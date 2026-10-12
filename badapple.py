#!/usr/bin/env python
"""Bad Apple → ShaderFlow WGSL 一键转换。

    uv run python ardu_video/badapple.py --fps 2.14 --size 32

给定视频，以及「帧率 / 分辨率 / 时长」里任意几项，脚本会在 ShaderFlow 的
65536 字节 editable 预算内自动调参，输出可直接粘贴的 WGSL，并逐步打印信息。

════════════════════════════════════════════════════════════════════════
管线
════════════════════════════════════════════════════════════════════════

  1. 探测视频（分辨率 / 帧率 / 时长）
  2. ffmpeg 抽帧：居中裁成正方形 → 缩放到主分辨率 → 灰度
  3. 二值化（默认阈值 96；原片本身强双峰，几乎无损）
  4. 自动调参：在预算内搜索 (存储分辨率, 时间步长, 关键帧间隔)
  5. 编码 + 生成 WGSL（紧凑文本，纯 ASCII）
  6. 自检：独立解码器逐像素回放，必须 0 差异；有 naga 就再编译一次

════════════════════════════════════════════════════════════════════════
为什么是这套编码（细节见 README.md）
════════════════════════════════════════════════════════════════════════

ShaderFlow 的 userfn 是 fragment shader，每个像素只知道 (x, y, t)，
无法顺序扫描变长码流。因此负载设计成：

    每帧 = [块变化位图] + [起始颜色 nibble] + [变化像素的游程流]
    关键帧 = 位图全 1（与增量帧同构 → 解码只有一条路径）

解码像素 (x,y)：
  · 从 t 得帧号 f；
  · 往前回退，找最近一次把本块标记为「变化」的帧（关键帧全 1，上界确定）；
  · 算 pos（扫描序子序列下标）后，从 run 流头部**全量遍历**到覆盖 pos 的游程。
    遍历实测只要 24~55 步/像素 —— 换来变长编码 14~38% 的体积收益。

体积上界是 editable 源码 ≤ 65536 字节（服务端 MaxCodeBytes），
u32 字面量按「十进制/十六进制取短、不缩进、每行 8 个」输出，约 10.5 字节/字。
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = HERE / "source.mp4"
MAX_CODE_BYTES = 65536
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)   # Rec.709

# 调参搜索用的候选（存储分辨率都取 4 的倍数，块大小固定 4）
SIZE_CANDIDATES = [128, 112, 96, 80, 64, 56, 48, 40, 36, 32, 28, 24, 20, 16]
BLOCK = 4
KEY_CANDIDATES = [16, 32]
BYTES_PER_WORD = 11.0      # 保守标定（实测 ~10.8），仅用于搜索阶段快速估算
FIXED_BYTES = 1200


# ══════════════════════════════════════════════════════════════════════
# 1. 视频探测与抽帧
# ══════════════════════════════════════════════════════════════════════

def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError:
        sys.exit("需要 imageio-ffmpeg：uv add imageio-ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: Path) -> dict:
    """用 ffmpeg 的 stderr 解析出分辨率 / 帧率 / 时长。"""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    text = proc.stderr.decode("utf-8", "replace")
    info = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0}

    import re
    for line in text.splitlines():
        if "Video:" in line and "x" in line:
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                info["width"], info["height"] = int(m.group(1)), int(m.group(2))
            m = re.search(r"([\d.]+) fps", line)
            if m:
                info["fps"] = float(m.group(1))
        if "Duration:" in line:
            m = re.search(r"(\d+):(\d+):([\d.]+)", line)
            if m:
                info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if not info["width"] or not info["fps"]:
        sys.exit(f"无法解析视频信息：{path}")
    return info


def extract_frames(video: Path, size: int) -> np.ndarray:
    """居中裁成正方形 → 缩放 → 灰度，返回 (F, size, size) uint8。"""
    info = probe_video(video)
    side = min(info["width"], info["height"])
    x0 = (info["width"] - side) // 2
    vf = f"crop={side}:{side}:{x0}:0,scale={size}:{size}:flags=lanczos"
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf", vf,
         "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE, check=True)
    per = size * size
    if len(proc.stdout) % per:
        sys.exit("ffmpeg 输出字节数不是帧大小整数倍")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, size, size)


# ══════════════════════════════════════════════════════════════════════
# 2. 编码：块变化位图 + 扫描序像素 RLE
# ══════════════════════════════════════════════════════════════════════

class BitWriter:
    """按位累积成 u32 小端序字流。"""

    def __init__(self) -> None:
        self.words: list[int] = []
        self.acc = 0
        self.n = 0

    def write(self, value: int, bits: int) -> None:
        self.acc |= (value & ((1 << bits) - 1)) << self.n
        self.n += bits
        while self.n >= 32:
            self.words.append(self.acc & 0xFFFFFFFF)
            self.acc >>= 32
            self.n -= 32

    def flush(self) -> None:
        if self.n:
            self.words.append(self.acc & 0xFFFFFFFF)
            self.acc = 0
            self.n = 0


def block_view(bw: np.ndarray, bs: int) -> np.ndarray:
    """(F,S,S) → (F, NB, bs*bs)：块栅格序、块内行优先（与 WGSL 的 local 一致）。"""
    F, S, _ = bw.shape
    n = S // bs
    return (bw.reshape(F, n, bs, n, bs).transpose(0, 1, 3, 2, 4)
              .reshape(F, n * n, bs * bs))


def block_of_pixel(size: int, bs: int) -> np.ndarray:
    """像素下标 → 块下标（两者都是扫描序）。"""
    n = size // bs
    rows = np.repeat(np.arange(size) // bs, size)
    cols = np.tile(np.arange(size) // bs, size)
    return rows * n + cols


def write_length(writer: BitWriter, length: int) -> int:
    """游程长度写入 nibble 流，返回 nibble 数。

    <=15 一个 nibble；否则 0 转义 +「3 位数据 + 1 位续」的分块链，最后 +16。
    """
    if length <= 15:
        writer.write(length, 4)
        return 1
    writer.write(0, 4)
    used = 1
    d = length - 16
    while d > 7:
        writer.write((d & 7) | 8, 4)
        d >>= 3
        used += 1
    writer.write(d & 7, 4)
    return used + 1


class Encoded:
    def __init__(self) -> None:
        self.words: list[int] = []
        self.offsets: list[int] = []      # 每帧负载起始字
        self.frames = 0
        self.key_frames = 0
        self.delta_frames = 0
        self.aliased = 0                  # 与上一帧完全相同、零字节
        self.max_nibbles = 0


def encode(bw: np.ndarray, bs: int, key_interval: int,
           budget_bytes: int = 0) -> Encoded:
    """编码全部帧；budget_bytes>0 时超预算就提前停止（用于搜索）。"""
    F, S, _ = bw.shape
    if S % bs:
        sys.exit(f"分辨率 {S} 必须是块大小 {bs} 的整数倍")
    n = S // bs
    NB = n * n
    bmap = math.ceil(NB / 32)
    blocks = block_view(bw, bs)
    blk_px = block_of_pixel(S, bs)

    dirty_all = np.zeros((F, NB), dtype=bool)
    if F > 1:
        dirty_all[1:] = (blocks[1:] != blocks[:-1]).any(axis=2)
    dirty_all[::key_interval] = True

    out = Encoded()
    writer = BitWriter()
    for f in range(F):
        dirty = dirty_all[f]
        if not dirty.any():
            out.offsets.append(out.offsets[-1])       # 与上一帧相同：零字节
            out.aliased += 1
            out.frames += 1
            continue
        out.offsets.append(len(writer.words))
        # ① 块变化位图（补齐到 u32）
        for i in range(0, NB, 32):
            word = 0
            for k in range(min(32, NB - i)):
                word |= int(dirty[i + k]) << k
            writer.write(word, 32)
        # ② 起始颜色 + 变化块像素的游程（图像行优先扫描序）
        vals = bw[f].reshape(-1)[dirty[blk_px]]
        writer.write(int(vals[0]), 4)
        runs = np.flatnonzero(vals[1:] != vals[:-1]) + 1
        lengths = np.diff(np.concatenate([[0], runs, [vals.size]]))
        nibs = 1
        for length in lengths:
            nibs += write_length(writer, int(length))
        writer.flush()
        out.max_nibbles = max(out.max_nibbles, nibs)
        out.frames += 1
        if dirty.all():
            out.key_frames += 1
        else:
            out.delta_frames += 1
        if budget_bytes and len(writer.words) * BYTES_PER_WORD > budget_bytes:
            break
    out.words = writer.words
    return out


# ══════════════════════════════════════════════════════════════════════
# 3. WGSL 生成
# ══════════════════════════════════════════════════════════════════════

def literal(value: int) -> str:
    """u32 字面量：十进制与十六进制取短（0x0→0 省 2 字符；从不更长）。"""
    hexed = f"0x{value:x}"
    decimal = str(value)
    return decimal if len(decimal) < len(hexed) else hexed


def format_words(words: list[int], per_line: int = 8) -> str:
    """紧凑数组文本：不缩进、不留空格、每行 8 个（最长约 88 列）。"""
    return "\n".join(",".join(literal(w) for w in words[i:i + per_line]) + ","
                     for i in range(0, len(words), per_line))


DECODE_RLE = """  let by = py / bs;
  let bx = px / bs;
  let blk = by * bn + bx;
  let bw_i = blk / 32u;
  let bw_s = blk % 32u;
  let low_mask = (1u << bw_s) - 1u;
  var bits = 0u;
  for (var back = 0u; back < {key}u; back = back + 1u) {{
    if (back > frame) {{ break; }}
    let cf = frame - back;
    let pk = index[cf / 2u];
    let ob = select(pk >> 16u, pk & 0xffffu, (cf & 1u) == 0u);
    if (((data[ob + bw_i] >> bw_s) & 1u) == 0u) {{ continue; }}
    var rank = 0u;
    for (var ri = 0u; ri < bmap; ri = ri + 1u) {{
      let w = data[ob + ri];
      if (ri < bw_i) {{ rank = rank + countOneBits(w); }}
      if (ri == bw_i) {{ rank = rank + countOneBits(w & low_mask); }}
    }}
{prefix_a}
{prefix_t}
    let pos = {img_bits}u * A + (py % bs) * {bs}u * (T - A) + {bs}u * (rank - A) + (px % bs);
    let nb = ob + bmap;
    var color = data[nb] & 0xfu;
    var acc = 0u;
    var k = 1u;
    for (var guard = 0u; guard < {max_nib}u; guard = guard + 1u) {{
      var len = (data[nb + k / 8u] >> ((k % 8u) * 4u)) & 0xfu;
      k = k + 1u;
      if (len == 0u) {{
        var q = 0u;
        var val = 0u;
        for (var g2 = 0u; g2 < 12u; g2 = g2 + 1u) {{
          let p = (data[nb + k / 8u] >> ((k % 8u) * 4u)) & 0xfu;
          k = k + 1u;
          val = val | ((p & 7u) << q);
          q = q + 3u;
          if ((p & 8u) == 0u) {{ break; }}
        }}
        len = val + 16u;
      }}
      if (acc + len > pos) {{ break; }}
      acc = acc + len;
      color = 1u - color;
    }}
    bits = color;
    break;
  }}"""


def _prefix(indent: str, name: str, target: str) -> str:
    """位图 [0, target) 的前缀 popcount（bmap 个 u32，2~8 次循环）。"""
    return f"""{indent}var {name} = 0u;
{indent}for (var pi = 0u; pi < bmap; pi = pi + 1u) {{
{indent}  let base = pi * 32u;
{indent}  if (base + 32u <= {target}) {{
{indent}    {name} = {name} + countOneBits(data[ob + pi]);
{indent}  }} else if (base < {target}) {{
{indent}    {name} = {name} + countOneBits(data[ob + pi] & ((1u << ({target} - base)) - 1u));
{indent}  }}
{indent}}}"""


def sample_code(indent: str, filt: str, key: int, bmap: int, img_bits: int,
                bs: int, max_nib: int) -> str:
    """生成 sample ∈ [0,1] 的采样代码。"""
    inner = DECODE_RLE.format(key=key, bmap=bmap, img_bits=img_bits, bs=bs,
                              max_nib=max_nib,
                              prefix_a=_prefix(indent + "    ", "A", "by * bn"),
                              prefix_t=_prefix(indent + "    ", "T", "(by + 1u) * bn"))
    inner = "\n".join(indent + "    " + line if line.strip() else line
                      for line in inner.splitlines())
    head = f"""{indent}let sf = f32(size);
{indent}let sx = clamp(x, 0.0, 1.0) * sf - 0.5;
{indent}let sy = (1.0 - clamp(y, 0.0, 1.0)) * sf - 0.5;
{indent}let x0 = floor(sx);
{indent}let y0 = floor(sy);
{indent}let fx = sx - x0;
{indent}let fy = sy - y0;
{indent}var sample = 0.0;
{indent}for (var jj = 0u; jj < {2 if filt != 'nearest' else 1}u; jj = jj + 1u) {{
{indent}  for (var ii = 0u; ii < {2 if filt != 'nearest' else 1}u; ii = ii + 1u) {{
{indent}    let px = u32(clamp(x0 + f32(ii), 0.0, sf - 1.0));
{indent}    let py = u32(clamp(y0 + f32(jj), 0.0, sf - 1.0));
{indent}    let wgt = select(1.0 - fx, fx, ii == 1u) * select(1.0 - fy, fy, jj == 1u);"""
    tail = f"""{indent}    sample = sample + select(0.0, 1.0, bits == 1u) * wgt;
{indent}  }}
{indent}}}"""
    if filt == "nearest":
        head = f"""{indent}let px = u32(clamp(floor(clamp(x, 0.0, 1.0) * f32(size)), 0.0, f32(size) - 1.0));
{indent}let py = u32(clamp(floor((1.0 - clamp(y, 0.0, 1.0)) * f32(size)), 0.0, f32(size) - 1.0));"""
        tail = f"""{indent}let sample = select(0.0, 1.0, bits == 1u);"""
    return head + "\n" + inner + "\n" + tail


def build_editable(enc: Encoded, size: int, bs: int, key: int, fps: float,
                   filt: str = "smooth", dissolve_window: float = 0.0) -> str:
    packed = []
    for i in range(0, len(enc.offsets), 2):
        lo = enc.offsets[i]
        hi = enc.offsets[i + 1] if i + 1 < len(enc.offsets) else 0
        if lo >= 65536 or hi >= 65536:
            sys.exit("帧偏移超出 u16，请减少帧数")
        packed.append(lo | (hi << 16))

    n = size // bs
    bmap = math.ceil(n * n / 32)
    fps_lit = f"{fps:.6f}".rstrip("0").rstrip(".")
    finalize = "smoothstep(0.3, 0.7, sample)" if filt == "smooth" else "sample"

    if dissolve_window > 0:
        body_sample = sample_code("    ", filt, key, bmap, bs * bs, bs, enc.max_nibbles)
        blend = (f"smoothstep({1.0 - dissolve_window:.4f}, 1.0, frac)"
                 if dissolve_window < 0.999 else "frac")
        tail = f"""  // frame + crossfade
  let clock = max(t, 0.0) * {fps_lit};
  let fid = floor(clock);
  let frac = clock - fid;
  let blendw = {blend};
  var out = 0.0;
  for (var k = 0u; k < 2u; k = k + 1u) {{
    let frame = (u32(fid) + k) % nf;
{body_sample}
    out = out + (2.0 * {finalize} - 1.0) * select(1.0 - blendw, blendw, k == 1u);
  }}
  return out;"""
    else:
        body_sample = sample_code("  ", filt, key, bmap, bs * bs, bs, enc.max_nibbles)
        tail = f"""  // frame index, loops
  let frame = u32(floor(max(t, 0.0) * {fps_lit})) % nf;

{body_sample}

  return clamp(2.0 * {finalize} - 1.0, -1.0, 1.0);"""

    return f"""// ShaderFlow editable area: binarized video hard-coded in userfn.
const positiveColor = vec3f(255., 255., 255.) / 256.;
const negativeColor = vec3f(0., 0., 0.) / 256.;
fn userfn(x: f32, y: f32, t: f32) -> f32 {{
  // {enc.frames} frames, {size}x{size}, key {key}
  var data = array<u32, {len(enc.words)}>(
{format_words(enc.words)}
  );
  var index = array<u32, {len(packed)}>(
{format_words(packed)}
  );
  let nf = {enc.frames}u;
  let size = {size}u;
  let bn = {n}u;
  let bs = {bs}u;
  let bmap = {bmap}u;

{tail}
}}
"""


LOCKED = """struct Frame {{ frame : u32, }};
@group(0) @binding(0) var<uniform> frameData : Frame;
struct VertexOut {{
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
}};
@vertex fn vs(@builtin(vertex_index) vertexIndex : u32) -> VertexOut {{
  let pos = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let p = pos[vertexIndex];
  var out : VertexOut;
  out.position = vec4f(p, 0.0, 1.0);
  out.uv = p * 0.5 + 0.5;
  return out;
}}
const positiveColor = vec3f(255., 255., 255.) / 256.;
const negativeColor = vec3f(0., 0., 0.) / 256.;
fn userfn(x: f32, y: f32, t: f32) -> f32 {{{body}}}
@fragment fn fs(in : VertexOut) -> @location(0) vec4f {{
  let t = f32(frameData.frame) * 0.01;
  let uv = in.uv;
  let ret = clamp(userfn(uv.x, uv.y, t), -1.0, 1.0);
  return vec4f(select(negativeColor * -ret, positiveColor * ret, ret > 0.0), 1.0);
}}"""


def build_full(editable: str) -> str:
    """拼上锁定模板，用于 naga 离线编译验证。"""
    head, _, rest = editable.partition("fn userfn(x: f32, y: f32, t: f32) -> f32 {")
    body = rest.rsplit("}", 1)[0]
    return LOCKED.format(body=body)


# ══════════════════════════════════════════════════════════════════════
# 4. 独立解码器（自检）
# ══════════════════════════════════════════════════════════════════════

def decode_editable(text: str) -> np.ndarray:
    """照搬 WGSL 的寻址逻辑，把 editable 还原成帧序列，用于无损校验。"""
    import re
    arrays = {}
    for name in ("data", "index"):
        m = re.search(rf"var {name} = array<u32, (\d+)>\((.*?)\n  \);", text, re.S)
        assert m is not None
        arrays[name] = [int(v, 0) for v in
                        re.findall(r"0x[0-9a-fA-F]+|\d+", m.group(2))]
    def get_constant(name: str) -> int:
        match = re.search(rf"let {name} = (\d+)u;", text)
        assert match is not None
        return int(match.group(1))

    nf, size, bn, bs, bmap = (get_constant(k)
                              for k in ("nf", "size", "bn", "bs", "bmap"))
    words, index_l = arrays["data"], arrays["index"]
    img_bits = bs * bs

    def prefix(ob, m):
        total = 0
        for i in range(bmap):
            base = i * 32
            if base + 32 <= m:
                total += bin(words[ob + i]).count("1")
            elif base < m:
                total += bin(words[ob + i] & ((1 << (m - base)) - 1)).count("1")
        return total

    out = np.zeros((nf, size, size), dtype=np.uint8)
    for f in range(nf):
        for y in range(size):
            by, dy = divmod(y, bs)
            for x in range(size):
                bx, dx = divmod(x, bs)
                b = by * bn + bx
                cf = f
                while True:
                    ob = index_l[cf // 2]
                    ob = (ob & 0xFFFF) if cf % 2 == 0 else (ob >> 16)
                    if (words[ob + b // 32] >> (b % 32)) & 1:
                        break
                    cf -= 1
                rank = 0
                for i in range(bmap):
                    w = words[ob + i]
                    if i < b // 32:
                        rank += bin(w).count("1")
                    elif i == b // 32:
                        rank += bin(w & ((1 << (b % 32)) - 1)).count("1")
                A = prefix(ob, by * bn)
                B = prefix(ob, (by + 1) * bn) - A
                pos = img_bits * A + dy * bs * B + bs * (rank - A) + dx

                def nib(n, ob=ob):
                    return (words[ob + bmap + n // 8] >> ((n % 8) * 4)) & 0xF

                acc, color, k = 0, nib(0), 1
                while True:
                    length = nib(k)
                    k += 1
                    if length == 0:
                        q, val = 0, 0
                        while True:
                            p = nib(k)
                            k += 1
                            val |= (p & 7) << q
                            q += 3
                            if not (p & 8):
                                break
                        length = val + 16
                    if acc + length > pos:
                        out[f, y, x] = color
                        break
                    acc += length
                    color ^= 1
    return out


# ══════════════════════════════════════════════════════════════════════
# 5. 自动调参
# ══════════════════════════════════════════════════════════════════════

def plan_size(enc_words: int, index_words: int) -> int:
    return int(FIXED_BYTES + BYTES_PER_WORD * (enc_words + index_words))


def auto_tune(binarized, video_fps: float, args, log) -> dict:
    """在预算内搜索 (size, stride, key)。

    约束（谁给了就满足谁）：--size / --fps / --duration。
    自由维度：分辨率优先（从大到小试，第一个可行即最大分辨率），
    同分辨率下帧率优先（步长从小到大试，第一个可行即最高帧率）。
    两者都没给时，还要求帧率不低于 --min-fps。
    """
    total = binarized.shape[0]
    sizes = [args.size] if args.size else [s for s in SIZE_CANDIDATES if s % BLOCK == 0]
    max_stride = max(1, total - 1)
    fallback = None
    log(f"可用源帧 {total}（{video_fps:g} fps），预算 {args.budget} 字节")

    for size in sizes:
        bw = binarized
        if bw.shape[1] != size:
            step = bw.shape[1] / size
            idx = np.minimum((np.arange(size) * step + step / 2).astype(int), bw.shape[1] - 1)
            bw = bw[:, idx][:, :, idx]

        if args.fps:
            strides = [max(1, round(video_fps / args.fps))]
        else:
            strides = range(1, min(120, max_stride))

        best_here = None
        tried = rejected = 0
        for key in KEY_CANDIDATES:
            for stride in strides:
                stored_fps = video_fps / stride
                sel = bw[args.start::stride]
                want = sel.shape[0]
                if args.duration:
                    want = min(want, max(1, int(round(args.duration * stored_fps))))
                if want < 1:
                    continue
                tried += 1
                enc = encode(sel[:want], BLOCK, key, budget_bytes=args.budget)
                fits = FIXED_BYTES + BYTES_PER_WORD * len(enc.words) <= args.budget
                short = args.duration is not None and enc.frames < want
                if not fits or short:
                    rejected += 1
                    continue
                cand = dict(size=size, stride=stride, key=key, fps=stored_fps,
                            nframes=enc.frames, words=len(enc.words), enc=enc)
                log(f"  试 {size:>3}x{size:<3} stride={stride:>2} ({stored_fps:5.2f} fps) "
                    f"key={key:>2} → {len(enc.words):>5} 字  ✓ 可行")
                if best_here is None or _better(cand, best_here):
                    best_here = cand
                if not args.fps:
                    break                        # 步长递增 → 第一个可行即最高帧率
        if rejected and not best_here:
            log(f"  试 {size:>3}x{size:<3} … {tried} 组组合均超预算 ✗")
        if best_here is None:
            continue

        if fallback is None:
            fallback = best_here
        if args.size is None and args.fps is None and best_here["fps"] < args.min_fps:
            continue                             # 帧率太低，换更小分辨率试试
        return best_here

    if fallback is None:
        sys.exit("预算内找不到可行参数：请降低 --size / --fps / --duration")
    log(f"  没有分辨率能在帧率 ≥ {args.min_fps:g} 下通过，退回最大可行分辨率")
    return fallback


def _better(a: dict, b: dict) -> bool:
    """同分辨率下：帧率更高更好；帧率相同则体积更小更好。"""
    if abs(a["fps"] - b["fps"]) > 1e-6:
        return a["fps"] > b["fps"]
    return a["words"] < b["words"]


# ══════════════════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="把 Bad Apple 视频转成 ShaderFlow 的 userfn WGSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  uv run python ardu_video/badapple.py --fps 2.14 --size 32\n"
               "  uv run python ardu_video/badapple.py --duration 219 --min-fps 2\n"
               "  uv run python ardu_video/badapple.py --fps 5 --out out/a.wgsl\n"
               + (__doc__ or ""))
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="输入视频")
    ap.add_argument("--fps", type=float, default=None, help="期望播放帧率")
    ap.add_argument("--size", type=int, default=None, help="存储分辨率（4 的倍数）")
    ap.add_argument("--duration", type=float, default=None, help="期望时长（秒）")
    ap.add_argument("--min-fps", type=float, default=2.0,
                    help="未指定 --size/--fps 时，可接受的最低帧率（默认 2.0）")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "badapple.wgsl")
    ap.add_argument("--start", type=int, default=0, help="从源视频第几帧开始")
    ap.add_argument("--thresh", type=int, default=96, help="二值化阈值")
    ap.add_argument("--key", type=int, default=None, help="关键帧间隔（默认自动）")
    ap.add_argument("--filter", choices=("smooth", "bilinear", "nearest"), default="smooth")
    ap.add_argument("--dissolve-window", type=float, default=0.0,
                    help="帧间交叉淡化窗口比例，0=关闭（>0 时每个像素多解一帧）")
    ap.add_argument("--budget", type=int, default=MAX_CODE_BYTES, help="源码字节上限")
    ap.add_argument("--master", type=int, default=128, help="抽帧主分辨率")
    ap.add_argument("--emit-full", type=Path, default=None, help="另存含锁定模板的完整着色器")
    ap.add_argument("--no-verify", action="store_true", help="跳过逐像素回放自检")
    args = ap.parse_args()

    t0 = time.time()

    def log(msg=""):
        print(msg, flush=True)

    log("═" * 68)
    log("Bad Apple → ShaderFlow WGSL")
    log("═" * 68)

    # 1. 探测
    log("\n[1/6] 探测视频")
    if not args.video.exists():
        sys.exit(f"找不到视频：{args.video}")
    info = probe_video(args.video)
    log(f"      {args.video.name}: {info['width']}x{info['height']}, "
        f"{info['fps']:g} fps, {info['duration']:.2f} s")

    # 2. 抽帧
    master = max(args.master, args.size or 0)
    log(f"\n[2/6] 抽帧（居中裁 {min(info['width'], info['height'])}x"
        f"{min(info['width'], info['height'])} → {master}x{master} 灰度）")
    gray = extract_frames(args.video, master)
    log(f"      {gray.shape[0]} 帧, {gray.nbytes / 1e6:.1f} MB")

    # 3. 二值化
    log(f"\n[3/6] 二值化（阈值 {args.thresh}）")
    bw = gray > args.thresh
    log(f"      白像素 {bw.mean() * 100:.1f}%（源片强双峰，二值化几乎无损）")

    # 4. 自动调参
    log(f"\n[4/6] 自动调参")
    plan = auto_tune(bw, info["fps"], args, log)
    if args.key:
        plan["key"] = args.key
    log(f"      ✓ 选定 {plan['size']}x{plan['size']} @ {plan['fps']:.2f} fps, "
        f"key={plan['key']}, {plan['nframes']} 帧, 时长 {plan['nframes'] / plan['fps']:.1f} s")

    # 5. 最终编码 + 生成
    log(f"\n[5/6] 编码 + 生成 WGSL")
    size = plan["size"]
    if bw.shape[1] != size:
        step = bw.shape[1] / size
        idx = np.minimum((np.arange(size) * step + step / 2).astype(int), bw.shape[1] - 1)
        bw = bw[:, idx][:, :, idx]
    stream = bw[args.start::plan["stride"]]
    n = size // BLOCK
    bmap = math.ceil(n * n / 32)

    # 先多编一些帧（搜索阶段是估算，这里按真实文本长度二分取满预算）
    cap = min(stream.shape[0], plan["nframes"] * 2 + 200)
    enc = encode(stream[:cap], BLOCK, plan["key"], budget_bytes=args.budget * 3)

    def build_n(count: int) -> tuple[str, int]:
        sub = Encoded()
        end = enc.offsets[count] if count < enc.frames else len(enc.words)
        sub.words = enc.words[:end]
        sub.offsets = enc.offsets[:count]
        sub.frames = count
        sub.max_nibbles = enc.max_nibbles
        text = build_editable(sub, size, BLOCK, plan["key"], plan["fps"],
                              args.filter, args.dissolve_window)
        return text, len(text.encode())

    lo, hi = 1, min(enc.frames, plan["nframes"] if args.duration else enc.frames)
    best_text, best_bytes = build_n(1)
    best_n = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        text, nb = build_n(mid)
        if nb <= args.budget and mid >= best_n:
            best_text, best_bytes, best_n = text, nb, mid
            lo = mid + 1
        else:
            hi = mid - 1
    text, nbytes = best_text, best_bytes
    count = best_n
    log(f"      关键帧 {enc.key_frames}, 增量帧 {enc.delta_frames}, "
        f"零字节复用 {enc.aliased}")
    log(f"      采用 {count} 帧, 时长 {count / plan['fps']:.1f} s")
    log(f"      源码 {nbytes} 字节 = 预算的 {nbytes / args.budget * 100:.1f}%"
        f"{'  ⚠ 超预算' if nbytes > args.budget else ''}")

    # 6. 自检
    log(f"\n[6/6] 自检")
    if not args.no_verify:
        t = time.time()
        back = decode_editable(text)
        ref = stream[:back.shape[0]].astype(np.uint8)
        diff = int(np.abs(back.astype(int) - ref).sum())
        log(f"      逐像素回放：{diff} / {ref.size} 差异 "
            f"→ {'无损 ✓' if diff == 0 else '有损 ✗'}（{time.time() - t:.1f}s）")
        if diff:
            return 1
    naga = shutil.which("naga")
    if naga:
        full_path = args.out.with_suffix(".full.wgsl")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(build_full(text), encoding="utf-8", newline="\n")
        proc = subprocess.run([naga, str(full_path), str(full_path.with_suffix(".spv"))],
                              capture_output=True)
        log(f"      naga 编译：{'通过 ✓' if proc.returncode == 0 else '失败 ✗'} "
            f"({full_path.name})")
    else:
        log("      未找到 naga，跳过 WGSL 编译验证（可选：cargo install naga-cli）")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    if args.emit_full:
        args.emit_full.parent.mkdir(parents=True, exist_ok=True)
        args.emit_full.write_text(build_full(text), encoding="utf-8", newline="\n")
    log(f"\n输出 → {args.out}（{nbytes} 字节）")
    log(f"把整段内容粘进 http://<host>/player/ 的编辑框即可。总耗时 {time.time() - t0:.1f}s")
    return 0 if nbytes <= args.budget else 1


if __name__ == "__main__":
    raise SystemExit(main())
