"""RAW 文件读取层（headerless / 带 header / 带 stride / MIPI packed）。

传感器 dump 出来的 raw 常见形态：
  * 纯裸数据：W*H*bpp，无 header，无 padding；
  * 带 header：前面若干字节是寄存器/时间戳，后面才是像素；
  * 带行 stride：行尾有 padding（DMA burst 对齐），此时 stride > W*bpp；
  * MIPI packed：RAW10/12/14 按 4/2/4 像素一组压到 5/3/7 字节；
  * 16bit 容器左对齐存放 10/12/14bit（需要右移 data_shift 位）。

本模块只做"磁盘 -> (H, W) 整数数组"，不做任何显示归一化。
显示变换在 utils/display.py，统计在 utils/stats.py。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict

import numpy as np

__all__ = [
    "RawReadError",
    "RawLoadSpec",
    "PACKING_MODES",
    "packing_group",
    "bytes_per_pixel",
    "row_bytes",
    "expected_size",
    "pack_array",
    "unpack_array",
    "load_raw",
    "describe_file",
    "suggest_geometries",
    "KNOWN_GEOMETRIES",
]

# packing 名称（unpacked=每像素占 1/2/4 字节；packed=MIPI 紧凑打包）
PACKING_MODES = ("unpacked", "packed")

# 常见 sensor 分辨率：用于打开文件时的几何推断（优先命中这些）
KNOWN_GEOMETRIES = [
    (640, 480), (1280, 720), (1280, 800), (1280, 960), (1600, 1200),
    (1920, 1080), (1920, 1200), (1936, 1096), (2048, 1536), (2048, 1088),
    (2192, 1232), (2448, 2048), (2592, 1944), (2608, 1960), (2688, 1520),
    (2736, 1824), (3264, 2448), (3280, 2464), (3840, 2160), (4000, 3000),
    (4032, 3024), (4096, 3072), (4208, 3120), (4624, 3472), (5312, 2988),
    (5472, 3648), (6000, 4000), (8000, 6000), (8192, 6144), (9248, 6936),
]

# 位深 -> 未打包时的容器字节数
_CONTAINER_BYTES = {8: 1, 10: 2, 12: 2, 14: 2, 16: 2}


class RawReadError(Exception):
    """RAW 读取失败（文件大小不符、位深/几何不支持等）。"""


@dataclass
class RawLoadSpec:
    """描述一份 RAW 数据在文件中的布局。"""

    width: int = 1920
    height: int = 1080
    bit_depth: int = 10
    packing: str = "unpacked"       # "unpacked" | "packed"
    header_bytes: int = 0           # 像素数据之前的字节数
    stride_bytes: int = 0           # 行 stride；0 表示按紧凑排布自动计算
    endian: str = "little"          # 仅 unpacked 且容器 2 字节时有效
    data_shift: int = 0             # 16bit 容器左对齐时右移的位数
    frame_index: int = 0            # 多帧文件中的第几帧
    frame_stride_bytes: int = 0     # 多帧文件每帧占用的字节数；0 表示紧凑

    def copy(self, **kw) -> "RawLoadSpec":
        d = asdict(self)
        d.update(kw)
        return RawLoadSpec(**d)

    # -- 派生量 --------------------------------------------------------
    @property
    def container_bytes(self) -> int:
        return _CONTAINER_BYTES.get(self.bit_depth, 2)

    @property
    def group_pixels(self) -> int:
        """packed 模式下一个打包组的像素数（unpacked 恒为 1）。"""
        return packing_group(self.bit_depth) if self.packing == "packed" else 1

    @property
    def effective_bit_depth(self) -> int:
        """去掉容器对齐位移后，真实的 ADC 位深。"""
        return max(1, self.bit_depth - self.data_shift)

    def row_bytes(self) -> int:
        return row_bytes(self.width, self.bit_depth, self.packing, self.stride_bytes)

    def frame_bytes(self) -> int:
        return self.row_bytes() * self.height

    def expected_size(self) -> int:
        return expected_size(self)

    def describe(self) -> str:
        parts = [
            f"{self.width}x{self.height}",
            f"{self.bit_depth}bit",
            ("MIPI packed" if self.packing == "packed" else "unpacked"),
        ]
        if self.stride_bytes:
            parts.append(f"stride={self.stride_bytes}B")
        if self.header_bytes:
            parts.append(f"header={self.header_bytes}B")
        if self.data_shift:
            parts.append(f"shift>>{self.data_shift}")
        if self.endian != "little":
            parts.append(self.endian)
        if self.frame_index:
            parts.append(f"frame#{self.frame_index}")
        return ", ".join(parts)


def _check_bit_depth(bit_depth: int, packing: str):
    if bit_depth not in _CONTAINER_BYTES:
        raise RawReadError(
            f"不支持的位深 {bit_depth}，可选：{sorted(_CONTAINER_BYTES)}")
    if packing == "packed" and bit_depth not in (10, 12, 14):
        raise RawReadError("packed 模式仅支持 10/12/14bit")


def packing_group(bit_depth: int) -> int:
    """MIPI packed：8/gcd(bit_depth,8) 个像素构成一个 packing group。"""
    return 8 // math.gcd(int(bit_depth), 8)


def bytes_per_pixel(bit_depth: int, packing: str = "unpacked") -> float:
    _check_bit_depth(int(bit_depth), packing)
    if packing == "packed":
        return bit_depth / 8.0
    return float(_CONTAINER_BYTES[int(bit_depth)])


def row_bytes(width: int, bit_depth: int, packing: str = "unpacked",
              stride_bytes: int = 0) -> int:
    _check_bit_depth(int(bit_depth), packing)
    if width <= 0:
        raise RawReadError("width 必须为正整数")
    if packing == "packed":
        # MIPI 的行按 packing group 对齐：宽度不是 group 倍数时行尾补位，
        # 多解出来的像素会被丢弃（真实 sensor 存在这种奇数宽度）。
        group = packing_group(bit_depth)
        group_bytes = group * bit_depth // 8
        groups = -(-int(width) // group)               # ceil
        tight = groups * group_bytes
    else:
        tight = width * _CONTAINER_BYTES[int(bit_depth)]
    if stride_bytes and stride_bytes < tight:
        raise RawReadError(f"stride({stride_bytes}B) 小于一行像素所需 {tight}B")
    return int(stride_bytes or tight)


def expected_size(spec: RawLoadSpec) -> int:
    """spec 描述的指定帧数据末尾偏移。"""
    base = spec.header_bytes + max(0, spec.frame_index) * (
        spec.frame_stride_bytes or spec.frame_bytes())
    return int(base + spec.frame_bytes())


# ----------------------------------------------------------------------
# packed <-> unpacked
# ----------------------------------------------------------------------
def unpack_array(buf: np.ndarray, bit_depth: int, width: int) -> np.ndarray:
    """把 MIPI packed 的 uint8 缓冲解成 (H, W) uint16。

    三种格式的共同点（本实现的依据）：
      每组的前 group 个字节分别是各像素的高 8 位；剩下的
      group*(bit_depth-8) 位是一段连续低位比特流，把它当作**小端位流**
      （第 j 位 = 第 j//8 个字节的第 j%8 位），第 i 个像素的低位字段就是
      该流上第 i 个长度为 low_bits 的字段，字段内部同样低位在前。
      RAW10: byte4 = P3[1:0] P2[1:0] P1[1:0] P0[1:0]
      RAW12: byte2 = P1[3:0] P0[3:0]
      RAW14: 3 字节 24 位流，P0 取流上最低 6 位，P1 取次 6 位 ……
      与 V4L2 内核文档 pixfmt-srggb14p / 常见 C 实现一致（注意网上不少
      RAW14 解析代码把跨字节的字段高低位写反了）。

    `buf` 必须已按行裁成紧凑 packed 数据（(H, row_bytes) uint8）。
    """
    _check_bit_depth(bit_depth, "packed")
    group = packing_group(bit_depth)
    group_bytes = group * bit_depth // 8
    low_bits = bit_depth - 8

    buf = np.ascontiguousarray(buf, dtype=np.uint8)
    if buf.ndim == 1:
        buf = buf[None, :]
    if buf.shape[1] % group_bytes:
        raise RawReadError(
            f"packed RAW{bit_depth} 行字节数 {buf.shape[1]} 不是 {group_bytes} 的倍数")

    h, rb = buf.shape
    ngroups = rb // group_bytes
    groups = buf.reshape(h, ngroups, group_bytes)

    msb = groups[:, :, :group].astype(np.uint16)           # 各像素高 8 位
    if low_bits:
        low_stream = groups[:, :, group:].reshape(h, -1)    # 每行连续低位字节
        # unpackbits 是字节内 MSB-first，逐字节翻转后即为小端位流
        # （注意只能按字节翻转，整段翻转会把 group 顺序也倒过来）
        bits = np.unpackbits(low_stream, axis=1).reshape(h, -1, 8)[:, :, ::-1]
        bits = bits.reshape(h, ngroups, group, low_bits)
        weights = (1 << np.arange(low_bits)).astype(np.uint16)   # 字段内低位在前
        low = (bits * weights).sum(axis=3).astype(np.uint16)
        values = (msb << low_bits) | low
    else:                                                   # pragma: no cover
        values = msb

    flat = values.reshape(h, -1)
    if flat.shape[1] < width:
        raise RawReadError("packed 行解出的像素数不足 width")
    return np.ascontiguousarray(flat[:, :width])


def pack_array(image: np.ndarray, bit_depth: int) -> np.ndarray:
    """unpack_array 的逆运算：供测试与回写 MIPI packed 文件使用。"""
    _check_bit_depth(bit_depth, "packed")
    if image.ndim != 2:
        raise RawReadError("pack_array 只接受 2D 数组")
    h, w = image.shape
    group = packing_group(bit_depth)
    if w % group:
        # 兼容奇数宽度：行尾补到 group 边界再用 0 填充
        pad = (-w) % group
        image = np.pad(image, ((0, 0), (0, pad)))
        w = image.shape[1]
    low_bits = bit_depth - 8
    max_val = (1 << bit_depth) - 1
    img = np.asarray(image).astype(np.uint16, copy=False)
    if img.size and int(img.max()) > max_val:
        raise RawReadError(f"像素值超出 {bit_depth}bit 范围")

    img = img.reshape(h, w // group, group)
    msb = (img >> low_bits).astype(np.uint8)
    if low_bits:
        ngroups = w // group
        low = (img & ((1 << low_bits) - 1)).astype(np.uint8)
        # 与小端位流互逆：字段内低位在前 -> 逐字节翻成 MSB-first -> packbits
        shifts = np.arange(low_bits)
        bits = ((low[..., None] >> shifts) & 1).astype(np.uint8)
        bits = bits.reshape(h, ngroups, -1, 8)[:, :, :, ::-1]
        low_bytes = np.packbits(bits, axis=3).reshape(h, ngroups, -1)
        return np.concatenate([msb, low_bytes], axis=2).reshape(h, -1)
    return msb.reshape(h, -1)                                # pragma: no cover


# ----------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------
def _read_bytes(path: str, offset: int, count: int) -> np.ndarray:
    """读取文件 [offset, offset+count) 字节，返回 uint8 数组。

    大文件走 memmap 避免整份拷贝；memmap 失败时回退普通读。
    """
    size = os.path.getsize(path)
    if offset < 0 or count < 0:
        raise RawReadError("offset/count 不能为负")
    if offset + count > size:
        raise RawReadError(
            f"文件不够大：需要 {offset + count} 字节，实际 {size} 字节")
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    try:
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(count,))
        return np.array(mm)            # 物化成 ndarray，之后可安全释放 memmap
    except (ValueError, OSError):
        with open(path, "rb") as fh:
            fh.seek(offset)
            return np.frombuffer(fh.read(count), dtype=np.uint8).copy()


def load_raw(path: str, spec: RawLoadSpec) -> np.ndarray:
    """按 spec 读取一份 RAW，返回 (H, W) 的 uint8/uint16 数组。

    行 stride 里多出来的 padding 会被丢弃；返回值是原始 DN 值，
    未做黑电平/位深归一化。多帧文件用 spec.frame_index 选帧。
    """
    if not os.path.exists(path):
        raise RawReadError(f"文件不存在：{path}")
    if spec.packing == "packed":
        return _load_packed(path, spec)
    return _load_unpacked(path, spec)


def _load_unpacked(path: str, spec: RawLoadSpec) -> np.ndarray:
    _check_bit_depth(spec.bit_depth, "unpacked")
    bpp = spec.container_bytes
    tight = spec.width * bpp
    stride = row_bytes(spec.width, spec.bit_depth, "unpacked", spec.stride_bytes)
    frame_bytes = stride * spec.height
    frame_stride = spec.frame_stride_bytes or frame_bytes
    start = spec.header_bytes + max(0, spec.frame_index) * frame_stride

    size = os.path.getsize(path)
    if start + frame_bytes > size:
        raise RawReadError(
            f"文件尺寸不足：按 {spec.describe()} 需要 {start + frame_bytes} 字节，"
            f"实际 {size} 字节（差 {start + frame_bytes - size}）")

    raw = _read_bytes(path, start, frame_bytes)
    endian = ">" if str(spec.endian).lower().startswith("b") else "<"
    if bpp == 1:
        rows = raw.reshape(spec.height, stride)[:, :tight]
        image = rows.reshape(spec.height, spec.width).astype(np.uint8)
    else:
        dtype = np.dtype(f"{endian}u{bpp}")
        rows = raw.reshape(spec.height, stride)[:, :tight]
        contig = np.ascontiguousarray(rows)
        image = contig.view(dtype).reshape(spec.height, spec.width)
        if spec.data_shift:
            image = image >> int(spec.data_shift)
        image = image.astype(np.uint16, copy=False)
    return np.ascontiguousarray(image)


def _load_packed(path: str, spec: RawLoadSpec) -> np.ndarray:
    _check_bit_depth(spec.bit_depth, "packed")
    tight = row_bytes(spec.width, spec.bit_depth, "packed")
    stride = row_bytes(spec.width, spec.bit_depth, "packed", spec.stride_bytes)
    frame_bytes = stride * spec.height
    frame_stride = spec.frame_stride_bytes or frame_bytes
    start = spec.header_bytes + max(0, spec.frame_index) * frame_stride

    size = os.path.getsize(path)
    if start + frame_bytes > size:
        raise RawReadError(
            f"文件尺寸不足：按 {spec.describe()} 需要 {start + frame_bytes} 字节，"
            f"实际 {size} 字节（差 {start + frame_bytes - size}）")

    raw = _read_bytes(path, start, frame_bytes)
    rows = raw.reshape(spec.height, stride)[:, :tight]
    image = unpack_array(rows, spec.bit_depth, spec.width)
    if spec.data_shift:
        image = (image >> int(spec.data_shift)).astype(np.uint16, copy=False)
    return image


# ----------------------------------------------------------------------
# 探测 / 推断
# ----------------------------------------------------------------------
def describe_file(path: str) -> dict:
    """文件基本信息，供打开对话框显示。"""
    size = os.path.getsize(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "size": size,
        "size_mb": size / 1048576.0,
    }


def _aspect_score(w: int, h: int) -> float:
    """越接近常见成像宽高比得分越低（4:3 / 3:2 / 16:9 / 5:4 / 1:1 …）。"""
    r = w / float(h)
    targets = [4 / 3, 3 / 2, 16 / 9, 5 / 4, 1.0, 8 / 5, 21 / 9]
    return min(abs(r - t) for t in targets)


def suggest_geometries(file_size: int, bit_depths=(10, 12, 14, 16, 8),
                       allow_packed=True, max_results: int = 8) -> list:
    """由文件大小反推可能的 (width, height, bit_depth, packing)。

    返回按可信度排序的列表：每项 {spec, score, note, confidence, total_bytes}。
    先匹配内置常见 sensor 分辨率，再按因数分解补齐因数对。
    用于"打开 dump 后忘了分辨率/位深"的场景。
    """
    results = []
    seen = set()

    def add(spec: RawLoadSpec, score: float, note: str):
        w, h = spec.width, spec.height
        if w < 16 or h < 16 or w % 2 or h % 2:
            return
        key = (w, h, spec.bit_depth, spec.packing, spec.stride_bytes)
        if key in seen:
            return
        if spec.frame_bytes() != file_size:
            return
        seen.add(key)
        results.append({"spec": spec, "score": score, "note": note,
                        "total_bytes": spec.frame_bytes()})

    for depth in bit_depths:
        modes = ["unpacked"]
        if allow_packed and depth in (10, 12, 14):
            modes.append("packed")
        for packing in modes:
            group = packing_group(depth) if packing == "packed" else 1
            bpp = bytes_per_pixel(depth, packing)
            total_pixels = int(round(file_size / bpp))
            if total_pixels <= 0:
                continue
            if abs(total_pixels * bpp - file_size) > 0.5:
                continue

            # 1) 命中内置分辨率表（紧凑排布）
            for (kw, kh) in KNOWN_GEOMETRIES:
                if kw * kh == total_pixels:
                    add(RawLoadSpec(kw, kh, depth, packing), 0.0, "匹配常见分辨率")
                    add(RawLoadSpec(kh, kw, depth, packing), 0.6,
                        "匹配常见分辨率(旋转)")

            # 2) 常见分辨率 + 行 padding
            for (kw, kh) in KNOWN_GEOMETRIES:
                tight = row_bytes(kw, depth, packing)
                extra = file_size - tight * kh
                if 0 < extra and extra % kh == 0:
                    pad = extra // kh
                    if pad % 4 == 0 and 0 < pad <= tight:
                        spec = RawLoadSpec(kw, kh, depth, packing,
                                           stride_bytes=tight + pad)
                        # 行 padding 越"小"越可信（按占一行的比例衡量）
                        add(spec, 0.3 + min(pad / max(tight, 1), 0.25),
                            f"常见分辨率 + {pad}B 行 padding")

            # 3) 因数分解兜底
            for w in range(16, min(total_pixels, 12000) + 1, 2):
                if total_pixels % w:
                    continue
                h = total_pixels // w
                if h < 16 or h % 2:
                    continue
                ratio = w / h
                if not (0.5 <= ratio <= 4.0):
                    continue
                add(RawLoadSpec(w, h, depth, packing),
                    1.0 + _aspect_score(w, h), "因数分解")

    results.sort(key=lambda r: r["score"])
    for r in results:
        r["confidence"] = "高" if r["score"] < 0.5 else ("中" if r["score"] < 1.3 else "低")
    return results[:max_results]
