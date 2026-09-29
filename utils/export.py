"""导出层：16bit PNG / TIFF、CSV 统计与缺陷清单、带标注的截图。

为什么不用 Qt 存图：QImage 存 16bit 灰度 PNG 支持并不完整，而 sensor 数据
最需要的就是"无损 16bit"。这里用标准库 zlib 手写 PNG、手写基线 TIFF，
既保证位深不丢，也让导出逻辑不依赖 Qt（可以单独跑单元测试/脚本批处理）。
"""
from __future__ import annotations

import csv
import struct
import zlib

import numpy as np

__all__ = [
    "write_png", "write_tiff", "write_csv", "stats_to_rows",
    "defects_to_rows", "draw_defects", "to_display_rgb", "write_image",
]


def _as_uint(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8 or a.dtype == np.uint16:
        return a
    if a.dtype.kind == "f":
        return np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8) \
            if a.max(initial=0) <= 1.0 else np.clip(a, 0, 65535).astype(np.uint16)
    return a.astype(np.uint16) if a.max(initial=0) > 255 else a.astype(np.uint8)


# ----------------------------------------------------------------------
# PNG
# ----------------------------------------------------------------------
def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload +
            struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def write_png(path: str, image: np.ndarray) -> str:
    """写 8/16bit 灰度或 8bit RGB 的 PNG（无损，无外部依赖）。

    2D 数组 -> 灰度；3D(H,W,3) -> RGB。
    """
    a = _as_uint(image)
    if a.ndim == 2:
        h, w = a.shape
        color_type = 0
        channels = 1
    elif a.ndim == 3 and a.shape[2] >= 3:
        h, w, _ = a.shape
        a = np.ascontiguousarray(a[:, :, :3])
        color_type = 2
        channels = 3
    else:
        raise ValueError(f"不支持的数组形状：{a.shape}")

    bit_depth = 16 if a.dtype == np.uint16 else 8
    if bit_depth == 16:
        raw_samples = a.astype(">u2", copy=False)
    else:
        raw_samples = a.astype(np.uint8, copy=False)

    row_bytes = w * channels * (bit_depth // 8)
    raw = bytearray()
    flat = raw_samples.reshape(h, -1)
    for y in range(h):
        raw.append(0)                                   # filter type 0 (None)
        raw += flat[y].tobytes()

    ihdr = struct.pack(">IIBBBBB", w, h, bit_depth, color_type, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) +
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6)) +
            _png_chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# ----------------------------------------------------------------------
# TIFF（基线、单 strip、无压缩）
# ----------------------------------------------------------------------
def write_tiff(path: str, image: np.ndarray) -> str:
    """写不压缩的基线 TIFF，支持 8/16bit 灰度与 8bit RGB。

    16bit 是给 ISP/算法同事复算用的，务必不要经过 8bit 压缩。
    文件布局：[8B header][BitsPerSample 外置数组(仅 RGB)][像素数据][IFD]
    """
    a = _as_uint(image)
    if a.ndim == 2:
        h, w = a.shape
        samples, photometric = 1, 1
        bits = 16 if a.dtype == np.uint16 else 8
    elif a.ndim == 3 and a.shape[2] >= 3:
        h, w, _ = a.shape
        a = np.ascontiguousarray(a[:, :, :3])
        samples, photometric = 3, 2
        bits = 8
    else:
        raise ValueError(f"不支持的数组形状：{a.shape}")

    if samples == 1 and bits == 16:
        data = a.astype("<u2", copy=False).tobytes()
    else:
        data = a.astype(np.uint8, copy=False).tobytes()

    SHORT, LONG = 3, 4
    extra = struct.pack("<3H", 8, 8, 8) if samples == 3 else b""
    bits_offset = 8 if samples == 3 else 0
    data_offset = 8 + len(extra)
    ifd_offset = data_offset + len(data)
    if ifd_offset % 2:
        ifd_offset += 1

    entries = [
        (256, LONG, 1, w),                    # ImageWidth
        (257, LONG, 1, h),                    # ImageLength
        (258, SHORT, samples, bits_offset if samples == 3 else bits),
        (259, SHORT, 1, 1),                   # Compression = none
        (262, SHORT, 1, photometric),         # 1=BlackIsZero, 2=RGB
        (273, LONG, 1, data_offset),          # StripOffsets
        (277, SHORT, 1, samples),             # SamplesPerPixel
        (278, LONG, 1, h),                    # RowsPerStrip
        (279, LONG, 1, len(data)),            # StripByteCounts
        (284, SHORT, 1, 1),                   # PlanarConfig = chunky
        (339, SHORT, 1, 1),                   # SampleFormat = unsigned int
    ]

    out = bytearray()
    out += b"II" + struct.pack("<H", 42) + struct.pack("<I", ifd_offset)
    out += extra
    out += data
    out += b"\x00" * (ifd_offset - len(out))
    out += struct.pack("<H", len(entries))
    for tag, typ, count, value in sorted(entries):   # IFD 标签必须升序
        out += struct.pack("<HHI", tag, typ, count)
        if typ == SHORT and count == 1:
            out += struct.pack("<HH", value, 0)
        else:
            out += struct.pack("<I", value)
    out += struct.pack("<I", 0)                       # 无后续 IFD
    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return path


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------
def write_csv(path: str, header, rows) -> str:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return path


def stats_to_rows(stats: dict):
    """把 utils.stats.roi_stats 的结果转成 (header, rows)。

    ChannelStat 的字段名是 name/snr_db，这里映射成表头的 channel/snr_dB，
    并把 ROI 作为独立列写在最后（方便直接丢进 Excel 透视）。
    """
    header = ["scope", "channel", "count", "mean", "std", "min", "max",
              "median", "p01", "p99", "sat", "zero", "snr_dB", "roi"]
    rows = []
    if not stats:
        return header, rows
    x0, y0, x1, y1 = stats.get("roi", (0, 0, 0, 0))
    roi_txt = f"x{x0}..{x1} y{y0}..{y1} ({x1 - x0}x{y1 - y0})"
    keymap = {"channel": "name", "snr_dB": "snr_db"}

    def emit(scope, st):
        rows.append([scope] + [st.get(keymap.get(f, f), "") for f in header[1:-1]] + [roi_txt])

    overall = stats.get("overall")
    if overall:
        emit("overall", overall)
    for st in stats.get("channels", []):
        emit("plane", st)
    return header, rows


def defects_to_rows(defects):
    """缺陷清单 -> (header, rows)，供产线/yield 分析。"""
    header = ["type", "x", "y", "channel", "value", "delta", "note", "source"]
    rows = []
    for d in defects or []:
        rows.append([d.get("type", ""), d.get("x", ""), d.get("y", ""),
                     d.get("channel", ""), d.get("value", ""),
                     d.get("delta", ""), d.get("note", ""), d.get("source", "")])
    return header, rows


# ----------------------------------------------------------------------
# 标注截图
# ----------------------------------------------------------------------
_PALETTE = {
    "hot": (255, 64, 64),
    "dead": (64, 160, 255),
    "cluster": (255, 170, 0),
    "row": (255, 255, 0),
    "col": (0, 255, 220),
    "sat": (255, 0, 255),
}


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    """任意显示数组 -> (H, W, 3) uint8，便于画标注。"""
    a = _as_uint(image)
    if a.ndim == 2:
        if a.dtype == np.uint16:
            a = (a.astype(np.uint32) * 255 // 65535).astype(np.uint8)
        return np.repeat(a[:, :, None], 3, axis=2)
    return np.ascontiguousarray(a[:, :, :3])


def draw_defects(image: np.ndarray, defects, radius: int = 3,
                 line_width: int = 1) -> np.ndarray:
    """把缺陷位置画到图上的 RGB8 数组里（返回新数组，不改原图）。

    point 类型画方框，line 类型画横/竖线。
    """
    rgb = to_display_rgb(image).copy()
    h, w = rgb.shape[:2]

    def rect(x0, y0, x1, y1, color):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        if x1 < x0 or y1 < y0:
            return
        for t in range(line_width):
            if y0 + t <= y1:
                rgb[y0 + t, x0:x1 + 1] = color
                rgb[y1 - t, x0:x1 + 1] = color
            if x0 + t <= x1:
                rgb[y0:y1 + 1, x0 + t] = color
                rgb[y0:y1 + 1, x1 - t] = color

    for d in defects or []:
        c = _PALETTE.get(str(d.get("type", "")).lower(), (255, 255, 0))
        if "x" not in d or "y" not in d:
            continue
        x, y = int(d["x"]), int(d["y"])
        rect(x - radius, y - radius, x + radius, y + radius, c)
    return rgb


def write_image(path: str, image: np.ndarray) -> str:
    """按扩展名选择 png/tiff 写出。"""
    lower = str(path).lower()
    if lower.endswith((".tif", ".tiff")):
        return write_tiff(path, image)
    return write_png(path, image)
