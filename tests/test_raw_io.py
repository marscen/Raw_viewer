"""RAW 读取层测试：packed 位序、header/stride/字节序/多帧、几何推断、错误路径。"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests._util import check, run  # noqa: E402

from utils.raw_io import (RawReadError, RawLoadSpec, load_raw, pack_array,  # noqa: E402
                          row_bytes, suggest_geometries, unpack_array)

TMP = tempfile.mkdtemp(prefix="rawv2_io_")


def _ref_unpack(row, depth, width):
    """独立参考实现（按 MIPI/V4L2 文档文本重写，不复用被测代码）。"""
    row = [int(v) for v in row]
    out = []
    if depth == 10:
        for i in range(0, len(row), 5):
            b0, b1, b2, b3, b4 = row[i:i + 5]
            out += [(b0 << 2) | (b4 & 3), (b1 << 2) | ((b4 >> 2) & 3),
                    (b2 << 2) | ((b4 >> 4) & 3), (b3 << 2) | ((b4 >> 6) & 3)]
    elif depth == 12:
        for i in range(0, len(row), 3):
            b0, b1, b2 = row[i:i + 3]
            out += [(b0 << 4) | (b2 & 0xF), (b1 << 4) | ((b2 >> 4) & 0xF)]
    else:  # 14bit，按 V4L2 pixfmt-srggb14p 表格
        for i in range(0, len(row), 7):
            b0, b1, b2, b3, b4, b5, b6 = row[i:i + 7]
            out += [(b0 << 6) | (b4 & 0x3F),
                    (b1 << 6) | ((b5 & 0xF) << 2) | ((b4 >> 6) & 3),
                    (b2 << 6) | ((b6 & 3) << 4) | ((b5 >> 4) & 0xF),
                    (b3 << 6) | ((b6 >> 2) & 0x3F)]
    return out[:width]


def test_packed_roundtrip_and_bit_order():
    rng = np.random.default_rng(0)
    for depth in (10, 12, 14):
        for w in (8, 64, 1920):
            img = rng.integers(0, 1 << depth, size=(6, w)).astype(np.uint16)
            packed = pack_array(img, depth)
            check(packed.shape[1] == row_bytes(w, depth, "packed"),
                  f"RAW{depth} w={w} 行字节数 {packed.shape[1]}")
            back = unpack_array(packed, depth, w)
            check(np.array_equal(img, back), f"RAW{depth} w={w} pack/unpack 往返")
            ref = np.array([_ref_unpack(r, depth, w) for r in packed], dtype=np.uint16)
            check(np.array_equal(ref, img),
                  f"RAW{depth} w={w} 与规范参考实现逐位一致（位序不能反）")


def test_packed_odd_width_padded_row():
    """宽度不是 packing group 倍数时，行尾补位、多解出的像素丢弃。"""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 1024, size=(4, 30)).astype(np.uint16)
    packed = pack_array(img, 10)
    check(packed.shape[1] == 40, "30 像素 RAW10 应补齐到 40B/行")
    check(np.array_equal(unpack_array(packed, 10, 30), img), "奇数宽度 packed 往返")


def test_file_layouts_roundtrip():
    """header / 行 padding / 字节序 / 左对齐位移 / 多帧 的文件级往返。"""
    import generate_sample as gs
    rng = np.random.default_rng(2)
    img = rng.integers(0, 1024, size=(32, 48)).astype(np.uint16)

    # unpacked + uint16
    spec = RawLoadSpec(48, 32, 16)
    path = os.path.join(TMP, "u16.raw")
    check(gs.write_sample(path, img, {"bit_depth": 16}, "unpacked")["bytes"] == 48 * 32 * 2,
          "16bit 紧凑文件大小")
    check(np.array_equal(load_raw(path, spec), img), "unpacked uint16 往返")

    # header + 行 padding + 大端（12bit 左对齐放 16bit 容器）
    headers = {"scene": "x", "bit_depth": 16}
    path = os.path.join(TMP, "hdr.raw")
    gs.write_sample(path, img << 4, headers, "unpacked", header=16, stride_pad=8,
                    endian="big")
    spec = RawLoadSpec(48, 32, 16, header_bytes=16, stride_bytes=48 * 2 + 8,
                       endian="big", data_shift=4)
    check(np.array_equal(load_raw(path, spec), img), "header+stride+大端+左对齐 往返")

    # MIPI packed + 行 padding
    path = os.path.join(TMP, "p10.raw")
    gs.write_sample(path, img, {"bit_depth": 10}, "packed", stride_pad=4)
    spec = RawLoadSpec(48, 32, 10, packing="packed", stride_bytes=row_bytes(48, 10, "packed") + 4)
    check(np.array_equal(load_raw(path, spec), img), "packed + stride 往返")

    # 多帧 + frame_index / frame_stride
    frames = np.stack([img, img[::-1], (img // 2)])
    path = os.path.join(TMP, "multi.raw")
    gs.write_sample(path, frames, {"bit_depth": 16}, "unpacked")
    for i in range(3):
        check(np.array_equal(load_raw(path, RawLoadSpec(48, 32, 16, frame_index=i)), frames[i]),
              f"多帧 frame_index={i}")
    fb = 48 * 32 * 2
    spec = RawLoadSpec(48, 32, 16, frame_index=2, frame_stride_bytes=fb)
    check(np.array_equal(load_raw(path, spec), frames[2]), "显式 frame_stride")


def test_size_validation_and_errors():
    path = os.path.join(TMP, "small.raw")
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 64)
    try:
        load_raw(path, RawLoadSpec(100, 100, 16))
        check(False, "尺寸不足应当报错")
    except RawReadError as exc:
        check("尺寸不足" in str(exc) and "实际" in str(exc), f"报错信息可读: {exc}")
    try:
        load_raw(os.path.join(TMP, "nope.raw"), RawLoadSpec(8, 8, 8))
        check(False, "文件不存在应当报错")
    except RawReadError:
        check(True, "文件不存在报错")
    try:
        unpack_array(np.zeros((2, 4), np.uint8), 10, 4)
        check(False, "packed 行字节数非法应当报错")
    except RawReadError:
        check(True, "packed 行字节数校验")
    try:
        row_bytes(16, 9, "unpacked")
        check(False, "非法位深应当报错")
    except RawReadError:
        check(True, "位深校验")


def test_chunked_packed_decode_matches_and_bounds_memory():
    """分块解码：结果必须与整块一致，且不再按文件体积倍数吃内存。"""
    rng = np.random.default_rng(7)
    for depth in (10, 12, 14):
        rows = 600                                  # > 512 行会走分块路径
        img = rng.integers(0, 1 << depth, (rows, 64)).astype(np.uint16)
        path = os.path.join(TMP, f"chunk{depth}.raw")
        with open(path, "wb") as fh:
            fh.write(pack_array(img, depth).tobytes())
        got = load_raw(path, RawLoadSpec(64, rows, depth, packing="packed"))
        check(np.array_equal(got, img), f"RAW{depth} 600 行分块解码一致")
    # 分块路径与不分块路径结果一致（同一份数据，行数不同）
    img = rng.integers(0, 1024, (513, 32)).astype(np.uint16)
    path = os.path.join(TMP, "chunk_edge.raw")
    with open(path, "wb") as fh:
        fh.write(pack_array(img, 10).tobytes())
    check(np.array_equal(load_raw(path, RawLoadSpec(32, 513, 10, packing="packed")), img),
          "分块边界（513 行）一致")


def test_frame_stride_validation():
    data = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    path = os.path.join(TMP, "stride.raw")
    with open(path, "wb") as fh:
        fh.write(data.tobytes() * 2)
    try:
        load_raw(path, RawLoadSpec(16, 16, 16, frame_stride_bytes=64))
        check(False, "frame_stride 过小应当报错（否则帧会静默重叠）")
    except RawReadError as exc:
        check("重叠" in str(exc), f"frame_stride 校验信息: {exc}")
    check(np.array_equal(load_raw(path, RawLoadSpec(16, 16, 16, frame_index=1)), data),
          "正常多帧仍可读")


def test_geometry_suggestion():
    cases = [
        (1920 * 1080 * 2, (1920, 1080)),
        (1920 * 1080 * 10 // 8, (1920, 1080)),
        ((1920 * 2 + 64) * 1080, (1920, 1080)),
        (4000 * 3000 * 2, (4000, 3000)),
        (2592 * 1944 * 12 // 8, (2592, 1944)),
        (1280 * 720 * 2, (1280, 720)),
    ]
    for size, expect in cases:
        res = suggest_geometries(size)
        check(bool(res), f"{size}B 能给出候选")
        got = (res[0]["spec"].width, res[0]["spec"].height)
        check(got == expect, f"{size}B -> {got}（期望 {expect}）")
        if len(expect) and expect == (1920, 1080) and size == (1920 * 2 + 64) * 1080:
            check(res[0]["spec"].stride_bytes == 1920 * 2 + 64,
                  "带行 padding 的候选 stride 正确")


def test_legacy_image_loader_api():
    """旧的 utils.image_loader 接口必须继续可用（老脚本/测试依赖）。"""
    from utils.image_loader import apply_bayer_mask, demosaic_image, load_raw_image
    path = os.path.join(TMP, "legacy.raw")
    rng = np.random.default_rng(3)
    img = rng.integers(0, 1024, size=(16, 16)).astype(np.uint16)
    img.tofile(path)
    disp, raw = load_raw_image(path, 16, 16, 10)
    check(disp is not None and disp.shape == (16, 16) and disp.dtype == np.uint8,
          "load_raw_image 返回 8bit 显示图")
    check(raw.shape == (16, 16) and np.array_equal(raw, img), "load_raw_image 返回原始 DN")
    rgb = apply_bayer_mask(img, "RGGB", 10)
    check(rgb.shape == (16, 16, 3) and rgb[0, 0, 0] == disp[0, 0],
          "apply_bayer_mask 相位着色（R 在 (0,0)）")
    rgb2 = demosaic_image(img, "RGGB", 10)
    check(rgb2.shape == (16, 16, 3), "demosaic_image 形状")


if __name__ == "__main__":
    sys.exit(run(globals(), "RAW 读取层"))
