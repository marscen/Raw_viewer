"""导出层测试：16bit PNG/TIFF 无损往返、CSV、标注截图。"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests._util import check, run  # noqa: E402

from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from utils import export as X  # noqa: E402

APP = QApplication.instance() or QApplication([])
TMP = tempfile.mkdtemp(prefix="rawv2_exp_")


def _load_qimage(path):
    img = QImage(path)
    if img.isNull():
        return None, None
    fmt = img.format()
    if fmt != QImage.Format.Format_Grayscale8 and fmt != QImage.Format.Format_Grayscale16:
        img = img.convertToFormat(QImage.Format.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    buf = buf.reshape(h, bpl)
    if img.format() == QImage.Format.Format_Grayscale16:
        arr = buf[:, :2 * w].copy().view(np.uint16).reshape(h, w)
        return "gray16", np.ascontiguousarray(arr)
    if img.format() == QImage.Format.Format_Grayscale8:
        return "gray8", np.ascontiguousarray(buf[:, :w])
    return "rgb8", np.ascontiguousarray(buf[:, :3 * w].reshape(h, w, 3))


def test_grayscale16_lossless():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 65536, (37, 29)).astype(np.uint16)
    for ext in ("png", "tif"):
        path = os.path.join(TMP, f"g16.{ext}")
        X.write_image(path, arr)
        kind, back = _load_qimage(path)
        check(kind == "gray16", f"{ext}: 读回为 16bit 灰度（实际 {kind}）")
        check(back.shape == arr.shape and np.array_equal(back, arr),
              f"{ext}: 16bit 无损往返（maxdiff="
              f"{np.abs(back.astype(int) - arr.astype(int)).max() if back is not None else 'n/a'}）")


def test_grayscale8_and_rgb_lossless():
    rng = np.random.default_rng(1)
    g8 = rng.integers(0, 256, (23, 41)).astype(np.uint8)
    rgb = rng.integers(0, 256, (23, 41, 3)).astype(np.uint8)
    for ext in ("png", "tif"):
        p1 = os.path.join(TMP, f"g8.{ext}")
        X.write_image(p1, g8)
        kind, back = _load_qimage(p1)
        check(kind == "gray8" and np.array_equal(back, g8), f"{ext}: 8bit 灰度无损")
        p2 = os.path.join(TMP, f"rgb.{ext}")
        X.write_image(p2, rgb)
        kind, back = _load_qimage(p2)
        check(kind == "rgb8" and np.array_equal(back, rgb), f"{ext}: RGB8 无损")


def test_png_dimensions_and_signature():
    arr = np.zeros((17, 33), np.uint16)
    path = os.path.join(TMP, "sig.png")
    X.write_png(path, arr)
    with open(path, "rb") as fh:
        head = fh.read(33)
    check(head[:8] == b"\x89PNG\r\n\x1a\n", "PNG 签名")
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    check((w, h) == (33, 17), f"PNG IHDR 尺寸 ({w},{h})")
    check(head[24] == 16 and head[25] == 0, "PNG 位深 16 / 灰度 color type 0")


def test_tiff_header_structure():
    arr = np.zeros((11, 13), np.uint16)
    path = os.path.join(TMP, "hdr.tif")
    X.write_tiff(path, arr)
    with open(path, "rb") as fh:
        data = fh.read()
    check(data[:2] == b"II" and int.from_bytes(data[2:4], "little") == 42, "TIFF 头")
    ifd = int.from_bytes(data[4:8], "little")
    n = int.from_bytes(data[ifd:ifd + 2], "little")
    check(n == 11, f"IFD 标签数 {n}")
    tags = {}
    for i in range(n):
        off = ifd + 2 + i * 12
        tag = int.from_bytes(data[off:off + 2], "little")
        typ = int.from_bytes(data[off + 2:off + 4], "little")
        cnt = int.from_bytes(data[off + 4:off + 8], "little")
        val = int.from_bytes(data[off + 8:off + 12], "little")
        tags[tag] = (typ, cnt, val)
    check(tags[256][2] == 13 and tags[257][2] == 11, "TIFF 宽高标签")
    check(tags[258][2] == 16, "TIFF BitsPerSample=16")
    check(tags[277][2] == 1 and tags[262][2] == 1, "TIFF 单通道黑底")
    check(list(tags) == sorted(tags), "IFD 标签升序（规范要求）")


def test_csv_writers():
    from utils import stats as S
    raw = np.full((32, 32), 100, np.uint16)
    st = S.roi_stats(raw, "Mono/None", (0, 0, 16, 16), 10)
    header, rows = X.stats_to_rows(st)
    path = os.path.join(TMP, "stats.csv")
    X.write_csv(path, header, rows)
    text = open(path, encoding="utf-8-sig").read().splitlines()
    check(text[0].split(",")[:4] == ["scope", "channel", "count", "mean"], "统计 CSV 表头")
    check(any("roi" in line for line in text), "统计 CSV 含 ROI 行")
    check(any("Mono" in line for line in text), "统计 CSV 含通道行")

    defects = [{"type": "hot", "x": 3, "y": 4, "channel": "R", "value": 1023,
                "delta": 900.0, "note": "single", "source": "Bad Pixel Detection"}]
    header, rows = X.defects_to_rows(defects)
    path = os.path.join(TMP, "def.csv")
    X.write_csv(path, header, rows)
    text = open(path, encoding="utf-8-sig").read().splitlines()
    check(text[0] == "type,x,y,channel,value,delta,note,source", "缺陷 CSV 表头（含来源列）")
    check(text[1].startswith("hot,3,4,R,1023"), f"缺陷 CSV 内容: {text[1]}")


def test_draw_defects_marks_pixels():
    base = np.full((40, 40, 3), 30, np.uint8)
    defects = [{"type": "hot", "x": 20, "y": 20}]
    out = X.draw_defects(base, defects, radius=4, line_width=1)
    check(out.shape == base.shape, "标注图尺寸不变")
    check(np.array_equal(base[20, 20], np.array([30, 30, 30], np.uint8)), "原图未被修改")
    check(out[16, 20, 0] == 255 and out[16, 20, 2] == 64, "hot 用红色方框标出")
    check(out[20, 16, 0] == 255, "方框左边也画上")
    check(np.array_equal(out[0, 0], base[0, 0]), "远处像素不受影响")
    # 越界坐标不应崩
    out2 = X.draw_defects(base, [{"type": "dead", "x": -5, "y": 100}], radius=3)
    check(out2.shape == base.shape, "越界坐标安全")
    # 灰度输入自动转 RGB
    out3 = X.draw_defects(np.zeros((10, 10), np.uint8), [{"type": "hot", "x": 5, "y": 5}],
                          radius=2)
    check(out3.ndim == 3 and out3.shape[2] == 3, "灰度输入自动扩成 RGB")


def test_write_image_dispatch():
    arr = np.zeros((8, 8), np.uint8)
    for name, expect16 in (("a.png", False), ("a.tif", False), ("a.tiff", False)):
        path = os.path.join(TMP, name)
        X.write_image(path, arr)
        check(os.path.getsize(path) > 8, f"{name} 写出成功")
    arr16 = np.zeros((8, 8), np.uint16)
    path = os.path.join(TMP, "b.tiff")
    X.write_image(path, arr16)
    kind, back = _load_qimage(path)
    check(kind == "gray16", "uint16 走 TIFF 分支")


if __name__ == "__main__":
    sys.exit(run(globals(), "导出层"))
