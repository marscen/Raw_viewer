"""加速层测试：OpenCV 路径与 numpy 回退**都要对**。

同一套断言会在两种后端下各跑一遍（tests/run_all.py 会用 RAWV2_NO_CV2=1
再跑一次），重点验证两条路径结果一致，而不是"有 cv2 就跳过"。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests._util import check, run  # noqa: E402

from utils import accel, cfa  # noqa: E402

BACKEND = accel.backend_name()
print(f"（当前后端: {BACKEND}）")


def _flat(pattern, shape=(32, 32)):
    vals = {"R": 800, "Gr": 400, "Gb": 400, "B": 200}
    img = np.zeros(shape, np.uint16)
    names = cfa.PHASE_LABELS[pattern]
    for ph, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        img[dy::2, dx::2] = vals[names[ph]]
    return img


def test_demosaic_channel_mapping():
    """cv2 的 Bayer 常量与我们的 pattern 是错位对应的，必须逐相位验证。"""
    for pattern in cfa.PATTERNS:
        rgb = accel.demosaic(_flat(pattern), pattern)
        inner = np.asarray(rgb[12:20, 12:20], dtype=np.float64).reshape(-1, 3).mean(axis=0)
        check(abs(inner[0] - 800) < 1.5 and abs(inner[1] - 400) < 1.5 and abs(inner[2] - 200) < 1.5,
              f"{pattern}: demosaic 通道应为 (800,400,200)，实际 "
              f"({inner[0]:.1f},{inner[1]:.1f},{inner[2]:.1f})")


def test_demosaic_consistency_and_edges():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4096, (64, 64)).astype(np.uint16)
    rgb = accel.demosaic(img, "RGGB")
    check(rgb.shape == (64, 64, 3), f"demosaic 形状 {rgb.shape}")
    check(np.asarray(rgb).dtype in (np.uint8, np.uint16, np.float32),
          f"demosaic dtype {np.asarray(rgb).dtype}")
    check(float(np.min(rgb)) >= 0 and float(np.max(rgb)) <= 4096,
          "demosaic 不应产生超出输入范围的数值")
    try:
        accel.demosaic(img, "Mono/None")
        check(False, "非 Bayer pattern 应报错")
    except ValueError:
        check(True, "非 Bayer pattern 报错")
    try:
        accel.demosaic(img, "NOPE")
        check(False, "非法 pattern 应报错")
    except ValueError:
        check(True, "非法 pattern 报错")


def test_connected_components_matches_reference():
    """与逐像素 BFS 参考实现比对（分区必须完全一致，编号也一致）。"""
    from collections import deque

    def bfs(mask, conn=8):
        h, w = mask.shape
        lab = np.zeros((h, w), np.int32)
        cur = 0
        nbrs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                if conn >= 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)])
        for y0, x0 in zip(*np.nonzero(mask)):
            if lab[y0, x0]:
                continue
            cur += 1
            q = deque([(y0, x0)])
            lab[y0, x0] = cur
            while q:
                y, x = q.popleft()
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        q.append((ny, nx))
        return lab, cur

    rng = np.random.default_rng(1)
    masks = [rng.random((50, 60)) < d for d in (0.005, 0.05, 0.3)]
    m = np.zeros((20, 30), bool)
    m[:, 5] = True
    m[19, :] = True
    masks.append(m)
    for conn in (8, 4):
        for i, mask in enumerate(masks):
            labels, sizes = accel.connected_components(mask, conn)
            ref, nref = bfs(mask, conn)
            mapping, same = {}, True
            for a, b in zip(labels.ravel(), ref.ravel()):
                if b == 0:
                    continue
                if mapping.setdefault(b, a) != a:
                    same = False
                    break
            check(same and int(labels.max()) == nref,
                  f"conn={conn} case{i}: 连通分区与参考实现一致")
            real = np.bincount(labels[mask], minlength=int(labels.max()) + 1)
            check(np.array_equal(real, sizes[:len(real)]),
                  f"conn={conn} case{i}: sizes 与标签一致")


def test_connected_components_large_blob_fast():
    """大块缺陷不能退化成逐像素 BFS（曾经 100 万像素要 2.6 秒）。"""
    mask = np.zeros((1500, 2000), bool)
    mask[100:600, 200:1200] = True          # 50 万像素
    t = time.perf_counter()
    labels, sizes = accel.connected_components(mask, 8)
    dt = time.perf_counter() - t
    check(sizes.max() == 500000, f"大块面积 {sizes.max()}")
    check(dt < 2.0, f"50 万像素连通标记耗时 {dt * 1000:.0f} ms（应 < 2 s）")


def test_morphology_and_fallback_equivalence():
    m = np.zeros((64, 64), bool)
    m[10:20, 10:20] = True
    m[40, 40] = True
    check(int(accel.dilate_mask(m).sum()) == 153, "膨胀像素数")
    check(int(accel.erode_mask(m).sum()) == 64, "腐蚀像素数")
    check(int(accel.open_mask(m).sum()) == 100, "开运算保留方块")
    # 与 numpy 回退实现比对
    check(np.array_equal(accel.dilate_mask(m, 2), accel._dilate_np(m, 2)),
          "cv2 与 numpy 膨胀结果一致")
    check(np.array_equal(accel.erode_mask(m, 1), accel._erode_np(m, 1)),
          "cv2 与 numpy 腐蚀结果一致")


def test_clahe():
    g = np.tile(np.linspace(40, 60, 256, dtype=np.uint8), (256, 1))
    out = accel.clahe(g, 2.0, 8)
    check(out.shape == g.shape and out.dtype == np.uint8, "CLAHE 形状/dtype")
    check(float(out.std()) > 0, "CLAHE 输出有对比度")
    flat = np.full((64, 64), 128, np.uint8)
    check(accel.clahe(flat).shape == (64, 64), "纯色输入不崩")


def test_phase_correlate_sign_and_accuracy():
    """符号约定必须与 cv2 一致：返回 b(x+dx, y+dy) ≈ a(x, y)。"""
    a = np.zeros((128, 128), np.float32)
    a[40:80, 40:80] = 1.0
    for dy, dx in ((5, -3), (-4, 2), (3, 3), (0, 7)):
        b = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        res = accel.phase_correlate(a, b)
        check(abs(res["dx"] - dx) < 0.2 and abs(res["dy"] - dy) < 0.2,
              f"roll(dy={dy},dx={dx}) -> ({res['dx']:+.2f},{res['dy']:+.2f})")
    check(accel.phase_correlate(a, np.zeros_like(a)) is not None, "全零参考不崩")
    check(accel.phase_correlate(a, a[:64]) is None, "尺寸不一致返回 None")


def test_shift_image():
    a = np.zeros((64, 64), np.uint16)
    a[20:30, 20:30] = 1000
    s = accel.shift_image(a, 5, -3)
    check(s.shape == a.shape and s.dtype == np.uint16, "平移后形状/dtype")
    check(int(s[17:27, 25:35].sum()) > 0, "内容确实移动了")
    check(np.array_equal(accel.shift_image(a, 0, 0), a), "零平移返回原图")
    # 平移后再相关应能测回来
    res = accel.phase_correlate(a, accel.shift_image(a, 4, -6))
    check(abs(res["dx"] - 4) < 0.5 and abs(res["dy"] + 6) < 0.5,
          f"shift 后可复原位移 ({res['dx']:+.2f},{res['dy']:+.2f})")


def test_resize_area_and_backend_info():
    img = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    out = accel.resize_area(img, 16, 16)
    if accel.available():
        check(out is not None and out.shape == (16, 16), "resize_area 输出")
    else:
        check(out is None, "无 OpenCV 时 resize_area 返回 None（由 Qt 回退）")
    check("OpenCV" in accel.backend_info() or "numpy" in accel.backend_info(),
          f"后端说明: {accel.backend_info()}")


if __name__ == "__main__":
    sys.exit(run(globals(), f"加速层 ({BACKEND})"))
