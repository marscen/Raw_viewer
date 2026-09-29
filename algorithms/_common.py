"""算法公用工具：同色邻域、鲁棒噪声估计、连通标记、坐标映射。

关键概念：对 Bayer 数据做缺陷检测必须"同色比较"——R 只能和 R 比、Gr 只能和
Gr/Gb 比，否则 R/B 行之间天然 1000+ DN 的差异会淹没真实坏点。这里统一用
cfa.phase_planes() 把数据按**全局相位**拆成子平面，子平面内相邻采样点即同色，
再把子平面坐标映射回全局坐标（兼容奇数起点的 ROI）。
"""
from __future__ import annotations

import numpy as np

from utils import cfa

__all__ = [
    "plane_views", "local_median", "robust_sigma", "label_mask", "dilate",
    "defect_overlay", "crop_with_origin", "paste_back",
]


def crop_with_origin(image, roi):
    """按 ROI 裁剪，返回 (crop, origin)。roi 为 None 时返回整图与 (0, 0)。"""
    if roi is None:
        return image, (0, 0)
    x0, y0, x1, y1 = (int(v) for v in roi)
    h, w = image.shape
    x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
    return image[y0:y1, x0:x1], (x0, y0)


def paste_back(base, patch, origin):
    """把 ROI 结果贴回整图（返回新数组）。"""
    out = base.copy()
    x0, y0 = origin
    out[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch
    return out


def plane_views(image, pattern, origin=(0, 0)):
    """按全局相位拆平面；返回 [{'name','plane','base_x','base_y','step'}]。"""
    return cfa.phase_planes(image, pattern, origin)


def local_median(plane: np.ndarray, neighbors: int = 4) -> np.ndarray:
    """同色邻域中值（边界用 edge 复制填充，避免漏检画幅边缘的坏点）。"""
    p = plane.astype(np.float32)
    padded = np.pad(p, ((1, 1), (1, 1)), mode="edge")
    h, w = plane.shape
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(neighbors) >= 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    stack = np.stack([padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                      for dy, dx in offsets])
    return np.median(stack, axis=0)


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 * MAD，对离群坏点不敏感的噪声估计。"""
    v = np.ravel(np.asarray(values, dtype=np.float32))
    if v.size == 0:
        return 0.0
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def label_mask(mask: np.ndarray, connectivity: int = 8, max_labels: int = 200000):
    """连通标记（BFS），返回 (labels, count)。

    缺陷数量通常不多，BFS 足够快；为避免病态输入卡死，超过 max_labels 就停。
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if not mask.any():
        return labels, 0
    if connectivity >= 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    from collections import deque
    ys, xs = np.nonzero(mask)
    cur = 0
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0, x0]:
            continue
        cur += 1
        if cur > max_labels:
            break
        q = deque([(y0, x0)])
        labels[y0, x0] = cur
        while q:
            y, x = q.popleft()
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = cur
                    q.append((ny, nx))
    return labels, cur


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3x3 形态学膨胀（纯 numpy）。

    用途：坏点簇判定。同色相邻坏点在整幅图上相隔 2 像素，2x2 坏点块则分布
    在四个不同相位；先膨胀再连通标记，才能把这两种情况都归成一个簇。
    """
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
               p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
               p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return out


def defect_overlay(defect: dict) -> dict:
    """缺陷记录 -> 画布叠加层（point）。"""
    return {
        "type": "point",
        "coords": (int(defect["x"]), int(defect["y"])),
        "kind": defect.get("type", "hot"),
        "color": defect.get("type", "hot"),
        "radius": int(defect.get("radius", 3)),
    }
