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
    """连通标记（行程 + 并查集），返回 (labels, count)。

    为什么不用逐像素 BFS：BFS 在 Python 里是逐像素推进的，一条坏列(3000 像素)
    要 25 ms、一个 100 万像素的过曝块要 **2.6 秒**（GUI 直接卡死）。这里改成
    经典的"按行取行程 + 并查集合并相邻行"：整体是向量化取行程 + 行程数量级的
    Python 循环，同样的 100 万像素块只需几十毫秒。

    labels 为 int32、0 为背景、区域从 1 开始按行优先首次出现顺序编号。
    """
    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if not mask.any():
        return labels, 0

    # 1) 逐行提取连续行程：差分找上升/下降沿（全向量化）
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:w + 1] = mask
    diff = np.diff(padded, axis=1)
    ys_s, xs_s = np.nonzero(diff == 1)          # 行程起点 (y, x0)
    ys_e, xs_e = np.nonzero(diff == -1)         # 行程终点 (y, x1)（不含）
    runs_y, runs_x0, runs_x1 = ys_s, xs_s, xs_e
    n_runs = runs_y.size

    # 2) 并查集（按行程编号）
    parent = np.arange(n_runs, dtype=np.int64)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:                # 路径压缩
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 按行分组的行程编号（np.nonzero 已是行优先，行内按列递增）
    row_slices = {}
    for idx in range(n_runs):
        row_slices.setdefault(int(runs_y[idx]), []).append(idx)

    # 连通容差：8 邻域允许两行之间"列号相差 1"，4 邻域必须列区间有重叠。
    # 行程是左闭右开的 [x0, x1)，所以"a 完全在 b 左侧"的判据是
    #     b0 - (a1 - 1) > tol   <=>   b0 - a1 + 1 > tol
    tol = 1 if int(connectivity) >= 8 else 0
    for y, cur in row_slices.items():
        nxt = row_slices.get(y + 1)
        if not nxt:
            continue
        i = j = 0
        while i < len(cur) and j < len(nxt):
            a, b = cur[i], nxt[j]
            a0, a1 = runs_x0[a], runs_x1[a]
            b0, b1 = runs_x0[b], runs_x1[b]
            if b0 - a1 + 1 > tol:               # a 完全在 b 左侧
                i += 1
            elif a0 - b1 + 1 > tol:             # b 完全在 a 左侧
                j += 1
            else:
                union(a, b)
                if a1 <= b1:                    # 谁先结束谁前进，另一个留着继续比
                    i += 1
                else:
                    j += 1

    # 3) 根 -> 新编号（按行程顺序 = 行优先首次出现，保证结果稳定可复现）
    root_to_label = {}
    run_label = np.zeros(n_runs, dtype=np.int64)
    next_label = 0
    for idx in range(n_runs):
        root = find(idx)
        label = root_to_label.get(root)
        if label is None:
            next_label += 1
            if next_label > max_labels:
                next_label -= 1
                label = 0
            else:
                label = next_label
            root_to_label[root] = label
        run_label[idx] = label

    # 4) 把行程编号刷回像素（按行程做切片赋值，速度快）
    for idx in range(n_runs):
        lab = run_label[idx]
        if lab:
            labels[runs_y[idx], runs_x0[idx]:runs_x1[idx]] = lab
    return labels, int(next_label)


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
