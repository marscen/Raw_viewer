"""CFA（Bayer）几何与通道拆分。

只依赖 numpy，供显示（utils/display.py）与统计（utils/stats.py）共用。
约定：
  * 通道编号 0=R, 1=G, 2=B；
  * 相位编号 phase = (y%2)*2 + (x%2)，对应 pattern 的四个采样点；
  * 4 个相位名：R / Gr / Gb / B，其中 Gr 是"位于 R 行"的绿点，
    Gb 是"位于 B 行"的绿点（这是 sensor 测试里判断 R/B 行噪声差异的基础）。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "PATTERNS", "PATTERN_NAMES", "PHASE_LABELS",
    "is_bayer", "normalize_pattern", "channel_index", "channel_map",
    "phase_map", "phase_names", "split_planes", "plane_name_at",
    "phase_channel", "phase_planes", "channel_gains",
]

PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

# pattern -> 四个相位（phase 0..3 = (0,0),(0,1),(1,0),(1,1)）的名字
PHASE_LABELS = {
    "RGGB": ("R", "Gr", "Gb", "B"),
    "BGGR": ("B", "Gb", "Gr", "R"),
    "GRBG": ("Gr", "R", "B", "Gb"),
    "GBRG": ("Gb", "B", "R", "Gr"),
}

PATTERN_NAMES = ("Mono/None",) + PATTERNS

# 每个相位的彩色通道：0=R 1=G 2=B
_PHASE_CHANNEL = {
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}


def normalize_pattern(pattern) -> str:
    """把用户/界面里的 pattern 字符串规整成 'RGGB' 之类，非法则返回 'Mono/None'。"""
    if pattern is None:
        return "Mono/None"
    p = str(pattern).strip().upper().replace(" ", "")
    if p in ("", "MONO", "MONO/NONE", "NONE", "MONOCHROME"):
        return "Mono/None"
    return p if p in PATTERNS else "Mono/None"


def is_bayer(pattern) -> bool:
    return normalize_pattern(pattern) in PATTERNS


def channel_index(x: int, y: int, pattern) -> int:
    """返回 (x, y) 处的颜色通道 0=R/1=G/2=B，非 Bayer 返回 -1。"""
    p = normalize_pattern(pattern)
    if p == "Mono/None":
        return -1
    return _PHASE_CHANNEL[p][(y % 2) * 2 + (x % 2)]


def phase_map(pattern, shape) -> np.ndarray:
    """(H, W) uint8，值为该像素的相位编号 0..3（phase = (y%2)*2 + x%2）。"""
    h, w = shape[0], shape[1]
    phases = (np.arange(h)[:, None] % 2) * 2 + (np.arange(w)[None, :] % 2)
    return phases.astype(np.uint8)


def channel_map(pattern, shape) -> np.ndarray:
    """(H, W) int8，值为颜色通道 0/1/2；Mono 时为 -1。"""
    p = normalize_pattern(pattern)
    if p == "Mono/None":
        return np.full(shape[:2], -1, dtype=np.int8)
    lut = np.array(_PHASE_CHANNEL[p], dtype=np.int8)
    return lut[phase_map(p, shape)]


def phase_names(pattern) -> dict:
    """{相位编号: 通道名}，例如 RGGB -> {0:'R',1:'Gr',2:'Gb',3:'B'}。"""
    p = normalize_pattern(pattern)
    if p == "Mono/None":
        return {0: "Mono"}
    return {i: name for i, name in enumerate(PHASE_LABELS[p])}


def split_planes(raw: np.ndarray, pattern) -> dict:
    """按相位拆成 4 个 (H/2, W/2) 平面。

    返回 {通道名: 平面}，Mono 时返回 {'Mono': raw}。
    这是所有分通道统计/噪声分析的入口。
    """
    p = normalize_pattern(pattern)
    if p == "Mono/None" or raw is None:
        return {"Mono": raw} if raw is not None else {}
    names = PHASE_LABELS[p]
    planes = {}
    for phase, name in enumerate(names):
        dy, dx = divmod(phase, 2)
        planes[name] = raw[dy::2, dx::2]
    return planes


def phase_planes(raw: np.ndarray, pattern, origin=(0, 0)) -> list:
    """按**全局相位**拆平面，返回 [{'name','plane','base_x','base_y','step'}]。

    子平面上的像素 (py, px) 对应全局坐标：
        x = base_x + step*px,  y = base_y + step*py
    这样即使 ROI 起点是奇数（从盘古坐标看相位被"平移"了），也不会串通道。
    Mono 时 step=1、base=origin。
    """
    p = normalize_pattern(pattern)
    ox, oy = int(origin[0]), int(origin[1])
    if p == "Mono/None":
        return [{"name": "Mono", "plane": raw, "base_x": ox, "base_y": oy, "step": 1}]
    out = []
    for phase, name in enumerate(PHASE_LABELS[p]):
        gy, gx = divmod(phase, 2)
        dy = (gy - oy) % 2
        dx = (gx - ox) % 2
        out.append({"name": name, "plane": raw[dy::2, dx::2],
                    "base_x": ox + dx, "base_y": oy + dy, "step": 2})
    return out


def plane_name_at(x: int, y: int, pattern) -> str:
    """(x, y) 属于哪个相位名（R/Gr/Gb/B），Mono 返回 'Mono'。"""
    p = normalize_pattern(pattern)
    if p == "Mono/None":
        return "Mono"
    return PHASE_LABELS[p][(y % 2) * 2 + (x % 2)]


def phase_channel(pattern):
    """4 个相位对应的颜色通道 (0=R,1=G,2=B)，顺序为 (0,0),(0,1),(1,0),(1,1)。"""
    p = normalize_pattern(pattern)
    if p == "Mono/None":
        return (0, 0, 0, 0)
    return _PHASE_CHANNEL[p]


def channel_gains(pattern) -> dict:
    """各相位的等效曝光增益（此处恒为 1，占位以便将来接 WB 增益）。"""
    return {name: 1.0 for name in phase_names(pattern).values()}
