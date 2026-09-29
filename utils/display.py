"""显示管线：RAW DN -> 屏幕像素。

测试工程师真正需要看的不是"原始 DN 直接线性映射"：暗场只有 0~100 DN、
亮场可能饱和、坏点只有几个 DN 的差异。所以这里把显示拆成可控的几步：

    DN --(黑/白电平)--> 归一化 --(拉伸模式)--> [0,1] --(gamma)--> 8bit
       --(伪彩/通道着色/单通道遮罩)--> RGB 或灰度

关键点：
  * 所有 DN->8bit 线性部分都是一张 LUT（10bit 只要 1024 项，16bit 65536 项），
    调滑块时对全图只是一次 numpy 花式索引，帧率够用；
  * 电平（lo/hi）默认按画面百分位自动求，避免暗场全黑、亮场全白；
  * 单通道视图（R/Gr/Gb/B plane）保持与 raw 相同的 HxW 坐标（非本相位像素置 0），
    这样 ROI、悬浮取点、像素检查器的坐标永远和 raw 一致，坏点定位不会错位。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from utils import cfa

__all__ = [
    "VIEW_MODES", "STRETCH_MODES", "COLORMAPS",
    "DisplayParams", "RenderInfo", "render_display",
    "compute_levels", "gray_from_codes", "apply_colormap",
    "bayer_colorize", "demosaic", "PLANE_VIEWS",
]

VIEW_MODES = [
    "Bayer Mosaic",     # 马赛克原貌，R/G/B 三色标识采样相位
    "Bayer Demosaic",   # 双线性插值彩色
    "Mono",             # 灰度（不区分 CFA）
    "R plane",          # 只显示 R 相位（其余相位置黑）
    "Gr plane",
    "Gb plane",
    "G plane",          # 两个绿相位一起显示
    "B plane",
]

PLANE_VIEWS = {
    "R plane": ("R",),
    "Gr plane": ("Gr",),
    "Gb plane": ("Gb",),
    "B plane": ("B",),
    "G plane": ("Gr", "Gb"),
}

STRETCH_MODES = [
    "Fixed (black/white)",   # 手动黑/白电平
    "Min-Max",               # 当前画面最小/最大
    "Percentile",            # 默认：按百分位裁掉两端极值
    "Sigma",                 # 均值 ± k*标准差
]

COLORMAPS = ["Gray", "Jet", "Turbo", "Hot", "Viridis"]


@dataclass
class DisplayParams:
    """一次渲染需要的全部显示参数（可整体存入设置）。"""

    bit_depth: int = 10
    view: str = "Bayer Mosaic"
    pattern: str = "RGGB"
    black_level: float = 0.0
    white_level: float = 0.0          # 0 -> 用 (2**bit_depth - 1)
    stretch: str = "Percentile"
    p_low: float = 0.5
    p_high: float = 99.5
    sigma_k: float = 3.0
    gamma: float = 1.0
    invert: bool = False
    colormap: str = "Gray"
    stretch_on_roi: bool = False      # 电平按 ROI 计算（ROI 内有强反光时有用）

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DisplayParams":
        known = {f for f in cls().to_dict()}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class RenderInfo:
    """渲染结果附带的测量信息（回填到侧栏/状态栏）。"""

    lo: float = 0.0
    hi: float = 1.0
    view: str = "Mono"
    is_color: bool = False
    masked: bool = False        # 是否用了单通道掩罩显示

    def describe(self) -> str:
        return f"{self.view} | level {self.lo:.0f}~{self.hi:.0f}"


# ----------------------------------------------------------------------
# 电平
# ----------------------------------------------------------------------
def _percentile_levels(vals: np.ndarray, p_low: float, p_high: float):
    vals = np.ravel(vals)
    if vals.size == 0:
        return 0.0, 1.0
    if vals.size > 4_000_000:                      # 大图抽样，保证交互流畅
        step = max(1, vals.size // 1_000_000)
        vals = vals[::step]
    lo, hi = np.percentile(vals, [p_low, p_high])
    return float(lo), float(hi)


def compute_levels(codes: np.ndarray, params: DisplayParams):
    """按拉伸模式算出 (lo, hi) 两个 DN 电平。codes 为待显示像素的 DN。"""
    max_code = float((1 << int(params.bit_depth)) - 1)
    white = float(params.white_level) if params.white_level else max_code
    if params.stretch == "Fixed (black/white)":
        lo, hi = float(params.black_level), white
    elif codes is None or codes.size == 0:
        lo, hi = float(params.black_level), white
    else:
        sample = codes
        if params.stretch == "Min-Max":
            lo, hi = float(np.min(sample)), float(np.max(sample))
        elif params.stretch == "Sigma":
            mu, sd = float(np.mean(sample)), float(np.std(sample))
            k = float(params.sigma_k)
            lo, hi = mu - k * sd, mu + k * sd
        else:                                       # Percentile（默认）
            lo, hi = _percentile_levels(sample, params.p_low, params.p_high)
            lo = max(lo, float(params.black_level))
            # 暗场/亮场里极少数坏点会让百分位区间退化成一个点（整幅变纯黑或纯白），
            # 此时退回 mean±3σ，既能看到噪声底，坏点也依旧饱和可见。
            if (hi - lo) < 0.01 * max_code:
                mu, sd = float(np.mean(sample)), float(np.std(sample))
                if sd > 0:
                    lo, hi = mu - 3.0 * sd, mu + 3.0 * sd
                else:
                    lo, hi = float(np.min(sample)), float(np.max(sample))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = max_code
    if hi <= lo:
        hi = lo + 1.0
    # 极端均匀的画面（例如纯黑参考帧）也要给出可看的中间灰，而不是全黑
    min_span = 0.01 * max_code
    if (hi - lo) < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
        lo = max(0.0, lo)
    return float(lo), float(hi)


# ----------------------------------------------------------------------
# LUT / 灰度
# ----------------------------------------------------------------------
def build_lut(params: DisplayParams, lo: float, hi: float) -> np.ndarray:
    """DN -> 8bit 的查找表（含 gamma / invert）。"""
    max_code = (1 << int(params.bit_depth)) - 1
    codes = np.arange(max_code + 1, dtype=np.float32)
    norm = (codes - lo) / max(hi - lo, 1e-6)
    np.clip(norm, 0.0, 1.0, out=norm)
    gamma = float(params.gamma) if params.gamma and params.gamma > 0 else 1.0
    if gamma != 1.0:
        norm = np.power(norm, 1.0 / gamma)
    if params.invert:
        norm = 1.0 - norm
    return np.rint(norm * 255.0).astype(np.uint8)


def gray_from_codes(codes: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """用 LUT 把 DN 映射成 8bit 灰度（自动按位深裁剪索引）。"""
    if codes is None:
        return None
    idx = codes.astype(np.uint32, copy=False)
    np.clip(idx, 0, len(lut) - 1, out=idx)
    return lut[idx]


# ----------------------------------------------------------------------
# 伪彩
# ----------------------------------------------------------------------
# 控制点插值出来的 colormap，够用且不引入 matplotlib 依赖
_COLORMAP_POINTS = {
    "Gray": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "Jet": [(0.0, (0, 0, 128)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
            (0.625, (255, 255, 0)), (0.875, (255, 0, 0)), (1.0, (128, 0, 0))],
    "Turbo": [(0.0, (48, 18, 59)), (0.1, (70, 107, 227)), (0.2, (33, 181, 236)),
              (0.3, (36, 229, 168)), (0.4, (155, 246, 78)), (0.5, (232, 225, 53)),
              (0.6, (253, 169, 44)), (0.7, (245, 105, 26)), (0.8, (209, 55, 12)),
              (0.9, (128, 26, 4)), (1.0, (122, 4, 3))],
    "Hot": [(0.0, (0, 0, 0)), (0.365, (255, 0, 0)), (0.746, (255, 255, 0)),
            (1.0, (255, 255, 255))],
    "Viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
                (0.75, (94, 201, 98)), (1.0, (253, 231, 37))],
}

_CMAP_CACHE: dict = {}


def colormap_lut(name: str) -> np.ndarray:
    """(256, 3) uint8 伪彩表。"""
    key = str(name or "Gray")
    if key in _CMAP_CACHE:
        return _CMAP_CACHE[key]
    pts = _COLORMAP_POINTS.get(key, _COLORMAP_POINTS["Gray"])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    cs = np.array([p[1] for p in pts], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(grid, xs, cs[:, i]) for i in range(3)], axis=1)
    lut = np.rint(lut).astype(np.uint8)
    _CMAP_CACHE[key] = lut
    return lut


def apply_colormap(gray8: np.ndarray, name: str) -> np.ndarray:
    """灰度 8bit -> (H, W, 3) RGB。Gray 直接复制三通道。"""
    cmap = colormap_lut(name)
    return cmap[gray8]


# ----------------------------------------------------------------------
# Bayer 着色 / demosaic
# ----------------------------------------------------------------------
def bayer_colorize(gray8: np.ndarray, pattern, intensity=None) -> np.ndarray:
    """马赛克视图：按相位把灰度值染成 R/G/B 三色。

    intensity 给定时用其做亮度权重（默认用 gray8 本身），
    这样暗场里坏点依旧是亮的彩色点，而不是满屏彩色噪点。
    """
    p = cfa.normalize_pattern(pattern)
    h, w = gray8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if p == "Mono/None":
        return np.repeat(gray8[:, :, None], 3, axis=2)
    intensity = gray8 if intensity is None else intensity
    phase_col = cfa.phase_channel(p)
    for phase, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        col = phase_col[phase]
        rgb[dy::2, dx::2, col] = intensity[dy::2, dx::2]
    return rgb


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """空出来的边补 0 的二维平移（插值用）。"""
    res = np.zeros_like(arr)
    h, w = arr.shape
    ys, ye = max(0, dy), h - max(0, -dy)
    xs, xe = max(0, dx), w - max(0, -dx)
    if ye > ys and xe > xs:
        res[ys:ye, xs:xe] = arr[ys - dy:ye - dy, xs - dx:xe - dx]
    return res


def demosaic(norm: np.ndarray, pattern) -> np.ndarray:
    """双线性插值 demosaic，输入/输出都是 float32 的 [0,1] 或 DN 值。

    输入必须是**按相位排布的原始采样**（未归一化也可，值域一致即可）。
    """
    p = cfa.normalize_pattern(pattern)
    h, w = norm.shape
    if p == "Mono/None":
        return np.repeat(norm[:, :, None], 3, axis=2)

    names = cfa.PHASE_LABELS[p]
    masks = {name: np.zeros((h, w), dtype=bool) for name in ("R", "Gr", "Gb", "B")}
    for phase, name in enumerate(names):
        dy, dx = divmod(phase, 2)
        masks[name][dy::2, dx::2] = True

    work = norm.astype(np.float32, copy=False)
    R = np.where(masks["R"], work, 0.0)
    G = np.where(masks["Gr"] | masks["Gb"], work, 0.0)
    B = np.where(masks["B"], work, 0.0)

    # 1. 绿：在 R/B 位置取四邻域均值
    g_sum = _shift(G, -1, 0) + _shift(G, 1, 0) + _shift(G, 0, -1) + _shift(G, 0, 1)
    rb = masks["R"] | masks["B"]
    G[rb] = g_sum[rb] / 4.0

    # 2. 红/蓝：先在对角位置取对角均值
    r_diag = _shift(R, -1, -1) + _shift(R, -1, 1) + _shift(R, 1, -1) + _shift(R, 1, 1)
    R[masks["B"]] = r_diag[masks["B"]] / 4.0
    b_diag = _shift(B, -1, -1) + _shift(B, -1, 1) + _shift(B, 1, -1) + _shift(B, 1, 1)
    B[masks["R"]] = b_diag[masks["R"]] / 4.0

    # 3. 红/蓝在绿位置：同行取水平邻居，异行取垂直邻居
    r_h = (_shift(R, 0, -1) + _shift(R, 0, 1)) / 2.0
    r_v = (_shift(R, -1, 0) + _shift(R, 1, 0)) / 2.0
    b_h = (_shift(B, 0, -1) + _shift(B, 0, 1)) / 2.0
    b_v = (_shift(B, -1, 0) + _shift(B, 1, 0)) / 2.0

    # 按定义 Gr 就是"R 行里的绿"，Gb 是"B 行里的绿"，四种 pattern 都统一
    r_row_g, b_row_g = masks["Gr"], masks["Gb"]

    R[r_row_g] = r_h[r_row_g]
    R[b_row_g] = r_v[b_row_g]
    B[b_row_g] = b_h[b_row_g]
    B[r_row_g] = b_v[r_row_g]

    return np.stack((R, G, B), axis=2)


# ----------------------------------------------------------------------
# 主渲染入口
# ----------------------------------------------------------------------
def _plane_selection(raw: np.ndarray, params: DisplayParams, view: str):
    """单通道视图 -> (待显示像素值, 掩罩 bool, 视图名)。"""
    p = cfa.normalize_pattern(params.pattern)
    if p == "Mono/None":
        return raw, None, "Mono"
    names = cfa.PHASE_LABELS[p]
    wanted = PLANE_VIEWS[view]
    mask = np.zeros(raw.shape, dtype=bool)
    for phase, name in enumerate(names):
        if name in wanted:
            dy, dx = divmod(phase, 2)
            mask[dy::2, dx::2] = True
    return raw, mask, view


def render_display(raw: np.ndarray, params: DisplayParams, roi=None):
    """把 RAW DN 渲染成可直接贴到 QImage 的 uint8 数组。

    raw: (H, W) 整数数组（raw_io.load_raw 的输出，已是真实 DN）。
    roi: 可选 (x0, y0, x1, y1)，仅当 params.stretch_on_roi 时用于电平统计。
    返回 (image, RenderInfo)：image 为 (H, W) 灰度或 (H, W, 3) RGB。
    """
    if raw is None:
        return None, RenderInfo()
    view = str(params.view or "Mono")
    is_plane = view in PLANE_VIEWS

    codes = raw
    mask = None
    view_name = view
    if is_plane:
        codes, mask, view_name = _plane_selection(raw, params, view)

    # 电平统计用的样本
    sample_src = codes
    if mask is not None:
        sample_src = codes[mask]
    elif params.stretch_on_roi and roi is not None:
        x0, y0, x1, y1 = (int(v) for v in roi)
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(raw.shape[1], max(x1, x0 + 1))
        y1 = min(raw.shape[0], max(y1, y0 + 1))
        sample_src = codes[y0:y1, x0:x1]

    lo, hi = compute_levels(sample_src, params)
    lut = build_lut(params, lo, hi)

    # 彩色路径与灰度路径分开处理
    if view == "Bayer Mosaic":
        gray = gray_from_codes(codes, lut)
        if mask is not None:                      # 理论上不会走到
            gray = np.where(mask, gray, 0).astype(np.uint8)
        return bayer_colorize(gray, params.pattern), RenderInfo(
            lo=lo, hi=hi, view=view_name, is_color=True)

    if view == "Bayer Demosaic":
        norm = (codes.astype(np.float32) - lo) / max(hi - lo, 1e-6)
        np.clip(norm, 0.0, 1.0, out=norm)
        gamma = float(params.gamma) if params.gamma and params.gamma > 0 else 1.0
        if gamma != 1.0:
            norm = np.power(norm, 1.0 / gamma)
        if params.invert:
            norm = 1.0 - norm
        rgb = demosaic(norm, params.pattern)
        rgb8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb8, RenderInfo(lo=lo, hi=hi, view=view_name, is_color=True)

    gray = gray_from_codes(codes, lut)
    if mask is not None:
        gray = np.where(mask, gray, 0).astype(np.uint8)

    if str(params.colormap) != "Gray":
        return apply_colormap(gray, params.colormap), RenderInfo(
            lo=lo, hi=hi, view=view_name, is_color=True, masked=mask is not None)

    return gray, RenderInfo(lo=lo, hi=hi, view=view_name,
                            is_color=False, masked=mask is not None)
