"""可选加速后端（OpenCV）。

设计原则：**OpenCV 是可选依赖，不是必需依赖**。

  * 没有装 OpenCV 时，所有功能仍然可用（走纯 numpy 回退实现）；
  * 装了 OpenCV 时，热点路径自动切过去，并在"关于"里显示当前后端；
  * 用环境变量 `RAWV2_NO_CV2=1` 可以强制禁用，便于测试两条路径；
  * 用 **opencv-python-headless**（不要用 opencv-python）：后者自带一套 Qt，
    会和 PyQt6 抢 Qt 平台插件，典型症状是启动直接崩在 "xcb plugin"。

实测收益（Apple Silicon, 10 线程）：
    demosaic 1920x1080   56 ms -> 0.4 ms      (~150x)
    demosaic 4000x3000  327 ms -> 1.8 ms      (~180x)
    连通标记 100 万像素大块 2590 ms -> 4.5 ms  (~575x)

懒加载：OpenCV 导入约 200~300 ms，所以只在第一次真正用到时才 import，
不拖慢程序启动。
"""
from __future__ import annotations

import os

import numpy as np

__all__ = [
    "available", "backend_info", "backend_name", "cv2_version",
    "demosaic", "connected_components", "dilate_mask", "erode_mask",
    "open_mask", "clahe", "phase_correlate", "resize_area", "shift_image",
    "CV_BAYER_MAP",
]

_FORCE_OFF = os.environ.get("RAWV2_NO_CV2", "").strip().lower() not in ("", "0", "false", "no")
_cv2 = None
_tried = False

# 我们的 CFA 名称 -> cv2 的 Bayer 常量名。
# 注意 cv2 的命名与我们的相位定义**差一位**（实测反推：纯色 CFA 场里，
# 只有 BayerBG 才能让 RGGB 的 R/G/B 落在正确通道上），不要凭名字猜。
CV_BAYER_MAP = {
    "RGGB": "COLOR_BayerBG2BGR",
    "BGGR": "COLOR_BayerRG2BGR",
    "GRBG": "COLOR_BayerGB2BGR",
    "GBRG": "COLOR_BayerGR2BGR",
}


def _cv2_module():
    """懒加载 OpenCV（只尝试一次）。不可用时返回 None。"""
    global _cv2, _tried
    if not _tried:
        _tried = True
        if not _FORCE_OFF:
            try:
                import cv2 as _m            # noqa: WPS433 (延迟导入是有意的)
                _cv2 = _m
            except Exception:
                _cv2 = None
    return _cv2


def available() -> bool:
    """当前是否使用 OpenCV 加速。"""
    return _cv2_module() is not None


def cv2_version() -> str:
    m = _cv2_module()
    return getattr(m, "__version__", "") if m is not None else ""


def backend_name() -> str:
    return f"OpenCV {cv2_version()}" if available() else "numpy (未启用 OpenCV)"


def backend_info() -> str:
    """给"关于"对话框/状态栏用的一句话说明。"""
    if not available():
        extra = "（已被 RAWV2_NO_CV2 禁用）" if _FORCE_OFF else "（pip install opencv-python-headless 可加速）"
        return f"加速后端: numpy{extra}"
    try:
        threads = _cv2_module().getNumThreads()
    except Exception:
        threads = "?"
    return f"加速后端: OpenCV {cv2_version()}（{threads} 线程）"


# ----------------------------------------------------------------------
# demosaic
# ----------------------------------------------------------------------
def demosaic(raw: np.ndarray, pattern: str) -> np.ndarray:
    """CFA 原始 DN -> (H, W, 3) RGB，**保持 DN 数值空间**（不归一化）。

    返回 dtype 可能是 uint16（OpenCV 路径）或 float32（numpy 回退），
    调用方应对两者都做四舍五入后再查表。
    要求 pattern 的原点在数组 (0,0)（即传整帧，不要传奇数起点的裁剪块）。
    """
    from utils import cfa

    p = cfa.normalize_pattern(pattern)
    if raw is None:
        raise ValueError("raw 为空")
    if p not in CV_BAYER_MAP:
        raise ValueError(f"demosaic 需要 Bayer pattern，收到 {pattern!r}")

    m = _cv2_module()
    if m is not None:
        # cv2 需要连续内存、且只接受 8/16bit
        src = np.ascontiguousarray(raw)
        if src.dtype not in (np.uint8, np.uint16):
            src = np.clip(np.rint(src), 0, 65535).astype(np.uint16)
        code = getattr(m, CV_BAYER_MAP[p])
        bgr = m.cvtColor(src, code, dstCn=3)
        return np.ascontiguousarray(bgr[:, :, ::-1])       # BGR -> RGB

    from utils.display import demosaic as _np_demosaic
    return _np_demosaic(raw.astype(np.float32, copy=False), p)


# ----------------------------------------------------------------------
# 连通标记 / 形态学
# ----------------------------------------------------------------------
def connected_components(mask: np.ndarray, connectivity: int = 8):
    """二值掩罩 -> (labels, sizes)。

    labels: int32，(H, W)，0 为背景，缺陷区域从 1 开始编号；
    sizes:  int64，长度 = 区域数 + 1，sizes[label] 即该区域的像素数；
            **sizes[0] 恒为 0**（背景不计入），避免调用方把背景面积当区域面积。
    """
    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    m = _cv2_module()
    if m is not None and mask.any():
        conn = 8 if int(connectivity) >= 8 else 4
        n, labels, stats, _cent = m.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=conn, ltype=m.CV_32S)
        sizes = np.bincount(labels.ravel(), minlength=n).astype(np.int64)
        sizes[0] = 0                                  # 背景不算区域
        return labels.astype(np.int32), sizes

    from algorithms import _common as C
    labels, n = C.label_mask(mask, connectivity=connectivity)
    sizes = np.bincount(labels.ravel(), minlength=max(1, n + 1)).astype(np.int64)
    sizes[0] = 0                                      # 背景不算区域
    return labels.astype(np.int32), sizes


def _dilate_np(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3x3 膨胀（纯 numpy，位移相或）。"""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
               p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
               p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return out


def _erode_np(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3x3 腐蚀 = 对补集做膨胀再取反（省掉一份 min 滤波实现）。"""
    return ~_dilate_np(~np.asarray(mask, dtype=bool), iterations)


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = _cv2_module()
    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if m is not None and int(iterations) > 0:
        kernel = np.ones((3, 3), np.uint8)
        return m.dilate(mask.astype(np.uint8), kernel,
                        iterations=int(iterations)).astype(bool)
    return _dilate_np(mask, iterations)


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = _cv2_module()
    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if m is not None and int(iterations) > 0:
        kernel = np.ones((3, 3), np.uint8)
        return m.erode(mask.astype(np.uint8), kernel,
                       iterations=int(iterations)).astype(bool)
    return _erode_np(mask, iterations)


def open_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """开运算：先腐蚀后膨胀，用来去掉孤立毛刺。"""
    return dilate_mask(erode_mask(mask, iterations), iterations)


# ----------------------------------------------------------------------
# 显示增强
# ----------------------------------------------------------------------
def clahe(gray8: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """局部对比度增强（CLAHE）。

    暗场/低对比场景下能看到更多结构。OpenCV 缺失时退化为**全局**直方图
    均衡（不是自适应的，只保证功能不缺失）。
    """
    g = np.ascontiguousarray(gray8.astype(np.uint8, copy=False))
    m = _cv2_module()
    if m is not None:
        tiles = max(2, int(tiles))
        obj = m.createCLAHE(clipLimit=float(max(0.1, clip_limit)),
                            tileGridSize=(tiles, tiles))
        return obj.apply(g)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    if cdf[-1] <= 0:
        return g
    lut = np.clip(np.rint(cdf / cdf[-1] * 255.0), 0, 255).astype(np.uint8)
    return lut[g]


def resize_area(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """高质量缩小（区域平均）。OpenCV 缺失时返回 None，由调用方回退到 Qt。"""
    m = _cv2_module()
    if m is None or width < 1 or height < 1:
        return None
    return m.resize(np.ascontiguousarray(img), (int(width), int(height)),
                    interpolation=m.INTER_AREA)


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """把图像平移 (dx, dy) 像素（正数=向右/向下），支持亚像素。

    用于参考帧对齐：把参考帧挪到与当前帧对齐后再做差分。OpenCV 缺失时
    退化为整数像素的 np.roll（够用，只是整像素精度）。
    """
    if img is None:
        return None
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img
    m = _cv2_module()
    if m is not None:
        h, w = img.shape[:2]
        mat = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
        interp = m.INTER_LINEAR if (abs(dx - round(dx)) > 1e-3 or
                                    abs(dy - round(dy)) > 1e-3) else m.INTER_NEAREST
        border = m.BORDER_REPLICATE
        return m.warpAffine(np.ascontiguousarray(img), mat, (w, h),
                            flags=interp, borderMode=border)
    return np.roll(img, (int(round(dy)), int(round(dx))), axis=(0, 1))


# ----------------------------------------------------------------------
# 配准
# ----------------------------------------------------------------------
def phase_correlate(a: np.ndarray, b: np.ndarray):
    """相位相关求两帧之间的平移量。

    返回 {"dx", "dy", "response", "method"}：
      b(x + dx, y + dy) ≈ a(x, y)，即要把 b 对齐到 a 需要平移 (-dx, -dy)。
    （实测校验：b = np.roll(a, (dy=+3, dx=-2)) 时返回 dx=-2, dy=+3。）

    两帧尺寸必须一致；OpenCV 缺失时用 numpy FFT 回退（整数峰 + 抛物线
    亚像素细化，加汉宁窗抑制边缘效应）。
    """
    if a is None or b is None or a.shape != b.shape:
        return None
    m = _cv2_module()
    if m is not None:
        fa = np.ascontiguousarray(a.astype(np.float32, copy=False))
        fb = np.ascontiguousarray(b.astype(np.float32, copy=False))
        (dx, dy), resp = m.phaseCorrelate(fa, fb)
        return {"dx": float(dx), "dy": float(dy), "response": float(resp),
                "method": "cv2.phaseCorrelate"}

    fa = a.astype(np.float32, copy=False)
    fb = b.astype(np.float32, copy=False)
    h, w = fa.shape
    wy = np.hanning(h).astype(np.float32)[:, None]
    wx = np.hanning(w).astype(np.float32)[None, :]
    win = wy * wx
    fa = (fa - fa.mean()) * win
    fb = (fb - fb.mean()) * win
    # 注意符号约定：conj(Fa)*Fb 的峰值才与 cv2.phaseCorrelate 一致
    # （Fa*conj(Fb) 会整体反号，实测三种位移都验证过）
    spec = np.conj(np.fft.rfft2(fa)) * np.fft.rfft2(fb)
    spec /= np.maximum(np.abs(spec), 1e-9)
    corr = np.fft.irfft2(spec, s=(h, w))
    peak = int(np.argmax(corr))
    py, px = divmod(peak, w)
    # 抛物线亚像素细化
    def _sub(v0, v1, v2):
        denom = (v0 - 2 * v1 + v2)
        return 0.0 if abs(denom) < 1e-12 else float(0.5 * (v0 - v2) / denom)

    sy = _sub(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    sx = _sub(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    dx = px + sx
    dy = py + sy
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    resp = float(corr[py, px] / max(np.abs(corr).max(), 1e-9))
    return {"dx": float(dx), "dy": float(dy), "response": resp,
            "method": "numpy FFT"}
