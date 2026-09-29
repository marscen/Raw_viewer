"""兼容层：保留早期 utils.image_loader 的三个函数接口。

真正的实现已经搬到 utils/raw_io.py（读盘）与 utils/display.py（显示变换），
这里只做转发，避免两份逻辑各自演化（历史上 demosaic 的 shift bug 就是因为
同一功能有多个副本）。老脚本 `from utils.image_loader import load_raw_image`
仍然可用。
"""
from __future__ import annotations

import numpy as np

from utils.display import DisplayParams, bayer_colorize, demosaic, render_display
from utils.raw_io import RawReadError, RawLoadSpec, load_raw

__all__ = ["load_raw_image", "apply_bayer_mask", "demosaic_image"]


def _normalize8(raw: np.ndarray, bit_depth: int) -> np.ndarray:
    """按满量程线性归一化到 8bit（旧的显示约定）。"""
    max_val = (1 << int(bit_depth)) - 1
    params = DisplayParams(bit_depth=int(bit_depth), view="Mono",
                           stretch="Fixed (black/white)", black_level=0,
                           white_level=max_val)
    img, _ = render_display(raw, params)
    return img


def load_raw_image(file_path, width, height, bit_depth):
    """读取裸 RAW，返回 (8bit 显示图, 原始 DN 数组)。

    读盘失败时保持旧行为：打印错误并返回 (None, None)。
    """
    try:
        spec = RawLoadSpec(width=int(width), height=int(height), bit_depth=int(bit_depth))
        raw = load_raw(file_path, spec)
    except (RawReadError, OSError, ValueError) as exc:
        print(f"Error loading image: {exc}")
        return None, None
    return _normalize8(raw, bit_depth), raw


def apply_bayer_mask(raw_data, pattern, bit_depth):
    """Bayer 马赛克着色预览：按相位把灰度值染成 R/G/B。"""
    return bayer_colorize(_normalize8(raw_data, bit_depth), pattern)


def demosaic_image(raw_data, pattern, bit_depth):
    """双线性插值 demosaic，返回 8bit RGB。"""
    norm = _normalize8(raw_data, bit_depth).astype(np.float32)
    rgb = demosaic(norm, pattern)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
