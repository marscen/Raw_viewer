"""批量/序列分析：把一叠 RAW 的关键指标汇成一张表。

产线/实验室里最常见的动作："这批 dump 跑一遍，出个汇总表"。这里只做
**只读**分析，不修改任何文件：

    每个文件 -> 分通道 mean/std/min/max、饱和比例、0DN 比例、
                黑电平估计、可选坏点数（用当前坏点参数）

主窗口负责选文件与进度条，这里负责单帧计算与 CSV 落盘。
"""
from __future__ import annotations

import os

import numpy as np

from utils import accel, cfa, stats as S

__all__ = ["analyze_frame", "write_batch_csv", "BATCH_HEADER"]

BATCH_HEADER = [
    "file", "size_bytes", "width", "height", "bit_depth", "pattern",
    "roi", "mean", "std", "min", "max", "median", "sat_pct", "zero_pct",
    "black_est", "defects", "defect_types", "backend", "note",
]


def estimate_black_level(raw: np.ndarray, bit_depth: int, roi=None) -> float:
    """估计黑电平。

    有 ROI 时直接用 ROI 的中位数（正确做法：先框光学黑区再估）；
    没有 ROI 时用全图直方图的**峰值**位置（暗场的峰值就是黑电平），
    并在峰值落在量程上半区时认为这不是暗场，返回 NaN 提示不可信。
    """
    from utils.stats import roi_region
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1]
    if sub.size == 0:
        return float("nan")
    if roi is not None:
        return float(np.median(sub))
    mc = (1 << int(bit_depth)) - 1
    counts = np.bincount(np.ravel(sub).astype(np.int64), minlength=mc + 1)[:mc + 1]
    if counts.sum() == 0:
        return float("nan")
    if counts.size >= 5:                      # 轻度平滑，避免单点噪声成为峰值
        k = np.ones(5) / 5.0
        counts = np.convolve(counts, k, mode="same")
    peak = int(np.argmax(counts))
    # 只有"暗场"（峰值落在量程很低的位置）时峰值才等于黑电平；
    # 亮场的峰值是信号电平，这里明确返回 NaN 让调用方提示不可信。
    if peak > 0.25 * mc:
        return float("nan")
    return float(peak)


def _roi_text(roi) -> str:
    """(x0, y0, x1, y1) -> "x100..500 y100..480"。

    注意别写成 "% tuple(roi)"：那样会把 y0 当成 x1 打印成 "x100..100"。
    """
    x0, y0, x1, y1 = (int(v) for v in roi)
    return f"x{x0}..{x1} y{y0}..{y1}"


def analyze_frame(raw: np.ndarray, pattern: str, bit_depth: int, roi=None,
                  max_code: int = 0, defect_params: dict | None = None) -> dict:
    """单帧关键指标。defect_params 给定时额外统计坏点数。"""
    mc = int(max_code) if max_code else (1 << int(bit_depth)) - 1
    st = S.roi_stats(raw, pattern, roi, bit_depth, max_code=mc)
    overall = st["overall"]
    sat_total = sum(int(c.get("sat") or 0) for c in st["channels"])
    zero_total = sum(int(c.get("zero") or 0) for c in st["channels"])
    total = max(1, sum(int(c.get("count") or 0) for c in st["channels"]))
    out = {
        "roi": _roi_text(st["roi"]) if roi is not None else "full",
        "mean": round(float(overall.get("mean") or 0.0), 3),
        "std": round(float(overall.get("std") or 0.0), 3),
        "min": float(overall.get("min") or 0.0),
        "max": float(overall.get("max") or 0.0),
        "median": float(overall.get("median") or 0.0),
        "sat_pct": round(100.0 * sat_total / total, 4),
        "zero_pct": round(100.0 * zero_total / total, 4),
        "black_est": None,
        "defects": "",
        "defect_types": "",
        "note": "",
    }
    black = estimate_black_level(raw, bit_depth, roi)
    out["black_est"] = None if black != black else round(float(black), 2)
    if black != black:
        out["note"] = "非暗场(峰值在半量程以上)，黑电平估计不可信"

    if defect_params:
        from algorithms.bad_pixel import BadPixelDetectionAlgorithm
        p = dict(defect_params)
        p.update({"pattern": pattern, "_bit_depth": int(bit_depth), "_roi": roi})
        res = BadPixelDetectionAlgorithm().run(raw, p)
        counts = res.get("report", {}).get("counts", {})
        out["defects"] = len(res.get("defects", []))
        out["defect_types"] = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    out["backend"] = accel.backend_name()
    return out


def write_batch_csv(path: str, rows: list) -> str:
    from utils import export as X
    body = [[r.get(h, "") for h in BATCH_HEADER] for r in rows]
    return X.write_csv(path, BATCH_HEADER, body)
