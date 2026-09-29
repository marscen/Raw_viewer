"""测量层：分通道统计、直方图、ROI 统计、行/列 profile、两帧差分指标。

sensor 测试里天天要回答的问题：
  * 暗场均值/标准差是多少（read noise / 黑电平漂移）？
  * 哪个通道（R/Gr/Gb/B）有问题？Gr 和 Gb 是否失配？
  * 有多少像素饱和/为 0（曝光与死点）？
  * 行方向/列方向的 FPN 有多大？哪一行跳变？
  * 校正前后差了多少（PSNR / 最大差 / 差异像素比例）？

这里的函数全部是纯 numpy，输入都是一次 raw_io.load_raw 得到的 DN 数组。
ROI 用全局坐标 (x0, y0, x1, y1)（左闭右开），允许奇数起点：
相位判定始终按**全局坐标奇偶性**来算，所以 ROI 不会串通道。
"""
from __future__ import annotations

import math

import numpy as np

from utils import cfa

__all__ = [
    "ChannelStat", "basic_stats", "roi_region", "global_phase_map",
    "plane_values", "roi_stats", "histogram", "profile",
    "diff_stats", "neighborhood", "saturation_stats",
]


class ChannelStat(dict):
    """一组像素的统计量（dict 子类，便于直接喂给表格/CSV）。"""

    FIELDS = ("name", "count", "mean", "std", "min", "max", "median",
              "p01", "p99", "sat", "zero", "snr_db")

    def __init__(self, name: str, values: np.ndarray, max_code: int):
        values = np.asarray(values)
        flat = np.ravel(values)
        n = int(flat.size)
        d = {k: "" for k in self.FIELDS}
        d["name"] = name
        if n:
            f = flat.astype(np.float64, copy=False)
            mean = float(np.mean(f))
            std = float(np.std(f))
            d.update(
                count=n,
                mean=mean,
                std=std,
                min=float(np.min(f)),
                max=float(np.max(f)),
                median=float(np.median(f)),
                p01=float(np.percentile(f, 1.0)),
                p99=float(np.percentile(f, 99.0)),
                sat=int(np.count_nonzero(f >= max_code)),
                zero=int(np.count_nonzero(f <= 0)),
                snr_db=(20.0 * math.log10(mean / std) if std > 0 and mean > 0 else float("inf")),
            )
            d["snr_db"] = round(d["snr_db"], 2) if np.isfinite(d["snr_db"]) else "inf"
        else:
            d.update(count=0)
        super().__init__(d)


def basic_stats(values: np.ndarray, max_code: int, name: str = "all") -> ChannelStat:
    return ChannelStat(name, values, max_code)


def roi_region(shape, roi):
    """把 ROI 规整成合法 (x0, y0, x1, y1)（左闭右开，至少 1x1）。"""
    h, w = shape[0], shape[1]
    if roi is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = (int(v) for v in roi)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def global_phase_map(shape, roi, pattern) -> np.ndarray:
    """ROI 内、按**全局坐标**奇偶性算出的相位图（0..3）。

    直接用 raw[y0:y1, x0:x1][dy::2, dx::2] 在做奇数起点 ROI 时会串通道，
    所以这里显式按全局坐标计算。
    """
    x0, y0, x1, y1 = roi_region(shape, roi)
    ys = (np.arange(y0, y1) % 2) * 2
    xs = np.arange(x0, x1) % 2
    return (ys[:, None] + xs[None, :]).astype(np.uint8)


def plane_values(raw: np.ndarray, pattern, roi=None) -> dict:
    """按相位取 ROI 内的像素值，返回 {通道名: 一维数组}。"""
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1]
    p = cfa.normalize_pattern(pattern)
    if p == "Mono/None":
        return {"Mono": np.ravel(sub)}
    phases = global_phase_map(raw.shape, (x0, y0, x1, y1), p)
    out = {}
    for phase, name in enumerate(cfa.PHASE_LABELS[p]):
        out[name] = sub[phases == phase]
    return out


def roi_stats(raw: np.ndarray, pattern, roi=None, bit_depth: int = 10,
              max_code: int = 0) -> dict:
    """ROI（或全图）统计：总体 + 分通道。

    返回 {"roi": (x0,y0,x1,y1), "overall": ChannelStat, "channels": [ChannelStat]}。
    """
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1]
    mc = int(max_code) if max_code else (1 << int(bit_depth)) - 1
    planes = plane_values(raw, pattern, (x0, y0, x1, y1))
    channels = [ChannelStat(name, vals, mc) for name, vals in planes.items()]
    return {
        "roi": (x0, y0, x1, y1),
        "size": (x1 - x0, y1 - y0),
        "overall": ChannelStat("ALL", sub, mc),
        "channels": channels,
    }


def histogram(raw: np.ndarray, pattern, roi=None, bit_depth: int = 10,
              bins: int = 256, per_channel: bool = False):
    """直方图。返回 (counts, edges)；per_channel 时 counts 为 {通道名: counts}。"""
    mc = (1 << int(bit_depth)) - 1
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1]

    def _hist(vals):
        counts = np.bincount(np.ravel(vals).astype(np.int64), minlength=mc + 1)
        if bins and bins < counts.size:                # 归并到更少的 bin
            edges = np.linspace(0, mc + 1, bins + 1).astype(np.int64)
            merged = np.zeros(bins, dtype=np.int64)
            for i in range(bins):
                lo, hi = edges[i], edges[i + 1]
                merged[i] = counts[lo:hi].sum()
            return merged
        return counts

    if not per_channel:
        counts = _hist(sub)
        edges = np.arange(counts.size + 1) if counts.size == mc + 1 else \
            np.linspace(0, mc + 1, counts.size + 1)
        return counts, edges

    out, edges = {}, None
    for name, vals in plane_values(raw, pattern, (x0, y0, x1, y1)).items():
        c = _hist(vals)
        if edges is None:
            edges = np.arange(c.size + 1) if c.size == mc + 1 else \
                np.linspace(0, mc + 1, c.size + 1)
        out[name] = c
    return out, edges


def profile(raw: np.ndarray, pattern, roi=None, axis: str = "rows"):
    """行/列 profile：沿另一轴做均值与标准差，并给出分通道曲线。

    axis="rows"  -> 每行的统计（横轴 = 行号 y），用来找行 FPN / 坏行；
    axis="cols"  -> 每列的统计（横轴 = 列号 x），用来找列 FPN / 坏列。
    返回 {"index", "mean", "std", "channels": {name: (mean, std)}, "axis"}
    """
    p = cfa.normalize_pattern(pattern)
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1].astype(np.float32)
    phases = global_phase_map(raw.shape, (x0, y0, x1, y1), p)

    if axis == "cols":
        index = np.arange(x0, x1)
        mat, ph = sub.T, phases.T            # 统一成 (沿轴长度, 横向长度)
    else:
        index = np.arange(y0, y1)
        mat, ph = sub, phases

    mean = mat.mean(axis=1)
    std = mat.std(axis=1)

    channels = {}
    if p != "Mono/None":
        for phase, name in enumerate(cfa.PHASE_LABELS[p]):
            sel = ph == phase
            cnt = sel.sum(axis=1).astype(np.float32)
            safe = np.maximum(cnt, 1.0)
            cm = np.where(sel, mat, 0.0).sum(axis=1) / safe
            cvar = (np.where(sel, mat - cm[:, None], 0.0) ** 2).sum(axis=1) / safe
            ok = cnt > 0
            channels[name] = (np.where(ok, cm, np.nan),
                              np.where(ok, np.sqrt(cvar), np.nan))

    return {"index": index, "mean": mean, "std": std, "channels": channels,
            "axis": axis}


def diff_stats(a: np.ndarray, b: np.ndarray, bit_depth: int = 10,
               max_code: int = 0, tol: int = 1) -> dict:
    """两帧差分指标（校正前后 / 与参考帧比较）。

    返回 psnr / rmse / mean_abs / max_abs / pct_diff / 差异图。
    """
    if a is None or b is None:
        return {}
    if a.shape != b.shape:
        return {"error": f"尺寸不一致：{a.shape} vs {b.shape}"}
    mc = int(max_code) if max_code else (1 << int(bit_depth)) - 1
    fa = a.astype(np.float64)
    fb = b.astype(np.float64)
    d = fa - fb
    ad = np.abs(d)
    mse = float(np.mean(d * d))
    psnr = float("inf") if mse <= 0 else 10.0 * math.log10((float(mc) ** 2) / mse)
    return {
        "psnr": psnr,
        "rmse": math.sqrt(mse),
        "mean_abs": float(np.mean(ad)),
        "max_abs": float(np.max(ad)),
        "pct_diff": float(100.0 * np.count_nonzero(ad > tol) / ad.size),
        "count_diff": int(np.count_nonzero(ad > tol)),
        "diff_map": ad,
        "signed_max": float(np.max(d)),
        "signed_min": float(np.min(d)),
    }


def saturation_stats(raw: np.ndarray, pattern, bit_depth: int = 10, roi=None) -> dict:
    """饱和/黑点统计：每个通道有多少像素顶到满量程或为 0。"""
    mc = (1 << int(bit_depth)) - 1
    planes = plane_values(raw, pattern, roi)
    out = {}
    for name, vals in planes.items():
        n = max(1, vals.size)
        out[name] = {
            "count": int(vals.size),
            "sat": int(np.count_nonzero(vals >= mc)),
            "sat_pct": 100.0 * np.count_nonzero(vals >= mc) / n,
            "zero": int(np.count_nonzero(vals <= 0)),
            "zero_pct": 100.0 * np.count_nonzero(vals <= 0) / n,
        }
    total = max(1, sum(v["count"] for v in out.values()))
    return {
        "max_code": mc,
        "channels": out,
        "sat_total": sum(v["sat"] for v in out.values()),
        "sat_pct": 100.0 * sum(v["sat"] for v in out.values()) / total,
        "zero_total": sum(v["zero"] for v in out.values()),
    }


def neighborhood(raw: np.ndarray, x: int, y: int, radius: int = 2, pattern=None):
    """像素检查器：返回中心点周围 (2r+1)^2 的窗口值与每个点的通道名。

    返回 {"values", "channels", "x0", "y0", "center", "center_value",
          "center_channel", "channel_values"}
    """
    h, w = raw.shape
    r = max(0, int(radius))
    x0 = max(0, x - r)
    y0 = max(0, y - r)
    x1 = min(w, x + r + 1)
    y1 = min(h, y + r + 1)
    vals = raw[y0:y1, x0:x1]
    p = cfa.normalize_pattern(pattern)

    if p == "Mono/None":
        chans = np.full(vals.shape, "Mono", dtype=object)
        center_ch = "Mono"
    else:
        phases = global_phase_map(raw.shape, (x0, y0, x1, y1), p)
        names = cfa.PHASE_LABELS[p]
        chans = np.empty(vals.shape, dtype=object)
        for phase, nm in enumerate(names):
            chans[phases == phase] = nm
        center_ch = cfa.plane_name_at(x, y, p)

    channel_values = {}
    if p != "Mono/None":
        for nm in cfa.PHASE_LABELS[p]:
            channel_values[nm] = int(vals[chans == nm].sum()) if np.any(chans == nm) else 0

    return {
        "values": vals,
        "channels": chans,
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "center": (int(x), int(y)),
        "center_value": int(raw[y, x]) if (0 <= y < h and 0 <= x < w) else None,
        "center_channel": center_ch,
        "channel_values": channel_values,
    }
