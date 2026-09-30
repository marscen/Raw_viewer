"""测量层：分通道统计、直方图、ROI 统计、行/列 profile、两帧差分指标。

sensor 测试里天天要回答的问题：
  * 暗场均值/标准差是多少（read noise / 黑电平漂移）？
  * 哪个通道（R/Gr/Gb/B）有问题？Gr 和 Gb 是否失配？
  * 有多少像素饱和/为 0（曝光与死点）？
  * 行方向/列方向的 FPN 有多大？哪一行跳变？
  * 校正前后差了多少（PSNR / 最大差 / 差异像素比例）？

这里的函数全部是纯 numpy，输入都是一次 raw_io.load_raw 得到的 DN 数组。
ROI 用全局坐标 (x0, y0, x1, y1)（左闭右开），允许奇数起点：相位判定始终按
**全局坐标奇偶性**来算，所以 ROI 不会串通道。

内存策略（8K 级 dump 很常见）：不做整块 float64/float32 转换，均值/标准差用
`dtype=np.float64` 累加，逐通道曲线在**相位子平面**上算（体积只有 1/4），
超大数组的百分位用奇步长抽样（`cfa.stratified_sample`）。
"""
from __future__ import annotations

import math

import numpy as np

from utils import cfa

__all__ = [
    "ChannelStat", "basic_stats", "roi_region", "global_phase_map",
    "plane_values", "roi_stats", "histogram", "profile",
    "diff_stats", "neighborhood", "saturation_stats", "SAMPLE_LIMIT",
]

# 超过这个像素数就抽样估百分位（中值/P1/P99），mean/std/min/max/饱和计数仍然精确
SAMPLE_LIMIT = 2_000_000


class ChannelStat(dict):
    """一组像素的统计量（dict 子类，便于直接喂给表格/CSV）。"""

    FIELDS = ("name", "count", "mean", "std", "min", "max", "median",
              "p01", "p99", "sat", "zero", "snr_db")

    def __init__(self, name: str, values: np.ndarray, max_code: int):
        flat = np.ravel(np.asarray(values))
        n = int(flat.size)
        d = {k: "" for k in self.FIELDS}
        d["name"] = name
        if n:
            # 关键：不要整块 astype(np.float64)，也不要用 np.std ——
            # np.std 内部会先算 mean 再整体相减，48MP 图会多出 384MB 临时量。
            # 这里用"和 / 平方和"，einsum 在累加器里用 float64，不产生副本。
            total = float(np.einsum("i->", flat, dtype=np.float64))
            mean = total / n
            sumsq = float(np.einsum("i,i->", flat, flat, dtype=np.float64))
            std = math.sqrt(max(sumsq / n - mean * mean, 0.0))
            mn = float(flat.min())
            mx = float(flat.max())
            if n > SAMPLE_LIMIT:
                samp = cfa.stratified_sample(flat, limit=SAMPLE_LIMIT, already_flat=True)
                median = float(np.median(samp))
                p01 = float(np.percentile(samp, 1.0))
                p99 = float(np.percentile(samp, 99.0))
            else:
                median = float(np.median(flat))
                p01 = float(np.percentile(flat, 1.0))
                p99 = float(np.percentile(flat, 99.0))
            # 退化情形（全黑/全常数）不该显示成 "inf"（看着像完美信号）
            if std > 0 and mean > 0:
                snr = round(20.0 * math.log10(mean / std), 2)
            else:
                snr = "n/a"
            d.update(count=n, mean=mean, std=std, min=mn, max=mx, median=median,
                     p01=p01, p99=p99,
                     sat=int(np.count_nonzero(flat >= max_code)),
                     zero=int(np.count_nonzero(flat <= 0)), snr_db=snr)
        else:
            d.update(count=0)
        super().__init__(d)


def basic_stats(values: np.ndarray, max_code: int, name: str = "all") -> ChannelStat:
    return ChannelStat(name, values, max_code)


def roi_region(shape, roi):
    """把 ROI 规整成合法 (x0, y0, x1, y1)（左闭右开，至少 1x1）。

    空数组（0 尺寸）也安全返回全 0，不抛异常。
    """
    h = int(shape[0]) if len(shape) >= 2 else 0
    w = int(shape[1]) if len(shape) >= 2 else 0
    if h <= 0 or w <= 0:
        return 0, 0, 0, 0
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
    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 0), dtype=np.uint8)
    # 用 uint8 相加：int64 的 (H,1)+(1,W) 广播会先产生 8 字节/像素的中间量
    # （48MP 就是 384MB），uint8 相加结果还是 uint8
    ys = ((np.arange(y0, y1) % 2) * 2).astype(np.uint8)
    xs = (np.arange(x0, x1) % 2).astype(np.uint8)
    return ys[:, None] + xs[None, :]


def plane_values(raw: np.ndarray, pattern, roi=None) -> dict:
    """按相位取 ROI 内的像素值，返回 {通道名: 一维数组}。"""
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1]
    p = cfa.normalize_pattern(pattern)
    if p == "Mono/None":
        return {"Mono": np.ravel(sub)}
    if sub.size == 0:
        return {name: np.empty(0, dtype=raw.dtype) for name in cfa.PHASE_LABELS[p]}
    # 按"全局奇偶性"取相位切片：这是**视图**，不像布尔掩罩那样要为每个相位
    # 分配整幅 bool 数组 + 抽取副本（48MP 图实测能省 500MB 峰值内存）
    out = {}
    for phase, name in enumerate(cfa.PHASE_LABELS[p]):
        gy, gx = divmod(phase, 2)
        dy = (gy - y0) % 2
        dx = (gx - x0) % 2
        out[name] = sub[dy::2, dx::2]
    return out


def roi_stats(raw: np.ndarray, pattern, roi=None, bit_depth: int = 10,
              max_code: int = 0) -> dict:
    """ROI（或全图）统计：总体 + 分通道。

    返回 {"roi": (x0,y0,x1,y1), "size", "overall", "channels": [ChannelStat]}。
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
        "max_code": mc,
    }


def histogram(raw: np.ndarray, pattern, roi=None, bit_depth: int = 10,
              bins: int = 256, per_channel: bool = False):
    """直方图。返回 (counts, edges)；per_channel 时 counts 为 {通道名: counts}。

    保证 `counts.sum() == 像素数`、`edges[-1] - 1 == 数据实际最大值`：
    数据超过 2^bit_depth-1 时（位深设小了 / 16bit 容器左对齐忘了 data_shift）
    旧实现会静默丢掉溢出部分、或把横轴画到 mc 而 counts 更长，导致"整幅发白
    但直方图近乎空"且看不出原因。现在溢出照实画出来，由界面提示去查位深设置。
    """
    mc = (1 << int(bit_depth)) - 1
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1]

    def _hist(vals):
        counts = np.bincount(np.ravel(vals), minlength=mc + 1)   # uint8/16 直接可用
        hi = int(counts.size)                    # 真实上界（可能 > mc+1）
        if bins and int(bins) < hi:
            edges = np.linspace(0, hi, int(bins) + 1)
            merged = np.add.reduceat(counts, edges[:-1].astype(np.int64))
            return merged, edges
        return counts, np.arange(hi + 1)

    if not per_channel:
        return _hist(sub)
    out = {}
    edges = None
    for name, vals in plane_values(raw, pattern, (x0, y0, x1, y1)).items():
        c, e = _hist(vals)
        edges = e if edges is None else edges
        out[name] = c
    if edges is None:
        edges = np.arange(mc + 2)
    return out, edges


def profile(raw: np.ndarray, pattern, roi=None, axis: str = "rows"):
    """行/列 profile：沿另一轴做均值与标准差，并给出分通道曲线。

    axis="rows"  -> 每行的统计（横轴 = 行号 y），用来找行 FPN / 坏行；
    axis="cols"  -> 每列的统计（横轴 = 列号 x），用来找列 FPN / 坏列。

    分通道曲线在**相位子平面**上计算（体积 1/4），不会为每个相位再复制整块
    ROI；缺失相位的位置留 NaN（明确表示"这一行/列没有该通道"）。
    返回 {"index", "mean", "std", "channels": {name: (mean, std)}, "axis"}
    """
    p = cfa.normalize_pattern(pattern)
    x0, y0, x1, y1 = roi_region(raw.shape, roi)
    sub = raw[y0:y1, x0:x1]
    if sub.size == 0:
        return {"index": np.empty(0), "mean": np.empty(0), "std": np.empty(0),
                "channels": {}, "axis": axis}

    mat = sub.T if axis == "cols" else sub          # (n_along, n_across) 视图
    n = mat.shape[0]
    sums = mat.sum(axis=1, dtype=np.float64)
    sumsq = np.einsum("ij,ij->i", mat, mat, dtype=np.float64)   # 不产生 float64 副本
    mean = sums / max(1, mat.shape[1])
    var = np.maximum(sumsq / max(1, mat.shape[1]) - mean ** 2, 0.0)
    std = np.sqrt(var)
    index = np.arange(x0, x1) if axis == "cols" else np.arange(y0, y1)

    channels = {}
    if p != "Mono/None":
        for view in cfa.phase_planes(sub, p, (x0, y0)):
            plane = view["plane"].astype(np.float32)
            if plane.size == 0:
                continue
            step, off = view["step"], (view["base_y"] - y0 if axis == "rows"
                                       else view["base_x"] - x0)
            along = plane.shape[0] if axis == "rows" else plane.shape[1]
            if along == 0:
                continue
            pm = plane.mean(axis=1) if axis == "rows" else plane.mean(axis=0)
            ps = plane.std(axis=1) if axis == "rows" else plane.std(axis=0)
            pos = off + step * np.arange(along)
            pos = pos[(pos >= 0) & (pos < n)]
            if pos.size == 0:
                continue
            cmean = np.full(n, np.nan)
            cstd = np.full(n, np.nan)
            cmean[pos] = pm[:pos.size]
            cstd[pos] = ps[:pos.size]
            channels[view["name"]] = (cmean, cstd)

    return {"index": index, "mean": mean, "std": std, "channels": channels,
            "axis": axis}


def diff_stats(a: np.ndarray, b: np.ndarray, bit_depth: int = 10,
               max_code: int = 0, tol: int = 1) -> dict:
    """两帧差分指标（校正前后 / 与参考帧比较）。

    返回 psnr / rmse / mean_abs / max_abs / pct_diff / 差异图。
    """
    if a is None or b is None:
        return {"error": "缺少一帧数据"}
    if a.shape != b.shape:
        return {"error": f"尺寸不一致：{a.shape} vs {b.shape}"}
    if a.size == 0:
        return {"error": "数据为空"}
    mc = int(max_code) if max_code else (1 << int(bit_depth)) - 1
    d = a.astype(np.float64) - b.astype(np.float64)
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

    坐标会被夹进画内（越界返回 None 而不是抛 IndexError —— 槽函数里抛异常在
    PyQt6 下会直接 abort 进程）。
    """
    h, w = int(raw.shape[0]), int(raw.shape[1])
    if h <= 0 or w <= 0:
        return None
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    r = max(0, int(radius))
    x0 = max(0, x - r)
    y0 = max(0, y - r)
    x1 = min(w, x + r + 1)
    y1 = min(h, y + r + 1)
    vals = raw[y0:y1, x0:x1]
    if vals.size == 0:
        return None
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
            sel = chans == nm
            channel_values[nm] = int(vals[sel].sum()) if np.any(sel) else 0

    return {
        "values": vals,
        "channels": chans,
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "center": (int(x), int(y)),
        "center_value": int(raw[y, x]),
        "center_channel": center_ch,
        "channel_values": channel_values,
    }
