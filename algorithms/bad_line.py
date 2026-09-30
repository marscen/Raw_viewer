"""坏行/坏列检测（行 FPN、列 FPN、部分坏线）。

思路与坏点一致：先按相位拆平面（避免 R/B 行交替造成的假坏行），再对行/列
统计量做**局部参考**比对，而不是和全图 median 比——否则画面本身有亮度梯度
（阴影/暗角）时，正常行也会被判成坏行。局部参考 = 相邻 ±window/2 行均值的
中值，这样只对"突兀的跳变"敏感。

支持 segment 模式：整行不一定全坏，把行按 block_size 分块，块内均值偏离参考
超过门限即报出该段（例如只有左半行有列驱动问题）。ROI 内的检测同样按全局
相位映射坐标。
"""
from __future__ import annotations

import numpy as np

from algorithms import _common as C
from utils import accel
from .base import Algorithm


class BadLineDetectionAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "Bad Line Detection"

    @property
    def description(self) -> str:
        return ("坏行/坏列检测：按相位分平面后，用相邻行的局部中值作参考，"
                "支持整行或分段(line segment)检测，避免阴影造成的误判。")

    def get_parameters(self) -> dict:
        return {
            "threshold": {
                "type": "float", "default": 100.0, "min": 0.0, "max": 65535.0,
                "label": "最小偏差 |Δ| (DN)",
            },
            "sigma_k": {
                "type": "float", "default": 6.0, "min": 0.5, "max": 50.0,
                "label": "自适应系数 k (×σ)",
            },
            "method": {
                "type": "list", "options": ["Adaptive (MAD)", "Absolute threshold"],
                "default": "Adaptive (MAD)", "label": "阈值方式",
            },
            "axis": {
                "type": "list", "options": ["Rows", "Cols", "Both"],
                "default": "Both", "label": "检测方向",
            },
            "window": {
                "type": "int", "default": 9, "min": 3, "max": 101,
                "label": "参考窗口 (行/列数)",
            },
            "block_size": {
                "type": "int", "default": 0, "min": 0, "max": 4096,
                "label": "分段长度 (0=整行)",
                "tooltip": ">0 时按该长度分段检测，可发现「只有局部坏」的线",
            },
            "min_segment": {
                "type": "int", "default": 32, "min": 1, "max": 4096,
                "label": "段最小像素数",
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _moving_reference(profile: np.ndarray, window: int) -> np.ndarray:
        """去掉中心点的滑动邻域中值（全向量化）。

        旧实现按位置写 Python 循环 + 每次 np.median，分段检测时是 2500 ms 级
        的开销；改成把 (w-1) 个移位切片一次性堆起来取中值，同样的活只要几十毫秒。
        """
        n = int(profile.size)
        w = max(3, int(window) | 1)          # 强制奇数
        half = w // 2
        padded = np.pad(profile.astype(np.float32), (half, half), mode="edge")
        offsets = [d for d in range(-half, half + 1) if d != 0]
        stack = np.stack([padded[half + d: half + d + n] for d in offsets])
        return np.median(stack, axis=0)

    @classmethod
    def _detect_axis(cls, profile: np.ndarray, floor: float, k: float,
                     method: str, window: int):
        """对一维 profile 做局部中值比对，返回 (偏差数组, 门限, sigma)。"""
        ref = cls._moving_reference(profile, window)
        dev = profile.astype(np.float32) - ref
        if method.startswith("Adaptive"):
            sigma = C.robust_sigma(dev)
            thr = max(k * sigma, floor)
        else:
            sigma = 0.0
            thr = floor
        return dev, (thr if thr > 0 else 1.0), sigma

    def run(self, image_data: np.ndarray, params: dict):
        pattern = params.get("pattern", "Mono/None")
        floor = float(params.get("threshold", 100.0) or 0.0)
        k = float(params.get("sigma_k", 6.0) or 0.0)
        method = params.get("method", "Adaptive (MAD)")
        axis = params.get("axis", "Both")
        window = int(params.get("window", 9) or 9)
        block = int(params.get("block_size", 0) or 0)
        min_seg = max(1, int(params.get("min_segment", 32) or 32))
        use_rows = axis in ("Rows", "Both")
        use_cols = axis in ("Cols", "Both")

        work, origin = C.crop_with_origin(image_data, params.get("_roi"))
        H, W = work.shape
        hits = []          # 原始命中：{type, index, a0, a1, delta, value, channel}

        def scan(profile, kind, along0, along1, line_index, channel):
            """在一维 profile 上找坏线/坏段，along0/along1 是该段的区间（沿剖面方向）。"""
            dev, thr, _sigma = self._detect_axis(profile, floor, k, method, window)
            for i in np.nonzero(np.abs(dev) > thr)[0]:
                i = int(i)
                hits.append({
                    "type": kind,
                    "index": int(line_index(i)),
                    "a0": int(along0), "a1": int(along1),
                    "delta": float(dev[i]),
                    "value": float(profile[i]),
                    "channel": channel,
                })

        for view in C.plane_views(work, pattern, origin):
            plane = view["plane"]
            if plane.size == 0:
                continue
            base_x, base_y, step = view["base_x"], view["base_y"], view["step"]
            pf = plane.astype(np.float32)
            ph, pw = pf.shape

            if use_rows:
                if block <= 0:
                    scan(pf.mean(axis=1), "row", 0, W - 1,
                         lambda i, b=base_y, st=step: b + st * i, view["name"])
                else:
                    for b0 in range(0, pw, block):
                        b1 = min(pw, b0 + block)
                        if (b1 - b0) < min_seg:
                            continue
                        scan(pf[:, b0:b1].mean(axis=1), "row",
                             base_x + step * b0, base_x + step * (b1 - 1),
                             lambda i, b=base_y, st=step: b + st * i, view["name"])
            if use_cols:
                if block <= 0:
                    scan(pf.mean(axis=0), "col", 0, H - 1,
                         lambda i, b=base_x, st=step: b + st * i, view["name"])
                else:
                    for b0 in range(0, ph, block):
                        b1 = min(ph, b0 + block)
                        if (b1 - b0) < min_seg:
                            continue
                        scan(pf[b0:b1, :].mean(axis=0), "col",
                             base_y + step * b0, base_y + step * (b1 - 1),
                             lambda i, b=base_x, st=step: b + st * i, view["name"])

        # ---- 合并同一行/列的搭接分段 ----
        # 分段模式下，不同相位、相邻分块会给出边界错开的重叠段（例如
        # "seg y0..126" 和 "seg y1..127"），不合并的话一条坏列能报出上百条。
        grouped = {}
        for h_ in hits:
            grouped.setdefault((h_["type"], h_["index"]), []).append(h_)

        merged = []
        for (dtype, index), items in sorted(grouped.items()):
            items.sort(key=lambda x: (x["a0"], x["a1"]))
            cur = None
            for it in items:
                if cur is None:
                    cur = {"a0": it["a0"], "a1": it["a1"], "wsum": it["delta"] * abs(it["delta"]),
                           "vsum": it["value"] * abs(it["delta"]), "w": abs(it["delta"]),
                           "best": it, "channels": {it["channel"]}}
                    continue
                if it["a0"] <= cur["a1"] + 1:                 # 相邻/重叠 -> 同一段
                    cur["a1"] = max(cur["a1"], it["a1"])
                    wgt = abs(it["delta"])
                    cur["wsum"] += it["delta"] * wgt
                    cur["vsum"] += it["value"] * wgt
                    cur["w"] += wgt
                    cur["channels"].add(it["channel"])
                    if abs(it["delta"]) > abs(cur["best"]["delta"]):
                        cur["best"] = it
                else:
                    merged.append(cur)
                    cur = {"a0": it["a0"], "a1": it["a1"], "wsum": it["delta"] * abs(it["delta"]),
                           "vsum": it["value"] * abs(it["delta"]), "w": abs(it["delta"]),
                           "best": it, "channels": {it["channel"]}}
            if cur is not None:
                merged.append(cur)

        full_extent = W - 1 if use_cols else H - 1
        defects, overlays = [], []
        for m in merged:
            dtype = m["best"]["type"]
            index = m["best"]["index"]
            weight = max(m["w"], 1e-9)
            delta = m["wsum"] / weight
            value = m["vsum"] / weight
            total_extent = (W - 1) if dtype == "row" else (H - 1)
            if m["a0"] <= 0.02 * total_extent and m["a1"] >= 0.98 * total_extent:
                note = f"full {dtype}"
            elif dtype == "row":
                note = f"seg x{m['a0']}..{m['a1']}"
            else:
                note = f"seg y{m['a0']}..{m['a1']}"
            channels = "/".join(sorted(m["channels"]))
            if len(m["channels"]) > 1:
                note += f" ({channels})"
            if dtype == "row":
                defects.append({"type": "row", "x": 0, "y": int(index),
                                "channel": m["best"]["channel"],
                                "value": round(value, 1), "delta": round(delta, 2),
                                "note": note})
                overlays.append({"type": "line", "kind": "row", "color": "row",
                                 "coords": (int(m["a0"]), int(index),
                                            int(m["a1"]) + 1, int(index))})
            else:
                defects.append({"type": "col", "x": int(index), "y": 0,
                                "channel": m["best"]["channel"],
                                "value": round(value, 1), "delta": round(delta, 2),
                                "note": note})
                overlays.append({"type": "line", "kind": "col", "color": "col",
                                 "coords": (int(index), int(m["a0"]),
                                            int(index), int(m["a1"]) + 1)})

        rows = sum(1 for d in defects if d["type"] == "row")
        cols = sum(1 for d in defects if d["type"] == "col")
        message = f"坏线 {len(defects)} 条（行 {rows} / 列 {cols}）"
        if defects:
            worst = max(defects, key=lambda d: abs(d["delta"]))
            message += (f"\n最大偏差: {worst['type']} "
                        f"{'y' if worst['type'] == 'row' else 'x'}="
                        f"{worst['y'] if worst['type'] == 'row' else worst['x']} "
                        f"Δ={worst['delta']:+.1f} DN")
        return {
            "image": image_data,
            "overlays": overlays,
            "defects": defects,
            "message": message,
            "report": {"rows": rows, "cols": cols, "raw_hits": len(hits),
                       "backend": accel.backend_name()},
        }
