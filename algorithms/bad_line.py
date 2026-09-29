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
    def _detect_axis(profile: np.ndarray, floor: float, k: float,
                     method: str, window: int):
        """对一维 profile 做局部中值比对，返回 (偏差数组, 门限)。"""
        n = profile.size
        w = max(3, int(window) | 1)
        half = w // 2
        padded = np.pad(profile, (half, half), mode="edge")
        # 每个位置的局部参考：去掉中心点的邻域中值
        ref = np.empty(n, dtype=np.float32)
        for i in range(n):
            seg = np.concatenate([padded[i:i + half], padded[i + half + 1:i + w]])
            ref[i] = np.median(seg) if seg.size else padded[i + half]
        dev = profile - ref
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
        h, w = work.shape
        defects = []

        for view in C.plane_views(work, pattern, origin):
            plane = view["plane"]
            if plane.size == 0:
                continue
            base_x, base_y, step = view["base_x"], view["base_y"], view["step"]
            pf = plane.astype(np.float32)
            ph, pw = pf.shape

            if use_rows:
                if block <= 0:
                    prof = pf.mean(axis=1)
                    dev, thr, sigma = self._detect_axis(prof, floor, k, method, window)
                    for i in np.nonzero(np.abs(dev) > thr)[0]:
                        gy = base_y + step * int(i)
                        defects.append({
                            "type": "row", "x": 0, "y": int(gy),
                            "channel": view["name"],
                            "value": round(float(prof[i]), 1),
                            "delta": round(float(dev[i]), 2),
                            "note": "full row",
                            "coords": (0, int(gy), image_data.shape[1], int(gy)),
                        })
                else:
                    for b0 in range(0, pw, block):
                        b1 = min(pw, b0 + block)
                        if (b1 - b0) < min_seg:
                            continue
                        prof = pf[:, b0:b1].mean(axis=1)
                        dev, thr, sigma = self._detect_axis(prof, floor, k, method, window)
                        for i in np.nonzero(np.abs(dev) > thr)[0]:
                            gy = base_y + step * int(i)
                            gx0 = base_x + step * b0
                            gx1 = base_x + step * (b1 - 1)
                            defects.append({
                                "type": "row", "x": int(gx0), "y": int(gy),
                                "channel": view["name"],
                                "value": round(float(prof[i]), 1),
                                "delta": round(float(dev[i]), 2),
                                "note": f"seg x{gx0}..{gx1}",
                                "coords": (int(gx0), int(gy), int(gx1) + 1, int(gy)),
                            })

            if use_cols:
                if block <= 0:
                    prof = pf.mean(axis=0)
                    dev, thr, sigma = self._detect_axis(prof, floor, k, method, window)
                    for i in np.nonzero(np.abs(dev) > thr)[0]:
                        gx = base_x + step * int(i)
                        defects.append({
                            "type": "col", "x": int(gx), "y": 0,
                            "channel": view["name"],
                            "value": round(float(prof[i]), 1),
                            "delta": round(float(dev[i]), 2),
                            "note": "full col",
                            "coords": (int(gx), 0, int(gx), image_data.shape[0]),
                        })
                else:
                    for b0 in range(0, ph, block):
                        b1 = min(ph, b0 + block)
                        if (b1 - b0) < min_seg:
                            continue
                        prof = pf[b0:b1, :].mean(axis=0)
                        dev, thr, sigma = self._detect_axis(prof, floor, k, method, window)
                        for i in np.nonzero(np.abs(dev) > thr)[0]:
                            gx = base_x + step * int(i)
                            gy0 = base_y + step * b0
                            gy1 = base_y + step * (b1 - 1)
                            defects.append({
                                "type": "col", "x": int(gx), "y": int(gy0),
                                "channel": view["name"],
                                "value": round(float(prof[i]), 1),
                                "delta": round(float(dev[i]), 2),
                                "note": f"seg y{gy0}..{gy1}",
                                "coords": (int(gx), int(gy0), int(gx), int(gy1) + 1),
                            })

        # 合并同一行/列（多相位可能同时命中），保留最强的偏差
        merged = {}
        for d in defects:
            key = (d["type"], d["y"] if d["type"] == "row" else d["x"], d["note"])
            cur = merged.get(key)
            if cur is None:
                d["channels"] = [d["channel"]]
                merged[key] = d
                continue
            if d["channel"] not in cur["channels"]:
                cur["channels"].append(d["channel"])
            if abs(d["delta"]) > abs(cur["delta"]):     # 保留偏差最大的相位做代表
                d["channels"] = cur["channels"]
                merged[key] = d

        final = sorted(merged.values(), key=lambda d: (d["type"], d["y"], d["x"]))
        overlays = [{
            "type": "line",
            "coords": tuple(d["coords"]),
            "kind": d["type"],
            "color": d["type"],
        } for d in final]

        rows = sum(1 for d in final if d["type"] == "row")
        cols = sum(1 for d in final if d["type"] == "col")
        message = f"坏线 {len(final)} 条（行 {rows} / 列 {cols}）"
        if final:
            worst = max(final, key=lambda d: abs(d["delta"]))
            message += (f"\n最大偏差: {worst['type']} "
                        f"{'y' if worst['type'] == 'row' else 'x'}="
                        f"{worst['y'] if worst['type'] == 'row' else worst['x']} "
                        f"Δ={worst['delta']:+.1f} DN")
        return {
            "image": image_data,
            "overlays": overlays,
            "defects": final,
            "message": message,
            "report": {"rows": rows, "cols": cols},
        }
