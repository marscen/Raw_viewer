"""阴影 / 暗角（Shading / Vignetting）分析。

把画面切成 blocks×blocks 的方块，比较各块均值：
  * 报告全画幅 shading 幅度（(max-min)/mean）、块极值、四角 vs 中心；
  * 超出阈值的块在图上用方框标出（同一块只画一次，列出涉及的通道），并进入缺陷清单；
  * 可选生成平坦场校正图（每相位各自的增益图 = 该相位全局均值/块均值，块最近邻上采样）。

为什么按相位分开算：R/Gr/Gb/B 的响应本来就不一样，混在一起算出的"阴影"
其实是通道差异（B 的暗角天然比 G 明显），分开算才能反映真实的光学 shading。
"""
from __future__ import annotations

import numpy as np

from algorithms import _common as C
from .base import Algorithm


class ShadingAnalysisAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "Shading / Vignetting"

    @property
    def description(self) -> str:
        return ("阴影/暗角分析：分块统计各相位均值，报告 shading 幅度、四角与中心差异，"
                "可选生成平坦场校正图。")

    def get_parameters(self) -> dict:
        return {
            "blocks": {
                "type": "int", "default": 16, "min": 2, "max": 128,
                "label": "分块数 (每轴)",
            },
            "metric": {
                "type": "list", "options": ["mean", "median"],
                "default": "mean", "label": "块统计量",
            },
            "warn_pct": {
                "type": "float", "default": 5.0, "min": 0.1, "max": 100.0,
                "label": "异常块阈值 (%)",
                "tooltip": "块均值偏离全图均值超过该百分比时标出",
            },
            "correct": {
                "type": "bool", "default": False,
                "label": "生成平坦场校正图",
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _block_stats(plane: np.ndarray, blocks: int, metric: str):
        """返回 (blocks, blocks) 的块统计量与块边界索引。"""
        ph, pw = plane.shape
        if ph < blocks or pw < blocks:
            return None, None, None
        ys = np.linspace(0, ph, blocks + 1).astype(int)
        xs = np.linspace(0, pw, blocks + 1).astype(int)
        data = plane.astype(np.float64)
        if metric == "median":
            out = np.empty((blocks, blocks), dtype=np.float64)
            for i in range(blocks):
                for j in range(blocks):
                    cell = data[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                    out[i, j] = np.median(cell) if cell.size else 0.0
            return out, ys, xs
        sums = np.add.reduceat(np.add.reduceat(data, ys[:-1], axis=0), xs[:-1], axis=1)
        counts = np.outer(np.diff(ys), np.diff(xs)).astype(np.float64)
        return sums / np.maximum(counts, 1.0), ys, xs

    def run(self, image_data: np.ndarray, params: dict):
        pattern = params.get("pattern", "Mono/None")
        blocks = max(2, int(params.get("blocks", 16) or 16))
        metric = params.get("metric", "mean")
        warn_pct = float(params.get("warn_pct", 5.0) or 5.0)
        do_correct = bool(params.get("correct", False))
        bit_depth = int(params.get("_bit_depth", 16) or 16)
        max_code = (1 << bit_depth) - 1

        work, origin = C.crop_with_origin(image_data, params.get("_roi"))
        corrected = work.astype(np.float64).copy() if do_correct else None

        flagged = {}          # (i, j) -> {"rect", "best", "phases"}
        reports = {}
        lines = []

        for view in C.plane_views(work, pattern, origin):
            plane = view["plane"]
            if plane.size == 0:
                continue
            stats, ys, xs = self._block_stats(plane, blocks, metric)
            if stats is None:
                continue
            base_x, base_y, step = view["base_x"], view["base_y"], view["step"]
            global_mean = float(np.mean(plane))
            bmean = float(np.mean(stats))
            bmin, bmax = float(np.min(stats)), float(np.max(stats))
            shading_pct = (bmax - bmin) / bmean * 100.0 if bmean else 0.0
            k = max(1, blocks // 8)
            mid = blocks // 2
            corners = np.concatenate([stats[:k, :k].ravel(), stats[:k, -k:].ravel(),
                                      stats[-k:, :k].ravel(), stats[-k:, -k:].ravel()])
            center = stats[max(0, mid - k):mid + k, max(0, mid - k):mid + k].ravel()
            corner_ratio = (float(np.mean(corners) / np.mean(center))
                            if center.size and np.mean(center) else 1.0)
            reports[view["name"]] = {
                "mean": global_mean, "block_min": bmin, "block_max": bmax,
                "shading_pct": shading_pct, "corner_over_center": corner_ratio,
                "block_means": stats.tolist(),
            }
            lines.append(f"{view['name']}: {shading_pct:.2f}% "
                         f"(min {bmin:.0f}/max {bmax:.0f}/mean {bmean:.0f}), "
                         f"四角/中心 {corner_ratio:.3f}")

            for i in range(blocks):
                for j in range(blocks):
                    dev_pct = (stats[i, j] - bmean) / bmean * 100.0 if bmean else 0.0
                    if abs(dev_pct) <= warn_pct:
                        continue
                    key = (i, j)
                    entry = flagged.get(key)
                    # 注意：xs/ys 是**相位子平面**的块边界，转全局必须乘 step
                    # （x = base_x + step*px）。漏乘会让 Bayer 下的方框只覆盖
                    # 画面左上 1/4，缺陷清单坐标也只有真值的一半。
                    rect = (int(base_x + step * xs[j]), int(base_y + step * ys[i]),
                            int(step * (xs[j + 1] - xs[j])),
                            int(step * (ys[i + 1] - ys[i])))
                    if entry is None:
                        flagged[key] = {"rect": rect, "best": abs(dev_pct),
                                        "phases": [view["name"]],
                                        "value": float(stats[i, j]),
                                        "delta": float(dev_pct),
                                        "mean": bmean}
                    else:
                        entry["phases"].append(view["name"])
                        if abs(dev_pct) > entry["best"]:
                            entry.update(best=abs(dev_pct), value=float(stats[i, j]),
                                         delta=float(dev_pct), mean=bmean)

            if do_correct and bmean > 0:
                gain = bmean / np.maximum(stats, 1e-6)
                gain_map = np.repeat(np.repeat(gain, np.maximum(np.diff(ys), 1), axis=0),
                                     np.maximum(np.diff(xs), 1), axis=1)
                gy = min(gain_map.shape[0], plane.shape[0])
                gx = min(gain_map.shape[1], plane.shape[1])
                off_y = view["base_y"] - origin[1]
                off_x = view["base_x"] - origin[0]
                sub = corrected[off_y::view["step"], off_x::view["step"]]
                hh, ww = sub.shape
                corrected[off_y::view["step"], off_x::view["step"]] = np.clip(
                    sub * gain_map[:hh, :ww], 0, max_code)

        overlays, defects = [], []
        for (i, j), e in sorted(flagged.items()):
            overlays.append({"type": "rect", "coords": e["rect"],
                             "kind": "shading", "color": "shading"})
            x, y, w, h = e["rect"]
            defects.append({
                "type": "shading", "x": int(x + w // 2), "y": int(y + h // 2),
                "channel": "/".join(sorted(set(e["phases"]))),
                "value": round(e["value"], 1),
                "delta": round(e["delta"], 2),
                "note": f"block[{i},{j}] {e['delta']:+.1f}% (ref {e['mean']:.0f})",
            })

        msg_lines = ["阴影分析（分相位）:"] + lines
        msg_lines.append(f"异常块 {len(defects)} 个（阈值 ±{warn_pct:.2f}%）" if defects
                         else f"无异常块（阈值 ±{warn_pct:.2f}%）")
        if do_correct:
            msg_lines.append("已生成平坦场校正图（块增益最近邻）")

        result_image = image_data
        if do_correct and corrected is not None:
            result_image = C.paste_back(image_data,
                                        np.rint(corrected).astype(image_data.dtype), origin)
        return {
            "image": result_image,
            "corrected": do_correct,
            "overlays": overlays,
            "defects": defects,
            "message": "\n".join(msg_lines),
            "report": reports,
        }
