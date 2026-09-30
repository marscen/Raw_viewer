"""饱和 / 裁剪（Clipping）与黑点检查。

曝光与动态范围验收的常规动作：
  * 有多少像素顶到满量程（过曝，信息已丢失）？分布在哪里（哪一块先饱和）？
  * 有多少像素是 0 DN（死点/黑电平异常）？
  * 各通道饱和比例是否失衡（例如 B 通道过早饱和 → 白平衡/增益配置问题）？

输出：分通道计数与百分比 + 过饱和区域的块状分布（避免在整片高光上逐像素标注，
块状分布既直观又不卡）+ 少量精确像素点（总数不多时）。
"""
from __future__ import annotations

import numpy as np

from algorithms import _common as C
from utils import accel, cfa
from .base import Algorithm


class ClipCheckAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "Saturation / Clipping"

    @property
    def description(self) -> str:
        return ("饱和与黑点检查：统计各通道顶到满量程 / 为 0 的像素比例，"
                "用块状分布指出过曝区域，用于曝光与动态范围验收。")

    def get_parameters(self) -> dict:
        return {
            "sat_level": {
                "type": "int", "default": 0, "min": 0, "max": 65535,
                "label": "饱和门限 (0=自动满量程)",
            },
            "zero_level": {
                "type": "int", "default": 0, "min": 0, "max": 65535,
                "label": "黑点门限",
            },
            "check_zero": {"type": "bool", "default": True, "label": "检查 0 DN 像素"},
            "warn_pct": {
                "type": "float", "default": 0.5, "min": 0.001, "max": 100.0,
                "label": "块告警阈值 (%)",
            },
            "blocks": {"type": "int", "default": 24, "min": 2, "max": 128,
                       "label": "分块数 (每轴)"},
            "point_limit": {"type": "int", "default": 2000, "min": 0, "max": 100000,
                            "label": "精确定位点数上限"},
            "region_limit": {"type": "int", "default": 50000, "min": 100, "max": 5000000,
                             "label": "区域分析像素上限",
                             "tooltip": "饱和像素超过该数量时退化为块状统计，避免百万级像素逐点分析"},
        }

    def run(self, image_data: np.ndarray, params: dict):
        pattern = params.get("pattern", "Mono/None")
        bit_depth = int(params.get("_bit_depth", 16) or 16)
        max_code = (1 << bit_depth) - 1
        sat_level = int(params.get("sat_level", 0) or 0) or max_code
        zero_level = int(params.get("zero_level", 0) or 0)
        check_zero = bool(params.get("check_zero", True))
        warn_pct = float(params.get("warn_pct", 0.5) or 0.5)
        blocks = max(2, int(params.get("blocks", 24) or 24))
        point_limit = int(params.get("point_limit", 2000) or 0)

        work, origin = C.crop_with_origin(image_data, params.get("_roi"))
        h, w = work.shape
        sat_mask = work >= sat_level
        zero_mask = work <= zero_level if check_zero else np.zeros_like(sat_mask)

        overlays, defects = [], []
        msg_lines = [f"饱和门限 {sat_level} DN（满量程 {max_code}），黑点门限 {zero_level} DN"]

        # ---- 分通道统计 ----
        for view in C.plane_views(work, pattern, origin):
            plane = view["plane"]
            if plane.size == 0:
                continue
            n = plane.size
            sat = int(np.count_nonzero(plane >= sat_level))
            zero = int(np.count_nonzero(plane <= zero_level)) if check_zero else 0
            msg_lines.append(
                f"{view['name']}: 饱和 {sat} ({100.0 * sat / n:.4f}%)  "
                f"0DN {zero} ({100.0 * zero / n:.4f}%)")

        # ---- 过曝区域：能算连通区域就算区域（有 bbox/面积/峰值），
        #      像素太多时才退化成块状统计（避免百万级像素的逐点开销）----
        sat_total = int(np.count_nonzero(sat_mask))
        region_limit = int(params.get("region_limit", 50000) or 50000)
        hot_blocks = 0
        if sat_total and sat_total <= region_limit:
            labels, sizes = accel.connected_components(sat_mask, 8)
            n_regions = int(labels.max())
            rows_map = {}
            ys_r, xs_r = np.nonzero(sat_mask)
            for py, px, lb in zip(ys_r.tolist(), xs_r.tolist(), labels[sat_mask].tolist()):
                e = rows_map.get(lb)
                if e is None:
                    rows_map[lb] = [py, py, px, px, 1, int(work[py, px])]
                else:
                    e[0] = min(e[0], py); e[1] = max(e[1], py)
                    e[2] = min(e[2], px); e[3] = max(e[3], px)
                    e[4] += 1
                    e[5] = max(e[5], int(work[py, px]))
            for lb, (ry0, ry1, rx0, rx1, area, peak) in sorted(rows_map.items()):
                gx0, gy0 = int(origin[0] + rx0), int(origin[1] + ry0)
                gx1, gy1 = int(origin[0] + rx1), int(origin[1] + ry1)
                overlays.append({"type": "rect", "kind": "sat", "color": "sat",
                                 "coords": (gx0, gy0, gx1 - gx0 + 1, gy1 - gy0 + 1)})
                defects.append({
                    "type": "sat", "x": int((gx0 + gx1) // 2), "y": int((gy0 + gy1) // 2),
                    "channel": cfa.plane_name_at((gx0 + gx1) // 2, (gy0 + gy1) // 2, pattern),
                    "value": peak, "delta": peak - sat_level,
                    "note": f"过热区域 x{area} bbox({gx0},{gy0})-({gx1},{gy1})",
                })
            hot_blocks = n_regions
            msg_lines.append(f"过曝区域 {n_regions} 个（已用方框标出 bbox）")
        else:
            blocks = max(2, int(params.get("blocks", 24) or 24))
            ys = np.linspace(0, h, blocks + 1).astype(int)
            xs = np.linspace(0, w, blocks + 1).astype(int)
            sat_sum = np.add.reduceat(
                np.add.reduceat(sat_mask.astype(np.int64), ys[:-1], axis=0), xs[:-1], axis=1)
            counts = np.outer(np.diff(ys), np.diff(xs))
            pct = 100.0 * sat_sum / np.maximum(counts, 1)
            for i in range(blocks):
                for j in range(blocks):
                    if pct[i, j] <= warn_pct:
                        continue
                    hot_blocks += 1
                    rect = (int(origin[0] + xs[j]), int(origin[1] + ys[i]),
                            int(xs[j + 1] - xs[j]), int(ys[i + 1] - ys[i]))
                    overlays.append({"type": "rect", "coords": rect,
                                     "kind": "sat", "color": "sat"})
                    defects.append({
                        "type": "sat", "x": int(rect[0] + rect[2] // 2),
                        "y": int(rect[1] + rect[3] // 2), "channel": "-",
                        "value": round(float(pct[i, j]), 2),
                        "delta": round(float(pct[i, j]), 2),
                        "note": f"饱和块 {pct[i, j]:.2f}%（像素过多，按块统计）",
                    })
            msg_lines.append(f"饱和块 {hot_blocks} 个（> {warn_pct:.3f}%，像素过多按块统计）")

        # ---- 精确点位（数量不多时）----
        if point_limit and 0 < sat_total <= point_limit:
            ys_i, xs_i = np.nonzero(sat_mask)
            for py, px in zip(ys_i.tolist(), xs_i.tolist()):
                gx, gy = int(origin[0] + px), int(origin[1] + py)
                overlays.append({"type": "point", "coords": (gx, gy),
                                 "kind": "sat", "color": "sat", "radius": 2})
                # 少量饱和像素通常是坏点/漏光，值得逐点进清单
                defects.append({
                    "type": "sat", "x": gx, "y": gy,
                    "channel": cfa.plane_name_at(int(origin[0] + px), int(origin[1] + py),
                                                 pattern),
                    "value": int(work[py, px]),
                    "delta": round(float(work[py, px] - sat_level), 2),
                    "note": "saturated pixel",
                })
        zero_total = int(np.count_nonzero(zero_mask))
        if check_zero and 0 < zero_total <= point_limit:
            ys_i, xs_i = np.nonzero(zero_mask)
            for py, px in zip(ys_i.tolist(), xs_i.tolist()):
                overlays.append({"type": "point",
                                 "coords": (int(origin[0] + px), int(origin[1] + py)),
                                 "kind": "dead", "color": "dead", "radius": 2})

        total = max(1, work.size)
        msg_lines.append(f"合计: 饱和 {sat_total} ({100.0 * sat_total / total:.4f}%)  "
                         f"0DN {zero_total} ({100.0 * zero_total / total:.4f}%)")
        if sat_total == 0:
            msg_lines.append("无饱和像素：曝光未削顶（也可据此加大曝光找上限）")
        return {
            "image": image_data,
            "overlays": overlays,
            "defects": defects,
            "message": "\n".join(msg_lines),
            "report": {"sat_level": sat_level, "sat_total": sat_total,
                       "zero_total": zero_total, "hot_blocks": hot_blocks},
        }
