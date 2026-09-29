"""坏点检测（Hot / Dead / Cluster）。

相对旧版的三点改进：
  1. **只和同色邻居比**：按相位拆平面后，平面内相邻采样点必然是同一颜色通道，
     因此不会再出现"R 和 G 比、1000 DN 差异"这种假坏点；
  2. **自适应阈值**：用 MAD(1.4826×median|x−median|) 估噪声 σ，判据是
     |Δ| > max(k·σ, 最小偏差)。这样换增益/换温度不用重新调阈值；
  3. **分类 + 校正**：区分 hot（偏亮）/ dead（偏暗）/ cluster（成簇），
     并可用同色邻域中值就地校正，返回校正图供与原因做差分对比。
"""
from __future__ import annotations

import numpy as np

from algorithms import _common as C
from .base import Algorithm


class BadPixelDetectionAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "Bad Pixel Detection"

    @property
    def description(self) -> str:
        return ("坏点检测：同色邻域中值比对 + 自适应阈值(MAD/σ)，区分 hot/dead/cluster，"
                "可选邻域中值校正。建议先在暗场图上跑。")

    def get_parameters(self) -> dict:
        return {
            "method": {
                "type": "list",
                "options": ["Adaptive (MAD)", "Sigma (global std)", "Absolute threshold"],
                "default": "Adaptive (MAD)",
                "label": "阈值方式",
                "tooltip": "MAD 对坏点本身不敏感，最适合暗场；Absolute 用于已知固定门限的验收测试",
            },
            "threshold": {
                "type": "float", "default": 100.0, "min": 0.0, "max": 65535.0,
                "label": "最小偏差 |Δ| (DN)",
                "tooltip": "硬下限：即使自适应阈值更低也不会把小于此值的差异判为坏点",
            },
            "sigma_k": {
                "type": "float", "default": 6.0, "min": 0.5, "max": 50.0,
                "label": "自适应系数 k (×σ)",
            },
            "neighbors": {
                "type": "list", "options": ["4 (上下左右)", "8 (含对角)"],
                "default": "4 (上下左右)", "label": "同色邻域",
            },
            "cluster_enable": {
                "type": "bool", "default": True, "label": "识别缺陷簇",
            },
            "cluster_min": {
                "type": "int", "default": 2, "min": 2, "max": 64,
                "label": "簇最小像素数",
            },
            "aggregate_min": {
                "type": "int", "default": 8, "min": 2, "max": 100000,
                "label": "簇聚合阈值 (像素数)",
                "tooltip": "簇内像素数达到该值时只出一条记录（带包围盒），"
                           "避免一条坏列刷出成千上万条明细",
            },
            "visualize_only": {
                "type": "bool", "default": True, "label": "只标记不校正",
            },
            "limit": {
                "type": "int", "default": 20000, "min": 100, "max": 500000,
                "label": "最大输出数",
            },
        }

    # ------------------------------------------------------------------
    def run(self, image_data: np.ndarray, params: dict):
        pattern = params.get("pattern", "Mono/None")
        method = params.get("method", "Adaptive (MAD)")
        floor = float(params.get("threshold", 100.0) or 0.0)
        k = float(params.get("sigma_k", 6.0) or 0.0)
        neighbors = 8 if str(params.get("neighbors", "4")).startswith("8") else 4
        cluster_enable = bool(params.get("cluster_enable", True))
        cluster_min = max(2, int(params.get("cluster_min", 2) or 2))
        aggregate_min = max(2, int(params.get("aggregate_min", 8) or 8))
        visualize_only = bool(params.get("visualize_only", True))
        limit = int(params.get("limit", 20000) or 20000)
        bit_depth = int(params.get("_bit_depth", 16) or 16)
        max_code = (1 << bit_depth) - 1

        work, origin = C.crop_with_origin(image_data, params.get("_roi"))
        corrected = work.copy() if not visualize_only else None

        # ---- 第一遍：逐相位（同色）判定，同时拼出整幅掩罩 ----
        per_plane = []
        full_mask = np.zeros(work.shape, dtype=bool)
        per_channel = {}
        total_pixels = 0
        for view in C.plane_views(work, pattern, origin):
            plane = view["plane"].astype(np.float32)
            if plane.size == 0:
                continue
            local = C.local_median(plane, neighbors)
            delta = plane - local
            if method.startswith("Adaptive"):
                sigma = C.robust_sigma(delta)
            elif method.startswith("Sigma"):
                sigma = float(np.std(plane))
            else:
                sigma = 0.0
            thr = max(k * sigma, floor) if not method.startswith("Absolute") else floor
            if thr <= 0:
                thr = 1.0
            mask = np.abs(delta) > thr
            count = int(np.count_nonzero(mask))
            total_pixels += plane.size
            step = view["step"]
            off_y = view["base_y"] - origin[1]
            off_x = view["base_x"] - origin[0]
            if count:
                full_mask[off_y::step, off_x::step] |= mask
            per_channel[view["name"]] = {
                "sigma": sigma, "threshold": thr, "count": count,
                "pixels": int(plane.size),
            }
            per_plane.append({"view": view, "plane": plane, "local": local,
                              "delta": delta, "mask": mask, "off_y": off_y,
                              "off_x": off_x, "step": step})

        # ---- 簇判定：整幅膨胀后连通标记（可同时覆盖 2x2 跨相位块与同色连片）----
        labels = None
        cluster_sizes = None
        if cluster_enable and full_mask.any():
            dilated = C.dilate(full_mask, iterations=1)
            labels, n_labels = C.label_mask(dilated, connectivity=8)
            if n_labels:
                # 只统计真正是坏点的像素数（膨胀出来的"桥"不算）
                idx = labels[full_mask]
                cluster_sizes = np.bincount(idx, minlength=n_labels + 1)

        # ---- 第二遍：生成缺陷记录 / 可选校正 ----
        # 小簇（< aggregate_min）逐点报，便于精确定位；大簇（坏列/坏块）只报一条
        # 带包围盒的记录，否则一条坏列就能刷出上千条明细，清单没法看。
        defects = []
        overlays = []
        big = {}
        for item in per_plane:
            mask = item["mask"]
            if not mask.any():
                continue
            view, plane, local, delta = item["view"], item["plane"], item["local"], item["delta"]
            off_y, off_x, step = item["off_y"], item["off_x"], item["step"]
            ys, xs = np.nonzero(mask)
            for py, px in zip(ys.tolist(), xs.tolist()):
                d = float(delta[py, px])
                kind = "hot" if d > 0 else "dead"
                note = "single"
                size = 1
                lbl = 0
                if labels is not None and cluster_sizes is not None:
                    lbl = int(labels[off_y + step * py, off_x + step * px])
                    size = int(cluster_sizes[lbl]) if 0 < lbl < cluster_sizes.size else 1
                    if size >= cluster_min:
                        kind = "cluster"
                        note = f"cluster x{size}"
                cy, cx = off_y + step * py, off_x + step * px
                gx = cx + origin[0]
                gy = cy + origin[1]
                if size >= aggregate_min:
                    agg = big.get(lbl)
                    if agg is None:
                        big[lbl] = {"px": [(int(gx), int(gy), int(plane[py, px]),
                                           float(d), view["name"])]}
                    else:
                        agg["px"].append((int(gx), int(gy), int(plane[py, px]),
                                          float(d), view["name"]))
                else:
                    defects.append({
                        "type": kind, "x": int(gx), "y": int(gy),
                        "channel": view["name"],
                        "value": int(plane[py, px]),
                        "delta": round(d, 2),
                        "note": note,
                    })
                    overlays.append(C.defect_overlay(defects[-1]))
                if corrected is not None:
                    corrected[cy, cx] = int(np.clip(round(local[py, px]), 0, max_code))
            if len(defects) > limit:
                defects = defects[:limit]
                break

        for _lbl, agg in sorted(big.items()):
            # 大簇再拆一层：占满整行/整列的像素单独报成 row/col（与坏线检测口径一致），
            # 剩下真正成块的部分才用包围盒 —— 否则"坏行+坏列"会被画成一个覆盖全图的
            # 大方框，完全看不出问题出在哪一行哪一列。
            px_list = agg["px"]
            xs_all = np.array([it[0] for it in px_list])
            ys_all = np.array([it[1] for it in px_list])
            d_all = np.array([it[3] for it in px_list])
            v_all = np.array([it[2] for it in px_list])
            ch_all = [it[4] for it in px_list]
            x0, x1 = int(xs_all.min()), int(xs_all.max())
            y0, y1 = int(ys_all.min()), int(ys_all.max())
            row_counts = np.bincount(ys_all - y0)
            col_counts = np.bincount(xs_all - x0)
            dense_rows = [y0 + i for i, c in enumerate(row_counts)
                          if c >= 0.5 * (x1 - x0 + 1)]
            dense_cols = [x0 + j for j, c in enumerate(col_counts)
                          if c >= 0.5 * (y1 - y0 + 1)]
            in_line = np.zeros(len(px_list), dtype=bool)
            for y in dense_rows:
                sel = ys_all == y
                in_line |= sel
                idx = int(np.argmax(np.abs(np.where(sel, d_all, -1.0))))
                defects.append({
                    "type": "row", "x": 0, "y": int(y),
                    "channel": ch_all[idx],
                    "value": int(round(float(v_all[sel].mean()))),
                    "delta": round(float(d_all[sel].mean()), 2),
                    "note": (f"row from cluster x{int(row_counts[y - y0])}"
                             f"（Δ 为中值法估计，幅度以坏线检测为准）"),
                })
                overlays.append({"type": "line", "coords": (x0, int(y), x1 + 1, int(y)),
                                 "kind": "row", "color": "row"})
            for x in dense_cols:
                sel = xs_all == x
                in_line |= sel
                idx = int(np.argmax(np.abs(np.where(sel, d_all, -1.0))))
                defects.append({
                    "type": "col", "x": int(x), "y": 0,
                    "channel": ch_all[idx],
                    "value": int(round(float(v_all[sel].mean()))),
                    "delta": round(float(d_all[sel].mean()), 2),
                    "note": (f"col from cluster x{int(col_counts[x - x0])}"
                             f"（Δ 为中值法估计，幅度以坏线检测为准）"),
                })
                overlays.append({"type": "line", "coords": (int(x), y0, int(x), y1 + 1),
                                 "kind": "col", "color": "col"})

            rest = [it for it, flag in zip(px_list, in_line) if not flag]
            if len(rest) >= max(2, cluster_min):
                rx = [it[0] for it in rest]
                ry = [it[1] for it in rest]
                worst = max(rest, key=lambda it: abs(it[3]))
                defects.append({
                    "type": "cluster",
                    "x": int((min(rx) + max(rx)) // 2),
                    "y": int((min(ry) + max(ry)) // 2),
                    "channel": worst[4],
                    "value": int(worst[2]),
                    "delta": round(float(worst[3]), 2),
                    "note": (f"cluster x{len(rest)} bbox({min(rx)},{min(ry)})-"
                             f"({max(rx)},{max(ry)})"),
                })
                overlays.append({
                    "type": "rect",
                    "coords": (int(min(rx)), int(min(ry)),
                               int(max(rx) - min(rx) + 1), int(max(ry) - min(ry) + 1)),
                    "kind": "cluster", "color": "cluster",
                })
        counts = {}
        for d in defects:
            counts[d["type"]] = counts.get(d["type"], 0) + 1
        noise_txt = "  ".join(
            f"{name}:σ={v['sigma']:.2f}/thr={v['threshold']:.0f}"
            for name, v in per_channel.items())
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "无"
        message = (f"坏点 {len(defects)} 个（{summary}）\n"
                   f"噪声/门限: {noise_txt}")
        if not visualize_only and corrected is not None:
            message += "\n已用同色邻域中值校正（可用差分视图查看改动量）"

        result_image = image_data if visualize_only else C.paste_back(image_data, corrected, origin)
        return {
            "image": result_image,
            "corrected": (not visualize_only),
            "overlays": overlays,
            "defects": defects,
            "message": message,
            "report": {"per_channel": per_channel, "counts": counts,
                       "total_pixels": total_pixels, "roi": params.get("_roi"),
                       "aggregated_clusters": len(big)},
        }
