"""坏点检测（Hot / Dead / Cluster / 坏行坏列）。

相对旧版的改进：
  1. **只和同色邻居比**：按相位拆平面后，平面内相邻采样点必然是同一颜色通道，
     因此不会再出现"R 和 G 比、1000 DN 差异"这种假坏点；
  2. **自适应阈值**：用 MAD(1.4826×median|x−median|) 估噪声 σ，判据是
     |Δ| > max(k·σ, 最小偏差)。换增益/换温度不用重新调阈值；
  3. **分类 + 校正**：区分 hot/dead/cluster，可用同色邻域中值就地校正；
  4. **大簇不刷屏**：上千像素的坏列/过曝块会按形状自动拆成 row/col 各一条
     （与坏线检测口径一致），实在成块的才用一条带包围盒的记录表示。

实现上全部走向量化：连通标记走 utils.accel（OpenCV 或 numpy 行程+并查集），
聚合阶段按"簇"分组用 numpy 归约，**不做逐像素 Python 循环** —— 早期版本在
100 万像素的缺陷块上因为逐像素建元组要 92 秒，现在几十毫秒。
"""
from __future__ import annotations

import numpy as np

from algorithms import _common as C
from utils import accel, cfa
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
            "report_min_cluster": {
                "type": "int", "default": 1, "min": 1, "max": 100000,
                "label": "只报簇(像素数≥)",
                "tooltip": "设为 2 以上可以滤掉噪声引起的孤立点，只保留成簇/成线缺陷",
            },
            "remove_isolated": {
                "type": "bool", "default": False,
                "label": "剔除孤立单点",
                "tooltip": "只保留连通面积 ≥2 的缺陷（不会破坏坏线/坏块），"
                           "用于噪声较大的图，避免把随机噪点当成坏点去校正",
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
        report_min_cluster = max(1, int(params.get("report_min_cluster", 1) or 1))
        remove_isolated = bool(params.get("remove_isolated", False))
        visualize_only = bool(params.get("visualize_only", True))
        limit = int(params.get("limit", 20000) or 20000)
        bit_depth = int(params.get("_bit_depth", 16) or 16)
        max_code = (1 << bit_depth) - 1

        work, origin = C.crop_with_origin(image_data, params.get("_roi"))
        corrected = work.copy() if not visualize_only else None

        # ---- 第一遍：逐相位（同色）判定，同时拼出整幅掩罩与偏差图 ----
        per_plane = []
        full_mask = np.zeros(work.shape, dtype=bool)
        delta_map = np.zeros(work.shape, dtype=np.float32)   # 供聚合阶段向量化归约
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
            delta_map[off_y::step, off_x::step] = delta
            if count:
                full_mask[off_y::step, off_x::step] |= mask
            per_channel[view["name"]] = {
                "sigma": sigma, "threshold": thr, "count": count,
                "pixels": int(plane.size),
            }
            per_plane.append({"view": view, "local": local, "delta": delta, "mask": mask,
                              "off_y": off_y, "off_x": off_x, "step": step})

        # ---- 连通标记（整幅 + 膨胀，覆盖 2x2 跨相位簇与同色连片）----
        labels = None
        sizes_real = None
        if full_mask.any():
            labels, _ = accel.connected_components(accel.dilate_mask(full_mask, 1), 8)
            if labels.max() > 0:
                # 面积只统计真缺陷像素，否则膨胀出来的"桥"会把孤立点算成 9 像素簇
                sizes_real = np.bincount(labels[full_mask],
                                         minlength=int(labels.max()) + 1)

        # 剔除孤立单点（按面积，保留坏线/坏块），并同步回各相位掩罩
        cleanup_removed = 0
        if remove_isolated and sizes_real is not None:
            before_px = int(sizes_real.sum())
            keep_area = sizes_real >= 2
            keep_area[0] = False
            full_mask = keep_area[labels] & full_mask
            for item in per_plane:
                item["mask"] = full_mask[item["off_y"]::item["step"],
                                         item["off_x"]::item["step"]]
            labels, _ = accel.connected_components(accel.dilate_mask(full_mask, 1), 8)
            sizes_real = (np.bincount(labels[full_mask], minlength=int(labels.max()) + 1)
                          if labels.max() > 0 else None)
            after_px = int(sizes_real.sum()) if sizes_real is not None else 0
            cleanup_removed = max(0, before_px - after_px)

        # ---- 校正（向量化）----
        # 放在 remove_isolated **之后**：被剔除的孤立噪点不应该被"修"，
        # 否则等于把随机噪声当成坏点写进图里。
        if corrected is not None:
            for item in per_plane:
                m = item["mask"]
                if not m.any():
                    continue
                tgt = corrected[item["off_y"]::item["step"], item["off_x"]::item["step"]]
                tgt[m] = np.clip(np.rint(item["local"][m]), 0,
                                 max_code).astype(corrected.dtype)

        # 只报"够大的簇"（不影响校正）
        reportable = None
        if report_min_cluster > 1 and sizes_real is not None:
            keep = sizes_real >= int(report_min_cluster)
            keep[0] = False
            reportable = keep[labels]

        cluster_sizes = sizes_real if cluster_enable else None

        # ---- 分簇：小簇逐点报；大簇按簇做向量化聚合 ----
        defects, overlays = [], []
        big_ids = np.array([], dtype=np.int64)
        big_pixels = {}
        big_lookup = None
        if sizes_real is not None:
            idx = np.nonzero(sizes_real >= aggregate_min)[0]
            big_ids = idx[idx > 0]
        if big_ids.size and labels is not None:
            sel = np.isin(labels, big_ids) & full_mask
            big_lookup = sel
            ys_b, xs_b = np.nonzero(sel)
            ids_b = labels[ys_b, xs_b]
            order = np.argsort(ids_b, kind="stable")
            ys_b, xs_b, ids_b = ys_b[order], xs_b[order], ids_b[order]
            uniq, starts = np.unique(ids_b, return_index=True)
            for t, lbl in enumerate(uniq.tolist()):
                a = int(starts[t])
                b = int(starts[t + 1]) if t + 1 < len(starts) else int(ids_b.size)
                big_pixels[lbl] = (ys_b[a:b], xs_b[a:b])

        # 小簇/孤立点：逐点（数量级可控）
        for item in per_plane:
            m = item["mask"]
            if not m.any():
                continue
            view, delta = item["view"], item["delta"]
            off_y, off_x, step = item["off_y"], item["off_x"], item["step"]
            ys, xs = np.nonzero(m)
            for py, px in zip(ys.tolist(), xs.tolist()):
                cy, cx = off_y + step * py, off_x + step * px
                if big_lookup is not None and big_lookup[cy, cx]:
                    continue                       # 该像素属于大簇，稍后聚合上报
                if reportable is not None and not reportable[cy, cx]:
                    continue
                d = float(delta[py, px])
                kind = "hot" if d > 0 else "dead"
                size = 1
                if cluster_sizes is not None and labels is not None:
                    lbl = int(labels[cy, cx])
                    size = int(cluster_sizes[lbl]) if 0 < lbl < cluster_sizes.size else 1
                    if size >= cluster_min:
                        kind = "cluster"
                gx, gy = cx + origin[0], cy + origin[1]
                defects.append({
                    "type": kind, "x": int(gx), "y": int(gy),
                    "channel": view["name"],
                    "value": int(work[cy, cx]),
                    "delta": round(d, 2),
                    "note": "single" if size < cluster_min else f"cluster x{size}",
                })
                overlays.append(C.defect_overlay(defects[-1]))
            if len(defects) > limit:
                defects = defects[:limit]
                overlays = overlays[:limit]      # 叠加层与清单必须一致
                break

        # 大簇：按形状拆成 row/col（与坏线检测口径一致），剩下成块的部分给一条包围盒
        # 大簇：按形状拆成 row/col（与坏线检测口径一致），剩下成块的部分给一条包围盒。
        # 关键：所有每行/每列的统计都用**一次** bincount 归约算出来，
        # 不能每拆一行就到整个簇上做一次 where/argmax —— 那在"整幅都是缺陷"
        # 的噪声掩罩上会退化成 O(行数 × 簇大小)，实测 80 秒。
        MAX_LINES = 64
        names = cfa.PHASE_LABELS.get(cfa.normalize_pattern(pattern), ("Mono",))

        for lbl, (cys, cxs) in big_pixels.items():
            if len(defects) > limit:
                break
            y0, y1 = int(cys.min()), int(cys.max())
            x0, x1 = int(cxs.min()), int(cxs.max())
            w_span, h_span = x1 - x0 + 1, y1 - y0 + 1
            dvals = delta_map[cys, cxs]
            vvals = work[cys, cxs].astype(np.float64)
            rr = cys - y0
            cc = cxs - x0
            row_cnt = np.bincount(rr, minlength=h_span)
            col_cnt = np.bincount(cc, minlength=w_span)
            row_dsum = np.bincount(rr, weights=dvals, minlength=h_span)
            row_vsum = np.bincount(rr, weights=vvals, minlength=h_span)
            col_dsum = np.bincount(cc, weights=dvals, minlength=w_span)
            col_vsum = np.bincount(cc, weights=vvals, minlength=w_span)

            def _row_channels(gy):
                if len(names) < 4:                      # Mono
                    return names[0]
                return "/".join(names[(gy % 2) * 2 + dx] for dx in (0, 1))

            def _col_channels(gx):
                if len(names) < 4:
                    return names[0]
                return "/".join(names[dy * 2 + (gx % 2)] for dy in (0, 1))

            def _emit_row(i):
                gy = y0 + i + origin[1]
                cnt = max(1, int(row_cnt[i]))
                note = (f"full row from cluster x{cnt}" if cnt >= 0.98 * work.shape[1]
                        else f"row from cluster x{cnt} (seg x{x0 + origin[0]}..{x1 + origin[0]})")
                defects.append({
                    "type": "row", "x": 0, "y": gy,
                    "channel": _row_channels(gy),
                    "value": int(round(float(row_vsum[i] / cnt))),
                    "delta": round(float(row_dsum[i] / cnt), 2),
                    "note": note + "（Δ 为中值法估计，幅度以坏线检测为准）",
                })
                overlays.append({"type": "line", "kind": "row", "color": "row",
                                 "coords": (x0 + origin[0], gy, x1 + origin[0] + 1, gy)})

            def _emit_col(j):
                gx = x0 + j + origin[0]
                cnt = max(1, int(col_cnt[j]))
                note = (f"full col from cluster x{cnt}" if cnt >= 0.98 * work.shape[0]
                        else f"col from cluster x{cnt} (seg y{y0 + origin[1]}..{y1 + origin[1]})")
                defects.append({
                    "type": "col", "x": gx, "y": 0,
                    "channel": _col_channels(gx),
                    "value": int(round(float(col_vsum[j] / cnt))),
                    "delta": round(float(col_dsum[j] / cnt), 2),
                    "note": note + "（Δ 为中值法估计，幅度以坏线检测为准）",
                })
                overlays.append({"type": "line", "kind": "col", "color": "col",
                                 "coords": (gx, y0 + origin[1], gx, y1 + origin[1] + 1)})

            row_flag = np.zeros(h_span, dtype=bool)
            col_flag = np.zeros(w_span, dtype=bool)
            if w_span <= 2 and h_span > 2:                  # 细长竖直 = 坏列
                for j in range(w_span):
                    col_flag[j] = True
                    _emit_col(j)
            elif h_span <= 2 and w_span > 2:                # 细长水平 = 坏行
                for i in range(h_span):
                    row_flag[i] = True
                    _emit_row(i)
            else:
                dense_rows = ([i for i in range(h_span) if row_cnt[i] >= 0.5 * w_span]
                              if w_span >= 8 else [])
                dense_cols = ([j for j in range(w_span) if col_cnt[j] >= 0.5 * h_span]
                              if h_span >= 8 else [])
                if len(dense_rows) + len(dense_cols) > MAX_LINES:
                    # 整个簇几乎铺满整幅（例如阈值给得太低、掩罩全是噪点）：
                    # 逐行报会产出几千条无意义记录，这里只给一条总览
                    dense_rows, dense_cols = [], []
                    defects.append({
                        "type": "cluster",
                        "x": int((x0 + x1) // 2 + origin[0]),
                        "y": int((y0 + y1) // 2 + origin[1]),
                        "channel": "/".join(names),
                        "value": int(work[cys[0], cxs[0]]),
                        "delta": round(float(dvals.mean()), 2),
                        "note": (f"cluster x{cys.size} 覆盖 bbox({x0 + origin[0]},"
                                 f"{y0 + origin[1]})-({x1 + origin[0]},{y1 + origin[1]})"
                                 f"，涉及 {len(dense_rows) + len(dense_cols)} 条行列，"
                                 f"未逐行拆分（阈值可能过低）"),
                    })
                    overlays.append({
                        "type": "rect",
                        "coords": (x0 + origin[0], y0 + origin[1],
                                   int(x1 - x0 + 1), int(y1 - y0 + 1)),
                        "kind": "cluster", "color": "cluster",
                    })
                for i in dense_rows:
                    row_flag[i] = True
                    _emit_row(i)
                for j in dense_cols:
                    col_flag[j] = True
                    _emit_col(j)

            rest = ~(row_flag[rr] | col_flag[cc])
            if int(rest.sum()) >= max(2, cluster_min):
                rys, rxs = cys[rest], cxs[rest]
                worst = int(np.argmax(np.abs(dvals[rest])))
                rx0, rx1 = int(rxs.min()), int(rxs.max())
                ry0, ry1 = int(rys.min()), int(rys.max())
                gx0, gy0 = rx0 + origin[0], ry0 + origin[1]
                gx1, gy1 = rx1 + origin[0], ry1 + origin[1]
                defects.append({
                    "type": "cluster",
                    "x": int((gx0 + gx1) // 2), "y": int((gy0 + gy1) // 2),
                    "channel": cfa.plane_name_at(int(rxs[worst]) + origin[0],
                                                 int(rys[worst]) + origin[1], pattern),
                    "value": int(work[rys[worst], rxs[worst]]),
                    "delta": round(float(dvals[rest][worst]), 2),
                    "note": f"cluster x{int(rest.sum())} bbox({gx0},{gy0})-({gx1},{gy1})",
                })
                overlays.append({
                    "type": "rect",
                    "coords": (gx0, gy0, int(rx1 - rx0 + 1), int(ry1 - ry0 + 1)),
                    "kind": "cluster", "color": "cluster",
                })

        # ---- 汇总 ----
        counts = {}
        for d in defects:
            counts[d["type"]] = counts.get(d["type"], 0) + 1
        noise_txt = "  ".join(
            f"{name}:σ={v['sigma']:.2f}/thr={v['threshold']:.0f}"
            for name, v in per_channel.items() if not name.startswith("_"))
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "无"
        message = f"坏点 {len(defects)} 个（{summary}）\n噪声/门限: {noise_txt}"
        if cleanup_removed:
            message += f"\n剔除孤立单点：{cleanup_removed} 个像素未列入（也不参与校正）"
        if reportable is not None:
            message += f"\n只报告簇（≥{report_min_cluster} 像素），孤立点未列入清单"
        if not visualize_only and corrected is not None:
            message += "\n已用同色邻域中值校正（可用差分视图查看改动量）"

        if len(overlays) > len(defects):         # 兜底：两者行数永远对齐
            overlays = overlays[:len(defects)]
        result_image = image_data if visualize_only else \
            C.paste_back(image_data, corrected, origin)
        return {
            "image": result_image,
            "corrected": (not visualize_only),
            "overlays": overlays,
            "defects": defects,
            "message": message,
            "report": {"per_channel": per_channel, "counts": counts,
                       "total_pixels": total_pixels, "roi": params.get("_roi"),
                       "aggregated_clusters": len(big_pixels),
                       "cleanup_removed": cleanup_removed,
                       "remove_isolated": remove_isolated,
                       "report_min_cluster": report_min_cluster,
                       "backend": accel.backend_name()},
        }
