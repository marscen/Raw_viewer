"""与参考帧比较（时域噪声 / 坏点确认 / 校正前后验证）。

工程上最常见的用法：
  * 同一曝光下连拍两帧暗场 → 差分即读出噪声与随机噪声；两帧都有、位置固定的
    大偏差就是真坏点（可以据此区分"随机噪点"与"固定坏点"）；
  * 跑完校正算法后与原始帧比较 → PSNR / 最大改动 / 改动像素比例，确认只动了
    该动的点，没有把整幅图拖偏。

参考帧由主窗口注入到 params['_reference']（比对用的原始 DN 数组）。
"""
from __future__ import annotations

import numpy as np

from algorithms import _common as C
from utils import stats as S
from .base import Algorithm


class FrameDiffAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "Frame Diff vs Reference"

    @property
    def description(self) -> str:
        return ("与参考帧比较：PSNR/RMSE/差异像素比例，列出偏差大的像素，"
                "用于时域噪声评估、坏点确认与校正前后验证。")

    def get_parameters(self) -> dict:
        return {
            "tol": {
                "type": "int", "default": 1, "min": 0, "max": 4096,
                "label": "差异计数门限 (DN)",
            },
            "hot_threshold": {
                "type": "float", "default": 64.0, "min": 1.0, "max": 65535.0,
                "label": "大偏差门限 (DN)",
                "tooltip": "超过该差值的像素会进入缺陷清单（可用来确认固定坏点）",
            },
            "list_limit": {
                "type": "int", "default": 5000, "min": 0, "max": 200000,
                "label": "清单上限",
            },
        }

    def run(self, image_data: np.ndarray, params: dict):
        reference = params.get("_reference")
        bit_depth = int(params.get("_bit_depth", 16) or 16)
        max_code = (1 << bit_depth) - 1
        tol = int(params.get("tol", 1) or 0)
        hot_threshold = float(params.get("hot_threshold", 64.0) or 64.0)
        limit = int(params.get("list_limit", 5000) or 0)

        if reference is None or getattr(reference, "shape", None) != image_data.shape:
            return {"image": image_data, "overlays": [], "defects": [],
                    "message": "需要一个同尺寸的参考帧：File → Open Reference Image…",
                    "report": {}}

        res = S.diff_stats(image_data, reference, bit_depth, max_code, tol=tol)
        if "error" in res:
            return {"image": image_data, "overlays": [], "defects": [],
                    "message": res["error"], "report": res}

        work, origin = C.crop_with_origin(image_data, params.get("_roi"))
        ref_work, _ = C.crop_with_origin(reference, params.get("_roi"))
        diff = np.abs(work.astype(np.int32) - ref_work.astype(np.int32))

        overlays, defects = [], []
        mask = diff > hot_threshold
        ys, xs = np.nonzero(mask)
        for py, px in zip(ys.tolist(), xs.tolist()):
            if limit and len(defects) >= limit:
                break
            gx, gy = int(origin[0] + px), int(origin[1] + py)
            signed = int(work[py, px]) - int(ref_work[py, px])
            defects.append({
                "type": "hot" if signed > 0 else "dead",
                "x": gx, "y": gy,
                "channel": "-",
                "value": int(work[py, px]),
                "delta": signed,
                "note": f"|Δ|={int(diff[py, px])} (ref {int(ref_work[py, px])})",
            })
            overlays.append({"type": "point", "coords": (gx, gy),
                             "kind": "hot" if signed > 0 else "dead",
                             "color": "hot" if signed > 0 else "dead", "radius": 3})

        psnr = res["psnr"]
        message = (
            f"PSNR {psnr:.2f} dB   RMSE {res['rmse']:.3f} DN\n"
            f"平均|Δ| {res['mean_abs']:.3f}   最大|Δ| {res['max_abs']:.0f}   "
            f"差异像素(> {tol} DN) {res['count_diff']} ({res['pct_diff']:.4f}%)\n"
            f"|Δ| > {hot_threshold:.0f} DN 的像素 {int(mask.sum())} 个（已列入缺陷清单）")
        return {
            "image": image_data,
            "overlays": overlays,
            "defects": defects,
            "message": message,
            "report": {k: v for k, v in res.items() if k != "diff_map"},
        }
