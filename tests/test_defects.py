"""缺陷检测算法测试：坏点/坏线/阴影/饱和/帧差，含与生成器注入缺陷的对拍。"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests._util import check, check_close, run  # noqa: E402

from algorithms.bad_line import BadLineDetectionAlgorithm  # noqa: E402
from algorithms.bad_pixel import BadPixelDetectionAlgorithm  # noqa: E402
from algorithms.clip_check import ClipCheckAlgorithm  # noqa: E402
from algorithms.frame_diff import FrameDiffAlgorithm  # noqa: E402
from algorithms.shading import ShadingAnalysisAlgorithm  # noqa: E402
from utils import cfa  # noqa: E402
from utils.raw_io import RawLoadSpec, load_raw  # noqa: E402

TMP = tempfile.mkdtemp(prefix="rawv2_def_")


def defaults(algo, **over):
    p = {k: v["default"] for k, v in algo.get_parameters().items()}
    p.update(over)
    return p


def _ctx(**over):
    ctx = {"pattern": "RGGB", "_bit_depth": 10, "_roi": None, "_reference": None}
    ctx.update(over)
    return ctx


def _bayer_flat(shape=(120, 160), pattern="RGGB", values=None, noise=0.0, seed=0):
    values = values or {"R": 500, "Gr": 400, "Gb": 400, "B": 300}
    out = np.zeros(shape, np.float32)
    names = cfa.PHASE_LABELS[cfa.normalize_pattern(pattern)]
    for phase, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        out[dy::2, dx::2] = values[names[phase]]
    if noise:
        out += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(out, 0, 1023).astype(np.uint16)


# ----------------------------------------------------------------------
def test_bad_pixel_hot_dead_cluster():
    algo = BadPixelDetectionAlgorithm()
    img = _bayer_flat(noise=1.0, seed=1)
    img[11, 21] = 1023        # 孤立 hot（离其它缺陷足够远，才能算 single）
    img[30, 40] = 0           # 孤立 dead
    img[80:82, 100:102] = 1023  # 2x2 簇：四个相位各一个
    res = algo.run(img, defaults(algo, **_ctx(visualize_only=True)))
    found = {(d["x"], d["y"]): d["type"] for d in res["defects"]}
    check((21, 11) in found and found[(21, 11)] == "hot", "检出 hot 坏点")
    check((40, 30) in found and found[(40, 30)] == "dead", "检出 dead 坏点（偏暗）")
    cluster = [d for d in res["defects"] if d["type"] == "cluster"]
    check(len(cluster) == 4, f"2x2 簇被识别为 cluster（实际 {len(cluster)} 个）")
    check(all(d["note"].startswith("cluster") for d in cluster), "簇的 note 标注")
    check("hot=1" in res["message"] and "cluster=4" in res["message"],
          f"消息含分类计数: {res['message']}")
    # 干净图不应有误报（阈值 100 DN，噪声 σ≈1）
    clean = _bayer_flat(noise=1.0, seed=2)
    res2 = algo.run(clean, defaults(algo, **_ctx()))
    check(len(res2["defects"]) == 0, f"干净 Bayer 场无假坏点（实际 {len(res2['defects'])}）")


def test_bad_pixel_correction_writes_correct_pixels():
    """校正必须写回**全局坐标**，不能写成相位子平面坐标（曾经的 bug）。"""
    algo = BadPixelDetectionAlgorithm()
    img = np.full((64, 64), 64, np.uint16)
    for (y, x) in ((10, 10), (20, 20), (21, 22), (33, 41)):
        img[y, x] = 1023
    res = algo.run(img, defaults(algo, visualize_only=False, **_ctx()))
    check(res["corrected"] is True, "标记为已校正")
    changed = np.argwhere(res["image"] != img)
    check(changed.shape[0] == 4, f"改动像素数 = 坏点数（实际 {changed.shape[0]}）")
    check(all(res["image"][y, x] == 64 for y, x in ((10, 10), (20, 20), (21, 22), (33, 41))),
          "坏点被替换为同色邻域中值")
    check(img[10, 10] == 1023, "原图未被修改（返回新数组）")


def test_bad_pixel_odd_roi_parity():
    """奇数起点的 ROI 内检测同样不能串通道。"""
    algo = BadPixelDetectionAlgorithm()
    img = _bayer_flat((64, 64), noise=0.0)
    img[20, 20] = 1023        # RGGB 下 (偶,偶) 是 R 相位；ROI(1,1,41,41) 起点为奇数
    res = algo.run(img, defaults(algo, **_ctx(_roi=(1, 1, 41, 41))))
    coords = [(d["x"], d["y"], d["channel"]) for d in res["defects"]]
    check((20, 20, "R") in coords, f"奇数 ROI 内坏点被正确归到 R 相位: {coords}")
    res_c = algo.run(img, defaults(algo, visualize_only=False, **_ctx(_roi=(1, 1, 41, 41))))
    check(int(res_c["image"][20, 20]) == 500, "奇数 ROI 内校正写回正确位置")


def test_bad_line_rows_cols_and_gradient_robustness():
    algo = BadLineDetectionAlgorithm()
    # 平坦场 + 坏行 + 坏列
    img = np.full((120, 160), 400, np.uint16)
    img[60, :] = 700
    img[:, 100] = 100
    res = algo.run(img, defaults(algo, pattern="Mono/None", axis="Both", threshold=100))
    kinds = {(d["type"], d["y"] if d["type"] == "row" else d["x"]) for d in res["defects"]}
    check(("row", 60) in kinds, f"检出坏行 60: {kinds}")
    check(("col", 100) in kinds, f"检出坏列 100: {kinds}")
    check(len(res["defects"]) == 2, f"无额外误报（实际 {len(res['defects'])}）")

    # 强亮度梯度（阴影）下仍不应误报 —— 用全图 median 作参考的旧算法会整片误报
    yy, xx = np.mgrid[0:120, 0:160].astype(np.float32)
    grad = (300 + 400 * xx / 159).astype(np.uint16)
    res2 = algo.run(grad, defaults(algo, pattern="Mono/None", axis="Rows", threshold=100))
    check(len(res2["defects"]) == 0,
          f"阴影梯度下不误报坏行（实际 {len(res2['defects'])}）")

    # Bayer 坏列：同色比较后应定位到全局列
    img3 = _bayer_flat((120, 160), "BGGR", noise=0.5, seed=3)
    img3[:, 50] = 1000
    res3 = algo.run(img3, defaults(algo, pattern="BGGR", axis="Cols", threshold=200))
    cols = {d["x"] for d in res3["defects"]}
    check(50 in cols, f"Bayer 图检出坏列 50（实际 {cols}）")
    check(all(d["type"] == "col" for d in res3["defects"]), "只报列")


def test_bad_line_segment_mode():
    """只有半行坏时，分段模式要能报出来，整行模式可能被平均掉。"""
    algo = BadLineDetectionAlgorithm()
    img = np.full((120, 200), 400, np.uint16)
    img[60, :100] = 900                       # 左半行坏
    res = algo.run(img, defaults(algo, pattern="Mono/None", axis="Rows",
                                 threshold=150, block_size=64, min_segment=32))
    segs = [d for d in res["defects"] if d["type"] == "row" and d["y"] == 60]
    check(len(segs) >= 1, f"分段模式检出局部坏行（{res['defects']}）")
    check(any("seg" in d["note"] for d in segs), f"标注为分段: {[d['note'] for d in segs]}")


def test_bad_pixel_large_cluster_decomposition():
    """整列/整行坏点不能刷出上千条明细，也不能画成覆盖全图的大方框。"""
    algo = BadPixelDetectionAlgorithm()
    img = np.full((128, 128), 400, np.uint16)
    img[:, 60] = 900                      # 整列偏亮
    img[100, :] = 900                     # 整行偏亮
    res = algo.run(img, defaults(algo, **_ctx(pattern="Mono/None", threshold=100)))
    typed = {(d["type"], d["x"] if d["type"] == "col" else d["y"]) for d in res["defects"]}
    check(("col", 60) in typed, f"整列被报成 col（{sorted(typed)}）")
    check(("row", 100) in typed, f"整行被报成 row（{sorted(typed)}）")
    check(len(res["defects"]) <= 4,
          f"聚合后条目很少（实际 {len(res['defects'])} 条）")
    big_box = [d for d in res["defects"]
               if d["type"] == "cluster" and "bbox(0,0)-(127,127)" in d["note"]]
    check(not big_box, "没有覆盖全图的无意义方框")
    check(any(o["type"] == "line" for o in res["overlays"]), "坏列/坏行用线标注")
    # 小簇仍然逐点列出（便于精确定位）
    img2 = np.full((64, 64), 400, np.uint16)
    img2[20:22, 30:32] = 900
    res2 = algo.run(img2, defaults(algo, **_ctx(pattern="Mono/None", threshold=100)))
    check(len(res2["defects"]) == 4, f"小簇逐点列出（{len(res2['defects'])} 条）")
    check(all(d["type"] == "cluster" for d in res2["defects"]), "小簇标为 cluster")


def test_shading_analysis():
    algo = ShadingAnalysisAlgorithm()
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    img = (600 * np.clip(1 - 0.5 * r ** 2, 0, 1)).astype(np.uint16)
    res = algo.run(img, defaults(algo, pattern="Mono/None", blocks=8, warn_pct=5.0))
    rep = res["report"]["Mono"]
    check(rep["shading_pct"] > 30, f"检出明显 shading（{rep['shading_pct']:.1f}%）")
    check(rep["corner_over_center"] < 0.8,
          f"四角/中心 < 0.8（{rep['corner_over_center']:.3f}）")
    check(len(res["defects"]) > 0, "标出异常块")
    # 平坦场不应报异常块
    res2 = algo.run(np.full((128, 128), 500, np.uint16),
                    defaults(algo, pattern="Mono/None", blocks=8, warn_pct=5.0))
    check(len(res2["defects"]) == 0, "平坦场无异常块")
    # 校正后 shading 幅度应显著下降
    res3 = algo.run(img, defaults(algo, pattern="Mono/None", blocks=8, correct=True))
    after = res3["image"].astype(np.float32)
    blocks = 8
    ys = np.linspace(0, h, blocks + 1).astype(int)
    center = after[ys[3]:ys[4], ys[3]:ys[4]].mean()
    corner = after[:ys[1], :ys[1]].mean()
    check(abs(corner - center) / center < 0.15,
          f"校正后四角/中心差异 < 15%（{100 * abs(corner - center) / center:.1f}%）")


def test_clip_check():
    algo = ClipCheckAlgorithm()
    img = _bayer_flat((64, 64), values={"R": 500, "Gr": 400, "Gb": 400, "B": 1023})
    res = algo.run(img, defaults(algo, **_ctx()))
    check("B: 饱和 1024" in res["message"] or "B: 饱和 1024" in res["message"].replace(",", ""),
          f"分通道饱和计数: {res['message'].splitlines()[2]}")
    pixels = [d for d in res["defects"] if d["note"] == "saturated pixel"]
    check(len(pixels) == 1024, f"逐点列出饱和像素（{len(pixels)}）")
    check(all(d["channel"] == "B" for d in pixels), "饱和像素通道名为 B")
    check(res["report"]["sat_total"] == 1024, "饱和总数")
    # 无饱和
    res2 = algo.run(_bayer_flat((64, 64)), defaults(algo, **_ctx()))
    check(res2["report"]["sat_total"] == 0, "无饱和像素")
    check("无饱和像素" in res2["message"], "给出无饱和的提示")


def test_frame_diff():
    algo = FrameDiffAlgorithm()
    a = _bayer_flat((48, 64), noise=1.0, seed=5)
    b = a.copy()
    b[10, 10] = a[10, 10] + 200
    res = algo.run(a, defaults(algo, **_ctx(_reference=b)))
    check(len(res["defects"]) == 1, f"检出唯一的改动像素（{len(res['defects'])}）")
    check(res["defects"][0]["x"] == 10 and res["defects"][0]["y"] == 10, "改动坐标")
    check("PSNR" in res["message"], "报告 PSNR")
    same = algo.run(a, defaults(algo, **_ctx(_reference=a.copy())))
    check(len(same["defects"]) == 0 and "inf" in same["message"], "相同帧 PSNR=inf")
    noraise = algo.run(a, defaults(algo, **_ctx()))
    check("参考帧" in noraise["message"], "缺少参考帧时给出提示")


def test_generated_sample_vs_injected_defects():
    """端到端对拍：生成器注入的缺陷必须被检出（含 packed/带 header 布局）。"""
    import generate_sample as gs
    from algorithms.manager import AlgorithmManager

    manager = AlgorithmManager()
    bp = manager.get_algorithm("Bad Pixel Detection")
    bl = manager.get_algorithm("Bad Line Detection")

    data, meta = gs.generate("dark", width=960, height=540, bit_depth=10,
                             pattern="RGGB", seed=11, dark_level=64)
    expected_hot = {(d["x"], d["y"]) for d in meta["injected"] if d["type"] in ("hot", "cluster")}
    expected_dead = {(d["x"], d["y"]) for d in meta["injected"] if d["type"] == "dead"}

    layouts = [("unpacked", {}), ("packed", {}), ("unpacked", {"header": 16, "stride_pad": 32})]
    for i, (packing, opt) in enumerate(layouts):
        path = os.path.join(TMP, f"gen_{i}.raw")
        info = gs.write_sample(path, data, meta, packing, **opt)
        back = load_raw(path, info["spec_object"])
        check(np.array_equal(back, data), f"{packing}{opt} 落盘/读回一致")
        # 暗场黑电平只有 64 DN，最小偏差门限要相应放低（界面上的"最小偏差"就是干这个的）
        res = bp.run(back, defaults(bp, _bit_depth=10, pattern="RGGB", threshold=30))
        found = {(d["x"], d["y"]) for d in res["defects"]}
        check(expected_hot <= found,
              f"{packing}{opt}: 注入的 hot/簇被发现（缺 {sorted(expected_hot - found)[:4]}）")
        check(expected_dead <= found, f"{packing}{opt}: 注入的 dead 被发现")
        res_line = bl.run(back, defaults(bl, _bit_depth=10, pattern="RGGB",
                                        axis="Both", threshold=60))
        kinds = {(d["type"], d["y"] if d["type"] == "row" else d["x"]) for d in res_line["defects"]}
        rows = {d["y"] for d in meta["injected"] if d["type"] == "row"}
        cols = {d["x"] for d in meta["injected"] if d["type"] == "col"}
        for y in rows:
            check(("row", y) in kinds, f"{packing}{opt}: 检出注入坏行 {y}")
        for x in cols:
            check(("col", x) in kinds, f"{packing}{opt}: 检出注入坏列 {x}")


if __name__ == "__main__":
    sys.exit(run(globals(), "缺陷检测算法"))
