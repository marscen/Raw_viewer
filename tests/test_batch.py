"""批量分析 / 黑电平估计测试。"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests._util import check, check_close, run  # noqa: E402

from utils import batch  # noqa: E402

TMP = tempfile.mkdtemp(prefix="rawv2_batch_")


def test_black_level_estimate():
    import generate_sample as gs
    data, _ = gs.generate("dark", width=320, height=240, bit_depth=10,
                          dark_level=64, seed=1)
    check_close(batch.estimate_black_level(data, 10), 64, 1.5, "暗场全图峰值≈黑电平")
    e_roi = batch.estimate_black_level(data, 10, (50, 50, 200, 200))
    check_close(e_roi, 64, 2.0, "ROI 中值估计")
    # 亮场：峰值不是黑电平，应给出 NaN 让界面提示不可信
    bright = np.full((100, 100), 900, np.uint16)
    check(batch.estimate_black_level(bright, 10) != batch.estimate_black_level(bright, 10),
          "亮场返回 NaN（不是数字）")
    # 16bit 容器 / 12bit 数据
    d12, _ = gs.generate("dark", width=200, height=200, bit_depth=12,
                         dark_level=200, seed=2)
    check_close(batch.estimate_black_level(d12, 12), 200, 6.0, "12bit 暗场估计")


def test_analyze_frame_fields():
    import generate_sample as gs
    data, meta = gs.generate("dark", width=320, height=240, bit_depth=10,
                             dark_level=64, seed=3)
    r = batch.analyze_frame(data, "RGGB", 10)
    for key in ("mean", "std", "min", "max", "median", "sat_pct", "zero_pct",
                "black_est", "roi", "backend"):
        check(key in r, f"结果含字段 {key}")
    check(r["roi"] == "full", "无 ROI 时标记 full")
    check_close(r["mean"], float(data.mean()), 0.01, "均值与 numpy 一致")
    check(r["max"] == float(data.max()), "最大值一致")
    # ROI 字符串格式（曾经把 y0 当成 x1 打印成 "x100..100"）
    r2 = batch.analyze_frame(data, "RGGB", 10, (10, 20, 110, 220))
    check(r2["roi"] == "x10..110 y20..220", f"ROI 文本正确: {r2['roi']}")
    # 带坏点统计
    r3 = batch.analyze_frame(data, "RGGB", 10, None, defect_params={"threshold": 30})
    check(r3["defects"] != "", "统计了坏点数")
    check("hot" in r3["defect_types"] or "cluster" in r3["defect_types"],
          f"坏点类型统计: {r3['defect_types']}")


def test_batch_csv_roundtrip():
    import generate_sample as gs
    rows = []
    for i in range(3):
        data, _ = gs.generate("flat", width=64, height=48, bit_depth=10, seed=i)
        r = batch.analyze_frame(data, "Mono/None", 10)
        r.update({"file": f"f{i}.raw", "size_bytes": data.nbytes,
                  "width": 64, "height": 48, "bit_depth": 10, "pattern": "Mono/None"})
        rows.append(r)
    path = os.path.join(TMP, "summary.csv")
    batch.write_batch_csv(path, rows)
    text = open(path, encoding="utf-8-sig").read().splitlines()
    check(text[0].split(",")[0] == "file", "表头首列是 file")
    check(len(text) == 4, f"3 行数据 + 表头（实际 {len(text)}）")
    check(all(len(line.split(",")) >= len(batch.BATCH_HEADER) - 1 for line in text[1:]),
          "每行列数合理")
    check("f0.raw" in text[1], "文件名写入")


def test_generator_supports_all_patterns():
    """生成器在 Mono 与四种 Bayer 下都要能出素材（曾经 Mono 会 KeyError）。"""
    import generate_sample as gs
    for scene in ("dark", "flat", "shading", "saturated", "bayer"):
        for pattern in ("Mono/None", "RGGB", "BGGR", "GRBG", "GBRG"):
            data, meta = gs.generate(scene, width=64, height=48, bit_depth=10,
                                     pattern=pattern, seed=1)
            check(data.shape == (48, 64) and data.dtype == np.uint16,
                  f"{scene}/{pattern} 生成 {data.shape} {data.dtype}")
            check(int(data.max()) <= 1023, f"{scene}/{pattern} 不超位深上限")
    # 落盘/读回
    from utils.raw_io import RawLoadSpec, load_raw
    data, meta = gs.generate("shading", width=64, height=48, bit_depth=12,
                             pattern="Mono/None", seed=2)
    path = os.path.join(TMP, "gen_mono.raw")
    info = gs.write_sample(path, data, meta, "unpacked")
    check(np.array_equal(load_raw(path, info["spec_object"]), data), "Mono 素材落盘往返")


if __name__ == "__main__":
    sys.exit(run(globals(), "批量分析"))
