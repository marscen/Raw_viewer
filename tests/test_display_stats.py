"""CFA / 显示管线 / 测量统计 测试。"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests._util import check, check_close, run  # noqa: E402

from utils import cfa  # noqa: E402
from utils import display as D  # noqa: E402
from utils import stats as S  # noqa: E402


def _rgbb_flat(shape=(32, 32), pattern="RGGB", values=None):
    """构造每个相位常数的 Bayer 图。"""
    values = values or {"R": 800, "Gr": 400, "Gb": 400, "B": 200}
    out = np.zeros(shape, np.uint16)
    names = cfa.PHASE_LABELS[cfa.normalize_pattern(pattern)]
    for phase, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        out[dy::2, dx::2] = values[names[phase]]
    return out


def test_phase_maps_and_split():
    for pattern, expect in {
        "RGGB": ("R", "Gr", "Gb", "B"),
        "BGGR": ("B", "Gb", "Gr", "R"),
        "GRBG": ("Gr", "R", "B", "Gb"),
        "GBRG": ("Gb", "B", "R", "Gr"),
    }.items():
        check(cfa.PHASE_LABELS[pattern] == expect, f"{pattern} 相位命名")
        raw = _rgbb_flat(pattern=pattern)
        planes = cfa.split_planes(raw, pattern)
        check(all(planes[n].size == 16 * 16 for n in expect), f"{pattern} 拆平面尺寸")
        check(all(planes[n].min() == planes[n].max() for n in expect),
              f"{pattern} 拆平面后每相位常数")
        check(cfa.plane_name_at(1, 1, pattern) == expect[3], f"{pattern} (1,1) 相位名")
    check(cfa.normalize_pattern("none") == "Mono/None", "pattern 归一化")
    check(cfa.normalize_pattern("gbrg") == "GBRG", "pattern 大小写")


def test_phase_planes_odd_origin():
    """奇数起点 ROI 不能串通道（这是 ROI 功能的正确性前提）。"""
    raw = _rgbb_flat(pattern="RGGB")
    for origin in ((0, 0), (1, 1), (0, 1), (1, 0)):
        x0, y0 = origin
        crop = raw[y0:y0 + 16, x0:x0 + 16]
        for view in cfa.phase_planes(crop, "RGGB", origin):
            plane = view["plane"]
            if plane.size == 0:
                continue
            check(plane.min() == plane.max(),
                  f"origin={origin} {view['name']} 平面应保持常数（未串通道）")
            # 全局坐标映射回来必须落在同一相位
            py, px = 0, 0
            gx = view["base_x"] + view["step"] * px
            gy = view["base_y"] + view["step"] * py
            check(cfa.plane_name_at(gx, gy, "RGGB") == view["name"],
                  f"origin={origin} {view['name']} 坐标映射回相位一致 "
                  f"({gx},{gy})")


def test_demosaic_flat_field_uniform():
    """纯色 Bayer 场 demosaic 后必须保持均匀（回归历史上的 shift bug）。"""
    for pattern in cfa.PATTERNS:
        raw = _rgbb_flat((32, 32), pattern).astype(np.float32)
        rgb = D.demosaic(raw, pattern)
        inner = rgb[8:24, 8:24]
        check_close(inner[..., 0].mean(), 800, 0.5, f"{pattern} demosaic R 均值")
        check_close(inner[..., 1].mean(), 400, 0.5, f"{pattern} demosaic G 均值")
        check_close(inner[..., 2].mean(), 200, 0.5, f"{pattern} demosaic B 均值")
        check(max(inner[..., i].std() for i in range(3)) < 0.5,
              f"{pattern} demosaic 均匀性（无条纹/无通道丢失）")


def test_render_views_and_planes():
    raw = _rgbb_flat((32, 32), "RGGB")
    base = dict(bit_depth=10, pattern="RGGB", stretch="Fixed (black/white)",
                white_level=1023)
    for view in D.VIEW_MODES:
        img, info = D.render_display(raw, D.DisplayParams(view=view, **base))
        check(img is not None and img.dtype == np.uint8, f"视图 {view} 输出 uint8")
        if view in ("Bayer Mosaic", "Bayer Demosaic"):
            check(img.ndim == 3 and img.shape[2] == 3, f"视图 {view} 为彩色")
        else:
            check(img.ndim == 2, f"视图 {view} 为灰度")
    # 单通道视图：非本相位像素必须为 0，本相位像素保持原值映射
    img, _ = D.render_display(raw, D.DisplayParams(view="R plane", **base))
    check(img[0, 0] == round(800 / 1023 * 255), "R plane 的 R 像素正确")
    check(img[0, 1] == 0 and img[1, 0] == 0 and img[1, 1] == 0, "R plane 其余相位为黑")
    img_g, _ = D.render_display(raw, D.DisplayParams(view="G plane", **base))
    check(img_g[0, 1] > 0 and img_g[1, 0] > 0 and img_g[0, 0] == 0,
          "G plane 只显示两个绿相位")


def test_stretch_modes_and_degenerate_levels():
    raw = _rgbb_flat((64, 64), "RGGB")
    p = D.DisplayParams(bit_depth=10, view="Mono", stretch="Min-Max")
    img, info = D.render_display(raw, p)
    check_close(info.lo, 200, 1e-6, "Min-Max 低电平")
    check_close(info.hi, 800, 1e-6, "Min-Max 高电平")
    check(img.min() == 0 and img.max() == 255, "Min-Max 拉伸铺满 0~255")

    p = D.DisplayParams(bit_depth=10, view="Mono", stretch="Sigma", sigma_k=2.0)
    img, info = D.render_display(raw, p)
    mu, sd = float(raw.mean()), float(raw.std())
    check_close(info.lo, mu - 2 * sd, 1e-3, "Sigma 低电平")
    check_close(info.hi, mu + 2 * sd, 1e-3, "Sigma 高电平")

    # 暗场里只有极少数亮坏点：百分位区间会退化，必须回退到 ±3σ 才看得见
    dark = np.full((256, 256), 64, np.uint16)
    dark.flat[:20] = 1023
    img, info = D.render_display(dark, D.DisplayParams(bit_depth=10, view="Mono",
                                                       stretch="Percentile"))
    check(info.hi - info.lo > 10, f"退化区间已展开（{info.lo:.1f}~{info.hi:.1f}）")
    check(60 < img[128, 128] < 200, f"暗场底色可见（映射到 {img[128, 128]}）")
    check(img[0, 0] == 255, "坏点仍饱和为白")

    # 完全均匀的画面也要给中间灰
    uni = np.full((64, 64), 64, np.uint16)
    img, info = D.render_display(uni, D.DisplayParams(bit_depth=10, view="Mono",
                                                      stretch="Percentile"))
    check(img[10, 10] > 10, f"纯均匀场不是全黑（{img[10, 10]}）")

    # gamma / invert 单调性
    p = D.DisplayParams(bit_depth=10, view="Mono", stretch="Min-Max", gamma=2.0)
    img_g, _ = D.render_display(raw, p)
    p2 = D.DisplayParams(bit_depth=10, view="Mono", stretch="Min-Max", invert=True)
    img_i, _ = D.render_display(raw, p2)
    check(img_g[0, 0] > 0, "gamma>1 提亮暗部")
    check(img_i[0, 0] < 255, "反相后亮部变暗")


def test_colormaps():
    gray = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
    for name in D.COLORMAPS:
        rgb = D.apply_colormap(gray, name)
        check(rgb.shape == (16, 16, 3) and rgb.dtype == np.uint8, f"{name} 伪彩形状")
    lut = D.colormap_lut("Gray")
    check(np.array_equal(lut[:, 0], lut[:, 1]) and np.array_equal(lut[:, 0], lut[:, 2]),
          "Gray 伪彩三通道一致")


def test_stats_exact_values():
    raw = _rgbb_flat((64, 64), "RGGB")
    st = S.roi_stats(raw, "RGGB", None, 10)
    check(st["overall"]["count"] == 4096, "总体像素数")
    check_close(st["overall"]["mean"], (800 + 400 + 400 + 200) / 4, 1e-6, "总体均值")
    chan = {c["name"]: c for c in st["channels"]}
    check_close(chan["R"]["mean"], 800, 1e-9, "R 均值")
    check_close(chan["B"]["mean"], 200, 1e-9, "B 均值")
    check(chan["R"]["std"] == 0, "常数平面 std=0")
    check(chan["R"]["sat"] == 0 and chan["R"]["zero"] == 0, "无饱和/黑点")

    # 奇数起点 ROI 不串通道
    st2 = S.roi_stats(raw, "RGGB", (1, 3, 33, 35), 10)
    chan2 = {c["name"]: c["mean"] for c in st2["channels"]}
    check_close(chan2["R"], 800, 1e-9, "奇数 ROI：R 仍是 800")
    check_close(chan2["B"], 200, 1e-9, "奇数 ROI：B 仍是 200")
    check(st2["size"] == (32, 32), "ROI 尺寸")

    # 饱和与 0DN 统计
    sat = _rgbb_flat((8, 8), "RGGB", {"R": 1023, "Gr": 0, "Gb": 0, "B": 1023})
    rep = S.saturation_stats(sat, "RGGB", 10)
    check(rep["channels"]["R"]["sat"] == 16 and rep["channels"]["Gr"]["zero"] == 16,
          "饱和/黑点计数")


def test_histogram_and_profile():
    raw = np.full((64, 64), 100, np.uint16)
    raw[::2, :] = 200                      # 偶数行 200，奇数行 100
    counts, edges = S.histogram(raw, "Mono/None", None, 10, bins=64)
    check(counts.sum() == raw.size, "直方图总数等于像素数")
    check(counts.size == 64 and edges.size == 65, "直方图 bin/edges 尺寸")
    chans, _ = S.histogram(raw, "Mono/None", None, 10, bins=64, per_channel=True)
    check("Mono" in chans, "Mono 分通道直方图")

    prof = S.profile(raw, "Mono/None", None, "rows")
    check(prof["index"].size == 64 and prof["mean"].size == 64, "行 profile 长度")
    check_close(prof["mean"][0], 200, 1e-6, "第 0 行均值")
    check_close(prof["mean"][1], 100, 1e-6, "第 1 行均值")
    prof_c = S.profile(raw, "Mono/None", None, "cols")
    check(prof_c["mean"].size == 64, "列 profile 长度")
    check_close(prof_c["mean"][0], 150, 1e-6, "列均值（200/100 各半）")

    # CFA 行 profile：每行只包含两个相位
    raw2 = _rgbb_flat((32, 32), "RGGB")
    prof2 = S.profile(raw2, "RGGB", None, "rows")
    r_mean = prof2["channels"]["R"][0]
    check(r_mean.shape == (32,), "分通道行 profile 形状")
    # 不含 R 的行应为 NaN（明确表示"该行没有这个相位"），含 R 的行恒为 800
    check(np.all(np.isnan(r_mean[1::2])), "无 R 的行给出 NaN")
    check(np.allclose(r_mean[0::2], 800), f"R 行 profile 恒 800（{r_mean[:4]}）")
    odd = S.profile(raw2, "RGGB", (1, 1, 31, 31), "rows")
    b_mean = odd["channels"]["B"][0]
    finite = b_mean[np.isfinite(b_mean)]
    check(finite.size > 0 and np.allclose(finite, 200),
          f"奇数起点 ROI 的 B 行 profile 仍为 200（{finite[:4]}）")


def test_diff_stats():
    a = np.full((32, 32), 100, np.uint16)
    b = a.copy()
    res = S.diff_stats(a, b, 10, tol=1)
    check(res["psnr"] == float("inf") and res["max_abs"] == 0, "相同帧 PSNR=inf")
    b[5, 5] = 140
    res = S.diff_stats(a, b, 10, tol=1)
    check_close(res["max_abs"], 40, 1e-9, "最大绝对差")
    check(res["count_diff"] == 1 and res["pct_diff"] > 0, "差异像素计数")
    check_close(res["rmse"], np.sqrt(40.0 ** 2 / a.size), 1e-9, "RMSE")
    check_close(res["psnr"], 10 * np.log10(1023.0 ** 2 / (1600.0 / a.size)), 1e-6, "PSNR")
    check(res["diff_map"].shape == a.shape, "差异图")
    mismatch = S.diff_stats(a, b[:16], 10)
    check("error" in mismatch, "尺寸不一致给出错误信息")


def test_neighborhood_inspector():
    raw = _rgbb_flat((16, 16), "RGGB")
    nb = S.neighborhood(raw, 5, 5, 2, "RGGB")
    check(nb["values"].shape == (5, 5), "邻域窗口尺寸")
    check(nb["center"] == (5, 5) and nb["center_value"] == int(raw[5, 5]), "中心点信息")
    check(nb["center_channel"] == cfa.plane_name_at(5, 5, "RGGB"), "中心点通道")
    check(nb["channels"][0, 0] == cfa.plane_name_at(3, 3, "RGGB"), "左上角通道名")
    check(sum(nb["channel_values"].values()) == int(nb["values"].sum()),
          "邻域通道和一致")


if __name__ == "__main__":
    sys.exit(run(globals(), "显示管线与统计"))
