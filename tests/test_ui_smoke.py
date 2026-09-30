"""无头 UI 集成测试：真实 QWidget 路径（offscreen 平台）走一遍主要工作流。

覆盖：加载 → 渲染 → ROI → 统计/直方图/profile → 像素检查器 → 视图切换
     → 自动电平 → 五个算法 → 校正与撤销 → 对比模式 → 导出 → 帧/文件切换
     → 参数对话框推断 → 视图状态在重渲染时保持。
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests._util import check, check_close, run  # noqa: E402

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

TMP = tempfile.mkdtemp(prefix="rawv2_ui_")
# 隔离设置存储：Qt 在 macOS 上忽略 setDefaultFormat（(org, app) 构造出来永远是
# plist），所以走应用自己的便携模式开关，把设置写进临时目录。
os.environ["RAWV2_SETTINGS_DIR"] = TMP

APP = QApplication.instance() or QApplication([])


DIALOG_RESULTS = {"open": "", "open_many": [], "save": "",
                  "question": "No"}      # question: "Yes" / "No"


def _patch_modals():
    """测试里绝不能弹模态框（会永久阻塞），替换成读 DIALOG_RESULTS 的桩。"""
    from PyQt6.QtWidgets import QMessageBox
    import ui.main_window as mw
    calls = []

    def make(kind):
        def _fn(parent, title, text, *a, **kw):
            calls.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return staticmethod(_fn)

    for kind in ("information", "warning", "critical"):
        setattr(mw.QMessageBox, kind, make(kind))

    def _question(parent, title, text, *a, **kw):
        calls.append(("question", title, text))
        return (QMessageBox.StandardButton.Yes
                if DIALOG_RESULTS.get("question") == "Yes"
                else QMessageBox.StandardButton.No)

    mw.QMessageBox.question = staticmethod(_question)
    # 文件选择 / 保存对话框
    mw.QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (DIALOG_RESULTS.get("open", ""), ""))
    mw.QFileDialog.getOpenFileNames = staticmethod(
        lambda *a, **k: (list(DIALOG_RESULTS.get("open_many", [])), ""))
    mw.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (DIALOG_RESULTS.get("save", ""), ""))
    # 参数对话框：直接返回"取消"，让调用方走无对话框分支
    import ui.dialogs as dlg
    dlg.ImageParamsDialog.exec = lambda self: 0
    mw.ImageParamsDialog.exec = lambda self: 0
    return calls


MODAL_CALLS = _patch_modals()

from algorithms.manager import AlgorithmManager  # noqa: E402
from ui.dialogs import ImageParamsDialog  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.sidebar import AlgorithmPanel, DisplayControlPanel  # noqa: E402
from ui.stats_panel import StatsPanel  # noqa: E402
from utils.raw_io import load_raw  # noqa: E402


def _make_sample(name="ui_dark.raw", width=320, height=240, packing="unpacked", **opt):
    import generate_sample as gs
    data, meta = gs.generate("dark", width=width, height=height, bit_depth=10,
                             pattern="RGGB", seed=5, dark_level=64)
    path = os.path.join(TMP, name)
    info = gs.write_sample(path, data, meta, packing, **opt)
    return path, data, meta, info


_WINDOWS = []          # 保住测试用窗口的引用：被 GC 提前销毁会引发悬空 paintEvent


def _window(path, params=None):
    w = MainWindow()
    _WINDOWS.append(w)
    w.resize(1200, 800)
    w.show()                      # 可见性/同步逻辑需要窗口真正 show
    p = dict(params or {})
    p.setdefault("width", 320)
    p.setdefault("height", 240)
    p.setdefault("bit_depth", 10)
    p.setdefault("packing", "unpacked")
    p.setdefault("pattern", "RGGB")
    p.setdefault("view", "Bayer Mosaic")
    w.load_image(path, p)
    return w


def test_load_and_render():
    path, data, meta, _ = _make_sample()
    w = _window(path)
    check(w.raw is not None and np.array_equal(w.raw, data), "加载的 DN 与生成数据一致")
    check(w.display is not None and w.display.ndim == 3, "Bayer Mosaic 渲染为彩色")
    check(w.canvas._src_np is not None and w.canvas.raw_data is w.raw, "画布持有显示数组与 DN")
    check("240x320" in w.status_label.text() or "320x240" in w.status_label.text(),
          f"状态栏显示尺寸: {w.status_label.text()[:60]}")
    w._do_measure()
    check(w.stats_panel.stats_table.rowCount() == 4, "统计表 4 个相位行")
    check(w.stats_panel.hist_plot.counts is not None, "直方图有数据")


def test_render_preserves_view_state():
    """只改显示参数时不能重置缩放/ROI（否则拖滑块画面会跳）。"""
    path, _, _, _ = _make_sample("ui_view.raw")
    w = _window(path)
    w.canvas.zoom_to(8.0)
    w.canvas.set_roi((20, 20, 200, 150))
    scale_before = w.canvas.scale
    roi_before = w.canvas.roi
    w.on_render_requested(dict(w.params, gamma=1.8, colormap="Turbo",
                               view="Mono", stretch="Fixed (black/white)",
                               black_level=50, white_level=200))
    check(abs(w.canvas.scale - scale_before) < 1e-6,
          f"重渲染后缩放保持（{scale_before} -> {w.canvas.scale}）")
    check(w.canvas.roi == roi_before, "重渲染后 ROI 保持")
    check(w.display.ndim == 3, "Turbo 伪彩输出 RGB")
    # 重新加载文件时才应该重置视图
    w.load_image(path, w.params)
    check(w.canvas.roi is None, "重新打开文件时 ROI 被清空")


def test_roi_stats_and_inspector():
    path, _, _, _ = _make_sample("ui_roi.raw")
    w = _window(path)
    w.canvas.set_roi((10, 11, 90, 71))
    check(w.roi == (10, 11, 90, 71), "ROI 记录")
    check("80x60" in w.roi_label.text(), f"ROI 状态栏: {w.roi_label.text()}")
    w._do_measure()
    first_line = w.stats_panel.stats_info.text().splitlines()[0]
    check("局部" in first_line and "80x60" in first_line, f"统计按 ROI 计算: {first_line}")
    w._on_pixel_clicked(11, 11)
    check("中心 (11,11)" in w.stats_panel.inspector_info.text(), "检查器定位到点选像素")
    check(w.stats_panel.inspector_table.rowCount() == 5, "检查器 5x5（半径 2）")
    w.stats_panel.inspector_radius.setValue(4)
    check(w.stats_panel.inspector_table.rowCount() == 9, "调整半径后 9x9")


def test_view_switching_and_auto_levels():
    path, _, _, _ = _make_sample("ui_views.raw")
    w = _window(path)
    for view in ("Mono", "R plane", "Gr plane", "Gb plane", "G plane", "B plane",
                 "Bayer Demosaic", "Bayer Mosaic"):
        w.on_render_requested(dict(w.params, view=view))
        check(w.display is not None, f"视图 {view} 渲染成功")
    check(w.display.ndim == 3, "马赛克为彩色")
    w.on_auto_levels()
    check(w.params["stretch"] == "Fixed (black/white)", "自动电平切换为固定电平")
    lo, hi = w.params["black_level"], w.params["white_level"]
    check(hi > lo, f"自动电平区间有效（{lo}~{hi}）")
    w.on_render_requested(dict(w.params, view="Mono"))
    check(int(w.display[120, 160]) > 0, f"暗场底色不是纯黑（{w.display[120, 160]}）")
    check(int(w.display.max()) == 255, "亮缺陷仍饱和为白（找坏点靠的就是这个对比）")

    # 干净暗场（只有噪声、没有坏点/坏行）：自动电平后底色应接近中间灰
    import generate_sample as gs
    clean_dir = os.path.join(TMP, "clean.raw")
    rng = np.random.default_rng(3)
    clean = np.clip(64 + rng.normal(0, 3.0, (240, 320)), 0, 1023).astype(np.uint16)
    clean.tofile(clean_dir)
    w2 = _window(clean_dir, {"width": 320, "height": 240, "pattern": "Mono/None",
                             "view": "Mono"})
    w2.on_auto_levels()
    w2.on_render_requested(dict(w2.params, view="Mono"))
    check(int(w2.display[120, 160]) > 60,
          f"干净暗场底色可见（{w2.display[120, 160]}，level "
          f"{w2.params['black_level']}~{w2.params['white_level']}）")


def test_algorithms_and_correction_flow():
    path, _, meta, _ = _make_sample("ui_algo.raw")
    w = _window(path)
    names = w.algorithm_manager.get_algorithm_names()
    check(len(names) == 5, f"注册了 5 个算法: {names}")
    for name in names:
        algo = w.algorithm_manager.get_algorithm(name)
        params = {k: v["default"] for k, v in algo.get_parameters().items()}
        w.run_algorithm(name, params)
        check(w.last_defects is not None, f"{name} 运行完成")
    hot = sum(1 for d in meta["injected"] if d["type"] in ("hot", "cluster"))
    bp = w.algorithm_manager.get_algorithm("Bad Pixel Detection")
    w.run_algorithm("Bad Pixel Detection", {k: v["default"] for k, v in bp.get_parameters().items()})
    found = {(d["x"], d["y"]) for d in w.last_defects}
    expected = {(d["x"], d["y"]) for d in meta["injected"] if d["type"] in ("hot", "cluster")}
    check(expected <= found, f"UI 流程检出注入坏点（缺 {sorted(expected - found)[:5]}）")
    check(w.stats_panel.defect_table.rowCount() == len(w.last_defects), "缺陷表行数一致")
    check(w.canvas.overlays, "叠加层已交给画布")

    # 校正 + 撤销
    params = {k: v["default"] for k, v in bp.get_parameters().items()}
    params["visualize_only"] = False
    before = w.raw.copy()
    w.run_algorithm("Bad Pixel Detection", params)
    check(len(w.undo_stack) == 1, "校正进入撤销栈")
    changed = int((w.raw != before).sum())
    check(changed >= hot, f"校正改动了 {changed} 个像素（>= 注入 {hot}）")
    check(w.ref_raw is not None and np.array_equal(w.ref_raw, before),
          "校正后自动把原图设为参考帧")
    w.undo_correction()
    check(np.array_equal(w.raw, before), "撤销后回到原图")
    check(len(w.undo_stack) == 0 and not w.last_defects, "撤销后清空缺陷与栈")

    # ROI 内运行
    w.canvas.set_roi((0, 0, 100, 100))
    params = {k: v["default"] for k, v in bp.get_parameters().items()}
    params["_roi_only"] = True
    w.run_algorithm("Bad Pixel Detection", params)
    check(all(0 <= d["x"] < 100 and 0 <= d["y"] < 100 for d in w.last_defects),
          "ROI 内运行时缺陷都在 ROI 内")


def test_compare_modes():
    path, _, _, _ = _make_sample("ui_cmp.raw")
    w = _window(path)
    w.open_reference_file_path = None
    # 直接注入参考帧（等价于加载同尺寸参考图）
    w.ref_raw = w.raw.copy()
    w.ref_raw[30, 40] = 1023
    w._update_reference_availability()
    w.toggle_compare_mode(True)
    check(w.ref_canvas.isVisibleTo(w), "对比模式打开参考画布")
    for mode in ("并排 Side-by-side", "差分 Difference (|A-B|)", "混合 Blend 50%"):
        w.set_compare_mode(mode)
        check(w.ref_canvas._src_np is not None, f"{mode} 参考画布有内容")
    w.set_compare_mode("差分 Difference (|A-B|)")
    check("PSNR" in w.status_label.text(), f"差分模式报告 PSNR: {w.status_label.text()[:60]}")
    check(w.ref_canvas._src_np.ndim == 3, "差分视图用 Hot 伪彩（RGB）")
    w.toggle_compare_mode(False)
    check(not w.ref_canvas.isVisibleTo(w), "关闭对比模式")


def test_exports_and_report():
    path, _, _, _ = _make_sample("ui_exp.raw")
    w = _window(path)
    bp = w.algorithm_manager.get_algorithm("Bad Pixel Detection")
    w.run_algorithm("Bad Pixel Detection", {k: v["default"] for k, v in bp.get_parameters().items()})
    w._do_measure()
    from utils import export as X
    png = os.path.join(TMP, "disp.png")
    X.write_png(png, w.display if w.display.ndim == 2 else w.display)
    check(os.path.getsize(png) > 100, "导出显示图")
    dn = os.path.join(TMP, "dn16.tif")
    X.write_image(dn, w.raw)
    check(os.path.getsize(dn) > 100, "导出原始 DN")
    header, rows = X.stats_to_rows(w._last_stats)
    X.write_csv(os.path.join(TMP, "s.csv"), header, rows)
    header, rows = X.defects_to_rows(w.last_defects)
    csvp = os.path.join(TMP, "d.csv")
    X.write_csv(csvp, header, rows)
    check(len(open(csvp, encoding="utf-8-sig").read().splitlines()) == len(w.last_defects) + 1,
          "缺陷 CSV 行数与缺陷数一致")
    annotated = X.draw_defects(w.display, w.last_defects, radius=5)
    X.write_png(os.path.join(TMP, "ann.png"), annotated)
    check(os.path.getsize(os.path.join(TMP, "ann.png")) > 100, "导出标注截图")


def test_frame_and_file_navigation():
    import generate_sample as gs
    data, meta = gs.generate("dark", width=160, height=120, bit_depth=10, frames=3, seed=9)
    path = os.path.join(TMP, "multi3.raw")
    gs.write_sample(path, data, meta, "unpacked")
    w = _window(path, {"width": 160, "height": 120})
    check(np.array_equal(w.raw, data[0]), "多帧文件默认读第 0 帧")
    w.navigate_frame(1)
    check(np.array_equal(w.raw, data[1]), "切到第 1 帧")
    w.navigate_frame(1)
    check(np.array_equal(w.raw, data[2]), "切到第 2 帧")
    w.navigate_frame(1)
    check("最后一帧" in w.status_label.text(), "越界提示")
    # 同目录另一个文件：故意做成同样大小（多帧），这样切换时不弹参数对话框
    other = os.path.join(TMP, "multi3_b.raw")
    data_b, meta_b = gs.generate("dark", width=160, height=120, bit_depth=10,
                                 frames=3, seed=77)
    gs.write_sample(other, data_b, meta_b, "unpacked")
    check(os.path.getsize(other) == os.path.getsize(path), "两个文件大小一致（避免弹窗）")
    w.navigate_file(1)
    check(w.current_file_path.endswith("multi3_b.raw"),
          f"切到同目录下一个文件: {os.path.basename(w.current_file_path)}")
    # 切文件时保留帧号（多帧序列逐帧对比的常用行为）
    check(w.params["frame_index"] == 2, "切文件保留当前帧号")
    check(np.array_equal(w.raw, data_b[2]), "读到的仍是同一帧号的数据")
    w.params["frame_index"] = 0
    w.load_image(other, w.params)
    check(np.array_equal(w.raw, data_b[0]), "帧号归零后读到第 0 帧")
    w.navigate_file(-1)
    check(w.current_file_path.endswith("multi3.raw"), "再切回上一个文件")
    check(not [c for c in MODAL_CALLS if c[0] == "critical"],
          f"流程中没有报错弹窗: {MODAL_CALLS[:2]}")


def test_params_dialog_detection():
    path, data, meta, info = _make_sample("ui_dlg.raw", width=256, height=192)
    dlg = ImageParamsDialog(None, path=path)
    check(dlg.candidate_list.count() >= 1, "给出布局候选")
    check("文件" in dlg.check_label.text(), f"初始校验信息: {dlg.check_label.text()[:80]}")
    check("少" in dlg.check_label.text(),
          "默认 1920x1080 参数与小文件不匹配时明确提示（旧的静默截断问题）")
    found = False
    for i in range(dlg.candidate_list.count()):
        item = dlg.candidate_list.item(i)
        cand = item.data(0x0100)  # Qt.ItemDataRole.UserRole
        if cand and cand["spec"].width == 256 and cand["spec"].bit_depth == 10:
            dlg.apply_candidate(item)
            found = True
            break
    check(found, "候选里能找到 256x192 10bit 并套用")
    p = dlg.get_params()
    check(p["width"] == 256 and p["height"] == 192 and p["bit_depth"] == 10, "套用后参数正确")
    check("完全匹配" in dlg.check_label.text(), f"套用候选后完全匹配: {dlg.check_label.text()[:80]}")
    check(p["pattern"] == "Mono/None", "默认 pattern 是 Mono（灰度数据不被染成彩色）")


def test_side_panels_signals():
    panel = DisplayControlPanel()
    seen = {}
    panel.reload_requested.connect(lambda d: seen.setdefault("reload", d))
    panel.render_requested.connect(lambda d: seen.setdefault("render", d))
    panel.set_params({"width": 640, "height": 480, "bit_depth": 12, "pattern": "BGGR"})
    check(panel.get_params()["width"] == 640, "面板写入参数")
    panel.pattern_combo.setCurrentText("RGGB")
    check(seen.get("render", {}).get("pattern") == "RGGB", "pattern 改变触发重渲染信号")
    panel.width_spin.setValue(800)
    check(seen.get("reload", {}).get("width") == 800, "几何改变触发重新读盘信号")

    ap = AlgorithmPanel(AlgorithmManager())
    check(ap.algo_combo.count() == 5, "算法下拉 5 项")
    ap.algo_combo.setCurrentText("Bad Line Detection")
    check("坏行" in ap.desc_label.text() or "Bad" in ap.desc_label.text(), "算法说明")
    params = ap.collect_params()
    params["_roi_only"] = ap.roi_only_check.isChecked()
    check("threshold" in params, "参数收集")
    sp = StatsPanel()
    sp.set_defects([{"type": "hot", "x": 1, "y": 2, "channel": "R", "value": 1023,
                     "delta": 900.0, "note": "single"}])
    check(sp.defect_table.rowCount() == 1 and "hot" in sp.defect_info.text(), "缺陷面板")


def test_coordinate_jump_box_parsing():
    from ui.goto_box import CoordinateJumpBox
    box = CoordinateJumpBox()
    for text, expect in (("100,200", (100, 200)), ("100 200", (100, 200)),
                         ("(100, 200)", (100, 200)), ("x=100 y=200", (100, 200)),
                         ("  100 ; 200 ", (100, 200)), ("100", None),
                         ("", None), ("abc", None),
                         ("-5,10", None), ("12.5,7", None), ("1e3,2", None),
                         ("-1,-1", None)):
        box.edit.setText(text)
        check(box.parse() == expect, f"解析 {text!r} -> {box.parse()}（期望 {expect}）")
    box.edit.setText("bad")
    check(not box.go_btn.isEnabled(), "非法输入禁用跳转按钮")
    box.edit.setText("5,6")
    check(box.go_btn.isEnabled(), "合法输入启用跳转按钮")
    # 1-based 换算（Matlab/Excel 习惯）
    box.one_based.setChecked(True)
    box.edit.setText("101,201")
    check(box.parse() == (100, 200), "1-based 输入转内部 0-based")
    box.set_coordinate(100, 200)
    check(box.edit.text() == "101,201", "1-based 回填坐标 +1")
    box.one_based.setChecked(False)
    box.set_coordinate(0, 0)
    check(box.edit.text() == "0,0", "0-based 回填坐标")
    # 缩放选项
    box.zoom_combo.setCurrentIndex(0)
    check(box.selected_zoom() is None, "默认“保持当前缩放”")
    box.zoom_combo.setCurrentIndex(3)
    check(box.selected_zoom() == 20.0, "20× 选项")


def test_goto_coordinate():
    path, _, _, _ = _make_sample("ui_goto.raw", width=400, height=300)
    w = _window(path, {"width": 400, "height": 300})
    check(w.raw.shape == (300, 400), f"按 400x300 解释文件（{w.raw.shape}）")
    check(w.goto_box.edit.text() == "", "打开新文件后坐标框清空")

    w.goto_coordinate(150, 120, 20.0)
    c = w.canvas
    check(c.selected_pixel == (150, 120), "跳转后选中该像素")
    check(abs(c.scale - 20.0) < 1e-6, f"跳转缩放生效（{c.scale}）")
    p = c.image_to_widget(150.5, 120.5)
    check(abs(p.x() - c.width() / 2) <= 1.5 and abs(p.y() - c.height() / 2) <= 1.5,
          f"像素中心落在视口中心（偏差 {p.x() - c.width() / 2:.1f},"
          f"{p.y() - c.height() / 2:.1f}）")
    check("跳转到 (150,120)" in w.status_label.text(), f"状态栏反馈: {w.status_label.text()}")
    check("中心 (150,120)" in w.stats_panel.inspector_info.text(), "像素检查器跟随跳转")
    check(w.stats_panel.tabs.currentIndex() == 2, "自动切到像素检查器页签")
    check(w.goto_box.edit.text() == "150,120", "坐标框回填")

    # 越界：夹回画内并提示
    w.goto_coordinate(99999, 99999)
    check(c.selected_pixel == (399, 299), f"越界夹回画内（{c.selected_pixel}）")
    check("越界" in w.status_label.text(), "越界有明确提示")

    # scale=None：缩放已经够大时保持不变
    c.zoom_to(12.0)
    w.goto_coordinate(20, 30, None)
    check(abs(c.scale - 12.0) < 1e-6, f"保持当前缩放（{c.scale}）")

    # 画布点选回填 -> 再从输入框跳转（模拟真实操作）
    w._on_pixel_clicked(7, 9)
    check(w.goto_box.edit.text() == "7,9", "画布点选回填坐标框")
    w.goto_box.zoom_combo.setCurrentIndex(5)      # 100×
    w.goto_box.edit.setText("88,66")
    w.goto_box.edit.returnPressed.emit()
    check(c.selected_pixel == (88, 66), f"输入框回车跳转（{c.selected_pixel}）")
    check(abs(c.scale - 100.0) < 1e-6, "输入框缩放选项生效")

    # 1-based 端到端
    w.goto_box.one_based.setChecked(True)
    w.goto_box.edit.setText("201,121")
    w.goto_box.edit.returnPressed.emit()
    check(c.selected_pixel == (200, 120), f"1-based 跳转（{c.selected_pixel}）")
    w.goto_box.one_based.setChecked(False)

    # 缺陷清单定位也会同步坐标框
    w.stats_panel.set_defects([{"type": "hot", "x": 33, "y": 44, "channel": "R",
                                "value": 1023, "delta": 900.0, "note": "single"}])
    w.locate_defect(33, 44)
    check(c.selected_pixel == (33, 44) and w.goto_box.edit.text() == "33,44",
          "缺陷定位同步坐标框")

    # 快捷键与菜单项存在
    jumps = [a for a in w.menuBar().actions()[3].menu().actions()
             if "Coordinate" in a.text()]
    check(len(jumps) == 1 and jumps[0].shortcut().toString() == "Ctrl+G",
          "Analyze 菜单里有 Ctrl+G 坐标跳转")
    w.focus_goto()
    check("输入坐标" in w.status_label.text(), "focus_goto 给出输入提示")


def test_black_level_estimate_action():
    path, data, _, _ = _make_sample("ui_black.raw", width=320, height=240)
    w = _window(path)
    w.params["black_level"] = 0
    w.estimate_black_level()
    check_close(w.params["black_level"], 64, 3.0, "黑电平估计填入参数")
    check("黑电平估计" in w.status_label.text(), f"状态栏提示: {w.status_label.text()}")
    w.params["black_level"] = 0
    w.canvas.set_roi((40, 40, 200, 200))
    w.estimate_black_level()
    check_close(w.params["black_level"], 64, 3.0, "带 ROI 的估计")
    check("ROI" in w.status_label.text(), "提示使用 ROI 中值")
    # 无 ROI 且画面不是暗场 → 不覆盖已有黑电平，并提示原因
    w.canvas.set_roi(None)
    bright = np.full((64, 64), 900, np.uint16)
    w.raw = bright
    w.params["black_level"] = 7
    w.estimate_black_level()
    check(w.params["black_level"] == 7, "非暗场时不覆盖已有黑电平")
    # 显式框了亮 ROI：按 ROI 算（用户指定了黑参考区），但要给出确认提示
    w.canvas.set_roi((10, 10, 50, 50))
    w.estimate_black_level()
    check_close(w.params["black_level"], 900, 1.0, "显式 ROI 时按 ROI 中值算")
    check("偏高" in w.status_label.text(), f"高电平 ROI 有确认提示: {w.status_label.text()[:70]}")


def test_reference_alignment():
    path, data, _, _ = _make_sample("ui_align.raw", width=320, height=240)
    w = _window(path)
    from utils import accel, stats as S
    shifted = accel.shift_image(data, 4.0, -3.0)
    w.ref_raw = shifted
    w._update_reference_availability()
    w.toggle_compare_mode(True)
    w.set_compare_mode("差分 Difference (|A-B|)")
    before = S.diff_stats(w.raw, w.ref_raw, 10)["psnr"]
    w.toggle_align_reference(True)
    check(w.ref_shift is not None, "算出了平移量")
    check(abs(w.ref_shift["dx"] - 4) < 0.5 and abs(w.ref_shift["dy"] + 3) < 0.5,
          f"平移量正确 ({w.ref_shift['dx']:+.2f},{w.ref_shift['dy']:+.2f})")
    after = S.diff_stats(w.raw, w._aligned_ref(), 10)["psnr"]
    check(after > before + 5, f"对齐后 PSNR 提升（{before:.1f} -> {after:.1f} dB）")
    check("对齐" in w.status_label.text(), "状态栏报告对齐结果")
    w.toggle_align_reference(False)
    check(w._aligned_ref() is w.ref_raw, "关闭对齐后返回原始参考帧")
    # 没有参考帧时不能打开对齐（动作状态要回退）
    w.ref_raw = None
    w.align_action.setChecked(True)
    w.toggle_align_reference(True)
    check(not w.align_reference and not w.align_action.isChecked(),
          "无参考帧时自动取消对齐并同步动作状态")


def test_defect_csv_import():
    path, _, _, _ = _make_sample("ui_imp.raw", width=200, height=160)
    w = _window(path, {"width": 200, "height": 160})
    from utils import export as X
    csv_path = os.path.join(TMP, "imp.csv")
    header, rows = X.defects_to_rows([
        {"type": "hot", "x": 30, "y": 40, "channel": "R", "value": 1023,
         "delta": 900.0, "note": "single", "source": "manual"},
        {"type": "col", "x": 55, "y": 0, "channel": "B", "value": 500,
         "delta": 100.0, "note": "full col", "source": "manual"},
    ])
    X.write_csv(csv_path, header, rows)
    DIALOG_RESULTS["open"] = csv_path
    w.import_defects_csv()
    check(len(w.last_defects) == 2, f"导入 2 条（实际 {len(w.last_defects)}）")
    check(len(w.canvas.overlays) == 2, "叠加层已生成")
    check(w.stats_panel.defect_table.rowCount() == 2, "缺陷表更新")
    check(any("CSV" in s for s in w.defect_sets), "缺陷来源标记为 CSV")
    # 坏 CSV 不应崩
    bad = os.path.join(TMP, "bad.csv")
    open(bad, "w", encoding="utf-8").write("foo,bar\n1,2\n")
    DIALOG_RESULTS["open"] = bad
    w.import_defects_csv()
    check(len(w.last_defects) == 2, "坏 CSV 被忽略且不破坏已有结果")


def test_clahe_and_hq_downscale():
    path, _, _, _ = _make_sample("ui_clahe.raw", width=320, height=240)
    w = _window(path)
    w.on_render_requested(dict(w.params, view="Mono", clahe_enable=True,
                               clahe_clip=3.0, clahe_tiles=4))
    check(w.display is not None and w.display.dtype == np.uint8, "CLAHE 渲染成功")
    check(w.params["clahe_enable"] is True, "参数已记录")
    w.toggle_hq_downscale(False)
    check(w.canvas.high_quality_downscale is False, "关闭高质量缩小")
    w.toggle_hq_downscale(True)
    check(w.canvas.high_quality_downscale is True, "开启高质量缩小")
    check("INTER_AREA" in w.status_label.text() or "最近邻" in w.status_label.text(),
          "状态栏反馈采样方式")
    check("OpenCV" in w.display_panel.backend_label.text()
          or "numpy" in w.display_panel.backend_label.text(), "面板显示加速后端")


CHR10 = chr(10)


def test_batch_analyze():
    files = []
    for i in range(3):
        p, _, _, _ = _make_sample(f"ui_batch{i}.raw", width=200, height=160)
        files.append(p)
    out = os.path.join(TMP, "batch_ui.csv")
    w = _window(files[0], {"width": 200, "height": 160})
    DIALOG_RESULTS["open_many"] = files
    DIALOG_RESULTS["save"] = out
    DIALOG_RESULTS["question"] = "Yes"          # 同时统计坏点数
    w.batch_analyze()
    check(os.path.exists(out), "汇总 CSV 已生成")
    lines = open(out, encoding="utf-8-sig").read().splitlines()
    check(len(lines) == 4, f"3 帧 + 表头（实际 {len(lines)}）")
    check("defects" in lines[0], "含坏点数列")
    check(all(os.path.basename(f) in CHR10.join(lines) for f in files),
          "三个文件都在表里")
    check("批量分析完成" in w.status_label.text(), f"状态栏: {w.status_label.text()[:50]}")
    DIALOG_RESULTS["question"] = "No"


def test_slot_exception_guard():
    """槽内异常必须被兜住（PyQt 里会打印 traceback、吞掉操作，还可能直接 abort）。"""
    path, _, _, _ = _make_sample("ui_guard.raw", width=160, height=120)
    w = _window(path, {"width": 160, "height": 120})

    def boom():
        raise RuntimeError("故意失败")

    guard = w._guard(boom)
    check(guard() is None, "异常被兜住且返回 None")
    check("操作失败" in w.status_label.text(), f"状态栏报告失败: {w.status_label.text()}")
    check(any(c[0] == "critical" for c in MODAL_CALLS), "弹出过一次错误提示")

    # 包装器要按形参个数裁剪 QAction.triggered 传来的 checked 参数
    check(w._guard(lambda: 42)(False) == 42, "无参可调用对象忽略多余参数")
    check(w._guard(lambda v: v)(False) is False, "单参可调用对象收到参数")
    check(w._guard(lambda *a: len(a))(True, 1, 2) == 3, "*args 可调用对象原样透传")

    # 真实的导出槽：写文件失败不能把异常抛出去
    from utils import export as X
    original = X.write_csv

    def fail(*a, **k):
        raise IsADirectoryError("路径是目录")

    X.write_csv = fail
    try:
        DIALOG_RESULTS["save"] = os.path.join(TMP, "whatever.csv")
        w._guard(w.export_stats_csv)()          # 走的是同一个保护逻辑
    finally:
        X.write_csv = original
    check("操作失败" in w.status_label.text(), "导出失败被报告而不是抛出")


def test_navigate_frame_robustness():
    """帧导航：帧数要扣 header、文件消失/坏帧号不能留下不一致状态。"""
    import generate_sample as gs
    data, meta = gs.generate("dark", width=64, height=48, bit_depth=10, frames=3, seed=1)
    path = os.path.join(TMP, "hdr_frames.raw")
    info = gs.write_sample(path, data, meta, "unpacked", header=1024)
    w = _window(path, {"width": 64, "height": 48, "header_bytes": 1024})
    check(w.raw is not None, "带 header 的多帧文件能打开")
    w.navigate_frame(1)
    check(w.params["frame_index"] == 1, "切到第 1 帧")
    w.navigate_frame(1)
    check(w.params["frame_index"] == 2, "切到第 2 帧")
    w.navigate_frame(1)
    check(w.params["frame_index"] == 2, "已经是最后一帧时停住（帧数按 header 扣减）")
    check("最后一帧" in w.status_label.text(), f"提示: {w.status_label.text()}")

    # 文件消失：不能抛异常，帧号保持不变
    missing = os.path.join(TMP, "hdr_frames_gone.raw")
    os.replace(path, missing)
    before = w.params["frame_index"]
    w.navigate_frame(-1)                     # 内部有 os.path.exists 保护
    check(w.params["frame_index"] == before, "文件不存在时帧号不变")
    check("不存在" in w.status_label.text(), f"提示文件不存在: {w.status_label.text()}")
    os.replace(missing, path)


def test_reference_lifecycle():
    """参考帧生命周期：自动参考要能随撤销清掉，换文件时尺寸不符要清除。"""
    path, data, _, _ = _make_sample("ui_ref1.raw", width=200, height=160)
    other, data2, _, _ = _make_sample("ui_ref2.raw", width=320, height=240)
    w = _window(path, {"width": 200, "height": 160})
    from algorithms.bad_pixel import BadPixelDetectionAlgorithm
    algo = w.algorithm_manager.get_algorithm("Bad Pixel Detection")
    params = {k: v["default"] for k, v in algo.get_parameters().items()}
    params["visualize_only"] = False
    w.run_algorithm("Bad Pixel Detection", params)
    check(w.ref_raw is not None and w._ref_auto, "校正后自动把原图设为参考帧")
    check("ref:" in w.info_label.text(), f"状态栏常驻显示参考帧: {w.info_label.text()}")
    w.undo_correction()
    check(w.ref_raw is None and not w._ref_auto, "撤销校正后参考帧一起清掉（否则差分全黑）")

    # 尺寸不一致的新文件要清参考帧
    w.ref_raw = data.copy()
    w.ref_file_path = "/tmp/fake.raw"
    w.load_image(other, {"width": 320, "height": 240, "bit_depth": 10,
                         "pattern": "RGGB", "view": "Mono"})
    check(w.ref_raw is None, "换到尺寸不同的文件时清除参考帧")
    # 尺寸一致则保留（序列对比常用）：参考帧 200x160，即将打开的 path 也是 200x160
    w.ref_raw = data.copy()
    w.ref_file_path = "/tmp/fake2.raw"
    w.load_image(path, {"width": 200, "height": 160, "bit_depth": 10,
                        "pattern": "RGGB", "view": "Mono"})
    check(w.ref_raw is not None, "换到同尺寸文件时保留参考帧")


def test_canvas_double_click_and_roi_clamp():
    """双击语义（放大→适应窗口）、图像外起拖的 ROI 起点要夹紧。"""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    path, _, _, _ = _make_sample("ui_dc.raw", width=640, height=480)
    w = _window(path, {"width": 640, "height": 480})
    c = w.canvas
    c.resize(300, 200)

    def dbl(x, y):
        return QMouseEvent(QEvent.Type.MouseButtonDblClick, QPointF(x, y), QPointF(x, y),
                           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier)

    c.zoom_to(8.0)
    c.mouseDoubleClickEvent(dbl(150, 100))
    check(c.scale < 1.0, f"放大状态双击 -> 适应窗口（scale={c.scale:.3f}）")
    check(abs(c.scale - min(300 / 640, 200 / 480)) < 0.05, f"适应窗口比例正确 ({c.scale:.3f})")
    c.zoom_to(0.5)
    c.mouseDoubleClickEvent(dbl(150, 100))
    check(abs(c.scale - 1.0) < 1e-6, f"缩小状态双击 -> 1:1（scale={c.scale}）")

    # 从图像外起拖：ROI 起点必须是画内坐标，且与统计区域一致
    from utils import stats as S
    c.set_view_params(1.0, __import__("PyQt6.QtCore", fromlist=["QPoint"]).QPoint(-200, -150))
    c.set_roi_enabled(True)
    c.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
                                  Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                  Qt.KeyboardModifier.NoModifier))
    c.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(120, 90), QPointF(120, 90),
                                 Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.NoModifier))
    c.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(120, 90),
                                    QPointF(120, 90), Qt.MouseButton.LeftButton,
                                    Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    roi = c.roi
    check(roi is not None and roi[0] >= 0 and roi[1] >= 0, f"ROI 起点已夹进画内: {roi}")
    if roi:
        check(S.roi_region(w.raw.shape, roi) == roi,
              f"标签坐标与实际统计区域一致（{roi}）")


def test_settings_roundtrip_is_isolated():
    """参数记忆：保存后新窗口应恢复（同时确认测试确实写在临时目录里）。"""
    path, _, _, _ = _make_sample("ui_cfg.raw", width=160, height=120)
    w = _window(path, {"width": 160, "height": 120})
    w.params["pattern"] = "BGGR"
    w.params["bit_depth"] = 12
    w.params["view"] = "Mono"
    w._save_settings()
    check(w.settings.fileName().startswith(TMP),
          f"设置写在临时目录（不会污染开发机偏好）: {w.settings.fileName()}")
    w2 = MainWindow()
    check(w2.params["pattern"] == "BGGR" and w2.params["bit_depth"] == 12,
          f"新窗口恢复上次参数（pattern={w2.params['pattern']}, "
          f"bit_depth={w2.params['bit_depth']}）")
    check(w2.display_panel.get_params()["pattern"] == "BGGR", "面板同步恢复")
    w2.params["pattern"] = "Mono/None"
    w2.params["bit_depth"] = 10
    w2._save_settings()


def test_monospace_font_is_real_and_fixed_pitch():
    """等宽字体必须挑到真实存在的固定宽度族。

    回归：曾经用 QFont("Monospace") 让 Qt 去枚举字体别名（启动多花 ~70ms 并打印
    告警）；而 QFontDatabase.systemFont(FixedFont) 在 Qt6/macOS 上返回的是
    .AppleSystemUIFont（比例字体、fixedPitch=False，还照样触发别名枚举）。
    """
    from PyQt6.QtGui import QFontInfo
    from ui.qtutil import mono_family, mono_font, mono_style
    fam = mono_family()
    check(fam.lower() not in ("monospace", "sans serif", "sans-serif", ""),
          f"字体族是具体名字（{fam}）")
    f = mono_font(12, bold=True)
    info = QFontInfo(f)
    check(info.fixedPitch(), f"挑到的是固定宽度字体（{info.family()}）")
    check(f.pixelSize() == 12 and f.bold(), "像素大小/粗体生效")
    check(mono_font(12, bold=True) is f, "同一规格的字体被缓存复用")
    check(mono_font(9).pixelSize() == 9, "不同像素大小分开缓存")
    check(mono_style("color: #9ad;").startswith("font-family:"), "样式表用具体族名")
    check("monospace;" not in mono_style(), "样式表里不再出现泛型 monospace")


def test_dock_and_toolbar_have_object_names():
    """saveState/restoreState 靠 objectName 匹配停靠窗口与工具栏。

    回归：没有 objectName 时 Qt 打印
      QMainWindow::saveState(): 'objectName' not set for QDockWidget ... 'Tools'
    并且布局记忆实际失效（restoreState 返回 False）。
    """
    path, _, _, _ = _make_sample("ui_state.raw", width=160, height=120)
    w = _window(path, {"width": 160, "height": 120})
    check(w.dock.objectName() == "toolsDock", f"Tools 停靠窗口 objectName: {w.dock.objectName()!r}")
    check(w.measure_dock.objectName() == "measureDock",
          f"测量停靠窗口 objectName: {w.measure_dock.objectName()!r}")
    from PyQt6.QtWidgets import QToolBar
    bars = w.findChildren(QToolBar)
    check(bars and all(b.objectName() for b in bars),
          f"工具栏都有 objectName: {[b.objectName() for b in bars]}")
    state = w.saveState()
    check(state.size() > 0, f"saveState 有内容（{state.size()} 字节）")
    w2 = _window(path, {"width": 160, "height": 120})
    check(w2.restoreState(state) is True, "restoreState 成功（objectName 齐备）")
    # 布局记忆要真的生效：隐藏测量面板后保存，新窗口恢复后应同样是隐藏的
    w.measure_dock.hide()
    state2 = w.saveState()
    w3 = _window(path, {"width": 160, "height": 120})
    w3.restoreState(state2)
    check(not w3.measure_dock.isVisibleTo(w3),
          "恢复后测量面板保持隐藏（停靠布局真的记住了）")


def test_main_window_grab_renders():
    path, _, _, _ = _make_sample("ui_grab.raw")
    w = _window(path)
    w.canvas.zoom_to(12.0)
    pm = w.canvas.grab()
    check(pm.width() > 100 and not pm.isNull(), "画布离屏渲染成功")
    pm2 = w.grab()
    check(pm2.width() > 100, "整窗离屏渲染成功")
    w.dock.show(); w.measure_dock.show()
    APP.processEvents()
    check(not w.grab().isNull(), "带面板渲染成功")


if __name__ == "__main__":
    sys.exit(run(globals(), "无头 UI 集成"))
