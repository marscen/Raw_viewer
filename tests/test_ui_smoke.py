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

from tests._util import check, run  # noqa: E402

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

TMP = tempfile.mkdtemp(prefix="rawv2_ui_")
# 隔离 QSettings，避免读到开发机上真实使用留下的参数
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TMP)

APP = QApplication.instance() or QApplication([])


def _patch_modals():
    """测试里绝不能弹模态框（会永久阻塞），替换成记录用的空实现。"""
    from PyQt6.QtWidgets import QMessageBox
    import ui.main_window as mw
    calls = []

    def make(kind):
        def _fn(parent, title, text, *a, **kw):
            calls.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return staticmethod(_fn)

    for kind in ("information", "warning", "critical", "question"):
        setattr(mw.QMessageBox, kind, make(kind))
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


def _window(path, params=None):
    w = MainWindow()
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
                         ("", None), ("abc", None)):
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
