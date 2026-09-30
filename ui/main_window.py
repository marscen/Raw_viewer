"""主窗口：把数据层/显示层/测量层/算法层组装成一个好用工具。

相比旧版新增的工作流能力：
  * 打开时自动推断布局 + 尺寸校验（dialogs.ImageParamsDialog）；
  * ROI 框选 → 统计/直方图/行列 profile 实时跟着走，算法也可"只在 ROI 内跑"；
  * 像素检查器（点选像素看邻域原始 DN）；
  * 缺陷清单（坏点/坏线/阴影/饱和/帧差）统一汇总、可定位、可导出 CSV；
  * 与参考帧对比：并排 / 差分（Hot 伪彩）/ 50% 混合，并给出 PSNR/RMSE；
  * 校正类算法（坏点校正、平坦场校正）结果可撤销（Ctrl+Z），并自动把原图设为
    参考帧 → 立刻用差分视图检查改动量；
  * 多文件/多帧浏览（PgUp/PgDn 换文件，,/. 换帧号）；
  * 导出：显示图 PNG、原始 DN 的 16bit PNG/TIFF（无损）、带缺陷标注的 PNG、
    统计 CSV、缺陷 CSV、测量报告 Markdown。
"""
from __future__ import annotations

import functools
import inspect
import os

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer, QPoint
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog,
                             QLabel, QMainWindow, QMessageBox, QSplitter,
                             QTabWidget, QToolBar)

from algorithms.manager import AlgorithmManager
from ui.canvas import ImageCanvas
from ui.dialogs import ImageParamsDialog
from ui.qtutil import mono_style
from ui.goto_box import CoordinateJumpBox
from ui.sidebar import AlgorithmPanel, DisplayControlPanel
from ui.stats_panel import StatsPanel
from utils import accel, cfa, export as X, stats as S
from utils.display import DisplayParams, render_display, compute_levels
from utils.raw_io import RawLoadSpec, RawReadError, load_raw
import traceback

APP_NAME = "RAWViewer"
RAW_EXTS = (".raw", ".bin", ".bayer", ".dump", ".dat")
SETTINGS_KEY_PARAMS = "raw_viewer/last_params"
SETTINGS_KEY_RECENT = "raw_viewer/recent_files"
SETTINGS_KEY_GEOMETRY = "raw_viewer/geometry"
SETTINGS_KEY_STATE = "raw_viewer/window_state"
# 指定这个环境变量时，参数/布局存到该目录下的独立 INI（便携模式；自动化测试也靠它
# 隔离 —— Qt 在 macOS 上会忽略 setDefaultFormat，(org, app) 构造出来永远是 plist）
SETTINGS_ENV = "RAWV2_SETTINGS_DIR"


def make_settings():
    """构造 QSettings：优先用 RAWV2_SETTINGS_DIR 指定的独立文件。"""
    folder = os.environ.get(SETTINGS_ENV, "").strip()
    if folder:
        try:
            os.makedirs(folder, exist_ok=True)
            return QSettings(os.path.join(folder, "raw_viewer.ini"),
                             QSettings.Format.IniFormat)
        except OSError:
            pass
    return QSettings(APP_NAME, APP_NAME)

COMPARE_MODES = ["并排 Side-by-side", "差分 Difference (|A-B|)", "混合 Blend 50%"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Viewer — CMOS 测试预览/分析")
        self.resize(1500, 900)
        self.setAcceptDrops(True)

        self.settings = make_settings()

        # ---------------- 状态 ----------------
        self.params: dict = self._default_params()
        self.raw = None                 # 当前帧原始 DN
        self.display = None             # 当前显示数组
        self.render_info = None
        self.current_file_path = None
        self.ref_raw = None
        self.ref_file_path = None
        self._ref_auto = False          # 参考帧是否由"校正后自动设置"而来
        self.ref_display = None
        self.roi = None
        self.selected_pixel = None
        self.undo_stack = []            # [(raw, label)]
        self.overlay_sets = {}          # {算法名: [叠加层]}
        self.last_defects = []
        self.defect_sets = {}           # {算法名: [缺陷]}，多次运行累积
        self.compare_enabled = False
        self.compare_mode = COMPARE_MODES[0]
        self.align_reference = False     # 是否用相位相关对齐参考帧
        self.ref_shift = None            # 对齐结果 (dx, dy, response)

        # ---------------- 画布 ----------------
        self.canvas = ImageCanvas()
        self.ref_canvas = ImageCanvas()
        self.ref_canvas.setVisible(False)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.ref_canvas)
        self.splitter.setCollapsible(0, False)
        self.setCentralWidget(self.splitter)

        self.canvas.view_changed.connect(self._guard(self._on_view_changed))
        self.ref_canvas.view_changed.connect(self._sync_to_main)
        self.canvas.hover_info.connect(self._guard(self._on_hover))
        self.canvas.pixel_clicked.connect(self._guard(self._on_pixel_clicked))
        self.canvas.roi_changed.connect(self._guard(self._on_roi_changed))
        self.ref_canvas.hover_info.connect(self._guard(self._on_hover))

        # ---------------- 状态栏 ----------------
        self.status_label = QLabel("就绪 —— File → Open RAW (Ctrl+O)")
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(mono_style())
        self.roi_label = QLabel("ROI: 全图")
        self.roi_label.setStyleSheet(mono_style("color: #6cf;"))
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.roi_label)
        self.statusBar().addPermanentWidget(self.info_label)

        # ---------------- 算法 / 面板 ----------------
        self.algorithm_manager = AlgorithmManager()
        self.display_panel = DisplayControlPanel()
        self.algo_panel = AlgorithmPanel(self.algorithm_manager)
        self.stats_panel = StatsPanel()

        self.side_tabs = QTabWidget()
        self.side_tabs.setObjectName("sideTabs")
        self.side_tabs.addTab(self.display_panel, "显示 Display")
        self.side_tabs.addTab(self.algo_panel, "算法 Algorithms")
        self.dock = QDockWidget("Tools", self)
        self.dock.setObjectName("toolsDock")          # saveState/restoreState 需要
        self.dock.setWidget(self.side_tabs)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                                  Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

        # 测量面板放在底部：直方图/profile/缺陷表都需要横向空间
        self.measure_dock = QDockWidget("测量 Measure", self)
        self.measure_dock.setObjectName("measureDock")
        self.measure_dock.setWidget(self.stats_panel)
        self.measure_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                                          Qt.DockWidgetArea.RightDockWidgetArea |
                                          Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.measure_dock)
        self.resizeDocks([self.measure_dock], [300], Qt.Orientation.Vertical)
        self.resizeDocks([self.dock], [400], Qt.Orientation.Horizontal)

        self.display_panel.reload_requested.connect(self._guard(self.on_reload_requested))
        self.display_panel.render_requested.connect(self._guard(self.on_render_requested))
        self.display_panel.auto_levels_requested.connect(self._guard(self.on_auto_levels))
        self.algo_panel.run_algorithm.connect(self._guard(self.run_algorithm))
        self.stats_panel.defect_activated.connect(self._guard(self.locate_defect))
        self.stats_panel.export_defects_requested.connect(self._guard(self.export_defects_csv))
        self.stats_panel.export_stats_requested.connect(self._guard(self.export_stats_csv))
        self.stats_panel.clear_defects_requested.connect(self._guard(self.clear_defects))
        self.stats_panel.histogram_bins_changed.connect(self._guard(self.update_histogram))
        self.stats_panel.inspector_radius.valueChanged.connect(self._guard(self.update_inspector))

        # 测量刷新去抖：拖动/缩放时不必每次都算百分位
        self.measure_timer = QTimer(self)
        self.measure_timer.setSingleShot(True)
        self.measure_timer.setInterval(120)
        self.measure_timer.timeout.connect(self._guard(self._do_measure))

        # ---------------- 菜单 / 工具栏 ----------------
        self._build_actions()
        self._restore_settings()

    # ==================================================================
    # 槽函数保护
    # ==================================================================
    def _guard(self, fn):
        """把槽函数包一层异常保护。

        PyQt 里槽内抛出的异常默认只打印 traceback（新版本还可能直接 abort
        进程），用户看到的是"点了没反应"，而且状态可能已经改了一半。这里统一
        转成状态栏 + 一次性提示，保证任何一个动作出错都不会拖垮整个程序。

        包装器会按被包函数的形参个数裁剪信号参数：PyQt 对 `*args` 的包装器会
        把 QAction.triggered 的 checked 参数一并传进来，不裁剪就会 TypeError。
        """
        try:
            params = list(inspect.signature(fn).parameters.values())
            n_pos = sum(1 for q in params if q.kind in (q.POSITIONAL_ONLY,
                                                        q.POSITIONAL_OR_KEYWORD))
            has_var = any(q.kind == q.VAR_POSITIONAL for q in params)
        except (TypeError, ValueError):
            n_pos, has_var = 0, True

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not has_var and len(args) > n_pos:
                args = args[:n_pos]
            try:
                return fn(*args, **kwargs)
            except Exception as exc:                       # noqa: BLE001
                traceback.print_exc()
                try:
                    self.status_label.setText(f"操作失败：{exc}")
                except Exception:
                    pass
                try:
                    QMessageBox.critical(self, "操作失败", f"{exc}")
                except Exception:
                    pass
                return None
        return wrapper

    # ==================================================================
    # 参数
    # ==================================================================
    @staticmethod
    def _default_params() -> dict:
        return {
            "width": 1920, "height": 1080, "bit_depth": 10, "packing": "unpacked",
            "header_bytes": 0, "stride_bytes": 0, "endian": "little",
            "data_shift": 0, "frame_index": 0, "frame_stride_bytes": 0,
            "pattern": "Mono/None", "view": "Mono", "colormap": "Gray",
            "stretch": "Percentile", "p_low": 0.5, "p_high": 99.5, "sigma_k": 3.0,
            "gamma": 1.0, "invert": False, "black_level": 0, "white_level": 0,
            "stretch_on_roi": False,
            "clahe_enable": False, "clahe_clip": 2.0, "clahe_tiles": 8,
        }

    def _spec(self) -> RawLoadSpec:
        p = self.params
        return RawLoadSpec(
            width=int(p["width"]), height=int(p["height"]),
            bit_depth=int(p["bit_depth"]), packing=p.get("packing", "unpacked"),
            header_bytes=int(p.get("header_bytes", 0)),
            stride_bytes=int(p.get("stride_bytes", 0)),
            endian=p.get("endian", "little"),
            data_shift=int(p.get("data_shift", 0)),
            frame_index=int(p.get("frame_index", 0)),
            frame_stride_bytes=int(p.get("frame_stride_bytes", 0)),
        )

    def _display_params(self) -> DisplayParams:
        p = self.params
        return DisplayParams(
            bit_depth=int(p["bit_depth"]), view=p.get("view", "Mono"),
            pattern=p.get("pattern", "Mono/None"),
            black_level=float(p.get("black_level", 0)),
            white_level=float(p.get("white_level", 0)),
            stretch=p.get("stretch", "Percentile"),
            p_low=float(p.get("p_low", 0.5)), p_high=float(p.get("p_high", 99.5)),
            sigma_k=float(p.get("sigma_k", 3.0)), gamma=float(p.get("gamma", 1.0)),
            invert=bool(p.get("invert", False)), colormap=p.get("colormap", "Gray"),
            stretch_on_roi=bool(p.get("stretch_on_roi", False)),
            clahe_enable=bool(p.get("clahe_enable", False)),
            clahe_clip=float(p.get("clahe_clip", 2.0)),
            clahe_tiles=int(p.get("clahe_tiles", 8)))

    def _merge_params(self, new: dict):
        for k, v in (new or {}).items():
            if k == "demosaic":                     # 旧字段
                self.params["view"] = "Bayer Demosaic" if v else "Mono"
                continue
            if k in self.params or k in ("packing", "header_bytes", "stride_bytes",
                                         "endian", "data_shift", "frame_index",
                                         "frame_stride_bytes"):
                self.params[k] = v
        self.display_panel.set_params(self.params, block=True)

    # ==================================================================
    # 动作 / 菜单
    # ==================================================================
    def _build_actions(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        view_menu = menubar.addMenu("View")
        analyze_menu = menubar.addMenu("Analyze")
        help_menu = menubar.addMenu("Help")

        def act(text, slot, shortcut=None, checkable=False, checked=False, tip=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setCheckable(checkable)
            a.setChecked(checked)
            a.triggered.connect(self._guard(slot))
            if tip:
                a.setToolTip(tip)
            return a

        self.open_action = act("Open RAW…", self.open_raw_file, "Ctrl+O")
        self.open_ref_action = act("Open Reference Image…", self.open_reference_file, "Ctrl+R")
        self.recent_menu = file_menu.addMenu("Recent Files")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        self.export_action = act("Export Display Image (PNG)…", self.export_display_image, "Ctrl+E")
        self.export_dn_action = act("Export RAW DN (16bit PNG/TIFF)…", self.export_raw_dn, "Ctrl+Shift+E",
                                    tip="导出未经显示变换的原始 DN，位深不丢，供算法同事复算")
        self.export_annot_action = act("Export Annotated Snapshot (PNG)…", self.export_annotated)
        self.export_report_action = act("Export Measurement Report (MD)…", self.export_report)
        file_menu.addSeparator()
        file_menu.addAction(act("Quit", self.close, "Ctrl+Q"))
        for a in (self.open_action, self.open_ref_action, self.export_action,
                  self.export_dn_action, self.export_annot_action, self.export_report_action):
            file_menu.addAction(a)

        edit_menu.addAction(act("Undo Correction", self.undo_correction, "Ctrl+Z"))
        edit_menu.addAction(act("Clear ROI", lambda: self.canvas.set_roi(None), "Esc"))
        edit_menu.addAction(act("ROI = Full Frame", self.select_full_roi, "Ctrl+A"))

        self.reload_action = act("Reload from Disk", self.reload_current, "Ctrl+Shift+R")
        edit_menu.addAction(self.reload_action)

        view_menu.addAction(self.dock.toggleViewAction())
        view_menu.addAction(self.measure_dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(act("Fit to Window", self.canvas.fit_to_window, "F"))
        view_menu.addAction(act("Zoom 1:1", self.canvas.zoom_1_1, "1"))
        view_menu.addAction(act("Zoom In", lambda: self.canvas.zoom_step(1), "+"))
        view_menu.addAction(act("Zoom Out", lambda: self.canvas.zoom_step(-1), "-"))
        view_menu.addSeparator()
        self.compare_action = act("Compare Mode", self.toggle_compare_mode, "Ctrl+D", checkable=True)
        view_menu.addAction(self.compare_action)
        view_menu.addSeparator()
        self.compare_group = QActionGroup(self)
        self.compare_group.setExclusive(True)
        self.compare_actions = {}
        for mode in COMPARE_MODES:
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(mode == self.compare_mode)
            a.triggered.connect(lambda _c, m=mode: self.set_compare_mode(m))
            self.compare_group.addAction(a)
            view_menu.addAction(a)
            self.compare_actions[mode] = a

        self.roi_action = act("ROI Select Mode", self.toggle_roi_mode, "Ctrl+Shift+A", checkable=True,
                              tip="开启后左键拖拽=框选 ROI；关闭时右键拖拽也能框选")
        view_menu.addAction(self.roi_action)
        view_menu.addAction(act("Toggle Pixel Grid/Values", self.toggle_grid, "G", checkable=True,
                                checked=True))
        view_menu.addAction(act("Toggle Crosshair", self.toggle_crosshair, "C", checkable=True,
                                checked=True))
        view_menu.addAction(act("High-quality Downscale (INTER_AREA)", self.toggle_hq_downscale,
                                "Ctrl+Shift+Q", checkable=True, checked=True,
                                tip="缩小时用区域平均而不是最近邻：RAW 噪声图看起来干净得多"))
        self.align_action = act("Auto-align Reference (phase correlation)",
                                self.toggle_align_reference, "Ctrl+Shift+D", checkable=True,
                                tip="先用相位相关测出参考帧的平移量再对齐，再做差分/混合")
        view_menu.addAction(self.align_action)

        analyze_menu.addAction(act("Next File (PgDn)", lambda: self.navigate_file(1), "PgDown"))
        analyze_menu.addAction(act("Previous File (PgUp)", lambda: self.navigate_file(-1), "PgUp"))
        analyze_menu.addAction(act("Next Frame", lambda: self.navigate_frame(1), "."))
        analyze_menu.addAction(act("Previous Frame", lambda: self.navigate_frame(-1), ","))
        analyze_menu.addSeparator()
        analyze_menu.addAction(act("Estimate Black Level (dark frame)", self.estimate_black_level, "Ctrl+B",
                                   tip="有 ROI 时用 ROI 中值；否则用全图直方图峰值（仅暗场可信）"))
        analyze_menu.addAction(act("Batch Analyze Files…", self.batch_analyze, "Ctrl+Shift+B",
                                   tip="选一批 RAW，导出每帧的均值/标准差/饱和/黑电平/坏点数汇总 CSV"))
        analyze_menu.addSeparator()
        analyze_menu.addAction(act("Import Defect CSV…", self.import_defects_csv,
                                   tip="载入之前导出的缺陷清单并叠加到画布上，用于和上游 defect map 对拍"))
        analyze_menu.addSeparator()
        analyze_menu.addAction(act("Jump to Coordinate…", self.focus_goto, "Ctrl+G",
                                   tip="输入 X,Y 后跳转（支持 1234,567 这类粘贴格式）"))
        analyze_menu.addAction(act("Auto Levels (auto black/white)", self.on_auto_levels, "Ctrl+L"))
        analyze_menu.addAction(act("Update Measurements", self.update_measurements, "Ctrl+M"))

        help_menu.addAction(act("Shortcuts / About", self.show_about))

        # ---------------- 工具栏 ----------------
        tb = QToolBar("Main")
        tb.setObjectName("mainToolBar")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self.open_action)
        tb.addAction(self.open_ref_action)
        tb.addAction(act("◀ 文件", lambda: self.navigate_file(-1), tip="上一个文件 (PgUp)"))
        tb.addAction(act("文件 ▶", lambda: self.navigate_file(1), tip="下一个文件 (PgDn)"))
        tb.addSeparator()
        tb.addAction(act("Fit", self.canvas.fit_to_window))
        tb.addAction(act("1:1", self.canvas.zoom_1_1))
        tb.addAction(self.roi_action)
        tb.addSeparator()
        tb.addAction(self.compare_action)
        # 工具栏上的下拉框与菜单里的互斥项保持同步
        self.compare_combo = QComboBox()
        self.compare_combo.addItems(COMPARE_MODES)
        self.compare_combo.currentTextChanged.connect(self.set_compare_mode)
        tb.addWidget(self.compare_combo)
        tb.addSeparator()
        self.goto_box = CoordinateJumpBox()
        self.goto_box.setToolTip("输入坐标跳转（Ctrl+G）")
        self.goto_box.jump_requested.connect(self.goto_coordinate)
        tb.addWidget(self.goto_box)
        tb.addSeparator()
        tb.addAction(act("Auto Levels", self.on_auto_levels))
        tb.addAction(act("Run Algorithm", self._run_selected_algorithm))

    def _run_selected_algorithm(self):
        self.side_tabs.setCurrentWidget(self.algo_panel)
        self.algo_panel.on_run_clicked()

    # ---- 高质量缩小 / 参考帧对齐 / 黑电平估计 / 缺陷导入 / 批量分析 ----
    def toggle_hq_downscale(self, checked):
        self.canvas.high_quality_downscale = bool(checked)
        self.ref_canvas.high_quality_downscale = bool(checked)
        self.canvas.update()
        self.ref_canvas.update()
        self.status_label.setText(
            f"缩小采样: {'高质量(INTER_AREA)' if checked else '最近邻'}")

    def toggle_align_reference(self, checked):
        self.align_reference = bool(checked)
        if self.align_reference and self.ref_raw is None:
            self.align_reference = False
            self.align_action.setChecked(False)          # 状态要和实际一致
            self.status_label.setText("还没有参考帧：File → Open Reference Image…")
            return
        if self.align_reference and self.raw is not None and self.ref_raw is not None \
                and self.ref_raw.shape != self.raw.shape:
            self.align_reference = False
            self.align_action.setChecked(False)
            self.status_label.setText("参考帧尺寸与当前帧不一致，无法对齐")
            return
        self._update_compare_display()
        self._report_alignment()

    def _report_alignment(self):
        if self.ref_raw is None or self.raw is None or self.ref_raw.shape != self.raw.shape:
            return
        if not self.align_reference:
            self.ref_shift = None
            return
        res = accel.phase_correlate(self.raw, self.ref_raw)
        if not res:
            self.ref_shift = None
            return
        self.ref_shift = res
        before = S.diff_stats(self.raw, self.ref_raw, int(self.params["bit_depth"]))
        aligned = accel.shift_image(self.ref_raw, -res["dx"], -res["dy"])
        after = S.diff_stats(self.raw, aligned, int(self.params["bit_depth"]))
        msg = (f"参考帧对齐: 平移 ({res['dx']:+.2f}, {res['dy']:+.2f}) px, "
               f"响应 {res['response']:.3f} [{res['method']}]")
        if before and after:
            msg += (f" | PSNR {before['psnr']:.2f} → {after['psnr']:.2f} dB")
        self.status_label.setText(msg)

    def estimate_black_level(self):
        """估计黑电平：有 ROI 用 ROI 中值，否则用全图直方图峰值（暗场）。"""
        if self.raw is None:
            QMessageBox.warning(self, "提示", "请先打开一张 RAW 图")
            return
        from utils.batch import estimate_black_level
        roi = self.roi
        value = estimate_black_level(self.raw, int(self.params["bit_depth"]), roi)
        if value != value:                       # NaN
            QMessageBox.information(
                self, "无法估计黑电平",
                "当前画面的直方图峰值落在量程较高位置，看起来不是暗场。\n"
                "请先框选一段光学黑(OB)区域再点本功能。")
            return
        self.params["black_level"] = float(round(value))
        self.display_panel.set_params(self.params, block=True)
        self._render(reset_view=False)
        mc = (1 << int(self.params["bit_depth"])) - 1
        note = ""
        if roi and value > 0.25 * mc:
            # 显式框选 ROI 时按它算是对的（OB 区本来就该是暗的），但电平这么高
            # 通常是框错了地方，提醒一句而不是悄悄给个错的黑电平
            note = "  ⚠ ROI 电平偏高，确认框的是光学黑(OB)区"
        self.status_label.setText(
            f"黑电平估计 = {value:.2f} DN"
            f"（{'ROI 中值' if roi else '全图直方图峰值'}），已填入黑电平{note}")

    def import_defects_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入缺陷清单 CSV", self._start_dir(),
                                              "CSV (*.csv);;All Files (*)")
        if not path:
            return
        try:
            import csv as _csv
            defects = []
            with open(path, newline="", encoding="utf-8-sig") as fh:
                for row in _csv.DictReader(fh):
                    try:
                        x = int(float(row.get("x", "")))
                        y = int(float(row.get("y", "")))
                    except (TypeError, ValueError):
                        continue
                    d = {"type": row.get("type", "hot") or "hot", "x": x, "y": y,
                         "channel": row.get("channel", "-") or "-",
                         "value": row.get("value", ""), "delta": row.get("delta", ""),
                         "note": row.get("note", "")}
                    defects.append(d)
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", f"{exc}")
            return
        if not defects:
            QMessageBox.information(self, "提示", "这个 CSV 里没有可用的 x/y 记录")
            return
        name = f"CSV: {os.path.basename(path)}"
        self.defect_sets[name] = defects
        self.overlay_sets[name] = [{"type": "point", "coords": (d["x"], d["y"]),
                                    "kind": d["type"], "color": d["type"],
                                    "radius": 4} for d in defects]
        self._rebuild_defect_union()
        self.measure_dock.show()
        self.stats_panel.tabs.setCurrentIndex(3)
        self.status_label.setText(
            f"已导入 {len(defects)} 条缺陷（{os.path.basename(path)}）并叠加到画布")

    def batch_analyze(self):
        """选一批文件，逐帧算指标并导出汇总 CSV。"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要批量分析的 RAW（可多选）", self._start_dir(),
            "RAW Files (*.raw *.bin *.bayer *.dump *.dat);;All Files (*)")
        if not files:
            return
        from PyQt6.QtWidgets import QProgressDialog
        from utils.batch import analyze_frame, write_batch_csv
        use_defects = QMessageBox.question(
            self, "批量分析",
            f"将对 {len(files)} 个文件做统计。\n是否同时统计坏点数？"
            f"（用当前坏点参数，耗时更长）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
        defect_params = None
        if use_defects:
            algo = self.algorithm_manager.get_algorithm("Bad Pixel Detection")
            defect_params = {k: v["default"] for k, v in algo.get_parameters().items()}
            defect_params["visualize_only"] = True
        out_path, _ = QFileDialog.getSaveFileName(
            self, "保存汇总 CSV", os.path.join(os.path.dirname(files[0]), "batch_summary.csv"),
            "CSV (*.csv)")
        if not out_path:
            return

        progress = QProgressDialog("批量分析中…", "取消", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        rows, skipped = [], []
        spec = self._spec()
        for i, path in enumerate(files):
            progress.setValue(i)
            progress.setLabelText(f"[{i + 1}/{len(files)}] {os.path.basename(path)}")
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                if not self._layout_matches(path):
                    skipped.append((os.path.basename(path), "布局与当前参数不匹配"))
                    continue
                raw = load_raw(path, spec.copy(frame_index=0))
                info = analyze_frame(raw, self.params["pattern"], int(self.params["bit_depth"]),
                                     self.roi, defect_params=defect_params)
                info.update({"file": os.path.basename(path),
                             "size_bytes": os.path.getsize(path),
                             "width": spec.width, "height": spec.height,
                             "bit_depth": spec.bit_depth, "pattern": self.params["pattern"]})
                rows.append(info)
            except Exception as exc:
                skipped.append((os.path.basename(path), str(exc)[:60]))
        progress.setValue(len(files))
        if not rows:
            QMessageBox.warning(self, "批量分析", "没有任何文件分析成功")
            return
        write_batch_csv(out_path, rows)
        msg = f"批量分析完成：{len(rows)} 帧 -> {os.path.basename(out_path)}"
        if skipped:
            msg += f"，跳过 {len(skipped)} 个（{skipped[0][1]}…）"
        self.status_label.setText(msg)
        QMessageBox.information(self, "批量分析", msg + "\n\n列说明见 CSV 表头。")

    def toggle_roi_mode(self, checked):
        self.canvas.set_roi_enabled(checked)
        self.status_label.setText("ROI 模式: 左键拖拽框选" if checked else "ROI 模式关闭（右键仍可框选）")

    def toggle_grid(self, checked):
        self.canvas.grid_threshold = 20.0 if checked else 1e9
        self.canvas.update()

    def toggle_crosshair(self, checked):
        self.canvas.show_crosshair = checked
        self.canvas.update()

    def select_full_roi(self):
        if self.raw is None:
            return
        h, w = self.raw.shape
        self.canvas.set_roi((0, 0, w, h))

    # ==================================================================
    # 打开 / 加载
    # ==================================================================
    def open_raw_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open RAW File", self._start_dir(),
            "RAW Files (*.raw *.bin *.bayer *.dump *.dat);;All Files (*)")
        if not path:
            return
        dialog = ImageParamsDialog(self, path=path, initial=self.params)
        if dialog.exec():
            self.load_image(path, dialog.get_params())
            self._add_recent(path)

    def load_image(self, path: str, params: dict):
        """读取并显示一帧（兼容旧的 load_image(path, params) 调用方式）。"""
        self._merge_params(params)
        try:
            spec = self._spec()
            raw = load_raw(path, spec)
        except RawReadError as exc:
            QMessageBox.critical(self, "读取失败 / Read error", str(exc))
            self.status_label.setText("读取失败")
            return
        except Exception as exc:                                  # pragma: no cover
            QMessageBox.critical(self, "错误", f"{exc}\n\n{traceback.format_exc()}")
            return

        # 换文件时参考帧的处理：尺寸还一致就保留（序列对比常用），否则清除，
        # 避免上一次的参考帧继续悄悄参与差分/报告
        if self.ref_raw is not None and self.ref_raw.shape != raw.shape:
            self.ref_raw = None
            self.ref_file_path = None
            self._ref_auto = False
            self.align_reference = False
            self.ref_shift = None
            self.status_label.setText("参考帧尺寸与当前帧不一致，已清除")
        self.current_file_path = path
        self.raw = raw
        self.goto_box.clear()
        self.undo_stack.clear()
        self.clear_defects()
        self.selected_pixel = None
        self.roi = None
        self._render(reset_view=True)
        self.update_measurements()
        name = os.path.basename(path)
        self.setWindowTitle(f"RAW Viewer — {name}  {spec.width}x{spec.height} "
                            f"{spec.bit_depth}bit {spec.packing}")
        self.status_label.setText(
            f"已加载 {name}  ({spec.describe()})  {raw.min()}~{raw.max()} DN")
        self.algo_panel.set_result("")
        self._update_reference_availability()

    def reload_current(self):
        if self.current_file_path:
            self.load_image(self.current_file_path, self.params)

    def on_reload_requested(self, params: dict):
        """几何/位深/packing/CFA 变了 —— 需要重新读盘。"""
        self._merge_params(params)
        if self.current_file_path:
            self.load_image(self.current_file_path, self.params)

    def on_render_requested(self, params: dict):
        """只是显示参数变了 —— 只重渲染，保留缩放/ROI/选中点。"""
        self._merge_params(params)
        self._render(reset_view=False)

    def _render(self, reset_view: bool = False):
        if self.raw is None:
            return
        dp = self._display_params()
        roi = self.roi if dp.stretch_on_roi else None
        disp, info = render_display(self.raw, dp, roi)
        self.display = disp
        self.render_info = info
        self.canvas.set_display(disp, self.raw, dp.pattern, reset_view=reset_view)
        if reset_view:
            self.canvas.roi = self.roi
        self.display_panel.set_level_info(
            f"level {info.lo:.0f}~{info.hi:.0f} DN | {info.view}")
        self._update_compare_display()
        self._refresh_info_label()

    def on_auto_levels(self):
        """把当前画面的百分位电平写死成黑/白电平，便于锁定显示做对比。"""
        if self.raw is None:
            return
        dp = self._display_params()
        codes = self.raw
        if dp.view in ("R plane", "Gr plane", "Gb plane", "B plane", "G plane"):
            from utils.display import PLANE_VIEWS
            wanted = PLANE_VIEWS[dp.view]
            p = cfa.normalize_pattern(dp.pattern)
            names = cfa.PHASE_LABELS.get(p, ())
            mask = np.zeros(self.raw.shape, dtype=bool)
            for phase, name in enumerate(names):
                if name in wanted:
                    dy, dx = divmod(phase, 2)
                    mask[dy::2, dx::2] = True
            if mask.any():
                codes = self.raw[mask]
        lo, hi = compute_levels(codes, dp)
        self.params["stretch"] = "Fixed (black/white)"
        self.params["black_level"] = float(round(lo))
        self.params["white_level"] = float(round(max(hi, lo + 1)))
        self.display_panel.set_params(self.params, block=True)
        self._render(reset_view=False)
        self.status_label.setText(f"自动电平: {lo:.0f} ~ {hi:.0f} DN（已锁定为固定电平）")

    # ==================================================================
    # 参考帧 / 对比
    # ==================================================================
    def _update_reference_availability(self):
        has_ref = self.ref_raw is not None and self.raw is not None and \
            self.ref_raw.shape == self.raw.shape
        self.compare_action.setEnabled(has_ref)

    def open_reference_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Reference File", self._start_dir(),
            "RAW Files (*.raw *.bin *.bayer *.dump *.dat);;All Files (*)")
        if not path:
            return
        # 布局一致时直接用当前参数加载，省一次点确认
        same_size = self._layout_matches(path)
        if same_size and self.raw is not None:
            params = dict(self.params)
        else:
            dialog = ImageParamsDialog(self, path=path, initial=self.params)
            if not dialog.exec():
                return
            params = dialog.get_params()
        try:
            self.ref_raw = load_raw(path, RawLoadSpec(
                width=int(params["width"]), height=int(params["height"]),
                bit_depth=int(params["bit_depth"]), packing=params.get("packing", "unpacked"),
                header_bytes=int(params.get("header_bytes", 0)),
                stride_bytes=int(params.get("stride_bytes", 0)),
                endian=params.get("endian", "little"),
                data_shift=int(params.get("data_shift", 0)),
                frame_index=int(params.get("frame_index", 0)),
                frame_stride_bytes=int(params.get("frame_stride_bytes", 0))))
        except RawReadError as exc:
            QMessageBox.critical(self, "参考帧读取失败", str(exc))
            return
        self.ref_file_path = path
        self._ref_auto = False
        self.status_label.setText(f"参考帧: {os.path.basename(path)} "
                                  f"({self.ref_raw.shape[1]}x{self.ref_raw.shape[0]})")
        self._update_reference_availability()
        self._update_compare_display()
        self.update_measurements()

    def toggle_compare_mode(self, checked):
        self.compare_enabled = bool(checked)
        self.ref_canvas.setVisible(self.compare_enabled)
        if self.compare_enabled:
            self._update_compare_display()
            self._sync_to_ref(self.canvas.scale, self.canvas.offset)

    def set_compare_mode(self, mode: str):
        if mode not in COMPARE_MODES:
            return
        self.compare_mode = mode
        self._syncing_compare = True
        try:
            if self.compare_combo.currentText() != mode:
                self.compare_combo.setCurrentText(mode)
            action = self.compare_actions.get(mode)
            if action is not None and not action.isChecked():
                action.setChecked(True)
            if not self.compare_enabled:
                self.compare_action.setChecked(True)      # 选模式即打开对比
                self.toggle_compare_mode(True)
        finally:
            self._syncing_compare = False
        self._update_compare_display()

    def _update_compare_display(self):
        if not self.compare_enabled or self.ref_raw is None or self.raw is None:
            return
        if self.ref_raw.shape != self.raw.shape:
            self.ref_canvas.set_display(
                np.zeros((64, 64), np.uint8), None, None, reset_view=False)
            self.status_label.setText(
                f"参考帧尺寸 {self.ref_raw.shape} 与当前帧 {self.raw.shape} 不一致，无法对比")
            return
        ref = self._aligned_ref()
        if self.compare_mode.startswith("差分"):
            diff = np.abs(self.raw.astype(np.int32) - ref.astype(np.int32))
            dp = DisplayParams(bit_depth=int(self.params["bit_depth"]), view="Mono",
                               stretch="Percentile", p_low=0.0, p_high=99.9,
                               colormap="Hot")
            disp, _ = render_display(diff.astype(np.uint16), dp)
            self.ref_canvas.set_display(disp, self.raw, "Mono", reset_view=False)
            res = S.diff_stats(self.raw, ref, int(self.params["bit_depth"]),
                               tol=int(self.params.get("diff_tol", 1)))
            if res:
                self.status_label.setText(
                    f"差分视图(|当前−参考|, Hot 伪彩)  PSNR {res['psnr']:.2f} dB  "
                    f"RMSE {res['rmse']:.3f}  max|Δ| {res['max_abs']:.0f}  "
                    f"差异像素 {res['count_diff']} ({res['pct_diff']:.4f}%)")
        elif self.compare_mode.startswith("混合"):
            if self.display is None:
                return
            ref_disp, _ = render_display(ref, self._display_params())
            if ref_disp.shape == self.display.shape:
                blend = ((self.display.astype(np.uint16) +
                          ref_disp.astype(np.uint16)) // 2).astype(np.uint8)
                self.ref_canvas.set_display(blend, self.raw, self.params["pattern"],
                                            reset_view=False)
        else:
            ref_disp, _ = render_display(ref, self._display_params())
            self.ref_canvas.set_display(ref_disp, ref, self.params["pattern"],
                                        reset_view=False)
        self.ref_canvas.set_overlays(self.canvas.overlays)
        self._sync_to_ref(self.canvas.scale, self.canvas.offset)

    def _on_view_changed(self, scale, offset):
        """任何缩放/平移（滚轮、fit、1:1、坐标跳转）都要刷新状态栏读数。"""
        self._sync_to_ref(scale, offset)
        self._refresh_info_label()

    def _aligned_ref(self):
        """按当前对齐设置返回（可能已平移的）参考帧。"""
        if self.ref_raw is None:
            return None
        if not self.align_reference:
            return self.ref_raw
        if self.ref_shift is None:
            self.ref_shift = accel.phase_correlate(self.raw, self.ref_raw)
        if not self.ref_shift:
            return self.ref_raw
        return accel.shift_image(self.ref_raw, -self.ref_shift["dx"], -self.ref_shift["dy"])

    def _sync_to_ref(self, scale, offset):
        if self.compare_enabled and self.ref_canvas.isVisible():
            self.ref_canvas.set_view_params(scale, offset)

    def _sync_to_main(self, scale, offset):
        if self.compare_enabled and self.ref_canvas.isVisible():
            self.canvas.set_view_params(scale, offset)

    # ==================================================================
    # 测量
    # ==================================================================
    def _on_roi_changed(self, roi):
        self.roi = roi
        if roi is None:
            self.roi_label.setText("ROI: 全图")
        else:
            x0, y0, x1, y1 = roi
            self.roi_label.setText(f"ROI: {x1 - x0}x{y1 - y0} @({x0},{y0})")
        self.update_measurements()

    def update_measurements(self):
        self.measure_timer.start()

    def _do_measure(self):
        if self.raw is None:
            return
        bit_depth = int(self.params["bit_depth"])
        roi = self.roi
        st = S.roi_stats(self.raw, self.params["pattern"], roi, bit_depth)
        self.stats_panel.set_stats(st, bit_depth, roi_only=roi is not None)
        self._last_stats = st
        self.update_histogram()
        rows = S.profile(self.raw, self.params["pattern"], roi, "rows")
        cols = S.profile(self.raw, self.params["pattern"], roi, "cols")
        self.stats_panel.set_profile(rows, cols)
        self._last_profiles = (rows, cols)
        if self.selected_pixel:
            self.update_inspector()

    def update_histogram(self):
        if self.raw is None:
            return
        bit_depth = int(self.params["bit_depth"])
        bins = self.stats_panel.bins_spin.value()
        counts, edges = S.histogram(self.raw, self.params["pattern"], self.roi,
                                    bit_depth, bins=bins)
        chans, _ = S.histogram(self.raw, self.params["pattern"], self.roi, bit_depth,
                               bins=bins, per_channel=True)
        self.stats_panel.set_histogram(counts, edges, chans,
                                       sat_line=(1 << bit_depth) - 1)

    def _on_pixel_clicked(self, x, y):
        self.selected_pixel = (x, y)
        self.goto_box.set_coordinate(x, y)
        self.update_inspector()

    def update_inspector(self):
        if self.raw is None:
            return
        x, y = self.selected_pixel if self.selected_pixel else (None, None)
        if x is None:
            return
        radius = self.stats_panel.inspector_radius_value()
        nb = S.neighborhood(self.raw, x, y, radius, self.params["pattern"])
        self.stats_panel.set_inspector(nb)

    def _on_hover(self, info):
        if not info:
            return
        x, y, val = info["x"], info["y"], info["value"]
        text = f"X:{x}  Y:{y}"
        raw = self.raw
        pattern = self.params["pattern"]
        if raw is not None and val is not None:
            if cfa.is_bayer(pattern) and (x + 1) < raw.shape[1] and (y + 1) < raw.shape[0]:
                x0, y0 = x - (x % 2), y - (y % 2)
                names = cfa.PHASE_LABELS[cfa.normalize_pattern(pattern)]
                parts = []
                for phase, name in enumerate(names):
                    dy, dx = divmod(phase, 2)
                    parts.append(f"{name}={int(raw[y0 + dy, x0 + dx])}")
                text += "  |  " + " ".join(parts) + f"  [{info['channel']}]"
            else:
                text += f"  |  {info['channel']}={val}"
        self.status_label.setText(text)

    def focus_goto(self):
        """Ctrl+G：聚焦坐标输入框（并全选，方便直接粘贴）。"""
        self.goto_box.focus_input()
        self.status_label.setText("输入坐标后回车跳转（支持 1234,567 / (1234, 567) 等格式）")

    def goto_coordinate(self, x: int, y: int, scale=None):
        """跳到指定像素：越界自动夹回画内，选中该点并刷新像素检查器。"""
        if self.raw is None:
            QMessageBox.warning(self, "提示", "请先打开一张 RAW 图")
            return
        h, w = self.raw.shape
        clamped = not (0 <= x < w and 0 <= y < h)
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        if scale is None:
            scale = self.canvas.scale if self.canvas.scale >= 4.0 else 20.0
        self.canvas.center_on(x, y, scale)
        self.canvas.selected_pixel = (x, y)
        self.canvas.update()
        self.goto_box.set_coordinate(x, y)
        self._on_pixel_clicked(x, y)
        self.measure_dock.show()
        self.stats_panel.tabs.setCurrentIndex(2)      # 像素检查器
        value = int(self.raw[y, x])
        channel = cfa.plane_name_at(x, y, self.params["pattern"])
        note = "（输入坐标越界，已夹到画内）" if clamped else ""
        self.status_label.setText(
            f"跳转到 ({x},{y})  {channel} = {value}  zoom {scale:g}×{note}")

    def locate_defect(self, x: int, y: int):
        if self.canvas.scale < 4:
            self.canvas.zoom_to(max(8.0, self.canvas.scale))
        self.canvas.center_on(x, y)
        self.selected_pixel = (x, y)
        self.goto_box.set_coordinate(x, y)
        self.update_inspector()
        self.side_tabs.setCurrentWidget(self.algo_panel)

    # ==================================================================
    # 算法
    # ==================================================================
    def _rebuild_defect_union(self):
        """把各算法累积的缺陷/叠加层合并（同一帧的不同检测结果放在一起看/导出）。"""
        union, overlays = [], []
        for name, items in self.defect_sets.items():
            for d in items:
                d = dict(d)
                d.setdefault("source", name)
                union.append(d)
            overlays.extend(self.overlay_sets.get(name, []))
        self.last_defects = union
        self.stats_panel.set_defects(union)
        self.canvas.set_overlays(overlays)
        if self.compare_enabled:
            self.ref_canvas.set_overlays(overlays)

    def clear_defects(self):
        self.defect_sets.clear()
        self.overlay_sets.clear()
        self.last_defects = []
        self.stats_panel.set_defects([])
        self.canvas.set_overlays([])
        if self.compare_enabled:
            self.ref_canvas.set_overlays([])
        self.status_label.setText("已清空缺陷清单")

    def run_algorithm(self, algo_name: str, params: dict):
        if self.raw is None:
            QMessageBox.warning(self, "提示", "请先打开一张 RAW 图")
            return
        algo = self.algorithm_manager.get_algorithm(algo_name)
        if algo is None:
            return
        roi_only = bool(params.pop("_roi_only", False))
        params["pattern"] = self.params["pattern"]
        params["_bit_depth"] = int(self.params["bit_depth"])
        params["_roi"] = self.roi if (roi_only and self.roi) else None
        params["_reference"] = self.ref_raw

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = algo.run(self.raw, params)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "算法失败", f"{exc}\n\n{traceback.format_exc()}")
            self.algo_panel.set_result("算法失败", error=True)
            return
        QApplication.restoreOverrideCursor()

        defects = result.get("defects", [])
        self.defect_sets[algo_name] = defects
        self.overlay_sets[algo_name] = result.get("overlays", [])
        self._rebuild_defect_union()

        new_image = result.get("image")
        if result.get("corrected") and new_image is not None and new_image is not self.raw:
            self.undo_stack.append((self.raw, algo_name))
            self.raw = new_image
            if self.ref_raw is None:          # 自动把原图设为参考 → 可直接看差分
                self.ref_raw = self.undo_stack[-1][0]
                self.ref_file_path = None
                self._ref_auto = True
                self._update_reference_availability()
            self._render(reset_view=False)
        self.update_measurements()

        msg = result.get("message", "完成")
        self.algo_panel.set_result(msg)
        self.status_label.setText(msg.replace("\n", "  |  "))
        if defects:
            self.measure_dock.show()
            self.stats_panel.tabs.setCurrentIndex(3)

    def undo_correction(self):
        if not self.undo_stack:
            self.status_label.setText("没有可撤销的校正")
            return
        raw, label = self.undo_stack.pop()
        self.raw = raw
        if self._ref_auto:                 # 参考帧是校正时自动设的原图 → 一起撤掉
            self.ref_raw = None
            self.ref_file_path = None
            self._ref_auto = False
            self.align_reference = False
            self.ref_shift = None
            self.align_action.setChecked(False)
            self._update_reference_availability()
            self._update_compare_display()
        self.clear_defects()
        self._render(reset_view=False)
        self.update_measurements()
        self.status_label.setText(f"已撤销「{label}」的校正结果")

    # ==================================================================
    # 浏览
    # ==================================================================
    def _sibling_files(self):
        if not self.current_file_path:
            return []
        folder = os.path.dirname(os.path.abspath(self.current_file_path))
        try:
            names = [f for f in sorted(os.listdir(folder))
                     if f.lower().endswith(RAW_EXTS)]
        except OSError:
            return []
        return [os.path.join(folder, n) for n in names]

    def navigate_file(self, step: int):
        files = self._sibling_files()
        if not files or not self.current_file_path:
            return
        try:
            idx = files.index(os.path.abspath(self.current_file_path))
        except ValueError:
            idx = 0
        idx = (idx + step) % len(files)
        path = files[idx]
        # 布局一致就直接套用当前参数，避免逐个弹窗
        if self._layout_matches(path):
            self.load_image(path, self.params)
            return
        dialog = ImageParamsDialog(self, path=path, initial=self.params)
        if dialog.exec():
            self.load_image(path, dialog.get_params())
            self._add_recent(path)

    def navigate_frame(self, step: int):
        if self.raw is None or not self.current_file_path:
            return
        if not os.path.exists(self.current_file_path):
            self.status_label.setText(f"文件已不存在：{self.current_file_path}")
            return
        spec = self._spec()
        cur = int(self.params.get("frame_index", 0))
        new = max(0, cur + step)
        frame_bytes = spec.frame_stride_bytes or spec.frame_bytes()
        # 帧数要扣掉 header，否则带 header 的多帧文件会报出多余帧数，
        # 翻到不存在的帧才发现（而且 frame_index 已经被改成坏值）
        total = max(1, (os.path.getsize(self.current_file_path) - spec.header_bytes)
                    // max(1, frame_bytes))
        if new >= total:
            self.status_label.setText(f"已是最后一帧（共 {total} 帧）")
            if cur >= total:                       # 之前可能停在越界帧上，纠正回来
                self.params["frame_index"] = total - 1
                self.display_panel.set_params(self.params, block=True)
            return
        prev = cur
        self.params["frame_index"] = new
        self.display_panel.set_params(self.params, block=True)
        self.load_image(self.current_file_path, self.params)
        if int(self.params.get("frame_index", new)) != new:
            self.params["frame_index"] = prev      # 读盘失败 -> 回滚，避免卡在坏帧号
            self.display_panel.set_params(self.params, block=True)
            return
        self.status_label.setText(f"帧 {new + 1}/{total}")

    def _layout_matches(self, path: str) -> bool:
        """文件是否能用当前布局直接读（多帧文件也算匹配）。

        不能拿"整文件大小 == 单帧大小"来判断：多帧 dump 会被误判为不匹配，
        于是每次都弹参数对话框。这里是"文件能容纳整数个当前布局的帧"。
        """
        try:
            size = os.path.getsize(path)
            spec = self._spec()
            frame_bytes = spec.frame_stride_bytes or spec.frame_bytes()
            if frame_bytes <= 0 or size < spec.header_bytes + frame_bytes:
                return False
            return (size - spec.header_bytes) % frame_bytes == 0
        except Exception:
            return False

    def _start_dir(self) -> str:
        if self.current_file_path:
            return os.path.dirname(self.current_file_path)
        return os.getcwd()

    def _add_recent(self, path: str):
        recent = self.settings.value(SETTINGS_KEY_RECENT, []) or []
        if isinstance(recent, str):
            recent = [recent]
        recent = [p for p in recent if p != path]
        recent.insert(0, path)
        self.settings.setValue(SETTINGS_KEY_RECENT, recent[:10])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self.recent_menu.clear()
        recent = self.settings.value(SETTINGS_KEY_RECENT, []) or []
        if isinstance(recent, str):
            recent = [recent]
        if not recent:
            a = QAction("(空)", self)
            a.setEnabled(False)
            self.recent_menu.addAction(a)
            return
        for p in recent[:10]:
            a = QAction(os.path.basename(p), self)
            a.setToolTip(p)
            a.triggered.connect(lambda _c, path=p: self._open_recent(path))
            self.recent_menu.addAction(a)

    def _open_recent(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(self, "提示", f"文件不存在：{path}")
            return
        if self._layout_matches(path):
            self.load_image(path, self.params)
            return
        dialog = ImageParamsDialog(self, path=path, initial=self.params)
        if dialog.exec():
            self.load_image(path, dialog.get_params())

    # ==================================================================
    # 导出
    # ==================================================================
    def _require_image(self) -> bool:
        if self.raw is None:
            QMessageBox.warning(self, "提示", "还没有加载图像")
            return False
        return True

    def export_display_image(self):
        if not self._require_image():
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出显示图", self._default_export_name(".png"),
                                             "PNG (*.png);;BMP (*.bmp)")
        if not path:
            return
        try:
            if self.display.ndim == 3:
                from PyQt6.QtGui import QImage
                arr = np.ascontiguousarray(self.display)
                h, w, _ = arr.shape
                ok = QImage(arr.data, w, h, 3 * w,
                            QImage.Format.Format_RGB888).copy().save(path)
                if not ok:
                    raise OSError(f"Qt 未能写入 {path}（路径/扩展名/权限问题）")
            else:
                X.write_png(path, self.display)
            self.status_label.setText(f"已导出显示图: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_raw_dn(self):
        if not self._require_image():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出原始 DN（无损 16bit）", self._default_export_name("_dn.png"),
            "PNG 16bit (*.png);;TIFF 16bit (*.tif *.tiff)")
        if not path:
            return
        try:
            X.write_image(path, self.raw)
            self.status_label.setText(
                f"已导出原始 DN: {os.path.basename(path)} （{self.raw.dtype}，未做显示变换）")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_annotated(self):
        if not self._require_image():
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出带标注截图",
                                              self._default_export_name("_annotated.png"),
                                              "PNG (*.png)")
        if not path:
            return
        try:
            base = self.display if self.display is not None else self.raw
            annotated = X.draw_defects(base, self.last_defects, radius=6)
            X.write_png(path, annotated)
            self.status_label.setText(
                f"已导出标注截图: {os.path.basename(path)}（含 {len(self.last_defects)} 个缺陷标注）")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_stats_csv(self):
        if not self._require_image():
            return
        st = getattr(self, "_last_stats", None) or \
            S.roi_stats(self.raw, self.params["pattern"], self.roi, int(self.params["bit_depth"]))
        path, _ = QFileDialog.getSaveFileName(self, "导出统计 CSV",
                                             self._default_export_name("_stats.csv"),
                                             "CSV (*.csv)")
        if not path:
            return
        header, rows = X.stats_to_rows(st)
        X.write_csv(path, header, rows)
        self.status_label.setText(f"已导出统计: {os.path.basename(path)}")

    def export_defects_csv(self):
        if not self.last_defects:
            QMessageBox.information(self, "提示", "当前没有缺陷数据（先跑一次检测算法）")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出缺陷清单 CSV",
                                             self._default_export_name("_defects.csv"),
                                             "CSV (*.csv)")
        if not path:
            return
        header, rows = X.defects_to_rows(self.last_defects)
        X.write_csv(path, header, rows)
        self.status_label.setText(
            f"已导出缺陷清单: {os.path.basename(path)}（{len(rows)} 条）")

    def export_report(self):
        if not self._require_image():
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出测量报告",
                                             self._default_export_name("_report.md"),
                                             "Markdown (*.md);;Text (*.txt)")
        if not path:
            return
        sections = []
        spec = self._spec()
        sections.append(("文件 / Frame", "\n".join([
            f"- 文件: `{self.current_file_path}`",
            f"- 布局: {spec.describe()}",
            f"- 帧号: {spec.frame_index}",
            f"- 显示: {self.render_info.describe() if self.render_info else '-'}",
            f"- ROI: {self.roi if self.roi else '全图'}",
        ])))
        st = getattr(self, "_last_stats", None)
        if st:
            lines = ["| channel | count | mean | std | min | max | median | p01 | p99 | sat | zero | snr_dB |",
                     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in st.get("channels", []):
                lines.append("| " + " | ".join(str(c.get(k, "")) for k in
                             ("name", "count", "mean", "std", "min", "max", "median",
                              "p01", "p99", "sat", "zero", "snr_db")) + " |")
            overall = st.get("overall", {})
            lines.append(f"\nALL: mean={overall.get('mean')} std={overall.get('std')} "
                         f"min={overall.get('min')} max={overall.get('max')}")
            sections.append(("ROI 统计", "\n".join(lines)))
        if getattr(self, "_last_profiles", None):
            rows, cols = self._last_profiles
            r_mean = np.asarray(rows.get("mean", []))
            c_mean = np.asarray(cols.get("mean", []))
            sections.append(("Profile 摘要", "\n".join([
                f"- 行均值: {r_mean.min():.2f} ~ {r_mean.max():.2f} DN "
                f"(峰谷差 {r_mean.max() - r_mean.min():.2f})",
                f"- 列均值: {c_mean.min():.2f} ~ {c_mean.max():.2f} DN "
                f"(峰谷差 {c_mean.max() - c_mean.min():.2f})",
            ])))
        if self.ref_raw is not None and self.ref_raw.shape == self.raw.shape:
            res = S.diff_stats(self.raw, self.ref_raw, int(self.params["bit_depth"]))
            sections.append(("与参考帧差分", "\n".join([
                f"- PSNR {res['psnr']:.3f} dB",
                f"- RMSE {res['rmse']:.4f} DN",
                f"- max|Δ| {res['max_abs']:.0f} DN",
                f"- 差异像素 {res['count_diff']} ({res['pct_diff']:.4f}%)",
            ])))
        if self.last_defects:
            counts = {}
            for d in self.last_defects:
                counts[d.get("type", "?")] = counts.get(d.get("type", "?"), 0) + 1
            lines = ["| 类型 | 数量 |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
            lines.append(f"\n合计 {len(self.last_defects)} 条缺陷，"
                         f"完整清单见导出的 defects CSV。")
            sections.append(("缺陷汇总", "\n".join(lines)))
        algo_msg = self.algo_panel.result_label.text()
        if algo_msg:
            sections.append(("最近一次算法输出", "```\n" + algo_msg + "\n```"))
        body = ["# RAW 测量报告", ""]
        for title, text in sections:
            body += [f"## {title}", "", text, ""]
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(body))
            self.status_label.setText(f"已导出报告: {os.path.basename(path)}")
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _default_export_name(self, suffix: str) -> str:
        base = os.path.splitext(os.path.basename(self.current_file_path or "frame"))[0]
        folder = os.path.dirname(self.current_file_path or os.getcwd())
        return os.path.join(folder, base + suffix)

    # ==================================================================
    # 拖放 / 设置 / 杂项
    # ==================================================================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            self._drop_event(event)
        except Exception as exc:                           # noqa: BLE001
            traceback.print_exc()
            self.status_label.setText(f"拖放失败：{exc}")

    def _drop_event(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return
        same = self._layout_matches(path)
        if same and self.raw is not None:
            self.load_image(path, self.params)
            self._add_recent(path)
        else:
            dialog = ImageParamsDialog(self, path=path, initial=self.params)
            if dialog.exec():
                self.load_image(path, dialog.get_params())
                self._add_recent(path)

    def _refresh_info_label(self):
        if self.render_info is None:
            self.info_label.setText("")
            return
        scale = self.canvas.scale
        zoom = f"{scale:g}×" if scale >= 1.0 else f"{scale * 100:.0f}%"
        ref = ""
        if self.ref_raw is not None:
            name = os.path.basename(self.ref_file_path) if self.ref_file_path else "原图(自动)"
            ref = f"   ref:{name}"
            if self.align_reference and self.ref_shift:
                ref += f" 对齐({self.ref_shift['dx']:+.1f},{self.ref_shift['dy']:+.1f})"
        self.info_label.setText(
            f"zoom {zoom}   level {self.render_info.lo:.0f}~{self.render_info.hi:.0f} DN   "
            f"{self.render_info.view}{ref}")

    def _restore_settings(self):
        self._state_restored = False
        geo = self.settings.value(SETTINGS_KEY_GEOMETRY)
        if geo:
            self.restoreGeometry(geo)
        state = self.settings.value(SETTINGS_KEY_STATE)
        if state:
            # 没有 objectName 时 Qt 无法匹配停靠窗口，restoreState 会失败
            self._state_restored = bool(self.restoreState(state))
        import json
        last = self.settings.value(SETTINGS_KEY_PARAMS, "")
        if last:
            try:
                self._merge_params(json.loads(last))
            except Exception:
                pass

    def _save_settings(self):
        """记忆窗口几何/停靠布局/上次参数（"下次打开还是这个布局"）。"""
        import json
        self.settings.setValue(SETTINGS_KEY_GEOMETRY, self.saveGeometry())
        self.settings.setValue(SETTINGS_KEY_STATE, self.saveState())
        self.settings.setValue(SETTINGS_KEY_PARAMS, json.dumps(self.params))

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def show_about(self):
        QMessageBox.information(self, "RAW Viewer — 快捷键", "\n".join([
            "Ctrl+O 打开 RAW     Ctrl+R 打开参考帧",
            "Ctrl+E 导出显示图   Ctrl+Shift+E 导出原始 DN (16bit)",
            "Ctrl+D 对比模式     Ctrl+L 自动电平",
            "Ctrl+A ROI=全图     Esc 清除 ROI     Ctrl+Z 撤销校正",
            "Ctrl+G 输入坐标跳转（支持 1234,567 / 1-based 坐标）",
            "F 适应窗口          1 1:1            +/- 缩放",
            "G 像素网格/数值     C 十字线",
            "PgUp/PgDn 上一个/下一个文件     ,/. 上一帧/下一帧",
            "",
            "鼠标: 左键拖拽=平移；开启 ROI 模式或右键拖拽=框选 ROI；单击=选点检查",
            "滚轮=以指针为中心缩放（放大时像素对齐到整数物理像素）",
            "",
            accel.backend_info(),
            "装了 opencv-python-headless 会自动启用加速（demosaic/连通域/缩小/CLAHE）；",
            "没装也能用，只是慢一些。",
        ]))

