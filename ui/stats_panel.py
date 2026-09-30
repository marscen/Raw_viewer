"""测量面板：统计 / Profile / 像素检查器 / 缺陷清单。

四页签对应测试现场的四个高频动作：
  1. Statistics —— ROI（或全图）的分通道 mean/std/min/max/百分位/饱和计数 + 直方图；
  2. Profiles   —— ROI 的行/列均值曲线，一眼看出行 FPN、坏行、阴影；
  3. Inspector  —— 点选像素周围 (2r+1)^2 的原始 DN 与通道归属，定位坏点时用；
  4. Defects    —— 算法检出的缺陷清单，可跳转定位、导出 CSV 交给 yield 分析。
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QSpinBox,
                             QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ui.qtutil import mono_font, mono_style
from ui.plots import HistogramPlot, ProfilePlot, color_for

STAT_COLUMNS = ["channel", "count", "mean", "std", "min", "max", "median",
                "p01", "p99", "sat", "zero", "snr_dB"]

DEFECT_COLUMNS = ["type", "x", "y", "channel", "value", "delta", "note", "source"]


def _fmt(v, digits=3):
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == float("inf"):
            return "inf"
        return f"{v:.{digits}f}"
    return "" if v is None else str(v)


def _make_table(columns) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.setAlternatingRowColors(True)
    return table


class StatsPanel(QWidget):
    defect_activated = pyqtSignal(int, int)
    export_defects_requested = pyqtSignal()
    export_stats_requested = pyqtSignal()
    histogram_bins_changed = pyqtSignal(int)
    clear_defects_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_stats_tab()
        self._build_profile_tab()
        self._rows_profile = None
        self._cols_profile = None
        self._build_inspector_tab()
        self._build_defect_tab()
        self._defects = []

    # ------------------------------------------------------------------
    def _build_stats_tab(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(4, 4, 4, 4)
        self.stats_info = QLabel("未加载图像")
        self.stats_info.setWordWrap(True)
        self.stats_info.setStyleSheet(mono_style())
        lay.addWidget(self.stats_info)

        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #ffb84d;")
        lay.addWidget(self.warn_label)

        self.stats_table = _make_table(STAT_COLUMNS)
        self.stats_table.setMaximumHeight(170)
        lay.addWidget(self.stats_table)

        ctrl = QHBoxLayout()
        self.hist_log_check = QCheckBox("对数纵轴")
        self.hist_log_check.setChecked(True)
        self.hist_channels_check = QCheckBox("分通道曲线")
        self.hist_channels_check.setChecked(True)
        self.bins_spin = QSpinBox(); self.bins_spin.setRange(16, 4096)
        self.bins_spin.setValue(256); self.bins_spin.setSingleStep(16)
        ctrl.addWidget(self.hist_log_check)
        ctrl.addWidget(self.hist_channels_check)
        ctrl.addWidget(QLabel("bins:"))
        ctrl.addWidget(self.bins_spin)
        ctrl.addStretch()
        self.export_stats_btn = QPushButton("导出统计 CSV")
        self.export_stats_btn.clicked.connect(self.export_stats_requested.emit)
        ctrl.addWidget(self.export_stats_btn)
        lay.addLayout(ctrl)

        self.hist_plot = HistogramPlot()
        lay.addWidget(self.hist_plot, 1)

        self.hist_log_check.toggled.connect(self.hist_plot.set_log_scale)
        self.hist_channels_check.toggled.connect(self.hist_plot.set_show_channels)
        self.bins_spin.valueChanged.connect(self.histogram_bins_changed.emit)
        self.tabs.addTab(page, "统计 Statistics")

    def _build_profile_tab(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(4, 4, 4, 4)
        head = QHBoxLayout()
        self.profile_info = QLabel("未计算")
        self.profile_info.setStyleSheet(mono_style())
        head.addWidget(self.profile_info, 1)
        self.profile_band_check = QCheckBox("±3σ 包络")
        self.profile_band_check.setChecked(True)
        self.profile_band_check.toggled.connect(self._refresh_band)
        head.addWidget(self.profile_band_check)
        lay.addLayout(head)

        lay.addWidget(QLabel("行 profile（每行均值，找行 FPN / 坏行）:"))
        self.row_plot = ProfilePlot()
        lay.addWidget(self.row_plot, 1)
        lay.addWidget(QLabel("列 profile（每列均值，找列 FPN / 坏列）:"))
        self.col_plot = ProfilePlot()
        lay.addWidget(self.col_plot, 1)
        self.tabs.addTab(page, "Profile")

    def _build_inspector_tab(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(4, 4, 4, 4)
        head = QHBoxLayout()
        self.inspector_info = QLabel("在图上单击像素以检查邻域")
        self.inspector_info.setWordWrap(True)
        self.inspector_info.setStyleSheet(mono_style())
        head.addWidget(self.inspector_info, 1)
        head.addWidget(QLabel("半径:"))
        self.inspector_radius = QSpinBox(); self.inspector_radius.setRange(1, 8)
        self.inspector_radius.setValue(2)
        head.addWidget(self.inspector_radius)
        lay.addLayout(head)
        self.inspector_table = _make_table(["邻域值"])
        self.inspector_table.setFont(mono_font(11))
        lay.addWidget(self.inspector_table, 1)
        self.inspector_note = QLabel("")
        self.inspector_note.setWordWrap(True)
        self.inspector_note.setStyleSheet(mono_style("color: #9ad;"))
        lay.addWidget(self.inspector_note)
        self.tabs.addTab(page, "像素检查 Inspector")

    def _build_defect_tab(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(4, 4, 4, 4)
        self.defect_info = QLabel("尚无缺陷结果（在算法页运行坏点/坏线检测）")
        self.defect_info.setWordWrap(True)
        self.defect_info.setStyleSheet(mono_style())
        lay.addWidget(self.defect_info)

        self.defect_table = _make_table(DEFECT_COLUMNS)
        self.defect_table.itemDoubleClicked.connect(self._on_defect_double_clicked)
        lay.addWidget(self.defect_table, 1)

        ctrl = QHBoxLayout()
        self.locate_btn = QPushButton("定位选中")
        self.locate_btn.clicked.connect(self._locate_selected)
        self.export_defect_btn = QPushButton("导出缺陷 CSV")
        self.export_defect_btn.clicked.connect(self.export_defects_requested.emit)
        self.clear_defect_btn = QPushButton("清空")
        self.clear_defect_btn.clicked.connect(self.clear_defects_requested.emit)
        ctrl.addWidget(self.locate_btn)
        ctrl.addWidget(self.export_defect_btn)
        ctrl.addWidget(self.clear_defect_btn)
        ctrl.addStretch()
        lay.addLayout(ctrl)
        self.tabs.addTab(page, "缺陷 Defects")

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------
    def set_stats(self, stats: dict, bit_depth: int = 10, roi_only: bool = False):
        if not stats:
            self.stats_table.setRowCount(0)
            self.stats_info.setText("未加载图像")
            return
        x0, y0, x1, y1 = stats.get("roi", (0, 0, 0, 0))
        w, h = stats.get("size", (x1 - x0, y1 - y0))
        overall = stats.get("overall", {})
        max_code = (1 << int(bit_depth)) - 1
        self.stats_info.setText(
            f"ROI: ({x0},{y0})-({x1},{y1})  {w}x{h} px   "
            f"{'局部' if roi_only else '全图'}   位深 {bit_depth}bit (max {max_code})\n"
            f"ALL   mean={_fmt(overall.get('mean'), 2)}  std={_fmt(overall.get('std'), 2)}  "
            f"min={_fmt(overall.get('min'), 0)}  max={_fmt(overall.get('max'), 0)}  "
            f"median={_fmt(overall.get('median'), 1)}")

        rows = stats.get("channels", [])
        self.stats_table.setRowCount(len(rows))
        # 表头名 -> ChannelStat 的字段名
        keymap = {"channel": "name", "snr_dB": "snr_db"}
        for r, st in enumerate(rows):
            for c, key in enumerate(STAT_COLUMNS):
                val = st.get(keymap.get(key, key), "")
                item = QTableWidgetItem(_fmt(val, 3) if key not in ("channel", "count", "sat", "zero") else _fmt(val, 0))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if key == "channel":
                    item.setForeground(QBrush(color_for(str(val))))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.stats_table.setItem(r, c, item)

        # 数据超出位深（位深设小了 / 16bit 容器左对齐忘了 data_shift）：
        # 这时显示会被 LUT 截白、直方图也怪，必须先提示去查设置
        msgs = []
        overall_max = float(overall.get("max") or 0.0)
        if overall_max > max_code:
            msgs.append(f"⚠ 数据最大值 {overall_max:.0f} 超过 {bit_depth}bit 上限 "
                        f"({max_code})：请检查位深 / 左对齐位移(data_shift)设置")
        sat_total = sum(int(st.get("sat") or 0) for st in rows)
        zero_total = sum(int(st.get("zero") or 0) for st in rows)
        total = max(1, sum(int(st.get("count") or 0) for st in rows))
        if sat_total:
            msgs.append(f"⚠ 饱和像素 {sat_total} ({100.0 * sat_total / total:.3f}%)，可能过曝")
        if zero_total:
            msgs.append(f"⚠ 0 DN 像素 {zero_total} ({100.0 * zero_total / total:.3f}%)，注意死点/黑电平")
        means = {st.get("name"): float(st["mean"]) for st in rows if st.get("mean") != ""}
        if "Gr" in means and "Gb" in means and means["Gr"] > 0:
            diff_pct = abs(means["Gr"] - means["Gb"]) / means["Gr"] * 100.0
            msgs.append(f"Gr/Gb 均值差 {diff_pct:.2f}%（绿通道失配，>2% 需关注）")
        self.warn_label.setText("\n".join(msgs))

    def set_histogram(self, counts, edges, channels=None, sat_line=None):
        self.hist_plot.set_data(counts, edges, channels, sat_line)

    def set_profile(self, rows_profile: dict, cols_profile: dict):
        self._rows_profile, self._cols_profile = rows_profile, cols_profile
        self._apply_profiles()

    def _apply_profiles(self):
        for plot, prof in ((self.row_plot, self._rows_profile),
                           (self.col_plot, self._cols_profile)):
            if not prof:
                plot.set_data(None, None)
                continue
            std = prof.get("std") if self.profile_band_check.isChecked() else None
            plot.set_data(prof.get("index"), prof.get("mean"), std,
                          prof.get("channels"), prof.get("axis", "rows"))
        rows_profile, cols_profile = self._rows_profile, self._cols_profile
        if rows_profile:
            r_mean = np.asarray(rows_profile.get("mean", []))
            r_std = np.asarray(rows_profile.get("std", []))
            c_std = np.asarray(cols_profile.get("std", [])) if cols_profile else np.array([0.0])
            self.profile_info.setText(
                f"行均值 {r_mean.min():.1f}~{r_mean.max():.1f} (std {r_std.max():.2f})   "
                f"列 std max {c_std.max():.2f}")

    def _refresh_band(self):
        self._apply_profiles()

    def set_inspector(self, nb: dict):
        if not nb or "values" not in nb or getattr(nb["values"], "size", 0) == 0:
            self.inspector_table.setRowCount(0)
            self.inspector_info.setText("在图上单击像素以检查邻域")
            self.inspector_note.setText("")
            return
        vals = nb["values"]
        chans = nb["channels"]
        h, w = vals.shape
        self.inspector_table.setRowCount(h)
        self.inspector_table.setColumnCount(w)
        self.inspector_table.setHorizontalHeaderLabels([str(nb["x0"] + i) for i in range(w)])
        self.inspector_table.setVerticalHeaderLabels([str(nb["y0"] + i) for i in range(h)])
        max_code = max(1, int(np.max(vals)))
        for r in range(h):
            for c in range(w):
                ch = str(chans[r, c])
                val = int(vals[r, c])
                item = QTableWidgetItem(f"{val}")
                color = color_for(ch)
                # 亮暗背景自动反差：高 DN 用亮底黑字
                if val > 0.6 * max_code:
                    item.setBackground(QBrush(color.lighter(160)))
                    item.setForeground(QBrush(QColor(20, 20, 20)))
                else:
                    item.setForeground(QBrush(color))
                item.setToolTip(f"{ch} @ ({nb['x0'] + c}, {nb['y0'] + r}) = {val}")
                if (nb["x0"] + c, nb["y0"] + r) == nb["center"]:
                    item.setBackground(QBrush(QColor(90, 90, 40)))
                    item.setText(f"[{val}]")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.inspector_table.setItem(r, c, item)

        cx, cy = nb["center"]
        self.inspector_info.setText(
            f"中心 ({cx},{cy})  {nb['center_channel']} = {nb['center_value']}   "
            f"邻域 {w}x{h}")
        cvals = nb.get("channel_values", {})
        self.inspector_note.setText(
            "邻域通道和: " + "  ".join(f"{k}={v}" for k, v in cvals.items()))

    def set_defects(self, defects):
        self._defects = list(defects or [])
        self.defect_table.setRowCount(len(self._defects))
        counts = {}
        for r, d in enumerate(self._defects):
            counts[d.get("type", "?")] = counts.get(d.get("type", "?"), 0) + 1
            for c, key in enumerate(DEFECT_COLUMNS):
                val = d.get(key, "")
                item = QTableWidgetItem(_fmt(val, 2) if isinstance(val, float) else str(val))
                if key == "type":
                    item.setForeground(QBrush(color_for(
                        {"hot": "R", "dead": "B", "cluster": "Gr"}.get(str(val), "ALL"))))
                self.defect_table.setItem(r, c, item)
        if self._defects:
            summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            sources = sorted({str(d.get("source", "")) for d in self._defects if d.get("source")})
            self.defect_info.setText(
                f"共 {len(self._defects)} 个缺陷    {summary}\n"
                f"来源: {', '.join(sources) if sources else '-'}"
                f"    （双击行定位到画布中心；多次运行的缺陷会累积）")
        else:
            self.defect_info.setText("尚无缺陷结果（在算法页运行坏点/坏线检测）")

    def get_defects(self):
        return self._defects

    def selected_defect(self):
        row = self.defect_table.currentRow()
        if 0 <= row < len(self._defects):
            return self._defects[row]
        return None

    def _locate_selected(self):
        d = self.selected_defect()
        if d and d.get("x") is not None and d.get("y") is not None:
            self.defect_activated.emit(int(d["x"]), int(d["y"]))

    def _on_defect_double_clicked(self, item):
        self._locate_selected()

    def inspector_radius_value(self) -> int:
        return self.inspector_radius.value()
