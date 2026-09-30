"""侧栏控件：

  * DisplayControlPanel —— 布局参数（几何/位深/packing/CFA）+ 显示参数
    （视图、拉伸、gamma、伪彩、黑/白电平）。
    两类改动成本不同：几何/位深/packing 必须重新读盘（reload_requested），
    视图/电平只要重新渲染（render_requested），所以分成两个信号。
  * AlgorithmPanel —— 算法选择与参数，额外提供"只在 ROI 内运行"开关。
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QSpinBox, QVBoxLayout, QWidget)

from ui.qtutil import mono_style
from utils import accel, display as D

LAYOUT_KEYS = ("width", "height", "bit_depth", "packing", "header_bytes",
               "stride_bytes", "endian", "data_shift", "frame_index",
               "frame_stride_bytes")


class DisplayControlPanel(QWidget):
    reload_requested = pyqtSignal(dict)     # 需要重新读盘（几何/位深/packing）
    render_requested = pyqtSignal(dict)     # 只需重新渲染（视图/电平/gamma）
    auto_levels_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ---------------- 布局 ----------------
        layout_group = QGroupBox("布局 / Layout")
        lform = QFormLayout(layout_group)
        self.width_spin = QSpinBox(); self.width_spin.setRange(1, 100000); self.width_spin.setSingleStep(2)
        self.height_spin = QSpinBox(); self.height_spin.setRange(1, 100000); self.height_spin.setSingleStep(2)
        self.bit_depth_combo = QComboBox(); self.bit_depth_combo.addItems(["8", "10", "12", "14", "16"])
        self.packing_combo = QComboBox(); self.packing_combo.addItems(["unpacked", "packed (MIPI)"])
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Mono/None", "RGGB", "BGGR", "GRBG", "GBRG"])
        lform.addRow("Width:", self.width_spin)
        lform.addRow("Height:", self.height_spin)
        lform.addRow("Bit Depth:", self.bit_depth_combo)
        lform.addRow("Packing:", self.packing_combo)
        lform.addRow("CFA Pattern:", self.pattern_combo)
        layout.addWidget(layout_group)

        # ---------------- 显示 ----------------
        view_group = QGroupBox("显示 / Display")
        vform = QFormLayout(view_group)
        self.view_combo = QComboBox(); self.view_combo.addItems(D.VIEW_MODES)
        self.colormap_combo = QComboBox(); self.colormap_combo.addItems(D.COLORMAPS)
        self.stretch_combo = QComboBox(); self.stretch_combo.addItems(D.STRETCH_MODES)
        self.stretch_combo.setCurrentText("Percentile")
        self.p_low_spin = QDoubleSpinBox(); self.p_low_spin.setRange(0.0, 49.0)
        self.p_low_spin.setValue(0.5); self.p_low_spin.setDecimals(2); self.p_low_spin.setSingleStep(0.1)
        self.p_high_spin = QDoubleSpinBox(); self.p_high_spin.setRange(51.0, 100.0)
        self.p_high_spin.setValue(99.5); self.p_high_spin.setDecimals(2); self.p_high_spin.setSingleStep(0.1)
        self.sigma_spin = QDoubleSpinBox(); self.sigma_spin.setRange(0.5, 20.0)
        self.sigma_spin.setValue(3.0); self.sigma_spin.setSingleStep(0.5)
        self.bit_depth_combo.currentTextChanged.connect(self._sync_white_default)
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0.05, 8.0)
        self.gamma_spin.setValue(1.0); self.gamma_spin.setSingleStep(0.05); self.gamma_spin.setDecimals(2)
        self.invert_check = QCheckBox("反相 (invert)")
        self.black_spin = QDoubleSpinBox(); self.black_spin.setRange(-65535, 65535)
        self.black_spin.setDecimals(0); self.black_spin.setSingleStep(1)
        self.white_spin = QDoubleSpinBox(); self.white_spin.setRange(0, 1 << 20)
        self.white_spin.setDecimals(0); self.white_spin.setSingleStep(1)
        self.white_spin.setToolTip("0 = 自动使用 (2^bit_depth - 1)")
        self.roi_stretch_check = QCheckBox("电平按 ROI 计算")
        # 局部对比度增强（CLAHE）：暗场/低对比画面里看结构用
        self.clahe_check = QCheckBox("局部对比度 CLAHE")
        self.clahe_check.setToolTip("分块自适应直方图均衡，暗场/低对比时能看清结构；"
                                    "有 OpenCV 时用它，否则退化为全局均衡")
        self.clahe_clip = QDoubleSpinBox(); self.clahe_clip.setRange(0.5, 10.0)
        self.clahe_clip.setValue(2.0); self.clahe_clip.setSingleStep(0.5)
        self.clahe_clip.setToolTip("对比度限制：越大越激进（噪声也会被放大）")
        self.clahe_tiles = QSpinBox(); self.clahe_tiles.setRange(2, 32)
        self.clahe_tiles.setValue(8)
        self.clahe_tiles.setToolTip("分块数：越大越局部（也越慢）")
        vform.addRow("视图:", self.view_combo)
        vform.addRow("拉伸:", self.stretch_combo)
        vform.addRow("百分位 low/high:", self._pair(self.p_low_spin, self.p_high_spin))
        vform.addRow("Sigma k:", self.sigma_spin)
        vform.addRow("黑电平:", self.black_spin)
        vform.addRow("白电平:", self.white_spin)
        vform.addRow("Gamma:", self.gamma_spin)
        vform.addRow("伪彩:", self.colormap_combo)
        vform.addRow("", self.invert_check)
        vform.addRow("", self.roi_stretch_check)
        vform.addRow("", self.clahe_check)
        vform.addRow("CLAHE clip/tiles:", self._pair(self.clahe_clip, self.clahe_tiles))
        layout.addWidget(view_group)

        btns = QHBoxLayout()
        self.auto_btn = QPushButton("自动电平 (Auto)")
        self.auto_btn.clicked.connect(self.auto_levels_requested.emit)
        self.reset_btn = QPushButton("重置显示")
        self.reset_btn.clicked.connect(self.reset_display)
        btns.addWidget(self.auto_btn)
        btns.addWidget(self.reset_btn)
        layout.addLayout(btns)

        self.level_label = QLabel("level: -")
        self.level_label.setStyleSheet(mono_style("color: #9ad;"))
        layout.addWidget(self.level_label)
        self.backend_label = QLabel(accel.backend_info())
        self.backend_label.setStyleSheet(mono_style("color: #8a8;"))
        self.backend_label.setWordWrap(True)
        layout.addWidget(self.backend_label)
        layout.addStretch()

        # 信号
        for w in (self.width_spin, self.height_spin):
            w.valueChanged.connect(self._emit_reload)
        for c in (self.bit_depth_combo, self.packing_combo):
            c.currentTextChanged.connect(self._emit_reload)
        for c in (self.pattern_combo, self.view_combo, self.colormap_combo, self.stretch_combo):
            c.currentTextChanged.connect(self._emit_render)
        for w in (self.p_low_spin, self.p_high_spin, self.sigma_spin, self.gamma_spin,
                  self.black_spin, self.white_spin):
            w.valueChanged.connect(self._emit_render)
        self.invert_check.toggled.connect(self._emit_render)
        self.roi_stretch_check.toggled.connect(self._emit_render)
        self.clahe_check.toggled.connect(self._emit_render)
        self.clahe_clip.valueChanged.connect(self._emit_render)
        self.clahe_tiles.valueChanged.connect(self._emit_render)

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        box = QWidget(); lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(a); lay.addWidget(b)
        return box

    def _sync_white_default(self):
        if self.white_spin.value() == 0:
            self._emit_render()

    # ---------------- 参数读写 ----------------
    def get_params(self) -> dict:
        return {
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "bit_depth": int(self.bit_depth_combo.currentText()),
            "packing": "packed" if self.packing_combo.currentText().startswith("packed") else "unpacked",
            "pattern": self.pattern_combo.currentText(),
            "view": self.view_combo.currentText(),
            "colormap": self.colormap_combo.currentText(),
            "stretch": self.stretch_combo.currentText(),
            "p_low": self.p_low_spin.value(),
            "p_high": self.p_high_spin.value(),
            "sigma_k": self.sigma_spin.value(),
            "gamma": self.gamma_spin.value(),
            "invert": self.invert_check.isChecked(),
            "black_level": self.black_spin.value(),
            "white_level": self.white_spin.value(),
            "stretch_on_roi": self.roi_stretch_check.isChecked(),
            "clahe_enable": self.clahe_check.isChecked(),
            "clahe_clip": self.clahe_clip.value(),
            "clahe_tiles": self.clahe_tiles.value(),
        }

    def set_params(self, params: dict, block: bool = True):
        """把参数填进控件（载入新文件时调用），不触发信号。"""
        if block:
            self.blockSignals(True)
            for w in self.findChildren(QWidget):
                w.blockSignals(True)
        try:
            if "width" in params: self.width_spin.setValue(int(params["width"]))
            if "height" in params: self.height_spin.setValue(int(params["height"]))
            if "bit_depth" in params: self.bit_depth_combo.setCurrentText(str(params["bit_depth"]))
            if "packing" in params:
                self.packing_combo.setCurrentText("packed (MIPI)" if params["packing"] == "packed" else "unpacked")
            if "pattern" in params: self.pattern_combo.setCurrentText(params["pattern"])
            if "view" in params: self.view_combo.setCurrentText(params["view"])
            if "colormap" in params: self.colormap_combo.setCurrentText(params["colormap"])
            if "stretch" in params: self.stretch_combo.setCurrentText(params["stretch"])
            if "p_low" in params: self.p_low_spin.setValue(float(params["p_low"]))
            if "p_high" in params: self.p_high_spin.setValue(float(params["p_high"]))
            if "sigma_k" in params: self.sigma_spin.setValue(float(params["sigma_k"]))
            if "gamma" in params: self.gamma_spin.setValue(float(params["gamma"]))
            if "invert" in params: self.invert_check.setChecked(bool(params["invert"]))
            if "black_level" in params: self.black_spin.setValue(float(params["black_level"]))
            if "white_level" in params: self.white_spin.setValue(float(params["white_level"]))
            if "stretch_on_roi" in params: self.roi_stretch_check.setChecked(bool(params["stretch_on_roi"]))
            if "clahe_enable" in params: self.clahe_check.setChecked(bool(params["clahe_enable"]))
            if "clahe_clip" in params: self.clahe_clip.setValue(float(params["clahe_clip"]))
            if "clahe_tiles" in params: self.clahe_tiles.setValue(int(params["clahe_tiles"]))
        finally:
            if block:
                for w in self.findChildren(QWidget):
                    w.blockSignals(False)
                self.blockSignals(False)

    def set_level_info(self, text: str):
        self.level_label.setText(text)

    def reset_display(self):
        self.set_params({"view": "Mono", "colormap": "Gray", "stretch": "Percentile",
                         "p_low": 0.5, "p_high": 99.5, "sigma_k": 3.0, "gamma": 1.0,
                         "invert": False, "black_level": 0, "white_level": 0,
                         "stretch_on_roi": False}, block=True)
        self._emit_render()

    def _emit_reload(self):
        self.reload_requested.emit(self.get_params())

    def _emit_render(self):
        self.render_requested.emit(self.get_params())


class AlgorithmPanel(QWidget):
    run_algorithm = pyqtSignal(str, dict)      # algorithm_name, params
    request_stats_roi = pyqtSignal()           # 请求在 ROI 内重算统计

    def __init__(self, algorithm_manager, parent=None):
        super().__init__(parent)
        self.algorithm_manager = algorithm_manager
        self.current_algo_name = None
        self.param_inputs = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        group = QGroupBox("算法 / Algorithms")
        algo_layout = QVBoxLayout(group)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self.algorithm_manager.get_algorithm_names())
        self.algo_combo.currentTextChanged.connect(self.on_algo_changed)
        algo_layout.addWidget(QLabel("选择算法:"))
        algo_layout.addWidget(self.algo_combo)
        self.desc_label = QLabel("")
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #9aa; font-style: italic;")
        algo_layout.addWidget(self.desc_label)
        layout.addWidget(group)

        self.params_group = QGroupBox("参数 / Parameters")
        self.params_layout = QFormLayout(self.params_group)
        layout.addWidget(self.params_group)

        self.roi_only_check = QCheckBox("只在 ROI 内运行")
        self.roi_only_check.setToolTip("先在图上框选 ROI，再勾选此项可只分析局部（例如局部缺陷扫描）")
        layout.addWidget(self.roi_only_check)

        self.run_btn = QPushButton("运行算法 (Run)")
        self.run_btn.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_btn)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #7ddc7d;")
        layout.addWidget(self.result_label)
        layout.addStretch()

        if self.algo_combo.count() > 0:
            self.on_algo_changed(self.algo_combo.currentText())

    def on_algo_changed(self, name):
        self.current_algo_name = name
        algo = self.algorithm_manager.get_algorithm(name)
        if not algo:
            return
        self.desc_label.setText(algo.description)
        self.rebuild_params_ui(algo.get_parameters())

    def rebuild_params_ui(self, params_spec):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.param_inputs.clear()

        for key, spec in (params_spec or {}).items():
            label_text = spec.get("label", key)
            param_type = spec.get("type", "str")
            default_val = spec.get("default")
            if param_type == "int":
                widget = QSpinBox(); widget.setRange(int(spec.get("min", -99999)), int(spec.get("max", 99999)))
                widget.setValue(int(default_val or 0))
            elif param_type == "float":
                widget = QDoubleSpinBox(); widget.setRange(float(spec.get("min", -99999.0)), float(spec.get("max", 99999.0)))
                widget.setDecimals(int(spec.get("decimals", 2)))
                widget.setValue(float(default_val or 0.0))
            elif param_type == "bool":
                widget = QCheckBox(); widget.setChecked(bool(default_val))
            elif param_type == "list":
                widget = QComboBox(); widget.addItems([str(o) for o in spec.get("options", [])])
                widget.setCurrentText(str(default_val))
            else:
                from PyQt6.QtWidgets import QLineEdit
                widget = QLineEdit(str(default_val if default_val is not None else ""))
            if spec.get("tooltip"):
                widget.setToolTip(spec["tooltip"])
            self.params_layout.addRow(label_text, widget)
            self.param_inputs[key] = (widget, param_type)

    def collect_params(self) -> dict:
        params = {}
        for key, (widget, param_type) in self.param_inputs.items():
            if param_type in ("int", "float"):
                params[key] = widget.value()
            elif param_type == "bool":
                params[key] = widget.isChecked()
            elif param_type == "list":
                params[key] = widget.currentText()
            else:
                params[key] = widget.text()
        return params

    def on_run_clicked(self):
        if not self.current_algo_name:
            return
        params = self.collect_params()
        params["_roi_only"] = self.roi_only_check.isChecked()
        self.run_algorithm.emit(self.current_algo_name, params)

    def set_result(self, text: str, error: bool = False):
        self.result_label.setStyleSheet("color: #ff8080;" if error else "color: #7ddc7d;")
        self.result_label.setText(text)
