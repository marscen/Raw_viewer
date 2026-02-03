from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QSpinBox, 
                             QComboBox, QGroupBox, QLabel, QPushButton, QCheckBox, 
                             QDoubleSpinBox, QLineEdit)
from PyQt6.QtCore import pyqtSignal

class ImageControlPanel(QWidget):
    params_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        # Group Box
        group = QGroupBox("Image Parameters")
        form_layout = QFormLayout()
        
        # Width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 65535)
        self.width_spin.setValue(1920)
        self.width_spin.setSingleStep(2) # Usually width is even
        self.width_spin.valueChanged.connect(self.emit_params)
        form_layout.addRow("Width:", self.width_spin)
        
        # Height
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 65535)
        self.height_spin.setValue(1080)
        self.height_spin.valueChanged.connect(self.emit_params)
        form_layout.addRow("Height:", self.height_spin)
        
        # Bit Depth
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["8", "10", "12", "14", "16"])
        self.bit_depth_combo.setCurrentText("10")
        self.bit_depth_combo.currentTextChanged.connect(self.emit_params)
        form_layout.addRow("Bit Depth:", self.bit_depth_combo)
        
        # Bayer Pattern
        self.pattern_combo = QComboBox()
        # Add a few common ones
        self.pattern_combo.addItems(["Mono/None", "RGGB", "BGGR", "GRBG", "GBRG"])
        self.pattern_combo.currentTextChanged.connect(self.emit_params)
        form_layout.addRow("Pattern:", self.pattern_combo)
        
        self.pattern_combo.currentTextChanged.connect(self.emit_params)
        form_layout.addRow("Pattern:", self.pattern_combo)
        
        group.setLayout(form_layout)
        layout.addWidget(group)
        layout.addStretch()
        
        # Block signals initially to prevent blast during setup? 
        # Actually usually fine, but set_params helper is useful.
        
    def set_params(self, params):
        self.blockSignals(True)
        if 'width' in params: self.width_spin.setValue(params['width'])
        if 'height' in params: self.height_spin.setValue(params['height'])
        if 'bit_depth' in params: self.bit_depth_combo.setCurrentText(str(params['bit_depth']))
        if 'pattern' in params: self.pattern_combo.setCurrentText(params['pattern'])
        self.blockSignals(False)
        
    def get_params(self):
        return {
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "bit_depth": int(self.bit_depth_combo.currentText()),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "bit_depth": int(self.bit_depth_combo.currentText()),
            "pattern": self.pattern_combo.currentText()
        }
        
    def emit_params(self):
        self.params_changed.emit(self.get_params())

class AlgorithmPanel(QWidget):
    run_algorithm = pyqtSignal(str, dict) # algorithm_name, params

    def __init__(self, algorithm_manager, parent=None):
        super().__init__(parent)
        self.algorithm_manager = algorithm_manager
        self.current_algo_name = None
        self.param_inputs = {} # Store input widgets

        layout = QVBoxLayout(self)

        # Algorithm Selection
        group = QGroupBox("Algorithms")
        algo_layout = QVBoxLayout()
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self.algorithm_manager.get_algorithm_names())
        self.algo_combo.currentTextChanged.connect(self.on_algo_changed)
        algo_layout.addWidget(QLabel("Select Algorithm:"))
        algo_layout.addWidget(self.algo_combo)

        # Description Label
        self.desc_label = QLabel("")
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: gray; font-style: italic;")
        algo_layout.addWidget(self.desc_label)

        group.setLayout(algo_layout)
        layout.addWidget(group)

        # Parameters Group (Dynamic)
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout()
        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        # Run Button
        self.run_btn = QPushButton("Run Algorithm")
        self.run_btn.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_btn)
        
        layout.addStretch()

        # Initialize with first algo
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
        # Clear existing
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.param_inputs.clear()

        # Build new
        for key, spec in params_spec.items():
            label_text = spec.get("label", key)
            param_type = spec.get("type", "str")
            default_val = spec.get("default")

            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(spec.get("min", -99999), spec.get("max", 99999))
                widget.setValue(int(default_val))
            elif param_type == "float":
                widget = QDoubleSpinBox()
                widget.setRange(spec.get("min", -99999.0), spec.get("max", 99999.0))
                widget.setValue(float(default_val))
            elif param_type == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(default_val))
            elif param_type == "list":
                widget = QComboBox()
                widget.addItems(spec.get("options", []))
                widget.setCurrentText(str(default_val))
            else:
                widget = QLineEdit(str(default_val))
            
            self.params_layout.addRow(label_text, widget)
            self.param_inputs[key] = (widget, param_type)

    def on_run_clicked(self):
        if not self.current_algo_name:
            return
        
        # Collect params
        params = {}
        for key, (widget, param_type) in self.param_inputs.items():
            if param_type == "int":
                params[key] = widget.value()
            elif param_type == "float":
                params[key] = widget.value()
            elif param_type == "bool":
                params[key] = widget.isChecked()
            elif param_type == "list":
                params[key] = widget.currentText()
            else:
                params[key] = widget.text()
        
        self.run_algorithm.emit(self.current_algo_name, params)
