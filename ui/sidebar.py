from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QSpinBox, 
                             QComboBox, QGroupBox, QLabel)
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
            "pattern": self.pattern_combo.currentText()
        }
        
    def emit_params(self):
        self.params_changed.emit(self.get_params())
