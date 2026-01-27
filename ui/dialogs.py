from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSpinBox, QComboBox, QDialogButtonBox, QFormLayout)
from PyQt6.QtCore import Qt

class ImageParamsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RAW Image Parameters")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # Width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 65535)
        self.width_spin.setValue(1920) # Default
        form_layout.addRow("Width:", self.width_spin)
        
        # Height
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 65535)
        self.height_spin.setValue(1080) # Default
        form_layout.addRow("Height:", self.height_spin)
        
        # Bit Depth
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["8", "10", "12", "14", "16"])
        self.bit_depth_combo.setCurrentText("10")
        form_layout.addRow("Bit Depth:", self.bit_depth_combo)
        
        # Bayer Pattern (Optional for now, but good to have)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["RGGB", "GRBG", "GBRG", "BGGR", "Mono/None"])
        form_layout.addRow("Bayer Pattern:", self.pattern_combo)

        layout.addLayout(form_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_params(self):
        return {
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "bit_depth": int(self.bit_depth_combo.currentText()),
            "pattern": self.pattern_combo.currentText()
        }
