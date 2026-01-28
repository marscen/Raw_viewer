from PyQt6.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QLabel)
from PyQt6.QtGui import QImage, QAction
from PyQt6.QtCore import Qt
import numpy as np
import os

from ui.canvas import ImageCanvas
from ui.dialogs import ImageParamsDialog
from utils.image_loader import load_raw_image, apply_bayer_mask

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Viewer")
        self.resize(1000, 800)
        
        # Central Widget
        self.canvas = ImageCanvas()
        self.setCentralWidget(self.canvas)
        
        # Status Bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        self.canvas.pixel_hovered.connect(self.status_label.setText)
        
        # Menu
        file_menu = self.menuBar().addMenu("File")
        
        open_action = QAction("Open RAW...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_raw_file)
        file_menu.addAction(open_action)
        
    def open_raw_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open RAW File", "", "RAW Files (*.raw *.bin *.bayer);;All Files (*)")
        if not file_path:
            return
            
        # Ask for parameters
        dialog = ImageParamsDialog(self)
        if dialog.exec():
            params = dialog.get_params()
            self.load_image(file_path, params)
            
    def load_image(self, file_path, params):
        try:
            width = params['width']
            height = params['height']
            bit_depth = params['bit_depth']
            
            self.status_label.setText(f"Loading {os.path.basename(file_path)}...")
            
            # Use utility to load
            display_data, raw_data = load_raw_image(file_path, width, height, bit_depth)
            
            if display_data is None:
                QMessageBox.critical(self, "Error", "Failed to load image data.")
                self.status_label.setText("Error loading.")
                return

            # Convert to QImage
            # display_data is uint8 numpy array (H, W)
            # We need to construct QImage. 
            
            # Check pattern
            pattern = params.get('pattern', 'Mono/None')
            if pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
                # Colorize
                rgb_data = apply_bayer_mask(raw_data, pattern, bit_depth)
                # QImage.Format_RGB888 needs contiguous data
                if not rgb_data.flags['C_CONTIGUOUS']:
                     rgb_data = np.ascontiguousarray(rgb_data)
                     
                height, width, _ = rgb_data.shape
                # QImage from RGB888
                q_img = QImage(rgb_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
            else:
                # Grayscale
                # Format_Grayscale8 is best for single channel
                q_img = QImage(display_data.data, width, height, width, QImage.Format.Format_Grayscale8)
            
            # Start strict separation: QImage buffer must persist or be copied. 
            # Passing numpy buffer directly to QImage is risky if numpy array is garbage collected.
            # safe approach: .copy()
            self.display_image = q_img.copy()
            
            self.canvas.set_image(self.display_image, raw_data, pattern)
            self.status_label.setText(f"Loaded: {width}x{height}, {bit_depth}-bit")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.status_label.setText("Error.")
