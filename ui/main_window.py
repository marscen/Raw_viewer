from PyQt6.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QLabel, QDockWidget, QSplitter)
from PyQt6.QtGui import QImage, QAction
from PyQt6.QtCore import Qt
import numpy as np
import os

from ui.canvas import ImageCanvas
from ui.dialogs import ImageParamsDialog
from ui.canvas import ImageCanvas
from ui.dialogs import ImageParamsDialog
from ui.sidebar import ImageControlPanel, AlgorithmPanel
from utils.image_loader import load_raw_image, apply_bayer_mask
from algorithms.manager import AlgorithmManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Viewer")
        self.resize(1000, 800)
        
        # Central Widget
        self.canvas = ImageCanvas()
        
        # Reference Canvas (for Dual View)
        self.ref_canvas = ImageCanvas()
        self.ref_canvas.setVisible(False) # Hidden by default
        
        # Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.ref_canvas)
        self.splitter.setCollapsible(0, False)
        
        self.setCentralWidget(self.splitter)

        # Sync Signals
        self.canvas.view_changed.connect(self.sync_to_ref)
        self.ref_canvas.view_changed.connect(self.sync_to_main)
        
        # Status Bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        self.canvas.pixel_hovered.connect(self.status_label.setText)
        
        # Current file state
        self.current_file_path = None
        self.reference_file_path = None
        
        # Sidebar (Dock)
        self.algorithm_manager = AlgorithmManager()
        
        self.sidebar = ImageControlPanel()
        self.algo_panel = AlgorithmPanel(self.algorithm_manager)
        
        # Use a Tab Widget for sidebar content
        from PyQt6.QtWidgets import QTabWidget
        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.sidebar, "Controls")
        self.side_tabs.addTab(self.algo_panel, "Algorithms")
        
        self.dock = QDockWidget("Tools", self)
        self.dock.setWidget(self.side_tabs)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.setVisible(False) # Hide by default
        
        # Connect Sidebar
        self.sidebar.params_changed.connect(self.reload_image_with_params)
        self.algo_panel.run_algorithm.connect(self.run_algorithm)
        
        # Menu
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        view_menu = menu_bar.addMenu("View")
        
        # File Actions
        open_action = QAction("Open RAW...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_raw_file)
        file_menu.addAction(open_action)
        
        open_ref_action = QAction("Open Reference Image...", self)
        open_ref_action.triggered.connect(self.open_reference_file)
        file_menu.addAction(open_ref_action)
        
        # View Actions
        toggle_view_action = self.dock.toggleViewAction()
        view_menu.addAction(toggle_view_action)

        toggle_compare_action = QAction("Compare Mode", self)
        toggle_compare_action.setCheckable(True)
        toggle_compare_action.toggled.connect(self.toggle_compare_mode)
        view_menu.addAction(toggle_compare_action)
        
    def open_raw_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open RAW File", "", "RAW Files (*.raw *.bin *.bayer);;All Files (*)")
        if not file_path:
            return
            
        # Ask for parameters
        dialog = ImageParamsDialog(self)
        if dialog.exec():
            params = dialog.get_params()
            self.current_file_path = file_path # Store path
            self.load_image(file_path, params)
            # Sync sidebar
            self.sidebar.set_params(params)
            
    def reload_image_with_params(self, params):
        if self.current_file_path:
            self.load_image(self.current_file_path, params)
            
    def run_algorithm(self, algo_name, params):
        if not self.canvas.raw_data is not None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        algo = self.algorithm_manager.get_algorithm(algo_name)
        if not algo:
            return
            
        try:
            self.status_label.setText(f"Running {algo_name}...")
            # Run algorithm
            result = algo.run(self.canvas.raw_data, params)
            
            # Handle result
            if "overlays" in result:
                self.canvas.set_overlays(result["overlays"])
            
            # TODO: If image modified, update it?
            # if "image" in result and result["image"] is not self.canvas.raw_data:
            #     ... update display ...
            
            msg = result.get("message", "Done.")
            self.status_label.setText(msg)
            
            if "message" in result:
                 QMessageBox.information(self, "Algorithm Result", result["message"])
                 
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Algorithm failed: {str(e)}")
            self.status_label.setText("Algorithm error.")
            
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

    def toggle_compare_mode(self, checked):
        self.ref_canvas.setVisible(checked)
        if checked:
            self.sync_to_ref(self.canvas.scale, self.canvas.offset)

    def sync_to_ref(self, scale, offset):
        if self.ref_canvas.isVisible():
            self.ref_canvas.set_view_params(scale, offset)

    def sync_to_main(self, scale, offset):
        if self.ref_canvas.isVisible():
            self.canvas.set_view_params(scale, offset)

    def open_reference_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Reference File", "", "RAW Files (*.raw *.bin *.bayer);;All Files (*)")
        if not file_path:
            return
            
        dialog = ImageParamsDialog(self)
        if dialog.exec():
            params = dialog.get_params()
            self.reference_file_path = file_path
            self.load_reference_image(file_path, params)

    def load_reference_image(self, file_path, params):
        try:
            width = params['width']
            height = params['height']
            bit_depth = params['bit_depth']
            
            self.status_label.setText(f"Loading Ref: {os.path.basename(file_path)}...")
            
            display_data, raw_data = load_raw_image(file_path, width, height, bit_depth)
            
            if display_data is None:
                QMessageBox.critical(self, "Error", "Failed to load reference image data.")
                self.status_label.setText("Error loading reference.")
                return

            pattern = params.get('pattern', 'Mono/None')
            if pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
                rgb_data = apply_bayer_mask(raw_data, pattern, bit_depth)
                if not rgb_data.flags['C_CONTIGUOUS']:
                     rgb_data = np.ascontiguousarray(rgb_data)
                height, width, _ = rgb_data.shape
                q_img = QImage(rgb_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
            else:
                q_img = QImage(display_data.data, width, height, width, QImage.Format.Format_Grayscale8)
            
            self.ref_canvas.set_image(q_img.copy(), raw_data, pattern)
            self.status_label.setText(f"Loaded Ref: {width}x{height}, {bit_depth}-bit")
            
            # If in compare mode, sync view
            if self.ref_canvas.isVisible():
                self.sync_to_ref(self.canvas.scale, self.canvas.offset)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ref Load Error: {str(e)}")
            self.status_label.setText("Error.")
