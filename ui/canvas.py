from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPaintEvent, QColor, QPen, QFont, QPalette
import numpy as np

class ImageCanvas(QWidget):
    # Signal to report pixel info (x, y, value) to status bar
    pixel_hovered = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None # This will hold the QImage for display
        self.raw_data = None # This will hold the original raw numpy array
        self.pattern = None # Bayer pattern
        
        # Interaction state
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.last_mouse_pos = QPoint()
        self.is_panning = False
        
        self.setMouseTracking(True) # Enable mouse tracking for pixel info
        self.setBackgroundRole(QPalette.ColorRole.NoRole) # Handle background painting manually
        
    def set_image(self, q_image, raw_data, pattern=None):
        self.image = q_image
        self.raw_data = raw_data
        self.pattern = pattern
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        
        # Center image initially
        if self.image:
             # Logic to center can go here, or just reset to 0,0
             pass
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        
        if self.image is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")
            return

        # Apply transformations
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        
        # Draw Image
        painter.drawImage(0, 0, self.image)
        
        # Draw Pixel Grid and Values if zoomed in enough
        # Threshold: e.g., when 1 pixel is at least 20 screen pixels
        if self.scale >= 20.0:
            self.draw_pixel_details(painter, event.rect())
            
    def draw_pixel_details(self, painter, debug_rect):
        # Calculate visible source rect to avoid iterating entire huge image
        # Inverse transform map_from_viewport... 
        # Simplified approach: iterate visible range
        
        # Get visible area in image coordinates
        # visible view rect in widget coords is (0,0) to (width, height)
        # map to image coords: (view_pos - offset) / scale
        
        widget_rect = self.rect()
        
        start_x = int((-self.offset.x()) / self.scale)
        start_y = int((-self.offset.y()) / self.scale)
        end_x = int((widget_rect.width() - self.offset.x()) / self.scale) + 1
        end_y = int((widget_rect.height() - self.offset.y()) / self.scale) + 1
        
        # Clamp to image bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(self.image.width(), end_x)
        end_y = min(self.image.height(), end_y)
        
        pen_grid = QPen(QColor(100, 100, 100, 128))
        pen_grid.setWidthF(1.0 / self.scale) # Keep line width const relative to screen? Or just 0 cosmetic?
        # Actually 0 width is 'cosmetic' (1 pixel wide always), but strictly we might want scaling.
        # Let's use 0 for hairline.
        pen_grid.setWidth(0)
        painter.setPen(pen_grid)
        
        font = painter.font()
        font.setPointSizeF(8) # Fixed font size? No, we transform, so font scales too?
        # If we use painter.scale, everything scales. So text "123" will become huge.
        # usually we want text to stay readable size.
        # So we should perhaps reset transform for text, or use constant pixel size.
        # Or, we just let it scale if we want it to be "part of the pixel".
        # Required: "Show pixel value". Usually implies readable text inside the pixel box.
        # If scale is 20, a pixel is 20x20. Text size 8-10 is fine.
        # But if scale is 50, text 10 is too small? No, scale applies to font size too technically if using painter.drawText.
        # Wait, if I do painter.scale(20, 20) and drawText with font 10, the text will be 200 units high? Yes.
        # So we need to scale the font DOWN or draw text in inverse transform.
        
        # Better approach for text: do not rely on painter scaling for text if we want fixed screen size,
        # OR set font pixel size to something small like 0.4 if we want it to fit in 1.0 box?
        # Let's try: font size = 0.4 (relative to pixel size 1.0).
        font.setPixelSize(1) # Too big? Pixel is 1x1. Font size should be e.g. 0.5 units height?
        # QFont setPointSizeF doesn't allow < 1 easily or might behave weird.
        # Let's use simpler approach: Only draw lines here relative to image.
        
        # Draw Grid
        # Vertical lines
        for x in range(start_x, end_x + 1):
            painter.drawLine(x, start_y, x, end_y)
        # Horizontal lines
        for y in range(start_y, end_y + 1):
            painter.drawLine(start_x, y, end_x, y)
            
        # Draw Text
        # We need check if raw_data exists and matches dimensions
        if self.raw_data is not None:
             painter.save()
             painter.resetTransform()
             
             # Set font for screen coordinates
             f = QFont("Monospace")
             f.setPixelSize(12) # 12px on screen
             f.setBold(True)
             painter.setFont(f)
             
             for y in range(start_y, end_y):
                 for x in range(start_x, end_x):
                     if y < self.raw_data.shape[0] and x < self.raw_data.shape[1]:
                         val = self.raw_data[y, x]
                         # Calculate screen position
                         sx = x * self.scale + self.offset.x()
                         sy = y * self.scale + self.offset.y()
                         
                         rect = QRectF(sx, sy, self.scale, self.scale)
                         
                         # Determine Color
                         text_color = QColor(255, 255, 0) # Default Yellow
                         if self.pattern and self.pattern not in ["Mono/None", None]:
                             channel = self.get_bayer_channel(x, y, self.pattern)
                             if channel == 0: # R
                                 text_color = QColor(255, 80, 80)
                             elif channel == 1: # G
                                 text_color = QColor(80, 255, 80)
                             elif channel == 2: # B
                                 text_color = QColor(80, 80, 255)
                         
                         # Draw Shadow/Outline for visibility
                         shadow_offset = 1
                         painter.setPen(Qt.GlobalColor.black)
                         # Draw shadow at +1, +1
                         shadow_rect = rect.translated(shadow_offset, shadow_offset)
                         painter.drawText(shadow_rect, Qt.AlignmentFlag.AlignCenter, str(val))
                         
                         # Draw Main Text
                         painter.setPen(text_color)
                         painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(val))
             
             painter.restore()

    def get_bayer_channel(self, x, y, pattern):
        # 0=R, 1=G, 2=B
        pattern = pattern.upper()
        # Row parity, Col parity
        r = y % 2
        c = x % 2
        
        if pattern == "RGGB":
            # R G
            # G B
            if r == 0: return 0 if c == 0 else 1
            else:      return 1 if c == 0 else 2
            
        elif pattern == "BGGR":
            # B G
            # G R
            if r == 0: return 2 if c == 0 else 1
            else:      return 1 if c == 0 else 0

        elif pattern == "GRBG":
            # G R
            # B G
            if r == 0: return 1 if c == 0 else 0
            else:      return 2 if c == 0 else 1
            
        elif pattern == "GBRG":
            # G B
            # R G
            if r == 0: return 1 if c == 0 else 2
            else:      return 0 if c == 0 else 1
            
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        # Update status bar info
        if self.image:
             # Calculate pixel coord
             img_x = int((event.pos().x() - self.offset.x()) / self.scale)
             img_y = int((event.pos().y() - self.offset.y()) / self.scale)
             
             if 0 <= img_x < self.image.width() and 0 <= img_y < self.image.height():
                 val = "N/A"
                 if self.raw_data is not None:
                     val = str(self.raw_data[img_y, img_x])
                 self.pixel_hovered.emit(f"X: {img_x}, Y: {img_y} | Value: {val}")
             else:
                 self.pixel_hovered.emit("")

        # Pan logic
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        # Zoom logic
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        old_scale = self.scale
        if event.angleDelta().y() > 0:
            self.scale *= zoom_in_factor
        else:
            self.scale *= zoom_out_factor
            
        # Clamp scale?
        self.scale = max(0.1, min(self.scale, 500.0))
        
        # Zoom towards mouse pointer
        # New offset needs to keep the point under mouse stable
        # moose_pos = (point * old_scale) + old_offset
        # mouse_pos = (point * new_scale) + new_offset
        # => new_offset = mouse_pos - point * new_scale
        # point = (mouse_pos - old_offset) / old_scale
        
        mouse_pos = event.position()
        point_x = (mouse_pos.x() - self.offset.x()) / old_scale
        point_y = (mouse_pos.y() - self.offset.y()) / old_scale
        
        new_offset_x = mouse_pos.x() - point_x * self.scale
        new_offset_y = mouse_pos.y() - point_y * self.scale
        
        self.offset = QPoint(int(new_offset_x), int(new_offset_y))
        
        self.update()
