"""图像画布：缩放/平移 + 像素级对齐渲染 + ROI 框选 + 缺陷叠加显示。

沿用之前修好的"整数物理像素吸附"渲染路径（scale/offset 吸附到 1/dpr，
放大时用 np.repeat 做严格块复制），保证放大到单像素时像素块等宽、与网格线
严格对齐 —— 这是判断坏点/坏线位置的前提。

在测试场景下新增：
  * ROI 框选（右键拖拽或开启 ROI 模式后左键拖拽），坐标用全局原始像素坐标；
  * 点选像素（单击）→ 驱动像素检查器；
  * 缺陷叠加在**屏幕坐标系**用 cosmetic 笔绘制，缩小时也看得见（旧实现
    在图像坐标系画 1px 线，缩小后完全不可见）；
  * fit / 1:1 / 居中到指定像素，便于快速定位到缺陷清单里的坐标。
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QLineF, pyqtSignal
from PyQt6.QtGui import (QPainter, QImage, QPaintEvent, QColor, QPen, QBrush,
                         QFont, QPalette)
from PyQt6.QtWidgets import QWidget

from utils import cfa

# 通道文字颜色（与绘图控件保持一致）
_TEXT_COLORS = {
    0: QColor(255, 80, 80),
    1: QColor(80, 255, 80),
    2: QColor(90, 140, 255),
}

# 叠加层配色（缺陷类型 -> 颜色），与 utils/export._PALETTE 保持一致
OVERLAY_COLORS = {
    "hot": QColor(255, 64, 64),
    "dead": QColor(64, 160, 255),
    "cluster": QColor(255, 170, 0),
    "row": QColor(255, 255, 0),
    "col": QColor(0, 255, 220),
    "sat": QColor(255, 0, 255),
    "shading": QColor(255, 170, 0),
    "red": QColor(255, 64, 64),
    "yellow": QColor(255, 255, 0),
    "green": QColor(120, 255, 120),
    "blue": QColor(64, 160, 255),
    "default": QColor(255, 80, 80),
}


class ImageCanvas(QWidget):
    """单张图像的查看器。"""

    pixel_hovered = pyqtSignal(str)          # 兼容旧接口
    hover_info = pyqtSignal(object)          # dict：x/y/value/channel
    pixel_clicked = pyqtSignal(int, int)
    roi_changed = pyqtSignal(object)         # (x0, y0, x1, y1) 或 None
    view_changed = pyqtSignal(float, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None            # QImage（由 set_display 生成，供兼容/导出）
        self._src_np = None          # 显示用 numpy 数组（灰度为 2D，彩色为 3D）
        self.raw_data = None         # 原始 DN 数组（算法/统计用）
        self.pattern = "Mono/None"
        self.overlays = []

        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.last_mouse_pos = QPoint()
        self.is_panning = False

        # ROI / 交互
        self.roi = None                     # (x0, y0, x1, y1) 图像坐标
        self.roi_enabled = False            # True 时左键拖拽 = 框选 ROI
        self._roi_dragging = False
        self._roi_start = None
        self._roi_cursor = None
        self.selected_pixel = None
        self.hover_pixel = None
        self.show_crosshair = True
        self.grid_threshold = 20.0          # 单像素 >= 20 屏幕像素才画网格/数值
        self.max_overlay_draw = 40000
        self._press_pos = None
        self._user_zoomed = False

        self.setMouseTracking(True)
        self.setBackgroundRole(QPalette.ColorRole.NoRole)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # 像素对齐（沿用既有修复，勿随意改动）
    # ------------------------------------------------------------------
    def _device_ratio(self):
        try:
            dpr = self.devicePixelRatioF()
        except Exception:
            dpr = 1.0
        if not dpr or dpr <= 0 or dpr != dpr:
            dpr = 1.0
        return dpr

    def effective_view(self):
        """把 scale/offset 吸附到整数物理像素，返回 (scale, QPointF)。

        图像、网格线、数值文字、叠加层必须共用这一组参数，否则会互相错开。
        """
        dpr = self._device_ratio()
        scale = self.scale
        if scale >= 1.0:
            n = round(scale * dpr)
            scale = max(1, n) / dpr
        ox = round(self.offset.x() * dpr) / dpr
        oy = round(self.offset.y() * dpr) / dpr
        return scale, QPointF(ox, oy)

    def image_to_widget(self, x, y, scale=None, offset=None):
        scale = self.scale if scale is None else scale
        offset = self.offset if offset is None else offset
        return QPointF(x * scale + offset.x(), y * scale + offset.y())

    def widget_to_image(self, pos, scale=None, offset=None):
        scale = self.scale if scale is None else scale
        offset = self.offset if offset is None else offset
        return ((pos.x() - offset.x()) / scale, (pos.y() - offset.y()) / scale)

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    def set_display(self, display_array, raw_data, pattern=None, reset_view: bool = True):
        """设置显示数组（uint8，2D 灰度或 3D RGB）与原始 DN 数据。

        reset_view=False 时保留缩放/平移/ROI/选中点 —— 调 gamma、换视图这类
        "只重渲染"的操作必须走这条路径，否则每拖一下滑块画面就跳回去。
        """
        self._src_np = None if display_array is None else np.ascontiguousarray(display_array)
        self.raw_data = raw_data
        self.pattern = pattern or "Mono/None"
        self.image = self._make_qimage(self._src_np) if self._src_np is not None else None
        if reset_view:
            self.overlays = []
            self.roi = None
            self.selected_pixel = None
            self._user_zoomed = False
            if self._src_np is not None:
                self._center_fit()
            self.update()
            self.view_changed.emit(self.scale, self.offset)
            self.roi_changed.emit(None)
        else:
            self.update()
            self.view_changed.emit(self.scale, self.offset)

    def set_image(self, q_image, raw_data, pattern=None, reset_view: bool = True):
        """兼容旧接口：从 QImage 起步（内部转成 numpy 再走同一路径）。"""
        arr = self._qimage_to_array(q_image) if q_image is not None else None
        self.set_display(arr, raw_data, pattern, reset_view)

    @staticmethod
    def _make_qimage(arr: np.ndarray):
        if arr is None:
            return None
        h, w = arr.shape[0], arr.shape[1]
        if arr.ndim == 2:
            return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    @staticmethod
    def _qimage_to_array(qimg):
        try:
            h, w, bpl = qimg.height(), qimg.width(), qimg.bytesPerLine()
            ptr = qimg.bits()
            ptr.setsize(bpl * h)
            buf = np.frombuffer(ptr, np.uint8).reshape(h, bpl)
            fmt = qimg.format()
            if fmt == QImage.Format.Format_RGB888:
                return np.ascontiguousarray(buf[:, :3 * w].reshape(h, w, 3))
            if fmt == QImage.Format.Format_Grayscale8:
                return np.ascontiguousarray(buf[:, :w])
            if fmt == QImage.Format.Format_RGB32:
                rgba = np.frombuffer(qimg.constBits().asstring(qimg.sizeInBytes()),
                                     np.uint8).reshape(h, bpl)[:, :4 * w].reshape(h, w, 4)
                return np.ascontiguousarray(rgba[:, :, [2, 1, 0]])
        except Exception:
            return None
        return None

    def set_overlays(self, overlays):
        self.overlays = list(overlays or [])
        self.update()

    def set_roi(self, roi, emit: bool = True):
        self.roi = None if roi is None else tuple(int(v) for v in roi)
        self.update()
        if emit:
            self.roi_changed.emit(self.roi)

    def set_roi_enabled(self, enabled: bool):
        self.roi_enabled = bool(enabled)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # 视图操作
    # ------------------------------------------------------------------
    def _image_size(self):
        if self._src_np is None:
            return 0, 0
        return self._src_np.shape[1], self._src_np.shape[0]

    def _center_fit(self):
        w, h = self._image_size()
        if w == 0 or h == 0:
            return
        vw, vh = max(1, self.width()), max(1, self.height())
        scale = min(vw / w, vh / h, 1.0)
        self.scale = max(0.02, scale)
        self.offset = QPoint(int((vw - w * self.scale) / 2), int((vh - h * self.scale) / 2))

    def fit_to_window(self):
        self._center_fit()
        self.update()
        self.view_changed.emit(self.scale, self.offset)

    def zoom_1_1(self):
        self.zoom_to(1.0)

    def zoom_to(self, scale: float, widget_pos=None):
        scale = max(0.02, min(float(scale), 500.0))
        self._user_zoomed = True
        if widget_pos is None:
            widget_pos = QPointF(self.width() / 2.0, self.height() / 2.0)
        px, py = self.widget_to_image(widget_pos)
        self.scale = scale
        self.offset = QPoint(int(widget_pos.x() - px * scale), int(widget_pos.y() - py * scale))
        self.update()
        self.view_changed.emit(self.scale, self.offset)

    def zoom_step(self, direction: int, widget_pos=None):
        """按 1.25 倍步进缩放（保持鼠标位置不动），并吸附到整数物理像素。"""
        self._user_zoomed = True
        factor = 1.25 if direction > 0 else 1 / 1.25
        if widget_pos is None:
            widget_pos = QPointF(self.width() / 2.0, self.height() / 2.0)
        old = self.scale
        px, py = self.widget_to_image(widget_pos)
        new = max(0.02, min(old * factor, 500.0))
        dpr = self._device_ratio()
        if new >= 1.0:
            n_old = int(round(old * dpr))
            n = int(round(new * dpr))
            n = n_old + 1 if (direction > 0 and n <= n_old) else n
            n = n_old - 1 if (direction < 0 and n >= n_old and n_old > 1) else n
            new = max(1, n) / dpr
        self.scale = new
        self.offset = QPoint(int(widget_pos.x() - px * new), int(widget_pos.y() - py * new))
        self.update()
        self.view_changed.emit(self.scale, self.offset)

    def pan_by(self, dx: float, dy: float):
        self.offset = QPoint(int(self.offset.x() + dx), int(self.offset.y() + dy))
        self.update()
        self.view_changed.emit(self.scale, self.offset)

    def center_on(self, x: int, y: int, scale: float | None = None):
        """把指定像素移到视口中心（坐标跳转 / 缺陷定位用）。

        注意要按像素**中心** (x+0.5, y+0.5) 对齐：按左上角对齐会整体偏半个
        像素块（放大 20× 时偏 10 个屏幕像素，看着就像没对准）。
        """
        if scale is not None:
            self.scale = max(0.02, min(float(scale), 500.0))
        self.offset = QPoint(
            int(round(self.width() / 2.0 - (x + 0.5) * self.scale)),
            int(round(self.height() / 2.0 - (y + 0.5) * self.scale)))
        self.selected_pixel = (int(x), int(y))
        self.update()
        self.view_changed.emit(self.scale, self.offset)

    def set_view_params(self, scale, offset):
        self._user_zoomed = True
        self.scale = max(0.02, min(float(scale), 500.0))
        self.offset = QPoint(offset)
        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(38, 40, 44))

        if self._src_np is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No Image Loaded\n(File → Open RAW...)")
            return

        scale, offset = self.effective_view()
        dpr = self._device_ratio()
        n = scale * dpr

        # ---- 1. 图像（整数倍块复制，避免 Qt 定点缩放漂移）----
        vw, vh = self.width(), self.height()
        ih, iw = self._src_np.shape[0], self._src_np.shape[1]
        ix0 = max(0, int(math.floor((0 - offset.x()) / scale)))
        iy0 = max(0, int(math.floor((0 - offset.y()) / scale)))
        ix1 = min(iw, int(math.ceil((vw - offset.x()) / scale)))
        iy1 = min(ih, int(math.ceil((vh - offset.y()) / scale)))

        if ix1 > ix0 and iy1 > iy0:
            px = round((ix0 * scale + offset.x()) * dpr)
            py = round((iy0 * scale + offset.y()) * dpr)
            if scale >= 1.0 and n >= 1.0:
                N = int(round(n))
                sub = self._src_np[iy0:iy1, ix0:ix1]
                if N > 1:
                    sub = np.repeat(np.repeat(sub, N, axis=0), N, axis=1)
                sub = np.ascontiguousarray(sub)
                qimg = self._make_qimage(sub)
            else:
                qimg = self._make_qimage(
                    np.ascontiguousarray(self._src_np[iy0:iy1, ix0:ix1]))
                tw = max(1, int(round((ix1 - ix0) * n)))
                th = max(1, int(round((iy1 - iy0) * n)))
                if qimg is not None and (tw != qimg.width() or th != qimg.height()):
                    qimg = qimg.scaled(tw, th, Qt.AspectRatioMode.IgnoreAspectRatio,
                                       Qt.TransformationMode.FastTransformation)
            if qimg is not None:
                qimg.setDevicePixelRatio(dpr)
                painter.drawImage(QPointF(px / dpr, py / dpr), qimg)

        # ---- 2. 叠加标记 / ROI / 十字线（屏幕坐标系）----
        self._draw_overlays(painter, scale, offset)
        self._draw_roi(painter, scale, offset)
        self._draw_selection(painter, scale, offset)
        self._draw_crosshair(painter, scale, offset)

        # ---- 3. 网格与数值 ----
        if scale >= self.grid_threshold:
            self.draw_pixel_details(painter, scale, offset)

    def _visible_image_rect(self, scale, offset):
        x0, y0 = self.widget_to_image(QPointF(0, 0), scale, offset)
        x1, y1 = self.widget_to_image(QPointF(self.width(), self.height()), scale, offset)
        return x0, y0, x1, y1

    def _draw_overlays(self, painter, scale, offset):
        if not self.overlays:
            return
        x0, y0, x1, y1 = self._visible_image_rect(scale, offset)
        pad = 4
        rects = {}
        lines = {}
        drawn = 0
        for ov in self.overlays:
            if drawn >= self.max_overlay_draw:
                break
            key = str(ov.get("kind") or ov.get("color", "")).lower()
            color = OVERLAY_COLORS.get(key)
            if color is None:
                oc = ov.get("color")
                color = QColor(oc) if isinstance(oc, str) and oc.startswith("#") \
                    else OVERLAY_COLORS["default"]
            otype = ov.get("type")
            coords = ov.get("coords") or ()
            if otype == "point" and len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if not (x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad):
                    continue
                r = ov.get("radius", 3)
                sx = round(x * scale + offset.x())
                sy = round(y * scale + offset.y())
                rects.setdefault(id(color), (color, []))[1].append(
                    QRectF(sx - r, sy - r, 2 * r, 2 * r))
            elif otype == "line" and len(coords) >= 4:
                ax, ay, bx, by = (float(v) for v in coords[:4])
                # 横线/竖线：整条线一般贯穿画面，按可见范围裁剪
                if ay == by:
                    if not (y0 - pad <= ay <= y1 + pad):
                        continue
                elif ax == bx:
                    if not (x0 - pad <= ax <= x1 + pad):
                        continue
                lines.setdefault(id(color), (color, []))[1].append(QLineF(
                    ax * scale + offset.x(), ay * scale + offset.y(),
                    bx * scale + offset.x(), by * scale + offset.y()))
            elif otype == "rect" and len(coords) >= 4:
                rx, ry, rw, rh = (float(v) for v in coords[:4])
                if not (rx <= x1 and rx + rw >= x0 and ry <= y1 and ry + rh >= y0):
                    continue
                rects.setdefault(id(color), (color, []))[1].append(QRectF(
                    rx * scale + offset.x(), ry * scale + offset.y(),
                    max(2.0, rw * scale), max(2.0, rh * scale)))
            drawn += 1

        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for color, rect_list in rects.values():
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.drawRects(rect_list)
        for color, line_list in lines.values():
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidthF(1.2)
            painter.setPen(pen)
            painter.drawLines(line_list)
        painter.restore()

    def _draw_roi(self, painter, scale, offset):
        roi = self.roi
        if self._roi_dragging and self._roi_start and self._roi_cursor:
            ax, ay = self._roi_start
            bx, by = self._roi_cursor
            roi = (int(min(ax, bx)), int(min(ay, by)),
                   int(max(ax, bx)), int(max(ay, by)))
        if roi is None:
            return
        rx0, ry0, rx1, ry1 = roi
        p0 = self.image_to_widget(rx0, ry0, scale, offset)
        p1 = self.image_to_widget(rx1, ry1, scale, offset)
        rect = QRectF(p0, p1).normalized()
        painter.save()
        pen = QPen(QColor(0, 220, 255))
        pen.setCosmetic(True)
        pen.setWidthF(1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 220, 255, 28)))
        painter.drawRect(rect)
        # 尺寸标签
        label = f"ROI {rx1 - rx0}x{ry1 - ry0} @({rx0},{ry0})"
        painter.setFont(QFont("Monospace", 10))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 8
        box = QRectF(rect.left(), max(0, rect.top() - 18), tw, 16)
        painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(box)
        painter.setPen(QColor(0, 235, 255))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()

    def _draw_selection(self, painter, scale, offset):
        if not self.selected_pixel:
            return
        x, y = self.selected_pixel
        c = self.image_to_widget(x + 0.5, y + 0.5, scale, offset)
        painter.save()
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setCosmetic(True)
        pen.setWidthF(1.2)
        painter.setPen(pen)
        w = max(4.0, scale)
        painter.drawLine(QPointF(c.x() - w, c.y() - w), QPointF(c.x() + w, c.y() + w))
        painter.drawLine(QPointF(c.x() - w, c.y() + w), QPointF(c.x() + w, c.y() - w))
        painter.restore()

    def _draw_crosshair(self, painter, scale, offset):
        if not self.show_crosshair or not self.hover_pixel:
            return
        x, y = self.hover_pixel
        c = self.image_to_widget(x + 0.5, y + 0.5, scale, offset)
        painter.save()
        pen = QPen(QColor(255, 255, 255, 90))
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(0, c.y()), QPointF(self.width(), c.y()))
        painter.drawLine(QPointF(c.x(), 0), QPointF(c.x(), self.height()))
        painter.restore()

    def draw_pixel_details(self, painter, scale, offset):
        """放大到一定倍数后画像素网格 + 每个像素的 DN 数值。"""
        start_x = max(0, int((-offset.x()) / scale))
        start_y = max(0, int((-offset.y()) / scale))
        end_x = min(self._src_np.shape[1], int((self.width() - offset.x()) / scale) + 1)
        end_y = min(self._src_np.shape[0], int((self.height() - offset.y()) / scale) + 1)
        if end_x <= start_x or end_y <= start_y:
            return

        dpr = self._device_ratio()

        def snap(v):
            return round(v * dpr) / dpr

        x0s, x1s = snap(start_x * scale + offset.x()), snap(end_x * scale + offset.x())
        y0s, y1s = snap(start_y * scale + offset.y()), snap(end_y * scale + offset.y())

        pen_grid = QPen(QColor(110, 110, 110, 140))
        pen_grid.setCosmetic(True)
        pen_grid.setWidthF(1.0)

        painter.save()
        painter.resetTransform()
        painter.setPen(pen_grid)
        lines = []
        for x in range(start_x, end_x + 1):
            X = snap(x * scale + offset.x())
            lines.append(QLineF(X, y0s, X, y1s))
        for y in range(start_y, end_y + 1):
            Y = snap(y * scale + offset.y())
            lines.append(QLineF(x0s, Y, x1s, Y))
        if lines:
            painter.drawLines(lines)
        painter.restore()

        if self.raw_data is None:
            return
        # 数值文字：像素太小就不画（避免糊成一片）
        if scale < max(self.grid_threshold, 22.0):
            return
        painter.save()
        painter.resetTransform()
        font = QFont("Monospace")
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)

        rh, rw = self.raw_data.shape[0], self.raw_data.shape[1]
        for y in range(start_y, end_y):
            if y >= rh:
                break
            Y = snap(y * scale + offset.y())
            for x in range(start_x, end_x):
                if x >= rw:
                    break
                X = snap(x * scale + offset.x())
                val = self.raw_data[y, x]
                rect = QRectF(X, Y, scale, scale)
                text_color = QColor(255, 255, 0)
                ch = cfa.channel_index(x, y, self.pattern)
                if ch in _TEXT_COLORS:
                    text_color = _TEXT_COLORS[ch]
                painter.setPen(Qt.GlobalColor.black)
                painter.drawText(rect.translated(1, 1), Qt.AlignmentFlag.AlignCenter, str(val))
                painter.setPen(text_color)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(val))
        painter.restore()

    # ------------------------------------------------------------------
    # 鼠标 / 键盘
    # ------------------------------------------------------------------
    def _pick_pixel(self, pos):
        scale, offset = self.effective_view()
        ix, iy = self.widget_to_image(QPointF(pos), scale, offset)
        return int(math.floor(ix)), int(math.floor(iy))

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        start_roi = (event.button() == Qt.MouseButton.RightButton or
                     (event.button() == Qt.MouseButton.LeftButton and self.roi_enabled))
        if start_roi and self._src_np is not None:
            x, y = self._pick_pixel(pos)
            self._roi_dragging = True
            self._roi_start = (x, y)
            self._roi_cursor = (x, y)
            self.update()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self.is_panning = True
            self.last_mouse_pos = pos
            self._press_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._src_np is not None:
            ix, iy = self._pick_pixel(pos)
            ih, iw = self.raw_data.shape if self.raw_data is not None else (0, 0)
            inside_show = 0 <= ix < self._src_np.shape[1] and 0 <= iy < self._src_np.shape[0]
            if inside_show:
                self.hover_pixel = (ix, iy)
                val = None
                if self.raw_data is not None and 0 <= ix < iw and 0 <= iy < ih:
                    val = int(self.raw_data[iy, ix])
                info = {
                    "x": ix, "y": iy, "value": val,
                    "channel": cfa.plane_name_at(ix, iy, self.pattern),
                    "pattern": self.pattern,
                }
                self.hover_info.emit(info)
                self.pixel_hovered.emit(
                    f"X: {ix}, Y: {iy} | {info['channel']}: {val}")
            else:
                self.hover_pixel = None
                self.pixel_hovered.emit("")
                self.hover_info.emit(None)
            if not self.is_panning and not self._roi_dragging:
                self.update()          # 刷新十字线

        if self._roi_dragging:
            x, y = self._pick_pixel(pos)
            ih = self._src_np.shape[0] if self._src_np is not None else 0
            iw = self._src_np.shape[1] if self._src_np is not None else 0
            self._roi_cursor = (min(max(x, 0), iw), min(max(y, 0), ih))
            self.update()
            return

        if self.is_panning:
            delta = pos - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = pos
            self.update()
            self.view_changed.emit(self.scale, self.offset)

    def mouseReleaseEvent(self, event):
        if self._roi_dragging and self._roi_start and self._roi_cursor:
            ax, ay = self._roi_start
            bx, by = self._roi_cursor
            self._roi_dragging = False
            self._roi_start = self._roi_cursor = None
            x0, x1 = int(min(ax, bx)), int(max(ax, bx))
            y0, y1 = int(min(ay, by)), int(max(ay, by))
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                self.set_roi((x0, y0, x1, y1))
            else:
                self.update()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            was_panning = self.is_panning
            self.is_panning = False
            if self.roi_enabled:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            # 单击（没有拖动）→ 选点，供像素检查器使用
            press = self._press_pos or event.position().toPoint()
            self._press_pos = None
            if (event.button() == Qt.MouseButton.LeftButton and was_panning
                    and (event.position().toPoint() - press).manhattanLength() <= 3):
                x, y = self._pick_pixel(event.position().toPoint())
                if self.raw_data is not None:
                    ih, iw = self.raw_data.shape
                    if 0 <= x < iw and 0 <= y < ih:
                        self.selected_pixel = (x, y)
                        self.pixel_clicked.emit(x, y)
                        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.zoom_to(1.0 if self.scale < 1.0 else 1.0, event.position())

    def wheelEvent(self, event):
        self.zoom_step(1 if event.angleDelta().y() > 0 else -1, event.position())

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self.roi is not None:
                self.set_roi(None)
            else:
                self.selected_pixel = None
                self.update()
            return
        if key == Qt.Key.Key_F:
            self.fit_to_window()
            return
        step = 40
        if key == Qt.Key.Key_Left:
            self.pan_by(step, 0)
        elif key == Qt.Key.Key_Right:
            self.pan_by(-step, 0)
        elif key == Qt.Key.Key_Up:
            self.pan_by(0, step)
        elif key == Qt.Key.Key_Down:
            self.pan_by(0, -step)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._src_np is not None and not self._user_zoomed:
            self._center_fit()
