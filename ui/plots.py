"""轻量绘图控件：直方图 / 行-列 profile。

不引入 matplotlib（项目只依赖 numpy+PyQt6）。两个控件都做了：
  * 自动量程 + 刻度 + 网格；
  * 鼠标悬停读数（找列/找行时非常有用）；
  * 通道配色统一（R 红 / Gr 亮绿 / Gb 深绿 / B 蓝 / ALL 灰）。
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPolygonF
from PyQt6.QtWidgets import QWidget

from ui.qtutil import mono_font, safe_paint

# 通道统一配色：R/Gr/Gb/B 与画布上的像素文字颜色保持一致
CHANNEL_COLORS = {
    "ALL": QColor(200, 200, 200),
    "Mono": QColor(200, 200, 200),
    "R": QColor(255, 80, 80),
    "Gr": QColor(80, 255, 80),
    "Gb": QColor(60, 190, 60),
    "B": QColor(90, 140, 255),
}


def color_for(name: str) -> QColor:
    return CHANNEL_COLORS.get(str(name), QColor(255, 220, 120))


class PlotBase(QWidget):
    """公共坐标轴/网格绘制。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.margin_left = 56
        self.margin_bottom = 26
        self.margin_top = 10
        self.margin_right = 12
        self._x_range = (0.0, 1.0)
        self._y_range = (0.0, 1.0)
        self.cursor_text = ""
        self.setMouseTracking(True)

    def plot_rect(self) -> QRectF:
        return QRectF(self.margin_left, self.margin_top,
                      max(1, self.width() - self.margin_left - self.margin_right),
                      max(1, self.height() - self.margin_top - self.margin_bottom))

    def to_screen(self, x: float, y: float) -> QPointF:
        r = self.plot_rect()
        x0, x1 = self._x_range
        y0, y1 = self._y_range
        sx = r.left() + (x - x0) / max(x1 - x0, 1e-9) * r.width()
        sy = r.bottom() - (y - y0) / max(y1 - y0, 1e-9) * r.height()
        return QPointF(sx, sy)

    def to_data(self, pos) -> tuple:
        r = self.plot_rect()
        x0, x1 = self._x_range
        y0, y1 = self._y_range
        fx = (pos.x() - r.left()) / max(r.width(), 1e-9)
        fy = (r.bottom() - pos.y()) / max(r.height(), 1e-9)
        return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)

    def _draw_axes(self, painter: QPainter, x_label: str, y_label: str,
                   x_ticks: int = 5, y_ticks: int = 4):
        r = self.plot_rect()
        painter.fillRect(r, QColor(28, 30, 34))
        pen_grid = QPen(QColor(70, 74, 80))
        pen_grid.setWidthF(1.0)
        pen_axis = QPen(QColor(150, 155, 160))
        painter.setFont(mono_font(10))

        x0, x1 = self._x_range
        y0, y1 = self._y_range
        for i in range(x_ticks + 1):
            v = x0 + (x1 - x0) * i / x_ticks
            p = self.to_screen(v, y0)
            painter.setPen(pen_grid)
            painter.drawLine(QPointF(p.x(), r.top()), QPointF(p.x(), r.bottom()))
            painter.setPen(QColor(190, 195, 200))
            txt = f"{v:.4g}"
            painter.drawText(QRectF(p.x() - 34, r.bottom() + 2, 68, 14),
                             Qt.AlignmentFlag.AlignCenter, txt)
        for i in range(y_ticks + 1):
            v = y0 + (y1 - y0) * i / y_ticks
            p = self.to_screen(x0, v)
            painter.setPen(pen_grid)
            painter.drawLine(QPointF(r.left(), p.y()), QPointF(r.right(), p.y()))
            painter.setPen(QColor(190, 195, 200))
            painter.drawText(QRectF(2, p.y() - 7, self.margin_left - 8, 14),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{v:.4g}")
        painter.setPen(pen_axis)
        painter.drawRect(r)
        painter.setPen(QColor(210, 215, 220))
        painter.drawText(QRectF(r.left(), r.bottom() + 14, r.width(), 14),
                         Qt.AlignmentFlag.AlignCenter, x_label)
        painter.save()
        painter.translate(12, r.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-r.height() / 2, -10, r.height(), 20),
                         Qt.AlignmentFlag.AlignCenter, y_label)
        painter.restore()


class HistogramPlot(PlotBase):
    """直方图：默认对数纵轴，方便同时看到主体分布和孤立坏点。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = None            # 总体 counts（用于自动量程）
        self.edges = None
        self.channels = {}            # {name: counts}
        self.log_scale = True
        self.show_channels = True
        self.sat_line = None

    def set_data(self, counts, edges, channels: dict | None = None,
                 sat_line: float | None = None):
        self.counts = None if counts is None else np.asarray(counts)
        self.edges = None if edges is None else np.asarray(edges)
        self.channels = {k: np.asarray(v) for k, v in (channels or {}).items()}
        self.sat_line = sat_line
        self._rescale()
        self.update()

    def set_log_scale(self, enabled: bool):
        self.log_scale = bool(enabled)
        self._rescale()
        self.update()

    def set_show_channels(self, enabled: bool):
        self.show_channels = bool(enabled)
        self.update()

    def _rescale(self):
        if self.edges is None or (self.counts is None and not self.channels):
            self._x_range, self._y_range = (0.0, 1.0), (0.0, 1.0)
            return
        self._x_range = (float(self.edges[0]), float(self.edges[-1]))
        peak = 1.0
        for c in [self.counts] + list(self.channels.values()):
            if c is not None and c.size:
                peak = max(peak, float(np.max(c)))
        self._y_range = (0.0, peak * 1.08)
        if self.log_scale:
            self._y_range = (0.0, math.log10(peak + 1.0) * 1.08 or 1.0)

    def _val(self, c):
        if self.log_scale:
            return np.log10(c.astype(np.float64) + 1.0)
        return c.astype(np.float64)

    def _draw_series(self, painter: QPainter, counts, color: QColor,
                     fill: bool, width: float = 1.2):
        if counts is None or counts.size == 0 or self.edges is None:
            return
        vals = self._val(counts)
        n = counts.size
        xs = np.linspace(self.edges[0], self.edges[-1], n)
        pts = [self.to_screen(float(xs[i]), float(vals[i])) for i in range(n)]
        painter.setPen(QPen(color, width))
        if n > 1:
            painter.drawPolyline(QPolygonF(pts))
        if fill:
            c = QColor(color)
            c.setAlpha(70)
            poly = QPolygonF([QPointF(self.to_screen(self.edges[0], self._y_range[0])),
                              QPointF(self.to_screen(self.edges[-1], self._y_range[0]))] + pts)
            painter.setBrush(QBrush(c))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(poly)

    @safe_paint
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(24, 26, 30))
        y_label = "log10(count)" if self.log_scale else "count"
        self._draw_axes(painter, "DN", y_label)

        if self.edges is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无数据")
            return

        if self.show_channels and self.channels:
            for name, c in self.channels.items():
                self._draw_series(painter, c, color_for(name), fill=False, width=1.2)
        if self.counts is not None:
            self._draw_series(painter, self.counts, QColor(215, 215, 215), fill=True, width=1.0)

        if self.sat_line is not None and self._x_range[1] > self._x_range[0]:
            p = self.to_screen(float(self.sat_line), self._y_range[1])
            pen = QPen(QColor(255, 120, 120), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(p.x(), self.plot_rect().top()),
                             QPointF(p.x(), self.plot_rect().bottom()))

        if self.cursor_text:
            painter.setPen(QColor(230, 230, 230))
            painter.setFont(mono_font(9))
            painter.drawText(QRectF(self.plot_rect().left() + 6, self.plot_rect().top() + 4,
                                    240, 14), Qt.AlignmentFlag.AlignLeft, self.cursor_text)

    def mouseMoveEvent(self, event):
        if self.edges is None:
            return
        x, _ = self.to_data(event.position())
        if self.counts is None:
            return
        n = self.counts.size
        span = (self.edges[-1] - self.edges[0]) / max(n, 1)
        idx = int((x - self.edges[0]) / max(span, 1e-9))
        idx = max(0, min(n - 1, idx))
        parts = [f"DN {self.edges[idx]:.0f}-{self.edges[min(idx+1, len(self.edges)-1)]:.0f}: {int(self.counts[idx])}"]
        for name, c in self.channels.items():
            if idx < c.size:
                parts.append(f"{name}={int(c[idx])}")
        self.cursor_text = "  ".join(parts)
        self.update()


class ProfilePlot(PlotBase):
    """行/列 profile 曲线：支持多条曲线 + 可选 ±std 包络。"""

    index_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.index = None
        self.series = []              # [(name, values)]
        self.band = None              # (mean, std) 画 ±3σ 包络
        self.show_channels = True
        self._nearest_idx = None

    def set_data(self, index, mean, std=None, channels=None, axis: str = "rows"):
        self.index = None if index is None else np.asarray(index)
        self.series = [("ALL", np.asarray(mean))] if mean is not None else []
        for name, (cm, cs) in (channels or {}).items():
            self.series.append((name, np.asarray(cm)))
        self.band = None if std is None else (np.asarray(mean), np.asarray(std))
        self._axis = axis
        self._rescale()
        self.update()

    def _rescale(self):
        vals = []
        for _, v in self.series:
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(v)
        if not vals or self.index is None or self.index.size == 0:
            self._x_range, self._y_range = (0.0, 1.0), (0.0, 1.0)
            return
        allv = np.concatenate(vals)
        lo, hi = float(np.min(allv)), float(np.max(allv))
        pad = max((hi - lo) * 0.08, 1.0)
        self._x_range = (float(self.index[0]), float(self.index[-1]) or 1.0)
        self._y_range = (lo - pad, hi + pad)

    @safe_paint
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(24, 26, 30))
        axis = getattr(self, "_axis", "rows")
        self._draw_axes(painter, "row index (y)" if axis == "rows" else "col index (x)", "DN")

        if self.index is None or not self.series:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无数据")
            return

        # ±3σ 包络
        if self.band is not None:
            mean, std = self.band
            upper = mean + 3 * std
            lower = mean - 3 * std
            poly = [self.to_screen(float(self.index[i]), float(upper[i])) for i in range(len(mean))]
            poly += [self.to_screen(float(self.index[i]), float(lower[i])) for i in range(len(mean) - 1, -1, -1)]
            c = QColor(120, 120, 140, 60)
            painter.setBrush(QBrush(c))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygonF(poly))

        for name, vals in self.series:
            if not self.show_channels and name != "ALL":
                continue
            color = color_for(name)
            pts = []
            for i in range(len(vals)):
                if not np.isfinite(vals[i]):
                    continue
                pts.append(self.to_screen(float(self.index[i]), float(vals[i])))
            if len(pts) > 1:
                painter.setPen(QPen(color, 1.2))
                painter.drawPolyline(QPolygonF(pts))

        if self._nearest_idx is not None:
            i = self._nearest_idx
            p = self.to_screen(float(self.index[i]), float(self.series[0][1][i]))
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(p.x(), self.plot_rect().top()), QPointF(p.x(), self.plot_rect().bottom()))
            painter.setPen(QColor(240, 240, 240))
            painter.setFont(mono_font(9))
            painter.drawText(QRectF(self.plot_rect().left() + 6, self.plot_rect().top() + 4, 320, 14),
                             Qt.AlignmentFlag.AlignLeft, self.cursor_text)

    def mouseMoveEvent(self, event):
        if self.index is None or self.index.size == 0:
            return
        x, _ = self.to_data(event.position())
        i = int(np.argmin(np.abs(self.index - x)))
        self._nearest_idx = i
        parts = [f"{'row' if getattr(self, '_axis', 'rows') == 'rows' else 'col'} {int(self.index[i])}"]
        for name, vals in self.series:
            if i < len(vals) and np.isfinite(vals[i]):
                parts.append(f"{name}={vals[i]:.1f}")
        self.cursor_text = "  ".join(parts)
        self.update()

    def mousePressEvent(self, event):
        if self._nearest_idx is not None:
            self.index_clicked.emit(int(self.index[self._nearest_idx]))
