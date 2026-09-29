"""坐标跳转输入框：输入 (X, Y) 后把画布定位到该像素。

设计取舍（按测试现场的习惯）：
  * 用一个输入框而不是两个 spinbox：缺陷坐标常常是从日志/CSV/Excel 里复制过来的
    （"1234,567"、"x=1234 y=567"、"(1234, 567)"），一个框能直接粘贴；
  * 支持 **1-based** 开关：Matlab / Excel / 部分测试报告的行列号从 1 开始，
    差 1 像素在坏点定位上就是致命的；
  * 跳转时可以顺手指定缩放倍数（定位坏点通常要放大到 20× 以上才看得清相位）；
  * 在画布上单击像素会把坐标回填到这个框里，方便"从这里再跳过去"。
"""
from __future__ import annotations

import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QWidget)

# 缩放选项：None = 保持当前缩放
ZOOM_CHOICES = [
    ("保持当前缩放", None),
    ("1:1", 1.0),
    ("8×", 8.0),
    ("20×", 20.0),
    ("50×", 50.0),
    ("100×", 100.0),
]

_NUMBER_RE = re.compile(r"\d+")


class CoordinateJumpBox(QWidget):
    """X/Y 输入 + 跳转按钮 + 缩放选择。"""

    jump_requested = pyqtSignal(int, int, object)   # x, y, scale(None=不变)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("坐标 X,Y:"))
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("如 1234,567")
        self.edit.setFixedWidth(130)
        self.edit.setToolTip(
            "支持 1234,567 / 1234 567 / (1234, 567) / x=1234 y=567\n"
            "X = 列号，Y = 行号，默认从 0 开始（可勾选 1-based）\n"
            "回车即跳转；Ctrl+G 可快速聚焦到本输入框")
        self.edit.returnPressed.connect(self._on_go)
        self.edit.textChanged.connect(self._validate)
        layout.addWidget(self.edit)

        self.one_based = QCheckBox("1-based")
        self.one_based.setToolTip("勾选后输入的坐标按“从 1 开始”解释（Matlab/Excel 习惯）")
        self.one_based.toggled.connect(self._validate)
        layout.addWidget(self.one_based)

        self.zoom_combo = QComboBox()
        for text, _value in ZOOM_CHOICES:
            self.zoom_combo.addItem(text)
        self.zoom_combo.setCurrentIndex(3)          # 默认 20×，适合看像素相位
        self.zoom_combo.setToolTip("跳转时同时把缩放设成该倍数")
        layout.addWidget(self.zoom_combo)

        self.go_btn = QPushButton("跳转")
        self.go_btn.setToolTip("定位到输入坐标 (Ctrl+G 聚焦输入框，回车跳转)")
        self.go_btn.clicked.connect(self._on_go)
        layout.addWidget(self.go_btn)

        self._validate()

    # ------------------------------------------------------------------
    def parse(self):
        """解析输入，返回 (x, y)（始终是 0-based 的内部坐标），非法返回 None。"""
        nums = _NUMBER_RE.findall(self.edit.text() or "")
        if len(nums) < 2:
            return None
        x, y = int(nums[0]), int(nums[1])
        if self.one_based.isChecked():
            x, y = x - 1, y - 1
        if x < 0 or y < 0:
            return None
        return x, y

    def selected_zoom(self):
        return ZOOM_CHOICES[self.zoom_combo.currentIndex()][1]

    def set_coordinate(self, x: int, y: int):
        """由画布点击/Tab 切换回填坐标（内部坐标 0-based）。"""
        if x is None or y is None:
            return
        if self.one_based.isChecked():
            x, y = x + 1, y + 1
        text = f"{int(x)},{int(y)}"
        if self.edit.text() != text:
            blocked = self.edit.blockSignals(True)
            self.edit.setText(text)
            self.edit.blockSignals(blocked)
        self._validate()

    def clear(self):
        self.edit.clear()

    def focus_input(self):
        self.edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.edit.selectAll()

    # ------------------------------------------------------------------
    def _validate(self):
        ok = self.parse() is not None
        self.edit.setStyleSheet("" if ok else "border: 1px solid #ff6b6b;")
        self.go_btn.setEnabled(ok)

    def _on_go(self):
        parsed = self.parse()
        if parsed is None:
            return
        self.jump_requested.emit(int(parsed[0]), int(parsed[1]), self.selected_zoom())
