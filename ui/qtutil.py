"""UI 小工具：等宽字体 + 绘制保护。

等宽字体

为什么不用 `QFont("Monospace")` / CSS 里的泛型 `monospace`：这两个名字在 macOS
上都不是真实字体族，Qt 为了别名会去枚举系统字体表，启动时打印

    qt.qpa.fonts: Populating font family aliases took 74 ms.
    Replace uses of missing font family "Monospace" with one that exists ...

这 60~100 ms 是纯浪费。

也不要用 `QFontDatabase.systemFont(FixedFont)`：实测（Qt 6.10 / macOS）
它返回的是 `.AppleSystemUIFont` —— 比例字体、`fixedPitch()` 为 False，而且
内部仍会走 "Monospace" 别名，既不好看也没省下那次枚举。

这里改成从**真实存在**的等宽族里按平台挑一个（macOS 上就是 Menlo），拿不到
再退回逐族探测，最后才用 Courier New 兜底。
"""
from __future__ import annotations

import functools

from PyQt6.QtGui import QFont, QFontDatabase, QFontInfo

__all__ = ["mono_font", "mono_family", "mono_style", "safe_paint"]

_FONT_CACHE: dict = {}
_FAMILY: str = ""

# 各平台常见的等宽族，按优先级排列
_CANDIDATES = (
    "Menlo", "SF Mono",                      # macOS
    "Consolas", "Cascadia Mono",             # Windows
    "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono",
    "Ubuntu Mono", "Ubuntu Sans Mono",       # Linux
    "Courier New", "Courier",
)


def mono_family() -> str:
    """挑一个真实存在的等宽族名（需要 QApplication 已创建）。"""
    global _FAMILY
    if _FAMILY:
        return _FAMILY
    try:
        db = QFontDatabase
        # 第一轮：要求"存在 + 等宽 + 精确匹配"，避免挑到 Qt 的替换字体
        for name in _CANDIDATES:
            if db.hasFamily(name):
                info = QFontInfo(QFont(name))
                if info.fixedPitch() and info.exactMatch():
                    _FAMILY = name
                    return _FAMILY
        # 第二轮：放宽精确匹配
        for name in _CANDIDATES:
            if db.hasFamily(name) and QFontInfo(QFont(name)).fixedPitch():
                _FAMILY = name
                return _FAMILY
        # 第三轮：在所有族里找第一个固定宽度的
        for name in list(db.families())[:400]:
            if QFontInfo(QFont(name)).fixedPitch():
                _FAMILY = name
                return _FAMILY
    except Exception:
        pass
    _FAMILY = "Courier New"
    return _FAMILY


def mono_font(pixel_size: int = 12, bold: bool = False) -> QFont:
    """等宽字体（缓存），替代 QFont("Monospace", ...)。"""
    key = (int(pixel_size), bool(bold))
    font = _FONT_CACHE.get(key)
    if font is None:
        font = QFont(mono_family())
        font.setStyleHint(QFont.StyleHint.Monospace)   # 换机器时让 Qt 选同类
        if pixel_size:
            font.setPixelSize(int(pixel_size))
        font.setBold(bool(bold))
        _FONT_CACHE[key] = font
    return font


def mono_style(extra: str = "") -> str:
    """给样式表用的等宽声明（具体族名，不用泛型 monospace）。"""
    style = f"font-family: '{mono_family()}';"
    return f"{style} {extra}".strip()


def safe_paint(fn):
    """绘制保护。

    窗口/控件正在析构时，Qt 可能仍给已销毁的 C++ 对象派发 paintEvent，
    此时访问 self 的任何 Qt 成员都会抛 `RuntimeError: wrapped C/C++ object
    of type X has been deleted`。绘制过程中抛异常还会连带打印
    "QPaintDevice: Cannot destroy paint device that is being painted"，
    所以这里只吞掉这一种异常，其他异常照旧暴露。
    """
    @functools.wraps(fn)
    def wrapper(self, event):
        try:
            return fn(self, event)
        except RuntimeError:
            return None
    return wrapper
