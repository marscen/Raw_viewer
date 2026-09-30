"""打开 RAW 的参数对话框。

老版本只有 宽/高/位深/pattern 四个框，且 pattern 默认落在 RGGB（灰度数据会被
误显示成彩色网点），文件比 W*H 大时还会被静默截断。这里补齐测试现场真正需要
的东西：

  * 布局参数：packing（MIPI RAW10/12/14 packed）、header 字节、行 stride、
    字节序、容器左对齐位移 data_shift、多帧文件的帧号/帧 stride；
  * **自动推断**：由文件大小反推可能的 分辨率/位深/packing（含行 padding），
    一键填入；
  * **实时校验**：随时显示"需要多少字节 / 文件实际多少字节 / 差多少"，
    尺寸不匹配当场看得见，不会等到加载后才失败；
  * 参数预设：常用的几种 dump 布局存成预设，下次一键套用（QSettings 持久化）。
"""
from __future__ import annotations

import json

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                             QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget)

from utils.raw_io import (PACKING_MODES, RawLoadSpec, describe_file,
                          expected_size, row_bytes, suggest_geometries)

SETTINGS_PRESET_KEY = "raw_viewer/presets"


def _settings():
    """和主窗口共用同一个设置后端（支持 RAWV2_SETTINGS_DIR 便携模式/测试隔离）。"""
    from ui.main_window import make_settings
    return make_settings()


def load_presets() -> dict:
    settings = _settings()
    raw = settings.value(SETTINGS_PRESET_KEY, "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def save_presets(presets: dict):
    _settings().setValue(SETTINGS_PRESET_KEY, json.dumps(presets, ensure_ascii=False))


class ImageParamsDialog(QDialog):
    """打开文件时确认/推断 RAW 布局参数。"""

    def __init__(self, parent=None, path: str | None = None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("RAW 参数 / RAW Layout")
        self.setModal(True)
        self.resize(560, 700)
        self.path = path
        self._presets = load_presets()
        initial = dict(initial or {})

        root = QVBoxLayout(self)

        # ---- 文件信息 ----
        info_group = QGroupBox("文件 / File")
        info_form = QFormLayout(info_group)
        self.file_label = QLabel("-")
        self.file_label.setWordWrap(True)
        info_form.addRow("文件:", self.file_label)
        self.size_label = QLabel("-")
        info_form.addRow("大小:", self.size_label)
        root.addWidget(info_group)

        # ---- 自动推断 ----
        detect_group = QGroupBox("自动推断布局 / Auto detect")
        detect_layout = QVBoxLayout(detect_group)
        row = QHBoxLayout()
        self.detect_btn = QPushButton("按文件大小推断 (Detect)")
        self.detect_btn.clicked.connect(self.run_detect)
        row.addWidget(self.detect_btn)
        row.addWidget(QLabel("（双击候选即应用）"))
        detect_layout.addLayout(row)
        self.candidate_list = QListWidget()
        self.candidate_list.setMaximumHeight(120)
        self.candidate_list.itemDoubleClicked.connect(self.apply_candidate)
        detect_layout.addWidget(self.candidate_list)
        root.addWidget(detect_group)

        # ---- 布局参数 ----
        layout_group = QGroupBox("布局参数 / Layout")
        form = QFormLayout(layout_group)

        self.width_spin = QSpinBox(); self.width_spin.setRange(1, 100000)
        self.height_spin = QSpinBox(); self.height_spin.setRange(1, 100000)
        self.bit_depth_combo = QComboBox(); self.bit_depth_combo.addItems(["8", "10", "12", "14", "16"])
        self.packing_combo = QComboBox(); self.packing_combo.addItems(["unpacked", "packed (MIPI)"])
        self.header_spin = QSpinBox(); self.header_spin.setRange(0, 1 << 24); self.header_spin.setSingleStep(4)
        self.stride_spin = QSpinBox(); self.stride_spin.setRange(0, 1 << 24)
        self.stride_spin.setToolTip("0 = 紧凑排布（无行 padding）")
        self.endian_combo = QComboBox(); self.endian_combo.addItems(["little", "big"])
        self.shift_spin = QSpinBox(); self.shift_spin.setRange(0, 16)
        self.shift_spin.setToolTip("16bit 容器左对齐存放 10/12/14bit 时右移的位数")
        self.frame_spin = QSpinBox(); self.frame_spin.setRange(0, 100000)
        self.frame_stride_spin = QSpinBox(); self.frame_stride_spin.setRange(0, 1 << 30)
        self.frame_stride_spin.setToolTip("0 = 每帧紧凑（W*H*bpp）")

        form.addRow("Width:", self.width_spin)
        form.addRow("Height:", self.height_spin)
        form.addRow("Bit Depth:", self.bit_depth_combo)
        form.addRow("Packing:", self.packing_combo)
        form.addRow("Header (bytes):", self.header_spin)
        form.addRow("Row stride (bytes):", self.stride_spin)
        form.addRow("Byte order:", self.endian_combo)
        form.addRow("Data shift >>:", self.shift_spin)
        form.addRow("Frame index:", self.frame_spin)
        form.addRow("Frame stride:", self.frame_stride_spin)
        root.addWidget(layout_group)

        # ---- 显示与 CFA ----
        view_group = QGroupBox("显示 / Display")
        vform = QFormLayout(view_group)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Mono/None", "RGGB", "BGGR", "GRBG", "GBRG"])
        self.pattern_combo.setCurrentText("Mono/None")   # 默认灰度：避免灰度数据被染成彩色网点
        self.view_combo = QComboBox()
        from utils.display import VIEW_MODES
        self.view_combo.addItems(VIEW_MODES)
        self.black_spin = QDoubleSpinBox(); self.black_spin.setRange(-65535, 65535); self.black_spin.setDecimals(0)
        self.white_spin = QDoubleSpinBox(); self.white_spin.setRange(0, 1 << 20); self.white_spin.setDecimals(0)
        self.white_spin.setToolTip("0 = 自动使用 (2^bit_depth - 1)")
        vform.addRow("Bayer Pattern:", self.pattern_combo)
        vform.addRow("初始视图:", self.view_combo)
        vform.addRow("黑电平 (DN):", self.black_spin)
        vform.addRow("白电平 (DN):", self.white_spin)
        root.addWidget(view_group)

        # ---- 预设 ----
        preset_group = QGroupBox("预设 / Presets")
        prow = QHBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(self._presets.keys()))
        apply_btn = QPushButton("套用")
        apply_btn.clicked.connect(self.apply_preset)
        save_btn = QPushButton("保存当前…")
        save_btn.clicked.connect(self.save_current_preset)
        prow.addWidget(self.preset_combo, 1)
        prow.addWidget(apply_btn)
        prow.addWidget(save_btn)
        root.addWidget(preset_group)

        # ---- 校验结果 ----
        self.check_label = QLabel("-")
        self.check_label.setWordWrap(True)
        root.addWidget(self.check_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                  QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        # 初始值
        self.width_spin.setValue(int(initial.get("width", 1920)))
        self.height_spin.setValue(int(initial.get("height", 1080)))
        self.bit_depth_combo.setCurrentText(str(initial.get("bit_depth", 10)))
        self.pattern_combo.setCurrentText(initial.get("pattern", "Mono/None"))
        self.view_combo.setCurrentText(initial.get("view", "Mono"))
        self.black_spin.setValue(float(initial.get("black_level", 0)))
        self.white_spin.setValue(float(initial.get("white_level", 0)))
        self.packing_combo.setCurrentText("packed (MIPI)" if initial.get("packing") == "packed" else "unpacked")
        self.header_spin.setValue(int(initial.get("header_bytes", 0)))
        self.stride_spin.setValue(int(initial.get("stride_bytes", 0)))
        self.endian_combo.setCurrentText(initial.get("endian", "little"))
        self.shift_spin.setValue(int(initial.get("data_shift", 0)))
        self.frame_spin.setValue(int(initial.get("frame_index", 0)))
        self.frame_stride_spin.setValue(int(initial.get("frame_stride_bytes", 0)))

        for w in (self.width_spin, self.height_spin, self.header_spin, self.stride_spin,
                  self.frame_spin, self.frame_stride_spin, self.shift_spin):
            w.valueChanged.connect(self.update_check)
        for c in (self.bit_depth_combo, self.packing_combo, self.endian_combo,
                  self.pattern_combo, self.view_combo):
            c.currentTextChanged.connect(self.update_check)
        self.white_spin.valueChanged.connect(self.update_check)
        self.black_spin.valueChanged.connect(self.update_check)

        if path:
            f = describe_file(path)
            self.file_label.setText(f["path"])
            self.size_label.setText(f"{f['size']:,} bytes ({f['size_mb']:.2f} MB)")
            self.run_detect()
        else:
            self.size_label.setText("(未选择文件)")
        self.update_check()

    # ------------------------------------------------------------------
    def spec(self) -> RawLoadSpec:
        return RawLoadSpec(
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            bit_depth=int(self.bit_depth_combo.currentText()),
            packing="packed" if self.packing_combo.currentText().startswith("packed") else "unpacked",
            header_bytes=self.header_spin.value(),
            stride_bytes=self.stride_spin.value(),
            endian=self.endian_combo.currentText(),
            data_shift=self.shift_spin.value(),
            frame_index=self.frame_spin.value(),
            frame_stride_bytes=self.frame_stride_spin.value(),
        )

    def get_params(self) -> dict:
        spec = self.spec()
        view = self.view_combo.currentText()
        pattern = self.pattern_combo.currentText()
        return {
            "width": spec.width, "height": spec.height, "bit_depth": spec.bit_depth,
            "packing": spec.packing, "header_bytes": spec.header_bytes,
            "stride_bytes": spec.stride_bytes, "endian": spec.endian,
            "data_shift": spec.data_shift, "frame_index": spec.frame_index,
            "frame_stride_bytes": spec.frame_stride_bytes,
            "pattern": pattern, "view": view,
            "black_level": self.black_spin.value(),
            "white_level": self.white_spin.value(),
            # 兼容旧字段
            "demosaic": view == "Bayer Demosaic",
        }

    def set_params(self, params: dict):
        self.width_spin.setValue(int(params.get("width", self.width_spin.value())))
        self.height_spin.setValue(int(params.get("height", self.height_spin.value())))
        self.bit_depth_combo.setCurrentText(str(params.get("bit_depth", self.bit_depth_combo.currentText())))
        if "packing" in params:
            self.packing_combo.setCurrentText("packed (MIPI)" if params["packing"] == "packed" else "unpacked")
        self.header_spin.setValue(int(params.get("header_bytes", self.header_spin.value())))
        self.stride_spin.setValue(int(params.get("stride_bytes", self.stride_spin.value())))
        self.endian_combo.setCurrentText(params.get("endian", self.endian_combo.currentText()))
        self.shift_spin.setValue(int(params.get("data_shift", self.shift_spin.value())))
        self.frame_spin.setValue(int(params.get("frame_index", self.frame_spin.value())))
        self.frame_stride_spin.setValue(int(params.get("frame_stride_bytes", self.frame_stride_spin.value())))
        if "pattern" in params:
            self.pattern_combo.setCurrentText(params["pattern"])
        if "view" in params:
            self.view_combo.setCurrentText(params["view"])
        if "black_level" in params:
            self.black_spin.setValue(float(params["black_level"]))
        if "white_level" in params:
            self.white_spin.setValue(float(params["white_level"]))

    # ------------------------------------------------------------------
    def run_detect(self):
        self.candidate_list.clear()
        if not self.path:
            return
        size = describe_file(self.path)["size"]
        candidates = suggest_geometries(size)
        if not candidates:
            item = QListWidgetItem("未能由文件大小唯一推断，请手动填写")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.candidate_list.addItem(item)
            return
        for c in candidates:
            s = c["spec"]
            text = (f"{s.width}x{s.height}  {s.bit_depth}bit  {s.packing}"
                    f"{f'  stride={s.stride_bytes}B' if s.stride_bytes else ''}"
                    f"   [{c['confidence']}] {c['note']}")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, c)
            self.candidate_list.addItem(item)

    def apply_candidate(self, item):
        c = item.data(Qt.ItemDataRole.UserRole)
        if not c:
            return
        self.apply_spec(c["spec"])

    def apply_spec(self, spec: RawLoadSpec):
        self.width_spin.setValue(spec.width)
        self.height_spin.setValue(spec.height)
        self.bit_depth_combo.setCurrentText(str(spec.bit_depth))
        self.packing_combo.setCurrentText("packed (MIPI)" if spec.packing == "packed" else "unpacked")
        self.header_spin.setValue(spec.header_bytes)
        self.stride_spin.setValue(spec.stride_bytes)
        self.endian_combo.setCurrentText(spec.endian)
        self.shift_spin.setValue(spec.data_shift)
        self.frame_spin.setValue(spec.frame_index)
        self.frame_stride_spin.setValue(spec.frame_stride_bytes)
        self.update_check()

    def apply_preset(self):
        name = self.preset_combo.currentText()
        if name and name in self._presets:
            self.set_params(self._presets[name])
            self.update_check()

    def save_current_preset(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "保存预设", "预设名称：")
        if not ok or not name.strip():
            return
        self._presets[name.strip()] = self.get_params()
        save_presets(self._presets)
        self.preset_combo.clear()
        self.preset_combo.addItems(sorted(self._presets.keys()))
        self.preset_combo.setCurrentText(name.strip())

    # ------------------------------------------------------------------
    def update_check(self):
        """实时显示尺寸校验结果，避免"文件大小不符"到加载时才暴露。"""
        try:
            spec = self.spec()
            need = expected_size(spec)
            rb = row_bytes(spec.width, spec.bit_depth, spec.packing, spec.stride_bytes)
        except Exception as exc:
            self.check_label.setStyleSheet("color: #ff8080;")
            self.check_label.setText(f"参数不合法：{exc}")
            return
        parts = [f"行字节：{rb}B"]
        if self.stride_spin.value():
            tight = row_bytes(spec.width, spec.bit_depth, spec.packing)
            parts.append(f"行 padding：{self.stride_spin.value() - tight}B")
        parts.append(f"单帧：{spec.frame_bytes():,}B")
        parts.append(f"需要：{need:,}B")
        if self.path:
            size = describe_file(self.path)["size"]
            diff = size - need
            parts.append(f"文件：{size:,}B")
            if diff == 0:
                self.check_label.setStyleSheet("color: #7ddc7d;")
                parts.append("✔ 完全匹配")
            elif diff > 0:
                self.check_label.setStyleSheet("color: #ffd479;")
                parts.append(f"文件多 {diff:,}B（可能是 header/多帧/尾部数据）")
            else:
                self.check_label.setStyleSheet("color: #ff8080;")
                parts.append(f"✘ 文件少 {-diff:,}B，无法读取")
        else:
            self.check_label.setStyleSheet("")
        self.check_label.setText("  |  ".join(parts))
