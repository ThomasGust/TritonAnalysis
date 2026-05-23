"""Stereo calibration GUI for TritonPilot stereo capture sessions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.crab_result_dialog import ImagePreviewPanel
from gui.responsive import resize_to_available_screen, vertical_scroll_area
from stereo_calibration import (
    CharucoBoardSpec,
    CheckerboardSpec,
    calibrate_stereo_from_observations,
    collect_charuco_observations,
    collect_checkerboard_observations,
    manifest_image_pairs,
    write_calibration_artifact,
)


class _SectionCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("stereoCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        layout.addWidget(title_label)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        layout.addLayout(self.body)


class _CalibrationWorker(QThread):
    completed = pyqtSignal(dict, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        manifest_path: Path,
        output_path: Path,
        board,
        board_kind: str,
        min_pairs: int,
        min_corners: int,
        rig_id: str,
        pair_name: str,
        parent=None,
    ):
        super().__init__(parent)
        self.manifest_path = Path(manifest_path)
        self.output_path = Path(output_path)
        self.board = board
        self.board_kind = str(board_kind)
        self.min_pairs = int(min_pairs)
        self.min_corners = int(min_corners)
        self.rig_id = str(rig_id)
        self.pair_name = str(pair_name)

    def run(self) -> None:
        try:
            image_pairs = manifest_image_pairs(self.manifest_path)
            if self.board_kind == "charuco":
                observations = collect_charuco_observations(
                    image_pairs,
                    self.board,
                    min_pairs=self.min_pairs,
                    min_corners=self.min_corners,
                )
            else:
                observations = collect_checkerboard_observations(
                    image_pairs,
                    self.board,
                    min_pairs=self.min_pairs,
                )
            artifact = calibrate_stereo_from_observations(
                observations,
                rig_id=self.rig_id,
                pair_name=self.pair_name,
                board_spec=self.board,
            )
            write_calibration_artifact(artifact, self.output_path)
            self.completed.emit(artifact, str(self.output_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class StereoCalibrationWindow(QMainWindow):
    """Standalone stereo calibration applet for saved left/right image pairs."""

    def __init__(self, manifest_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stereo Calibration")
        self.manifest_path: Path | None = None
        self.manifest: dict = {}
        self.image_pairs: list[tuple[Path, Path]] = []
        self._worker: _CalibrationWorker | None = None

        self._build_ui()
        if manifest_path:
            self.load_manifest(Path(manifest_path))
        resize_to_available_screen(self, 1360, 860, min_width=940, min_height=620)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.setCentralWidget(central)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.manifest_edit = QLineEdit()
        self.manifest_edit.setPlaceholderText("Open TritonPilot stereo manifest.json")
        self.manifest_edit.setReadOnly(True)
        self.open_manifest_btn = QPushButton("Open Manifest")
        self.open_manifest_btn.clicked.connect(self._choose_manifest)
        top.addWidget(self.manifest_edit, 1)
        top.addWidget(self.open_manifest_btn, 0)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        splitter.addWidget(left_panel)

        preview_row = QSplitter(Qt.Orientation.Horizontal)
        self.left_preview = ImagePreviewPanel("Left")
        self.right_preview = ImagePreviewPanel("Right")
        preview_row.addWidget(self.left_preview)
        preview_row.addWidget(self.right_preview)
        left_layout.addWidget(preview_row, 2)

        self.pairs_table = QTableWidget(0, 5)
        self.pairs_table.setHorizontalHeaderLabels(["#", "Delta", "Left Seq", "Right Seq", "Stem"])
        self.pairs_table.verticalHeader().hide()
        self.pairs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pairs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pairs_table.currentCellChanged.connect(lambda row, *_args: self._show_pair(row))
        left_layout.addWidget(self.pairs_table, 1)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        splitter.addWidget(vertical_scroll_area(controls))

        session_card = _SectionCard("Session")
        session_grid = QGridLayout()
        self.pair_lbl = self._value_label()
        self.rig_lbl = self._value_label()
        self.image_size_lbl = self._value_label()
        self.frame_count_lbl = self._value_label()
        for row, (label_text, value) in enumerate(
            [
                ("Pair", self.pair_lbl),
                ("Rig", self.rig_lbl),
                ("Image Size", self.image_size_lbl),
                ("Frames", self.frame_count_lbl),
            ]
        ):
            session_grid.addWidget(QLabel(label_text), row, 0)
            session_grid.addWidget(value, row, 1)
        session_card.body.addLayout(session_grid)
        controls_layout.addWidget(session_card)

        board_card = _SectionCard("Board")
        self.board_type_combo = QComboBox()
        self.board_type_combo.addItems(["Checkerboard", "ChArUco"])
        self.board_type_combo.setCurrentText("ChArUco")
        self.board_type_combo.currentTextChanged.connect(self._refresh_board_mode)
        board_card.body.addWidget(self.board_type_combo)

        board_form = QFormLayout()
        self.columns_spin = self._int_spin(2, 80, 9)
        self.rows_spin = self._int_spin(2, 80, 6)
        self.squares_x_spin = self._int_spin(2, 80, 24)
        self.squares_y_spin = self._int_spin(2, 80, 17)
        self.square_size_spin = self._float_spin(0.001, 1000.0, 30.0, 3)
        self.marker_size_spin = self._float_spin(0.001, 1000.0, 22.0, 3)
        self.units_edit = QLineEdit("mm")
        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_250"])
        board_form.addRow("Checker Columns", self.columns_spin)
        board_form.addRow("Checker Rows", self.rows_spin)
        board_form.addRow("ChArUco Squares X", self.squares_x_spin)
        board_form.addRow("ChArUco Squares Y", self.squares_y_spin)
        board_form.addRow("Square Size", self.square_size_spin)
        board_form.addRow("Marker Size", self.marker_size_spin)
        board_form.addRow("Units", self.units_edit)
        board_form.addRow("Dictionary", self.dictionary_combo)
        board_card.body.addLayout(board_form)
        controls_layout.addWidget(board_card)

        run_card = _SectionCard("Calibration")
        run_form = QFormLayout()
        self.min_pairs_spin = self._int_spin(1, 500, 8)
        self.min_corners_spin = self._int_spin(4, 500, 8)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("auto: stereo_calibration.json next to manifest")
        self.output_btn = QPushButton("Choose Output")
        self.output_btn.clicked.connect(self._choose_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_btn, 0)
        run_form.addRow("Min Pairs", self.min_pairs_spin)
        run_form.addRow("Min ChArUco Corners", self.min_corners_spin)
        run_form.addRow("Output", output_row)
        run_card.body.addLayout(run_form)
        self.calibrate_btn = QPushButton("Run Calibration")
        self.calibrate_btn.clicked.connect(self._run_calibration)
        self.calibrate_btn.setEnabled(False)
        run_card.body.addWidget(self.calibrate_btn)
        controls_layout.addWidget(run_card)

        result_card = _SectionCard("Result")
        self.result_lbl = QLabel("Open a manifest to begin.")
        self.result_lbl.setObjectName("summaryHint")
        self.result_lbl.setWordWrap(True)
        result_card.body.addWidget(self.result_lbl)
        self.rejected_table = QTableWidget(0, 2)
        self.rejected_table.setHorizontalHeaderLabels(["Pair", "Reason"])
        self.rejected_table.verticalHeader().hide()
        self.rejected_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rejected_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        result_card.body.addWidget(self.rejected_table)
        controls_layout.addWidget(result_card, 1)
        controls_layout.addStretch(1)

        splitter.setSizes([900, 420])
        self._refresh_board_mode()

    def _value_label(self) -> QLabel:
        label = QLabel("-")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def _int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _float_spin(self, minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        return spin

    def _refresh_board_mode(self) -> None:
        charuco = self.board_type_combo.currentText().lower().startswith("ch")
        for widget in (self.squares_x_spin, self.squares_y_spin, self.marker_size_spin, self.dictionary_combo, self.min_corners_spin):
            widget.setEnabled(charuco)
        for widget in (self.columns_spin, self.rows_spin):
            widget.setEnabled(not charuco)

    def _choose_manifest(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open TritonPilot stereo manifest",
            str(Path.cwd()),
            "Stereo manifest (manifest.json);;JSON files (*.json);;All files (*)",
        )
        if path:
            self.load_manifest(Path(path))

    def _choose_output(self) -> None:
        start = self._default_output_path()
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save stereo calibration artifact",
            str(start),
            "Calibration JSON (*.json);;All files (*)",
        )
        if path:
            self.output_edit.setText(str(Path(path)))

    def _default_output_path(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path.parent / "stereo_calibration.json"
        return Path.cwd() / "stereo_calibration.json"

    def load_manifest(self, path: Path) -> None:
        self.manifest_path = Path(path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.image_pairs = manifest_image_pairs(self.manifest_path)
        self.manifest_edit.setText(str(self.manifest_path))
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(self._default_output_path()))
        self._populate_session()
        self._populate_pairs_table()
        self.calibrate_btn.setEnabled(bool(self.image_pairs))
        if self.image_pairs:
            self.pairs_table.selectRow(0)
            self._show_pair(0)
        self.statusBar().showMessage(f"Loaded {len(self.image_pairs)} stereo pair(s)", 4000)

    def _populate_session(self) -> None:
        pair = self.manifest.get("pair") or {}
        self.pair_lbl.setText(str(pair.get("name") or self.manifest.get("pair_name") or "-"))
        self.rig_lbl.setText(str(pair.get("rig_id") or "-"))
        frames = self.manifest.get("frames") or []
        self.frame_count_lbl.setText(str(len(frames)))
        size = "-"
        first = frames[0] if frames else {}
        shape = (first.get("left") or {}).get("shape")
        if isinstance(shape, list) and len(shape) >= 2:
            size = f"{shape[1]}x{shape[0]}"
        self.image_size_lbl.setText(size)

    def _populate_pairs_table(self) -> None:
        frames = self.manifest.get("frames") or []
        self.pairs_table.setRowCount(0)
        for row, frame in enumerate(frames):
            self.pairs_table.insertRow(row)
            values = [
                str(frame.get("index", row + 1)),
                f"{float(frame.get('pair_delta_ms', 0.0)):.1f} ms",
                str((frame.get("left") or {}).get("seq", "-")),
                str((frame.get("right") or {}).get("seq", "-")),
                str(frame.get("stem", "")),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pairs_table.setItem(row, col, item)
        self.pairs_table.resizeColumnsToContents()

    def _show_pair(self, row: int) -> None:
        if row < 0 or row >= len(self.image_pairs):
            self.left_preview.clear("No pair selected")
            self.right_preview.clear("No pair selected")
            return
        left_path, right_path = self.image_pairs[row]
        self.left_preview.set_frame(cv2.imread(str(left_path)), placeholder_text="Left image missing")
        self.right_preview.set_frame(cv2.imread(str(right_path)), placeholder_text="Right image missing")

    def _board_spec(self):
        units = self.units_edit.text().strip() or "cm"
        if self.board_type_combo.currentText().lower().startswith("ch"):
            return "charuco", CharucoBoardSpec(
                squares_x=self.squares_x_spin.value(),
                squares_y=self.squares_y_spin.value(),
                square_size=float(self.square_size_spin.value()),
                marker_size=float(self.marker_size_spin.value()),
                dictionary=self.dictionary_combo.currentText(),
                units=units,
            )
        return "checkerboard", CheckerboardSpec(
            columns=self.columns_spin.value(),
            rows=self.rows_spin.value(),
            square_size=float(self.square_size_spin.value()),
            units=units,
        )

    def _run_calibration(self) -> None:
        if self.manifest_path is None:
            return
        board_kind, board = self._board_spec()
        pair = self.manifest.get("pair") or {}
        self._worker = _CalibrationWorker(
            manifest_path=self.manifest_path,
            output_path=Path(self.output_edit.text().strip() or self._default_output_path()),
            board=board,
            board_kind=board_kind,
            min_pairs=self.min_pairs_spin.value(),
            min_corners=self.min_corners_spin.value(),
            rig_id=str(pair.get("rig_id") or "stereo_rig"),
            pair_name=str(pair.get("name") or "stereo_pair"),
            parent=self,
        )
        self._worker.completed.connect(self._on_calibration_completed)
        self._worker.failed.connect(self._on_calibration_failed)
        self._worker.finished.connect(lambda: self.calibrate_btn.setEnabled(True))
        self.calibrate_btn.setEnabled(False)
        self.result_lbl.setText("Running stereo calibration...")
        self.statusBar().showMessage("Stereo calibration running", 3000)
        self._worker.start()

    def _on_calibration_completed(self, artifact: dict, output_path: str) -> None:
        rms = artifact.get("rms") or {}
        stereo = artifact.get("stereo") or {}
        board = artifact.get("board") or {}
        units = board.get("units") or ""
        self.result_lbl.setText(
            "Accepted: {accepted} | Stereo RMS: {rms:.4f} | Baseline: {baseline:.4f} {units}\n{path}".format(
                accepted=artifact.get("observation_count", 0),
                rms=float(rms.get("stereo", 0.0)),
                baseline=float(stereo.get("baseline", 0.0)),
                units=units,
                path=output_path,
            )
        )
        self._populate_rejections(artifact.get("rejected_observations") or [])
        self.statusBar().showMessage(f"Calibration saved: {output_path}", 7000)

    def _on_calibration_failed(self, error: str) -> None:
        self.result_lbl.setText(f"Calibration failed: {error}")
        self.statusBar().showMessage(f"Calibration failed: {error}", 7000)

    def _populate_rejections(self, rejected: list[dict]) -> None:
        self.rejected_table.setRowCount(0)
        for row, item in enumerate(rejected):
            self.rejected_table.insertRow(row)
            values = [str(item.get("index", row + 1)), str(item.get("reason", ""))]
            for col, value in enumerate(values):
                self.rejected_table.setItem(row, col, QTableWidgetItem(value))
        self.rejected_table.resizeColumnsToContents()
