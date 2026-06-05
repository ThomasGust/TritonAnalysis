"""Stereo calibration GUI for TritonPilot stereo capture sessions."""

from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
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

from triton_analysis.gui.file_dialogs import ThumbnailFileDialog as QFileDialog

from triton_analysis.workspace import latest_pilot_stereo_sessions_dir, workspace_paths
from triton_analysis.gui.image_preview import ImagePreviewPanel
from triton_analysis.gui.responsive import resize_to_available_screen, vertical_scroll_area
from triton_analysis.stereo.calibration import (
    CHARUCO_DICTIONARIES,
    DEFAULT_CHARUCO_DICTIONARY,
    DEFAULT_CHARUCO_MARKER_SIZE,
    DEFAULT_CHARUCO_SQUARES_X,
    DEFAULT_CHARUCO_SQUARES_Y,
    DEFAULT_CHARUCO_SQUARE_SIZE,
    CharucoBoardSpec,
    CheckerboardSpec,
    annotate_board_detection,
    calibrate_stereo_from_observations,
    collect_charuco_observations,
    collect_checkerboard_observations,
    detect_stereo_board,
    load_manifest_collection,
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
        manifest_paths: list[Path],
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
        self.manifest_paths = [Path(path) for path in manifest_paths]
        self.output_path = Path(output_path)
        self.board = board
        self.board_kind = str(board_kind)
        self.min_pairs = int(min_pairs)
        self.min_corners = int(min_corners)
        self.rig_id = str(rig_id)
        self.pair_name = str(pair_name)

    def run(self) -> None:
        try:
            _manifest, image_pairs = load_manifest_collection(self.manifest_paths)
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
        self.manifest_paths: list[Path] = []
        self.manifest: dict = {}
        self.image_pairs: list[tuple[Path, Path]] = []
        self._worker: _CalibrationWorker | None = None
        self._checker_rows: list[tuple[QLabel, QWidget]] = []
        self._charuco_rows: list[tuple[QLabel, QWidget]] = []

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
        self.manifest_edit.setPlaceholderText("Open TritonPilot stereo manifest(s) or a session folder")
        self.manifest_edit.setReadOnly(True)
        self.open_manifest_btn = QPushButton("Open Manifest(s)")
        self.open_manifest_btn.clicked.connect(self._choose_manifest)
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self._choose_manifest_folder)
        top.addWidget(self.manifest_edit, 1)
        top.addWidget(self.open_manifest_btn, 0)
        top.addWidget(self.open_folder_btn, 0)
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

        self.detection_lbl = QLabel("Open a manifest to inspect board detections.")
        self.detection_lbl.setObjectName("summaryHint")
        self.detection_lbl.setWordWrap(True)
        self.detection_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left_layout.addWidget(self.detection_lbl, 0)

        self.pairs_table = QTableWidget(0, 8)
        self.pairs_table.setHorizontalHeaderLabels(["#", "Delta", "Left Seq", "Right Seq", "Matched", "Status", "Stem", "Source"])
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
        self.squares_x_spin = self._int_spin(2, 80, DEFAULT_CHARUCO_SQUARES_X)
        self.squares_y_spin = self._int_spin(2, 80, DEFAULT_CHARUCO_SQUARES_Y)
        self.square_size_spin = self._float_spin(
            0.001,
            1000.0,
            DEFAULT_CHARUCO_SQUARE_SIZE,
            3,
        )
        self.marker_size_spin = self._float_spin(
            0.001,
            1000.0,
            DEFAULT_CHARUCO_MARKER_SIZE,
            3,
        )
        self.units_edit = QLineEdit("cm")
        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(CHARUCO_DICTIONARIES)
        self.dictionary_combo.setCurrentText(DEFAULT_CHARUCO_DICTIONARY)
        self._checker_rows.extend(
            [
                self._add_form_row(board_form, "Checker Columns", self.columns_spin),
                self._add_form_row(board_form, "Checker Rows", self.rows_spin),
            ]
        )
        self._charuco_rows.extend(
            [
                self._add_form_row(board_form, "ChArUco Squares X", self.squares_x_spin),
                self._add_form_row(board_form, "ChArUco Squares Y", self.squares_y_spin),
            ]
        )
        self._add_form_row(board_form, "Square Size", self.square_size_spin)
        self._charuco_rows.append(self._add_form_row(board_form, "Marker Size", self.marker_size_spin))
        self._add_form_row(board_form, "Units", self.units_edit)
        self._charuco_rows.append(self._add_form_row(board_form, "Dictionary", self.dictionary_combo))
        board_card.body.addLayout(board_form)
        controls_layout.addWidget(board_card)

        run_card = _SectionCard("Calibration")
        run_form = QFormLayout()
        self.min_pairs_spin = self._int_spin(1, 500, 8)
        self.min_corners_spin = self._int_spin(4, 500, 24)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("auto: stereo_calibration.json next to manifest")
        self.output_btn = QPushButton("Choose Output")
        self.output_btn.clicked.connect(self._choose_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_btn, 0)
        self._add_form_row(run_form, "Min Pairs", self.min_pairs_spin)
        self._charuco_rows.append(self._add_form_row(run_form, "Min ChArUco Corners", self.min_corners_spin))
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
        self.quality_table = QTableWidget(0, 2)
        self.quality_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.quality_table.verticalHeader().hide()
        self.quality_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.quality_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.quality_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.quality_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        result_card.body.addWidget(self.quality_table)
        self.rejected_table = QTableWidget(0, 2)
        self.rejected_table.setHorizontalHeaderLabels(["Pair", "Reason"])
        self.rejected_table.verticalHeader().hide()
        self.rejected_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rejected_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        result_card.body.addWidget(self.rejected_table)
        controls_layout.addWidget(result_card, 1)
        controls_layout.addStretch(1)

        splitter.setSizes([900, 420])
        self._connect_detection_controls()
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

    def _add_form_row(self, form: QFormLayout, label_text: str, field: QWidget) -> tuple[QLabel, QWidget]:
        label = QLabel(label_text)
        form.addRow(label, field)
        return label, field

    def _set_form_rows_visible(self, rows: list[tuple[QLabel, QWidget]], visible: bool) -> None:
        for label, field in rows:
            label.setVisible(visible)
            field.setVisible(visible)
            field.setEnabled(visible)

    def _refresh_board_mode(self) -> None:
        charuco = self.board_type_combo.currentText().lower().startswith("ch")
        self._set_form_rows_visible(self._charuco_rows, charuco)
        self._set_form_rows_visible(self._checker_rows, not charuco)
        self._refresh_current_pair_detection()

    def _connect_detection_controls(self) -> None:
        for spin in (
            self.columns_spin,
            self.rows_spin,
            self.squares_x_spin,
            self.squares_y_spin,
            self.square_size_spin,
            self.marker_size_spin,
            self.min_corners_spin,
        ):
            spin.valueChanged.connect(lambda _value: self._refresh_current_pair_detection())
        self.dictionary_combo.currentTextChanged.connect(lambda _text: self._refresh_current_pair_detection())

    def _choose_manifest(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(
            self,
            "Open TritonPilot stereo manifest(s)",
            str(self._stereo_session_start()),
            "Stereo manifest (manifest.json);;JSON files (*.json);;All files (*)",
        )
        if paths:
            self.load_manifests([Path(path) for path in paths])

    def _choose_manifest_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Open TritonPilot stereo session folder",
            str(self._stereo_session_start()),
        )
        if path:
            self.load_manifests([Path(path)])

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
        output_dir = workspace_paths(create=True).calibrations
        if self.manifest_paths:
            first = self.manifest_paths[0]
            session_name = first.parent.name if first.name.lower() == "manifest.json" else first.stem
            return output_dir / f"{session_name}_stereo_calibration.json"
        return output_dir / "stereo_calibration.json"

    @staticmethod
    def _stereo_session_start() -> Path:
        return latest_pilot_stereo_sessions_dir(create=True)

    def load_manifest(self, path: Path) -> None:
        self.load_manifests([path])

    def load_manifests(self, paths: list[Path]) -> None:
        self.manifest, self.image_pairs = load_manifest_collection(paths)
        self.manifest_paths = [Path(source["path"]) for source in self.manifest.get("sources", [])]
        source_count = len(self.manifest_paths)
        if source_count == 1:
            self.manifest_edit.setText(str(self.manifest_paths[0]))
        else:
            first_parent = self.manifest_paths[0].parent if self.manifest_paths else self._stereo_session_start()
            self.manifest_edit.setText(f"{source_count} manifests | {first_parent}")
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(self._default_output_path()))
        self._populate_session()
        self._populate_pairs_table()
        self.calibrate_btn.setEnabled(bool(self.image_pairs))
        if self.image_pairs:
            self.pairs_table.selectRow(0)
            self._show_pair(0)
        self.statusBar().showMessage(
            f"Loaded {len(self.image_pairs)} stereo pair(s) from {source_count} manifest(s)",
            4000,
        )

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
                "-",
                "-",
                str(frame.get("stem", "")),
                str(frame.get("source_session", "")),
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
            self.detection_lbl.setText("No pair selected.")
            return
        left_path, right_path = self.image_pairs[row]
        left_image = cv2.imread(str(left_path))
        right_image = cv2.imread(str(right_path))
        if left_image is None or right_image is None:
            self.left_preview.set_frame(left_image, placeholder_text="Left image missing")
            self.right_preview.set_frame(right_image, placeholder_text="Right image missing")
            self.detection_lbl.setText("Detection unavailable: one or both images could not be read.")
            self._set_pair_detection_cells(row, matched="-", status="missing image")
            return

        try:
            _kind, board = self._board_spec()
            detection = detect_stereo_board(
                left_image,
                right_image,
                board,
                min_corners=self.min_corners_spin.value(),
            )
            matched_ids = detection.get("matched_ids") or []
            self.left_preview.set_frame(
                annotate_board_detection(left_image, detection["left"], matched_ids=matched_ids),
                placeholder_text="Left image missing",
            )
            self.right_preview.set_frame(
                annotate_board_detection(right_image, detection["right"], matched_ids=matched_ids),
                placeholder_text="Right image missing",
            )
            self.detection_lbl.setText(self._detection_summary_text(detection))
            self._set_pair_detection_cells(
                row,
                matched=str(detection.get("matched_count", 0)),
                status="ok" if detection.get("accepted") else "review",
            )
        except Exception as exc:
            self.left_preview.set_frame(left_image, placeholder_text="Left image missing")
            self.right_preview.set_frame(right_image, placeholder_text="Right image missing")
            self.detection_lbl.setText(f"Detection failed: {exc}")
            self._set_pair_detection_cells(row, matched="-", status="failed")

    def _refresh_current_pair_detection(self) -> None:
        if not hasattr(self, "pairs_table"):
            return
        row = self.pairs_table.currentRow()
        if 0 <= row < len(self.image_pairs):
            self._show_pair(row)

    def _set_pair_detection_cells(self, row: int, *, matched: str, status: str) -> None:
        if row < 0 or row >= self.pairs_table.rowCount():
            return
        for col, value in ((4, matched), (5, status)):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pairs_table.setItem(row, col, item)
        self.pairs_table.resizeColumnsToContents()

    def _detection_summary_text(self, detection: dict) -> str:
        left = detection.get("left") or {}
        right = detection.get("right") or {}
        status = "accepted" if detection.get("accepted") else str(detection.get("reason") or "review")
        if detection.get("kind") == "charuco":
            return (
                "Detection: {status} | matched {matched} | "
                "left {left_corners} corners / {left_markers} markers | "
                "right {right_corners} corners / {right_markers} markers"
            ).format(
                status=status,
                matched=detection.get("matched_count", 0),
                left_corners=left.get("corner_count", 0),
                left_markers=left.get("marker_count", 0),
                right_corners=right.get("corner_count", 0),
                right_markers=right.get("marker_count", 0),
            )
        return (
            "Detection: {status} | matched {matched} | left {left_corners} corners | right {right_corners} corners"
        ).format(
            status=status,
            matched=detection.get("matched_count", 0),
            left_corners=left.get("corner_count", 0),
            right_corners=right.get("corner_count", 0),
        )

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
        if not self.manifest_paths:
            return
        board_kind, board = self._board_spec()
        pair = self.manifest.get("pair") or {}
        self._worker = _CalibrationWorker(
            manifest_paths=list(self.manifest_paths),
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
        warnings = (artifact.get("quality") or {}).get("warnings") or []
        units = board.get("units") or ""
        warning_text = f"\nWarnings: {len(warnings)} item(s) to review" if warnings else ""
        self.result_lbl.setText(
            "Accepted: {accepted} | Stereo RMS: {rms:.4f} px | Baseline: {baseline:.4f} {units}{warnings}\n{path}".format(
                accepted=artifact.get("observation_count", 0),
                rms=float(rms.get("stereo", 0.0)),
                baseline=float(stereo.get("baseline", 0.0)),
                units=units,
                warnings=warning_text,
                path=output_path,
            )
        )
        self._populate_quality_table(artifact)
        self._populate_rejections(artifact.get("rejected_observations") or [])
        self.statusBar().showMessage(f"Calibration saved: {output_path}", 7000)

    def _on_calibration_failed(self, error: str) -> None:
        self.result_lbl.setText(f"Calibration failed: {error}")
        self.quality_table.setRowCount(0)
        self.statusBar().showMessage(f"Calibration failed: {error}", 7000)

    def _format_float(self, value, decimals: int = 3, suffix: str = "") -> str:
        if value is None:
            return "-"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{number:.{decimals}f}{suffix}"

    def _format_percent(self, value) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value) * 100.0:.0f}%"
        except (TypeError, ValueError):
            return "-"

    def _coverage_text(self, coverage: dict) -> str:
        return "{area} bbox, {grid} grid".format(
            area=self._format_percent((coverage or {}).get("area_fraction")),
            grid=self._format_percent((coverage or {}).get("grid_4x4_fraction")),
        )

    def _populate_quality_table(self, artifact: dict) -> None:
        quality = artifact.get("quality") or {}
        rms = artifact.get("rms") or {}
        stereo = artifact.get("stereo") or {}
        board = artifact.get("board") or {}
        units = board.get("units") or ""
        left_reproj = quality.get("left_reprojection") or {}
        right_reproj = quality.get("right_reprojection") or {}
        epipolar = quality.get("epipolar") or {}
        left_coverage = quality.get("left_coverage") or {}
        right_coverage = quality.get("right_coverage") or {}
        warnings = quality.get("warnings") or []
        epipolar_label = (
            "Rectified Epipolar" if epipolar.get("space") == "rectified_pixels" else "Epipolar Error"
        )

        rows = [
            ("Accepted Pairs", str(artifact.get("observation_count", 0))),
            ("Stereo RMS", self._format_float(rms.get("stereo"), suffix=" px")),
            ("Left Reprojection", self._format_float(left_reproj.get("rms_px"), suffix=" px RMS")),
            ("Right Reprojection", self._format_float(right_reproj.get("rms_px"), suffix=" px RMS")),
            (epipolar_label, self._format_float(epipolar.get("rms_px"), suffix=" px RMS")),
            ("Left Coverage", self._coverage_text(left_coverage)),
            ("Right Coverage", self._coverage_text(right_coverage)),
            ("Baseline", f"{self._format_float(stereo.get('baseline'), suffix='')} {units}".strip()),
            ("Rejected Pairs", str(len(artifact.get("rejected_observations") or []))),
        ]
        if warnings:
            rows.extend((f"Warning {index}", warning) for index, warning in enumerate(warnings, start=1))
        else:
            rows.append(("Warnings", "None"))

        self.quality_table.setRowCount(0)
        for row, (metric, value) in enumerate(rows):
            self.quality_table.insertRow(row)
            self.quality_table.setItem(row, 0, QTableWidgetItem(metric))
            self.quality_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.quality_table.resizeRowsToContents()

    def _populate_rejections(self, rejected: list[dict]) -> None:
        self.rejected_table.setRowCount(0)
        for row, item in enumerate(rejected):
            self.rejected_table.insertRow(row)
            values = [str(item.get("index", row + 1)), str(item.get("reason", ""))]
            for col, value in enumerate(values):
                self.rejected_table.setItem(row, col, QTableWidgetItem(value))
        self.rejected_table.resizeColumnsToContents()
