"""PyQt applet for iceberg tracking threat assessment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QBrush, QFont, QLinearGradient, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
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

from triton_analysis.workspace import fresh_output_subdir, workspace_paths
from triton_analysis.iceberg.tracking import (
    DEFAULT_PLATFORMS,
    THREAT_GREEN,
    THREAT_RED,
    THREAT_YELLOW,
    ThreatAssessment,
    SurveyStatus,
    assess_all_platforms,
    build_judge_report,
    count_levels,
    decimal_degrees_from_dms,
    decimal_degrees_to_dms,
    evaluate_survey_numbers,
    format_dms_coordinate,
    format_level,
    heading_unit_vector,
    local_offset_nm,
    local_point_to_lat_lon,
    normalize_heading_deg,
)
from triton_analysis.gui.responsive import resize_to_available_screen, vertical_scroll_area
from triton_analysis.gui.widgets import BlankZeroDoubleSpinBox, BlankZeroSpinBox


LEVEL_COLORS = {
    THREAT_GREEN: QColor("#4cc878"),
    THREAT_YELLOW: QColor("#ffd166"),
    THREAT_RED: QColor("#ff5d73"),
}
LEVEL_TEXT_COLORS = {
    THREAT_GREEN: QColor("#102719"),
    THREAT_YELLOW: QColor("#2d2206"),
    THREAT_RED: QColor("#320d14"),
}


def _level_color(level: str) -> QColor:
    return LEVEL_COLORS.get(str(level).lower(), LEVEL_COLORS[THREAT_GREEN])


def _level_text_color(level: str) -> QColor:
    return LEVEL_TEXT_COLORS.get(str(level).lower(), LEVEL_TEXT_COLORS[THREAT_GREEN])


class IcebergMapWidget(QWidget):
    """Map-style visualization of iceberg track and platform threat zones."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._iceberg_latitude_deg = 46.5
        self._iceberg_longitude_deg = -48.45
        self._heading_deg = 180.0
        self._future_track_only = True
        self._assessments: list[ThreatAssessment] = []
        self._awaiting_position = True
        self._grid_caption = ""
        self.setMinimumSize(220, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_tracking_state(
        self,
        *,
        iceberg_latitude_deg: float,
        iceberg_longitude_deg: float,
        heading_deg: float,
        assessments: Sequence[ThreatAssessment],
        future_track_only: bool,
    ) -> None:
        self._iceberg_latitude_deg = float(iceberg_latitude_deg)
        self._iceberg_longitude_deg = float(iceberg_longitude_deg)
        self._heading_deg = float(heading_deg)
        self._future_track_only = bool(future_track_only)
        self._assessments = list(assessments)
        self._awaiting_position = False
        self.update()

    def set_awaiting_position(self) -> None:
        """Show a neutral 'enter the iceberg position' prompt over the ocean."""
        self._assessments = []
        self._awaiting_position = True
        self.update()

    def _map_bounds(self) -> tuple[float, float, float, float]:
        heading_east, heading_north = heading_unit_vector(self._heading_deg)
        points: list[tuple[float, float]] = [(0.0, 0.0)]
        max_forward = 70.0
        for assessment in self._assessments:
            east = assessment.geometry.east_nm
            north = assessment.geometry.north_nm
            points.append((east, north))
            for radius in (5.0, 10.0, 25.0):
                points.extend(
                    [
                        (east - radius, north),
                        (east + radius, north),
                        (east, north - radius),
                        (east, north + radius),
                    ]
                )
            if assessment.geometry.along_track_nm > 0.0:
                max_forward = max(max_forward, assessment.geometry.along_track_nm + 28.0)

        points.append((heading_east * max_forward, heading_north * max_forward))
        points.append((-heading_east * 12.0, -heading_north * 12.0))

        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        if max_x - min_x < 40.0:
            center = (min_x + max_x) * 0.5
            min_x = center - 20.0
            max_x = center + 20.0
        if max_y - min_y < 40.0:
            center = (min_y + max_y) * 0.5
            min_y = center - 20.0
            max_y = center + 20.0
        return min_x, max_x, min_y, max_y

    def _projection(self, plot_rect: QRectF):
        min_x, max_x, min_y, max_y = self._map_bounds()
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min(plot_rect.width() / span_x, plot_rect.height() / span_y)
        draw_width = span_x * scale
        draw_height = span_y * scale
        origin_x = plot_rect.x() + (plot_rect.width() - draw_width) * 0.5
        origin_y = plot_rect.y() + (plot_rect.height() - draw_height) * 0.5

        def project(east_nm: float, north_nm: float) -> QPointF:
            return QPointF(
                origin_x + (east_nm - min_x) * scale,
                origin_y + draw_height - (north_nm - min_y) * scale,
            )

        return project, scale, (min_x, max_x, min_y, max_y)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Ocean-blue chart background to match the MATE example imagery.
        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0.0, QColor("#0b3a59"))
        background.setColorAt(1.0, QColor("#11537e"))
        painter.fillRect(self.rect(), background)

        if self._awaiting_position:
            self._draw_awaiting_prompt(painter)
            painter.end()
            return

        plot_rect = QRectF(self.rect()).adjusted(24.0, 34.0, -22.0, -22.0)
        project, scale, bounds = self._projection(plot_rect)

        self._draw_grid(painter, project, bounds)
        self._draw_platform_zones(painter, project, scale)
        self._draw_track(painter, project)
        self._draw_platforms(painter, project)
        self._draw_iceberg(painter, project)
        self._draw_caption(painter)
        painter.end()

    def _draw_caption(self, painter: QPainter) -> None:
        # Bottom-left, clear of the longitude labels along the top and the
        # latitude labels down the right edge.
        painter.setPen(QColor("#bcd7ea"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(16, self.height() - 9, self._grid_caption)

    def _draw_awaiting_prompt(self, painter: QPainter) -> None:
        painter.setPen(QColor("#dceaf6"))
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        flags = int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap)
        painter.drawText(
            QRectF(self.rect()).adjusted(24.0, 24.0, -24.0, -24.0),
            flags,
            "Enter the iceberg latitude, longitude, heading, and keel depth "
            "to plot the platform threat map.",
        )

    def _draw_grid(self, painter: QPainter, project, bounds: tuple[float, float, float, float]) -> None:
        min_x, max_x, min_y, max_y = bounds
        corners = [
            local_point_to_lat_lon(self._iceberg_latitude_deg, self._iceberg_longitude_deg, x, y)
            for x in (min_x, max_x)
            for y in (min_y, max_y)
        ]
        min_lat = min(point[0] for point in corners)
        max_lat = max(point[0] for point in corners)
        min_lon = min(point[1] for point in corners)
        max_lon = max(point[1] for point in corners)
        lat_step = self._coordinate_grid_step(max_lat - min_lat)
        lon_step = self._coordinate_grid_step(max_lon - min_lon)

        painter.setPen(QPen(QColor(255, 255, 255, 48), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        lon = math.ceil(min_lon / lon_step) * lon_step
        while lon <= max_lon + 1.0e-9:
            top_offset = local_offset_nm(
                self._iceberg_latitude_deg,
                self._iceberg_longitude_deg,
                max_lat,
                lon,
            )
            bottom_offset = local_offset_nm(
                self._iceberg_latitude_deg,
                self._iceberg_longitude_deg,
                min_lat,
                lon,
            )
            top = project(top_offset[0], top_offset[1])
            bottom = project(bottom_offset[0], bottom_offset[1])
            painter.drawLine(top, bottom)
            self._draw_longitude_label(painter, top, lon)
            lon += lon_step

        lat = math.ceil(min_lat / lat_step) * lat_step
        while lat <= max_lat + 1.0e-9:
            left_offset = local_offset_nm(
                self._iceberg_latitude_deg,
                self._iceberg_longitude_deg,
                lat,
                min_lon,
            )
            right_offset = local_offset_nm(
                self._iceberg_latitude_deg,
                self._iceberg_longitude_deg,
                lat,
                max_lon,
            )
            left = project(left_offset[0], left_offset[1])
            right = project(right_offset[0], right_offset[1])
            painter.drawLine(left, right)
            self._draw_latitude_label(painter, right, lat)
            lat += lat_step

        self._grid_caption = (
            f"Grid {self._format_grid_step(lat_step)} lat / {self._format_grid_step(lon_step)} lon"
            f"   ·   heading {normalize_heading_deg(self._heading_deg):.1f}° true"
        )

    @staticmethod
    def _coordinate_grid_step(span_degrees: float) -> float:
        span = abs(float(span_degrees))
        if span > 4.0:
            return 1.0
        if span > 1.0:
            return 0.5
        if span > 0.45:
            return 0.25
        if span > 0.16:
            return 1.0 / 12.0
        return 1.0 / 60.0

    @staticmethod
    def _format_grid_step(step_degrees: float) -> str:
        minutes = abs(float(step_degrees)) * 60.0
        if minutes >= 60.0:
            return f"{minutes / 60.0:g} deg"
        return f"{minutes:g}'"

    @staticmethod
    def _draw_longitude_label(painter: QPainter, point: QPointF, longitude_deg: float) -> None:
        text = format_dms_coordinate(longitude_deg, "lon").replace("o", "°")
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        rect = QRectF(
            point.x() - metrics.horizontalAdvance(text) * 0.5 - 5.0,
            8.0,
            metrics.horizontalAdvance(text) + 10.0,
            20.0,
        )
        painter.setPen(QColor("#d6e8f6"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        painter.drawLine(QPointF(point.x(), 30.0), QPointF(point.x(), 38.0))

    @staticmethod
    def _draw_latitude_label(painter: QPainter, point: QPointF, latitude_deg: float) -> None:
        text = format_dms_coordinate(latitude_deg, "lat").replace("o", "°")
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QColor("#d6e8f6"))
        painter.drawText(
            QRectF(point.x() - 96.0, point.y() - 9.0, 90.0, 18.0),
            int(Qt.AlignmentFlag.AlignRight) | int(Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        painter.drawLine(QPointF(point.x() - 4.0, point.y()), QPointF(point.x(), point.y()))

    def _draw_platform_zones(self, painter: QPainter, project, scale: float) -> None:
        rings = [
            (25.0, QColor(108, 168, 255, 95), Qt.PenStyle.DashLine),
            (10.0, QColor(255, 209, 102, 135), Qt.PenStyle.DashLine),
            (5.0, QColor(255, 93, 115, 160), Qt.PenStyle.SolidLine),
        ]
        for assessment in self._assessments:
            center = project(assessment.geometry.east_nm, assessment.geometry.north_nm)
            for radius_nm, color, style in rings:
                radius = radius_nm * scale
                pen = QPen(color, 1)
                pen.setStyle(style)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0))

    def _draw_track(self, painter: QPainter, project) -> None:
        heading_east, heading_north = heading_unit_vector(self._heading_deg)
        max_forward = 70.0
        for assessment in self._assessments:
            max_forward = max(max_forward, assessment.geometry.along_track_nm + 28.0)
        start = project(-heading_east * 10.0, -heading_north * 10.0)
        end = project(heading_east * max_forward, heading_north * max_forward)

        painter.setPen(QPen(QColor("#ffd166"), 3))
        painter.drawLine(start, end)

        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        arrow_size = 15.0
        arrow = QPolygonF(
            [
                end,
                QPointF(
                    end.x() - arrow_size * math.cos(angle - 0.45),
                    end.y() - arrow_size * math.sin(angle - 0.45),
                ),
                QPointF(
                    end.x() - arrow_size * math.cos(angle + 0.45),
                    end.y() - arrow_size * math.sin(angle + 0.45),
                ),
            ]
        )
        painter.setBrush(QColor("#ffd166"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(arrow)

        painter.setPen(QPen(QColor(255, 209, 102, 90), 1))
        if self._future_track_only:
            origin = project(0.0, 0.0)
            painter.drawLine(start, origin)

    def _draw_platforms(self, painter: QPainter, project) -> None:
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        placed_label_rects: list[QRectF] = []
        for assessment in self._assessments:
            point = project(assessment.geometry.east_nm, assessment.geometry.north_nm)
            surface_color = _level_color(assessment.surface_level)
            subsea_color = _level_color(assessment.subsea_level)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(subsea_color, 3))
            painter.drawEllipse(QRectF(point.x() - 11, point.y() - 11, 22, 22))

            painter.setBrush(surface_color)
            painter.setPen(QPen(QColor("#07111a"), 2))
            painter.drawEllipse(QRectF(point.x() - 6, point.y() - 6, 12, 12))

            label = assessment.platform.name
            label_rect = self._place_platform_label(point, metrics, label, placed_label_rects)
            placed_label_rects.append(label_rect)
            painter.setBrush(QColor(7, 24, 39, 220))
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
            painter.drawRoundedRect(label_rect, 6.0, 6.0)
            painter.setPen(QColor("#f3f9ff"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _place_platform_label(self, point: QPointF, metrics, label: str, placed: list[QRectF]) -> QRectF:
        """Choose a label box near the marker that avoids previously placed boxes."""
        width = metrics.horizontalAdvance(label) + 16.0
        height = 21.0
        # Candidate anchors: right, left, above, below the marker.
        candidates = [
            QRectF(point.x() + 12.0, point.y() - height / 2.0, width, height),
            QRectF(point.x() - 12.0 - width, point.y() - height / 2.0, width, height),
            QRectF(point.x() - width / 2.0, point.y() - 16.0 - height, width, height),
            QRectF(point.x() - width / 2.0, point.y() + 16.0, width, height),
        ]
        for rect in candidates:
            if not any(rect.intersects(other) for other in placed):
                return rect
        # All anchors collide: stack downward from the first until clear.
        rect = QRectF(candidates[0])
        while any(rect.intersects(other) for other in placed):
            rect.translate(0.0, height + 3.0)
        return rect

    def _draw_iceberg(self, painter: QPainter, project) -> None:
        center = project(0.0, 0.0)
        heading_east, heading_north = heading_unit_vector(self._heading_deg)
        screen_dx = heading_east
        screen_dy = -heading_north
        norm = max(math.hypot(screen_dx, screen_dy), 1.0e-9)
        screen_dx /= norm
        screen_dy /= norm
        perp_x = -screen_dy
        perp_y = screen_dx
        tip = QPointF(center.x() + screen_dx * 16.0, center.y() + screen_dy * 16.0)
        left = QPointF(center.x() - screen_dx * 10.0 + perp_x * 9.0, center.y() - screen_dy * 10.0 + perp_y * 9.0)
        right = QPointF(center.x() - screen_dx * 10.0 - perp_x * 9.0, center.y() - screen_dy * 10.0 - perp_y * 9.0)
        marker = QPolygonF([tip, left, right])

        painter.setBrush(QColor("#f2f8ff"))
        painter.setPen(QPen(QColor("#0a2740"), 2))
        painter.drawPolygon(marker)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QColor("#eaf3fb"))
        painter.drawText(QRectF(center.x() + 15.0, center.y() + 7.0, 120.0, 24.0), Qt.AlignmentFlag.AlignLeft, "Iceberg")

class IcebergTrackingWindow(QMainWindow):
    """Operator window for entering iceberg data and reviewing threat results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Iceberg Tracking Threat Assessment")
        resize_to_available_screen(self, 1280, 820, min_width=880, min_height=600)
        self._platforms = list(DEFAULT_PLATFORMS)
        self._assessments: list[ThreatAssessment] = []
        self._survey_status = evaluate_survey_numbers([None, None, None, None, None])
        self._syncing_coordinates = False
        self._last_coordinate_mode = "dms"

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        input_scroll = vertical_scroll_area(self._build_input_panel())
        # Keep the floor low enough that the whole window still fits a small
        # screen; the default split below gives the sheet enough room for the
        # spaced-out DMS fields on a normal competition laptop.
        input_scroll.setMinimumWidth(330)
        input_scroll.setMaximumWidth(560)
        splitter.addWidget(input_scroll)          # entry sheet
        splitter.addWidget(self._build_map_panel())   # tall ocean threat map
        splitter.addWidget(self._build_info_panel())  # results table + report
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([450, 415, 415])

        container = QWidget()
        container.setObjectName("icebergTrackingRoot")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter)
        self.setCentralWidget(container)
        self._apply_local_style()
        self._connect_inputs()
        self._recalculate()
        resize_to_available_screen(self, 1280, 820, min_width=880, min_height=600)

    def _build_input_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("trackingPanel")
        panel.setMinimumWidth(420)
        panel.setMaximumWidth(560)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Iceberg Tracking")
        title.setObjectName("trackingTitle")
        subtitle = QLabel("Survey numbers, iceberg sheet values, and judge-ready threat calls.")
        subtitle.setObjectName("trackingSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        sheet_frame = QFrame()
        sheet_frame.setObjectName("trackingSection")
        sheet_layout = QGridLayout(sheet_frame)
        sheet_layout.setContentsMargins(12, 12, 12, 12)
        sheet_layout.setHorizontalSpacing(10)
        sheet_layout.setVerticalSpacing(8)

        self.coordinate_format_combo = QComboBox()
        self.coordinate_format_combo.addItem("Degrees minutes seconds", "dms")
        self.coordinate_format_combo.addItem("Decimal degrees", "decimal")

        self.latitude_spin = self._make_spinbox(-90.0, 90.0, 0.0, 5, 0.0001)
        self.longitude_spin = self._make_spinbox(-180.0, 180.0, 0.0, 5, 0.0001)
        self.latitude_dms_widget = self._make_dms_coordinate_widget("lat")
        self.longitude_dms_widget = self._make_dms_coordinate_widget("lon")
        self.heading_spin = self._make_spinbox(0.0, 359.9, 0.0, 1, 1.0)
        self.heading_spin.setWrapping(True)
        self.keel_depth_spin = self._make_spinbox(0.0, 500.0, 0.0, 1, 1.0)
        self.future_track_check = QCheckBox("Forward heading only")
        self.future_track_check.setChecked(True)
        self.future_track_check.setToolTip("Use the future iceberg track from its current location instead of the infinite heading line.")

        self._add_labeled_widget(sheet_layout, 0, "Coordinate format", self.coordinate_format_combo)
        self.latitude_dms_label = self._add_labeled_widget(sheet_layout, 1, "Latitude", self.latitude_dms_widget)
        self.longitude_dms_label = self._add_labeled_widget(sheet_layout, 2, "Longitude", self.longitude_dms_widget)
        self.latitude_decimal_label = self._add_labeled_widget(sheet_layout, 3, "Latitude", self.latitude_spin)
        self.longitude_decimal_label = self._add_labeled_widget(sheet_layout, 4, "Longitude", self.longitude_spin)
        self._add_labeled_widget(sheet_layout, 5, "Heading (°true)", self.heading_spin)
        self._add_labeled_widget(sheet_layout, 6, "Keel depth (m)", self.keel_depth_spin)
        sheet_layout.addWidget(self.future_track_check, 7, 1)
        self._refresh_coordinate_format_visibility()
        layout.addWidget(sheet_frame)

        survey_frame = QFrame()
        survey_frame.setObjectName("trackingSection")
        survey_layout = QGridLayout(survey_frame)
        survey_layout.setContentsMargins(12, 12, 12, 12)
        survey_layout.setHorizontalSpacing(8)
        survey_layout.setVerticalSpacing(8)
        survey_title = QLabel("Survey Numbers")
        survey_title.setObjectName("trackingSectionTitle")
        survey_layout.addWidget(survey_title, 0, 0, 1, 2)
        self.survey_combos: list[QComboBox] = []
        for row, label in enumerate(["Corner A", "Corner B", "Corner C", "Corner D", "Keel pipe"], start=1):
            combo = QComboBox()
            combo.addItem("-", None)
            for number in range(10):
                combo.addItem(str(number), number)
            combo.setMinimumWidth(74)
            self.survey_combos.append(combo)
            survey_layout.addWidget(QLabel(label), row, 0)
            survey_layout.addWidget(combo, row, 1)
        layout.addWidget(survey_frame)

        self.survey_status_label = self._make_status_card("Survey: 0/5")
        self.surface_count_label = self._make_status_card("Surface: -")
        self.subsea_count_label = self._make_status_card("Subsea: -")
        self.nearest_label = self._make_status_card("Nearest: -")
        layout.addWidget(self.survey_status_label)
        layout.addWidget(self.surface_count_label)
        layout.addWidget(self.subsea_count_label)
        layout.addWidget(self.nearest_label)

        button_row = QHBoxLayout()
        self.copy_report_btn = QPushButton("Copy Report")
        self.save_report_btn = QPushButton("Save Report")
        button_row.addWidget(self.copy_report_btn)
        button_row.addWidget(self.save_report_btn)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return panel

    def _build_map_panel(self) -> QWidget:
        self.map_widget = IcebergMapWidget()
        self.map_widget.setMinimumWidth(220)
        return self.map_widget

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.result_table = QTableWidget(0, 8)
        self.result_table.setObjectName("trackingTable")
        self.result_table.setHorizontalHeaderLabels(
            [
                "Platform",
                "CPA (NM)",
                "Track",
                "Water",
                "Keel/depth",
                "Surface",
                "Subsea",
                "Basis",
            ]
        )
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.verticalHeader().setVisible(False)
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.result_table, 1)

        self.report_preview = QPlainTextEdit()
        self.report_preview.setObjectName("trackingReport")
        self.report_preview.setReadOnly(True)
        self.report_preview.setMinimumHeight(150)
        layout.addWidget(self.report_preview, 1)
        return panel

    @staticmethod
    def _make_spinbox(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
        step: float,
        *,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spin = BlankZeroDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(126)
        return spin

    def _make_dms_coordinate_widget(self, coordinate: str, value: float | None = None) -> QWidget:
        coordinate_type = str(coordinate).strip().lower()
        max_degrees = 90 if coordinate_type == "lat" else 180
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        # Pin the row to its content width so the grid can never squeeze the
        # fields together (which is how seconds got mistaken for minutes).
        row.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        def _make_dms_spin(spin, width: int, tooltip: str):
            # DMS values are typed, not stepped: drop the arrows to free space
            # and remove a source of accidental changes.
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setFixedWidth(width)
            spin.setToolTip(tooltip)
            return spin

        degree_spin = _make_dms_spin(BlankZeroSpinBox(), 44, "Degrees")
        degree_spin.setRange(0, max_degrees)
        minute_spin = _make_dms_spin(BlankZeroSpinBox(), 40, "Minutes")
        minute_spin.setRange(0, 59)
        second_spin = _make_dms_spin(BlankZeroDoubleSpinBox(), 58, "Seconds")
        second_spin.setRange(0.0, 59.99)
        second_spin.setDecimals(2)
        second_spin.setSingleStep(1.0)
        hemisphere_combo = QComboBox()
        if coordinate_type == "lat":
            hemisphere_combo.addItems(["N", "S"])
        else:
            hemisphere_combo.addItems(["W", "E"])
        hemisphere_combo.setFixedWidth(52)
        hemisphere_combo.setToolTip("Hemisphere")

        def _unit(symbol: str) -> QLabel:
            unit = QLabel(symbol)
            unit.setObjectName("dmsUnit")
            unit.setFixedWidth(14)
            return unit

        # Group each value with its unit and leave clear gaps between the
        # degree / minute / second groups so they are never confused.
        row.addWidget(degree_spin)
        row.addWidget(_unit("°"))
        row.addSpacing(13)
        row.addWidget(minute_spin)
        row.addWidget(_unit("′"))
        row.addSpacing(13)
        row.addWidget(second_spin)
        row.addWidget(_unit("″"))
        row.addSpacing(13)
        row.addWidget(hemisphere_combo)
        row.addStretch(1)

        if coordinate_type == "lat":
            self.latitude_degree_spin = degree_spin
            self.latitude_minute_spin = minute_spin
            self.latitude_second_spin = second_spin
            self.latitude_hemisphere_combo = hemisphere_combo
        else:
            self.longitude_degree_spin = degree_spin
            self.longitude_minute_spin = minute_spin
            self.longitude_second_spin = second_spin
            self.longitude_hemisphere_combo = hemisphere_combo
        # Leave the degree/minute/second fields blank until the operator enters
        # a value, so a fresh sheet never looks pre-filled.
        if value is not None:
            self._set_dms_widgets_from_decimal(coordinate_type, value)
        return container

    @staticmethod
    def _add_labeled_widget(layout: QGridLayout, row: int, label: str, widget: QWidget) -> QLabel:
        text = QLabel(label)
        text.setObjectName("trackingFormLabel")
        layout.addWidget(text, row, 0)
        layout.addWidget(widget, row, 1)
        return text

    @staticmethod
    def _make_status_card(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("trackingStatusCard")
        label.setWordWrap(True)
        label.setMinimumHeight(44)
        return label

    def _connect_inputs(self) -> None:
        self.coordinate_format_combo.currentIndexChanged.connect(self._coordinate_format_changed)
        for spin in (self.latitude_spin, self.longitude_spin, self.heading_spin, self.keel_depth_spin):
            spin.valueChanged.connect(self._recalculate)
        for widget in (
            self.latitude_degree_spin,
            self.latitude_minute_spin,
            self.latitude_second_spin,
            self.longitude_degree_spin,
            self.longitude_minute_spin,
            self.longitude_second_spin,
        ):
            widget.valueChanged.connect(self._coordinate_inputs_changed)
        self.latitude_hemisphere_combo.currentIndexChanged.connect(self._coordinate_inputs_changed)
        self.longitude_hemisphere_combo.currentIndexChanged.connect(self._coordinate_inputs_changed)
        self.future_track_check.stateChanged.connect(self._recalculate)
        for combo in self.survey_combos:
            combo.currentIndexChanged.connect(self._recalculate)
        self.copy_report_btn.clicked.connect(self._copy_report)
        self.save_report_btn.clicked.connect(self._save_report)

    def _coordinate_mode(self) -> str:
        return str(self.coordinate_format_combo.currentData() or "dms")

    def _coordinate_format_changed(self, *_args) -> None:
        if self._syncing_coordinates:
            return
        previous_mode = self._last_coordinate_mode
        new_mode = self._coordinate_mode()
        if previous_mode == new_mode:
            self._refresh_coordinate_format_visibility()
            return

        self._syncing_coordinates = True
        try:
            if previous_mode == "dms" and new_mode == "decimal":
                self.latitude_spin.setValue(self._latitude_from_dms())
                self.longitude_spin.setValue(self._longitude_from_dms())
            elif previous_mode == "decimal" and new_mode == "dms":
                self._set_dms_widgets_from_decimal("lat", self.latitude_spin.value())
                self._set_dms_widgets_from_decimal("lon", self.longitude_spin.value())
            self._last_coordinate_mode = new_mode
            self._refresh_coordinate_format_visibility()
        finally:
            self._syncing_coordinates = False
        self._recalculate()

    def _coordinate_inputs_changed(self, *_args) -> None:
        if self._syncing_coordinates:
            return
        if self._coordinate_mode() == "dms":
            self._recalculate()

    def _refresh_coordinate_format_visibility(self) -> None:
        use_dms = self._coordinate_mode() == "dms"
        self.latitude_dms_label.setVisible(use_dms)
        self.latitude_dms_widget.setVisible(use_dms)
        self.longitude_dms_label.setVisible(use_dms)
        self.longitude_dms_widget.setVisible(use_dms)
        self.latitude_decimal_label.setVisible(not use_dms)
        self.latitude_spin.setVisible(not use_dms)
        self.longitude_decimal_label.setVisible(not use_dms)
        self.longitude_spin.setVisible(not use_dms)

    def _set_dms_widgets_from_decimal(self, coordinate: str, value: float) -> None:
        degrees, minutes, seconds = decimal_degrees_to_dms(value, seconds_decimals=2)
        if coordinate == "lat":
            hemisphere = "N" if float(value) >= 0.0 else "S"
            widgets = (
                self.latitude_degree_spin,
                self.latitude_minute_spin,
                self.latitude_second_spin,
                self.latitude_hemisphere_combo,
            )
        else:
            hemisphere = "E" if float(value) >= 0.0 else "W"
            widgets = (
                self.longitude_degree_spin,
                self.longitude_minute_spin,
                self.longitude_second_spin,
                self.longitude_hemisphere_combo,
            )

        degree_spin, minute_spin, second_spin, hemisphere_combo = widgets
        degree_spin.setValue(degrees)
        minute_spin.setValue(minutes)
        second_spin.setValue(seconds)
        index = hemisphere_combo.findText(hemisphere)
        if index >= 0:
            hemisphere_combo.setCurrentIndex(index)

    def _latitude_from_dms(self) -> float:
        return decimal_degrees_from_dms(
            self.latitude_degree_spin.value(),
            self.latitude_minute_spin.value(),
            self.latitude_second_spin.value(),
            self.latitude_hemisphere_combo.currentText(),
        )

    def _longitude_from_dms(self) -> float:
        return decimal_degrees_from_dms(
            self.longitude_degree_spin.value(),
            self.longitude_minute_spin.value(),
            self.longitude_second_spin.value(),
            self.longitude_hemisphere_combo.currentText(),
        )

    def _current_latitude_deg(self) -> float:
        if self._coordinate_mode() == "dms":
            return self._latitude_from_dms()
        return float(self.latitude_spin.value())

    def _current_longitude_deg(self) -> float:
        if self._coordinate_mode() == "dms":
            return self._longitude_from_dms()
        return float(self.longitude_spin.value())

    def _survey_numbers(self) -> list[int | None]:
        return [combo.currentData() for combo in self.survey_combos]

    def _position_entered(self) -> bool:
        """True once a real iceberg position is set (both coordinates non-zero)."""
        return abs(self._current_latitude_deg()) > 1.0e-6 and abs(self._current_longitude_deg()) > 1.0e-6

    def _recalculate(self, *_args) -> None:
        self._survey_status = evaluate_survey_numbers(self._survey_numbers())
        if not self._position_entered():
            self._assessments = []
            self.map_widget.set_awaiting_position()
            self._show_awaiting_state()
            return

        latitude = self._current_latitude_deg()
        longitude = self._current_longitude_deg()
        self._assessments = assess_all_platforms(
            iceberg_latitude_deg=latitude,
            iceberg_longitude_deg=longitude,
            heading_deg=self.heading_spin.value(),
            keel_depth_m=self.keel_depth_spin.value(),
            platforms=self._platforms,
            future_track_only=self.future_track_check.isChecked(),
        )
        self.map_widget.set_tracking_state(
            iceberg_latitude_deg=latitude,
            iceberg_longitude_deg=longitude,
            heading_deg=self.heading_spin.value(),
            assessments=self._assessments,
            future_track_only=self.future_track_check.isChecked(),
        )
        self._update_status_cards()
        self._update_result_table()
        self._update_report_preview()

    def _show_awaiting_state(self) -> None:
        """Neutral state shown before an iceberg position is entered."""
        self.survey_status_label.setText(self._survey_status.message)
        self._set_card_tone(self.survey_status_label, "ok" if self._survey_status.complete else "warn")
        self.surface_count_label.setText("Surface: awaiting position")
        self.subsea_count_label.setText("Subsea: awaiting position")
        self.nearest_label.setText("Nearest: enter the iceberg position")
        for card in (self.surface_count_label, self.subsea_count_label, self.nearest_label):
            self._set_card_tone(card, "info")
        self.result_table.setRowCount(0)
        self.report_preview.setPlainText(
            "Enter the iceberg latitude, longitude, heading, and keel depth to generate the threat report."
        )

    def _update_status_cards(self) -> None:
        self.survey_status_label.setText(self._survey_status.message)
        self._set_card_tone(self.survey_status_label, "ok" if self._survey_status.complete else "warn")

        surface_counts = count_levels(self._assessments, "surface_level")
        subsea_counts = count_levels(self._assessments, "subsea_level")
        self.surface_count_label.setText(
            "Surface: "
            f"{surface_counts[THREAT_RED]} red / "
            f"{surface_counts[THREAT_YELLOW]} yellow / "
            f"{surface_counts[THREAT_GREEN]} green"
        )
        self.subsea_count_label.setText(
            "Subsea: "
            f"{subsea_counts[THREAT_RED]} red / "
            f"{subsea_counts[THREAT_YELLOW]} yellow / "
            f"{subsea_counts[THREAT_GREEN]} green"
        )
        self._set_card_tone(self.surface_count_label, self._counts_tone(surface_counts))
        self._set_card_tone(self.subsea_count_label, self._counts_tone(subsea_counts))

        nearest = min(self._assessments, key=lambda item: item.geometry.closest_approach_nm)
        self.nearest_label.setText(
            f"Nearest: {nearest.platform.name} at {nearest.geometry.closest_approach_nm:.2f} NM"
        )
        self._set_card_tone(self.nearest_label, "info")

    @staticmethod
    def _counts_tone(counts: dict[str, int]) -> str:
        if counts.get(THREAT_RED, 0) > 0:
            return "alert"
        if counts.get(THREAT_YELLOW, 0) > 0:
            return "warn"
        return "ok"

    @staticmethod
    def _set_card_tone(label: QLabel, tone: str) -> None:
        label.setProperty("tone", tone)
        label.style().unpolish(label)
        label.style().polish(label)
        label.update()

    def _update_result_table(self) -> None:
        self.result_table.setRowCount(len(self._assessments))
        for row, assessment in enumerate(self._assessments):
            geometry = assessment.geometry
            relative = "Ahead" if geometry.platform_ahead else "Behind"
            basis = f"{assessment.surface_reason} {assessment.subsea_reason}"
            values = [
                assessment.platform.name,
                f"{geometry.closest_approach_nm:.2f}",
                f"{relative}, {geometry.along_track_nm:.1f} NM",
                f"{assessment.platform.water_depth_m:.0f} m",
                f"{assessment.keel_to_depth_ratio * 100.0:.0f}%",
                format_level(assessment.surface_level),
                format_level(assessment.subsea_level),
                basis,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(basis)
                if column in (1, 2, 3, 4, 5, 6):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 5:
                    self._style_level_item(item, assessment.surface_level)
                elif column == 6:
                    self._style_level_item(item, assessment.subsea_level)
                self.result_table.setItem(row, column, item)
            self.result_table.setRowHeight(row, 42)

    @staticmethod
    def _style_level_item(item: QTableWidgetItem, level: str) -> None:
        item.setBackground(QBrush(_level_color(level)))
        item.setForeground(QBrush(_level_text_color(level)))
        font = item.font()
        font.setBold(True)
        item.setFont(font)

    def _update_report_preview(self) -> None:
        self.report_preview.setPlainText(self._report_text())

    def _report_text(self) -> str:
        return build_judge_report(
            iceberg_latitude_deg=self._current_latitude_deg(),
            iceberg_longitude_deg=self._current_longitude_deg(),
            heading_deg=self.heading_spin.value(),
            keel_depth_m=self.keel_depth_spin.value(),
            survey_status=self._survey_status,
            assessments=self._assessments,
            future_track_only=self.future_track_check.isChecked(),
        )

    def _copy_report(self) -> None:
        QApplication.clipboard().setText(self._report_text())
        self.statusBar().showMessage("Iceberg tracking report copied.", 4000)

    def _save_report(self) -> None:
        results_dir = fresh_output_subdir(workspace_paths(create=True).reports / "iceberg_tracking", "iceberg_tracking_report")
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            results_dir = Path.cwd()
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save iceberg tracking report",
            str(results_dir / "iceberg_tracking_report.txt"),
            "Text files (*.txt);;All files (*)",
        )
        if not selected_path:
            return
        path = Path(selected_path)
        try:
            path.write_text(self._report_text(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Iceberg Tracking", f"Could not save report:\n{exc}")
            return
        self.statusBar().showMessage(f"Saved iceberg tracking report: {path}", 5000)

    def _apply_local_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#icebergTrackingRoot {
                background: #18181c;
            }
            QFrame#trackingPanel {
                background: #1b1d24;
                border: 1px solid #303340;
                border-radius: 8px;
            }
            QLabel#trackingTitle {
                font-size: 22px;
                font-weight: 800;
                color: #f5f7fb;
            }
            QLabel#trackingSubtitle {
                color: #b8c0d0;
            }
            QFrame#trackingSection {
                background: #151821;
                border: 1px solid #2a2f3d;
                border-radius: 8px;
            }
            QLabel#trackingSectionTitle {
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#trackingFormLabel {
                color: #cdd3df;
            }
            QLabel#dmsUnit {
                color: #aeb8c8;
                font-size: 14px;
                font-weight: 800;
            }
            QLabel#trackingStatusCard {
                background: #151821;
                border: 1px solid #2a2f3d;
                border-radius: 8px;
                padding: 8px 10px;
                font-weight: 700;
            }
            QLabel#trackingStatusCard[tone="ok"] {
                color: #d9ffea;
                border-color: #2f7a4f;
                background: #1d3527;
            }
            QLabel#trackingStatusCard[tone="warn"] {
                color: #ffe6ae;
                border-color: #a07e34;
                background: #332b1d;
            }
            QLabel#trackingStatusCard[tone="alert"] {
                color: #ffd9d9;
                border-color: #995252;
                background: #402222;
            }
            QLabel#trackingStatusCard[tone="info"] {
                color: #dbe6ff;
                border-color: #4468aa;
                background: #1f2c42;
            }
            QDoubleSpinBox,
            QComboBox {
                padding: 4px 7px;
                border: 1px solid #343a4b;
                border-radius: 6px;
                background: #10131a;
            }
            QPushButton {
                padding: 8px 10px;
                border: 1px solid #3b465f;
                border-radius: 7px;
                background: #26324a;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #314061;
            }
            QTableWidget#trackingTable {
                background: #11141b;
                alternate-background-color: #161a23;
                border: 1px solid #2a2f3d;
                border-radius: 8px;
                gridline-color: #252b38;
            }
            QPlainTextEdit#trackingReport {
                background: #10131a;
                border: 1px solid #2a2f3d;
                border-radius: 8px;
                padding: 8px;
                color: #e8edf5;
                font-family: Consolas, monospace;
            }
            """
        )
