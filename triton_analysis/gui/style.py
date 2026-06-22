"""Shared Qt palette and stylesheet for TritonAnalysis applets."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


def apply_modern_style(app: QApplication) -> None:
    """Apply the standalone TritonAnalysis Qt theme."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(24, 24, 28))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 22))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(28, 28, 34))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.Text, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.Button, QColor(32, 32, 38))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(70, 120, 255))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    try:
        app.setPalette(palette)
    except Exception:
        pass

    qss = """
    QMainWindow { background: #18181c; }
    QWidget { font-size: 12px; }
    QScrollArea#responsiveControlStrip {
        background: transparent;
        border: none;
    }
    QScrollArea#responsiveControlStrip > QWidget > QWidget {
        background: transparent;
    }
    QTabWidget::pane {
        border: 1px solid #2a2a32;
        border-radius: 10px;
    }
    QTabBar::tab {
        padding: 8px 12px;
        margin: 2px;
        border-radius: 10px;
    }
    QTabBar::tab:selected { background: #2a2a36; }
    QStatusBar { border-top: 1px solid #2a2a32; }
    QWidget#missionClock {
        background: #151820;
        border: 1px solid #303849;
        border-radius: 8px;
    }
    QLabel#missionClockLabel {
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid #3a4354;
        color: #f7fbff;
        background: #20232c;
        font-family: Consolas, Menlo, monospace;
        font-weight: 900;
    }
    QLabel#missionClockLabel[tone="green"] {
        color: #e9fff1;
        background: #1d5130;
        border: 1px solid #4fb36e;
    }
    QLabel#missionClockLabel[tone="blue"] {
        color: #f0f6ff;
        background: #214d82;
        border: 1px solid #5d9cec;
    }
    QLabel#missionClockLabel[tone="yellow"] {
        color: #171207;
        background: #e7bd42;
        border: 1px solid #ffd56a;
    }
    QLabel#missionClockLabel[tone="orange"] {
        color: #fff6e6;
        background: #a5531d;
        border: 1px solid #ee9142;
    }
    QLabel#missionClockLabel[tone="red"] {
        color: #fff0f0;
        background: #7b2424;
        border: 1px solid #e36b6b;
    }
    QLabel#missionClockLabel[tone="off"] {
        color: #b7bdca;
        background: #2a2d35;
        border: 1px solid #4a4f5e;
    }
    QLabel#missionClockLabel[state="paused"] {
        color: #fff8e8;
    }
    QPushButton#missionClockButton,
    QToolButton#missionClockButton {
        min-width: 48px;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #414a5c;
        background: #202733;
        color: #eef3ff;
        font-weight: 700;
    }
    QPushButton#missionClockButton:hover,
    QToolButton#missionClockButton:hover {
        border: 1px solid #6d80aa;
        background: #283242;
    }
    QPushButton#missionClockButton:disabled,
    QToolButton#missionClockButton:disabled {
        color: #7e8491;
        background: #1d2028;
        border: 1px solid #303541;
    }
    QSpinBox#missionClockDuration {
        min-width: 66px;
        padding: 2px 4px;
        border: 1px solid #414a5c;
        border-radius: 6px;
        background: #15161d;
        color: #edf3ff;
        font-weight: 700;
    }
    QSpinBox#missionClockDuration:disabled {
        color: #7e8491;
        background: #1d2028;
        border: 1px solid #303541;
    }
    QLabel#summaryCard {
        background: #202028;
        border: 1px solid #2f2f3a;
        border-radius: 10px;
        padding: 2px 4px;
        font-size: 13px;
    }
    QLabel#summaryCard[tone="alert"] {
        background: #3b2525;
        border: 1px solid #9c4a4a;
        color: #ffd9d9;
        font-weight: 700;
    }
    QLabel#summaryCard[tone="warn"] {
        background: #332b1d;
        border: 1px solid #a07e34;
        color: #ffe6ae;
    }
    QLabel#summaryHint {
        color: #b6bac8;
        padding: 2px 4px 6px 4px;
    }
    QFrame#stereoCard {
        background: #202028;
        border: 1px solid #2f2f3a;
        border-radius: 10px;
    }
    QTableWidget {
        border: 1px solid #2a2a32;
        border-radius: 10px;
        gridline-color: #2a2a32;
    }
    QHeaderView::section {
        background: #202028;
        padding: 6px 8px;
        border: none;
        border-bottom: 1px solid #2a2a32;
    }
    QLabel { color: #ebebeb; }
    """
    try:
        app.setStyleSheet(qss)
    except Exception:
        pass
