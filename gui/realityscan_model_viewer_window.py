"""Three.js viewer launcher for RealityScan OBJ models."""

from __future__ import annotations

import functools
import html
import os
import socket
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.responsive import resize_to_available_screen, vertical_scroll_area

try:  # Optional dependency. The external-browser path works without it.
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - depends on optional local install
    QWebEngineView = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "realityscan"


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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    """Serve the viewer page and model directory from localhost only."""

    server_version = "TritonAnalysisModelViewer/1.0"

    def log_message(self, _format: str, *_args) -> None:
        return

    @property
    def model_dir(self) -> Path:
        return self.server.model_dir  # type: ignore[attr-defined]

    @property
    def model_name(self) -> str:
        return self.server.model_name  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("", "/", "/viewer.html"):
            self._send_viewer_html()
            return
        if parsed.path.startswith("/files/"):
            self._send_model_file(parsed.path[len("/files/") :], include_body=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/files/"):
            self._send_model_file(parsed.path[len("/files/") :], include_body=False)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _send_viewer_html(self) -> None:
        body = _viewer_html(model_url=f"/files/{quote(self.model_name)}", model_name=self.model_name).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_model_file(self, rel_url: str, *, include_body: bool) -> None:
        rel_path = Path(unquote(rel_url))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        path = (self.model_dir / rel_path).resolve()
        if not _is_relative_to(path, self.model_dir) or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ctype = self.guess_type(str(path))
        try:
            data = path.read_bytes() if include_body else b""
            size = path.stat().st_size
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if include_body:
            self.wfile.write(data)


class ModelViewerServer:
    """Small localhost server for one OBJ model directory."""

    def __init__(self, obj_path: Path):
        self.obj_path = Path(obj_path).expanduser().resolve()
        self.model_dir = self.obj_path.parent
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        self.stop()
        port = _find_free_port()
        handler = functools.partial(_ViewerRequestHandler, directory=str(self.model_dir))
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        server.model_dir = self.model_dir  # type: ignore[attr-defined]
        server.model_name = self.obj_path.name  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        return f"http://127.0.0.1:{port}/viewer.html"

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None


class RealityScanModelViewerPanel(QWidget):
    """Launch and optionally embed a Three.js OBJ measurement viewport."""

    def __init__(self, model_path: str | None = None, parent=None):
        super().__init__(parent)
        self._server: ModelViewerServer | None = None
        self._viewer_url = ""

        self._build_ui()
        if model_path:
            self.model_edit.setText(str(Path(model_path)))
        self._set_status("Select a metric OBJ model.", "idle")

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        controls_panel = QWidget()
        controls_panel.setMinimumWidth(360)
        controls_panel.setMaximumWidth(470)
        controls = QVBoxLayout(controls_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        splitter.addWidget(vertical_scroll_area(controls_panel))

        title = QLabel("Metric Model Viewer")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        controls.addWidget(title)

        input_card = _SectionCard("Model")
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("underwater_model_metric.obj")
        self.model_edit.setClearButtonEnabled(True)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_edit, 1)
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self._choose_model)
        model_row.addWidget(self.browse_model_btn)
        input_card.body.addWidget(QLabel("OBJ"))
        input_card.body.addLayout(model_row)

        self.unit_combo = QComboBox()
        self.unit_combo.addItem("Meters", "m")
        self.unit_combo.addItem("Centimeters", "cm")
        self.unit_combo.addItem("Millimeters", "mm")
        self.unit_combo.setCurrentIndex(0)
        input_grid = QGridLayout()
        input_grid.addWidget(QLabel("Display Units"), 0, 0)
        input_grid.addWidget(self.unit_combo, 0, 1)
        input_card.body.addLayout(input_grid)
        controls.addWidget(input_card)

        launch_card = _SectionCard("Viewport")
        self.status_label = QLabel("-")
        self.status_label.setObjectName("summaryCard")
        self.status_label.setWordWrap(True)
        launch_card.body.addWidget(self.status_label)
        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Load Viewport")
        self.start_btn.clicked.connect(self._start_viewer)
        self.browser_btn = QPushButton("Open Browser")
        self.browser_btn.clicked.connect(self._open_browser)
        self.browser_btn.setEnabled(False)
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.clicked.connect(self._reload_view)
        self.reload_btn.setEnabled(False)
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.browser_btn)
        button_row.addWidget(self.reload_btn)
        launch_card.body.addLayout(button_row)
        self.url_edit = QLineEdit()
        self.url_edit.setReadOnly(True)
        launch_card.body.addWidget(QLabel("Viewer URL"))
        launch_card.body.addWidget(self.url_edit)
        controls.addWidget(launch_card)

        controls.addStretch(1)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        use_embedded_webview = QWebEngineView is not None and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"
        if use_embedded_webview:
            self.web_view = QWebEngineView()
            viewer_layout.addWidget(self.web_view, 1)
        else:
            self.web_view = None
            placeholder = QLabel(
                "Embedded Qt WebEngine is not available in this environment.\n"
                "Use Load Viewport, then Open Browser for the Three.js viewer."
            )
            placeholder.setObjectName("summaryHint")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setWordWrap(True)
            viewer_layout.addWidget(placeholder, 1)

    def _choose_model(self) -> None:
        start = self._dialog_start()
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select RealityScan OBJ model",
            str(start),
            "OBJ models (*.obj);;All files (*)",
        )
        if path:
            self.model_edit.setText(str(Path(path)))

    def _dialog_start(self) -> Path:
        text = self.model_edit.text().strip().strip('"')
        if text:
            path = Path(text).expanduser()
            if path.exists():
                return path.parent
        return DEFAULT_RESULTS_DIR if DEFAULT_RESULTS_DIR.exists() else REPO_ROOT

    def _model_path(self) -> Path:
        return Path(self.model_edit.text().strip().strip('"')).expanduser()

    def _start_viewer(self) -> None:
        model = self._model_path()
        if not model.exists() or model.suffix.lower() != ".obj":
            QMessageBox.warning(self, "Model Viewer", f"Select an OBJ model first:\n{model}")
            return

        self.shutdown()
        self._server = ModelViewerServer(model)
        self._viewer_url = self._server.start()
        self.url_edit.setText(self._viewer_url)
        self.browser_btn.setEnabled(True)
        self.reload_btn.setEnabled(True)
        self._set_status(f"Serving {model.name} from localhost.", "ok")
        self._load_embedded_view()

    def _load_embedded_view(self) -> None:
        if self.web_view is None or not self._viewer_url:
            return
        url = QUrl(self._viewer_url)
        if self.unit_combo.currentData():
            url.setQuery(f"unit={self.unit_combo.currentData()}")
        self.web_view.setUrl(url)

    def _reload_view(self) -> None:
        if not self._viewer_url:
            self._start_viewer()
            return
        self._load_embedded_view()

    def _open_browser(self) -> None:
        if not self._viewer_url:
            self._start_viewer()
        if self._viewer_url:
            url = QUrl(self._viewer_url)
            if self.unit_combo.currentData():
                url.setQuery(f"unit={self.unit_combo.currentData()}")
            QDesktopServices.openUrl(url)

    def _set_status(self, text: str, tone: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("tone", tone)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.status_label.update()


class RealityScanModelViewerWindow(QMainWindow):
    """Top-level window wrapper for the reusable model-viewer panel."""

    def __init__(self, model_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RealityScan Model Viewer")
        self.panel = RealityScanModelViewerPanel(model_path=model_path)
        self.setCentralWidget(self.panel)
        resize_to_available_screen(self, 1500, 900, min_width=1020, min_height=680)

    def __getattr__(self, name: str):
        panel = self.__dict__.get("panel")
        if panel is not None and hasattr(panel, name):
            return getattr(panel, name)
        raise AttributeError(name)

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        self.panel.shutdown()


def _viewer_html(*, model_url: str, model_name: str) -> str:
    safe_name = html.escape(model_name)
    safe_url = html.escape(model_url)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_name}</title>
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #0d1117;
      color: #eef2ff;
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    #viewport {{
      position: fixed;
      inset: 0;
    }}
    #toolbar {{
      position: fixed;
      top: 12px;
      left: 12px;
      right: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      pointer-events: none;
      z-index: 5;
    }}
    #toolbar > * {{
      pointer-events: auto;
    }}
    #labels {{
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 4;
      overflow: hidden;
    }}
    #loupe {{
      position: fixed;
      left: 0;
      top: 0;
      width: 180px;
      height: 180px;
      display: none;
      pointer-events: none;
      z-index: 7;
      border: 2px solid rgba(34, 211, 238, 0.92);
      border-radius: 50%;
      background: rgba(15, 23, 42, 0.82);
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
      image-rendering: pixelated;
    }}
    .measurement-label {{
      position: absolute;
      left: 0;
      top: 0;
      transform: translate3d(var(--label-x, -9999px), var(--label-y, -9999px), 0) translate(-50%, -115%);
      will-change: transform;
      border: 1px solid rgba(250, 204, 21, 0.72);
      border-radius: 5px;
      background: rgba(15, 23, 42, 0.82);
      color: #f8fafc;
      padding: 2px 6px;
      font-size: 11px;
      line-height: 1.1;
      white-space: nowrap;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.75);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }}
    body.labels-hidden .measurement-label {{
      display: none !important;
    }}
    .measurement-label.active {{
      background: rgba(30, 41, 59, 0.94);
      color: #ffffff;
      font-weight: 700;
    }}
    button, select, input {{
      border: 1px solid rgba(148, 163, 184, 0.5);
      border-radius: 7px;
      background: rgba(15, 23, 42, 0.88);
      color: #eef2ff;
      padding: 7px 10px;
      font-size: 13px;
    }}
    button.active, button[aria-pressed="true"] {{
      background: rgba(37, 99, 235, 0.92);
      border-color: rgba(147, 197, 253, 0.95);
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.52;
    }}
    input {{
      width: 92px;
    }}
    input::placeholder {{
      color: #94a3b8;
      opacity: 1;
    }}
    #rollStepSelect {{
      width: 78px;
    }}
    #truthUnitSelect {{
      width: 64px;
    }}
    #readout {{
      min-width: 210px;
      border: 1px solid rgba(148, 163, 184, 0.45);
      border-radius: 7px;
      background: rgba(15, 23, 42, 0.86);
      padding: 7px 10px;
      font-size: 13px;
      line-height: 1.3;
    }}
    #status {{
      position: fixed;
      left: 12px;
      bottom: 12px;
      max-width: min(760px, calc(100vw - 24px));
      border: 1px solid rgba(148, 163, 184, 0.38);
      border-radius: 7px;
      background: rgba(15, 23, 42, 0.82);
      color: #cbd5e1;
      padding: 7px 10px;
      font-size: 12px;
      z-index: 5;
    }}
    canvas {{
      display: block;
    }}
  </style>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
      }}
    }}
  </script>
</head>
<body>
  <div id="viewport"></div>
  <div id="labels"></div>
  <canvas id="loupe" width="180" height="180" aria-hidden="true"></canvas>
  <div id="toolbar">
    <button id="measureBtn" title="Pick two mesh points">Measure</button>
    <button id="centerBtn" title="Set orbit center from mesh">Set Center</button>
    <button id="deleteBtn" title="Delete selected point or measurement">Delete</button>
    <button id="clearBtn" title="Clear measurements">Clear</button>
    <button id="pickAssistBtn" title="Show exact mesh hit marker and cursor magnifier while picking" aria-pressed="true">Pick Assist</button>
    <button id="edgeBtn" title="Toggle cyan geometry edges for surface boundaries" aria-pressed="false">Edges</button>
    <button id="fitBtn" title="Fit model to view">Fit</button>
    <button id="resetBtn" title="Restore fitted camera">Reset</button>
    <button id="levelViewBtn" title="Remove camera roll relative to the floor">Level</button>
    <button id="floorViewBtn" title="Top-down floor view aligned to the model footprint">Floor View</button>
    <button id="rollLeftBtn" title="Roll camera counterclockwise">Roll -</button>
    <button id="rollRightBtn" title="Roll camera clockwise">Roll +</button>
    <select id="rollStepSelect" title="Camera roll step">
      <option value="1">1 deg</option>
      <option value="5" selected>5 deg</option>
      <option value="15">15 deg</option>
      <option value="45">45 deg</option>
    </select>
    <button id="floorBtn" title="Pick three mesh points to make that plane level" aria-pressed="false">Set Floor</button>
    <button id="resetFloorBtn" title="Undo floor alignment" disabled>Reset Floor</button>
    <button id="gridBtn" title="Toggle 3D reference grid" aria-pressed="false">3D Grid</button>
    <button id="labelsBtn" title="Toggle measurement labels" aria-pressed="true">Labels</button>
    <select id="viewSelect" title="Camera view">
      <option value="">View</option>
      <option value="iso">Isometric</option>
      <option value="floor">Floor View</option>
      <option value="top">Top</option>
      <option value="front">Front</option>
      <option value="right">Right</option>
    </select>
    <select id="measurementSelect" title="Measurements">
      <option value="">Measurements</option>
    </select>
    <select id="unitSelect" title="Distance units">
      <option value="m">m</option>
      <option value="cm">cm</option>
      <option value="mm">mm</option>
    </select>
    <input id="truthInput" title="Ground-truth length for the selected measurement" type="number" min="0" step="any" placeholder="Truth">
    <select id="truthUnitSelect" title="Ground-truth units">
      <option value="m">m</option>
      <option value="cm">cm</option>
      <option value="mm">mm</option>
    </select>
    <button id="scaleBtn" title="Rescale model so selected measurement equals ground truth">Apply Scale</button>
    <div id="readout">Distance: -</div>
  </div>
  <div id="status">Loading {safe_name}</div>
  <script type="module">
    import * as THREE from 'three';
    import {{ TrackballControls }} from 'three/addons/controls/TrackballControls.js';
    import {{ MTLLoader }} from 'three/addons/loaders/MTLLoader.js';
    import {{ OBJLoader }} from 'three/addons/loaders/OBJLoader.js';

    const modelUrl = '{safe_url}';
    const viewport = document.getElementById('viewport');
    const labelsLayer = document.getElementById('labels');
    const loupe = document.getElementById('loupe');
    const loupeContext = loupe.getContext('2d');
    const statusEl = document.getElementById('status');
    const readout = document.getElementById('readout');
    const measureBtn = document.getElementById('measureBtn');
    const centerBtn = document.getElementById('centerBtn');
    const deleteBtn = document.getElementById('deleteBtn');
    const clearBtn = document.getElementById('clearBtn');
    const pickAssistBtn = document.getElementById('pickAssistBtn');
    const edgeBtn = document.getElementById('edgeBtn');
    const fitBtn = document.getElementById('fitBtn');
    const resetBtn = document.getElementById('resetBtn');
    const levelViewBtn = document.getElementById('levelViewBtn');
    const floorViewBtn = document.getElementById('floorViewBtn');
    const rollLeftBtn = document.getElementById('rollLeftBtn');
    const rollRightBtn = document.getElementById('rollRightBtn');
    const rollStepSelect = document.getElementById('rollStepSelect');
    const floorBtn = document.getElementById('floorBtn');
    const resetFloorBtn = document.getElementById('resetFloorBtn');
    const gridBtn = document.getElementById('gridBtn');
    const labelsBtn = document.getElementById('labelsBtn');
    const viewSelect = document.getElementById('viewSelect');
    const measurementSelect = document.getElementById('measurementSelect');
    const unitSelect = document.getElementById('unitSelect');
    const truthInput = document.getElementById('truthInput');
    const truthUnitSelect = document.getElementById('truthUnitSelect');
    const scaleBtn = document.getElementById('scaleBtn');
    const urlUnit = new URLSearchParams(window.location.search).get('unit');
    if (urlUnit && [...unitSelect.options].some(option => option.value === urlUnit)) {{
      unitSelect.value = urlUnit;
    }}

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);
    const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 100000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    viewport.appendChild(renderer.domElement);

    const controls = new TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 0.85;
    controls.panSpeed = 0.34;
    controls.zoomSpeed = 0.52;
    controls.dynamicDampingFactor = 0.12;
    controls.staticMoving = false;
    controls.noRotate = false;
    controls.noPan = false;
    controls.noZoom = false;
    controls.mouseButtons = {{
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    }};
    controls.addEventListener('change', requestLabelUpdate);
    controls.addEventListener('start', () => {{
      if (!applyingViewPreset) viewSelect.value = '';
    }});

    const hemi = new THREE.HemisphereLight(0xffffff, 0x334155, 1.8);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(3, -4, 5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xbfd7ff, 0.9);
    fill.position.set(-5, 3, 2);
    scene.add(fill);

    const grid = new THREE.Group();
    grid.visible = false;
    scene.add(grid);

    const edgeHelpers = [];

    function makeCursorLine(start, end, color) {{
      const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
      return new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({{ color, depthTest: false, transparent: true, opacity: 0.95 }})
      );
    }}

    const orbitCursor = new THREE.Group();
    orbitCursor.renderOrder = 10;
    orbitCursor.add(makeCursorLine(new THREE.Vector3(-0.5, 0, 0), new THREE.Vector3(0.5, 0, 0), 0xff5f57));
    orbitCursor.add(makeCursorLine(new THREE.Vector3(0, -0.5, 0), new THREE.Vector3(0, 0.5, 0), 0x5eead4));
    orbitCursor.add(makeCursorLine(new THREE.Vector3(0, 0, -0.5), new THREE.Vector3(0, 0, 0.5), 0x60a5fa));
    orbitCursor.add(new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 12, 6),
      new THREE.MeshBasicMaterial({{ color: 0xffffff, depthTest: false }})
    ));
    scene.add(orbitCursor);

    const hoverMarker = new THREE.Group();
    hoverMarker.visible = false;
    hoverMarker.renderOrder = 30;
    const hoverDot = new THREE.Mesh(
      new THREE.SphereGeometry(1.0, 18, 10),
      new THREE.MeshBasicMaterial({{ color: 0xfacc15, depthTest: false, transparent: true, opacity: 0.96 }})
    );
    const hoverRing = new THREE.Mesh(
      new THREE.RingGeometry(2.0, 2.35, 36),
      new THREE.MeshBasicMaterial({{ color: 0x22d3ee, depthTest: false, transparent: true, opacity: 0.9, side: THREE.DoubleSide }})
    );
    const hoverNormalGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 4.5, 0)
    ]);
    const hoverNormalLine = new THREE.Line(
      hoverNormalGeometry,
      new THREE.LineBasicMaterial({{ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.86 }})
    );
    hoverMarker.add(hoverDot, hoverRing, hoverNormalLine);
    scene.add(hoverMarker);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let rootObject = null;
    let meshObjects = [];
    let modelRadius = 1.0;
    let cumulativeModelScale = 1.0;
    const measurements = [];
    const markerPickObjects = [];
    let nextMeasurementId = 1;
    let activeMeasurement = null;
    let selectedMarker = null;
    let measureMode = false;
    let centerMode = false;
    let floorMode = false;
    let labelsVisible = true;
    let labelsDirty = true;
    let measurementSelectSignature = '';
    let applyingViewPreset = false;
    let floorAlignment = new THREE.Quaternion();
    const floorPickPoints = [];
    const floorPickMarkers = [];
    let downPoint = null;
    let draggingMarker = null;
    let dragMoved = false;
    let pickAssistEnabled = true;
    let edgesVisible = false;
    let pointerInsideViewport = false;
    let loupeDrawFailed = false;
    const lastPointer = {{ x: 0, y: 0 }};
    const dragPlane = new THREE.Plane();
    const dragPlaneHit = new THREE.Vector3();
    const labelMidpoint = new THREE.Vector3();
    const labelScreen = new THREE.Vector3();
    const worldUp = new THREE.Vector3(0, 1, 0);
    const defaultFloorScreenUp = new THREE.Vector3(0, 0, -1);
    const savedViewState = {{
      position: new THREE.Vector3(),
      up: new THREE.Vector3(0, 1, 0),
      target: new THREE.Vector3()
    }};
    const viewPresets = {{
      iso: {{
        direction: new THREE.Vector3(0.72, 0.62, 0.58).normalize(),
        up: new THREE.Vector3(0, 1, 0)
      }},
      top: {{
        direction: new THREE.Vector3(0, 1, 0).normalize(),
        up: new THREE.Vector3(0, 0, -1)
      }},
      front: {{
        direction: new THREE.Vector3(0, 0, 1).normalize(),
        up: new THREE.Vector3(0, 1, 0)
      }},
      right: {{
        direction: new THREE.Vector3(1, 0, 0).normalize(),
        up: new THREE.Vector3(0, 1, 0)
      }}
    }};

    function setStatus(text) {{
      statusEl.textContent = text;
    }}

    function requestLabelUpdate() {{
      labelsDirty = true;
    }}

    function roundToDevicePixel(value) {{
      const dpr = Math.max(window.devicePixelRatio || 1, 1);
      return Math.round(value * dpr) / dpr;
    }}

    function setButtonPressed(button, pressed) {{
      button.classList.toggle('active', pressed);
      button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
    }}

    function setControlTargets(target) {{
      controls.target.copy(target);
      orbitCursor.position.copy(target);
    }}

    function saveViewState() {{
      savedViewState.position.copy(camera.position);
      savedViewState.up.copy(camera.up);
      savedViewState.target.copy(controls.target);
    }}

    function restoreSavedViewState() {{
      camera.position.copy(savedViewState.position);
      camera.up.copy(savedViewState.up);
      camera.lookAt(savedViewState.target);
      camera.updateProjectionMatrix();
      setControlTargets(savedViewState.target);
      controls.update();
      requestLabelUpdate();
    }}

    function floorProjected(vector) {{
      return vector.clone().addScaledVector(worldUp, -vector.dot(worldUp));
    }}

    function orientFloorAxis(axis) {{
      const oriented = axis.clone().normalize();
      let currentUp = floorProjected(camera.up);
      if (currentUp.lengthSq() < 1e-8) currentUp = defaultFloorScreenUp.clone();
      if (oriented.dot(currentUp.normalize()) < 0) oriented.negate();
      return oriented;
    }}

    function computeFootprintMajorAxis() {{
      if (!rootObject) return null;
      rootObject.updateMatrixWorld(true);

      let vertexCount = 0;
      rootObject.traverse(child => {{
        const position = child.isMesh && child.geometry ? child.geometry.attributes.position : null;
        if (position) vertexCount += position.count;
      }});
      if (vertexCount < 3) return null;

      const stride = Math.max(1, Math.ceil(vertexCount / 12000));
      const samples = [];
      const point = new THREE.Vector3();
      let seen = 0;
      rootObject.traverse(child => {{
        const position = child.isMesh && child.geometry ? child.geometry.attributes.position : null;
        if (!position) return;
        for (let index = 0; index < position.count; index += 1) {{
          if (seen % stride === 0) {{
            point.fromBufferAttribute(position, index).applyMatrix4(child.matrixWorld);
            samples.push(point.x, point.z);
          }}
          seen += 1;
        }}
      }});

      const count = samples.length / 2;
      if (count < 3) return null;
      let meanX = 0;
      let meanZ = 0;
      for (let index = 0; index < samples.length; index += 2) {{
        meanX += samples[index];
        meanZ += samples[index + 1];
      }}
      meanX /= count;
      meanZ /= count;

      let xx = 0;
      let xz = 0;
      let zz = 0;
      for (let index = 0; index < samples.length; index += 2) {{
        const dx = samples[index] - meanX;
        const dz = samples[index + 1] - meanZ;
        xx += dx * dx;
        xz += dx * dz;
        zz += dz * dz;
      }}
      if (xx + zz < 1e-12) return null;

      const angle = 0.5 * Math.atan2(2 * xz, xx - zz);
      return orientFloorAxis(new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle)));
    }}

    function setCameraPose(direction, up, distance = null) {{
      const viewDirection = direction.clone().normalize();
      const viewDistance = distance || Math.max(camera.position.distanceTo(controls.target), modelRadius * 2.5, 0.01);
      let safeUp = up.clone().addScaledVector(viewDirection, -up.dot(viewDirection));
      if (safeUp.lengthSq() < 1e-8) {{
        safeUp = (Math.abs(viewDirection.dot(worldUp)) > 0.96 ? defaultFloorScreenUp : worldUp).clone();
        safeUp.addScaledVector(viewDirection, -safeUp.dot(viewDirection));
      }}

      applyingViewPreset = true;
      camera.up.copy(safeUp.normalize());
      camera.position.copy(controls.target).addScaledVector(viewDirection, viewDistance);
      camera.lookAt(controls.target);
      camera.updateProjectionMatrix();
      controls.update();
      setControlTargets(controls.target);
      applyingViewPreset = false;
      requestLabelUpdate();
    }}

    function setFloorView(distance = null, announce = true) {{
      const screenUp = computeFootprintMajorAxis() || defaultFloorScreenUp.clone();
      setCameraPose(worldUp, screenUp, distance);
      viewSelect.value = 'floor';
      if (announce) setStatus('Floor view squared to the model footprint.');
    }}

    function levelCameraToFloor(announce = true) {{
      const offset = camera.position.clone().sub(controls.target);
      const distance = offset.length();
      let direction = distance > 1e-6 ? offset.clone().normalize() : viewPresets.iso.direction.clone();
      if (direction.dot(worldUp) < -0.02) {{
        direction = direction.addScaledVector(worldUp, -2 * direction.dot(worldUp)).normalize();
      }}
      const up = Math.abs(direction.dot(worldUp)) > 0.96
        ? computeFootprintMajorAxis() || defaultFloorScreenUp.clone()
        : worldUp.clone();
      setCameraPose(direction, up, Math.max(distance, modelRadius * 2.5, 0.01));
      viewSelect.value = '';
      if (announce) setStatus('View leveled to the floor plane.');
    }}

    function rollStepDegrees() {{
      const value = Number(rollStepSelect.value);
      return Number.isFinite(value) && value > 0 ? value : 5;
    }}

    function rollCamera(degrees) {{
      const axis = camera.position.clone().sub(controls.target);
      if (axis.lengthSq() < 1e-12) return;
      const quaternion = new THREE.Quaternion().setFromAxisAngle(axis.normalize(), THREE.MathUtils.degToRad(degrees));
      camera.up.applyQuaternion(quaternion).normalize();
      camera.lookAt(controls.target);
      camera.updateProjectionMatrix();
      controls.update();
      viewSelect.value = '';
      requestLabelUpdate();
      setStatus(`Camera rolled ${{degrees > 0 ? '+' : ''}}${{degrees}} deg.`);
    }}

    function updateFloorResetState() {{
      resetFloorBtn.disabled = Math.abs(floorAlignment.x) < 1e-9
        && Math.abs(floorAlignment.y) < 1e-9
        && Math.abs(floorAlignment.z) < 1e-9
        && Math.abs(floorAlignment.w - 1) < 1e-9;
    }}

    function resize() {{
      const width = Math.max(1, window.innerWidth);
      const height = Math.max(1, window.innerHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      if (typeof controls.handleResize === 'function') controls.handleResize();
      requestLabelUpdate();
    }}
    window.addEventListener('resize', resize);
    resize();

    function modelBaseUrl() {{
      const idx = modelUrl.lastIndexOf('/');
      return idx >= 0 ? modelUrl.slice(0, idx + 1) : './';
    }}

    function mtlUrlForObj() {{
      return modelUrl.replace(/\\.obj($|\\?)/i, '.mtl$1');
    }}

    async function mtlExists(url) {{
      try {{
        const response = await fetch(url, {{ method: 'HEAD', cache: 'no-store' }});
        return response.ok;
      }} catch (_err) {{
        return false;
      }}
    }}

    function normalizeMaterials(object) {{
      object.traverse(child => {{
        if (!child.isMesh) return;
        child.geometry.computeBoundingSphere();
        if (Array.isArray(child.material)) {{
          child.material.forEach(material => {{
            material.side = THREE.DoubleSide;
            material.needsUpdate = true;
          }});
        }} else if (child.material) {{
          child.material.side = THREE.DoubleSide;
          child.material.needsUpdate = true;
        }}
      }});
    }}

    function collectMeshes(object) {{
      meshObjects = [];
      object.traverse(child => {{
        if (child.isMesh) meshObjects.push(child);
      }});
    }}

    function clearEdgeOverlay() {{
      while (edgeHelpers.length) {{
        const helper = edgeHelpers.pop();
        if (helper.parent) helper.parent.remove(helper);
        if (helper.geometry) helper.geometry.dispose();
        if (helper.material) helper.material.dispose();
      }}
    }}

    function rebuildEdgeOverlay() {{
      clearEdgeOverlay();
      if (!edgesVisible || !meshObjects.length) return;
      let totalVertices = 0;
      meshObjects.forEach(mesh => {{
        const position = mesh.geometry && mesh.geometry.attributes ? mesh.geometry.attributes.position : null;
        if (position) totalVertices += position.count;
      }});
      if (totalVertices > 800000) {{
        edgesVisible = false;
        setButtonPressed(edgeBtn, false);
        setStatus('Edge overlay skipped: this mesh is too dense for interactive edges.');
        return;
      }}

      meshObjects.forEach(mesh => {{
        if (!mesh.geometry || !mesh.geometry.attributes || !mesh.geometry.attributes.position) return;
        const geometry = new THREE.EdgesGeometry(mesh.geometry, 35);
        const helper = new THREE.LineSegments(
          geometry,
          new THREE.LineBasicMaterial({{ color: 0x22d3ee, transparent: true, opacity: 0.42, depthTest: true, depthWrite: false }})
        );
        helper.renderOrder = 8;
        mesh.add(helper);
        edgeHelpers.push(helper);
      }});
    }}

    function setEdgeOverlayVisible(visible) {{
      edgesVisible = Boolean(visible);
      setButtonPressed(edgeBtn, edgesVisible);
      if (edgesVisible) {{
        rebuildEdgeOverlay();
        if (edgesVisible) setStatus('Geometry edges enabled.');
      }} else {{
        clearEdgeOverlay();
        setStatus('Geometry edges hidden.');
      }}
    }}

    function configureGridMaterials(helper, opacity) {{
      const materials = Array.isArray(helper.material) ? helper.material : [helper.material];
      materials.forEach(material => {{
        material.transparent = true;
        material.opacity = opacity;
        material.depthWrite = false;
      }});
    }}

    function makeGridHelper(size, divisions, centerColor, gridColor, opacity) {{
      const helper = new THREE.GridHelper(size, divisions, centerColor, gridColor);
      configureGridMaterials(helper, opacity);
      return helper;
    }}

    function rebuildGridForBox(box) {{
      const wasVisible = grid.visible;
      while (grid.children.length) {{
        const child = grid.children.pop();
        disposeObject(child);
      }}

      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const gridSize = Math.max(size.x, size.y, size.z, 1);
      const divisions = 12;
      const floorY = box.min.y;
      const centerY = floorY + gridSize * 0.5;

      const floorGrid = makeGridHelper(gridSize, divisions, 0x64748b, 0x1f2937, 0.72);
      floorGrid.position.set(center.x, floorY, center.z);
      grid.add(floorGrid);

      const xyGrid = makeGridHelper(gridSize, divisions, 0x475569, 0x1e293b, 0.34);
      xyGrid.rotation.x = Math.PI / 2;
      xyGrid.position.set(center.x, centerY, center.z);
      grid.add(xyGrid);

      const yzGrid = makeGridHelper(gridSize, divisions, 0x475569, 0x1e293b, 0.30);
      yzGrid.rotation.z = Math.PI / 2;
      yzGrid.position.set(center.x, centerY, center.z);
      grid.add(yzGrid);

      const frameBox = new THREE.Box3(
        new THREE.Vector3(center.x - gridSize * 0.5, floorY, center.z - gridSize * 0.5),
        new THREE.Vector3(center.x + gridSize * 0.5, floorY + gridSize, center.z + gridSize * 0.5)
      );
      const frame = new THREE.Box3Helper(frameBox, 0x475569);
      configureGridMaterials(frame, 0.42);
      grid.add(frame);
      grid.visible = wasVisible;
    }}

    function setOrbitCenter(point, keepView = true) {{
      const target = point.clone();
      if (keepView) {{
        const delta = target.clone().sub(controls.target);
        camera.position.add(delta);
      }}
      setControlTargets(target);
      controls.update();
      requestLabelUpdate();
    }}

    function updateCursorScale() {{
      orbitCursor.scale.setScalar(Math.max(modelRadius * 0.055, 0.004));
    }}

    function centerObjectAtOrigin(object) {{
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      object.position.sub(center);
      object.updateMatrixWorld(true);
    }}

    function fitCamera(object) {{
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.length() * 0.5, 0.001);
      modelRadius = radius;
      camera.near = Math.max(radius / 1000, 0.0001);
      camera.far = Math.max(radius * 1000, 10);
      const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov * 0.5)) * 1.15;
      camera.updateProjectionMatrix();
      setControlTargets(center);
      controls.minDistance = Math.max(radius * 0.015, 0.0005);
      controls.maxDistance = Math.max(radius * 80, 1);
      updateCursorScale();
      setCameraView('iso', distance);
      saveViewState();

      rebuildGridForBox(new THREE.Box3().setFromObject(object));
      requestLabelUpdate();
    }}

    function setCameraView(name, distance = null) {{
      if (name === 'floor') {{
        setFloorView(distance);
        return;
      }}
      const preset = viewPresets[name] || viewPresets.iso;
      setCameraPose(preset.direction, preset.up, distance);
      viewSelect.value = name;
    }}

    function loadObj(materials = null) {{
      const loader = new OBJLoader();
      if (materials) loader.setMaterials(materials);
      loader.load(
        modelUrl,
        object => {{
          clearEdgeOverlay();
          if (rootObject) scene.remove(rootObject);
          rootObject = object;
          clearFloorPickMarkers();
          floorAlignment.identity();
          cumulativeModelScale = 1.0;
          updateFloorResetState();
          normalizeMaterials(rootObject);
          centerObjectAtOrigin(rootObject);
          scene.add(rootObject);
          fitCamera(rootObject);
          collectMeshes(rootObject);
          if (edgesVisible) rebuildEdgeOverlay();
          setStatus(`Loaded {safe_name} | ${{meshObjects.length}} mesh part(s)`);
        }},
        event => {{
          if (event.lengthComputable) {{
            setStatus(`Loading {safe_name} | ${{Math.round(event.loaded / event.total * 100)}}%`);
          }}
        }},
        error => {{
          console.error(error);
          setStatus('OBJ load failed. Check the browser console and model files.');
        }}
      );
    }}

    async function loadModel() {{
      const mtlUrl = mtlUrlForObj();
      if (await mtlExists(mtlUrl)) {{
        const mtlLoader = new MTLLoader();
        mtlLoader.setPath(modelBaseUrl());
        mtlLoader.load(
          mtlUrl.split('/').pop(),
          materials => {{
            materials.preload();
            loadObj(materials);
          }},
          undefined,
          () => loadObj()
        );
      }} else {{
        loadObj();
      }}
    }}

    function unitMultiplier(unit) {{
      if (unit === 'cm') return 0.01;
      if (unit === 'mm') return 0.001;
      return 1.0;
    }}

    function formatDistance(meters, unit = unitSelect.value) {{
      const value = meters / unitMultiplier(unit);
      const digits = Math.abs(value) >= 10 ? 3 : 4;
      return `${{value.toPrecision(digits)}} ${{unit}}`;
    }}

    function markerRadius() {{
      return Math.max(modelRadius * 0.0014, 0.0008);
    }}

    function markerHitRadius() {{
      return Math.max(modelRadius * 0.006, 0.003);
    }}

    function measurementColor(measurement) {{
      const colors = [0xfacc15, 0x6ee7b7, 0x93c5fd, 0xf0abfc, 0xfdba74, 0xc4b5fd];
      return colors[(measurement.id - 1) % colors.length];
    }}

    function colorCss(color) {{
      return `#${{color.toString(16).padStart(6, '0')}}`;
    }}

    function createMeasurement() {{
      const measurement = {{
        id: nextMeasurementId++,
        points: [],
        markers: [],
        line: null,
        label: null,
        labelState: {{ x: null, y: null, visible: null, text: '', active: false }},
      }};
      measurements.push(measurement);
      activeMeasurement = measurement;
      updateMeasurementSelect();
      updateReadout();
      return measurement;
    }}

    function removeMarkerPickObjects(marker) {{
      marker.traverse(child => {{
        const index = markerPickObjects.indexOf(child);
        if (index >= 0) markerPickObjects.splice(index, 1);
      }});
    }}

    function disposeObject(object) {{
      object.traverse(child => {{
        if (child.geometry) child.geometry.dispose();
        if (child.material) {{
          if (Array.isArray(child.material)) child.material.forEach(material => material.dispose());
          else child.material.dispose();
        }}
      }});
    }}

    function createMarker(measurement, pointIndex, point) {{
      const marker = new THREE.Group();
      marker.position.copy(point);
      marker.userData.measurement = measurement;
      marker.userData.pointIndex = pointIndex;

      const visible = new THREE.Mesh(
        new THREE.SphereGeometry(markerRadius(), 16, 8),
        new THREE.MeshStandardMaterial({{ color: measurementColor(measurement), emissive: 0x111827, roughness: 0.45 }})
      );
      visible.userData.marker = marker;
      marker.userData.visible = visible;
      marker.add(visible);

      const hit = new THREE.Mesh(
        new THREE.SphereGeometry(markerHitRadius(), 12, 6),
        new THREE.MeshBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.0, depthWrite: false }})
      );
      hit.userData.marker = marker;
      marker.add(hit);
      markerPickObjects.push(visible, hit);
      scene.add(marker);
      return marker;
    }}

    function setSelectedMarker(marker) {{
      selectedMarker = marker || null;
      measurements.forEach(measurement => {{
        measurement.markers.forEach(item => {{
          const visible = item.userData.visible;
          if (!visible || !visible.material) return;
          visible.material.color.setHex(item === selectedMarker ? 0xffffff : measurementColor(measurement));
          visible.material.emissive.setHex(item === selectedMarker ? 0x334155 : 0x111827);
        }});
      }});
      if (selectedMarker) {{
        activeMeasurement = selectedMarker.userData.measurement;
        updateMeasurementSelect();
      }}
      updateReadout();
    }}

    function ensureMeasurementLabel(measurement) {{
      if (measurement.label) return measurement.label;
      const label = document.createElement('div');
      label.className = 'measurement-label';
      label.dataset.measurementId = String(measurement.id);
      label.style.borderColor = colorCss(measurementColor(measurement));
      labelsLayer.appendChild(label);
      measurement.label = label;
      measurement.labelState = {{ x: null, y: null, visible: null, text: '', active: false }};
      return label;
    }}

    function removeMeasurementLabel(measurement) {{
      if (!measurement.label) return;
      measurement.label.remove();
      measurement.label = null;
      measurement.labelState = {{ x: null, y: null, visible: null, text: '', active: false }};
    }}

    function updateMeasurementLabel(measurement) {{
      if (!measurement || measurement.points.length !== 2) {{
        if (measurement) removeMeasurementLabel(measurement);
        return;
      }}
      const label = ensureMeasurementLabel(measurement);
      const nextText = measurementLabel(measurement);
      const active = measurement === activeMeasurement;
      if (measurement.labelState.text !== nextText) {{
        label.textContent = nextText;
        measurement.labelState.text = nextText;
      }}
      if (measurement.labelState.active !== active) {{
        label.classList.toggle('active', active);
        measurement.labelState.active = active;
      }}

      labelMidpoint.copy(measurement.points[0]).add(measurement.points[1]).multiplyScalar(0.5);
      labelScreen.copy(labelMidpoint).project(camera);
      const rendererRect = renderer.domElement.getBoundingClientRect();
      const labelsRect = labelsLayer.getBoundingClientRect();
      const visible = labelsVisible
        && labelScreen.z >= -1
        && labelScreen.z <= 1
        && labelScreen.x >= -1.05
        && labelScreen.x <= 1.05
        && labelScreen.y >= -1.05
        && labelScreen.y <= 1.05;
      if (!visible) {{
        if (measurement.labelState.visible !== false) {{
          label.style.display = 'none';
          measurement.labelState.visible = false;
        }}
        return;
      }}
      const x = roundToDevicePixel(rendererRect.left - labelsRect.left + (labelScreen.x * 0.5 + 0.5) * rendererRect.width);
      const y = roundToDevicePixel(rendererRect.top - labelsRect.top + (-labelScreen.y * 0.5 + 0.5) * rendererRect.height - 12);
      if (measurement.labelState.x !== x) {{
        label.style.setProperty('--label-x', `${{x}}px`);
        measurement.labelState.x = x;
      }}
      if (measurement.labelState.y !== y) {{
        label.style.setProperty('--label-y', `${{y}}px`);
        measurement.labelState.y = y;
      }}
      if (measurement.labelState.visible !== true) {{
        label.style.display = 'block';
        measurement.labelState.visible = true;
      }}
    }}

    function updateAllMeasurementLabels(force = false) {{
      if (!force && !labelsDirty) return;
      labelsDirty = false;
      measurements.forEach(updateMeasurementLabel);
    }}

    function updateLine(measurement) {{
      if (measurement.points.length === 2) {{
        if (!measurement.line) {{
          measurement.line = new THREE.Line(
            new THREE.BufferGeometry(),
            new THREE.LineBasicMaterial({{ color: measurementColor(measurement), linewidth: 2 }})
          );
          scene.add(measurement.line);
        }}
        measurement.line.geometry.dispose();
        measurement.line.geometry = new THREE.BufferGeometry().setFromPoints(measurement.points);
        measurement.line.geometry.computeBoundingSphere();
      }} else if (measurement.line) {{
        scene.remove(measurement.line);
        measurement.line.geometry.dispose();
        measurement.line.material.dispose();
        measurement.line = null;
      }}
      requestLabelUpdate();
      updateMeasurementLabel(measurement);
    }}

    function measurementDistance(measurement) {{
      if (!measurement || measurement.points.length !== 2) return null;
      return measurement.points[0].distanceTo(measurement.points[1]);
    }}

    function measurementLabel(measurement) {{
      const distance = measurementDistance(measurement);
      const suffix = distance === null ? `${{measurement.points.length}}/2` : formatDistance(distance);
      return `M${{measurement.id}}  ${{suffix}}`;
    }}

    function updateMeasurementSelect() {{
      const selectedId = activeMeasurement ? String(activeMeasurement.id) : '';
      const signature = `${{selectedId}}|${{measurements.map(measurement => `${{measurement.id}}:${{measurementLabel(measurement)}}`).join('|')}}`;
      if (signature === measurementSelectSignature) {{
        measurementSelect.value = selectedId;
        return;
      }}
      measurementSelectSignature = signature;
      measurementSelect.innerHTML = '<option value="">Measurements</option>';
      measurements.forEach(measurement => {{
        const option = document.createElement('option');
        option.value = String(measurement.id);
        option.textContent = measurementLabel(measurement);
        measurementSelect.appendChild(option);
      }});
      measurementSelect.value = selectedId;
    }}

    function updateReadout() {{
      updateMeasurementSelect();
      if (!measurements.length) {{
        readout.textContent = 'Distance: -';
        return;
      }}
      const measurement = activeMeasurement || measurements[measurements.length - 1];
      const distance = measurementDistance(measurement);
      if (distance === null) {{
        readout.textContent = `M${{measurement.id}}: pick ${{2 - measurement.points.length}} point(s)`;
      }} else {{
        readout.textContent = `M${{measurement.id}}: ${{formatDistance(distance)}}`;
      }}
      requestLabelUpdate();
      updateAllMeasurementLabels(true);
    }}

    function scaleVectorAbout(vector, origin, factor) {{
      return vector.sub(origin).multiplyScalar(factor).add(origin);
    }}

    function refreshModelBounds() {{
      if (!rootObject) return;
      const box = new THREE.Box3().setFromObject(rootObject);
      const size = box.getSize(new THREE.Vector3());
      modelRadius = Math.max(size.length() * 0.5, 0.001);
      camera.near = Math.max(modelRadius / 1000, 0.0001);
      camera.far = Math.max(modelRadius * 1000, 10);
      camera.updateProjectionMatrix();
      controls.minDistance = Math.max(modelRadius * 0.015, 0.0005);
      controls.maxDistance = Math.max(modelRadius * 80, 1);
      updateCursorScale();
      rebuildGridForBox(box);
    }}

    function selectedCompleteMeasurement() {{
      const selectedId = Number(measurementSelect.value || 0);
      const selected = measurements.find(item => item.id === selectedId);
      if (selected && selected.points.length === 2) return selected;
      if (activeMeasurement && activeMeasurement.points.length === 2) return activeMeasurement;
      return [...measurements].reverse().find(item => item.points.length === 2) || null;
    }}

    function groundTruthDistanceMeters() {{
      const value = Number(truthInput.value);
      if (!Number.isFinite(value) || value <= 0) return null;
      return value * unitMultiplier(truthUnitSelect.value);
    }}

    function applyUniformModelScale(scaleFactor) {{
      if (!rootObject || !Number.isFinite(scaleFactor) || scaleFactor <= 0) return false;
      const origin = new THREE.Vector3(0, 0, 0);

      scaleVectorAbout(rootObject.position, origin, scaleFactor);
      rootObject.scale.multiplyScalar(scaleFactor);
      rootObject.updateMatrixWorld(true);

      measurements.forEach(measurement => {{
        measurement.points.forEach(point => scaleVectorAbout(point, origin, scaleFactor));
        measurement.markers.forEach(marker => {{
          scaleVectorAbout(marker.position, origin, scaleFactor);
          marker.scale.multiplyScalar(scaleFactor);
        }});
        updateLine(measurement);
      }});

      floorPickPoints.forEach(point => scaleVectorAbout(point, origin, scaleFactor));
      floorPickMarkers.forEach(marker => {{
        scaleVectorAbout(marker.position, origin, scaleFactor);
        marker.scale.multiplyScalar(scaleFactor);
      }});

      scaleVectorAbout(camera.position, origin, scaleFactor);
      scaleVectorAbout(savedViewState.position, origin, scaleFactor);
      scaleVectorAbout(savedViewState.target, origin, scaleFactor);
      setControlTargets(scaleVectorAbout(controls.target.clone(), origin, scaleFactor));
      controls.update();

      cumulativeModelScale *= scaleFactor;
      refreshModelBounds();
      requestLabelUpdate();
      updateReadout();
      return true;
    }}

    function rescaleFromGroundTruth() {{
      if (!rootObject) {{
        setStatus('Load a model before applying scale.');
        return;
      }}
      const measurement = selectedCompleteMeasurement();
      if (!measurement) {{
        setStatus('Select or create a completed measurement before applying scale.');
        return;
      }}
      const measuredDistance = measurementDistance(measurement);
      const truthDistance = groundTruthDistanceMeters();
      if (!truthDistance) {{
        setStatus('Enter a positive ground-truth length.');
        return;
      }}
      if (!measuredDistance || measuredDistance <= 0) {{
        setStatus(`M${{measurement.id}} is too short to use for scaling.`);
        return;
      }}

      const scaleFactor = truthDistance / measuredDistance;
      if (!applyUniformModelScale(scaleFactor)) {{
        setStatus('Scale could not be applied.');
        return;
      }}
      activeMeasurement = measurement;
      updateMeasurementSelect();
      const afterDistance = measurementDistance(measurement);
      setStatus(
        `Scaled model by ${{scaleFactor.toPrecision(6)}}x using M${{measurement.id}}: ` +
        `${{formatDistance(measuredDistance)}} -> ${{formatDistance(afterDistance)}}. ` +
        `Cumulative scale ${{cumulativeModelScale.toPrecision(6)}}x.`
      );
    }}

    function addMeasurementPoint(point) {{
      let measurement = activeMeasurement;
      if (!measurement || measurement.points.length >= 2) measurement = createMeasurement();
      const pointIndex = measurement.points.length;
      measurement.points.push(point.clone());
      const marker = createMarker(measurement, pointIndex, point);
      measurement.markers.push(marker);
      setSelectedMarker(marker);
      updateLine(measurement);
      updateReadout();
    }}

    function removeMeasurement(measurement) {{
      if (!measurement) return;
      measurement.markers.forEach(marker => {{
        removeMarkerPickObjects(marker);
        scene.remove(marker);
        disposeObject(marker);
      }});
      if (measurement.line) {{
        scene.remove(measurement.line);
        measurement.line.geometry.dispose();
        measurement.line.material.dispose();
      }}
      removeMeasurementLabel(measurement);
      const index = measurements.indexOf(measurement);
      if (index >= 0) measurements.splice(index, 1);
      if (activeMeasurement === measurement) activeMeasurement = measurements[Math.max(0, index - 1)] || null;
      if (selectedMarker && selectedMarker.userData.measurement === measurement) selectedMarker = null;
      updateMeasurementSelect();
      updateReadout();
    }}

    function removeMarker(marker) {{
      if (!marker) return;
      const measurement = marker.userData.measurement;
      const index = measurement.markers.indexOf(marker);
      if (index < 0) return;
      measurement.points.splice(index, 1);
      measurement.markers.splice(index, 1);
      removeMarkerPickObjects(marker);
      scene.remove(marker);
      disposeObject(marker);
      measurement.markers.forEach((item, itemIndex) => item.userData.pointIndex = itemIndex);
      selectedMarker = null;
      if (measurement.points.length === 0) {{
        removeMeasurement(measurement);
        return;
      }}
      updateLine(measurement);
      activeMeasurement = measurement;
      updateMeasurementSelect();
      updateReadout();
    }}

    function clearMeasurements() {{
      [...measurements].forEach(removeMeasurement);
      activeMeasurement = null;
      selectedMarker = null;
      updateReadout();
    }}

    function deleteSelected() {{
      if (selectedMarker) removeMarker(selectedMarker);
      else if (activeMeasurement) removeMeasurement(activeMeasurement);
    }}

    function pointerToRay(clientX, clientY) {{
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
    }}

    function pickMeshAt(clientX, clientY) {{
      pointerToRay(clientX, clientY);
      const hits = raycaster.intersectObjects(meshObjects, true);
      return hits.length > 0 ? hits[0] : null;
    }}

    function pickMarkerAt(clientX, clientY) {{
      pointerToRay(clientX, clientY);
      const hits = raycaster.intersectObjects(markerPickObjects, true);
      return hits.length > 0 ? hits[0].object.userData.marker : null;
    }}

    function pickingModeActive() {{
      return measureMode || centerMode || floorMode;
    }}

    function hidePickAssistVisuals() {{
      hoverMarker.visible = false;
      loupe.style.display = 'none';
    }}

    function setPickAssistEnabled(enabled) {{
      pickAssistEnabled = Boolean(enabled);
      setButtonPressed(pickAssistBtn, pickAssistEnabled);
      if (!pickAssistEnabled) {{
        hidePickAssistVisuals();
        setStatus('Pick assist hidden.');
      }} else {{
        setStatus('Pick assist enabled.');
        if (pointerInsideViewport) updatePickAssist(lastPointer.x, lastPointer.y);
      }}
    }}

    function surfaceNormalFromHit(hit) {{
      if (hit && hit.face) {{
        return hit.face.normal.clone().transformDirection(hit.object.matrixWorld).normalize();
      }}
      return camera.getWorldDirection(new THREE.Vector3()).negate().normalize();
    }}

    function updateHoverMarker(hit) {{
      if (!hit) {{
        hoverMarker.visible = false;
        return;
      }}
      const normal = surfaceNormalFromHit(hit);
      const scale = Math.max(modelRadius * 0.006, camera.position.distanceTo(hit.point) * 0.0045, 0.0008);
      hoverMarker.position.copy(hit.point).addScaledVector(normal, scale * 0.35);
      hoverMarker.scale.setScalar(scale);
      hoverRing.quaternion.copy(camera.quaternion);
      const positions = hoverNormalGeometry.attributes.position;
      positions.setXYZ(0, 0, 0, 0);
      positions.setXYZ(1, normal.x * 5.0, normal.y * 5.0, normal.z * 5.0);
      positions.needsUpdate = true;
      hoverNormalGeometry.computeBoundingSphere();
      hoverMarker.visible = true;
    }}

    function updatePickAssist(clientX, clientY, meshHit = null) {{
      if (!pickAssistEnabled || !pickingModeActive()) {{
        hidePickAssistVisuals();
        return null;
      }}
      const hit = meshHit || pickMeshAt(clientX, clientY);
      updateHoverMarker(hit);
      return hit;
    }}

    function updateLoupe(markerVisible = hoverMarker.visible) {{
      if (!pickAssistEnabled || !pickingModeActive() || !pointerInsideViewport || !markerVisible || loupeDrawFailed) {{
        loupe.style.display = 'none';
        return;
      }}
      const rect = renderer.domElement.getBoundingClientRect();
      if (
        lastPointer.x < rect.left || lastPointer.x > rect.right ||
        lastPointer.y < rect.top || lastPointer.y > rect.bottom
      ) {{
        loupe.style.display = 'none';
        return;
      }}

      const sourceCssSize = 62;
      const scaleX = renderer.domElement.width / rect.width;
      const scaleY = renderer.domElement.height / rect.height;
      const sourceWidth = sourceCssSize * scaleX;
      const sourceHeight = sourceCssSize * scaleY;
      const sourceX = THREE.MathUtils.clamp((lastPointer.x - rect.left) * scaleX - sourceWidth * 0.5, 0, renderer.domElement.width - sourceWidth);
      const sourceY = THREE.MathUtils.clamp((lastPointer.y - rect.top) * scaleY - sourceHeight * 0.5, 0, renderer.domElement.height - sourceHeight);

      try {{
        const size = loupe.width;
        loupeContext.clearRect(0, 0, size, size);
        loupeContext.imageSmoothingEnabled = false;
        loupeContext.drawImage(renderer.domElement, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, size, size);
        const center = size * 0.5;
        loupeContext.save();
        loupeContext.strokeStyle = 'rgba(34, 211, 238, 0.98)';
        loupeContext.lineWidth = 2;
        loupeContext.beginPath();
        loupeContext.moveTo(center - 24, center);
        loupeContext.lineTo(center + 24, center);
        loupeContext.moveTo(center, center - 24);
        loupeContext.lineTo(center, center + 24);
        loupeContext.stroke();
        loupeContext.strokeStyle = 'rgba(250, 204, 21, 0.98)';
        loupeContext.beginPath();
        loupeContext.arc(center, center, 11, 0, Math.PI * 2);
        loupeContext.stroke();
        loupeContext.restore();
      }} catch (_err) {{
        loupeDrawFailed = true;
        loupe.style.display = 'none';
        return;
      }}

      const displaySize = 180;
      let left = lastPointer.x + 18;
      let top = lastPointer.y + 18;
      if (left + displaySize + 12 > window.innerWidth) left = lastPointer.x - displaySize - 18;
      if (top + displaySize + 12 > window.innerHeight) top = lastPointer.y - displaySize - 18;
      left = THREE.MathUtils.clamp(left, 12, Math.max(12, window.innerWidth - displaySize - 12));
      top = THREE.MathUtils.clamp(top, 12, Math.max(12, window.innerHeight - displaySize - 12));
      loupe.style.transform = `translate3d(${{roundToDevicePixel(left)}}px, ${{roundToDevicePixel(top)}}px, 0)`;
      loupe.style.display = 'block';
    }}

    function moveMarker(marker, point) {{
      const measurement = marker.userData.measurement;
      const index = marker.userData.pointIndex;
      marker.position.copy(point);
      measurement.points[index].copy(point);
      updateLine(measurement);
      updateReadout();
    }}

    function clearFloorPickMarkers() {{
      floorPickMarkers.forEach(marker => {{
        scene.remove(marker);
        disposeObject(marker);
      }});
      floorPickMarkers.length = 0;
      floorPickPoints.length = 0;
    }}

    function createFloorPickMarker(point, index) {{
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(markerRadius() * 1.8, 0.003), 16, 8),
        new THREE.MeshBasicMaterial({{ color: index === 0 ? 0x38bdf8 : index === 1 ? 0x22c55e : 0xf97316, depthTest: false }})
      );
      marker.renderOrder = 20;
      marker.position.copy(point);
      scene.add(marker);
      floorPickMarkers.push(marker);
    }}

    function applyWorldAlignment(quaternion) {{
      if (!rootObject) return;
      rootObject.position.applyQuaternion(quaternion);
      rootObject.quaternion.premultiply(quaternion);
      rootObject.updateMatrixWorld(true);

      measurements.forEach(measurement => {{
        measurement.points.forEach(point => point.applyQuaternion(quaternion));
        measurement.markers.forEach(marker => marker.position.applyQuaternion(quaternion));
        updateLine(measurement);
      }});

      const nextTarget = controls.target.clone().applyQuaternion(quaternion);
      setOrbitCenter(nextTarget, false);
      rebuildGridForBox(new THREE.Box3().setFromObject(rootObject));
      floorAlignment.premultiply(quaternion);
      updateFloorResetState();
      saveViewState();
      requestLabelUpdate();
      updateReadout();
    }}

    function alignFloorFromPickedPoints() {{
      if (floorPickPoints.length < 3) return;
      const a = floorPickPoints[0];
      const b = floorPickPoints[1];
      const c = floorPickPoints[2];
      const normal = new THREE.Vector3()
        .subVectors(b, a)
        .cross(new THREE.Vector3().subVectors(c, a));
      if (normal.lengthSq() < 1e-12) {{
        setStatus('Floor alignment needs three non-collinear mesh points.');
        clearFloorPickMarkers();
        return;
      }}
      normal.normalize();
      if (normal.dot(worldUp) < 0) normal.negate();
      const quaternion = new THREE.Quaternion().setFromUnitVectors(normal, worldUp);
      applyWorldAlignment(quaternion);
      setFloorView(null, false);
      saveViewState();
      clearFloorPickMarkers();
      setStatus('Floor aligned; floor view squared to mesh footprint.');
    }}

    function addFloorPickPoint(point) {{
      floorPickPoints.push(point.clone());
      createFloorPickMarker(point, floorPickPoints.length - 1);
      if (floorPickPoints.length >= 3) {{
        alignFloorFromPickedPoints();
        setFloorMode(false, false);
      }} else {{
        setStatus(`Set floor: pick ${{3 - floorPickPoints.length}} more point(s).`);
      }}
    }}

    function resetFloorAlignment() {{
      if (resetFloorBtn.disabled || !rootObject) return;
      const inverse = floorAlignment.clone().invert();
      floorAlignment.identity();
      applyWorldAlignment(inverse);
      floorAlignment.identity();
      updateFloorResetState();
      setStatus('Floor alignment reset.');
    }}

    renderer.domElement.addEventListener('pointerdown', event => {{
      if (event.button !== 0) return;
      pointerInsideViewport = true;
      lastPointer.x = event.clientX;
      lastPointer.y = event.clientY;
      downPoint = {{ x: event.clientX, y: event.clientY }};
      if (measureMode) {{
        const marker = pickMarkerAt(event.clientX, event.clientY);
        if (marker) {{
          draggingMarker = marker;
          dragMoved = false;
          setSelectedMarker(marker);
          controls.enabled = false;
          dragPlane.setFromNormalAndCoplanarPoint(
            camera.getWorldDirection(new THREE.Vector3()).normalize(),
            marker.position
          );
          renderer.domElement.setPointerCapture(event.pointerId);
          event.preventDefault();
          event.stopPropagation();
        }}
      }}
    }}, true);
    renderer.domElement.addEventListener('pointermove', event => {{
      pointerInsideViewport = true;
      lastPointer.x = event.clientX;
      lastPointer.y = event.clientY;
      if (!draggingMarker) {{
        updatePickAssist(event.clientX, event.clientY);
        return;
      }}
      const dx = event.clientX - downPoint.x;
      const dy = event.clientY - downPoint.y;
      dragMoved = dragMoved || Math.hypot(dx, dy) > 2;
      const meshHit = pickMeshAt(event.clientX, event.clientY);
      if (meshHit) {{
        moveMarker(draggingMarker, meshHit.point);
        updatePickAssist(event.clientX, event.clientY, meshHit);
      }} else {{
        updateHoverMarker(null);
        pointerToRay(event.clientX, event.clientY);
        if (raycaster.ray.intersectPlane(dragPlane, dragPlaneHit)) {{
          moveMarker(draggingMarker, dragPlaneHit);
        }}
      }}
      event.preventDefault();
      event.stopPropagation();
    }}, true);
    renderer.domElement.addEventListener('pointerup', event => {{
      if (event.button !== 0 || !downPoint) return;
      pointerInsideViewport = true;
      lastPointer.x = event.clientX;
      lastPointer.y = event.clientY;
      if (draggingMarker) {{
        try {{ renderer.domElement.releasePointerCapture(event.pointerId); }} catch (_err) {{}}
        controls.enabled = true;
        draggingMarker = null;
        downPoint = null;
        event.preventDefault();
        event.stopPropagation();
        return;
      }}
      const dx = event.clientX - downPoint.x;
      const dy = event.clientY - downPoint.y;
      if (Math.hypot(dx, dy) <= 4) {{
        const marker = pickMarkerAt(event.clientX, event.clientY);
        const meshHit = pickMeshAt(event.clientX, event.clientY);
        if (meshHit && floorMode) {{
          addFloorPickPoint(meshHit.point);
        }} else if (marker && !centerMode && !measureMode) {{
          setSelectedMarker(marker);
        }} else if (meshHit && centerMode) {{
          setOrbitCenter(meshHit.point);
          setCenterMode(false);
        }} else if (meshHit && measureMode) {{
          addMeasurementPoint(meshHit.point);
        }}
      }}
      downPoint = null;
      updatePickAssist(event.clientX, event.clientY);
    }}, true);
    renderer.domElement.addEventListener('pointercancel', event => {{
      if (draggingMarker) {{
        try {{ renderer.domElement.releasePointerCapture(event.pointerId); }} catch (_err) {{}}
      }}
      controls.enabled = true;
      draggingMarker = null;
      downPoint = null;
      hidePickAssistVisuals();
    }}, true);
    renderer.domElement.addEventListener('pointerleave', () => {{
      pointerInsideViewport = false;
      hidePickAssistVisuals();
    }});
    renderer.domElement.addEventListener('contextmenu', event => event.preventDefault());

    function setMeasureMode(enabled) {{
      measureMode = Boolean(enabled);
      setButtonPressed(measureBtn, measureMode);
      if (measureMode) setCenterMode(false);
      if (measureMode) setFloorMode(false);
      if (measureMode) {{
        setStatus('Measure: hover marker shows the exact mesh point that will be picked.');
        if (pointerInsideViewport) updatePickAssist(lastPointer.x, lastPointer.y);
      }} else {{
        hidePickAssistVisuals();
      }}
    }}

    function setCenterMode(enabled) {{
      centerMode = Boolean(enabled);
      setButtonPressed(centerBtn, centerMode);
      if (centerMode) setMeasureMode(false);
      if (centerMode) setFloorMode(false);
      if (centerMode) {{
        setStatus('Set center: hover marker shows the exact mesh point that will become the orbit pivot.');
        if (pointerInsideViewport) updatePickAssist(lastPointer.x, lastPointer.y);
      }} else {{
        hidePickAssistVisuals();
      }}
    }}

    function setFloorMode(enabled, clearPicks = true) {{
      floorMode = Boolean(enabled);
      setButtonPressed(floorBtn, floorMode);
      if (clearPicks) clearFloorPickMarkers();
      if (floorMode) {{
        setMeasureMode(false);
        setCenterMode(false);
        setStatus('Set floor: pick 3 mesh points. Hover marker shows the exact surface hit.');
        if (pointerInsideViewport) updatePickAssist(lastPointer.x, lastPointer.y);
      }} else {{
        hidePickAssistVisuals();
      }}
    }}

    measureBtn.addEventListener('click', () => setMeasureMode(!measureMode));
    centerBtn.addEventListener('click', () => setCenterMode(!centerMode));
    deleteBtn.addEventListener('click', deleteSelected);
    clearBtn.addEventListener('click', clearMeasurements);
    pickAssistBtn.addEventListener('click', () => setPickAssistEnabled(!pickAssistEnabled));
    edgeBtn.addEventListener('click', () => setEdgeOverlayVisible(!edgesVisible));
    fitBtn.addEventListener('click', () => {{
      if (rootObject) fitCamera(rootObject);
    }});
    resetBtn.addEventListener('click', () => {{
      restoreSavedViewState();
      viewSelect.value = '';
    }});
    levelViewBtn.addEventListener('click', () => levelCameraToFloor());
    floorViewBtn.addEventListener('click', () => setFloorView());
    rollLeftBtn.addEventListener('click', () => rollCamera(-rollStepDegrees()));
    rollRightBtn.addEventListener('click', () => rollCamera(rollStepDegrees()));
    floorBtn.addEventListener('click', () => setFloorMode(!floorMode));
    resetFloorBtn.addEventListener('click', resetFloorAlignment);
    gridBtn.addEventListener('click', () => {{
      grid.visible = !grid.visible;
      setButtonPressed(gridBtn, grid.visible);
    }});
    labelsBtn.addEventListener('click', () => {{
      labelsVisible = !labelsVisible;
      document.body.classList.toggle('labels-hidden', !labelsVisible);
      setButtonPressed(labelsBtn, labelsVisible);
      requestLabelUpdate();
      updateAllMeasurementLabels(true);
    }});
    viewSelect.addEventListener('change', () => {{
      if (viewSelect.value) setCameraView(viewSelect.value);
    }});
    measurementSelect.addEventListener('change', () => {{
      const id = Number(measurementSelect.value || 0);
      activeMeasurement = measurements.find(measurement => measurement.id === id) || null;
      setSelectedMarker(null);
      updateReadout();
    }});
    unitSelect.addEventListener('change', () => {{
      updateReadout();
      updateAllMeasurementLabels();
    }});
    scaleBtn.addEventListener('click', rescaleFromGroundTruth);
    truthInput.addEventListener('keydown', event => {{
      if (event.key === 'Enter') rescaleFromGroundTruth();
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      updateAllMeasurementLabels();
      const drawCleanLoupe = pickAssistEnabled && pickingModeActive() && pointerInsideViewport && hoverMarker.visible && !loupeDrawFailed;
      if (drawCleanLoupe) {{
        hoverMarker.visible = false;
        renderer.render(scene, camera);
        hoverMarker.visible = true;
        updateLoupe(true);
      }} else {{
        updateLoupe(false);
      }}
      renderer.render(scene, camera);
    }}
    loadModel();
    animate();
  </script>
</body>
</html>
"""
