import os
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from triton_analysis.gui.realityscan_model_viewer_window import ModelViewerServer, RealityScanModelViewerWindow
from triton_analysis.gui.style import apply_modern_style


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        apply_modern_style(app)
    return app


def _sample_obj(tmp_path: Path) -> Path:
    model = tmp_path / "underwater_model_metric.obj"
    model.write_text(
        "mtllib underwater_model_metric.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )
    (tmp_path / "underwater_model_metric.mtl").write_text(
        "newmtl material_0\nKd 0.8 0.8 0.8\n",
        encoding="utf-8",
    )
    return model


def test_model_viewer_server_serves_threejs_viewer_and_model(tmp_path: Path):
    model = _sample_obj(tmp_path)
    server = ModelViewerServer(model)
    try:
        url = server.start()
        html = urllib.request.urlopen(url, timeout=5).read().decode("utf-8")
        obj_text = urllib.request.urlopen(url.replace("/viewer.html", "/files/underwater_model_metric.obj"), timeout=5).read().decode("utf-8")

        assert "three@0.160.0" in html
        assert "TrackballControls" in html
        assert "OrbitControls" not in html
        assert "OBJLoader" in html
        assert "Distance:" in html
        assert "id=\"labels\"" in html
        assert "id=\"loupe\"" in html
        assert "measurement-label" in html
        assert "ensureMeasurementLabel" in html
        assert "updateAllMeasurementLabels" in html
        assert "requestLabelUpdate" in html
        assert "roundToDevicePixel" in html
        assert "renderer.domElement.getBoundingClientRect" in html
        assert "labelState" in html
        assert "measurementSelect" in html
        assert "measurements = []" in html
        assert "deleteSelected" in html
        assert "draggingMarker" in html
        assert "setOrbitCenter" in html
        assert "orbitCursor" in html
        assert "viewSelect" in html
        assert "resetBtn" in html
        assert "pickAssistBtn" in html
        assert "edgeBtn" in html
        assert "levelViewBtn" in html
        assert "floorViewBtn" in html
        assert "rollLeftBtn" in html
        assert "rollRightBtn" in html
        assert "rollStepSelect" in html
        assert "truthInput" in html
        assert "truthUnitSelect" in html
        assert '<option value="cm" selected>cm</option>' in html
        assert "distanceSignificantDigits = 4" in html
        assert "formatSignificantValue" in html
        assert "value.toPrecision(digits)" not in html
        assert "scaleBtn" in html
        assert "freeOrbitBtn" not in html
        assert "floorBtn" in html
        assert "resetFloorBtn" in html
        assert "gridBtn" in html
        assert "labelsBtn" in html
        assert "controls.mouseButtons" in html
        assert "controls.staticMoving = false" in html
        assert "controls.dynamicDampingFactor = 0.12" in html
        assert "controls.noRotate = false" in html
        assert "function setControlMode" not in html
        assert "function saveViewState" in html
        assert "saveState" not in html
        assert "function rebuildGridForBox" in html
        assert "function setCameraPose" in html
        assert "function setFloorView" in html
        assert "function levelCameraToFloor" in html
        assert "function computeFootprintMajorAxis" in html
        assert "function updatePickAssist" in html
        assert "function updateHoverMarker" in html
        assert "function updateLoupe" in html
        assert "function rebuildEdgeOverlay" in html
        assert "function setEdgeOverlayVisible" in html
        assert "function rollCamera" in html
        assert "function applyUniformModelScale" in html
        assert "function rescaleFromGroundTruth" in html
        assert "function selectedCompleteMeasurement" in html
        assert "function centerObjectAtOrigin" in html
        assert "function applyWorldAlignment" in html
        assert "function alignFloorFromPickedPoints" in html
        assert "cumulativeModelScale" in html
        assert "hoverMarker" in html
        assert "hoverMarker.visible = false" in html
        assert "updateLoupe(true)" in html
        assert "new THREE.EdgesGeometry" in html
        assert "new THREE.Vector3(0, 0, 1).normalize()" in html
        assert "pickAssistBtn.addEventListener('click'" in html
        assert "scaleBtn.addEventListener('click', rescaleFromGroundTruth)" in html
        assert "controls.enableRotate = !measureMode" not in html
        assert "3D Grid" in html
        assert "* 0.0014" in html
        assert "v 1 0 0" in obj_text
    finally:
        server.stop()


def test_model_viewer_window_starts_local_viewer(tmp_path: Path):
    app = _app()
    model = _sample_obj(tmp_path)
    window = RealityScanModelViewerWindow(model_path=str(model))
    try:
        window._start_viewer()

        assert window.panel.unit_combo.currentData() == "cm"
        assert window.url_edit.text().startswith("http://127.0.0.1:")
        assert window.browser_btn.isEnabled()
        assert window.reload_btn.isEnabled()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_model_viewer_reuses_existing_server_for_same_model(tmp_path: Path, monkeypatch):
    app = _app()
    model = _sample_obj(tmp_path)
    window = RealityScanModelViewerWindow(model_path=str(model))
    original_server = None
    original_stop = None
    try:
        window._start_viewer()
        original_server = window.panel._server
        assert original_server is not None
        original_stop = original_server.stop

        def fail_stop():
            raise AssertionError("same model should not restart the viewer server")

        monkeypatch.setattr(original_server, "stop", fail_stop)
        window._start_viewer()

        assert window.panel._server is original_server
    finally:
        if original_server is not None and original_stop is not None:
            original_stop()
            window.panel._server = None
        window.close()
        window.deleteLater()
        app.processEvents()
