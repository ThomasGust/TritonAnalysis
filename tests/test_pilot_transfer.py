import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest

import triton_analysis.sync.pilot_transfer as pilot_transfer
from triton_analysis.sync.pilot_transfer import (
    PilotTransferSummary,
    fetch_pilot_index,
    sync_from_pilot,
)


class _TransferHandler(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}

    def log_message(self, _format: str, *_args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/index.json":
            payload = {
                "type": "triton-analysis-transfer-index",
                "version": 1,
                "files": [
                    {"path": path, "size": len(data), "mtime_ns": 1_700_000_000_000_000_000}
                    for path, data in sorted(self.files.items())
                ],
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/files/"):
            rel = unquote(self.path[len("/files/") :])
            data = self.files.get(rel)
            if data is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)


def _serve(files: dict[str, bytes]):
    handler = type("TransferHandler", (_TransferHandler,), {"files": files})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def test_fetch_pilot_index_and_sync(tmp_path: Path):
    served_files = {
        "run_01/frame.txt": b"hello\n",
        "stereo_sessions/session-a/manifest.json": b"{}\n",
    }
    server, thread, base_url = _serve(served_files)
    try:
        files = fetch_pilot_index(base_url)
        assert [item.path for item in files] == [
            "run_01/frame.txt",
            "stereo_sessions/session-a/manifest.json",
        ]

        incoming = tmp_path / "incoming"
        events = []
        summary = sync_from_pilot(base_url, incoming, progress_callback=events.append)
        assert summary.scanned == 2
        assert summary.copied == 2
        assert summary.bytes_scanned == sum(len(data) for data in served_files.values())
        assert (incoming / "run_01" / "frame.txt").read_text(encoding="utf-8") == "hello\n"
        assert any(event["event"] == "copy_start" for event in events)
        assert any(event["event"] == "copy_progress" for event in events)
        assert events[-1]["event"] == "complete"

        second = sync_from_pilot(base_url, incoming)
        assert second.copied == 0
        assert second.skipped == 2

        shutil.rmtree(incoming)
        third = sync_from_pilot(base_url, incoming)
        assert third.copied == 2
        assert third.skipped == 0
        assert (incoming / "run_01" / "frame.txt").read_text(encoding="utf-8") == "hello\n"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_sync_dry_run_does_not_write(tmp_path: Path):
    server, thread, base_url = _serve({"run_01/frame.txt": b"hello\n"})
    try:
        summary = sync_from_pilot(base_url, tmp_path / "incoming", dry_run=True)

        assert summary.copied == 1
        assert not (tmp_path / "incoming").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_fetch_pilot_index_sorts_newest_run_folders_first():
    served_files = {
        "20260605-120000/frame_a.txt": b"older\n",
        "20260605-130000/frame_a.txt": b"newer\n",
        "20260605-130000/frame_b.txt": b"newer-b\n",
        "run_01/frame.txt": b"legacy\n",
    }
    server, thread, base_url = _serve(served_files)
    try:
        files = fetch_pilot_index(base_url)

        assert [item.path for item in files] == [
            "20260605-130000/frame_a.txt",
            "20260605-130000/frame_b.txt",
            "20260605-120000/frame_a.txt",
            "run_01/frame.txt",
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_sync_from_pilot_requires_reachable_transfer_server(monkeypatch, tmp_path: Path):
    source = tmp_path / "TritonPilot" / "recordings"
    run_file = source / "20260605-130000" / "frame.txt"
    run_file.parent.mkdir(parents=True)
    run_file.write_text("local\n", encoding="utf-8")

    def _raise_network_error(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(pilot_transfer, "fetch_pilot_index", _raise_network_error)
    incoming = tmp_path / "incoming"
    events = []

    with pytest.raises(OSError, match="network unavailable"):
        sync_from_pilot(
            "http://10.77.0.1:8765",
            incoming,
            progress_callback=events.append,
        )

    assert not incoming.exists()
    assert run_file.read_text(encoding="utf-8") == "local\n"
    assert [event["event"] for event in events] == ["index_start"]
    assert all(event["event"] != "local_fallback" for event in events)


def test_sync_from_pilot_copies_from_http_transfer_server(tmp_path: Path):
    server, thread, base_url = _serve({"20260605-130000/frame.txt": b"served\n"})
    try:
        incoming = tmp_path / "incoming"
        summary = sync_from_pilot(
            base_url,
            incoming,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    assert summary.base_url == base_url
    assert summary.copied == 1
    assert (incoming / "20260605-130000" / "frame.txt").read_text(encoding="utf-8") == "served\n"


def test_pilot_transfer_sync_worker_reports_success(tmp_path: Path):
    pytest.importorskip("PyQt6")
    from triton_analysis.gui.pilot_transfer_sync import PilotTransferSyncWorker

    payloads = []

    progress_payloads = []

    def _fake_sync(base_url, destination, *, overwrite=False, timeout=10.0, progress_callback=None):
        assert base_url == "http://pilot.test:8765"
        assert Path(destination) == tmp_path / "incoming"
        assert overwrite is False
        assert timeout == 1.5
        assert progress_callback is not None
        progress_callback({"event": "copy_start", "path": "run_01/frame.txt"})
        return PilotTransferSummary(
            base_url=base_url,
            destination=destination,
            scanned=2,
            copied=1,
            skipped=1,
            bytes_copied=42,
        )

    worker = PilotTransferSyncWorker(
        base_url="http://pilot.test:8765",
        destination=tmp_path / "incoming",
        timeout=1.5,
        sync_fn=_fake_sync,
    )
    worker.finished.connect(payloads.append)
    worker.progress.connect(progress_payloads.append)
    worker.run()

    assert payloads
    assert payloads[0]["ok"] is True
    assert payloads[0]["summary"].copied == 1
    assert [payload["event"] for payload in progress_payloads] == ["sync_start", "copy_start"]
