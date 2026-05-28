import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest

from pilot_transfer import PilotTransferSummary, fetch_pilot_index, sync_from_pilot


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
    server, thread, base_url = _serve(
        {
            "run_01/frame.txt": b"hello\n",
            "stereo_sessions/session-a/manifest.json": b"{}\n",
        }
    )
    try:
        files = fetch_pilot_index(base_url)
        assert [item.path for item in files] == [
            "run_01/frame.txt",
            "stereo_sessions/session-a/manifest.json",
        ]

        summary = sync_from_pilot(base_url, tmp_path / "incoming")
        assert summary.scanned == 2
        assert summary.copied == 2
        assert (tmp_path / "incoming" / "run_01" / "frame.txt").read_text(encoding="utf-8") == "hello\n"

        second = sync_from_pilot(base_url, tmp_path / "incoming")
        assert second.copied == 0
        assert second.skipped == 2
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


def test_pilot_transfer_sync_worker_reports_success(tmp_path: Path):
    pytest.importorskip("PyQt6")
    from gui.pilot_transfer_sync import PilotTransferSyncWorker

    payloads = []

    def _fake_sync(base_url, destination, *, overwrite=False, timeout=10.0):
        assert base_url == "http://pilot.test:8765"
        assert Path(destination) == tmp_path / "incoming"
        assert overwrite is False
        assert timeout == 1.5
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
    worker.run()

    assert payloads
    assert payloads[0]["ok"] is True
    assert payloads[0]["summary"].copied == 1
