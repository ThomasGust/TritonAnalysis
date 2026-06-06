import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pytest

import triton_analysis.sync.pilot_transfer as pilot_transfer
from triton_analysis.sync.pilot_transfer import (
    PilotTransferEvent,
    PilotTransferSummary,
    fetch_pilot_index,
    sync_from_pilot,
    wait_for_pilot_change,
)


class _TransferHandler(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}

    def log_message(self, _format: str, *_args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/index.json":
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
        if parsed.path == "/events":
            query = parse_qs(parsed.query)
            since = int((query.get("since") or ["0"])[0] or "0")
            event_id = len(self.files) + 1
            total_bytes = sum(len(data) for data in self.files.values())
            payload = {
                "type": "triton-analysis-transfer-event",
                "version": 1,
                "event_id": event_id,
                "changed": event_id != since,
                "file_count": len(self.files),
                "total_bytes": total_bytes,
                "generated_at": 1_700_000_000.0,
                "stable_seconds": 0.75,
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith("/files/"):
            rel = unquote(parsed.path[len("/files/") :])
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


def test_wait_for_pilot_change_reads_event_endpoint():
    server, thread, base_url = _serve({"run_01/frame.txt": b"hello\n"})
    try:
        event = wait_for_pilot_change(base_url, since_event_id=0, timeout=0.1)

        assert event.base_url == base_url
        assert event.changed is True
        assert event.event_id == 2
        assert event.file_count == 1
        assert event.total_bytes == len(b"hello\n")
        assert event.stable_seconds == 0.75

        unchanged = wait_for_pilot_change(base_url, since_event_id=event.event_id, timeout=0.1)
        assert unchanged.changed is False
        assert unchanged.event_id == event.event_id
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


def test_pilot_transfer_sync_worker_skips_sync_when_watch_has_no_change(tmp_path: Path):
    pytest.importorskip("PyQt6")
    from triton_analysis.gui.pilot_transfer_sync import PilotTransferSyncWorker

    payloads = []
    progress_payloads = []

    def _fake_event(base_url, *, since_event_id=0, timeout=20.0, request_timeout=None):
        assert base_url == "http://pilot.test:8765"
        assert since_event_id == 7
        return PilotTransferEvent(base_url=base_url, event_id=7, changed=False)

    def _unexpected_sync(*_args, **_kwargs):
        raise AssertionError("sync should not run when Pilot reports no change")

    worker = PilotTransferSyncWorker(
        base_url="http://pilot.test:8765",
        destination=tmp_path / "incoming",
        watch_for_changes=True,
        since_event_id=7,
        event_timeout=0.1,
        event_fn=_fake_event,
        sync_fn=_unexpected_sync,
    )
    worker.finished.connect(payloads.append)
    worker.progress.connect(progress_payloads.append)
    worker.run()

    assert payloads[0]["ok"] is True
    assert payloads[0]["no_change"] is True
    assert payloads[0]["event_id"] == 7
    assert [payload["event"] for payload in progress_payloads] == ["watch_start", "watch_done"]


def test_pilot_transfer_sync_worker_syncs_after_watch_change(tmp_path: Path):
    pytest.importorskip("PyQt6")
    from triton_analysis.gui.pilot_transfer_sync import PilotTransferSyncWorker

    payloads = []
    progress_payloads = []

    def _fake_event(base_url, *, since_event_id=0, timeout=20.0, request_timeout=None):
        return PilotTransferEvent(base_url=base_url, event_id=8, changed=True, file_count=1, total_bytes=42)

    def _fake_sync(base_url, destination, *, overwrite=False, timeout=10.0, progress_callback=None):
        return PilotTransferSummary(base_url=base_url, destination=destination, scanned=1, copied=1, bytes_copied=42)

    worker = PilotTransferSyncWorker(
        base_url="http://pilot.test:8765",
        destination=tmp_path / "incoming",
        watch_for_changes=True,
        since_event_id=7,
        event_timeout=0.1,
        event_fn=_fake_event,
        sync_fn=_fake_sync,
    )
    worker.finished.connect(payloads.append)
    worker.progress.connect(progress_payloads.append)
    worker.run()

    assert payloads[0]["ok"] is True
    assert payloads[0]["event_id"] == 8
    assert payloads[0]["summary"].copied == 1
    assert [payload["event"] for payload in progress_payloads][:2] == ["watch_start", "watch_done"]
