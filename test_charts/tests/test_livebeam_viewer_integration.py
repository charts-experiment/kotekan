#!/usr/bin/env python3
"""test_livebeam_viewer_integration.py - Automated smoke test for Kotekan Viewer REST Telemetry.

Spins up a lightweight mock HTTP server simulating Kotekan's REST API on port 12048,
launches `livebeam_server.py`, validates that proxy endpoints (/telemetry, /targets, /trackers, /mode)
return proper JSON telemetry, and cleanly shuts down.
"""

from __future__ import annotations

import http.server
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
pytest.importorskip("twisted")
pytest.importorskip("autobahn")


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class MockKotekanHandler(http.server.BaseHTTPRequestHandler):
    """Simulates Kotekan REST endpoints."""

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_GET(self):
        if self.path == "/direct_beam_tracker/telemetry":
            payload = {
                "timestamp_fpga_ns": 1729900000000000000,
                "kernel_time_ms": 0.284,
                "realtime_budget_utilization_pct": 2.22,
                "active_beams": 4,
                "max_beams": 16,
                "vram_allocated_mb": 244.0,
                "vram_total_mb": 32110.0,
                "gpu_temp_c": 30.0,
                "gpu_power_w": 70.5,
                "status": "HEALTHY",
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/beam_tracker/targets":
            payload = [
                {
                    "beam_id": 0,
                    "name": "B0329+54",
                    "ra_deg": 53.25,
                    "dec_deg": 54.58,
                    "l": 0.05,
                    "m": -0.02,
                    "n": 0.998,
                    "active": True,
                },
                {
                    "beam_id": 1,
                    "name": "B0833-45",
                    "ra_deg": 128.8,
                    "dec_deg": -45.1,
                    "l": -0.10,
                    "m": 0.04,
                    "n": 0.994,
                    "active": True,
                },
            ]
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/trackers":
            payload = {
                "cudaDirectBeamTracker_kernel_latency_ms": 0.284,
                "cudaDirectBeamTracker_budget_utilization_pct": 2.22,
                "cudaDirectBeamTracker_active_beams": 4,
                "cudaDirectBeamTracker_vram_mb": 244.0,
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()


def test_viewer_integration():
    repo_root = Path(__file__).resolve().parent.parent.parent
    server_script = repo_root / "python" / "scripts" / "js_viewer" / "livebeam_server.py"

    rest_port = get_free_port()
    http_port = get_free_port()
    ws_port = get_free_port()
    kotekan_port = get_free_port()

    # 1. Start Mock Kotekan REST server
    mock_server = http.server.HTTPServer(("127.0.0.1", rest_port), MockKotekanHandler)
    server_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[TEST] Mock Kotekan REST Server running on port {rest_port}")

    # 2. Launch livebeam_server.py
    cmd = [
        sys.executable,
        str(server_script),
        "--kotekan-rest-port", str(rest_port),
        "--http-port", str(http_port),
        "--ws-port", str(ws_port),
        "--kotekan-port", str(kotekan_port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[TEST] Launched livebeam_server.py (PID {proc.pid}) on HTTP port {http_port}")

    # Wait for HTTP server to become responsive
    ready = False
    for _ in range(25):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/mode", timeout=0.5) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass

    assert ready, "livebeam_server failed to start within timeout"
    print("[TEST] livebeam_server is responsive.")

    try:
        # 3. Test /telemetry endpoint
        telemetry_url = f"http://127.0.0.1:{http_port}/telemetry"
        with urllib.request.urlopen(telemetry_url, timeout=2.0) as resp:
            assert resp.status == 200, f"Expected 200 from /telemetry, got {resp.status}"
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[TEST] /telemetry response: {data}")
            assert data["active_beams"] == 4
            assert data["kernel_time_ms"] == 0.284
            assert data["gpu_temp_c"] == 30.0
            assert data["status"] == "HEALTHY"

        # 4. Test /targets endpoint
        targets_url = f"http://127.0.0.1:{http_port}/targets"
        with urllib.request.urlopen(targets_url, timeout=2.0) as resp:
            assert resp.status == 200, f"Expected 200 from /targets, got {resp.status}"
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[TEST] /targets response: {len(data)} targets")
            assert len(data) == 2
            assert data[0]["name"] == "B0329+54"
            assert data[1]["name"] == "B0833-45"

        # 5. Test /trackers endpoint
        trackers_url = f"http://127.0.0.1:{http_port}/trackers"
        with urllib.request.urlopen(trackers_url, timeout=2.0) as resp:
            assert resp.status == 200, f"Expected 200 from /trackers, got {resp.status}"
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[TEST] /trackers response: {data}")
            assert "cudaDirectBeamTracker_kernel_latency_ms" in data
            assert data["cudaDirectBeamTracker_kernel_latency_ms"] == 0.284

        print("\n>>> ALL VIEWER REST INTEGRATION SMOKE TESTS PASSED! <<<")

    finally:
        # Clean shutdown
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        mock_server.shutdown()
        server_thread.join(timeout=1.0)
        print("[TEST] Cleanly shut down livebeam_server and mock REST server.")


if __name__ == "__main__":
    test_viewer_integration()
