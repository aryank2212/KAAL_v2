"""
test_server.py — Integration test for the FastAPI backend.

Starts the app (with a stubbed YOLO so no model file/GPU is required), hits
`/api/python-data`, and verifies the JSON schema the frontend consumes.

Mirrors the Phase 2 roadmap requirement:
  "Server: start FastAPI, hit /api/python-data, verify JSON schema"
"""

import importlib.util
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


SERVER_PATH = Path(__file__).resolve().parents[2] / "Frontend_Backend" / "backend" / "server.py"


@pytest.fixture(scope="module")
def client():
    import os

    # Never touch the real camera in tests.
    os.environ["KAAL_ENABLE_CAMERA"] = "0"
    spec = importlib.util.spec_from_file_location("kaal_server_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from fastapi.testclient import TestClient

    with TestClient(module.app) as c:
        yield c


def test_python_data_schema(client):
    resp = client.get("/api/python-data")
    assert resp.status_code == 200
    payload = resp.json()

    assert "status" in payload
    assert "noise_score" in payload
    assert "detections" in payload
    assert isinstance(payload["detections"], list)
    assert "is_attack" in payload
    assert "fft_threshold" in payload
    assert "flagged_count" in payload

    metrics = payload.get("systemMetrics", {})
    for key in ("cpu", "memory", "gpu", "bandwidth", "latency", "uptime"):
        assert key in metrics

    assert "data" in payload
    assert isinstance(payload["data"], list)


def test_python_data_types(client):
    payload = client.get("/api/python-data").json()
    assert isinstance(payload["noise_score"], (int, float))
    assert isinstance(payload["status"], str)


def test_video_feed_route_registered(client):
    """Streaming route yields jpeg multipart frames — read bounded chunks."""
    import asyncio

    module = sys.modules["kaal_server_under_test"]
    assert any(r.path == "/api/video_feed" for r in module.app.routes)

    # No worker thread is running, so seed a frame so the stream can yield
    module.latest_frame = b"jpeg-frame-bytes"
    response = asyncio.run(module.video_feed())
    assert "multipart/x-mixed-replace" in response.media_type

    frames = asyncio.run(_collect_frames(response.body_iterator, max_frames=2))
    assert len(frames) == 2
    assert b"--frame" in frames[0] and b"image/jpeg" in frames[0]


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("status", "mode", "is_attack", "camera", "model_loaded", "fft_threshold", "uptime"):
        assert key in body
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_settings_get_put(client):
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    initial = resp.json()
    for key in ("fft_threshold", "conf"):
        assert key in initial
        assert isinstance(initial[key], (int, float))

    new_threshold = 0.777
    resp = client.put("/api/settings", json={"fft_threshold": new_threshold, "conf": 0.5})
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["fft_threshold"] == pytest.approx(new_threshold)
    assert updated["conf"] == pytest.approx(0.5)

    # Values are clamped to sane ranges.
    resp = client.put("/api/settings", json={"fft_threshold": 5.0, "conf": 0.0})
    assert resp.json()["fft_threshold"] <= 1.0
    assert resp.json()["conf"] >= 0.05


def test_websocket_broadcasts_snapshot(client):
    """On connect the /ws endpoint immediately sends the latest snapshot."""
    module = sys.modules["kaal_server_under_test"]
    assert any(getattr(r, "path", None) == "/ws" for r in module.app.routes)

    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert "status" in msg
        assert "noise_score" in msg
        assert "detections" in msg


async def _collect_frames(gen, max_frames):

    out = []
    async for chunk in gen:
        out.append(chunk)
        if len(out) >= max_frames:
            break
    await gen.aclose()
    return out
