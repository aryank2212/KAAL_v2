"""
test_server.py — Integration test for the FastAPI backend.

Starts the app with the stock YOLO weights (stubbed when Ultralytics is absent), hits
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
    with pytest.MonkeyPatch.context() as mp:
        # Never touch the real camera. Pin the stock YOLO weights (stubbed when
        # Ultralytics is absent) instead of the registry's production model,
        # so the test never depends on trained weights being on disk.
        mp.setenv("KAAL_ENABLE_CAMERA", "0")
        mp.setenv("KAAL_MODEL_PATH", str(SERVER_PATH.parents[2] / "MODEL" / "yolov8m.pt"))
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
    for key in ("status", "mode", "operating_mode", "feed_state", "frame_age_seconds", "detector_health", "is_attack", "camera", "model_loaded", "fft_threshold", "uptime"):
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


# ── Regression tests for the Phase 4 review findings ─────────────────────


def _server():
    return sys.modules["kaal_server_under_test"]


def test_worker_emit_reaches_broadcast_queue(client):
    """A snapshot emitted from the worker thread must land in the asyncio queue.

    Regression: `_emit` used to schedule an `async def` with call_soon_threadsafe,
    creating a coroutine that was never awaited — live WS clients froze.
    """
    import asyncio
    import threading

    module = _server()

    async def scenario():
        queue = asyncio.Queue(maxsize=4)
        original = module.state.queue
        module.state.queue = queue
        try:
            worker = module.CameraWorker(module.DetectorAdapter("", "yolo"))
            worker._loop = asyncio.get_running_loop()
            t = threading.Thread(target=worker._emit, args=({"frame_no": 1},))
            t.start()
            t.join()
            await asyncio.sleep(0.1)
            return queue.qsize(), (queue.get_nowait() if queue.qsize() else None)
        finally:
            module.state.queue = original

    size, item = asyncio.run(scenario())
    assert size == 1
    assert item == {"frame_no": 1}


def test_enqueue_latest_drops_oldest_when_full():
    import asyncio

    module = _server()
    queue = asyncio.Queue(maxsize=2)
    for i in range(5):
        module.enqueue_latest(queue, {"i": i})
    assert [queue.get_nowait()["i"] for _ in range(queue.qsize())] == [3, 4]


def test_settings_reject_non_finite(client):
    resp = client.put(
        "/api/settings",
        content=b'{"fft_threshold": NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert client.get("/api/settings").json()["fft_threshold"] == pytest.approx(
        _server().settings.snapshot()["fft_threshold"]
    )


def test_settings_token_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(_server(), "API_TOKEN", "s3cret")
    assert client.put("/api/settings", json={"conf": 0.5}).status_code == 401
    bad = client.put("/api/settings", json={"conf": 0.5}, headers={"X-KAAL-Token": "nope"})
    assert bad.status_code == 401
    ok = client.put("/api/settings", json={"conf": 0.5}, headers={"X-KAAL-Token": "s3cret"})
    assert ok.status_code == 200


def test_cors_rejects_foreign_origin(client):
    resp = client.options(
        "/api/settings",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example"


def test_cors_allows_dashboard_origin(client):
    resp = client.options(
        "/api/settings",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "PUT"},
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_websocket_rejects_foreign_origin(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()


class _RecordingDetector:
    """DetectorAdapter stand-in that records the confidence it was called with."""

    def __init__(self):
        self.confs = []
        self.names = {0: "tank"}

    def info(self):
        return {"backend": "fake", "weights": None, "fallback": False}

    def predict(self, frame, conf):
        self.confs.append(conf)
        return [
            {"bbox": [100, 100, 300, 260], "confidence": 0.9, "class_id": 0, "class_name": "tank"}
        ]


def test_process_frame_uses_live_conf_and_scores_regions(client):
    import numpy as np

    module = _server()
    detector = _RecordingDetector()
    worker = module.CameraWorker(detector)
    module.settings.update(fft_threshold=None, conf=0.7)

    frame = np.random.RandomState(0).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    snap = worker.process_frame(frame)

    assert detector.confs == [pytest.approx(0.7)]
    det = snap["detections"][0]
    assert det["label"] == "tank"
    assert det["anomaly_score"] is not None  # region was FFT-scored
    assert "temporal" in snap and "detector" in snap
    assert snap["frame"].shape == (480, 640, 3)  # within PROCESS_SIDE: not resized


def test_metrics_sampler_throttles_nvidia_smi(monkeypatch):
    module = _server()
    calls = []
    monkeypatch.setattr(module, "_gpu_metrics", lambda: calls.append(1) or None)
    sampler = module.MetricsSampler(interval=60.0)
    for _ in range(10):
        metrics = sampler.sample(latency_ms=12.5)
    assert len(calls) == 1
    assert metrics["latency"] == 12.5


def test_operating_mode_and_feed_freshness_are_explicit(client):
    """The API distinguishes configured source type from actual feed freshness."""
    import time

    module = _server()
    original = (
        module.ENABLE_CAMERA,
        module.SOURCE_IS_FILE,
        module.state.camera_connected,
        module.last_frame_at,
    )
    try:
        module.ENABLE_CAMERA = False
        module.SOURCE_IS_FILE = False
        module.state.camera_connected = False
        module.last_frame_at = None
        payload = module._snapshot()
        assert payload["operating_mode"] == "SIMULATION"
        assert payload["feed_state"] == "SIMULATED"

        module.ENABLE_CAMERA = True
        module.SOURCE_IS_FILE = True
        module.state.camera_connected = True
        module.last_frame_at = time.time()
        payload = module._snapshot()
        assert payload["operating_mode"] == "REPLAY"
        assert payload["feed_state"] == "CONNECTED"

        module.SOURCE_IS_FILE = False
        module.last_frame_at = time.time()
        payload = module._snapshot()
        assert payload["operating_mode"] == "LIVE"
        assert payload["feed_state"] == "CONNECTED"
        assert payload["frame_age_seconds"] is not None

        module.last_frame_at = time.time() - module.FEED_STALE_SECONDS - 1
        assert module._snapshot()["feed_state"] == "STALE"

        module.state.camera_connected = False
        assert module._snapshot()["feed_state"] == "NO_SIGNAL"
    finally:
        (
            module.ENABLE_CAMERA,
            module.SOURCE_IS_FILE,
            module.state.camera_connected,
            module.last_frame_at,
        ) = original
