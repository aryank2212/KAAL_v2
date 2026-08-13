"""
server.py — Phase 4 FastAPI backend for the KAAL dashboard.

Replaces the Phase 0 thread + global-dict server with an asyncio
producer-consumer pipeline:

    Camera worker (thread)
        -> capture -> detector -> FFT region scoring
        -> snapshot (dict) -> asyncio.Queue -> WebSocket broadcast
                                           -> latest snapshot (HTTP)

Endpoints
    GET  /api/python-data   latest snapshot (legacy polled endpoint)
    GET  /api/video_feed    MJPEG stream of the annotated frame
    GET  /api/health        liveness / mode / camera / model status
    GET  /api/settings      live FFT threshold + detector confidence
    PUT  /api/settings      update them at runtime
    WS   /ws                real-time snapshot broadcast

Phase 4 roadmap items addressed:
  - asyncio producer-consumer + WebSocket (replaces 1.5 s polling)
  - real system metrics via psutil (fallback) + nvidia-smi
  - health check endpoint, env-variable configuration (KAAL_*)
  - graceful shutdown and camera disconnect recovery with backoff
  - per-detection adversarial flags (FFT region scores, like the model)
"""

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

logging.basicConfig(
    level=os.environ.get("KAAL_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Allow importing MODEL's model-free scoring modules (fft_scorer, calibration).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from MODEL.fft_scorer import compute_fft_anomaly_score, compute_region_scores  # noqa: E402
from MODEL.calibration import load_calibrated_threshold  # noqa: E402

# ── Configuration (environment variables) ────────────────────────────────


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


HOST = _env_str("KAAL_HOST", "0.0.0.0")
PORT = _env_int("KAAL_PORT", 3000)
CAMERA_INDEX = _env_int("KAAL_CAMERA_INDEX", 0)
ENABLE_CAMERA = _env_str("KAAL_ENABLE_CAMERA", "1") not in ("0", "false", "False")
MODEL_PATH = _env_str(
    "KAAL_MODEL_PATH",
    str(_REPO_ROOT / "MODEL" / "yolov8m.pt"),
)
CALIBRATION_PROFILE = _env_str("KAAL_CALIBRATION", "")
DEFAULT_CONF = _env_float("KAAL_CONF", 0.4)
DEFAULT_FFT_THRESHOLD = _env_float("KAAL_FFT_THRESHOLD", 0.5)
HIGH_FREQ_RATIO = _env_float("KAAL_HIGH_FREQ_RATIO", 0.25)
HISTORY_LEN = _env_int("KAAL_HISTORY", 60)
REGION_MIN_SIZE = _env_int("KAAL_REGION_MIN_SIZE", 16)

# Distance estimation constants.
KNOWN_WIDTHS: Dict[str, float] = {
    "person": 50,
    "cell phone": 7,
    "laptop": 35,
    "bottle": 8,
    "cup": 8,
    "chair": 50,
    "backpack": 30,
    "car": 180,
}
FOCAL_LENGTH = 600


def estimate_distance(pixel_width: float, real_width: float) -> float:
    if pixel_width == 0:
        return 0.0
    return (real_width * FOCAL_LENGTH) / pixel_width


# ── Thread-safe live settings ────────────────────────────────────────────


class Settings:
    """Live-adjustable defense parameters (FFT threshold, detector conf)."""

    def __init__(self, fft_threshold: float, conf: float) -> None:
        self._lock = threading.Lock()
        self._fft_threshold = float(fft_threshold)
        self._conf = float(conf)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {"fft_threshold": self._fft_threshold, "conf": self._conf}

    def update(self, fft_threshold: Optional[float], conf: Optional[float]) -> Dict[str, float]:
        with self._lock:
            if fft_threshold is not None:
                self._fft_threshold = float(np.clip(fft_threshold, 0.0, 1.0))
            if conf is not None:
                self._conf = float(np.clip(conf, 0.05, 0.95))
            return {"fft_threshold": self._fft_threshold, "conf": self._conf}


settings = Settings(
    load_calibrated_threshold(CALIBRATION_PROFILE, DEFAULT_FFT_THRESHOLD, HIGH_FREQ_RATIO),
    DEFAULT_CONF,
)

# ── Detector adapter ─────────────────────────────────────────────────────


class DetectorAdapter:
    """Wraps an Ultralytics YOLO model. Tests may swap in a fake via app.state."""

    def __init__(self, model_path: str, conf: float) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.names: Dict[int, str] = getattr(self.model, "names", {}) or {}

    def predict(self, frame: np.ndarray) -> Any:
        return self.model.predict(source=frame, conf=self.conf, verbose=False)


# ── App state ────────────────────────────────────────────────────────────


class AppState:
    def __init__(self) -> None:
        self.queue: Optional[asyncio.Queue] = None
        self.worker: Optional["CameraWorker"] = None
        self.broadcaster: Optional[asyncio.Task] = None
        self.clients: Set[WebSocket] = set()
        self.camera_connected = False
        self.model_loaded = False


state = AppState()

# Legacy module globals kept so existing tests/integrations can seed frames.
latest_frame: Optional[bytes] = None
latest_data: Dict[str, Any] = {
    "status": "INITIALIZING",
    "noise_score": 0.0,
    "fft_threshold": settings.snapshot()["fft_threshold"],
    "is_attack": False,
    "detections": [],
    "flagged_count": 0,
    "systemMetrics": {
        "cpu": 0,
        "memory": 0,
        "gpu": None,
        "bandwidth": 0,
        "latency": 0,
        "uptime": "00:00:00",
    },
    "data": [],
}
latest_data_lock = threading.Lock()

# ── System metrics (real, with graceful fallbacks) ──────────────────────


def _gpu_metrics() -> Optional[Dict[str, Any]]:
    """Query nvidia-smi once; return None when no NVIDIA GPU is present."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        util, mem_used, mem_total = (float(x) for x in out.stdout.splitlines()[0].split(","))
        return {
            "utilization": int(util),
            "memory_used": int(mem_used),
            "memory_total": int(mem_total),
            "memory_pct": int(100.0 * mem_used / mem_total) if mem_total else 0,
        }
    except Exception:
        return None


def _system_metrics(
    start_time: float,
    latency_ms: float,
    prev_net: Optional[List[int]],
    net_interval: float,
) -> Dict[str, Any]:
    uptime_sec = int(time.time() - start_time)
    uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime_sec))

    next_net: Optional[List[int]] = prev_net
    try:
        import psutil  # type: ignore[import-not-found]

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        net = psutil.net_io_counters()
        cur_net = [net.bytes_sent, net.bytes_recv]
        bandwidth = 0.0
        if prev_net is not None and net_interval > 0:
            total = sum(cur_net) - sum(prev_net)
            bandwidth = total / net_interval / 1024.0  # KiB/s
        next_net = cur_net
    except ImportError:
        # No psutil: approximate with os.loadavg() + best-effort memory.
        cpu = round(float(os.getloadavg()[0]) * 10.0, 1) if hasattr(os, "getloadavg") else 0.0
        mem = round(_memory_usage_fallback(), 1)
        bandwidth = 0.0

    gpu = _gpu_metrics()
    gpu_pct = gpu["utilization"] if gpu else None
    return {
        "cpu": round(cpu, 1),
        "memory": round(mem, 1),
        "gpu": gpu_pct,
        "gpu_details": gpu,
        "bandwidth": round(bandwidth, 1),
        "latency": round(latency_ms, 1),
        "uptime": uptime_str,
        "prev_net": next_net,
    }


def _memory_usage_fallback() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            lines = dict(line.split(":", 1) for line in f)
        total = float(lines["MemTotal"].strip().split()[0])
        available = float(lines["MemAvailable"].strip().split()[0])
        return 100.0 * (1.0 - available / total) if total else 0.0
    except Exception:
        return 0.0


# ── Camera worker (producer thread) ──────────────────────────────────────


class CameraWorker:
    """Reads the camera, runs detector + FFT scoring, pushes snapshots."""

    def __init__(self, detector: DetectorAdapter) -> None:
        self.detector = detector
        self._stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.start_time = time.time()
        self.prev_net: Optional[List[int]] = None
        self.net_interval = 1.0
        self._last_net_sample = time.time()
        self.history: List[Dict[str, Any]] = []

    def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        self.thread = threading.Thread(target=self._run, name="kaal-camera-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, snapshot: Dict[str, Any]) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._publish, snapshot)

    async def _publish(self, snapshot: Dict[str, Any]) -> None:
        if state.queue is not None:
            await state.queue.put(snapshot)

    def _open_camera(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap is None or not cap.isOpened():
            cap.release()
            return None
        return cap

    def _run(self) -> None:
        global latest_frame, latest_data
        backoff = 0.5
        while not self._stop.is_set():
            cap = self._open_camera()
            if cap is None:
                state.camera_connected = False
                logger.warning("Camera %d unavailable; retrying in %.1fs", CAMERA_INDEX, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff = 0.5
            state.camera_connected = True
            logger.info("Camera %d connected", CAMERA_INDEX)
            try:
                self._camera_loop(cap)
            finally:
                cap.release()
                state.camera_connected = False

    def _camera_loop(self, cap: cv2.VideoCapture) -> None:
        global latest_frame, latest_data
        while not self._stop.is_set():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera read failed; reconnecting")
                return
            try:
                snapshot = self._process_frame(frame)
            except Exception as exc:  # keep the worker alive on any per-frame error
                logger.warning("Frame processing error: %s", exc)
                self._stop.wait(0.05)
                continue

            _, buffer = cv2.imencode(".jpg", snapshot.pop("frame"))
            latest_frame = buffer.tobytes()
            snapshot["latency_ms"] = round((time.time() - t0) * 1000.0, 1)

            with latest_data_lock:
                latest_data = snapshot
            self._emit(snapshot)

    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        frame = cv2.resize(frame, (640, 480))
        st = settings.snapshot()

        results = self.detector.predict(frame)
        out_frame = results[0].plot()
        detections: List[Dict[str, Any]] = []
        boxes: List[tuple] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.detector.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pixel_width = x2 - x1
                real_width = KNOWN_WIDTHS.get(cls_name.lower(), 50)
                distance_m = estimate_distance(pixel_width, real_width) / 100.0
                boxes.append((x1, y1, x2, y2))
                detections.append(
                    {
                        "label": cls_name,
                        "conf": round(float(box.conf[0]), 3),
                        "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                        "distance": round(distance_m, 2),
                        "anomaly_score": None,
                        "is_adversarial": False,
                    }
                )

        # Per-region FFT scoring (matches the model's dual-stream path).
        region_scores = compute_region_scores(frame, boxes, HIGH_FREQ_RATIO, REGION_MIN_SIZE)
        for det, score in zip(detections, region_scores):
            if score is not None:
                det["anomaly_score"] = round(float(score), 4)
                det["is_adversarial"] = float(score) > st["fft_threshold"]

        noise_score = compute_fft_anomaly_score(frame, HIGH_FREQ_RATIO)
        flagged = sum(1 for d in detections if d["is_adversarial"])
        is_attack = noise_score > st["fft_threshold"] or flagged > 0

        # Draw status overlay.
        color = (0, 0, 255) if is_attack else (0, 255, 0)
        status_text = "ADVERSARIAL ATTACK" if is_attack else "SYSTEM NOMINAL"
        cv2.putText(
            out_frame, f"FFT: {noise_score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        cv2.putText(out_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if is_attack:
            cv2.rectangle(out_frame, (0, 0), (640, 480), (0, 0, 255), 10)
        for det, score in zip(detections, region_scores):
            if det["is_adversarial"]:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Metrics.
        latency_ms = 0.0
        now = time.time()
        if now - self._last_net_sample >= self.net_interval:
            prev = self.prev_net
            metrics = _system_metrics(self.start_time, 0.0, prev, self.net_interval)
            self.prev_net = metrics.pop("prev_net", None)
            self._last_net_sample = now
        else:
            metrics = _system_metrics(self.start_time, latency_ms, self.prev_net, self.net_interval)
            self.prev_net = metrics.pop("prev_net", None)

        # Timeline point.
        uptime_sec = int(time.time() - self.start_time)
        self.history.append(
            {
                "time": uptime_sec,
                "value": round(float(noise_score), 4),
                "value2": round(float(noise_score) * 100.0, 2),
                "value3": round(100.0 * flagged, 1) if detections else 0.0,
            }
        )
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

        return {
            "frame": out_frame,
            "status": "ENGAGING" if is_attack or detections else "NOMINAL",
            "noise_score": round(float(noise_score), 4),
            "fft_threshold": st["fft_threshold"],
            "is_attack": bool(is_attack),
            "flagged_count": flagged,
            "detections": detections,
            "systemMetrics": metrics,
            "data": list(self.history),
        }


# ── FastAPI app ──────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(_: FastAPI):
    logger.info("Loading YOLO model: %s", MODEL_PATH)
    detector = DetectorAdapter(MODEL_PATH, settings.snapshot()["conf"])
    state.model_loaded = True
    state.queue = asyncio.Queue(maxsize=4)
    if ENABLE_CAMERA:
        state.worker = CameraWorker(detector)
        state.worker.start()
    else:
        logger.info("Camera worker disabled (KAAL_ENABLE_CAMERA=0)")
    state.broadcaster = asyncio.create_task(_broadcast_loop())
    try:
        yield
    finally:
        logger.info("Shutting down backend")
        if state.worker is not None:
            state.worker.stop()
            if state.worker.thread is not None:
                state.worker.thread.join(timeout=3.0)
        if state.broadcaster is not None:
            state.broadcaster.cancel()
        try:
            if state.broadcaster is not None:
                await state.broadcaster
        except asyncio.CancelledError:
            pass


app = FastAPI(title="KAAL Defense Backend", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _broadcast_loop() -> None:
    assert state.queue is not None
    while True:
        snapshot = await state.queue.get()
        for ws in list(state.clients):
            try:
                await ws.send_json(snapshot)
            except Exception:
                state.clients.discard(ws)


def _snapshot() -> Dict[str, Any]:
    with latest_data_lock:
        return latest_data


# ── REST endpoints ───────────────────────────────────────────────────────


@app.get("/api/python-data")
async def get_python_data() -> Dict[str, Any]:
    return _snapshot()


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    snap = _snapshot()
    return {
        "status": "ok",
        "mode": snap["status"],
        "is_attack": snap["is_attack"],
        "camera": state.camera_connected,
        "model_loaded": state.model_loaded,
        "fft_threshold": settings.snapshot()["fft_threshold"],
        "uptime": snap["systemMetrics"]["uptime"],
    }


@app.get("/api/settings")
async def get_settings() -> Dict[str, float]:
    return settings.snapshot()


@app.put("/api/settings")
async def put_settings(body: Dict[str, Any]) -> Dict[str, float]:
    fft = body.get("fft_threshold")
    conf = body.get("conf")
    return settings.update(
        float(fft) if fft is not None else None,
        float(conf) if conf is not None else None,
    )


@app.get("/api/video_feed")
async def video_feed():
    async def gen():
        while True:
            frame = latest_frame
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── WebSocket ────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state.clients.add(websocket)
    await websocket.send_json(_snapshot())
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
