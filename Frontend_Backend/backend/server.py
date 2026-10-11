"""
server.py — FastAPI backend for the KAAL dashboard.

Asyncio producer-consumer pipeline:

    Camera worker (thread)
        -> capture -> detector (YOLO or RF-DETR) -> FFT region scoring
        -> temporal-consistency cue -> snapshot (dict)
        -> asyncio.Queue -> WebSocket broadcast
                         -> latest snapshot (HTTP)

Endpoints
    GET  /api/python-data   latest snapshot (legacy polled endpoint)
    GET  /api/video_feed    MJPEG stream of the annotated frame
    GET  /api/health        liveness / mode / camera / detector / temporal status
    GET  /api/settings      live FFT threshold + detector confidence
    PUT  /api/settings      update them at runtime (token-protected if KAAL_API_TOKEN)
    WS   /ws                real-time snapshot broadcast (origin-checked)

Configuration (environment variables)
    KAAL_HOST                 bind address (default 127.0.0.1 — local only)
    KAAL_PORT                 default 3000
    KAAL_CORS_ORIGINS         comma-separated dashboard origins
                              (default http://localhost:5173,http://127.0.0.1:5173)
    KAAL_API_TOKEN            if set, PUT /api/settings requires header X-KAAL-Token
    KAAL_DETECTOR             yolo | rfdetr | auto (default auto: from weights suffix)
    KAAL_MODEL_PATH           weights (.pt -> YOLO, .pth/.ckpt -> RF-DETR)
    KAAL_RFDETR_SIZE          nano | small | medium (COCO weights when no path given)
    KAAL_CAMERA_SOURCE        video file / stream URL instead of a webcam (files loop)
    KAAL_CAMERA_INDEX, KAAL_ENABLE_CAMERA, KAAL_CONF, KAAL_FFT_THRESHOLD,
    KAAL_CALIBRATION, KAAL_HIGH_FREQ_RATIO, KAAL_REGION_MIN_SIZE, KAAL_HISTORY
    KAAL_TEMPORAL             1/0 — enable the temporal cue (default 1)
    KAAL_TEMPORAL_THRESHOLD   fixed temporal threshold (default: warm-up calibrated)
    KAAL_TEMPORAL_WARMUP      warm-up frames for calibration (default 30)
    KAAL_PROCESS_SIDE         longest frame side fed to the pipeline (default 960)
"""

import asyncio
import logging
import os
import secrets
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
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.environ.get("KAAL_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Allow importing MODEL's model-free modules (fft_scorer, calibration, ...).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from MODEL.calibration import load_calibrated_threshold  # noqa: E402
from MODEL.detectors import infer_backend, load_detector  # noqa: E402
from MODEL.fft_scorer import (  # noqa: E402
    DEFAULT_FFT_THRESHOLD as MODEL_DEFAULT_FFT_THRESHOLD,
    compute_fft_anomaly_score,
    compute_region_scores,
)
from MODEL.temporal_scorer import StreamingTemporalCue  # noqa: E402

# ── Configuration (environment variables) ────────────────────────────────


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw not in ("0", "false", "False", "")


HOST = _env_str("KAAL_HOST", "127.0.0.1")
PORT = _env_int("KAAL_PORT", 3000)
CORS_ORIGINS = [
    o.strip()
    for o in _env_str("KAAL_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
API_TOKEN = _env_str("KAAL_API_TOKEN", "")
CAMERA_INDEX = _env_int("KAAL_CAMERA_INDEX", 0)
# A file path or stream URL replaces the webcam; local video files loop (demo mode).
CAMERA_SOURCE: int | str = _env_str("KAAL_CAMERA_SOURCE", "") or CAMERA_INDEX
SOURCE_IS_FILE = isinstance(CAMERA_SOURCE, str) and Path(CAMERA_SOURCE).is_file()
ENABLE_CAMERA = _env_bool("KAAL_ENABLE_CAMERA", True)
MODEL_PATH = _env_str("KAAL_MODEL_PATH", str(_REPO_ROOT / "MODEL" / "yolov8m.pt"))
DETECTOR_BACKEND = _env_str("KAAL_DETECTOR", "auto").lower()
RFDETR_SIZE = _env_str("KAAL_RFDETR_SIZE", "small")
CALIBRATION_PROFILE = _env_str("KAAL_CALIBRATION", "")
DEFAULT_CONF = _env_float("KAAL_CONF", 0.4)
DEFAULT_FFT_THRESHOLD = _env_float("KAAL_FFT_THRESHOLD", MODEL_DEFAULT_FFT_THRESHOLD)
HIGH_FREQ_RATIO = _env_float("KAAL_HIGH_FREQ_RATIO", 0.25)
HISTORY_LEN = _env_int("KAAL_HISTORY", 60)
REGION_MIN_SIZE = _env_int("KAAL_REGION_MIN_SIZE", 16)
TEMPORAL_ENABLED = _env_bool("KAAL_TEMPORAL", True)
TEMPORAL_THRESHOLD = os.environ.get("KAAL_TEMPORAL_THRESHOLD")
TEMPORAL_WARMUP = _env_int("KAAL_TEMPORAL_WARMUP", 30)
PROCESS_SIDE = _env_int("KAAL_PROCESS_SIDE", 960)  # longest side fed to the pipeline
METRICS_INTERVAL_S = 1.0  # psutil net counters + nvidia-smi are sampled at most this often

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
    "vehicle": 180,
    "tank": 360,
    "armored_vehicle": 300,
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


class SettingsUpdate(BaseModel):
    """PUT /api/settings body. Non-finite values (NaN/inf) are rejected with 422."""

    fft_threshold: Optional[float] = Field(default=None, allow_inf_nan=False)
    conf: Optional[float] = Field(default=None, allow_inf_nan=False)


settings = Settings(
    load_calibrated_threshold(CALIBRATION_PROFILE, DEFAULT_FFT_THRESHOLD, HIGH_FREQ_RATIO),
    DEFAULT_CONF,
)

# ── Detector adapter ─────────────────────────────────────────────────────


class DetectorAdapter:
    """
    Wraps a ``MODEL.detectors`` backend. Falls back to a NO-DETECTOR demo mode
    (FFT + temporal scoring, metrics and streams still active) when the ML
    stack or the weights are unavailable. Tests may swap in a fake.
    """

    def __init__(self, model_path: str, backend: str = "auto") -> None:
        self.backend = infer_backend(model_path) if backend == "auto" else backend
        self.model_path = model_path
        self.detector: Any = None
        self.is_fallback = True
        try:
            self.detector = load_detector(self.backend, model_path or None, rfdetr_size=RFDETR_SIZE)
            self.is_fallback = False
        except ImportError as exc:
            logger.warning(
                "%s stack not installed (%s) — running in NO-DETECTOR demo mode", self.backend, exc
            )
        except Exception as exc:  # missing/corrupt weights should not kill the server
            logger.error(
                "Could not load %s weights %s: %s — demo mode", self.backend, model_path, exc
            )
        self.names: Dict[int, str] = getattr(self.detector, "names", {}) or {}

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "weights": Path(self.model_path).name if self.model_path else None,
            "fallback": self.is_fallback,
        }

    def predict(self, frame: np.ndarray, conf: float) -> List[Dict[str, Any]]:
        if self.is_fallback or self.detector is None:
            return []
        return self.detector.predict(frame, conf=conf)


# ── App state ────────────────────────────────────────────────────────────


class AppState:
    def __init__(self) -> None:
        self.queue: Optional[asyncio.Queue] = None
        self.worker: Optional["CameraWorker"] = None
        self.broadcaster: Optional[asyncio.Task] = None
        self.clients: Set[WebSocket] = set()
        self.camera_connected = False
        self.model_loaded = False
        self.detector_info: Dict[str, Any] = {}


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
    "temporal": None,
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


def enqueue_latest(queue: asyncio.Queue, snapshot: Dict[str, Any]) -> None:
    """Non-blocking put that drops the oldest snapshot when the queue is full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(snapshot)


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


def _memory_usage_fallback() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            lines = dict(line.split(":", 1) for line in f)
        total = float(lines["MemTotal"].strip().split()[0])
        available = float(lines["MemAvailable"].strip().split()[0])
        return 100.0 * (1.0 - available / total) if total else 0.0
    except Exception:
        return 0.0


class MetricsSampler:
    """
    System metrics for the dashboard. CPU/memory are cheap and read per call;
    network throughput and the nvidia-smi subprocess are refreshed at most
    once per ``interval`` seconds, and bandwidth is computed over the real
    elapsed time between network samples.
    """

    def __init__(self, interval: float = METRICS_INTERVAL_S) -> None:
        self.interval = interval
        self.start_time = time.time()
        self._last_sample = 0.0
        self._prev_net: Optional[tuple] = None  # (bytes_total, timestamp)
        self._bandwidth = 0.0
        self._gpu: Optional[Dict[str, Any]] = None

    def sample(self, latency_ms: float) -> Dict[str, Any]:
        now = time.time()
        try:
            import psutil  # type: ignore[import-not-found]

            cpu = float(psutil.cpu_percent(interval=None))
            mem = float(psutil.virtual_memory().percent)
            if now - self._last_sample >= self.interval:
                net = psutil.net_io_counters()
                total = net.bytes_sent + net.bytes_recv
                if self._prev_net is not None and now > self._prev_net[1]:
                    self._bandwidth = (
                        (total - self._prev_net[0]) / (now - self._prev_net[1]) / 1024.0
                    )
                self._prev_net = (total, now)
        except ImportError:
            cpu = round(float(os.getloadavg()[0]) * 10.0, 1) if hasattr(os, "getloadavg") else 0.0
            mem = _memory_usage_fallback()

        if now - self._last_sample >= self.interval:
            self._gpu = _gpu_metrics()
            self._last_sample = now

        uptime = time.strftime("%H:%M:%S", time.gmtime(int(now - self.start_time)))
        return {
            "cpu": round(cpu, 1),
            "memory": round(mem, 1),
            "gpu": self._gpu["utilization"] if self._gpu else None,
            "gpu_details": self._gpu,
            "bandwidth": round(self._bandwidth, 1),  # KiB/s
            "latency": round(latency_ms, 1),
            "uptime": uptime,
        }


# ── Frame annotation ─────────────────────────────────────────────────────


def annotate(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    noise_score: float,
    is_attack: bool,
    temporal: Optional[Dict[str, Any]],
) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        color = (0, 0, 255) if det["is_adversarial"] else (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['label']} {det['conf']:.2f}"
        if det["anomaly_score"] is not None:
            label += f" fft={det['anomaly_score']:.3f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    color = (0, 0, 255) if is_attack else (0, 255, 0)
    cv2.putText(out, f"FFT: {noise_score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if temporal is not None:
        txt = f"TMP: {temporal['score']:.4f}" if temporal["ready"] else "TMP: calibrating"
        cv2.putText(out, txt, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    status_text = "ADVERSARIAL ATTACK" if is_attack else "SYSTEM NOMINAL"
    cv2.putText(out, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if is_attack:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
    return out


# ── Camera worker (producer thread) ──────────────────────────────────────


class CameraWorker:
    """Reads the camera, runs detector + FFT + temporal scoring, pushes snapshots."""

    def __init__(self, detector: DetectorAdapter) -> None:
        self.detector = detector
        self._stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.metrics = MetricsSampler()
        self.start_time = self.metrics.start_time
        self.history: List[Dict[str, Any]] = []
        self._last_latency_ms = 0.0
        self.temporal: Optional[StreamingTemporalCue] = (
            StreamingTemporalCue(
                threshold=float(TEMPORAL_THRESHOLD) if TEMPORAL_THRESHOLD else None,
                warmup_frames=TEMPORAL_WARMUP,
            )
            if TEMPORAL_ENABLED
            else None
        )

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.thread = threading.Thread(target=self._run, name="kaal-camera-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, snapshot: Dict[str, Any]) -> None:
        """Hand a snapshot to the event loop from the worker thread."""
        loop, queue = self._loop, state.queue
        if loop is not None and queue is not None and loop.is_running():
            loop.call_soon_threadsafe(enqueue_latest, queue, snapshot)

    def _open_camera(self):
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if cap is None or not cap.isOpened():
            cap.release()
            return None
        return cap

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            cap = self._open_camera()
            if cap is None:
                state.camera_connected = False
                logger.warning("Camera %s unavailable; retrying in %.1fs", CAMERA_SOURCE, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff = 0.5
            state.camera_connected = True
            if self.temporal is not None:
                self.temporal.reset()  # new capture session -> new background
            logger.info("Camera %s connected", CAMERA_SOURCE)
            try:
                self._camera_loop(cap)
            finally:
                cap.release()
                state.camera_connected = False

    def _camera_loop(self, cap: cv2.VideoCapture) -> None:
        global latest_frame, latest_data
        # Pace video files at their native frame rate (webcams pace themselves).
        frame_interval = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if SOURCE_IS_FILE else 0.0
        rewinds = 0
        while not self._stop.is_set():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret and SOURCE_IS_FILE and rewinds < 2:
                rewinds += 1  # end of file: loop the demo video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if self.temporal is not None:
                    self.temporal.reset()  # the jump back is a scene cut
                continue
            if not ret:
                logger.warning("Camera read failed; reconnecting")
                return
            rewinds = 0
            try:
                snapshot = self.process_frame(frame)
            except Exception as exc:  # keep the worker alive on any per-frame error
                logger.warning("Frame processing error: %s", exc)
                self._stop.wait(0.05)
                continue

            _, buffer = cv2.imencode(".jpg", snapshot.pop("frame"))
            latest_frame = buffer.tobytes()
            # Smoothed end-to-end processing time, reported with the next frame.
            elapsed = (time.time() - t0) * 1000.0
            self._last_latency_ms = 0.8 * self._last_latency_ms + 0.2 * elapsed

            with latest_data_lock:
                latest_data = snapshot
            self._emit(snapshot)
            if frame_interval:
                self._stop.wait(max(0.0, frame_interval - (time.time() - t0)))

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        # Longest side -> PROCESS_SIDE, keeping the aspect ratio (a fixed
        # 640x480 squashed non-4:3 sources and blurred small patches).
        h, w = frame.shape[:2]
        scale = PROCESS_SIDE / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(
                frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        st = settings.snapshot()

        detections: List[Dict[str, Any]] = []
        boxes: List[tuple] = []
        for det in self.detector.predict(frame, conf=st["conf"]):
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            real_width = KNOWN_WIDTHS.get(cls_name.lower(), 50)
            boxes.append((x1, y1, x2, y2))
            detections.append(
                {
                    "label": cls_name,
                    "conf": round(float(det["confidence"]), 3),
                    "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                    "distance": round(estimate_distance(x2 - x1, real_width) / 100.0, 2),
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
        temporal = self.temporal.observe(frame) if self.temporal is not None else None
        flagged = sum(1 for d in detections if d["is_adversarial"])
        temporal_flag = bool(temporal and temporal["flagged"])
        is_attack = noise_score > st["fft_threshold"] or flagged > 0 or temporal_flag

        uptime_sec = int(time.time() - self.start_time)
        self.history.append(
            {
                "time": uptime_sec,
                "value": round(float(noise_score), 4),
                "value2": round(100.0 * flagged / len(detections), 1) if detections else 0.0,
                "value3": round(float(temporal["score"]) * 100.0, 2) if temporal else 0.0,
            }
        )
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

        return {
            "frame": annotate(frame, detections, noise_score, is_attack, temporal),
            "status": "ENGAGING" if is_attack or detections else "NOMINAL",
            "noise_score": round(float(noise_score), 4),
            "fft_threshold": st["fft_threshold"],
            "is_attack": bool(is_attack),
            "flagged_count": flagged,
            "temporal": temporal,
            "detector": self.detector.info(),
            "detections": detections,
            "systemMetrics": self.metrics.sample(self._last_latency_ms),
            "data": list(self.history),
        }


# ── FastAPI app ──────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(_: FastAPI):
    backend = DETECTOR_BACKEND if DETECTOR_BACKEND in ("yolo", "rfdetr") else "auto"
    logger.info("Loading detector (%s): %s", backend, MODEL_PATH)
    detector = DetectorAdapter(MODEL_PATH, backend)
    state.model_loaded = not detector.is_fallback
    state.detector_info = detector.info()
    state.queue = asyncio.Queue(maxsize=4)
    if HOST not in ("127.0.0.1", "localhost", "::1") and not API_TOKEN:
        logger.warning(
            "Listening on %s without KAAL_API_TOKEN: anyone on the network can change "
            "defense settings. Set KAAL_API_TOKEN.",
            HOST,
        )
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
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "PUT"],
    allow_headers=["Content-Type", "X-KAAL-Token"],
)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # The default handler echoes the offending input back, which itself fails
    # to serialize for NaN/inf (-> 500). Report location + message only.
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


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


def _origin_allowed(origin: Optional[str], host: Optional[str]) -> bool:
    """Browsers always send Origin on WebSockets; non-browser clients may not."""
    if not origin:
        return True
    if "*" in CORS_ORIGINS or origin in CORS_ORIGINS:
        return True
    return host is not None and origin.split("://", 1)[-1] == host  # same-origin


# ── REST endpoints ───────────────────────────────────────────────────────


@app.get("/api/python-data")
async def get_python_data() -> Dict[str, Any]:
    return _snapshot()


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    snap = _snapshot()
    temporal = state.worker.temporal if state.worker is not None else None
    return {
        "status": "ok",
        "mode": snap["status"],
        "is_attack": snap["is_attack"],
        "camera": state.camera_connected,
        "model_loaded": state.model_loaded,
        "detector": state.detector_info,
        "fft_threshold": settings.snapshot()["fft_threshold"],
        "temporal": {
            "enabled": temporal is not None,
            "ready": bool(temporal and temporal.ready),
            "threshold": temporal.threshold if temporal is not None else None,
            "rebaselines": temporal.rebaselines if temporal is not None else 0,
        },
        "uptime": snap["systemMetrics"]["uptime"],
    }


@app.get("/api/settings")
async def get_settings() -> Dict[str, float]:
    return settings.snapshot()


@app.put("/api/settings")
async def put_settings(
    body: SettingsUpdate,
    x_kaal_token: Optional[str] = Header(default=None),
) -> Dict[str, float]:
    if API_TOKEN and not secrets.compare_digest(x_kaal_token or "", API_TOKEN):
        raise HTTPException(status_code=401, detail="invalid or missing X-KAAL-Token")
    return settings.update(body.fft_threshold, body.conf)


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
    if not _origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008)  # policy violation: cross-site socket
        return
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
