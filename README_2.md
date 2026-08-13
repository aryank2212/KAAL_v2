# KAAL — Kill-chain Aware Autonomous Lattice

A dual-layer defense framework for autonomous drone swarms that solves two critical vulnerabilities: **adversarial patch attacks** on drone vision and **GPS jamming** of swarm coordination.

---

## What It Does

### Layer 1 — Anti-Deception Eye
Every drone runs two parallel vision streams on each camera frame:

- **Stream 1 (YOLO):** Standard object detection — "I see a tank at position X."
- **Stream 2 (FFT):** Frequency-domain anomaly scan — adversarial patches leave unnatural high-frequency signatures invisible to the human eye but detectable via 2D FFT.

If the FFT stream flags an anomaly, the detection is marked unreliable and the system falls back to alternative sensors.

### Layer 2 — GPS-Free Swarm Brain
Drones coordinate using only local information (IMU, camera, nearest 5 neighbors) via three flocking rules — separation, alignment, cohesion. No GPS or central server needed. If 30% of the swarm is destroyed, the rest self-reorganizes.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Adversarial Defense | 2D FFT anomaly scoring, PGD/FGSM attacks for training |
| Swarm Intelligence | MAPPO (Multi-Agent PPO) |
| Simulation | AirSim + Unreal Engine |
| Backend | FastAPI + Python (real-time YOLO + FFT inference) |
| Frontend | React + Vite + Recharts (tactical dashboard) |
| Hardware | Raspberry Pi Zero 2W, F450 frames, Pixhawk controllers |

---

## Project Structure

```
KAAL/
├── MODEL/                  # ML pipeline
│   ├── config.py           # Centralized hyperparameters
│   ├── adversarial_patch.py# Patch generation & injection
│   ├── dual_stream_defense.py # FFT + YOLO fusion
│   ├── trainer.py          # Adversarial fine-tuning
│   ├── evaluator.py        # 3-pass robustness evaluation
│   ├── pipeline.py         # Full train → eval CLI
│   ├── attacks/            # FGSM, PGD implementations
│   ├── defenses/           # JPEG compression, bit-depth, blur
│   └── kaal_eye_v2.pt      # Trained model weights
├── Frontend_Backend/
│   ├── backend/server.py   # FastAPI server (video + data endpoints)
│   └── dashboard-app/      # React tactical dashboard
├── SCRIPTS/                # AirSim drone control & live demos
└── Docs/                   # Reference images
```

---

## Quick Start

```bash
# Backend
pip install -r MODEL/requirements.txt
pip install fastapi uvicorn
cd Frontend_Backend/backend && python server.py

# Frontend (new terminal)
cd Frontend_Backend/dashboard-app
npm install && npm run dev
```

Backend runs on `localhost:3000`, dashboard on `localhost:5173`.

---

## How the Defense Works

```
Camera Frame
    ├──→ [YOLO Stream]  →  Detections (class, bbox, confidence)
    │                              ↓
    └──→ [FFT Stream]   →  Anomaly Score (high-freq energy ratio)
                                   ↓
                          ┌────────────────────┐
                          │   Fusion Decision   │
                          │                     │
                          │ score ≤ threshold → TRUST detections
                          │ score > threshold → FLAG as adversarial
                          └────────────────────┘
```

---

## Evaluation Output

The pipeline evaluates robustness across three scenarios:

| Scenario | Description |
|----------|-------------|
| **Clean** | mAP on unmodified images (baseline) |
| **Adversarial** | mAP after patch injection (no defense) |
| **Defended** | mAP with dual-stream defense active |

Typical result: attack drops mAP by ~0.37, defense recovers ~0.27.
