# KAAL: Autonomous Swarm & Adversarial Defense Framework
**KAAL is a hierarchical multi-agent framework for sustainable and autonomous drone swarms combining flocking instincts with adversarial threat detection.**

[![CI — lint, typecheck, tests](https://github.com/aryank2212/KAAL/actions/workflows/ci.yml/badge.svg)](https://github.com/aryank2212/KAAL/actions/workflows/ci.yml)

---

## 🛑 The Problem

Current unmanned aerial vehicle (UAV) and drone systems face two major, critically fatal vulnerabilities in field deployments:

### 1. The Visual Deception Vulnerability (Adversarial Attacks)
Machine vision systems, including state-of-the-art military object detection models, are highly susceptible to adversarial patch attacks. A simple printed sticker placed on a target (like a tank or an enemy asset) can completely fool the drone's advanced camera capabilities. The drone camera sees the compromised target, processes it, and falsely identifies it (e.g., misclassifying a tank as a harmless vehicle or a civilian object). Lab testing on military-grade targets has proven that a cost-effective printed patch or geometric sticker can defeat million-dollar guidance systems.

### 2. The Centralized Signal Vulnerability (GPS Jamming)
Swarm operations heavily depend on GPS and centralized command hubs. During warfare or high-risk active missions, enemies deploy generic signal jammers—often costing as little as $200. The second the GPS and radio links are severed, the entire drone swarm goes blind, loses formation, and fails the mission, often crashing.

*Both attacks cost the enemy pennies. Both wipe out multimillion-dollar missions.*

---

## 💡 The Idea to Solve

We need a system where **drones don't rely on central commands and don't trust their own singular vision blindly**. 

Our idea is to implement a dual-layer strategy:
1. **The Anti-Deception Eye (Adversarial Defense)**: Equip each drone with a "second brain" that acts as an anomaly detector. It will identify if the visual input has been tampered with or contains unnatural adversarial noise that is invisible to bare human eyes.
2. **GPS-Free Swarm Brain (MARL)**: Enable the drones to coordinate locally like a flock of birds (murmurations), completely independent of GPS or central communication servers. If 30% of the swarm is taken out, the rest naturally reorganizes and completes the mission.

---

## 🛠️ Tech Stack

### Core AI & Computer Vision
- **YOLOv8 (Ultralytics)**: Core object detection model for identifying targets (e.g., tanks, aircraft, military vehicles)
- **PyTorch**: Deep learning framework for training and fine-tuning YOLO and running MARL pipelines
- **OpenCV**: Computer vision library for handling video frames, FFT conversion, and image manipulation
- **FFT Analysis**: Frequency domain transformation for detecting the unnatural statistical signatures of adversarial patches

### Swarm Intelligence & Simulation
- **MAPPO (Multi-Agent Proximal Policy Optimization)**: Reinforcement learning architecture for decentralized execution and centralized training of drone agents
- **PyBullet / Gazebo**: Realistic 3D physics simulators for replicating swarm aerodynamics and sensor inputs (IMU, distance, Lidar)

### Dashboard & Control Center
- **React.js & Vite**: Lightning-fast, modern frontend framework for building the real-time visualization dashboard
- **Python Backend**: Fast API / Flask handlers to orchestrate local hardware communication and feed YOLO inferences to the dashboard

### Hardware Specs (Physical Prototyping)
- Raspberry Pi Zero 2W (On-board inference)
- Standard F450 Quadcopter Frames
- Pixhawk Flight Controllers (Matek F405)
- Basic onboard camera & IMU modules

---

## 🚀 Our Solution: KAAL Architecture

The system is defined by its two primary operational layers:

### Layer 1: The Anti-Deception Eye (Adversarial Robustness)
Every drone runs two parallel vision systems on each camera frame it captures:

* **Stream 1 - Normal Vision**: The image is passed through the standard YOLOv8 model ("I see a tank at position X").
* **Stream 2 - Frequency Analysis**: The exact same image frame is converted to the frequency domain using Fast Fourier Transform (FFT). While adversarial patches look normal in RGB space, they cast massive, unnatural frequency signatures in FFT space. 

**The Fusion Layer:**
* If both streams agree that the target is natural, the detection is trusted.
* If the FFT stream detects anomalies (meaning the object has adversarial noise), the system flags an **Adversarial Attack** and gracefully switches to alternative detection methods (thermal/IR sensors) while ignoring the tampered visual inference.

### Layer 2: GPS-Free Swarm Brain
Each drone in the system is defined as an independent autonomous agent using MAPPO. A drone operates strictly on localized knowledge:
* Its own speed and direction via IMU.
* Sightline within 50 meters (Camera + Lidar).
* Distance and bearing of its closest 5 neighboring drones.

The drones follow **3 Fundamental Rules**:
1. **Wait and Avoid (Separation):** Prevent collision by maintaining a minimum safe distance from neighbors.
2. **Align (Alignment):** Match the average heading or spatial direction of neighboring drones.
3. **Move Together (Cohesion):** Steer towards the average physical position of the local flock.

Because of this, if a jammer disconnects them from the base, the swarm logic takes over. No single point of failure exists; the swarm self-heals and continues tracking the target.

---

## 🌍 Real World Impact

* **Defense & Military**: Securing drone swarms over contested borders where GPS jamming is an active, daily threat. Ensuring automated targeting failsafes.
* **Civilian Disaster Response**: Swarms dynamically sweeping through earthquake rubble without needing satellite connectivity beneath concrete.
* **Smart Cities & Agriculture**: Remote crop monitoring and urban grid monitoring without relying on massive bandwidth or permanent internet infrastructure.
---

## 🧪 Testing & CI

The repo runs three automated checks on every push (GitHub Actions, see `.github/workflows/ci.yml`):

1. **Lint** — `ruff check MODEL tests`
2. **Type check** — `mypy MODEL tests`
3. **Unit + integration tests** — `pytest tests/` (74 passed; 1 GPU-marked test auto-skipped without torch)

The integration suite exercises the full defense path: `DualStreamDefense.predict()` on clean vs patched images (asserts detection rate > 80% with a calibration-derived threshold) and the FastAPI server JSON schema.

```bash
pip install -r requirements.txt ruff mypy pytest
ruff check MODEL tests
mypy MODEL tests
pytest tests/            # skip GPU tests via -m "not gpu"
```

---

## 🛠️ Installation & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/aryank2212/KAAL.git
cd KAAL
```

### 2. Backend Setup (Python API & YOLO Inference)
The backend uses FastAPI and runs the YOLOv8 model for inference.
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install the required dependencies
pip install -r MODEL/requirements.txt
pip install -r requirements-server.txt  # FastAPI + uvicorn + psutil + websockets

# Run the backend server (config via KAAL_* env vars; e.g. KAAL_CAMERA_INDEX, KAAL_FFT_THRESHOLD)
python Frontend_Backend/backend/server.py
```
*The backend server will start on `http://localhost:3000` with `/api/health`, `/api/python-data`, `/api/settings`, and a WebSocket stream at `/ws`.*

### 3. Frontend Setup (React Dashboard)
The frontend dashboard visualizes the swarm status and adversarial alerts. Open a **new terminal window** and run:
```bash
cd Frontend_Backend/dashboard-app

# Install Node.js dependencies
npm install

# Start the Vite development server
npm run dev
```
*The frontend will typically run on `http://localhost:5173`. Open this URL in your browser to view the KAAL Control Dashboard!*

---
*“KAAL is a swarm intelligence system where the drones share the instincts of a flock of birds and the paranoia of a soldier — they need no signal to coordinate, and they trust nothing their cameras show them.”*
