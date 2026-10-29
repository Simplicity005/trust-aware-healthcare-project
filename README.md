# Trust-Aware Adaptive Multi-Agent Edge Intelligence Framework for Smart Healthcare

An edge-intelligence, fault-tolerant clinical telemetry testbed designed for resilient hospital ICU monitoring. The system combines autonomous multi-agent sensor telemetry over MQTT, real-time edge deep learning inference (1D-CNN for arrhythmia detection), zero-latency bedside alerting, and a mathematical trust evaluation engine capable of dynamic compute offloading and Byzantine fault isolation.

---

## 1. System Overview & Architecture

The architecture models a two-tier edge computing topology: **Bedside Edge Intelligence** (NVIDIA Jetson Orin or local CPU testbed) and a **Central Nurse Station Gateway** (Linux Laptop).

```
 ┌──────────────────────────────────────────────────────────────┐
 │                  Autonomous Sensor Agents                    │
 │    [Agent_ECG (250 Hz)]    [Agent_SpO2]    [Agent_HR]        │
 └──────────────────────────────┬───────────────────────────────┘
                                │ MQTT: hospital/sensors/<agent_id>
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │         Bedside Edge Brain (NVIDIA Jetson Orin / CPU)        │
 │  • 1D-CNN Arrhythmia Inference (TensorRT / ONNX Runtime)     │
 │  • Sub-5ms Local Bedside Emergency Alarms                    │
 │  • Cross-Sensor Physiological Plausibility Validation        │
 └──────────────────────────────┬───────────────────────────────┘
                                │ MQTT: hospital/edge/summary
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │       Central Nurse Station, Trust Engine & Dashboard        │
 │  • Mathematical Multi-Metric Trust Scoring T_i(t) ∈ [0.0, 1.0]│
 │  • Dynamic State Machine & Traffic Enforcement:               │
 │      - TRUSTED (T_i ≥ 0.70)  -> Primary Orin GPU Path        │
 │      - OFFLOADED (0.45 ≤ T_i < 0.70 or Latency > 180ms)      │
 │          -> Gateway CPU Fallback & Secondary Relay B         │
 │      - ISOLATED (T_i < 0.45) -> Byzantine Quarantine         │
 │  • Real-Time Dual-View Streamlit Clinical Interface          │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │            Live Fault Injection (Chaos Engine)               │
 │  • Real-time mutation of network delay, jitter, packet loss, │
 │    and spoofed/malicious physiological telemetry             │
 └──────────────────────────────────────────────────────────────┘
```

---

## 2. Target Project Directory Structure

```text
trust-aware-healthcare-project/
├── .gitignore               # Git ignore patterns for venv, caches, binaries
├── README.md                # Comprehensive project documentation & status
├── requirements.txt         # Core Python dependencies
├── chaos.json               # Live fault-injection configuration
├── make_model.py            # PyTorch 1D-CNN generator exporting ecg_anomaly.onnx
├── sensor_agents.py         # Autonomous sensor streamer with broker failover
├── jetson_edge_brain.py     # Bedside GPU Hero Agent & local ICU monitor
├── dashboard.py             # Dual-view Streamlit dashboard & Trust Engine
├── run_benchmarks.py        # Automated benchmarking suite (CSV + PNG plots)
├── start_local_demo.sh      # One-click startup script for local demonstration
└── benchmarks/              # Benchmark output directory (created by benchmarks)
    ├── 1_inference_speed.png
    ├── 2_alert_latency_routing.png
    └── 3_trust_isolation_time.png
```

---

## 3. Development Phases & Roadmap

The project is structured into **5 distinct development phases**:

### Phase 1: Environment Setup & Foundational Model Architecture
- [x] Configure repository directory structure and clean `.gitignore`.
- [x] Author technical documentation and project roadmap.
- [ ] Define `requirements.txt` with locked dependencies (`paho-mqtt>=2.0.0`, `streamlit>=1.32.0`, `torch>=2.1.0`, `onnxruntime>=1.17.0`, `pandas`, `numpy`, `matplotlib`).
- [ ] Initialize baseline `chaos.json` fault-injection schema.
- [ ] Implement `make_model.py` to construct a 1D-CNN in PyTorch (Conv1D + BatchNorm + MaxPool + AdaptiveAvgPool) accepting shape `(1, 1, 250)` and exporting to `ecg_anomaly.onnx`.

### Phase 2: Multi-Agent Sensor System & Bedside Edge Brain
- [ ] Implement `sensor_agents.py`:
  - Three threaded autonomous agents: `Agent_ECG`, `Agent_SpO2`, `Agent_HR`.
  - Realistic physiological synthesis: 250 Hz PQRST synthetic waveform, normal SpO2 (96–100%), resting HR (65–92 bpm).
  - Dynamic `chaos.json` watcher for real-time packet loss, latency/jitter injection, and spoofed vitals.
  - Automatic MQTT broker failover mechanism (`--primary-broker` to `--backup-broker`).
- [ ] Implement `jetson_edge_brain.py`:
  - ONNX Runtime execution session with fallback chain: `TensorrtExecutionProvider` -> `CUDAExecutionProvider` -> `CPUExecutionProvider`.
  - Sub-5ms inference latency tracking on ECG packets.
  - Immediate local bedside alarm flag if arrhythmia probability $> 0.70$.
  - Multi-sensor physiological cross-checking (e.g., normal ECG + SpO2 vs. impossible HR).
  - Telemetry aggregator publishing unified summaries to `hospital/edge/summary`.

### Phase 3: Central Trust Engine & Dual-View Streamlit Dashboard
- [ ] Implement Mathematical Trust Engine in `dashboard.py`:
  - Sliding-window ($N = 20$) latency score: $S_L = \max(0.0, 1.0 - \frac{\bar{L}_i}{400.0})$.
  - Jitter score: $S_J = \max(0.0, 1.0 - \frac{\max(L_i) - \min(L_i)}{200.0})$.
  - Packet loss score: $S_P = \max(0.0, 1.0 - 2.5 \times \text{loss\_rate}_i)$.
  - Plausibility score ($S_D$) with asymmetric EWMA decay on clinical bounds violations.
  - Composite Trust Score: $T_i(t) = 0.25 S_L + 0.15 S_J + 0.25 S_P + 0.35 S_D$.
  - Tri-state dynamic routing: **TRUSTED**, **OFFLOADED**, **ISOLATED**.
- [ ] Implement Dual-View Streamlit UI:
  - **Left View (Bedside Monitor)**: Live ECG line chart, execution provider status, inference speed, local alarm banner.
  - **Right View (Central Nurse Station)**: Per-agent trust cards, historical trust charts, active routing indicator.
  - **Sidebar (Chaos Controller)**: Real-time sliders mutating `chaos.json`.

### Phase 4: Automated Benchmarking & Evaluation Suite
- [ ] Implement `run_benchmarks.py`:
  - **Benchmark 1**: Batch inference speed across 200 iterations (CPU vs. TensorRT/GPU emulation).
  - **Benchmark 2**: End-to-end alert latency under link congestion (static routing vs. trust-aware adaptive rerouting).
  - **Benchmark 3**: Node isolation response time ($T_i < 0.45$ in $<1.5$ seconds) upon fault injection.
  - Auto-generate publication-ready figures and CSV logs in `benchmarks/`.

### Phase 5: Automation, Orchestration & Hardware Deployment
- [ ] Implement `start_local_demo.sh` for one-command local execution (venv setup, Mosquitto daemon check, model generation, background agent startup, Streamlit launch, clean SIGINT teardown).
- [ ] Hardware testbed documentation and deployment procedures for NVIDIA Jetson Orin (FP16 TensorRT compilation via `trtexec`, GPU telemetry monitoring with `jtop`).

---

## 4. Current Implementation Status

| Component | Target File | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Project Setup** | `.gitignore`, `README.md` | **Completed** | Clean workspace initialized without legacy files |
| **Dependencies** | `requirements.txt` | *Pending* | Specifications defined in Section 5 |
| **Chaos Configuration** | `chaos.json` | *Pending* | Schema defined; ready for generation |
| **ECG 1D-CNN Model** | `make_model.py` | *Pending* | Architecture designed; ready for export |
| **Sensor Agents** | `sensor_agents.py` | *Pending* | Multi-threaded design ready |
| **Bedside Brain** | `jetson_edge_brain.py` | *Pending* | TensorRT/ONNX fallback designed |
| **Clinical Dashboard** | `dashboard.py` | *Pending* | UI layout & mathematical trust engine defined |
| **Benchmarks** | `run_benchmarks.py` | *Pending* | Benchmark specs & plotting routines planned |
| **Startup Script** | `start_local_demo.sh` | *Pending* | Process lifecycle & signal traps planned |

**Overall Project Completion:** ~15% (Phase 0 & Architecture completed; ready for Phase 1 code generation).

---

## 5. Quick-Start Guide (Local Mode)

Once files are generated:

### 1. Prerequisites
Ensure Python 3.10+ and Mosquitto MQTT broker are installed on your system:
```bash
sudo apt update && sudo apt install -y mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

### 2. Single-Command Launch
```bash
chmod +x start_local_demo.sh
./start_local_demo.sh
```
This script will:
1. Create and activate a Python virtual environment (`venv`).
2. Install all dependencies from `requirements.txt`.
3. Check and ensure the local `mosquitto` broker is active.
4. Export the 1D-CNN anomaly model to `ecg_anomaly.onnx` if absent.
5. Launch `sensor_agents.py` and `jetson_edge_brain.py` in the background.
6. Open the Streamlit Dashboard at `http://localhost:8501`.

### 3. Run Benchmarks
To execute the automated evaluation and generate figures:
```bash
source venv/bin/activate
python3 run_benchmarks.py
```
Generated graphs and metrics will be saved in `benchmarks/`.
