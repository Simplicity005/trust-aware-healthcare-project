# Trust-Aware Smart Healthcare Monitoring System

A smart, fault-tolerant patient monitoring system for hospital ICUs. 

It monitors patient vitals (ECG, Heart Rate, Oxygen), spots emergencies with on-device AI, and automatically detects when network lag or glitchy sensors are sending bad data—rerouting alerts or muting broken sensors so nurses don't get flooded with false alarms.

---

## What Does This Project Do? (The Big Picture)

In a real hospital, medical sensors stream live data over the network to central nurse stations. But networks can lag, Wi-Fi can drop packets, and sensors can fail or glitch (sending crazy numbers like a heart rate of -40 or oxygen of 15%).

This project solves that with **three core ideas**:

1. **Instant Bedside Alarms (Edge AI):** A small bedside computer (like an NVIDIA Jetson Orin or simply a laptop CPU) runs a lightweight AI model directly at the patient's bed. If the ECG heart wave looks dangerous (arrhythmia), the bedside alarm sounds in under 5 milliseconds—without waiting for slow cloud or Wi-Fi servers.
2. **Smart Trust Engine (0% to 100% Score):** The system continuously grades each sensor. If a sensor starts lagging, dropping packets, or reporting physically impossible vitals, its **Trust Score** drops.
3. **Automatic Rerouting & Quarantine:**
   - **Trusted (Green / ≥ 70%):** Everything is healthy; primary fast path is active.
   - **Offloaded (Yellow / 45%–70% or High Lag):** Wi-Fi is congested; the system automatically reroutes alerts through a backup network path.
   - **Isolated (Red / < 45%):** Sensor is glitching or sending garbage data; it gets quarantined so it cannot trigger false clinical alarms.
4. **Live Nurse Dashboard & Chaos Slider:** A clean split-screen web dashboard showing patient vitals in real time, with interactive sliders that let you inject network lag or fake data on the fly to see how the system reacts.

---

## How It Works (Simple Architecture)

```
[ Patient Sensors ] ───▶ Sends vitals every 0.4s (ECG Wave, Oxygen, Heart Rate)
        │
        ▼
[ Bedside AI Brain ] ──▶ Runs instant AI check on heart rhythm (<5 ms)
        │                Checks if readings make sense together
        ▼
[ Nurse Dashboard ] ───▶ Calculates live Trust Scores (0 to 100%)
                         Shows live ECG graph, alerts, and active routes
                         Includes Chaos Sliders to test network failures
```

---

## Target Project Files

Everything is kept clean and simple in the project folder:

```text
trust-aware-healthcare-project/
├── README.md               # This guide
├── .gitignore              # Files to ignore (venv, temp files)
├── requirements.txt        # Python libraries needed
├── chaos.json              # Live settings for injecting lag / fake data
├── make_model.py           # Creates the lightweight ECG AI model (.onnx)
├── sensor_agents.py        # Simulates the 3 patient sensors
├── jetson_edge_brain.py    # The bedside AI computer
├── dashboard.py            # The nurse station live dashboard
├── run_benchmarks.py       # Generates charts and benchmark reports
├── start_local_demo.sh     # One-click script to start the whole demo
└── benchmarks/             # Folder where test graphs get saved
```

---

## Development Roadmap (5 Simple Phases)

### Phase 1: Basic Setup & AI Model
- [x] Create project structure and clean repository.
- [x] Write plain-English documentation and roadmap.
- [x] Create `requirements.txt` with standard libraries (`paho-mqtt`, `streamlit`, `torch`, `onnxruntime`, etc.).
- [x] Create `chaos.json` to store lag and error injection settings.
- [x] Write `make_model.py` to train and export the simple ECG heart check AI model (`ecg_anomaly.onnx`).
- [x] Verify virtual environment and test ONNX model inference.

### Phase 2: Sensor Simulators & Bedside Brain
- [x] Write `sensor_agents.py` to stream realistic vitals (ECG waveform, oxygen level 96-100%, heart rate 65-92 bpm) every 0.4 seconds.
- [x] Add broker failover so sensors reconnect automatically if a network disconnects.
- [x] Write `jetson_edge_brain.py` to run the ECG AI model and trigger bedside alarms in under 5ms.
- [x] Add sanity checks (e.g. if ECG is normal but Heart Rate says -40, flag a sensor conflict).
- [x] Validate end-to-end telemetry streaming and chaos fault-injection under live Mosquitto broker.

### Phase 3: Nurse Dashboard & Trust Scoring
- [x] Write the mathematical Trust Engine (grades sensors from 0.0 to 1.0 based on delay, jitter, dropped packets, and crazy values).
- [x] Build the interactive Streamlit dashboard (`dashboard.py`):
  - **Left side:** Bedside view (live ECG line chart, AI speed, alarm banner).
  - **Right side:** Nurse station view (sensor trust cards, status badges, routing mode).
  - **Sidebar:** Sliders to add delay, packet loss, or fake data in real time.

### Phase 4: Automated Testing & Graphs
- [x] Write `run_benchmarks.py` to automatically run tests and save clean PNG charts:
  1. AI inference speed comparison (CPU vs GPU).
  2. Alert delay with normal routing vs smart adaptive rerouting.
  3. Time taken to isolate a broken sensor (reaches quarantine in < 1.5 seconds).

### Phase 5: One-Click Startup & Hardware Mode
- [x] Create `start_local_demo.sh` to start Mosquitto, run the model generator, start background sensors, and open the dashboard in one command.
- [x] Document how to run on actual hardware (NVIDIA Jetson Orin + Laptop) over local Wi-Fi/LAN if hardware is available.

---

## Current Progress

| Component | File | Status | What it does |
| :--- | :--- | :---: | :--- |
| **Project Setup** | `.gitignore`, `README.md` | **Completed** | Clean repo and simplified guide |
| **Dependencies** | `requirements.txt` | **Completed** | Standard Python libraries + ONNX tooling |
| **Chaos Config** | `chaos.json` | **Completed** | Baseline fault-injection settings |
| **ECG AI Model** | `make_model.py` | **Completed** | Self-contained 1D-CNN ONNX anomaly detector |
| **Sensor Agents** | `sensor_agents.py` | **Completed** | Multi-agent telemetry streamer with failover |
| **Bedside Brain** | `jetson_edge_brain.py` | **Completed** | Edge AI inference (<0.5 ms) & local alarms |
| **Nurse Dashboard** | `dashboard.py` | **Completed** | Split-screen live UI & Mathematical Trust Engine |
| **Benchmarks** | `run_benchmarks.py` | **Completed** | Generates test charts (`.png`) and performance CSVs |
| **Start Script** | `start_local_demo.sh` | **Completed** | One-click demo launcher with automatic cleanup |

**Current Status:** 100% completed (All 5 Phases finished and verified).

---

## Quick-Start (How to Run Everything Locally)

You can run 100% of this project on a standard laptop (no GPU or NVIDIA hardware required):

### 1. Install Mosquitto (The message broker)
Mosquitto is a tiny, standard program that lets the sensors talk to the bedside brain and dashboard:
```bash
sudo apt update && sudo apt install -y mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

### 2. Launch the Demo
Run the one-command launcher:
```bash
chmod +x start_local_demo.sh
./start_local_demo.sh
```
This automatically:
- Creates/activates the Python virtual environment and installs requirements.
- Generates the ECG AI model (`ecg_anomaly.onnx`).
- Starts the sensors (`sensor_agents.py`) and bedside brain (`jetson_edge_brain.py`) in the background.
- Opens the live dashboard in your browser at `http://localhost:8501`.

### 3. Generate Benchmark Reports
```bash
source venv/bin/activate
python3 run_benchmarks.py
```
Test graphs (`1_inference_speed.png`, `2_alert_latency_routing.png`, `3_trust_isolation_time.png`) and CSV logs will be saved directly into `benchmarks/`.

---

## Two-Device Hardware Mode (Bedside Jetson Orin + Nurse Station Laptop)

In a real hospital ICU, the **patient sensors and the Edge AI brain sit together at the patient's bed (on the Jetson Orin)**, while the **Nurse Station Dashboard sits on the nurse's table (your Laptop)** connected over hospital Wi-Fi/LAN:

* **On the Bedside Jetson Orin (`<ORIN_IP>`):** Runs `sensor_agents.py` (patient's bedside sensors) + `jetson_edge_brain.py` (bedside GPU AI monitor) + `ecg_anomaly.onnx`.
* **On the Nurse's Table Laptop (`<LAPTOP_IP>`):** Runs **only** `mosquitto` (the MQTT broker) and `dashboard.py` (the Nurse Station UI & Trust Engine). Any Chaos Slider changes on the Laptop are broadcast over MQTT (`hospital/control/chaos`) to the Jetson automatically.

### 1. On the Bedside Jetson Orin — Copy Bedside Files & Start Sensors + Edge Brain
```bash
scp ecg_anomaly.onnx jetson_edge_brain.py sensor_agents.py chaos.json user@<ORIN_IP>:~/
ssh user@<ORIN_IP>

# Start the bedside patient sensors and bedside GPU brain (pointing to the Laptop's broker)
python3 sensor_agents.py --primary-broker <LAPTOP_IP> &
python3 jetson_edge_brain.py --broker <LAPTOP_IP>
```

### 2. On the Nurse Station Laptop — Start the Nurse Dashboard Only
```bash
source venv/bin/activate
MQTT_BROKER=<LAPTOP_IP> streamlit run dashboard.py
```
