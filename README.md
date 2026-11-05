# Trust-Aware Multi-Patient Smart Healthcare Monitoring System

A fault-tolerant, multi-patient ICU monitoring system powered by **Bedside Edge AI (NVIDIA Jetson Orin)** and a **Central Nurse Station Trust & Routing Engine**.

It monitors multiple patient beds (`Bed_1`, `Bed_2`, `Bed_3`, ...) simultaneously, detects cardiac arrhythmias in **`< 5 ms`** directly at each patient's bed, grades every sensor's **Trust Score ($0.0 - 1.0$)** in real time, reroutes alerts over backup paths when hospital Wi-Fi lags, and automatically quarantines broken or spoofed sensors so nurses never get flooded with false alarms.

---

## Modular Folder Structure (Separated by Hardware Role)

The repository includes a root `requirements.txt` for the full project, and is cleanly divided into **two self-contained folders** (`jetson_bedside/` and `nurse_station/`), each with its own `requirements.txt` and launcher:

```text
trust-aware-healthcare-project/
├── README.md                        # Master project guide
├── requirements.txt                 # Full project Python dependencies
├── start_local_demo.sh              # One-click local simulator (runs 3 patient beds + nurse dashboard on 1 laptop)
│
├── jetson_bedside/                  # 🛏️ CODE THAT RUNS ON EACH BEDSIDE JETSON ORIN (PATIENT'S BED)
│   ├── README.md                    # Jetson setup & multi-bed launch instructions
│   ├── requirements.txt             # Lightweight Edge AI & MQTT dependencies (paho-mqtt, numpy, onnxruntime)
│   ├── chaos.json                   # Bedside fault config (auto-synced via MQTT from Nurse Station)
│   ├── ecg_anomaly.onnx             # Pre-built 1D-CNN ECG anomaly detector model
│   ├── make_model.py                # Re-generates ecg_anomaly.onnx if needed
│   ├── sensor_agents.py             # Patient bedside sensors (Agent_ECG, Agent_SpO2, Agent_HR)
│   ├── jetson_edge_brain.py         # Bedside GPU AI (<5 ms arrhythmia inference & local alarm)
│   └── start_bedside.sh             # One-command Jetson launcher (--patient-id Bed_1 --broker <LAPTOP_IP>)
│
└── nurse_station/                   # 🖥️ CODE THAT RUNS ON THE NURSE'S TABLE LAPTOP
    ├── README.md                    # Nurse laptop setup & LAN broker instructions
    ├── requirements.txt             # Dashboard, Trust Engine & benchmark dependencies
    ├── chaos.json                   # Per-patient fault injection settings
    ├── ecg_anomaly.onnx             # Pre-built 1D-CNN ECG model for Gateway CPU fallback
    ├── make_model.py                # Re-generates ecg_anomaly.onnx if needed
    ├── dashboard.py                 # Multi-Patient Nurse Station Dashboard & Trust Engine
    ├── run_benchmarks.py            # Automated benchmark & plot generator
    ├── start_nurse_station.sh       # One-command Nurse Station launcher
    └── benchmarks/                  # Generated 200-DPI PNG charts & CSV reports
```

---

## How It Works (Multi-Jetson ICU Architecture)

```text
┌─────────────────────────────────────────────────────────┐
│ BEDSIDE JETSON #1 (jetson_bedside/ — Patient Bed_1)     │
│  [sensor_agents.py] ──▶ [jetson_edge_brain.py (<5ms)]   │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────┐
│ BEDSIDE JETSON #2 (jetson_bedside/ — Patient Bed_2)     │
│  [sensor_agents.py] ──▶ [jetson_edge_brain.py (<5ms)]   │
└───────────────────────────┼─────────────────────────────┘
                            │  Hospital Wi-Fi / MQTT (hospital/<Bed_ID>/...)
                            ▼
┌─────────────────────────────────────────────────────────┐
│ CENTRAL NURSE STATION LAPTOP (nurse_station/)           │
│  • Multi-Patient Ward Overview (Bed_1, Bed_2, Bed_3...) │
│  • Mathematical Trust Engine T_i(t) in [0.0, 1.0]       │
│  • Dynamic Offloading (Gateway CPU + Relay Path B)      │
│  • Byzantine Sensor Quarantine (< 1.5s isolation)       │
│  • Per-Patient Live Chaos Sliders (synced over MQTT)    │
└─────────────────────────────────────────────────────────┘
```

---

## How to Download & Install Requirements

You can either download the **entire project** (for local testing or development) or download **only the specific folder** needed for a Jetson Orin or Nurse Station Laptop:

### Option A: Download the Entire Project & Install `requirements.txt`

```bash
git clone https://github.com/Simplicity005/trust-aware-healthcare-project.git
cd trust-aware-healthcare-project

# Create virtual environment and install all requirements
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*(Or without `git`:)*
```bash
curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz
cd trust-aware-healthcare-project-main
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

---

### Option B: Download Only the Folder You Need (Device-Specific)

* **On a Bedside Jetson Orin (downloads ONLY `jetson_bedside/` & installs its `requirements.txt`):**
  ```bash
  curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1 trust-aware-healthcare-project-main/jetson_bedside
  cd jetson_bedside
  chmod +x start_bedside.sh

  # Optional manual venv setup (start_bedside.sh also does this automatically):
  python3 -m venv --system-site-packages venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

* **On the Nurse's Table Laptop (downloads ONLY `nurse_station/` & installs its `requirements.txt`):**
  ```bash
  curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1 trust-aware-healthcare-project-main/nurse_station
  cd nurse_station
  chmod +x start_nurse_station.sh

  # Optional manual venv setup (start_nurse_station.sh also does this automatically):
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

---

## Mode 1: Run the Full Multi-Patient Demo Locally on 1 Laptop

You can simulate **3 Bedside Jetsons (`Bed_1`, `Bed_2`, `Bed_3`) + the Central Nurse Station** on a single laptop:

```bash
chmod +x start_local_demo.sh
./start_local_demo.sh
```
Open `http://localhost:8501` in your browser:
- View all 3 patient beds (`Bed_1`, `Bed_2`, `Bed_3`) side by side in the **ICU Ward Multi-Patient Overview**.
- Click any patient bed to inspect its live 250 Hz ECG wave and sensor trust cards.
- Use the **Sidebar Chaos Engine** to pick a target patient bed (e.g. `Bed_2`) and inject lag, packet drops, or spoofed vitals while `Bed_1` and `Bed_3` remain healthy!

### Generate Benchmark Charts & CSV Logs
```bash
source venv/bin/activate
python3 nurse_station/run_benchmarks.py
```
Charts and CSVs are saved to `nurse_station/benchmarks/`.

---

## Mode 2: Multi-Device Hardware Deployment (Real Jetsons + Nurse Laptop)

Assume your Nurse Station Laptop's Wi-Fi/LAN IP is `<LAPTOP_IP>` (check with `hostname -I | awk '{print $1}'`):

### 1. On the Nurse's Table Laptop (`nurse_station/`):
Enable LAN connections on Mosquitto once, then start the Nurse Station:
```bash
sudo bash -c 'cat > /etc/mosquitto/conf.d/lan.conf << EOF
listener 1883 0.0.0.0
allow_anonymous true
EOF'
sudo systemctl restart mosquitto

cd nurse_station
./start_nurse_station.sh
```

### 2. On Each Bedside NVIDIA Jetson Orin (`jetson_bedside/`):
Run `start_bedside.sh` with a unique `--patient-id` on each Jetson:
```bash
# On Jetson Orin #1 (Patient Bed 1):
cd jetson_bedside
./start_bedside.sh --patient-id Bed_1 --broker <LAPTOP_IP>

# On Jetson Orin #2 (Patient Bed 2):
cd jetson_bedside
./start_bedside.sh --patient-id Bed_2 --broker <LAPTOP_IP>

# On Jetson Orin #3 (Patient Bed 3):
cd jetson_bedside
./start_bedside.sh --patient-id Bed_3 --broker <LAPTOP_IP>
```
Each Jetson automatically streams its patient's sensors and `<5 ms` GPU AI summaries into the Nurse Station Dashboard, and receives per-bed Chaos Slider commands over MQTT!
