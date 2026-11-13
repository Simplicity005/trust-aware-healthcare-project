# Trust-Aware Multi-Patient Smart Healthcare Monitoring System

A fault-tolerant, multi-patient ICU monitoring system powered by **Bedside Edge AI (NVIDIA Jetson Orin)** and a **Central Nurse Station Trust & Routing Engine**.

It monitors multiple patient beds (`Bed_1`, `Bed_2`, `Bed_3`, ...) simultaneously, detects cardiac arrhythmias in **`< 5 ms`** directly at each patient's bed, grades every sensor's **Trust Score ($T_i(t) \in [0.0, 1.0]$)** in real time, reroutes alerts over backup paths when hospital Wi-Fi lags, and automatically quarantines broken or spoofed sensors so nurses never get flooded with false alarms.

---

## Project Structure

```text
trust-aware-healthcare-project/
├── README.md                              # Complete setup & usage guide (macOS, Linux, Windows WSL, Jetson)
├── requirements.txt                       # Full Python dependency list
├── start_local_demo.sh                    # One-command local simulator (macOS/Linux)
├── start_local_demo.bat                   # One-command local simulator (Windows)
├── start_nurse_station.bat                # Windows launcher for Central Nurse Station (:8501)
├── start_bedside.bat                      # Windows launcher for Bedside Monitor (:8080)
│
├── jetson_bedside/                        # Bedside Node (runs on NVIDIA Jetson Orin or simulated locally)
│   ├── bedside_monitor.html               # Local Bedside Monitor UI (http://localhost:8080)
│   ├── jetson_edge_brain.py               # Edge 1D-CNN inference, local alarm, and HTTP server (:8080)
│   ├── sensor_agents.py                   # Bedside sensor streamers (Agent_ECG, Agent_SpO2, Agent_HR)
│   ├── ecg_anomaly.onnx                   # Pre-built 1D-CNN ECG arrhythmia model
│   ├── make_model.py                      # Script to regenerate ecg_anomaly.onnx
│   ├── chaos.json                         # Local bedside vitals & fault configuration
│   ├── requirements.txt                   # Lightweight bedside dependencies
│   └── start_bedside.sh                   # Bedside node launcher
│
└── nurse_station/                         # Central Nurse Station (runs on the Nurse's Laptop)
    ├── dashboard.html                     # Nurse Station Bento Box UI templates & CSS
    ├── dashboard.py                       # Multi-patient Trust Engine, HA failover & Streamlit app (:8501)
    ├── ecg_anomaly.onnx                   # Gateway CPU fallback model (used during Jetson GPU offload)
    ├── make_model.py                      # Script to regenerate ecg_anomaly.onnx
    ├── run_benchmarks.py                  # Automated benchmark & chart generator
    ├── start_nurse_station.sh             # Single Nurse Station launcher
    ├── start_dual_nurse_stations.sh       # High-Availability dual Nurse Station launcher (:8501 + :8502)
    ├── requirements.txt                   # Nurse Station Python dependencies
    └── benchmarks/                        # Generated 200-DPI PNG plots & CSV logs
```

---

## Quick Start on a Brand-New Laptop (No Requirements Installed)

Follow the steps for your operating system below. Once **Python 3** and **Mosquitto** are installed, the launcher script automatically creates a virtual environment and installs every Python package for you.

### Step 1: Install System Prerequisites (One-Time Setup)

#### macOS (Apple Silicon M1/M2/M3/M4 or Intel)
1. **Install Homebrew** (skip if `brew --version` already works in Terminal):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   *(After Homebrew finishes on Apple Silicon, run the two `echo` / `eval` commands it prints in your terminal so `brew` is added to your `PATH`.)*
2. **Install Python 3, Mosquitto (MQTT broker), and Git:**
   ```bash
   brew install python@3.12 mosquitto git
   ```
   *(You do **not** need to run Mosquitto as a background service — `./start_local_demo.sh` starts it automatically when needed.)*

#### Ubuntu / Debian Linux
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip mosquitto git
```

#### Windows 10 / 11 (via WSL2 Ubuntu)
1. Open PowerShell as Administrator and install Ubuntu WSL:
   ```powershell
   wsl --install
   ```
2. Reboot if prompted, open the **Ubuntu** app from the Start Menu, and run:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip mosquitto git
   ```

---

### Step 2: Download the Project

```bash
git clone https://github.com/Simplicity005/trust-aware-healthcare-project.git
cd trust-aware-healthcare-project
```

*(If you don't have `git`, download and extract it with `curl`:)*
```bash
curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz
cd trust-aware-healthcare-project-main
```

---

### Step 3: Run Everything with One Command (Local Simulation Mode)

No Jetson hardware is required. Run the one-click launcher from the project root:

**macOS / Linux / WSL:**
```bash
chmod +x start_local_demo.sh
./start_local_demo.sh
```

**Windows Native (PowerShell or Double-Click):**
Double-click `start_local_demo.bat` in File Explorer, or run in PowerShell:
```powershell
.\start_local_demo.bat
```
Or launch just the Nurse Station directly:
```powershell
streamlit run nurse_station/dashboard.py
```

What this script does automatically:
1. Creates a local Python virtual environment (`venv/`) and installs all packages from `requirements.txt` (`paho-mqtt`, `streamlit`, `pandas`, `numpy`, `onnxruntime`, `matplotlib`).
2. Checks if an MQTT broker is listening on port `1883` and starts `mosquitto` if needed.
3. Launches 3 simulated Bedside Jetson Nodes (`Bed_1`, `Bed_2`, `Bed_3`) streaming live 250 Hz ECG, SpO2, and HR telemetry.
4. Starts the Local Bedside Monitor (`:8080`) and Central Nurse Station (`:8501`).

---

### Step 4: Open Both Dashboards in Your Browser

| Dashboard | URL | What It Shows |
| :--- | :--- | :--- |
| **Central Nurse Station** | **`http://localhost:8501`** | Multi-bed ICU ward overview, room directory, dynamic bed discovery (`+ Add New Bed`), composite Trust Scores $T_i(t)$, arrhythmia alarms, compute offloading, and clinical alert log. |
| **Local Bedside Monitor** | **`http://localhost:8080`** | Bedside Jetson Orin screen: live 250 Hz ECG canvas, Heart Rate, SpO2, inference latency, and **Bedside Vitals & Fault Injection Controls** (trigger arrhythmia, spoof sensor data, inject network delay/packet drops, or overload GPU). |

#### Things to Try Once Both Pages Are Open:
1. **Admit More Beds:** In the Nurse Station (`:8501`), `Bed_1` is admitted at startup. Click **+ Add New Bed** (or look under **Unassigned Jetson Nodes** in the left sidebar) to admit `Bed_2` and `Bed_3` into ICU rooms.
2. **Manipulate Vitals / Inject Faults at the Bedside:** Open **`http://localhost:8080`**, switch beds using the top-right dropdown if desired, and click presets like **[PRESET: Ventricular Arrhythmia]**, **[PRESET: Byzantine Spoof]**, or **[PRESET: Network Latency]**. Watch the Nurse Station (`:8501`) detect the arrhythmia or drop the faulty sensor's trust score below `0.45` and quarantine it.
3. **Edit UI Layouts Live:** Edit [`jetson_bedside/bedside_monitor.html`](jetson_bedside/bedside_monitor.html) or [`nurse_station/dashboard.html`](nurse_station/dashboard.html) and refresh your browser — changes take effect immediately without restarting Python.

Press **`Ctrl + C`** in the terminal when you are done to stop all processes cleanly.

---

## Optional: Run in Separate Terminals / Test High Availability (HA)

If you want separate logs for the Bedside Node and Nurse Station, open two terminals:

**Terminal 1 — Simulated Bedside Nodes (`http://localhost:8080`):**
```bash
cd trust-aware-healthcare-project/jetson_bedside
chmod +x start_bedside.sh
./start_bedside.sh --patients Bed_1 Bed_2 Bed_3 --broker 127.0.0.1
```

**Terminal 2 — Central Nurse Station (`http://localhost:8501`):**
```bash
cd trust-aware-healthcare-project/nurse_station
chmod +x start_nurse_station.sh
./start_nurse_station.sh --broker 127.0.0.1 --port 8501
```

**Test Dual Nurse Station Failover (`:8501` Primary + `:8502` Hot Standby):**
```bash
cd trust-aware-healthcare-project/nurse_station
chmod +x start_dual_nurse_stations.sh
./start_dual_nurse_stations.sh
```
Open `http://localhost:8501` and `http://localhost:8502`. Expand **High Availability (HA) Failover Controls** in the sidebar and click **Test Failover / Switch Role** to see automatic leader promotion.

---

## Generate Benchmark Charts & CSV Reports

```bash
cd trust-aware-healthcare-project
source venv/bin/activate
python3 nurse_station/run_benchmarks.py
```
Outputs 200-DPI PNG plots and CSVs into `nurse_station/benchmarks/`.

---

## Multi-Device Hardware Deployment (Real NVIDIA Jetson Orin + Laptop)

Find your Nurse Station laptop's Wi-Fi/LAN IP (`ipconfig getifaddr en0` on macOS, or `hostname -I | awk '{print $1}'` on Linux):

### 1. On the Nurse Station Laptop (`nurse_station/`):
Allow LAN devices to connect to Mosquitto on port `1883`:
- **Linux:**
  ```bash
  sudo bash -c 'cat > /etc/mosquitto/conf.d/lan.conf << EOF
  listener 1883 0.0.0.0
  allow_anonymous true
  EOF'
  sudo systemctl restart mosquitto
  ```
- **macOS:**
  Create `/opt/homebrew/etc/mosquitto/mosquitto.conf` (or `/usr/local/etc/mosquitto/mosquitto.conf` on Intel Mac) with:
  ```text
  listener 1883 0.0.0.0
  allow_anonymous true
  ```
  Then start Mosquitto with `mosquitto -c /opt/homebrew/etc/mosquitto/mosquitto.conf -d`.

Start the Nurse Station:
```bash
cd nurse_station
./start_nurse_station.sh
```

### 2. On Each Bedside NVIDIA Jetson Orin (`jetson_bedside/`):
```bash
curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1 trust-aware-healthcare-project-main/jetson_bedside
cd jetson_bedside
chmod +x start_bedside.sh
./start_bedside.sh --patient-id Bed_1 --broker <LAPTOP_IP>
```
*(Use `--patient-id Bed_2`, `Bed_3`, etc. on additional Jetsons. Each Jetson serves its own local bedside monitor at `http://localhost:8080`.)*

---

## Troubleshooting

- **`Mosquitto MQTT broker is not installed` (macOS):** Run `brew install mosquitto`, then re-run `./start_local_demo.sh`. If Homebrew was just installed on Apple Silicon, make sure `/opt/homebrew/bin` and `/opt/homebrew/sbin` are in your `PATH`.
- **`Address already in use` on port `8080` or `8501`:** An older instance is still running. Stop it with:
  ```bash
  pkill -f jetson_edge_brain.py; pkill -f sensor_agents.py; pkill -f streamlit
  ```
- **macOS `python3` prompts for Command Line Developer Tools:** Click **Install** (or run `xcode-select --install`), or install Python via `brew install python@3.12`.
- **`onnxruntime` fails to install on Python 3.14+:** Both the Bedside Edge Brain and Nurse Station automatically fall back to pure-NumPy inference if `onnxruntime` is unavailable, or install `python@3.12` via Homebrew (`python3.12 -m venv venv`) for full ONNX Runtime support.
