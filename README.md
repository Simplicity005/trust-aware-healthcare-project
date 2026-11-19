# Trust-Aware Multi-Patient Smart Healthcare Monitoring System

A fault-tolerant, multi-patient ICU monitoring system powered by **Bedside Edge AI (NVIDIA Jetson Orin)** and a **Central Nurse Station Trust & Routing Engine**.

It monitors multiple patient beds (`Bed_1`, `Bed_2`, `Bed_3`, ...) simultaneously, detects cardiac arrhythmias in **`< 5 ms`** directly at each patient's bed, grades every sensor's **Trust Score ($T_i(t) \in [0.0, 1.0]$)** in real time, reroutes alerts over backup paths when hospital Wi-Fi lags, and automatically quarantines broken or spoofed sensors so nurses never get flooded with false alarms.

---

## Project Structure

```text
trust-aware-healthcare-project/
├── README.md                              # Complete setup & usage guide (macOS, Linux, Windows WSL, Jetson)
├── requirements.txt                       # Full Python dependency list
├── start_local_demo.sh / .bat             # Full demo with three simulated beds
├── start_bedside.sh / .bat                # Bedside sensors and monitor (:8080)
├── start_nurse_station.sh / .bat          # Nurse Station dashboard and API (:8501)
│
├── jetson_bedside/                        # Bedside Node (runs on NVIDIA Jetson Orin or simulated locally)
│   ├── bedside_monitor.html               # Local Bedside Monitor UI (http://localhost:8080)
│   ├── jetson_edge_brain.py               # Edge 1D-CNN inference, local alarm, and HTTP server (:8080)
│   ├── sensor_agents.py                   # Bedside sensor streamers (ECG, SpO2, HR, EtCO2, IBP, NIBP)
│   ├── ecg_anomaly.onnx                   # Pre-built 1D-CNN ECG arrhythmia model
│   ├── make_model.py                      # Script to regenerate ecg_anomaly.onnx
│   ├── chaos.json                         # Local bedside vitals & fault configuration
│   └── requirements.txt                   # Lightweight bedside dependencies
│
└── nurse_station/                         # Central Nurse Station (runs on the Nurse's Laptop)
    ├── web/                               # Next.js nurse dashboard (:8501)
    ├── api.py                             # FastAPI snapshots, live events, and station actions
    ├── engine.py                          # Multi-patient trust engine, MQTT, and HA failover
    ├── launch.py                          # Shared Python launcher for all three modes
    ├── ecg_anomaly.onnx                   # Gateway CPU fallback model (used during Jetson GPU offload)
    ├── run_benchmarks.py                  # Automated benchmark & chart generator
    ├── requirements.txt                   # Nurse Station Python dependencies
    └── benchmarks/                        # Generated 200-DPI PNG plots & CSV logs
```

---

## Quick Start on a New Laptop

Install **Python 3** and **Mosquitto**. The Total Local Demo and Nurse Station also need **Node.js 20.9 or newer with npm**. The launchers set up Python dependencies on first use. The Nurse Station also installs and builds the Next.js app.

### Step 1: Install System Prerequisites (One-Time Setup)

#### macOS (Apple Silicon M1/M2/M3/M4 or Intel)
1. **Install Homebrew** (skip if `brew --version` already works in Terminal):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   *(After Homebrew finishes on Apple Silicon, run the two `echo` / `eval` commands it prints in your terminal so `brew` is added to your `PATH`.)*
2. **Install Python 3, Node.js, Mosquitto, and Git:**
   ```bash
   brew install python@3.12 node mosquitto git
   ```
   *(You do **not** need to run Mosquitto as a background service — `./start_local_demo.sh` starts it automatically when needed.)*

#### Ubuntu / Debian Linux
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip mosquitto git
```

#### Windows

For the batch launchers, install Python and Mosquitto and make their commands available on `PATH`. Install Node.js and npm for the Nurse Station or Total Local Demo. On WSL, follow the Ubuntu setup and use the shell launchers.

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

### Step 3: Choose a Launcher

Run from the repository root. On Windows, double-click the batch file or run it from PowerShell.

| Mode | Linux, macOS, or WSL | Windows PowerShell | Opens at |
| :--- | :--- | :--- | :--- |
| Total Local Demo | `./start_local_demo.sh` | `.\start_local_demo.bat` | Bedside at `localhost:8080` and Nurse Station at `localhost:8501` |
| Bedside | `./start_bedside.sh` | `.\start_bedside.bat` | `localhost:8080` |
| Nurse Station | `./start_nurse_station.sh` | `.\start_nurse_station.bat` | `localhost:8501` |

The Total Local Demo starts three simulated beds and both dashboards. Bedside starts the sensors and monitor for one bed by default. Nurse Station starts the Next.js dashboard and FastAPI, and can provide simulated data when bedside feeds are absent. Each launcher can use a running MQTT broker or start local Mosquitto when needed.

---

### Step 4: Open Both Dashboards in Your Browser

| Dashboard | URL | What It Shows |
| :--- | :--- | :--- |
| **Central Nurse Station** | **`http://localhost:8501`** | Multi-bed ICU ward overview, room directory, dynamic bed discovery (`+ Add New Bed`), composite Trust Scores $T_i(t)$, arrhythmia alarms, compute offloading, and clinical alert log. |
| **Local Bedside Monitor** | **`http://localhost:8080`** | Bedside Jetson Orin screen: live 250 Hz ECG canvas, Heart Rate, SpO2, inference latency, and **Bedside Vitals & Fault Injection Controls** (trigger arrhythmia, spoof sensor data, inject network delay/packet drops, or overload GPU). |

#### Things to Try Once Both Pages Are Open:
1. **Admit More Beds:** The three demo beds are admitted at startup. Click **+ Add new bed** to discover another bedside device or simulate an additional bed. Use **Release bed** on a bed dashboard to take that bed out of active monitoring.
2. **Manipulate Vitals / Inject Faults at the Bedside:** Open **`http://localhost:8080`**, switch beds using the top-right dropdown if desired, and click presets like **[PRESET: Ventricular Arrhythmia]**, **[PRESET: Byzantine Spoof]**, or **[PRESET: Network Latency]**. Watch the Nurse Station (`:8501`) detect the arrhythmia or drop the faulty sensor's trust score below `0.45` and quarantine it.
3. **Edit UI Layouts Live:** Run the [Nurse Station frontend in development mode](nurse_station/README.md#run-the-web-and-api-separately) to enable live reload for [`nurse_station/web/app/`](nurse_station/web/app/). Edit [`jetson_bedside/bedside_monitor.html`](jetson_bedside/bedside_monitor.html) and refresh the bedside page to see changes.

Press **`Ctrl + C`** in the terminal when you are done to stop all processes cleanly.

---

## Optional: Test High Availability

Run `./start_nurse_station.sh --dual` or `.\start_nurse_station.bat --dual` to start the primary on port `8501` and standby on `8502`. Open **High availability** in the primary sidebar and click **Test failover / switch role** once the peer appears.

---

## Generate Benchmark Charts & CSV Reports

```bash
cd trust-aware-healthcare-project
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 nurse_station/run_benchmarks.py
```
Outputs 200-DPI PNG plots and CSVs into `nurse_station/benchmarks/`.

---

## Multi-Device Hardware Deployment (Real NVIDIA Jetson Orin + Laptop)

Find your Nurse Station laptop's Wi-Fi/LAN IP (`ipconfig getifaddr en0` on macOS, or `hostname -I | awk '{print $1}'` on Linux):

### 1. On the Nurse Station Laptop
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

Run the Nurse Station launcher from the repository root.

### 2. On Each Bedside NVIDIA Jetson Orin

Run `./start_bedside.sh --patient-id Bed_1 --broker 192.168.1.50` from the repository root. Replace the IP address with the Nurse Station's LAN address and choose a different bed ID for each Jetson. The [Bedside guide](jetson_bedside/README.md) covers the files needed for a minimal deployment.

---

## Troubleshooting

- **`Mosquitto MQTT broker is not installed` (macOS):** Run `brew install mosquitto`, then re-run `./start_local_demo.sh`. If Homebrew was just installed on Apple Silicon, make sure `/opt/homebrew/bin` and `/opt/homebrew/sbin` are in your `PATH`.
- **`Address already in use`:** Stop the earlier launcher with Ctrl+C in its terminal, or select different ports with `--port`, `--api-port`, and `--bedside-port`.
- **macOS `python3` prompts for Command Line Developer Tools:** Click **Install** (or run `xcode-select --install`), or install Python via `brew install python@3.12`.
- **`onnxruntime` fails to install on Python 3.14+:** The demo can start without ONNX Runtime. To use the ONNX models on a platform without a wheel for your Python version, create the virtual environment with a supported Python version.
