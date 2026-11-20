# Jetson Bedside Monitor

This folder runs on **each patient's bedside device** (an NVIDIA Jetson or a friend's laptop acting as a Jetson). Each bed gets its own private dashboard, its own login password, and its own simulator.

**This folder is 100% standalone.** You only need to copy the `jetson_bedside/` folder onto a Jetson or bedside laptop — you do **not** need `nurse_station/`, Node.js, Mosquitto, or Docker.

---

## Step 1 — Install Requirements (First Time Only)

Make sure **Python 3.8+** (with `pip` and `venv`) is installed:
- **Ubuntu / Debian / Jetson:**
  ```bash
  sudo apt update && sudo apt install -y python3 python3-pip python3-venv
  ```
  *(If your Jetson uses Python 3.8 specifically, also install `python3.8-venv`: `sudo apt install -y python3.8-venv python3-pip`. If you previously created a `venv/` without `pip`, remove it with `rm -rf venv` before running `./setup.sh`.)*
- **macOS / Windows:** install from [python.org](https://www.python.org/downloads/) *(on Windows, check **"Add Python to PATH"** during install)*

Then open a terminal **inside `jetson_bedside/`** and run the setup script:

- **Linux / macOS / Jetson:**
  ```bash
  ./setup.sh
  ```
- **Windows (PowerShell / Command Prompt):**
  ```powershell
  .\setup.bat
  ```

This creates a local virtual environment (reusing Jetson system CUDA/TensorRT packages when available) and downloads all required Python packages (`paho-mqtt`, `numpy`, and `onnxruntime`).

---

## Step 2 — Start the Bedside Monitor

Run from **inside `jetson_bedside/`**:

### A) Connecting to a separate Nurse Station (different Wi-Fi via Tailscale, or same Wi-Fi)

1. If you are on different Wi-Fis, install free **Tailscale** ([tailscale.com/download](https://tailscale.com/download)) and connect to the Nurse Station's Tailscale network.
2. Save the public **`ca.pem`** file sent by the Nurse Station inside this `jetson_bedside/` folder.
3. Paste the bedside commands printed by the Nurse Station's `start_network_broker` script:

- **Linux / macOS / Jetson:**
  ```bash
  export MQTT_HOST='100.x.y.z'
  export MQTT_PORT=8883
  export MQTT_USERNAME='jetson-bed-1'
  export MQTT_PASSWORD='your-bed-password'
  export MQTT_CA_FILE='./ca.pem'

  ./start_bedside.sh --patient-id Bed_1
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:MQTT_HOST = '100.x.y.z'
  $env:MQTT_PORT = '8883'
  $env:MQTT_USERNAME = 'jetson-bed-1'
  $env:MQTT_PASSWORD = 'your-bed-password'
  $env:MQTT_CA_FILE = '.\ca.pem'

  .\start_bedside.bat --patient-id Bed_1
  ```

### B) Running locally on the same computer as the Nurse Station

No environment variables needed — it connects to the local Nurse Station on port `1884` automatically:

- **Linux / macOS / Jetson:**
  ```bash
  ./start_bedside.sh --patient-id Bed_1
  ```
- **Windows (PowerShell / Command Prompt):**
  ```powershell
  .\start_bedside.bat --patient-id Bed_1
  ```

*(Change `Bed_1` to `Bed_2`, `Bed_3`, etc. for other beds. Use `--web-port 8081` if running more than one bed on the same computer. The device ID is fixed once launched.)*

---

## Step 3 — Register the Patient

1. Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in a browser **on that bedside device**.
2. Fill in:
   - **Department** (e.g., ICU, Cardiology)
   - **Room**
   - **Patient Name**
   - **Local Password** *(must be at least 8 characters)*
3. Click **Register**.

As soon as you register, the 8 virtual sensors start running and the patient automatically appears on the Nurse Station.

---

## Step 4 — Use the Bedside Simulator (Optional)

- Click **Simulation** on the monitor, or go directly to [http://127.0.0.1:8080/simulation](http://127.0.0.1:8080/simulation).
- Log in with the same bedside password you created in Step 3.
- From here you can change vital targets, unplug/disconnect individual sensors, or simulate network delay, packet loss, and heavy CPU/GPU load.
- Everything here affects **only this bed** and resets back to normal when you restart the bedside app.

---

## Everyday Operations

- **Locking the screen:** Logs out the browser UI so nobody walking by can view or touch it, while the sensors keep sending live data to the Nurse Station in the background.
- **Restarting a bed:** Registration info and the salted password hash are saved in `~/.local/state/nurse-station/<device-id>.json`. If you restart using the same `--patient-id`, OS user, and state folder, the sensors automatically resume and you just log back into the dashboard.
- **Changing where state is saved / running a clean test:** Set `BEDSIDE_STATE_DIR` to another folder before starting:
  ```bash
  BEDSIDE_STATE_DIR="$(mktemp -d)" ./start_bedside.sh --patient-id Bed_1
  ```
- **Moving a physical Jetson to a brand-new patient:**
  1. On the Nurse Station, click **Release bed** for the old patient.
  2. Stop the bedside monitor (`Ctrl + C`).
  3. Move/archive `~/.local/state/nurse-station/<device-id>.json` to a safe backup folder (or delete it for a test).
  4. Relaunch `./start_bedside.sh --patient-id <same-id>` and register the new patient.

---

## How It Works Under the Hood

- **Privacy & isolation:** The bedside web UI only listens on `127.0.0.1` (this device only — other machines on the same Wi-Fi cannot open it) and only subscribes to MQTT control topics for its own bed ID.
- **Sensors (250 Hz clock):** Eight independent sensor threads generate timestamped waveforms and vitals:
  - Cardiac traces (ECG, pleth, arterial pressure): 6-second sweep
  - Breathing traces (respiration, capnography): 12-second sweep
  - Blood pressure cuff (NIBP): updates every 15 seconds
- **Edge AI:** Runs the included `ecg_anomaly.onnx` model via ONNX Runtime, automatically choosing the fastest available accelerator: **TensorRT → CUDA → CPU**.
