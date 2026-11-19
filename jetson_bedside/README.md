# Jetson Bedside Monitor

This folder runs on **each patient's bedside device** (a Jetson or a local test computer). Each bed gets its own private dashboard, its own login password, and its own simulator.

> **Using separate machines (different Wi-Fis via Tailscale, or real Jetsons)?**
> Before starting, run the 1-command Mosquitto broker on the Nurse Station computer and use the login + `ca.pem` from [Option 2: Run on Separate Machines](../README.md#option-2-run-on-separate-machines). If you are running everything on one computer, you can skip this — it connects to the local Nurse Station on port `1884` automatically.

---

## Step 1 — Start the Bedside Monitor

Always run this command from the **main (root) project folder**, not inside `jetson_bedside/`:

- **Linux / macOS / Jetson:**
  ```bash
  ./start_bedside.sh --patient-id Bed_1
  ```
- **Windows (PowerShell / Command Prompt):**
  ```powershell
  .\start_bedside.bat --patient-id Bed_1
  ```

*(Change `Bed_1` to `Bed_2`, `Bed_3`, etc. for other beds. The device ID is fixed once launched.)*

---

## Step 2 — Register the Patient

1. Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in a browser **on that bedside device**.
2. Fill in:
   - **Department** (e.g., ICU, Cardiology)
   - **Room**
   - **Patient Name**
   - **Local Password** *(must be at least 8 characters)*
3. Click **Register**.

As soon as you register, the 8 virtual sensors start running and the patient automatically appears on the Nurse Station.

---

## Step 3 — Use the Bedside Simulator (Optional)

- Click **Simulation** on the monitor, or go directly to [http://127.0.0.1:8080/simulation](http://127.0.0.1:8080/simulation).
- Log in with the same bedside password you created in Step 2.
- From here you can change vital targets, unplug/disconnect individual sensors, or simulate network delay, packet loss, and heavy CPU/GPU load.
- Everything here affects **only this bed** and resets back to normal when you restart the bedside app.

---

## Everyday Operations

- **Locking the screen:** Locks/logs out the browser UI so nobody walking by can view or touch it, while the sensors keep sending live data to the Nurse Station in the background.
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
- **Edge AI:** Runs the included ONNX ECG anomaly model via ONNX Runtime, automatically choosing the fastest available accelerator: **TensorRT → CUDA → CPU**. *(On a Jetson, a new Python virtual environment can reuse system-installed packages; keep the full root repo checkout so the launcher and shared Mosquitto config stay available.)*
