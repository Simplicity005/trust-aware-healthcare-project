# Central Nurse Station

This folder runs the **Central Nurse Station** — the main screen where nurses monitor every registered bed at once, see sensor trust scores and alerts, and manage station failover.

Under the hood, it runs two local services on the station computer:
- **Web Dashboard (Next.js):** `http://127.0.0.1:8501`
- **Backend API & Engine (FastAPI + Python):** `http://127.0.0.1:9501` (talks to beds through Mosquitto)

> **Using separate machines (different Wi-Fis via Tailscale, or real Jetsons)?**
> Start the private Mosquitto broker first using [Option 2: Run on Separate Machines](../README.md#option-2-run-on-separate-machines) and paste the Nurse Station commands it prints. If you are running locally on one computer, the launcher automatically starts Mosquitto on port `1884` (using native Mosquitto or Docker).

---

## Step 1 — Start the Nurse Station

Run this from the **main (root) project folder**, not inside `nurse_station/`:

- **Linux / macOS / WSL:**
  ```bash
  ./start_nurse_station.sh
  ```
- **Windows (PowerShell / Command Prompt):**
  ```powershell
  .\start_nurse_station.bat
  ```

---

## Step 2 — Open & Read the Dashboard

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser on the station computer.

- **Starts empty:** A new station starts with **0 patients**. As soon as someone registers a bed on a Jetson, it appears automatically. Beds re-announce themselves periodically, so starting the Nurse Station *after* the beds also works.
- **Room Directory (left sidebar):** Click a department to expand its rooms. Box colors mean:
  - **Grey:** Empty / vacant bed
  - **Green:** Patient is stable
  - **Amber:** Sensor issue or connection problem
  - **Red:** Emergency / critical alert
- **View a patient:** Click any occupied bed number to see the patient report, live vitals, ECG waveform, and bed activity log.
- **Offline & unplugged sensors:** If a bed stops sending data for **8 seconds**, it is marked offline. If a sensor is disconnected at the bedside, the dashboard leaves it blank/alerted instead of making up fake values.
- **Release & Readmit:**
  - Click **Release bed** to remove a patient from this station.
  - Click **Readmit bed** (which appears for released beds that are still registered) to bring them back.
  - *(Registering new patients and running sensor simulations are always done on the bedside monitor itself.)*

---

## Optional: Test High Availability (Primary + Standby Failover)

Want to test automatic backup failover between two Nurse Stations? Run from the **root project folder**:

- **Linux / macOS / WSL:**
  ```bash
  ./start_nurse_station.sh --dual
  ```
- **Windows:**
  ```powershell
  .\start_nurse_station.bat --dual
  ```

This starts two stations that exchange heartbeats over Mosquitto:
- **Primary Station:** [http://127.0.0.1:8501](http://127.0.0.1:8501)
- **Standby Station:** [http://127.0.0.1:8502](http://127.0.0.1:8502)

Once the standby peer shows up, expand **High availability** in the sidebar and click **Test failover / switch role**.

---

## Developer Mode (Run Backend & Frontend Separately)

Use this only if you are editing the code and want live auto-reload. Make sure Mosquitto is already running first.

1. **Terminal 1 — Start the FastAPI backend** (from inside `nurse_station/` with your Python venv active):
   ```bash
   python -m pip install -r requirements.txt
   python -m uvicorn api:app --host 127.0.0.1 --port 9501
   ```
2. **Terminal 2 — Start the Next.js web UI** (from inside `nurse_station/web/`):
   ```bash
   npm ci
   npm run dev -- --hostname 127.0.0.1 --port 8501
   ```

*(Optional: Set `NURSE_API_URL` in the frontend terminal if your FastAPI backend is running on a different address. The browser talks to the Next.js origin for live events and station actions.)*

---

## Benchmarks & Rebuilding the ECG AI Model

- **Run performance benchmarks:** From inside `nurse_station/` (with `requirements.txt` installed), run:
  ```bash
  python run_benchmarks.py
  ```
  This saves CSV reports and 200-DPI charts in `nurse_station/benchmarks/`.
- **Regenerate the ECG ONNX model:** Run `python jetson_bedside/make_model.py`, then copy the generated `ecg_anomaly.onnx` file into `nurse_station/`.
