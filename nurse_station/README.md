# Central Nurse Station (`nurse_station/`) - Nurse's Table Laptop Hi

This folder contains **everything that runs on the Nurse's Table Laptop** (including its own `requirements.txt` and pre-built `ecg_anomaly.onnx`):

1. `requirements.txt` - Nurse Station dependencies (`paho-mqtt`, `streamlit`, `pandas`, `numpy`, `onnxruntime`, `matplotlib`).
2. `ecg_anomaly.onnx` - Pre-built 1D-CNN ECG model for Gateway CPU fallback inference.
3. `dashboard.py` - **Multi-Patient Central Nurse Station Dashboard & Mathematical Trust Engine**. Monitors multiple bedside Jetson Orin units (`Bed_1`, `Bed_2`, `Bed_3`, ...) simultaneously, grades sensor trust $T_i(t) \in [0.0, 1.0]$, handles Gateway CPU fallback offloading, quarantines broken/malicious sensors, and provides a per-patient Chaos Controller over MQTT.
4. `run_benchmarks.py` & `benchmarks/` - Automated benchmark suite and generated PNG charts + CSV logs.
5. `start_nurse_station.sh` - One-command launcher that automatically sets up `venv`, installs `requirements.txt`, starts Mosquitto, and opens the dashboard.

---

## Download Only This Folder & Install Requirements on the Nurse's Laptop

Run this in the Nurse's Laptop terminal to download `nurse_station/` (which includes `requirements.txt` and `ecg_anomaly.onnx`):

```bash
curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1 trust-aware-healthcare-project-main/nurse_station
cd nurse_station
chmod +x start_nurse_station.sh

# Optional manual install (start_nurse_station.sh also runs this automatically):
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run on the Nurse's Table Laptop

### 1. Ensure Mosquitto Broker Listens on LAN (for Physical Jetsons)
If physical Jetson Orin boards are connecting over Wi-Fi/LAN to this Laptop, make sure Mosquitto allows LAN connections:
```bash
sudo bash -c 'cat > /etc/mosquitto/conf.d/lan.conf << EOF
listener 1883 0.0.0.0
allow_anonymous true
EOF'
sudo systemctl restart mosquitto
```

### 2. Launch the Nurse Station Dashboard

#### Option A: Single Nurse Station (Primary Leader on Port 8501)
```bash
./start_nurse_station.sh
```
Open `http://localhost:8501` in your browser.

#### Option B: Dual Nurse Stations (High-Availability Redundant Pair)
Run both stations simultaneously for fault tolerance. If Nurse Station 1 fails or crashes, Nurse Station 2 automatically assumes primary leadership:
```bash
chmod +x start_dual_nurse_stations.sh start_secondary_nurse_station.sh
./start_dual_nurse_stations.sh
```
- **Nurse Station 1 (Primary Leader):** `http://localhost:8501`
- **Nurse Station 2 (Hot-Standby Peer):** `http://localhost:8502`
- Both stations coordinate over MQTT via `hospital/nurse_stations/heartbeat` and `hospital/nurse_stations/failover`. Use the **`Test Failover`** button in the UI to demonstrate instant leader failover!

### 3. Room-Based Device Directory (Left Sidebar)
The Nurse Station left sidebar organizes all available and admitted devices by ICU Room:
- **Room Groups**: e.g., `Room 101 (Cardiac ICU)`, `Room 102 (Cardiac ICU)`, `Room 103 (Neuro ICU)`, etc.
- **Device Status Badges**: Shows each device's real-time state (`[TRUSTED]`, `[ARRHYTHMIA ALARM]`, `[QUARANTINED]`, `[OFFLOADED]`, `[OFFLINE]`).
- **Room Filter**: Filter the ward view to monitor a specific room or all rooms.
- **Available / Unassigned Jetsons**: Lists newly detected devices waiting to be commissioned, with a room assignment dropdown.
- **Direct Inspect Button**: Clicking any bed in the sidebar instantly loads its live 250 Hz ECG waveform and sensor trust metrics into the detailed inspection view.

### 4. Bedside Patient Vitals & Sensor Manipulation (Hosted on the Jetson)
In compliance with edge healthcare architectures, **patient vitals manipulation is hosted directly on the Bedside Jetson Orin** (`http://<JETSON_IP>:8080`), rather than simulated on the central nurse laptop:
- Accessible at `http://localhost:8080` (or the Jetson's screen).
- Doctors/technicians at the bedside can set Target HR, Target SpO2, and ECG rhythm (Normal Sinus vs. Arrhythmia vs. Byzantine Malicious Spoofing).
- Changes sync immediately to streaming sensor agents and the Nurse Station Trust Engine over MQTT!

### 5. Dynamic Bed Admission & Jetson Auto-Detection ("Add New Bed")
The Nurse Station features a dynamic bed admission and auto-discovery workflow:
1. **"+ Add New Bed" Action Button**: Located at the top of the ICU Ward overview. If unassigned Jetson Orin nodes are transmitting on the LAN or MQTT broker, the button displays a live badge: `+ Add New Bed (N Detected)`.
2. **Auto-Detection**: Clicking the button scans the network and lists all newly discovered Jetson devices ready to pair with live signal strength, vitals preview, and hardware execution provider (e.g. TensorRT/CUDA).
3. **Room Assignment & Admission**: Choose the room (e.g. `Room 101`, `Room 102`, `Room 103`) and click `+ Admit Jetson (Bed_X)` to officially admit the device to the ward.
4. **Bed Release / Discharge**: The nurse can release any bed back to the unassigned pool using the `Release Bed` button.
5. **Built-In Simulator**: If physical Jetson boards are not currently running, the panel includes a `+ Simulate New Jetson Node` button to test the complete discovery and admission flow locally.
