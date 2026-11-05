# Central Nurse Station (`nurse_station/`) — Nurse's Table Laptop

This folder contains **everything that runs on the Nurse's Table Laptop** (including its own `requirements.txt` and pre-built `ecg_anomaly.onnx`):

1. `requirements.txt` — Nurse Station dependencies (`paho-mqtt`, `streamlit`, `pandas`, `numpy`, `onnxruntime`, `matplotlib`).
2. `ecg_anomaly.onnx` — Pre-built 1D-CNN ECG model for Gateway CPU fallback inference.
3. `dashboard.py` — **Multi-Patient Central Nurse Station Dashboard & Mathematical Trust Engine**. Monitors multiple bedside Jetson Orin units (`Bed_1`, `Bed_2`, `Bed_3`, ...) simultaneously, grades sensor trust $T_i(t) \in [0.0, 1.0]$, handles Gateway CPU fallback offloading, quarantines broken/malicious sensors, and provides a per-patient Chaos Controller over MQTT.
4. `run_benchmarks.py` & `benchmarks/` — Automated benchmark suite and generated PNG charts + CSV logs.
5. `start_nurse_station.sh` — One-command launcher that automatically sets up `venv`, installs `requirements.txt`, starts Mosquitto, and opens the dashboard.

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
```bash
./start_nurse_station.sh
```
Open `http://localhost:8501` in your browser. As each Jetson Orin starts (`./start_bedside.sh --patient-id Bed_1 --broker <LAPTOP_IP>`), its patient bed streams live into the dashboard!
