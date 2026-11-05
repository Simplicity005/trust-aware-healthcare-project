# Bedside Node (`jetson_bedside/`) — NVIDIA Jetson Orin

This folder contains **everything that runs at the patient's bed** on an **NVIDIA Jetson Orin** (including its own `requirements.txt` and pre-built `ecg_anomaly.onnx`):

1. `requirements.txt` — Lightweight Jetson dependencies (`paho-mqtt`, `numpy`, `onnxruntime`).
2. `ecg_anomaly.onnx` — Pre-built 1D-CNN ECG Arrhythmia detector model (ready to use without needing PyTorch).
3. `sensor_agents.py` — Streams live patient vitals (`Agent_ECG`, `Agent_SpO2`, `Agent_HR`) every `0.4s` for `--patient-id <Bed_ID>`.
4. `jetson_edge_brain.py` — Runs the `<5 ms` 1D-CNN ECG Arrhythmia AI model (`TensorRT / CUDA / CPU`), triggers instant bedside alarms, checks cross-sensor physiological plausibility, and streams verified summaries to the Nurse Station Laptop.
5. `start_bedside.sh` — One-command launcher that automatically sets up `venv`, installs `requirements.txt`, and launches the bedside node.

---

## Download Only This Folder & Install Requirements on a Jetson Orin

Run this in the Jetson Orin terminal to download `jetson_bedside/` (which includes `requirements.txt` and `ecg_anomaly.onnx`):

```bash
curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1 trust-aware-healthcare-project-main/jetson_bedside
cd jetson_bedside
chmod +x start_bedside.sh

# Optional manual install (start_bedside.sh also runs this automatically):
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run on Each Jetson Orin

Assume the Nurse Station Laptop's IP address on Wi-Fi/LAN is `<LAPTOP_IP>`:

### On Jetson #1 (Patient Bed 1):
```bash
./start_bedside.sh --patient-id Bed_1 --broker <LAPTOP_IP>
```

### On Jetson #2 (Patient Bed 2):
```bash
./start_bedside.sh --patient-id Bed_2 --broker <LAPTOP_IP>
```

### On Jetson #3 (Patient Bed 3):
```bash
./start_bedside.sh --patient-id Bed_3 --broker <LAPTOP_IP>
```
