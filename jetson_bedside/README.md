# Bedside Node (`jetson_bedside/`) — NVIDIA Jetson Orin

This folder contains **everything that runs at the patient's bed** on an **NVIDIA Jetson Orin** (or simulated bedside node):

1. `sensor_agents.py` — Streams live patient vitals (`Agent_ECG`, `Agent_SpO2`, `Agent_HR`) every `0.4s` for `--patient-id <Bed_ID>`.
2. `jetson_edge_brain.py` — Runs the `<5 ms` 1D-CNN ECG Arrhythmia AI model (`TensorRT / CUDA / CPU`), triggers instant bedside alarms, checks cross-sensor physiological plausibility, and streams verified summaries to the Nurse Station Laptop.
3. `make_model.py` — Generates the self-contained `ecg_anomaly.onnx` model.
4. `start_bedside.sh` — One-command launcher for a Jetson Orin.

---

## Download Only This Folder onto a Jetson Orin

Run this single command on the Jetson Orin terminal to download **only** `jetson_bedside/`:

```bash
curl -L https://github.com/Simplicity005/trust-aware-healthcare-project/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1 trust-aware-healthcare-project-main/jetson_bedside
cd jetson_bedside
chmod +x start_bedside.sh
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

### Simulate Multiple Beds from One Machine:
```bash
./start_bedside.sh --patients Bed_1 Bed_2 Bed_3 --broker <LAPTOP_IP>
```
