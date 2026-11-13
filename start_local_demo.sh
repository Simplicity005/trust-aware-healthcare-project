#!/usr/bin/env bash
# ==============================================================================
# One-Click Multi-Patient Local Demo Launcher
# Simulates 3 Bedside Jetson Units (Bed_1, Bed_2, Bed_3) from jetson_bedside/
# and launches the Central Nurse Station Dashboard from nurse_station/.
# ==============================================================================

set -e
export PATH="$PATH:/opt/homebrew/sbin:/opt/homebrew/bin:/usr/local/sbin:/usr/local/bin"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BEDSIDE_PID=""
MOSQ_PID=""

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping simulated bedside Jetson nodes..."
    if [[ -n "$BEDSIDE_PID" ]] && kill -0 "$BEDSIDE_PID" 2>/dev/null; then
        kill "$BEDSIDE_PID" 2>/dev/null || true
    fi
    if [[ -n "$MOSQ_PID" ]] && kill -0 "$MOSQ_PID" 2>/dev/null; then
        kill "$MOSQ_PID" 2>/dev/null || true
    fi
    echo "[OK] All background processes terminated cleanly."
}

trap cleanup EXIT INT TERM

echo "======================================================================"
echo "  Trust-Aware Multi-Patient ICU Healthcare Demo Launcher (3 Beds)"
echo "======================================================================"

# 0. Ensure root requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    cat > requirements.txt << 'EOF'
paho-mqtt>=1.6.1
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
onnxruntime>=1.14.0
matplotlib>=3.6.0
EOF
fi

# 1. Activate or create shared Python virtual environment
if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
    echo "[1/5] Activating Python virtual environment (venv)..."
    source venv/bin/activate
else
    echo "[1/5] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
fi

# Verify all dependencies are installed in venv
if ! python3 -c "import paho.mqtt.client, streamlit, pandas, numpy, matplotlib" >/dev/null 2>&1; then
    echo "  -> Installing missing packages from requirements.txt..."
    pip install "paho-mqtt>=1.6.1" "streamlit>=1.28.0" "pandas>=1.5.0" "numpy>=1.21.0" "matplotlib>=3.6.0"
    pip install "onnxruntime>=1.14.0" || echo "  -> Note: onnxruntime wheel not available for this Python version; using NumPy fallback."
elif ! python3 -c "import onnxruntime" >/dev/null 2>&1; then
    pip install "onnxruntime>=1.14.0" >/dev/null 2>&1 || true
fi

# 2. Ensure Mosquitto MQTT broker is running on port 1883
echo "[2/5] Checking Mosquitto MQTT broker..."
if python3 -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', 1883)); s.close()" >/dev/null 2>&1; then
    echo "  -> MQTT broker is already listening on 127.0.0.1:1883."
elif command -v mosquitto >/dev/null 2>&1; then
    echo "  -> Starting local Mosquitto daemon on port 1883..."
    mosquitto -p 1883 -d 2>/dev/null || {
        mosquitto -p 1883 >/dev/null 2>&1 &
        MOSQ_PID=$!
    }
    sleep 0.6
else
    echo ""
    echo "[ERROR] Mosquitto MQTT broker is not installed on this machine."
    echo "  Install it once, then re-run ./start_local_demo.sh:"
    echo "    - macOS (Homebrew):  brew install mosquitto"
    echo "    - Ubuntu / Debian:   sudo apt update && sudo apt install -y mosquitto"
    echo ""
    exit 1
fi

# 3. Ensure 1D-CNN ONNX model exists in both folders
echo "[3/5] Verifying 1D-CNN ECG ONNX models..."
if [[ ! -f "jetson_bedside/ecg_anomaly.onnx" ]] && python3 -c "import torch, onnx" >/dev/null 2>&1; then
    python3 jetson_bedside/make_model.py
fi
if [[ -f "jetson_bedside/ecg_anomaly.onnx" ]] && [[ ! -f "nurse_station/ecg_anomaly.onnx" ]]; then
    cp jetson_bedside/ecg_anomaly.onnx nurse_station/ecg_anomaly.onnx
fi

# Reset chaos.json to clean 3-bed baseline
cat > nurse_station/chaos.json << 'EOF'
{
  "Bed_1": {
    "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": false},
    "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": false},
    "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": false},
    "Orin_Link": {"overload_gpu": false, "use_relay_b": false}
  },
  "Bed_2": {
    "Agent_ECG": {"delay_ms": 14, "drop_rate": 0.0, "malicious": false},
    "Agent_SpO2": {"delay_ms": 12, "drop_rate": 0.0, "malicious": false},
    "Agent_HR": {"delay_ms": 11, "drop_rate": 0.0, "malicious": false},
    "Orin_Link": {"overload_gpu": false, "use_relay_b": false}
  },
  "Bed_3": {
    "Agent_ECG": {"delay_ms": 10, "drop_rate": 0.0, "malicious": false},
    "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": false},
    "Agent_HR": {"delay_ms": 12, "drop_rate": 0.0, "malicious": false},
    "Orin_Link": {"overload_gpu": false, "use_relay_b": false}
  }
}
EOF
cp nurse_station/chaos.json jetson_bedside/chaos.json

# 4. Start 3 Simulated Bedside Jetsons (Bed_1, Bed_2, Bed_3)
echo "[4/5] Starting 3 Simulated Bedside Jetson Nodes (Bed_1, Bed_2, Bed_3)..."
chmod +x jetson_bedside/start_bedside.sh nurse_station/start_nurse_station.sh
./jetson_bedside/start_bedside.sh --patients Bed_1 Bed_2 Bed_3 --broker 127.0.0.1 &
BEDSIDE_PID=$!
sleep 1.2

# 5. Launch Multi-Patient Nurse Station Dashboard
echo "[5/5] Launching Multi-Patient Nurse Station Dashboard..."
echo "  -> Nurse Station Dashboard: http://localhost:8501"
echo "  -> Local Bedside Monitor:   http://localhost:8080"
echo "======================================================================"
./nurse_station/start_nurse_station.sh
