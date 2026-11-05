z#!/usr/bin/env bash
# ==============================================================================
# One-Click Multi-Patient Local Demo Launcher
# Simulates 3 Bedside Jetson Units (Bed_1, Bed_2, Bed_3) from jetson_bedside/
# and launches the Central Nurse Station Dashboard from nurse_station/.
# ==============================================================================

set -e

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

# 1. Activate or create shared Python virtual environment
if [[ ! -d "venv" ]]; then
    echo "[1/5] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[1/5] Activating Python virtual environment (venv)..."
    source venv/bin/activate
fi

# 2. Ensure Mosquitto MQTT broker is running on port 1883
echo "[2/5] Checking Mosquitto MQTT broker..."
if systemctl is-active --quiet mosquitto 2>/dev/null; then
    echo "  -> Mosquitto systemd service is active."
elif pgrep -x "mosquitto" >/dev/null 2>&1; then
    echo "  -> Mosquitto process is running."
else
    echo "  -> Starting local Mosquitto daemon on port 1883..."
    mosquitto -p 1883 -d || {
        mosquitto -p 1883 &
        MOSQ_PID=$!
    }
    sleep 0.5
fi

# 3. Ensure 1D-CNN ONNX model exists in both folders
echo "[3/5] Verifying 1D-CNN ECG ONNX models..."
if [[ ! -f "jetson_bedside/ecg_anomaly.onnx" ]]; then
    python3 jetson_bedside/make_model.py
fi
if [[ ! -f "nurse_station/ecg_anomaly.onnx" ]]; then
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
echo "======================================================================"
./nurse_station/start_nurse_station.sh
