#!/usr/bin/env bash
# ==============================================================================
# One-Click Launcher for Trust-Aware Smart Healthcare Monitoring System
# Starts Mosquitto MQTT broker, verifies Python venv & ONNX model, launches
# background Sensor Agents + Bedside Edge Brain, and opens the Nurse Dashboard.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SENSOR_PID=""
BRAIN_PID=""
MOSQ_PID=""

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping background healthcare agents..."
    if [[ -n "$SENSOR_PID" ]] && kill -0 "$SENSOR_PID" 2>/dev/null; then
        kill "$SENSOR_PID" 2>/dev/null || true
    fi
    if [[ -n "$BRAIN_PID" ]] && kill -0 "$BRAIN_PID" 2>/dev/null; then
        kill "$BRAIN_PID" 2>/dev/null || true
    fi
    if [[ -n "$MOSQ_PID" ]] && kill -0 "$MOSQ_PID" 2>/dev/null; then
        kill "$MOSQ_PID" 2>/dev/null || true
    fi
    echo "[OK] All background processes terminated cleanly."
}

trap cleanup EXIT INT TERM

echo "======================================================================"
echo "  Trust-Aware Adaptive Multi-Agent Smart Healthcare Demo Launcher"
echo "======================================================================"

# 1. Activate or initialize Python virtual environment
if [[ ! -d "venv" ]]; then
    echo "[1/5] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[1/5] Activating existing Python virtual environment (venv)..."
    source venv/bin/activate
fi

# 2. Ensure Mosquitto MQTT broker is running on port 1883
echo "[2/5] Checking Mosquitto MQTT broker status..."
if systemctl is-active --quiet mosquitto 2>/dev/null; then
    echo "  -> Mosquitto systemd service is active."
elif pgrep -x "mosquitto" >/dev/null 2>&1; then
    echo "  -> Mosquitto process is already running."
else
    if command -v mosquitto >/dev/null 2>&1; then
        echo "  -> Starting local Mosquitto daemon on port 1883..."
        mosquitto -p 1883 -d || {
            mosquitto -p 1883 &
            MOSQ_PID=$!
        }
        sleep 0.5
    else
        echo "[ERROR] Mosquitto broker not found! Install via: sudo apt install -y mosquitto"
        exit 1
    fi
fi

# 3. Ensure 1D-CNN ONNX model exists
if [[ ! -f "ecg_anomaly.onnx" ]]; then
    echo "[3/5] Generating 1D-CNN ECG Anomaly Detector (ecg_anomaly.onnx)..."
    python3 make_model.py
else
    echo "[3/5] Verified ecg_anomaly.onnx is present."
fi

# Reset chaos.json to clean baseline on fresh launch
cat > chaos.json << 'EOF'
{
  "Agent_ECG": {
    "delay_ms": 12,
    "drop_rate": 0.0,
    "malicious": false
  },
  "Agent_SpO2": {
    "delay_ms": 10,
    "drop_rate": 0.0,
    "malicious": false
  },
  "Agent_HR": {
    "delay_ms": 10,
    "drop_rate": 0.0,
    "malicious": false
  },
  "Orin_Link": {
    "overload_gpu": false,
    "use_relay_b": false
  }
}
EOF

# 4. Start Autonomous Sensor Agents and Bedside Edge Brain in background
echo "[4/5] Starting Multi-Agent Sensor Streamer & Bedside Edge Brain..."
python3 sensor_agents.py --primary-broker 127.0.0.1 --backup-broker 127.0.0.1 &
SENSOR_PID=$!
sleep 0.5

python3 jetson_edge_brain.py --broker 127.0.0.1 --model ecg_anomaly.onnx &
BRAIN_PID=$!
sleep 0.8

# 5. Launch Streamlit Nurse Station Live Dashboard
echo "[5/5] Launching Live Split-Screen Nurse Station Dashboard..."
echo "======================================================================"
streamlit run dashboard.py

