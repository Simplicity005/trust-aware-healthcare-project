#!/usr/bin/env bash
# ==============================================================================
# Nurse Station Laptop Launcher (Runs on the Nurse's Table Laptop)
# Starts Mosquitto MQTT Broker and opens the Multi-Patient Nurse Dashboard.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${MQTT_BROKER:-127.0.0.1}"

if [[ -d "../venv" ]]; then
    source ../venv/bin/activate
elif [[ -d "venv" ]]; then
    source venv/bin/activate
else
    echo "[1/3] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

if [[ ! -f "ecg_anomaly.onnx" ]]; then
    echo "[2/3] Generating Gateway CPU fallback ECG model (ecg_anomaly.onnx)..."
    python3 make_model.py
fi

if ! systemctl is-active --quiet mosquitto 2>/dev/null && ! pgrep -x "mosquitto" >/dev/null 2>&1; then
    if command -v mosquitto >/dev/null 2>&1; then
        echo "Starting Mosquitto MQTT broker on port 1883..."
        mosquitto -p 1883 -d || true
    fi
fi

echo "[3/3] Launching Multi-Patient Nurse Station Dashboard..."
MQTT_BROKER="$BROKER" streamlit run dashboard.py
