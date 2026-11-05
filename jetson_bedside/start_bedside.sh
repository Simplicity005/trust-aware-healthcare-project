#!/usr/bin/env bash
# ==============================================================================
# Bedside Node Launcher (Runs on an NVIDIA Jetson Orin or Simulated Bedside Unit)
# Automatically ensures requirements.txt exists, sets up venv, verifies all
# required packages are installed, and starts bedside sensors + Edge Brain.
#
# Usage:
#   ./start_bedside.sh --patient-id Bed_1 --broker <NURSE_LAPTOP_IP>
#   ./start_bedside.sh --patients Bed_1 Bed_2 Bed_3 --broker 127.0.0.1
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="127.0.0.1"
PATIENT_ID="Bed_1"
PATIENTS_LIST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --broker)
            BROKER="$2"
            shift 2
            ;;
        --patient-id)
            PATIENT_ID="$2"
            shift 2
            ;;
        --patients)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                PATIENTS_LIST="$PATIENTS_LIST $1"
                shift
            done
            ;;
        *)
            shift
            ;;
    esac
done

SENSOR_PID=""

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping bedside sensor agents..."
    if [[ -n "$SENSOR_PID" ]] && kill -0 "$SENSOR_PID" 2>/dev/null; then
        kill "$SENSOR_PID" 2>/dev/null || true
    fi
    echo "[OK] Bedside node stopped."
}

trap cleanup EXIT INT TERM

# 1. Ensure requirements.txt exists in this folder
if [[ ! -f "requirements.txt" ]]; then
    echo "[INFO] Generating requirements.txt in $SCRIPT_DIR..."
    cat > requirements.txt << 'EOF'
paho-mqtt>=1.6.1
numpy>=1.21.0
onnxruntime>=1.14.0
EOF
fi

# 2. Activate or create virtual environment
if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -d "../venv" ]] && [[ -f "../venv/bin/activate" ]]; then
    source ../venv/bin/activate
else
    echo "[1/3] Creating Python virtual environment (venv)..."
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    pip install --upgrade pip
fi

# 3. Always verify required packages are installed in the active venv
if ! python3 -c "import paho.mqtt.client, numpy, onnxruntime" >/dev/null 2>&1; then
    echo "[1/3] Installing missing dependencies from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "[1/3] Verified Python dependencies (paho-mqtt, numpy, onnxruntime) are installed."
fi

# 4. Verify ecg_anomaly.onnx exists
if [[ ! -f "ecg_anomaly.onnx" ]]; then
    if python3 -c "import torch, onnx" >/dev/null 2>&1; then
        echo "[2/3] Generating 1D-CNN ECG model (ecg_anomaly.onnx)..."
        python3 make_model.py
    else
        echo "[2/3] Note: ecg_anomaly.onnx not found; using built-in NumPy 1D-CNN fallback."
    fi
else
    echo "[2/3] Verified ecg_anomaly.onnx is present."
fi

# 5. Start Bedside Sensors & Edge Brain
echo "[3/3] Starting Bedside Sensors & Edge Brain (Broker: $BROKER)..."
if [[ -n "$PATIENTS_LIST" ]]; then
    python3 sensor_agents.py --patients $PATIENTS_LIST --primary-broker "$BROKER" --backup-broker "$BROKER" &
    SENSOR_PID=$!
    sleep 0.5
    python3 jetson_edge_brain.py --patients $PATIENTS_LIST --broker "$BROKER"
else
    python3 sensor_agents.py --patient-id "$PATIENT_ID" --primary-broker "$BROKER" --backup-broker "$BROKER" &
    SENSOR_PID=$!
    sleep 0.5
    python3 jetson_edge_brain.py --patient-id "$PATIENT_ID" --broker "$BROKER"
fi
