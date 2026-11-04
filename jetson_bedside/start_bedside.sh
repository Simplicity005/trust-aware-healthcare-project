#!/usr/bin/env bash
# ==============================================================================
# Bedside Node Launcher (Runs on an NVIDIA Jetson Orin or Simulated Bedside Unit)
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
    echo "[2/3] Generating 1D-CNN ECG model (ecg_anomaly.onnx)..."
    python3 make_model.py
fi

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
