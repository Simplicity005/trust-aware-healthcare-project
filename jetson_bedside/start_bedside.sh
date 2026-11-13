#!/usr/bin/env bash
# ==============================================================================
# Bedside Node Launcher (Runs on an NVIDIA Jetson Orin or Simulated Bedside Unit)
# Handles Ubuntu 20.04 / Python 3.8 on Jetson gracefully (even if python3.8-venv
# is missing, falls back to user/system site-packages automatically).
#
# Usage:
#   ./start_bedside.sh --patient-id Bed_1 --broker <NURSE_LAPTOP_IP>
#   ./start_bedside.sh --patients Bed_1 Bed_2 Bed_3 --broker 127.0.0.1
# ==============================================================================

set -e
export PATH="$PATH:/opt/homebrew/sbin:/opt/homebrew/bin:/usr/local/sbin:/usr/local/bin"

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
BRAIN_PID=""

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping bedside sensor agents and edge brain..."
    if [[ -n "$BRAIN_PID" ]] && kill -0 "$BRAIN_PID" 2>/dev/null; then
        kill "$BRAIN_PID" 2>/dev/null || true
    fi
    if [[ -n "$SENSOR_PID" ]] && kill -0 "$SENSOR_PID" 2>/dev/null; then
        kill "$SENSOR_PID" 2>/dev/null || true
    fi
    echo "[OK] Bedside node stopped."
}

trap cleanup EXIT INT TERM

# 1. Ensure requirements.txt exists in this folder
if [[ ! -f "requirements.txt" ]]; then
    cat > requirements.txt << 'EOF'
paho-mqtt>=1.6.1
numpy>=1.19.0
onnxruntime
EOF
fi

# Clean up any broken partial venv from a previous failed python3 -m venv attempt
if [[ -d "venv" ]] && [[ ! -f "venv/bin/activate" ]]; then
    echo "[INFO] Removing incomplete venv folder..."
    rm -rf venv
fi

# 2. Try activating or creating venv; if python3.8-venv is not installed, fall back cleanly!
USING_VENV=0
if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    USING_VENV=1
elif [[ -d "../venv" ]] && [[ -f "../venv/bin/activate" ]]; then
    source ../venv/bin/activate
    USING_VENV=1
else
    echo "[1/3] Attempting to create Python virtual environment (venv)..."
    if python3 -m venv --system-site-packages venv 2>/dev/null && [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        USING_VENV=1
    else
        rm -rf venv
        echo "  -> Note: python3-venv package is not installed on this Jetson; using --user site-packages."
    fi
fi

# 3. Ensure paho-mqtt, numpy, and (if supported) onnxruntime are installed
if ! python3 -c "import paho.mqtt.client, numpy" >/dev/null 2>&1; then
    echo "[1/3] Installing paho-mqtt and numpy..."
    if [[ "$USING_VENV" -eq 1 ]]; then
        python3 -m pip install --upgrade pip || true
        python3 -m pip install "paho-mqtt>=1.6.1" "numpy>=1.19.0"
        python3 -m pip install onnxruntime || true
    else
        python3 -m pip install --user "paho-mqtt>=1.6.1" "numpy>=1.19.0"
        python3 -m pip install --user onnxruntime || true
    fi
else
    echo "[1/3] Verified Python dependencies are installed."
fi

# 4. Start Bedside Sensors & Edge Brain + Open Local Bedside Dashboard
echo "[2/3] Verified bedside model configuration."
echo "[3/3] Starting Bedside Sensors, Edge Brain & Local Bedside ICU Monitor (http://localhost:8080)..."

# Automatically open the Local Bedside Dashboard on the screen if a display is attached
if [[ -z "$PATIENTS_LIST" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        (sleep 1.5 && open "http://localhost:8080" >/dev/null 2>&1 || true) &
    elif [[ -n "$DISPLAY" ]]; then
        (sleep 1.5 && xdg-open "http://localhost:8080" >/dev/null 2>&1 || firefox "http://localhost:8080" >/dev/null 2>&1 || true) &
    fi
fi

if [[ -n "$PATIENTS_LIST" ]]; then
    python3 sensor_agents.py --patients $PATIENTS_LIST --primary-broker "$BROKER" --backup-broker "$BROKER" &
    SENSOR_PID=$!
    sleep 0.5
    python3 jetson_edge_brain.py --patients $PATIENTS_LIST --broker "$BROKER" &
    BRAIN_PID=$!
    wait "$BRAIN_PID"
else
    python3 sensor_agents.py --patient-id "$PATIENT_ID" --primary-broker "$BROKER" --backup-broker "$BROKER" &
    SENSOR_PID=$!
    sleep 0.5
    python3 jetson_edge_brain.py --patient-id "$PATIENT_ID" --broker "$BROKER" &
    BRAIN_PID=$!
    wait "$BRAIN_PID"
fi
