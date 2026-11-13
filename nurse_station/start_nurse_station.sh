#!/usr/bin/env bash
# ==============================================================================
# Nurse Station Laptop Launcher (Runs on the Nurse's Table Laptop)
# Automatically ensures requirements.txt exists, sets up venv, verifies all
# required packages are installed, starts Mosquitto, and opens the Dashboard.
# ==============================================================================

set -e
export PATH="$PATH:/opt/homebrew/sbin:/opt/homebrew/bin:/usr/local/sbin:/usr/local/bin"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${MQTT_BROKER:-127.0.0.1}"
PORT="${STREAMLIT_SERVER_PORT:-8501}"
STATION_ID="${NURSE_STATION_ID:-Nurse_Station_1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --station-id)
            STATION_ID="$2"
            shift 2
            ;;
        --broker)
            BROKER="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 1. Ensure requirements.txt exists in this folder
if [[ ! -f "requirements.txt" ]]; then
    echo "[INFO] Generating requirements.txt in $SCRIPT_DIR..."
    cat > requirements.txt << 'EOF'
paho-mqtt>=1.6.1
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
onnxruntime>=1.14.0
matplotlib>=3.6.0
EOF
fi

# 2. Activate or create virtual environment
if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -d "../venv" ]] && [[ -f "../venv/bin/activate" ]]; then
    source ../venv/bin/activate
else
    echo "[1/3] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
fi

# 3. Always verify required packages are installed in the active venv
if ! python3 -c "import paho.mqtt.client, streamlit, pandas, numpy, matplotlib" >/dev/null 2>&1; then
    echo "[1/3] Installing missing dependencies from requirements.txt..."
    pip install "paho-mqtt>=1.6.1" "streamlit>=1.28.0" "pandas>=1.5.0" "numpy>=1.21.0" "matplotlib>=3.6.0"
    pip install "onnxruntime>=1.14.0" || echo "  -> Note: onnxruntime wheel unavailable; using NumPy fallback."
else
    echo "[1/3] Verified all Nurse Station Python dependencies are installed."
fi

# 4. Verify ecg_anomaly.onnx exists
if [[ ! -f "ecg_anomaly.onnx" ]]; then
    if python3 -c "import torch, onnx" >/dev/null 2>&1; then
        echo "[2/3] Generating Gateway CPU fallback ECG model (ecg_anomaly.onnx)..."
        python3 make_model.py
    fi
else
    echo "[2/3] Verified ecg_anomaly.onnx is present."
fi

# 5. Ensure Mosquitto MQTT broker is running
if ! python3 -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', 1883)); s.close()" >/dev/null 2>&1; then
    if command -v mosquitto >/dev/null 2>&1; then
        echo "Starting Mosquitto MQTT broker on port 1883..."
        mosquitto -p 1883 -d 2>/dev/null || mosquitto -p 1883 >/dev/null 2>&1 &
        sleep 0.5
    else
        echo "[WARN] Mosquitto MQTT broker is not installed (brew install mosquitto / apt install mosquitto)."
    fi
fi

echo "[3/3] Launching Nurse Station Dashboard ($STATION_ID on port $PORT)..."
NURSE_STATION_ID="$STATION_ID" STREAMLIT_SERVER_PORT="$PORT" MQTT_BROKER="$BROKER" streamlit run dashboard.py --server.address 0.0.0.0 --server.port "$PORT"
