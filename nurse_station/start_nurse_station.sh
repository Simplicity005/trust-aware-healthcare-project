#!/usr/bin/env bash
# ==============================================================================
# Nurse Station Laptop Launcher (Runs on the Nurse's Table Laptop)
# Automatically ensures requirements.txt exists, sets up venv, verifies all
# required packages are installed, starts Mosquitto, and opens the Dashboard.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${MQTT_BROKER:-127.0.0.1}"

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
if ! python3 -c "import paho.mqtt.client, streamlit, pandas, numpy, onnxruntime, matplotlib" >/dev/null 2>&1; then
    echo "[1/3] Installing missing dependencies from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
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
if ! systemctl is-active --quiet mosquitto 2>/dev/null && ! pgrep -x "mosquitto" >/dev/null 2>&1; then
    if command -v mosquitto >/dev/null 2>&1; then
        echo "Starting Mosquitto MQTT broker on port 1883..."
        mosquitto -p 1883 -d || true
    fi
fi

echo "[3/3] Launching Multi-Patient Nurse Station Dashboard..."
MQTT_BROKER="$BROKER" streamlit run dashboard.py
