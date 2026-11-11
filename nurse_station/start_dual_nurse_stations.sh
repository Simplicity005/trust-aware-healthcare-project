#!/usr/bin/env bash
# ==============================================================================
# Dual Nurse Station Launcher (High Availability Redundant Pair)
# Launches Nurse Station 1 (Primary, Port 8501) and Nurse Station 2 (Secondary, Port 8502)
# both connected to the central MQTT broker.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${MQTT_BROKER:-127.0.0.1}"
STATION1_PID=""

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping Primary Nurse Station (Port 8501)..."
    if [[ -n "$STATION1_PID" ]] && kill -0 "$STATION1_PID" 2>/dev/null; then
        kill "$STATION1_PID" 2>/dev/null || true
    fi
    echo "[OK] Terminated cleanly."
}

trap cleanup EXIT INT TERM

# Activate virtual environment
if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -d "../venv" ]] && [[ -f "../venv/bin/activate" ]]; then
    source ../venv/bin/activate
fi

# Ensure Mosquitto is active
if ! systemctl is-active --quiet mosquitto 2>/dev/null && ! pgrep -x "mosquitto" >/dev/null 2>&1; then
    if command -v mosquitto >/dev/null 2>&1; then
        echo "Starting Mosquitto MQTT broker on port 1883..."
        mosquitto -p 1883 -d || true
    fi
fi

echo "======================================================================"
echo "  Launching Dual Redundant Nurse Stations (High Availability Mode)"
echo "  Station 1 (Primary Leader):   http://localhost:8501"
echo "  Station 2 (Hot-Standby Peer): http://localhost:8502"
echo "======================================================================"

# Start Nurse Station 1 in background
NURSE_STATION_ID="Nurse_Station_1" \
STREAMLIT_SERVER_PORT=8501 \
MQTT_BROKER="$BROKER" \
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 &
STATION1_PID=$!
sleep 2

# Start Nurse Station 2 in foreground
NURSE_STATION_ID="Nurse_Station_2" \
STREAMLIT_SERVER_PORT=8502 \
MQTT_BROKER="$BROKER" \
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8502

