#!/usr/bin/env bash
# ==============================================================================
# Secondary Nurse Station Launcher (High Availability Hot-Standby Dashboard)
# Runs Nurse Station 2 on port 8502 concurrently with Primary Nurse Station (8501).
# If Primary Nurse Station fails, this dashboard automatically assumes active ward leadership.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${MQTT_BROKER:-127.0.0.1}"
PORT=8502
STATION_ID="Nurse_Station_2"

# Activate virtual environment
if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -d "../venv" ]] && [[ -f "../venv/bin/activate" ]]; then
    source ../venv/bin/activate
fi

echo "======================================================================"
echo "  Launching Secondary Nurse Station Dashboard ($STATION_ID)"
echo "  URL:  http://localhost:$PORT"
echo "  Role: Hot-Standby Failover Dashboard (Synchronized via MQTT)"
echo "======================================================================"

NURSE_STATION_ID="$STATION_ID" \
STREAMLIT_SERVER_PORT="$PORT" \
MQTT_BROKER="$BROKER" \
streamlit run dashboard.py --server.address 0.0.0.0 --server.port "$PORT"

