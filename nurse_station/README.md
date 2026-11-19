# Central Nurse Station

The Nurse Station is a Next.js dashboard backed by FastAPI. The Python engine subscribes to the MQTT broker, tracks sensor trust, handles gateway ONNX inference and station failover, and provides simulated telemetry when bedside feeds are absent.

| Path | Purpose |
| --- | --- |
| `web/` | Next.js ward and bed dashboards, live ECG, trust history, and station controls |
| `api.py` | Snapshots, server-sent telemetry events, and station actions |
| `engine.py` | Python trust calculation, MQTT ingestion and discovery, fallback simulation, and failover |
| `launch.py` | Shared process launcher for Linux, macOS, and Windows |

## Run locally

Install Python 3, Node.js 20.9 or newer with npm, and Mosquitto. Launch from the repository root:

```bash
./start_nurse_station.sh
```

The dashboard is available at http://localhost:8501. The launcher sets up dependencies, builds the app, and runs the Python API on loopback port 9501. It reuses a local broker when one is already running. Use `--broker <host>` for an external broker. On Windows, run `start_nurse_station.bat` from the repository root. If you deployed this folder by itself, run `python3 launch.py` here (`python launch.py` on Windows).

The repository-level local demo starts a bedside monitor and three simulated MQTT feeds. With this folder alone, the engine supplies its own simulated feeds and uses this folder’s `chaos.json` for the emergency simulator.

Both components ship the same ECG model. To regenerate it, run `python3 jetson_bedside/make_model.py` from the repository root and copy the resulting `ecg_anomaly.onnx` into `nurse_station/`.

## Use the dashboard

- **All wards** and the department filters show admitted beds. Open a bed to inspect ECG, compute and routing, recent alerts, and the six sensor trust scores.
- **Add new bed** probes the broker for up to 2.5 minutes. Assign a department and an optional room to a discovered device, or add a simulated bed.
- **Release bed** removes a device from active monitoring. Physical devices still sending telemetry will appear in discovery again.
- **Emergency simulator** in the sidebar triggers or clears an alarm for demonstration. **Open emergency bed** jumps directly from a ward-wide alert to the affected bed.
- Fault controls for latency, packet loss, routing, and GPU load are on the [bedside monitor](../jetson_bedside/README.md).

## Test high availability

```bash
./start_nurse_station.sh --dual
```

Open http://localhost:8501 and http://localhost:8502. Wait for the stations to see each other under **High availability**, then use **Test failover / switch role** on the primary to promote the standby. Each station runs its own Python engine and exchanges heartbeats over the shared MQTT broker. To start the secondary separately, pass `--station-id Nurse_Station_2 --port 8502` to the Nurse Station launcher. The Windows launcher accepts the same options.

## Run the web and API separately

Activate the project’s Python virtual environment and install dependencies with `python -m pip install -r requirements.txt` from this folder. In one terminal, start the API:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 9501
```

In another, run the frontend:

```bash
cd web
npm ci
npm run dev -- --port 8501
```

Set `NURSE_API_URL` in the frontend process to use a different API address. The browser connects to Next.js on the same origin for both live events and dashboard actions.

## Benchmarks

Install the Python requirements and run `python run_benchmarks.py` to generate the CSV reports and plots under `benchmarks/`.
