# Nurse Station

The Next.js Central Dashboard receives snapshots and live events from a local FastAPI service. The Python engine subscribes to registrations, sensor data and alerts through Mosquitto, tracks sensor trust, and coordinates station failover.

## Run the station

From the repository root, run `./start_nurse_station.sh` or `.\start_nurse_station.bat`. Open [localhost:8501](http://localhost:8501) on the station computer. The API binds to loopback port 9501. With default settings the launcher starts Mosquitto (native executable or Docker Compose) on loopback port 1884. For separate hardware, configure the station’s credentials using the [shared Mosquitto setup](../README.md#connect-separate-jetsons-and-the-nurse-station).

## Monitor patients

- A fresh station has zero monitored patients. A Jetson sends its assignment after bedside registration and repeats it periodically, so a station that starts later can discover it.
- Expand a department under **Room directory**. Grey boxes are vacant. Green indicates stable readings, amber indicates a monitoring or connection issue, and red flags an emergency. Select an occupied number for the patient report, readings, ECG and bed activity.
- If an admitted Jetson stops reporting for eight seconds, the station marks it offline. Sensor removal and disconnection do not generate replacement measurements.
- **Release bed** removes the patient from this running station. **Readmit bed** appears when a registered bed has been released. Configure new patients and simulate sensor changes at the bedside.

## Test high availability

Run `./start_nurse_station.sh --dual` to start the primary on port 8501 and standby on 8502. Once the peer appears, expand **High availability** in the sidebar and use **Test failover / switch role**. The stations exchange heartbeats over Mosquitto.

## Run the web and API separately

With Mosquitto running and the project Python environment activated, start the API from this directory.

```bash
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 9501
```

Run the frontend in a second terminal.

```bash
cd web
npm ci
npm run dev -- --hostname 127.0.0.1 --port 8501
```

Set `NURSE_API_URL` in the frontend process to use a different API address. The browser uses the Next.js origin for live events and station actions.

## Benchmarks

With this directory’s Python requirements installed, run `python run_benchmarks.py` here to write the CSV reports and 200-DPI plots under `benchmarks/`. To regenerate the included ECG model, run `jetson_bedside/make_model.py` and copy the resulting model to this directory.
