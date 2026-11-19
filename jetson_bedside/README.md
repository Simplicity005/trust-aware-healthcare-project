# Jetson bedside

Start one Jetson per bed from the repository root. Use its assigned device ID and its HiveMQ credential from the [shared setup guide](../README.md#connect-separate-jetsons-and-the-nurse-station).

```bash
./start_bedside.sh --patient-id Bed_1
```

Open the monitor in a browser on that Jetson at [localhost:8080](http://localhost:8080). Register the department, room, patient name and a local password of at least eight characters. The device ID is fixed at launch. Registration starts the sensors and announces the assignment to the station.

The local [simulation page](http://localhost:8080/simulation) uses the same bedside login. It controls only this device’s vital targets, sensor connections, network delay, packet loss and compute load. Locking the screen ends that login while the sensors continue running. Each Jetson serves its web pages on loopback and subscribes only to control messages in its own MQTT topic.

## Registration and restart

The bedside saves its registration and salted password hash in `~/.local/state/nurse-station/<device-id>.json`. Set `BEDSIDE_STATE_DIR` to select a different storage directory. Keep the same device ID, directory and OS account across restarts. The next launch resumes the registered patient's sensors and prompts for a fresh dashboard login. Simulation settings reset on restart.

For a fresh local demonstration, point `BEDSIDE_STATE_DIR` at a new empty directory before starting the demo. To reuse a physical Jetson for a different patient, release the old bed in the station, stop the bedside, and archive that device's registration file in a protected location. Relaunch the same ID and complete registration for the new assignment.

## Sensor simulation and inference

Eight independent sensor threads share a 250 Hz patient simulation clock. ECG, pleth, respiration, capnography and arterial pressure traces use timestamped samples. The monitor uses a six-second sweep for cardiac traces and a twelve-second sweep for breathing traces. The simulated blood pressure cuff holds its measurement for 15 seconds between cycles.

The bedside runs the included ECG model through ONNX Runtime when available, preferring TensorRT, then CUDA, then CPU. A new bedside Python environment can reuse installed system packages on a Jetson. Keep the root checkout so the launcher and shared HiveMQ configuration are available.
