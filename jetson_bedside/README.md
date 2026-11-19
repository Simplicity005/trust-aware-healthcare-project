# Bedside Node

The Bedside runs the sensor agents, Jetson edge inference, and a local monitor at http://localhost:8080. It uses Python and sends telemetry over MQTT to the Nurse Station.

## Start the Bedside

From the repository root, run `./start_bedside.sh` on Linux, macOS, or WSL, or run `start_bedside.bat` on Windows. The launcher starts the sensors and the monitor together.

For a Jetson connected to a Nurse Station on the LAN, pass the bed ID and the Nurse Station's MQTT broker address:

```bash
./start_bedside.sh --patient-id Bed_1 --broker 192.168.1.50
```

Replace the address with your MQTT broker's address and use a unique bed ID for each Jetson. For multiple simulated beds on one machine, use `--patients Bed_1 Bed_2 Bed_3`.

Each bed runs eight independent MQTT workers. The ECG, pleth, respiration, capnography, and arterial pressure traces share a 250 Hz simulation clock. The monitor uses a six-second sweep for cardiac traces and a twelve-second sweep for breathing traces. Uncovered samples appear as gaps. The simulated blood pressure cuff holds its reading for 15 seconds between measurements.

## Minimal Jetson deployment

Keep `jetson_bedside/`, `nurse_station/launch.py`, and `start_bedside.sh` at their repository paths. That includes the prebuilt ECG model, monitor HTML, and Python programs. The shared launcher uses system site packages for a new Bedside virtual environment so the Jetson can reuse its installed ONNX Runtime.
