# Trust-aware bedside monitoring

Each Jetson runs a local patient monitor and simulator. The Next.js Nurse Station collects registered beds over Eclipse Mosquitto and displays their readings, ECG traces, sensor trust and alerts.

## Total Local Demo

Test the entire system on one computer using simulated Jetsons. All connections use loopback. No physical sensors, cloud account or Wi-Fi setup is needed.

1. Install Python 3.10+ with venv support and Node.js 20.9+ with npm. The launcher installs project dependencies and builds the dashboard on its first run.
2. Install Eclipse Mosquitto 2.x. On Ubuntu or Debian, run `sudo apt install mosquitto`. On macOS, run `brew install mosquitto`. Windows users can use the [Mosquitto installer](https://mosquitto.org/download/). If Docker with Compose is already installed and running, the launcher can use that instead.
3. From the repository root, run **Total Local Demo**.

| Mode | Linux, macOS or WSL | Windows | Open in a local browser |
| --- | --- | --- | --- |
| Total Local Demo | `./start_local_demo.sh` | `.\start_local_demo.bat` | Nurse Station on port 8501 and independent beds on 8080, 8081 and 8082 |
| Bedside | `./start_bedside.sh --patient-id Bed_1` | `.\start_bedside.bat --patient-id Bed_1` | Bedside on port 8080 |
| Nurse Station | `./start_nurse_station.sh` | `.\start_nurse_station.bat` | Nurse Station on port 8501 |

1. Open the [Central Dashboard](http://127.0.0.1:8501). On a fresh setup the monitored-bed count is zero. The grey numbered boxes represent available beds in the room directory.
2. Open [Bed 1](http://127.0.0.1:8080). Enter a demo patient name, department, room and a bedside password. Registration starts the eight simulated sensors. The patient appears automatically on the station. Open their numbered box to see the bed report.
3. Open **Simulation** on that bedside to change its readings or disconnect a sensor. To set up another patient, register on [Bed 2](http://127.0.0.1:8081) or [Bed 3](http://127.0.0.1:8082). Each monitor and simulation page uses its own device login.

The demo runs a private local broker on port 1884, the Nurse Station, and three separate simulated Jetsons. The default demo ignores broker addresses and credentials from your hardware setup. Use `--mqtt-port` to select another local port if 1884 is occupied. To deliberately use an existing local broker, pass `--broker 127.0.0.1 --mqtt-port PORT`.

Registrations persist between runs. Total Local Demo stores them in `~/.local/state/nurse-station/demo/` by default. Set `BEDSIDE_STATE_DIR` to a new empty directory for a fresh test. In Bash, `BEDSIDE_STATE_DIR="$(mktemp -d)" ./start_local_demo.sh` starts with zero registrations without deleting existing ones.

Press Ctrl+C in the launcher terminal to stop the apps and the native broker. When using the Docker fallback, run `docker compose down` from the repository root to stop its broker after the apps exit.

## Connect separate Jetsons and the Nurse Station

On physical hardware, each Jetson and the station connect to the same Mosquitto hostname. Wi-Fi, Ethernet and different routed network segments all work when each machine can reach the broker’s TLS port. Keep the broker on the nurse laptop for a lab or on an always-on internal server. The one-computer demo listener stays on loopback.

1. Install Mosquitto on the broker host. Configure a network listener with a server certificate for its DNS name and allow the devices to reach TCP port 8883. A typical Mosquitto 2.x configuration uses the following settings. Substitute the paths to the actual server certificate chain, private key, password file and topic permissions file. The broker service needs read access to these files. [Mosquitto configuration reference](https://mosquitto.org/man/mosquitto-conf-5.html)

```conf
listener 8883
certfile /etc/mosquitto/certs/broker.crt
keyfile /etc/mosquitto/certs/broker.key
allow_anonymous false
password_file /etc/mosquitto/passwd
acl_file /etc/mosquitto/acl
```

2. Create a separate MQTT username and password for each device using `mosquitto_passwd`. In the topic permissions file, grant the station access across beds and each Jetson access only to its own device ID. For example, for two beds:

```text
user nurse-station
topic readwrite hospital/#

user jetson-bed-1
topic readwrite hospital/Bed_1/#

user jetson-bed-2
topic readwrite hospital/Bed_2/#
```

3. Start the configured broker. In each machine’s launcher terminal, set the same broker hostname and use that machine’s own credentials. For a private certificate authority, set `MQTT_CA_FILE` to its trusted PEM certificate on each machine. The bedside dashboard password is set separately during patient registration.

```bash
export MQTT_HOST='your-broker-hostname'
export MQTT_PORT=8883
read -r -p 'MQTT username: ' MQTT_USERNAME
read -rs -p 'MQTT password: ' MQTT_PASSWORD
echo
export MQTT_USERNAME MQTT_PASSWORD
```

In Windows PowerShell, use the following before starting the batch file.

```powershell
$env:MQTT_HOST = 'your-broker-hostname'
$env:MQTT_PORT = '8883'
$credential = Get-Credential -Message 'MQTT credential for this machine'
$env:MQTT_USERNAME = $credential.UserName
$env:MQTT_PASSWORD = $credential.GetNetworkCredential().Password
```

4. On the nurse laptop run `./start_nurse_station.sh`. On each Jetson run `./start_bedside.sh --patient-id Bed_1`, substituting its device ID. On Windows use the corresponding batch file. An explicit broker host connects to the existing service.
5. Open each device’s own dashboard and register the patient at the bedside. The registration survives a restart under the same OS account. Registered Jetsons announce themselves again when the station starts later.

## Development and benchmarks

The [bedside guide](jetson_bedside/README.md) covers local storage, simulation and edge inference. The [station guide](nurse_station/README.md) covers the directory, development servers, high availability and benchmarks.
