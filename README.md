# Trust-aware bedside monitoring

Each Jetson runs a local patient monitor and simulator. The Next.js Nurse Station collects registered beds over HiveMQ and displays their readings, ECG traces, sensor trust and alerts.

## Total Local Demo

Install Python 3.10 or newer with venv support, Node.js 20.9 or newer with npm, and Docker with Compose v2. Start Docker, then run one launcher from the repository root. The launchers install their dependencies on first use.

| Mode | Linux, macOS or WSL | Windows | Open in a local browser |
| --- | --- | --- | --- |
| Total Local Demo | `./start_local_demo.sh` | `.\start_local_demo.bat` | Nurse Station on port 8501 and independent beds on 8080, 8081 and 8082 |
| Bedside | `./start_bedside.sh --patient-id Bed_1` | `.\start_bedside.bat --patient-id Bed_1` | Bedside on port 8080 |
| Nurse Station | `./start_nurse_station.sh` | `.\start_nurse_station.bat` | Nurse Station on port 8501 |

1. Open the [Central Dashboard](http://127.0.0.1:8501). On a fresh setup the monitored-bed count is zero. The grey numbered boxes represent available beds in the room directory.
2. Open [Bed 1](http://127.0.0.1:8080). Enter a demo patient name, department, room and a bedside password. Registration starts the eight simulated sensors. The patient appears automatically on the station. Open their numbered box to see the bed report.
3. Open **Simulation** on that bedside to change its readings or disconnect a sensor. To set up another patient, register on [Bed 2](http://127.0.0.1:8081) or [Bed 3](http://127.0.0.1:8082). Each monitor and simulation page uses its own device login.

The bundled HiveMQ Community Edition broker listens on loopback port 1884. Press Ctrl+C in the launcher terminal to stop its application processes. Run `docker compose down` from the project root to stop the shared local broker after all launchers have stopped.

## Connect separate Jetsons and the Nurse Station

1. Create a private HiveMQ Cloud cluster or use a managed HiveMQ broker with a trusted TLS certificate. Copy the cluster hostname and TLS MQTT port, normally 8883. Each machine needs an outbound connection to this endpoint. [HiveMQ Cloud quick start](https://docs.hivemq.com/hivemq-cloud/quick-start-guide.html)
2. Create a credential for the station and a different credential for each Jetson. Configure topic permissions on the broker as shown below. Each device ID must be unique and match its permission exactly. Use a HiveMQ deployment that supports topic-level permissions. [HiveMQ access management](https://docs.hivemq.com/hivemq-cloud/authn-authz.html)

| Credential | MQTT topic filter | Permission |
| --- | --- | --- |
| Nurse Station | `hospital/#` | Publish and subscribe |
| Jetson for Bed_1 | `hospital/Bed_1/#` | Publish and subscribe |
| Jetson for Bed_2 | `hospital/Bed_2/#` | Publish and subscribe |

3. Set the following environment variables in the terminal or service that starts each machine. Use that machine’s broker credential. The password set later in the bedside registration form protects the local dashboard.

| Variable | Value |
| --- | --- |
| `HIVEMQ_HOST` | Cluster hostname, without a scheme or port |
| `HIVEMQ_PORT` | TLS listener port, normally `8883` |
| `HIVEMQ_USERNAME` | This machine’s broker username |
| `HIVEMQ_PASSWORD` | This machine’s broker password |
| `HIVEMQ_CA_FILE` | Optional PEM file for a privately issued broker certificate |

For example, in Bash on each machine, enter its own credentials at the prompts.

```bash
export HIVEMQ_HOST='your-cluster-hostname'
export HIVEMQ_PORT=8883
read -r -p 'HiveMQ username: ' HIVEMQ_USERNAME
read -rs -p 'HiveMQ password: ' HIVEMQ_PASSWORD
echo
export HIVEMQ_USERNAME HIVEMQ_PASSWORD
```

In Windows PowerShell, use the following before starting the batch file.

```powershell
$env:HIVEMQ_HOST = 'your-cluster-hostname'
$env:HIVEMQ_PORT = '8883'
$credential = Get-Credential -Message 'HiveMQ credential for this machine'
$env:HIVEMQ_USERNAME = $credential.UserName
$env:HIVEMQ_PASSWORD = $credential.GetNetworkCredential().Password
```

4. Run `./start_nurse_station.sh` on the nurse laptop. On each Jetson, run `./start_bedside.sh --patient-id Bed_1`, substituting its own device ID. Open the dashboard on each device’s **own** browser. Both roles bind their web interface to loopback by default. An explicit HiveMQ host uses the existing broker, so Docker is optional on those machines.
5. Register the patient at their Jetson. The registration survives a restart under the same OS account. If the station starts later, registered Jetsons announce themselves again automatically.

## Development and benchmarks

The [bedside guide](jetson_bedside/README.md) covers local storage, simulation and edge inference. The [station guide](nurse_station/README.md) covers the directory, development servers, high availability and benchmarks.
