# Trust-Aware Bedside Monitoring

A hospital monitoring system with two parts:
- **Bedside Monitors** (one per patient / Jetson): live waveforms, vitals, local login, and a separate simulation page.
- **Central Nurse Station**: watches all registered beds in one place, tracks sensor trust scores, and shows alerts.

They talk to each other through **Eclipse Mosquitto** (MQTT).

---

## Option 1: Run Everything on One Computer (Local Demo)

Use this to test the whole project on a single laptop. No Jetson hardware, real sensors, or Wi-Fi setup needed.

### Step 1 — Install the 3 prerequisites

1. **Python 3.10+** (with `venv`)
   - Ubuntu / Debian: `sudo apt install python3 python3-venv`
   - macOS / Windows: install from [python.org](https://www.python.org/downloads/)
2. **Node.js 20.9+** (includes `npm`)
   - Install from [nodejs.org](https://nodejs.org/)
3. **Eclipse Mosquitto 2.x** *(the message broker)*
   - Ubuntu / Debian: `sudo apt install mosquitto`
   - macOS: `brew install mosquitto`
   - Windows: download from [mosquitto.org/download](https://mosquitto.org/download/)
   - *(Already have Docker running? You can skip installing Mosquitto — the launcher will use Docker automatically.)*

### Step 2 — Start the demo

Open a terminal in this project folder and run **one command**:

- **Linux / macOS / WSL:**
  ```bash
  ./start_local_demo.sh
  ```
- **Windows (PowerShell / Command Prompt):**
  ```powershell
  .\start_local_demo.bat
  ```

> **Note:** The very first run takes a minute or two because it installs Python & Node packages and builds the web app automatically. Wait until you see the `http://localhost:...` links printed in the terminal.

### Step 3 — Open the dashboards in your browser

1. **Open the Nurse Station:** [http://127.0.0.1:8501](http://127.0.0.1:8501)
   - At first, 0 patients are monitored. The grey numbered boxes in the sidebar are empty beds.
2. **Open Bed 1:** [http://127.0.0.1:8080](http://127.0.0.1:8080)
   - Enter any patient name, pick a department (e.g., `ICU`), room, and set a password (at least 8 characters).
   - Click **Register**. Bed 1's sensors start immediately and the patient pops up on the Nurse Station!
3. **Try the Simulator:**
   - On Bed 1's monitor, click **Simulation** to open its simulation tab.
   - Change heart rate / SpO₂, trigger an arrhythmia, or remove/unplug a sensor and watch both dashboards react.
4. **Add more beds (optional):**
   - Open [Bed 2 (port 8081)](http://127.0.0.1:8081) or [Bed 3 (port 8082)](http://127.0.0.1:8082) and register them the same way.
   - Each bed is completely isolated and has its own login and simulation page.

### Step 4 — How to stop (or reset)

- **Stop everything:** Press `Ctrl + C` in the terminal where you started the demo.
  - *(If you used Docker instead of native Mosquitto, also run `docker compose down`.)*
- **Start fresh with 0 registered beds:**
  - Demo beds are saved in `~/.local/state/nurse-station/demo/` so they survive restarts.
  - For a clean one-time test without deleting your saved beds (Linux/macOS):
    ```bash
    BEDSIDE_STATE_DIR="$(mktemp -d)" ./start_local_demo.sh
    ```
  - Or delete `~/.local/state/nurse-station/demo/` to wipe them permanently.

---

## Option 2: Run on Separate Machines

Use this when the Nurse Station is on one computer and each Bedside Monitor runs on its own Jetson over Wi-Fi or Ethernet.

### Step 1 — Set up Mosquitto on one machine (the Nurse Laptop or a server)

1. Install **Eclipse Mosquitto 2.x** on that machine and allow TCP port `8883` through its firewall.
2. Put your TLS certificate & key in `/etc/mosquitto/certs/` (make sure the `mosquitto` user can read them).
3. Create one login for the Nurse Station and one login per Jetson:
   ```bash
   sudo mosquitto_passwd -c /etc/mosquitto/passwd nurse-station
   sudo mosquitto_passwd /etc/mosquitto/passwd jetson-bed-1
   sudo mosquitto_passwd /etc/mosquitto/passwd jetson-bed-2
   ```
4. Create `/etc/mosquitto/acl` so each Jetson can only publish to its own bed:
   ```text
   user nurse-station
   topic readwrite hospital/#

   user jetson-bed-1
   topic readwrite hospital/Bed_1/#

   user jetson-bed-2
   topic readwrite hospital/Bed_2/#
   ```
5. Add these settings to `/etc/mosquitto/mosquitto.conf` (or a file in `/etc/mosquitto/conf.d/`) and restart Mosquitto:
   ```conf
   listener 8883
   certfile /etc/mosquitto/certs/broker.crt
   keyfile /etc/mosquitto/certs/broker.key
   allow_anonymous false
   password_file /etc/mosquitto/passwd
   acl_file /etc/mosquitto/acl
   ```

### Step 2 — Start the Nurse Station (on the Nurse Laptop)

Set the broker details in your terminal, then start the station:

- **Linux / macOS:**
  ```bash
  export MQTT_HOST='your-broker-hostname'
  export MQTT_PORT=8883
  export MQTT_USERNAME='nurse-station'
  read -rs -p 'MQTT password: ' MQTT_PASSWORD && echo
  export MQTT_PASSWORD
  # If using a self-signed/private CA certificate, also run:
  # export MQTT_CA_FILE='/path/to/ca.pem'

  ./start_nurse_station.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:MQTT_HOST = 'your-broker-hostname'
  $env:MQTT_PORT = '8883'
  $cred = Get-Credential -UserName 'nurse-station' -Message 'Nurse Station MQTT password'
  $env:MQTT_USERNAME = $cred.UserName
  $env:MQTT_PASSWORD = $cred.GetNetworkCredential().Password
  # Optional private CA: $env:MQTT_CA_FILE = 'C:\path\to\ca.pem'

  .\start_nurse_station.bat
  ```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) on the Nurse Laptop.

### Step 3 — Start each Bedside Unit (on each Jetson)

On **Jetson 1**, set the same `MQTT_HOST` and `MQTT_PORT`, use `jetson-bed-1` credentials, and launch:

```bash
export MQTT_HOST='your-broker-hostname'
export MQTT_PORT=8883
export MQTT_USERNAME='jetson-bed-1'
read -rs -p 'MQTT password: ' MQTT_PASSWORD && echo
export MQTT_PASSWORD
# Optional private CA: export MQTT_CA_FILE='/path/to/ca.pem'

./start_bedside.sh --patient-id Bed_1
```

On **Jetson 2**, do the same with `jetson-bed-2` and `--patient-id Bed_2` (and on Windows, use `.\start_bedside.bat --patient-id Bed_1`).

### Step 4 — Register each patient at the bedside

1. On each Jetson, open [http://127.0.0.1:8080](http://127.0.0.1:8080) in its local browser.
2. Fill in the patient's details, ward/room, and set a bedside password.
3. The bed will immediately appear on the Nurse Station. Registrations are saved automatically, so if a Jetson restarts it will reconnect on its own.

---

## Quick Reference & Troubleshooting

| What you want to run | Linux / macOS / WSL | Windows | Browser URL |
| --- | --- | --- | --- |
| Everything on 1 PC (Demo) | `./start_local_demo.sh` | `.\start_local_demo.bat` | Station: `:8501`, Beds: `:8080`, `:8081`, `:8082` |
| Nurse Station only | `./start_nurse_station.sh` | `.\start_nurse_station.bat` | `http://127.0.0.1:8501` |
| Single Bedside only | `./start_bedside.sh --patient-id Bed_1` | `.\start_bedside.bat --patient-id Bed_1` | `http://127.0.0.1:8080` |

- **"Port 1884 is in use":** Pick another port with `./start_local_demo.sh --mqtt-port 1885`.
- **Want to use an existing local broker instead of starting a new one?** Pass `--broker 127.0.0.1 --mqtt-port <PORT>`.
- **More technical details:** See [`jetson_bedside/README.md`](jetson_bedside/README.md) (edge AI, simulation, state storage) and [`nurse_station/README.md`](nurse_station/README.md) (high availability failover, dev mode, benchmarks).
