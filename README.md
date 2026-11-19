# Trust-Aware Bedside Monitoring

A hospital monitoring system with two **standalone** folders:
- **[`jetson_bedside/`](jetson_bedside/README.md)** (one per patient / Jetson): live waveforms, vitals, local login, and its own simulation page. Needs **only Python**.
- **[`nurse_station/`](nurse_station/README.md)** (central station laptop): watches all registered beds in one place, tracks sensor trust scores, and shows alerts.

They talk to each other through **Eclipse Mosquitto** (100% free, self-hosted MQTT — no paid cloud accounts or usage limits).

> **Installing on separate devices?** You do **not** have to copy the whole repository everywhere!
> - Copy **only `jetson_bedside/`** onto each Jetson (or a friend's laptop) → run `./setup.sh` (or `.\setup.bat`) inside it.
> - Copy **only `nurse_station/`** onto the Nurse Laptop → run `./setup.sh` (or `.\setup.bat`) inside it.
> - Or keep the full repo on one computer to run the 1-command local demo below.

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

Use this to test across **two laptops on completely different Wi-Fis** (e.g., your laptop as the Nurse Station and a friend's laptop acting as a Jetson), or with real Jetsons.

It uses **Eclipse Mosquitto** (running free on your laptop) + **Tailscale** (free personal VPN that makes laptops on different Wi-Fis act like they're on the same private network — no router setup, no paid cloud broker, no data limits).

### Step 1 — Install each folder on its own machine

- **On YOUR laptop (Nurse Station):** Copy `nurse_station/` (or the whole repo), `cd nurse_station`, and run `./setup.sh` (Linux/macOS) or `.\setup.bat` (Windows).
- **On your FRIEND'S laptop or each JETSON (Bedside):** Copy **only `jetson_bedside/`**, `cd jetson_bedside`, and run `./setup.sh` (Linux/macOS/Jetson) or `.\setup.bat` (Windows). *(Needs only Python 3.10+ — no Node.js, no Mosquitto, no Docker!)*

### Step 2 — Connect both machines with Tailscale *(skip if on the same Wi-Fi)*

1. Install **Tailscale** (free) on **your laptop** and **your friend's laptop (or each Jetson)** from [tailscale.com/download](https://tailscale.com/download).
2. Sign in and turn Tailscale **On** on both devices (either sign into the same Tailscale account, or share your laptop with your friend from the Tailscale web dashboard).

*(If both devices are already on the exact same Wi-Fi or mobile hotspot, you can skip Tailscale — Step 3 will automatically use your Wi-Fi IP instead.)*

### Step 3 — Start the private Mosquitto broker (1 command on YOUR laptop)

From inside `nurse_station/` (or the repo root), run **one command**:

- **Ubuntu / Debian:**
  ```bash
  sudo ./start_network_broker.sh --install-etc
  ```
- **macOS / other Linux:**
  ```bash
  ./start_network_broker.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  .\start_network_broker.bat
  ```

This **automatically** detects your Tailscale IP, generates the TLS encryption certificates and per-bed passwords, starts Mosquitto on port `8883`, and **prints the exact copy-paste commands** for Steps 4 and 5!

> **If your friend gets a connection timeout (firewall):**
> - **Ubuntu:** run `sudo ufw allow 8883/tcp` (only needed if `ufw` is enabled).
> - **Windows / macOS:** click **Allow** when the firewall popup asks about `mosquitto`.

### Step 4 — Start the Nurse Station (on YOUR laptop)

From inside `nurse_station/` (or the repo root), copy and paste **block `1)`** printed by Step 3, then open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser.

### Step 5 — Start the Bedside Monitor (on your friend's laptop or a Jetson)

1. Send your friend the public **`ca.pem`** file printed in Step 3 (`~/.local/state/nurse-station/broker/ca.pem` — completely safe to send over WhatsApp, Discord, or email; never send `server.key`).
2. Have her save `ca.pem` inside her `jetson_bedside/` folder.
3. From inside `jetson_bedside/`, have her copy and paste **block `2)`** printed in Step 3.
4. She opens [http://127.0.0.1:8080](http://127.0.0.1:8080), enters a patient name, department, room, and an 8+ character password, and clicks **Register**.
5. Within a second, that patient appears live on **your** Nurse Station — even across completely different Wi-Fi networks!

> **Useful tips:**
> - **Need more than 3 beds?** Pass `--beds Bed_1 Bed_2 Bed_3 Bed_4` to `start_network_broker`.
> - **Forgot the printed commands?** Re-run the Step 3 command — it reuses your saved passwords and reprints everything.
> - **Want to remove the Ubuntu system service config later?** Run `sudo ./start_network_broker.sh --uninstall-etc`.

---

## Quick Reference & Troubleshooting

| What you want to run | Where to run it | Linux / macOS / WSL | Windows | Browser URL |
| --- | --- | --- | --- | --- |
| Full Local Demo (1 PC) | Repo root | `./start_local_demo.sh` | `.\start_local_demo.bat` | Station: `:8501`, Beds: `:8080–8082` |
| Install Nurse Station | `nurse_station/` | `./setup.sh` | `.\setup.bat` | — |
| Private Network Broker | `nurse_station/` or root | `sudo ./start_network_broker.sh --install-etc` *(Ubuntu)* or `./start_network_broker.sh` | `.\start_network_broker.bat` | TLS MQTT on `:8883` |
| Nurse Station only | `nurse_station/` or root | `./start_nurse_station.sh` | `.\start_nurse_station.bat` | `http://127.0.0.1:8501` |
| Install Jetson Bedside | `jetson_bedside/` | `./setup.sh` | `.\setup.bat` | — |
| Single Bedside only | `jetson_bedside/` or root | `./start_bedside.sh --patient-id Bed_1` | `.\start_bedside.bat --patient-id Bed_1` | `http://127.0.0.1:8080` |

- **"Port 1884 is in use":** Pick another port with `./start_local_demo.sh --mqtt-port 1885`.
- **Want to use an existing local broker instead of starting a new one?** Pass `--broker 127.0.0.1 --mqtt-port <PORT>`.
- **Detailed folder guides:** See [`jetson_bedside/README.md`](jetson_bedside/README.md) and [`nurse_station/README.md`](nurse_station/README.md).
