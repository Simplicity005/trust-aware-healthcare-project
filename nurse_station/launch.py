#!/usr/bin/env python3
"""Run the full local demo, the Bedside, or the Nurse Station."""

import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import venv

STATION = Path(__file__).resolve().parent
ROOT = STATION.parent
WEB = STATION / "web"
BEDSIDE = ROOT / "jetson_bedside"
NEXT = WEB / "node_modules" / "next" / "dist" / "bin" / "next"
IS_WINDOWS = os.name == "nt"

sys.path.insert(0, str(ROOT))
from mqtt_connection import broker_port, local_host, mqtt_setting


def use_virtual_environment(bedside_only):
    if sys.prefix != sys.base_prefix:
        return None
    executable = "Scripts/python.exe" if IS_WINDOWS else "bin/python"
    folders = (BEDSIDE / "venv", ROOT / "venv") if bedside_only else (ROOT / "venv", STATION / "venv")
    for folder in folders:
        candidate = folder / executable
        if candidate.is_file():
            return candidate
    folder = folders[0] if bedside_only or BEDSIDE.is_dir() else STATION / "venv"
    print(f"Creating Python virtual environment in {folder}", flush=True)
    try:
        venv.EnvBuilder(with_pip=True, system_site_packages=bedside_only).create(folder)
    except (OSError, subprocess.CalledProcessError) as error:
        if not bedside_only:
            raise
        print(f"Using the system Python for the Bedside: {error}", file=sys.stderr, flush=True)
        return None
    return folder / executable


def prepare_python(bedside_only):
    runtime = ["paho", "numpy"]
    packages = ["paho-mqtt>=1.6.1", "numpy>=1.21.0"]
    if not bedside_only:
        runtime.extend(["fastapi", "uvicorn", "pydantic"])
        packages.extend(["fastapi>=0.115,<1", "uvicorn>=0.30,<1"])
    if any(find_spec(package) is None for package in runtime):
        pip_install = [sys.executable, "-m", "pip", "install"]
        if sys.prefix == sys.base_prefix:
            pip_install.append("--user")
        subprocess.run([*pip_install, *packages], check=True)
        if find_spec("onnxruntime") is None:
            result = subprocess.run([*pip_install, "onnxruntime>=1.14.0"])
            if result.returncode:
                print("ONNX Runtime is unavailable. Starting with the existing fallback.", flush=True)


def prepare_web(parser):
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        parser.error("Node.js 20.9+ and npm are required for the Nurse Station.")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if tuple(int(part) for part in version[1:].split(".")[:2]) < (20, 9):
        parser.error(f"Node.js 20.9+ is required (found {version}).")

    lockfile = WEB / "package-lock.json"
    installed = WEB / "node_modules" / ".package-lock.json"
    if not NEXT.is_file() or not installed.is_file() or installed.stat().st_mtime < lockfile.stat().st_mtime:
        command = [npm, "ci", "--no-audit"]
        if IS_WINDOWS:
            command = ["cmd.exe", "/d", "/c", *command]
        subprocess.run(command, cwd=WEB, check=True)

    build_id = WEB / ".next" / "BUILD_ID"
    sources = [*WEB.joinpath("app").rglob("*"), WEB / "next.config.ts", WEB / "tsconfig.json", lockfile]
    newest = max(path.stat().st_mtime for path in sources if path.is_file())
    if not build_id.is_file() or build_id.stat().st_mtime < newest:
        subprocess.run([node, NEXT, "build"], cwd=WEB, check=True)
    return node


def start(children, label, command, cwd=STATION, env=None, foreground=False):
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS else {"start_new_session": not foreground}
    child = subprocess.Popen(command, cwd=cwd, env=env, **options)
    children.append((label, child))
    return child


def stop(children):
    pending = []
    for label, child in reversed(children):
        if child.poll() is not None:
            continue
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/pid", str(child.pid), "/t", "/f"], capture_output=True)
            else:
                child.terminate()
        except ProcessLookupError:
            continue
        except PermissionError as error:
            # A foreground broker can already be stopping in response to the terminal's Ctrl+C.
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                print(f"Could not stop {label} (PID {child.pid}): {error}", file=sys.stderr, flush=True)
            continue
        pending.append((label, child))
    for label, child in pending:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                child.kill()
                child.wait(timeout=5)
            except ProcessLookupError:
                pass
            except (PermissionError, subprocess.TimeoutExpired) as error:
                print(f"Could not stop {label} (PID {child.pid}): {error}", file=sys.stderr, flush=True)


def broker_listening(port):
    with socket.socket() as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_local_mosquitto(children, port):
    executable = shutil.which("mosquitto")
    if not executable and not IS_WINDOWS:
        executable = shutil.which("mosquitto", path="/usr/sbin")
    if not executable and IS_WINDOWS:
        installed = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "mosquitto" / "mosquitto.exe"
        if installed.is_file():
            executable = str(installed)
    if executable:
        if broker_listening(port):
            raise RuntimeError(f"Port {port} is in use. Use --mqtt-port to choose a free port, "
                               "or --broker 127.0.0.1 to use an existing broker intentionally.")
        help_text = subprocess.run([executable, "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        if b"mosquitto version 2." not in help_text.lower():
            raise RuntimeError("Install Mosquitto 2.x for its loopback-only local mode.")
        broker = start(children, "Local Mosquitto", [executable, "-p", str(port)], foreground=True)
        deadline = time.monotonic() + 5
        while broker.poll() is None and time.monotonic() < deadline:
            if broker_listening(port):
                print(f"Local Mosquitto listening on 127.0.0.1:{port}", flush=True)
                return
            time.sleep(0.1)
        raise RuntimeError(f"Mosquitto could not start on 127.0.0.1:{port}.")

    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Install Eclipse Mosquitto or Docker with Compose to run the local broker.")
    subprocess.run([docker, "compose", "-f", str(ROOT / "compose.yaml"), "up", "-d", "mosquitto"],
                   env={**os.environ, "MQTT_PORT": str(port)}, check=True)
    print("Mosquitto is running in Docker. Use docker compose down to stop the broker.", flush=True)


def start_station(children, node, args, station_id, port, api_port):
    environment = {
        **os.environ,
        "MQTT_HOST": args.broker,
        "MQTT_PORT": str(args.mqtt_port),
        "NURSE_STATION_ID": station_id,
        "NURSE_STATION_PORT": str(port),
    }
    start(children, f"{station_id} API", [
        sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(api_port),
        "--workers", "1", "--timeout-graceful-shutdown", "3",
    ], env=environment)
    environment["NURSE_API_URL"] = f"http://127.0.0.1:{api_port}"
    start(children, f"{station_id} dashboard", [
        node, NEXT, "start", "--hostname", args.host, "--port", str(port),
    ], cwd=WEB, env=environment)
    print(f"{station_id} dashboard at http://localhost:{port}", flush=True)


def start_bedside(children, args):
    for index, device_id in enumerate(args.patients):
        port = args.bedside_port + index
        start(children, f"Bedside {device_id}", [
            sys.executable, "jetson_edge_brain.py", "--patient-id", device_id, "--broker", args.broker,
            "--port", str(args.mqtt_port), "--web-port", str(port),
        ], cwd=BEDSIDE)
        print(f"{device_id} at http://127.0.0.1:{port}", flush=True)


def interrupt(_signal, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Run the Nurse Station and three bedside nodes")
    mode.add_argument("--bedside", action="store_true", help="Run only the Bedside")
    parser.add_argument("--port", type=int, default=int(os.environ.get("NURSE_STATION_PORT", "8501")),
                        help="Nurse Station dashboard port")
    parser.add_argument("--station-id", default=os.environ.get("NURSE_STATION_ID", "Nurse_Station_1"))
    parser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1",
                        help="Local Nurse Station dashboard address")
    parser.add_argument("--api-port", type=int, default=None, help="Local API port, defaults to 9501 or 9502")
    parser.add_argument("--broker", help="Use an existing MQTT broker at this hostname")
    parser.add_argument("--mqtt-port", type=int, default=None, help="MQTT listener port")
    parser.add_argument("--dual", action="store_true", help="Run a standby station on the next dashboard and API port")
    parser.add_argument("--bedside-port", type=int, default=8080, help="Bedside monitor port")
    beds = parser.add_mutually_exclusive_group()
    beds.add_argument("--patient-id", help="Run a single bedside node with this bed ID")
    beds.add_argument("--patients", nargs="+", help="Run bedside nodes with these bed IDs")
    args = parser.parse_args()

    if args.demo:
        os.environ.setdefault("BEDSIDE_STATE_DIR", str(Path.home() / ".local" / "state" / "nurse-station" / "demo"))
        if not args.broker:
            # The one-command demo stays local even in a terminal configured for physical Jetsons.
            for prefix in ("MQTT_", "HIVEMQ_"):
                for setting in ("HOST", "PORT", "USERNAME", "PASSWORD", "CA_FILE"):
                    os.environ.pop(prefix + setting, None)
    else:
        args.broker = args.broker or mqtt_setting("HOST")
    managed_broker = not args.broker and not args.bedside
    args.broker = args.broker or "127.0.0.1"
    local_broker = local_host(args.broker)
    try:
        if args.mqtt_port is None:
            args.mqtt_port = broker_port(args.broker)
    except ValueError:
        parser.error("MQTT_PORT must be a number.")
    args.patients = args.patients or ([args.patient_id] if args.patient_id else
                                     ["Bed_1", "Bed_2", "Bed_3"] if args.demo else ["Bed_1"])
    if len(set(args.patients)) != len(args.patients):
        parser.error("Use a unique ID for each Jetson.")
    if args.demo and not local_broker:
        parser.error("For a remote MQTT broker, start each Jetson and station separately with its own credentials.")
    has_bedside = args.demo or args.bedside
    if has_bedside and not local_broker and len(args.patients) != 1:
        parser.error("Start one Jetson per remote MQTT credential.")
    if not 1 <= args.mqtt_port <= 65535:
        parser.error("MQTT port must be within 1–65535.")
    if has_bedside and not BEDSIDE.is_dir():
        parser.error("The Bedside requires the jetson_bedside/ folder.")
    if args.bedside and args.dual:
        parser.error("--dual requires the Nurse Station.")
    if args.dual and args.station_id == "Nurse_Station_2":
        parser.error("The primary and standby must have different station IDs.")

    api_port = args.api_port if args.api_port is not None else 9502 if args.port == 8502 else 9501
    ports = [] if args.bedside else [args.port, api_port]
    if args.dual:
        ports.extend([args.port + 1, api_port + 1])
    if has_bedside:
        ports.extend(range(args.bedside_port, args.bedside_port + len(args.patients)))
    if local_broker:
        ports.append(args.mqtt_port)
    if len(set(ports)) != len(ports) or any(port < 1 or port > 65535 for port in ports):
        parser.error("Dashboard, API, local broker, and bedside ports must be distinct and within 1–65535.")

    try:
        python = use_virtual_environment(args.bedside)
        if python:
            command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
            if IS_WINDOWS:
                return subprocess.call(command)
            os.execv(str(python), command)
        prepare_python(args.bedside)
        if not args.bedside:
            node = prepare_web(parser)
        shared = []
        stations = {}
        signal.signal(signal.SIGTERM, interrupt)
        try:
            if managed_broker:
                start_local_mosquitto(shared, args.mqtt_port)
            if has_bedside:
                start_bedside(shared, args)
            if not args.bedside:
                stations[args.station_id] = []
                start_station(stations[args.station_id], node, args, args.station_id, args.port, api_port)
            if args.dual:
                stations["Nurse_Station_2"] = []
                start_station(stations["Nurse_Station_2"], node, args, "Nurse_Station_2", args.port + 1, api_port + 1)

            while True:
                for label, child in shared:
                    if child.poll() is not None:
                        raise RuntimeError(f"{label} stopped with exit code {child.returncode}.")
                for station_id, processes in list(stations.items()):
                    for label, child in processes:
                        if child.poll() is not None:
                            message = f"{label} stopped with exit code {child.returncode}."
                            if len(stations) == 1:
                                raise RuntimeError(message)
                            print(message + " Keeping the other station running.", file=sys.stderr, flush=True)
                            stop(processes)
                            del stations[station_id]
                            break
                time.sleep(0.25)
        finally:
            stop(shared + [child for processes in stations.values() for child in processes])
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
