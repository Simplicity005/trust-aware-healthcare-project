#!/usr/bin/env python3
"""Standalone launcher for the Jetson Bedside Monitor."""

import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import signal
import subprocess
import sys
import venv

BEDSIDE = Path(__file__).resolve().parent
ROOT = BEDSIDE.parent
IS_WINDOWS = os.name == "nt"

sys.path.insert(0, str(BEDSIDE))
from mqtt_connection import broker_port, mqtt_setting, valid_device_id


def use_virtual_environment():
    if sys.prefix != sys.base_prefix:
        return None
    executable = "Scripts/python.exe" if IS_WINDOWS else "bin/python"
    for folder in (BEDSIDE / "venv", ROOT / "venv"):
        candidate = folder / executable
        if candidate.is_file():
            return candidate
    folder = BEDSIDE / "venv"
    print(f"Creating Python virtual environment in {folder}", flush=True)
    try:
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(folder)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Using the system Python for the Bedside: {error}", file=sys.stderr, flush=True)
        return None
    return folder / executable


def prepare_python():
    if any(find_spec(pkg) is None for pkg in ("paho", "numpy")):
        pip_install = [sys.executable, "-m", "pip", "install"]
        if sys.prefix == sys.base_prefix:
            pip_install.append("--user")
        subprocess.run([*pip_install, "paho-mqtt>=1.6.1", "numpy>=1.21.0"], check=True)
        if find_spec("onnxruntime") is None:
            result = subprocess.run([*pip_install, "onnxruntime>=1.14.0"])
            if result.returncode:
                print("ONNX Runtime is unavailable. Starting with the existing fallback.", flush=True)


def interrupt(_signal, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", default="Bed_1", help="Unique bed / device ID (default: Bed_1)")
    parser.add_argument("--broker", help="MQTT broker IP or hostname (defaults to MQTT_HOST or 127.0.0.1)")
    parser.add_argument("--mqtt-port", "--port", dest="mqtt_port", type=int, default=None, help="MQTT broker port")
    parser.add_argument("--web-port", "--bedside-port", dest="web_port", type=int, default=8080,
                        help="Local bedside dashboard port (default: 8080)")
    args = parser.parse_args()

    if not valid_device_id(args.patient_id):
        parser.error(f"Invalid --patient-id: {args.patient_id}")
    if not 1 <= args.web_port <= 65535:
        parser.error("Web port must be within 1-65535.")

    host = args.broker or mqtt_setting("HOST") or "127.0.0.1"
    try:
        port = args.mqtt_port if args.mqtt_port is not None else broker_port(host)
    except ValueError:
        parser.error("MQTT_PORT must be a number.")
    if not 1 <= port <= 65535:
        parser.error("MQTT port must be within 1-65535.")

    python = use_virtual_environment()
    if python:
        command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
        if IS_WINDOWS:
            return subprocess.call(command)
        os.execv(str(python), command)

    try:
        prepare_python()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Failed to install Python dependencies: {error}", file=sys.stderr)
        return 1

    signal.signal(signal.SIGTERM, interrupt)
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS else {}
    child = subprocess.Popen(
        [
            sys.executable, "jetson_edge_brain.py",
            "--patient-id", args.patient_id,
            "--broker", host,
            "--port", str(port),
            "--web-port", str(args.web_port),
        ],
        cwd=BEDSIDE,
        **options,
    )
    print(f"{args.patient_id} bedside monitor at http://127.0.0.1:{args.web_port}", flush=True)
    try:
        return child.wait()
    except KeyboardInterrupt:
        if child.poll() is None:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/pid", str(child.pid), "/t", "/f"], capture_output=True)
            else:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
