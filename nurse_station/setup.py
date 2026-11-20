#!/usr/bin/env python3
"""Install Python and Node.js dependencies and build the Nurse Station dashboard."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

STATION = Path(__file__).resolve().parent
WEB = STATION / "web"
NEXT = WEB / "node_modules" / "next" / "dist" / "bin" / "next"
IS_WINDOWS = os.name == "nt"


def ensure_venv():
    if sys.prefix != sys.base_prefix:
        return Path(sys.executable)
    venv_dir = STATION / "venv"
    executable = venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if executable.is_file():
        return executable
    print(f"Creating virtual environment in {venv_dir}...", flush=True)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    return executable


def check_mosquitto():
    extra = (
        [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "mosquitto"]
        if IS_WINDOWS
        else ["/usr/sbin", "/usr/local/sbin", "/opt/homebrew/sbin", "/opt/homebrew/bin"]
    )
    if shutil.which("mosquitto"):
        return True
    for directory in extra:
        candidate = Path(directory) / ("mosquitto.exe" if IS_WINDOWS else "mosquitto")
        if candidate.is_file():
            return True
    return bool(shutil.which("docker"))


def main():
    if sys.version_info < (3, 8):
        print("Python 3.8 or newer is required.", file=sys.stderr)
        return 1

    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        print("Node.js 20.9+ and npm are required. Download from https://nodejs.org/", file=sys.stderr)
        return 1
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if tuple(int(part) for part in version[1:].split(".")[:2]) < (20, 9):
        print(f"Node.js 20.9+ is required (found {version}).", file=sys.stderr)
        return 1

    python = ensure_venv()
    pip_cmd = [str(python), "-m", "pip", "install", "--upgrade"]

    print("Installing Nurse Station Python packages...", flush=True)
    core_packages = [
        "paho-mqtt>=1.6.1",
        "fastapi>=0.115,<1",
        "uvicorn>=0.30,<1",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
    ]
    subprocess.run([*pip_cmd, *core_packages], check=True)

    print("Installing ONNX Runtime for ECG AI inference (optional)...", flush=True)
    result = subprocess.run([*pip_cmd, "onnxruntime>=1.14.0"])
    if result.returncode:
        print("ONNX Runtime wheel unavailable on this platform — station will use built-in fallback.", flush=True)

    print("Installing Nurse Station web packages (npm ci)...", flush=True)
    npm_cmd = [npm, "ci", "--no-audit"]
    if IS_WINDOWS:
        npm_cmd = ["cmd.exe", "/d", "/c", *npm_cmd]
    subprocess.run(npm_cmd, cwd=WEB, check=True)

    print("Building Nurse Station web dashboard (next build)...", flush=True)
    subprocess.run([node, str(NEXT), "build"], cwd=WEB, check=True)

    if not check_mosquitto():
        print(
            "\nNote: Eclipse Mosquitto 2.x was not detected on this computer.\n"
            "Install it to host the MQTT broker:\n"
            "  Ubuntu / Debian: sudo apt install mosquitto\n"
            "  macOS:           brew install mosquitto\n"
            "  Windows:         https://mosquitto.org/download/",
            flush=True,
        )

    if IS_WINDOWS:
        print("\nNurse Station setup complete!")
        print("  Start locally:            .\\start_nurse_station.bat")
        print("  Host for remote Jetsons:  .\\start_network_broker.bat")
    else:
        print("\nNurse Station setup complete!")
        print("  Start locally:            ./start_nurse_station.sh")
        print("  Host for remote Jetsons:  sudo ./start_network_broker.sh --install-etc   (Ubuntu/Debian)")
        print("                            ./start_network_broker.sh                      (macOS/other Linux)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
