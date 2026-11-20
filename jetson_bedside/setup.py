#!/usr/bin/env python3
"""Install Python dependencies for a standalone Jetson Bedside Monitor."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

BEDSIDE = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"


def has_pip(python):
    return subprocess.run(
        [str(python), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def ensure_pip(python, venv_dir):
    if has_pip(python):
        return python
    print("pip not found in virtual environment — attempting ensurepip...", flush=True)
    subprocess.run(
        [str(python), "-m", "ensurepip", "--upgrade"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if has_pip(python):
        return python
    if venv_dir is not None and venv_dir.is_dir():
        print("Recreating virtual environment cleanly...", flush=True)
        shutil.rmtree(venv_dir, ignore_errors=True)
        try:
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
            executable = venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
            if executable.is_file() and has_pip(executable):
                return executable
        except (OSError, subprocess.CalledProcessError):
            pass
    if has_pip(sys.executable):
        print("Falling back to system Python (which has pip).", flush=True)
        return Path(sys.executable)
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(
        "\npip is not installed for this Python.\n"
        "On Ubuntu / Debian / Jetson, install it with:\n"
        f"  sudo apt update && sudo apt install -y python3-pip python3-venv python{ver}-venv\n"
        "Then delete any broken venv (`rm -rf venv`) and run ./setup.sh again.",
        file=sys.stderr,
    )
    return None


def ensure_venv():
    if sys.prefix != sys.base_prefix:
        return Path(sys.executable), None
    venv_dir = BEDSIDE / "venv"
    executable = venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if executable.is_file():
        return executable, venv_dir
    print(f"Creating virtual environment in {venv_dir}...", flush=True)
    try:
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
        return executable, venv_dir
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not create virtual environment ({error}); installing with current Python instead.", flush=True)
        return Path(sys.executable), None


def main():
    if sys.version_info < (3, 8):
        print("Python 3.8 or newer is required.", file=sys.stderr)
        return 1

    python, venv_dir = ensure_venv()
    python = ensure_pip(python, venv_dir)
    if python is None:
        return 1
    pip_cmd = [str(python), "-m", "pip", "install", "--upgrade"]
    if python == Path(sys.executable) and sys.prefix == sys.base_prefix:
        pip_cmd.append("--user")

    print("Installing core bedside packages (paho-mqtt, numpy)...", flush=True)
    subprocess.run([*pip_cmd, "paho-mqtt>=1.6.1", "numpy>=1.21.0"], check=True)

    print("Installing ONNX Runtime for ECG AI inference (optional)...", flush=True)
    result = subprocess.run([*pip_cmd, "onnxruntime>=1.14.0"])
    if result.returncode:
        print("ONNX Runtime wheel unavailable on this platform — bedside will use built-in fallback.", flush=True)

    start_cmd = ".\\start_bedside.bat --patient-id Bed_1" if IS_WINDOWS else "./start_bedside.sh --patient-id Bed_1"
    print("\nJetson Bedside setup complete!")
    print(f"Start the bedside monitor with:\n  {start_cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
