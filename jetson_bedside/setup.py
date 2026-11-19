#!/usr/bin/env python3
"""Install Python dependencies for a standalone Jetson Bedside Monitor."""

import os
from pathlib import Path
import subprocess
import sys
import venv

BEDSIDE = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"


def ensure_venv():
    if sys.prefix != sys.base_prefix:
        return Path(sys.executable)
    venv_dir = BEDSIDE / "venv"
    executable = venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if executable.is_file():
        return executable
    print(f"Creating virtual environment in {venv_dir}...", flush=True)
    try:
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
        return executable
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not create virtual environment ({error}); installing with current Python instead.", flush=True)
        return Path(sys.executable)


def main():
    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is required.", file=sys.stderr)
        return 1

    python = ensure_venv()
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
