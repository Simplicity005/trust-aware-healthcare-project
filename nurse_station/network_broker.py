#!/usr/bin/env python3
"""Start a private TLS-encrypted Eclipse Mosquitto broker for Tailscale or LAN."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import string
import subprocess
import sys
import tempfile
import time

if os.name != "nt":
    import grp
    import pwd

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"

sys.path.insert(0, str(ROOT))
from mqtt_connection import valid_device_id


def invoking_user_info():
    """Return (home_path, uid, gid) for the real user even when invoked under sudo."""
    if not IS_WINDOWS and os.geteuid() == 0 and os.environ.get("SUDO_USER"):
        try:
            entry = pwd.getpwnam(os.environ["SUDO_USER"])
            return Path(entry.pw_dir), entry.pw_uid, entry.pw_gid
        except KeyError:
            pass
    return Path.home(), None, None


def default_state_dir():
    home, _, _ = invoking_user_info()
    return home / ".local" / "state" / "nurse-station" / "broker"


def find_executable(name, extra_dirs=()):
    found = shutil.which(name)
    if found:
        return found
    for directory in extra_dirs:
        candidate = Path(directory) / (f"{name}.exe" if IS_WINDOWS else name)
        if candidate.is_file():
            return str(candidate)
    return None


def find_mosquitto():
    extra = (
        [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "mosquitto"]
        if IS_WINDOWS
        else ["/usr/sbin", "/usr/local/sbin", "/opt/homebrew/sbin", "/opt/homebrew/bin"]
    )
    return find_executable("mosquitto", extra), find_executable("mosquitto_passwd", extra)


def find_openssl():
    extra = (
        [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "usr" / "bin",
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "OpenSSL-Win64" / "bin",
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Git" / "usr" / "bin",
        ]
        if IS_WINDOWS
        else ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]
    )
    return find_executable("openssl", extra)


def detect_tailscale_ip():
    extra = (
        [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tailscale"]
        if IS_WINDOWS
        else ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/Applications/Tailscale.app/Contents/MacOS"]
    )
    tailscale = find_executable("tailscale", extra)
    if not tailscale:
        return None
    try:
        result = subprocess.run([tailscale, "ip", "-4"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                ip = line.strip()
                if ip:
                    ipaddress.ip_address(ip)
                    return ip
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def detect_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return None


def is_ip_address(value):
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def bed_username(bed_id):
    return "jetson-" + bed_id.lower().replace("_", "-")


def random_password(length=16):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def restrict_permissions(path, mode=0o600):
    if not IS_WINDOWS:
        os.chmod(path, mode)


def chown_to_invoking_user(path):
    _, uid, gid = invoking_user_info()
    if uid is not None and gid is not None:
        os.chown(path, uid, gid)


def ensure_certificates(state_dir, openssl, host):
    ca_key = state_dir / "ca.key"
    ca_cert = state_dir / "ca.pem"
    server_key = state_dir / "server.key"
    server_cert = state_dir / "server.crt"
    meta_file = state_dir / "cert_meta.json"

    if not ca_key.is_file() or not ca_cert.is_file():
        with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as cfg:
            cfg.write(
                "[req]\n"
                "distinguished_name = dn\n"
                "x509_extensions = v3_ca\n"
                "prompt = no\n\n"
                "[dn]\n"
                "CN = Nurse Station Local MQTT CA\n\n"
                "[v3_ca]\n"
                "basicConstraints = critical, CA:TRUE\n"
                "keyUsage = critical, keyCertSign, cRLSign\n"
                "subjectKeyIdentifier = hash\n"
            )
            ca_cfg = cfg.name
        try:
            subprocess.run(
                [
                    openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(ca_key), "-out", str(ca_cert),
                    "-days", "3650", "-config", ca_cfg,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            restrict_permissions(ca_key, 0o600)
            chown_to_invoking_user(ca_key)
            chown_to_invoking_user(ca_cert)
        finally:
            os.unlink(ca_cfg)

    existing_hosts = []
    if meta_file.is_file():
        try:
            existing_hosts = json.loads(meta_file.read_text(encoding="utf-8")).get("hosts", [])
        except (OSError, ValueError):
            existing_hosts = []

    need_server_cert = not server_key.is_file() or not server_cert.is_file() or host not in existing_hosts
    if need_server_cert:
        hosts = sorted(set(existing_hosts) | {host})
        san_entries = ["IP:127.0.0.1", "DNS:localhost"]
        for entry in hosts:
            prefix = "IP" if is_ip_address(entry) else "DNS"
            item = f"{prefix}:{entry}"
            if item not in san_entries:
                san_entries.append(item)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csr_file = tmp / "server.csr"
            ext_file = tmp / "server.ext"
            ext_file.write_text(
                "basicConstraints = critical, CA:FALSE\n"
                "keyUsage = critical, digitalSignature, keyEncipherment\n"
                "extendedKeyUsage = serverAuth\n"
                "subjectKeyIdentifier = hash\n"
                "authorityKeyIdentifier = keyid,issuer\n"
                f"subjectAltName = {', '.join(san_entries)}\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    openssl, "req", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(server_key), "-out", str(csr_file),
                    "-subj", f"/CN={host}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                [
                    openssl, "x509", "-req", "-in", str(csr_file),
                    "-CA", str(ca_cert), "-CAkey", str(ca_key), "-CAcreateserial",
                    "-out", str(server_cert), "-days", "825", "-extfile", str(ext_file),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            restrict_permissions(server_key, 0o600)
            chown_to_invoking_user(server_key)
            chown_to_invoking_user(server_cert)
        meta_file.write_text(json.dumps({"hosts": hosts}, indent=2), encoding="utf-8")
        chown_to_invoking_user(meta_file)
        srl_file = state_dir / "ca.srl"
        if srl_file.is_file():
            chown_to_invoking_user(srl_file)

    return ca_cert, server_cert, server_key


def ensure_credentials_and_config(state_dir, mosquitto_passwd, beds, port, ca_cert, server_cert, server_key, reset):
    creds_file = state_dir / "credentials.json"
    creds = {}
    if creds_file.is_file() and not reset:
        try:
            creds = json.loads(creds_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            creds = {}

    users = ["nurse-station"] + [bed_username(bed) for bed in beds]
    for user in users:
        if not creds.get(user):
            creds[user] = random_password()

    creds_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    restrict_permissions(creds_file, 0o600)
    chown_to_invoking_user(creds_file)

    passwd_file = state_dir / "passwd"
    if passwd_file.exists():
        passwd_file.unlink()
    for index, user in enumerate(creds):
        flag = ["-c"] if index == 0 else []
        subprocess.run(
            [mosquitto_passwd, "-b", *flag, str(passwd_file), user, creds[user]],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    restrict_permissions(passwd_file, 0o600)
    chown_to_invoking_user(passwd_file)

    acl_lines = [
        "user nurse-station",
        "topic readwrite hospital/#",
        "",
    ]
    for bed in beds:
        acl_lines.extend([
            f"user {bed_username(bed)}",
            f"topic readwrite hospital/{bed}/#",
            "",
        ])
    acl_file = state_dir / "acl"
    acl_file.write_text("\n".join(acl_lines), encoding="utf-8")
    restrict_permissions(acl_file, 0o600)
    chown_to_invoking_user(acl_file)

    def mosquitto_path(path):
        return path.resolve().as_posix()

    conf_file = state_dir / "mosquitto.conf"
    conf_file.write_text(
        f"listener {port} 0.0.0.0\n"
        f"cafile {mosquitto_path(ca_cert)}\n"
        f"certfile {mosquitto_path(server_cert)}\n"
        f"keyfile {mosquitto_path(server_key)}\n"
        "allow_anonymous false\n"
        f"password_file {mosquitto_path(passwd_file)}\n"
        f"acl_file {mosquitto_path(acl_file)}\n",
        encoding="utf-8",
    )
    chown_to_invoking_user(conf_file)

    docker_conf_file = state_dir / "mosquitto-docker.conf"
    docker_conf_file.write_text(
        f"listener {port} 0.0.0.0\n"
        "cafile /mosquitto/config/ca.pem\n"
        "certfile /mosquitto/config/server.crt\n"
        "keyfile /mosquitto/config/server.key\n"
        "allow_anonymous false\n"
        "password_file /mosquitto/config/passwd\n"
        "acl_file /mosquitto/config/acl\n",
        encoding="utf-8",
    )
    chown_to_invoking_user(docker_conf_file)

    return creds, conf_file, docker_conf_file, passwd_file, acl_file


def install_into_etc_mosquitto(port, ca_cert, server_cert, server_key, passwd_file, acl_file):
    if IS_WINDOWS:
        raise RuntimeError("--install-etc is only needed on Linux (Ubuntu/Debian).")
    if os.geteuid() != 0:
        raise RuntimeError("Run with sudo: sudo ./start_network_broker.sh --install-etc")

    etc_dir = Path("/etc/mosquitto")
    certs_dir = etc_dir / "certs"
    conf_d_dir = etc_dir / "conf.d"
    certs_dir.mkdir(parents=True, exist_ok=True)
    conf_d_dir.mkdir(parents=True, exist_ok=True)

    try:
        mosq_gid = grp.getgrnam("mosquitto").gr_gid
    except KeyError:
        mosq_gid = 0

    targets = [
        (ca_cert, certs_dir / "nurse-station-ca.pem", 0o644),
        (server_cert, certs_dir / "nurse-station-server.crt", 0o644),
        (server_key, certs_dir / "nurse-station-server.key", 0o640),
        (passwd_file, etc_dir / "nurse-station.passwd", 0o640),
        (acl_file, etc_dir / "nurse-station.acl", 0o640),
    ]
    for src, dst, mode in targets:
        shutil.copyfile(src, dst)
        os.chown(dst, 0, mosq_gid)
        os.chmod(dst, mode)

    conf_dst = conf_d_dir / "nurse-station.conf"
    conf_dst.write_text(
        f"listener {port} 0.0.0.0\n"
        "cafile /etc/mosquitto/certs/nurse-station-ca.pem\n"
        "certfile /etc/mosquitto/certs/nurse-station-server.crt\n"
        "keyfile /etc/mosquitto/certs/nurse-station-server.key\n"
        "allow_anonymous false\n"
        "password_file /etc/mosquitto/nurse-station.passwd\n"
        "acl_file /etc/mosquitto/nurse-station.acl\n",
        encoding="utf-8",
    )
    os.chown(conf_dst, 0, mosq_gid)
    os.chmod(conf_dst, 0o644)

    systemctl = shutil.which("systemctl")
    if not systemctl:
        raise RuntimeError("Installed config in /etc/mosquitto/, but systemctl was not found. Restart mosquitto manually.")
    subprocess.run([systemctl, "enable", "--now", "mosquitto"], check=True)
    subprocess.run([systemctl, "restart", "mosquitto"], check=True)


def uninstall_from_etc_mosquitto():
    if IS_WINDOWS or os.geteuid() != 0:
        raise RuntimeError("Run with sudo on Linux: sudo ./start_network_broker.sh --uninstall-etc")
    files = [
        Path("/etc/mosquitto/conf.d/nurse-station.conf"),
        Path("/etc/mosquitto/certs/nurse-station-ca.pem"),
        Path("/etc/mosquitto/certs/nurse-station-server.crt"),
        Path("/etc/mosquitto/certs/nurse-station-server.key"),
        Path("/etc/mosquitto/nurse-station.passwd"),
        Path("/etc/mosquitto/nurse-station.acl"),
    ]
    for path in files:
        if path.exists():
            path.unlink()
    systemctl = shutil.which("systemctl")
    if systemctl:
        subprocess.run([systemctl, "restart", "mosquitto"], check=False)
    print("Removed Nurse Station config from /etc/mosquitto/ and restarted mosquitto.")


def port_listening(port):
    with socket.socket() as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def print_connection_guide(host, port, ca_cert, beds, creds, source_label, service_mode=False):
    ca_path = str(ca_cert.resolve())
    station_pw = creds["nurse-station"]
    first_bed = beds[0]
    first_user = bed_username(first_bed)
    first_pw = creds[first_user]

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  Mosquitto TLS Broker READY on {host}:{port} ({source_label})")
    print(f"{bar}\n")

    print("1) ON THIS LAPTOP — Run the Nurse Station in a terminal:")
    if IS_WINDOWS:
        print(
            f"   $env:MQTT_HOST = '{host}'\n"
            f"   $env:MQTT_PORT = '{port}'\n"
            f"   $env:MQTT_USERNAME = 'nurse-station'\n"
            f"   $env:MQTT_PASSWORD = '{station_pw}'\n"
            f"   $env:MQTT_CA_FILE = '{ca_path}'\n"
            f"   .\\start_nurse_station.bat\n"
        )
    else:
        print(
            f"   export MQTT_HOST='{host}'\n"
            f"   export MQTT_PORT={port}\n"
            f"   export MQTT_USERNAME='nurse-station'\n"
            f"   export MQTT_PASSWORD='{station_pw}'\n"
            f"   export MQTT_CA_FILE='{ca_path}'\n"
            f"   ./start_nurse_station.sh\n"
        )

    print("2) ON YOUR FRIEND'S LAPTOP / JETSON (different Wi-Fi via Tailscale or same Wi-Fi):")
    print(f"   First, send ONLY this public certificate file:\n     {ca_path}")
    print("   (Have them save it inside the project folder as ca.pem)\n")
    print(f"   Then they run ({first_bed}):")
    print("   [Linux / macOS / Jetson]")
    print(
        f"     export MQTT_HOST='{host}'\n"
        f"     export MQTT_PORT={port}\n"
        f"     export MQTT_USERNAME='{first_user}'\n"
        f"     export MQTT_PASSWORD='{first_pw}'\n"
        f"     export MQTT_CA_FILE='./ca.pem'\n"
        f"     ./start_bedside.sh --patient-id {first_bed}\n"
    )
    print("   [Windows PowerShell]")
    print(
        f"     $env:MQTT_HOST = '{host}'\n"
        f"     $env:MQTT_PORT = '{port}'\n"
        f"     $env:MQTT_USERNAME = '{first_user}'\n"
        f"     $env:MQTT_PASSWORD = '{first_pw}'\n"
        f"     $env:MQTT_CA_FILE = '.\\ca.pem'\n"
        f"     .\\start_bedside.bat --patient-id {first_bed}\n"
    )

    if len(beds) > 1:
        print("   Extra bed logins already created (same MQTT_HOST, MQTT_PORT, and ca.pem):")
        for bed in beds[1:]:
            user = bed_username(bed)
            print(f"     {bed}: username='{user}'  password='{creds[user]}'")
        print()

    if service_mode:
        print("Mosquitto is running in the background as a system service (starts automatically on boot).")
        print("To stop/remove this config later: sudo ./start_network_broker.sh --uninstall-etc\n", flush=True)
    else:
        print("Keep this terminal open. Press Ctrl+C here to stop the broker.\n", flush=True)


def interrupt(_signal, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        help="Your Tailscale IP (100.x.y.z), Tailscale MagicDNS name, or Wi-Fi IP. Auto-detected if omitted.",
    )
    parser.add_argument("--port", type=int, default=8883, help="TLS MQTT port (default: 8883)")
    parser.add_argument(
        "--beds",
        nargs="+",
        default=["Bed_1", "Bed_2", "Bed_3"],
        help="Bed IDs to create logins for (default: Bed_1 Bed_2 Bed_3)",
    )
    parser.add_argument(
        "--state-dir",
        default=str(default_state_dir()),
        help="Folder where certificates, passwords, and Mosquitto config are stored",
    )
    parser.add_argument(
        "--reset-credentials",
        action="store_true",
        help="Generate brand-new passwords for all accounts",
    )
    parser.add_argument(
        "--install-etc",
        action="store_true",
        help="Install generated config into /etc/mosquitto/ and restart the system mosquitto service (Ubuntu/Debian)",
    )
    parser.add_argument(
        "--uninstall-etc",
        action="store_true",
        help="Remove Nurse Station config from /etc/mosquitto/ and restart mosquitto",
    )
    args = parser.parse_args()

    if args.uninstall_etc:
        try:
            uninstall_from_etc_mosquitto()
            return 0
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1

    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    for bed in args.beds:
        if not valid_device_id(bed):
            parser.error(f"Invalid bed ID: {bed}")

    mosquitto, mosquitto_passwd = find_mosquitto()
    docker = shutil.which("docker")
    if not mosquitto_passwd:
        print(
            "Eclipse Mosquitto 2.x is required to host the broker.\n"
            "  Ubuntu/Debian: sudo apt install mosquitto\n"
            "  macOS:         brew install mosquitto\n"
            "  Windows:       https://mosquitto.org/download/",
            file=sys.stderr,
        )
        return 1

    openssl = find_openssl()
    if not openssl:
        print(
            "OpenSSL is required to generate the TLS certificate.\n"
            "  Ubuntu/Debian: sudo apt install openssl\n"
            "  macOS:         included by default\n"
            "  Windows:       install Git for Windows (https://git-scm.com) — it includes openssl",
            file=sys.stderr,
        )
        return 1

    if args.host:
        host = args.host.strip()
        source_label = "provided via --host"
    else:
        tailscale_ip = detect_tailscale_ip()
        if tailscale_ip:
            host = tailscale_ip
            source_label = "Tailscale IP"
        else:
            lan_ip = detect_lan_ip()
            if lan_ip:
                host = lan_ip
                source_label = "Local Wi-Fi/LAN IP (Tailscale not detected)"
            else:
                print(
                    "Could not detect your Tailscale or Wi-Fi IP automatically.\n"
                    "Start Tailscale (or pass your IP manually): ./start_network_broker.sh --host 100.x.y.z",
                    file=sys.stderr,
                )
                return 1

    if not args.install_etc and port_listening(args.port):
        print(
            f"Port {args.port} is already in use. Stop the other Mosquitto instance or pick another port with --port 8884.",
            file=sys.stderr,
        )
        return 1

    state_dir = Path(args.state_dir).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    restrict_permissions(state_dir, 0o700)
    chown_to_invoking_user(state_dir)

    try:
        ca_cert, server_cert, server_key = ensure_certificates(state_dir, openssl, host)
        creds, conf_file, docker_conf_file, passwd_file, acl_file = ensure_credentials_and_config(
            state_dir, mosquitto_passwd, args.beds, args.port,
            ca_cert, server_cert, server_key, args.reset_credentials,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode(errors="replace") if error.stderr else str(error)
        print(f"Broker setup failed: {details}", file=sys.stderr)
        return 1

    if args.install_etc:
        try:
            install_into_etc_mosquitto(args.port, ca_cert, server_cert, server_key, passwd_file, acl_file)
        except (RuntimeError, subprocess.CalledProcessError) as error:
            print(f"Could not install into /etc/mosquitto/: {error}", file=sys.stderr)
            return 1
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not port_listening(args.port):
            time.sleep(0.1)
        if not port_listening(args.port):
            print(
                f"Mosquitto restarted, but port {args.port} is not listening yet. Check: sudo journalctl -u mosquitto -n 30",
                file=sys.stderr,
            )
            return 1
        print_connection_guide(host, args.port, ca_cert, args.beds, creds, source_label, service_mode=True)
        return 0

    signal.signal(signal.SIGTERM, interrupt)
    broker = None
    broker_log = None
    using_docker = False
    if mosquitto:
        log_path = state_dir / "broker.log"
        broker_log = open(log_path, "w", encoding="utf-8")
        broker = subprocess.Popen(
            [mosquitto, "-c", str(conf_file)],
            stderr=broker_log,
        )
        deadline = time.monotonic() + 5
        while broker.poll() is None and time.monotonic() < deadline:
            if port_listening(args.port):
                break
            time.sleep(0.1)
        if broker.poll() is not None:
            broker_log.close()
            broker_log = None
            err_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            if "Unable to open config file" in err_text:
                if docker:
                    using_docker = True
                    broker = None
                else:
                    print(
                        "Ubuntu's Mosquitto security policy (AppArmor) requires broker config files to live in /etc/mosquitto/.\n"
                        "Run this one command instead (it installs everything into /etc/mosquitto/ automatically):\n\n"
                        f"  sudo ./start_network_broker.sh --install-etc"
                        + (f" --host {args.host}" if args.host else "")
                        + (f" --port {args.port}" if args.port != 8883 else "")
                        + "\n",
                        file=sys.stderr,
                    )
                    return 1
            else:
                if err_text:
                    print(err_text.strip(), file=sys.stderr)
                print(f"Mosquitto exited unexpectedly with code {broker.returncode}.", file=sys.stderr)
                return broker.returncode or 1

    if using_docker or not mosquitto:
        if not docker:
            print("Install Eclipse Mosquitto 2.x or Docker to host the broker.", file=sys.stderr)
            return 1
        broker = subprocess.Popen([
            docker, "run", "--rm",
            "-p", f"0.0.0.0:{args.port}:{args.port}",
            "-v", f"{state_dir.resolve()}:/mosquitto/config:ro",
            "eclipse-mosquitto:2",
            "mosquitto", "-c", "/mosquitto/config/mosquitto-docker.conf",
        ])
        deadline = time.monotonic() + 15
        while broker.poll() is None and time.monotonic() < deadline:
            if port_listening(args.port):
                break
            time.sleep(0.2)
        if broker.poll() is not None:
            print(f"Docker Mosquitto exited unexpectedly with code {broker.returncode}.", file=sys.stderr)
            return broker.returncode or 1

    try:
        print_connection_guide(host, args.port, ca_cert, args.beds, creds, source_label, service_mode=False)
        while broker.poll() is None:
            time.sleep(0.25)
        return broker.returncode or 0
    except KeyboardInterrupt:
        print("\nStopping Mosquitto broker...", flush=True)
        return 0
    finally:
        if broker and broker.poll() is None:
            broker.terminate()
            try:
                broker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                broker.kill()
        if broker_log:
            broker_log.close()


if __name__ == "__main__":
    sys.exit(main())
