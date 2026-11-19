#!/usr/bin/env python3
"""
Bedside GPU Edge Brain & Local Bedside Monitor (Runs on NVIDIA Jetson Orin)
Executes 1D-CNN ECG anomaly inference, triggers local bedside alarms,
validates cross-sensor physiological plausibility, streams bedside summaries to
the Nurse Station, and serves an isolated, password-protected Local Bedside Monitor
for each bedside node on loopback (127.0.0.1).
"""

import os
import sys
import time
import json
import hmac
import signal
import copy
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs, urlsplit
import hashlib
import secrets
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from sensor_agents import SensorAgentThread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mqtt_connection import DEPARTMENTS, broker_host, broker_port, mqtt_client, valid_device_id
from patient_simulation import PatientSimulation, SAMPLE_RATE, SENSOR_AGENT_IDS, WAVEFORM_KEYS
from trust_evaluator import AgentTrustTracker

try:
    import onnxruntime as ort
except ImportError:
    ort = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
BEDSIDE_HTML_FILE = os.path.join(BASE_DIR, "bedside_monitor.html")
BEDSIDE_SIM_HTML_FILE = os.path.join(BASE_DIR, "bedside_simulation.html")


def load_bedside_html():
    """Reads bedside_monitor.html from disk so HTML/CSS edits take effect on refresh."""
    if os.path.exists(BEDSIDE_HTML_FILE):
        with open(BEDSIDE_HTML_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "<!DOCTYPE html><html><body><h3>bedside_monitor.html not found</h3></body></html>"


def load_bedside_sim_html():
    """Reads bedside_simulation.html from disk so HTML/CSS edits take effect on refresh."""
    if os.path.exists(BEDSIDE_SIM_HTML_FILE):
        with open(BEDSIDE_SIM_HTML_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "<!DOCTYPE html><html><body><h3>bedside_simulation.html not found</h3></body></html>"


def default_bed_chaos():
    return {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False, "removed": False, "condition": "normal"},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "removed": False, "target_spo2": 98},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "removed": False, "target_hr": 75},
        "Agent_CO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "removed": False, "target_co2": 38},
        "Agent_Respiration": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "removed": False},
        "Agent_EtCO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "removed": False},
        "Agent_IBP": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "removed": False, "target_sys": 120, "target_dia": 80},
        "Agent_NIBP": {"delay_ms": 15, "drop_rate": 0.0, "malicious": False, "removed": False, "target_sys": 120, "target_dia": 80},
        "Agent_Temperature": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "removed": False},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False},
        "vitals": {
            "target_hr": 75,
            "target_spo2": 98,
            "target_co2": 38,
            "target_sys": 120,
            "target_dia": 80,
            "target_respiration": 16,
            "target_etco2": 38,
            "ecg_condition": "normal",
        },
    }


class BedsideAuthController:
    """Own the login and durable registration of exactly one local unit."""

    def __init__(self, brain, web_port):
        self.brain = brain
        self.web_port = web_port
        self.lock = threading.Lock()
        self.registered = False
        self.sessions = set()
        directory = Path(os.environ.get("BEDSIDE_STATE_DIR") or Path.home() / ".local" / "state" / "nurse-station")
        self.record = directory / f"{brain.patient_id}.json"

    def restore(self):
        if self.record.exists():
            self.activate(json.loads(self.record.read_text(encoding="utf-8")))

    def activate(self, record):
        self.password_salt = bytes.fromhex(record["salt"])
        self.password_hash = bytes.fromhex(record["hash"])
        self.brain.start(record["profile"], record["registration_id"])
        self.registered = True

    def _hash_password(self, password, salt):
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)

    def register(self, payload):
        password = payload.get("password")
        if not isinstance(password, str) or len(password) < 8:
            raise ValueError("Use a bedside password of at least 8 characters.")
        if payload.get("patient_id", self.brain.patient_id) != self.brain.patient_id:
            raise ValueError("The device ID is fixed by the launcher.")
        profile = {}
        for key in ("department", "room", "patient_name", "diagnosis"):
            value = payload.get(key, "")
            if not isinstance(value, str) or len(value) > 256:
                raise ValueError(f"Enter a valid {key.replace('_', ' ')}.")
            profile[key] = value.strip()
        if profile["department"] not in dict(DEPARTMENTS) or not profile["room"] or not profile["patient_name"]:
            raise ValueError("Enter the patient, room and department.")
        salt = secrets.token_bytes(16)
        record = {"profile": profile, "registration_id": secrets.token_hex(16), "salt": salt.hex(),
                  "hash": self._hash_password(password, salt).hex()}
        with self.lock:
            if self.registered:
                raise ValueError("This device is already registered.")
            self.record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.record.parent, delete=False) as output:
                temporary = Path(output.name)
                json.dump(record, output)
            try:
                os.link(temporary, self.record)
            except FileExistsError:
                raise ValueError("This device is already registered.") from None
            finally:
                temporary.unlink(missing_ok=True)
            self.activate(record)
            token = secrets.token_urlsafe(32)
            self.sessions.add(token)
            return token

    def login(self, password):
        with self.lock:
            if not self.registered:
                raise ValueError("This bedside unit has not been registered yet.")
            candidate = self._hash_password(str(password or ""), self.password_salt)
            if not hmac.compare_digest(candidate, self.password_hash):
                raise ValueError("Incorrect bedside password.")
            token = secrets.token_urlsafe(32)
            self.sessions.add(token)
            return token

    def logout(self, token=None):
        with self.lock:
            self.sessions.discard(token)

    def is_authenticated(self, token):
        if not token:
            return False
        with self.lock:
            return self.registered and token in self.sessions

    def status_payload(self, token):
        with self.lock:
            reg = self.registered
            authed = reg and bool(token) and (token in self.sessions)
        return {
            "registered": reg,
            "authenticated": authed,
            "web_port": self.web_port,
            "profile": self.brain.get_profile() if authed else {"patient_id": self.brain.patient_id},
        }


def make_bedside_handler(brain, auth_ctrl):
    class BedsideMonitorHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _get_session_token(self):
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header.removeprefix("Bearer ").strip()
            return None

        def _send_json(self, obj, code=200):
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def local_request(self):
            host = self.headers.get("Host", "")
            allowed = {f"localhost:{auth_ctrl.web_port}", f"127.0.0.1:{auth_ctrl.web_port}"}
            origin = self.headers.get("Origin")
            if host not in allowed or (origin and origin != f"http://{host}"):
                self._send_json({"error": "Local bedside access only"}, code=403)
                return False
            query = parse_qs(urlsplit(self.path).query)
            if any(value != brain.patient_id for key in ("bed", "patient_id") for value in query.get(key, [])):
                self._send_json({"error": "Unknown bed"}, code=404)
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path_only = self.path.split("?", 1)[0]
            token = self._get_session_token()

            if path_only == "/bedside-api.js":
                body = Path(BASE_DIR, "bedside-api.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path_only == "/api/auth/status":
                self._send_json(auth_ctrl.status_payload(token))
                return

            if path_only == "/api/state":
                if not auth_ctrl.is_authenticated(token):
                    self._send_json({"error": "Authentication required", "auth": auth_ctrl.status_payload(token)}, code=401)
                    return

                pid = brain.patient_id
                chaos = brain.get_config()
                sensors_snap = brain.get_sensor_snapshot()
                with brain.lock:
                    bed_state = {
                        "patient_id": pid,
                        "profile": brain.get_profile(),
                        "hr": None if sensors_snap["Agent_HR"]["removed"] else brain.latest_hr,
                        "spo2": None if sensors_snap["Agent_SpO2"]["removed"] else brain.latest_spo2,
                        "co2": None if sensors_snap["Agent_EtCO2"]["removed"] else brain.latest_co2,
                        "ibp": None if sensors_snap["Agent_IBP"]["removed"] else brain.latest_ibp,
                        "nibp": None if sensors_snap["Agent_NIBP"]["removed"] else brain.latest_nibp,
                        "temperature_c": None if sensors_snap["Agent_Temperature"]["removed"] else brain.latest_sensors["Agent_Temperature"]["value"],
                        "respiration_rate": None if sensors_snap["Agent_Respiration"]["removed"] else brain.latest_sensors["Agent_Respiration"]["value"],
                        "etco2_mmhg": None if sensors_snap["Agent_EtCO2"]["removed"] else brain.latest_sensors["Agent_EtCO2"]["value"],
                        "hr_valid": brain.hr_plausible and not sensors_snap["Agent_HR"]["removed"],
                        "spo2_valid": brain.spo2_plausible and not sensors_snap["Agent_SpO2"]["removed"],
                        "co2_valid": brain.co2_plausible and not sensors_snap["Agent_EtCO2"]["removed"],
                        "ibp_valid": brain.ibp_plausible and not sensors_snap["Agent_IBP"]["removed"],
                        "nibp_valid": brain.nibp_plausible and not sensors_snap["Agent_NIBP"]["removed"],
                        "anomaly_prob": 0.0 if sensors_snap["Agent_ECG"]["removed"] else round(brain.last_prob, 4),
                        "arrhythmia": False if sensors_snap["Agent_ECG"]["removed"] else brain.last_arrhythmia,
                        "inference_ms": round(brain.last_inf_ms, 2),
                        "offloaded": brain.last_offloaded,
                        "relay_b": brain.last_relay_b,
                        "packets": brain.packets_processed,
                        "ecg_wave": [] if sensors_snap["Agent_ECG"]["removed"] else (brain.latest_ecg_wave[-250:] if brain.latest_ecg_wave else []),
                        "sensors": sensors_snap,
                        "sensor_links": copy.deepcopy(brain.sensor_links),
                        "chaos": chaos,
                    }
                self._send_json(
                    {
                        "broker": f"{brain.broker}:{brain.port}",
                        "provider": brain.active_provider,
                        "server_time": time.time(),
                        "patient_id": pid,
                        "profile": brain.get_profile(),
                        "bed": bed_state,
                        "beds": {pid: bed_state},
                    }
                )
            elif path_only == "/api/chaos":
                if not auth_ctrl.is_authenticated(token):
                    self._send_json({"error": "Authentication required"}, code=401)
                    return
                self._send_json(brain.get_config())
            elif path_only in ("/bedside_simulation.html", "/bedside_simulation", "/simulation"):
                body = load_bedside_sim_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path_only in ("/", "/bedside_monitor.html"):
                body = load_bedside_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            else:
                self.send_error(404)

        def do_POST(self):
            if not self.local_request():
                return
            if self.headers.get_content_type() != "application/json":
                self._send_json({"error": "Expected application/json"}, code=415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 65536:
                    self._send_json({"error": "Invalid request size"}, code=413)
                    return
                raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Expected an object")
            except Exception:
                self._send_json({"ok": False, "error": "Invalid JSON"}, code=400)
                return

            path_only = self.path.split("?", 1)[0]
            token = self._get_session_token()

            if data.get("patient_id", brain.patient_id) != brain.patient_id:
                self._send_json({"error": "Unknown bed"}, code=404)
                return

            if path_only == "/api/auth/register":
                try:
                    new_token = auth_ctrl.register(data)
                    self._send_json(
                        {"ok": True, "token": new_token, **auth_ctrl.status_payload(new_token)},
                    )
                except ValueError as err:
                    self._send_json({"ok": False, "error": str(err)}, code=400)
                return

            if path_only == "/api/auth/login":
                try:
                    new_token = auth_ctrl.login(data.get("password", ""))
                    self._send_json(
                        {"ok": True, "token": new_token, **auth_ctrl.status_payload(new_token)},
                    )
                except ValueError as err:
                    self._send_json({"ok": False, "error": str(err)}, code=401)
                return

            if path_only == "/api/auth/logout":
                auth_ctrl.logout(token)
                self._send_json({"ok": True, **auth_ctrl.status_payload(None)})
                return

            if not auth_ctrl.is_authenticated(token):
                self._send_json({"ok": False, "error": "Authentication required"}, code=401)
                return

            try:
                with brain.lock:
                    self.update_simulation(path_only, data)
            except (TypeError, ValueError, OverflowError):
                self._send_json({"error": "Invalid simulation settings"}, code=400)

        def update_simulation(self, path_only, data):
            if path_only == "/api/chaos/reset":
                clean = default_bed_chaos()
                brain.set_config(clean)
                self._send_json({"ok": True, "chaos": clean})
            elif path_only == "/api/chaos":
                chaos_cfg = data.get("chaos", default_bed_chaos())
                if not isinstance(chaos_cfg, dict) or not all(isinstance(v, dict) for v in chaos_cfg.values()):
                    raise ValueError("Expected sensor settings")
                brain.set_config(chaos_cfg)
                self._send_json({"ok": True, "chaos": chaos_cfg})
            elif path_only in ("/api/manipulate", "/api/vitals"):
                if data.get("reset"):
                    clean = default_bed_chaos()
                    brain.set_config(clean)
                    self._send_json({"ok": True, "chaos": clean})
                    return

                chaos = brain.get_config()
                if "vitals" not in chaos:
                    chaos["vitals"] = {}

                if "target_hr" in data:
                    chaos["vitals"]["target_hr"] = int(data["target_hr"])
                    chaos.setdefault("Agent_HR", {})["target_hr"] = int(data["target_hr"])
                if "target_spo2" in data:
                    chaos["vitals"]["target_spo2"] = int(data["target_spo2"])
                    chaos.setdefault("Agent_SpO2", {})["target_spo2"] = int(data["target_spo2"])
                if "target_co2" in data:
                    chaos["vitals"]["target_co2"] = int(data["target_co2"])
                    chaos.setdefault("Agent_CO2", {})["target_co2"] = int(data["target_co2"])
                    chaos["vitals"]["target_etco2"] = int(data["target_co2"])
                    chaos.setdefault("Agent_EtCO2", {})["target_etco2"] = int(data["target_co2"])
                if "target_respiration" in data:
                    chaos["vitals"]["target_respiration"] = int(data["target_respiration"])
                if "target_sys" in data:
                    chaos["vitals"]["target_sys"] = int(data["target_sys"])
                    chaos.setdefault("Agent_IBP", {})["target_sys"] = int(data["target_sys"])
                    chaos.setdefault("Agent_NIBP", {})["target_sys"] = int(data["target_sys"])
                if "target_dia" in data:
                    chaos["vitals"]["target_dia"] = int(data["target_dia"])
                    chaos.setdefault("Agent_IBP", {})["target_dia"] = int(data["target_dia"])
                    chaos.setdefault("Agent_NIBP", {})["target_dia"] = int(data["target_dia"])
                if "ecg_condition" in data:
                    chaos["vitals"]["ecg_condition"] = str(data["ecg_condition"])
                    chaos.setdefault("Agent_ECG", {})["condition"] = str(data["ecg_condition"])

                all_agents = (*SENSOR_AGENT_IDS, "Agent_CO2")
                if "delay_ms" in data:
                    d_ms = float(data["delay_ms"])
                    for aid in all_agents:
                        chaos.setdefault(aid, {})["delay_ms"] = d_ms
                if "drop_rate" in data:
                    dr = float(data["drop_rate"])
                    for aid in all_agents:
                        chaos.setdefault(aid, {})["drop_rate"] = dr
                if "malicious" in data:
                    m = bool(data["malicious"])
                    for aid in all_agents:
                        chaos.setdefault(aid, {})["malicious"] = m
                if "sensor_faults" in data and isinstance(data["sensor_faults"], dict):
                    for aid, is_mal in data["sensor_faults"].items():
                        if aid in all_agents:
                            chaos.setdefault(aid, {})["malicious"] = bool(is_mal)
                            if aid == "Agent_EtCO2":
                                chaos.setdefault("Agent_CO2", {})["malicious"] = bool(is_mal)
                            elif aid == "Agent_CO2":
                                chaos.setdefault("Agent_EtCO2", {})["malicious"] = bool(is_mal)
                if "removed_sensors" in data and isinstance(data["removed_sensors"], dict):
                    for aid, is_rem in data["removed_sensors"].items():
                        if aid in all_agents:
                            chaos.setdefault(aid, {})["removed"] = bool(is_rem)
                            if aid == "Agent_EtCO2":
                                chaos.setdefault("Agent_CO2", {})["removed"] = bool(is_rem)
                            elif aid == "Agent_CO2":
                                chaos.setdefault("Agent_EtCO2", {})["removed"] = bool(is_rem)
                if "overload_gpu" in data:
                    ov = bool(data["overload_gpu"])
                    chaos.setdefault("Orin_Link", {})["overload_gpu"] = ov
                    chaos.setdefault("Orin_Link", {})["use_relay_b"] = ov

                brain.set_config(chaos)
                self._send_json({"ok": True, "chaos": chaos})
            else:
                self._send_json({"ok": False}, code=404)

    return BedsideMonitorHandler


def start_bedside_web_server(brain, web_port=8080, web_host="127.0.0.1"):
    auth_ctrl = BedsideAuthController(brain, web_port)
    handler_cls = make_bedside_handler(brain, auth_ctrl)
    try:
        server = ThreadingHTTPServer((web_host, web_port), handler_cls)
        auth_ctrl.restore()
        server.daemon_threads = True
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(
            f"[BEDSIDE MONITOR:{brain.patient_id}] Isolated Bedside UI live at http://localhost:{web_port} "
            f"(bound to {web_host}:{web_port})"
        )
        return server
    except Exception as e:
        print(f"[BEDSIDE MONITOR:{brain.patient_id}] Notice: Could not bind {web_host}:{web_port} ({e})")
        return None


class JetsonEdgeBrain:
    def __init__(self, patient_id, broker, port, session, active_provider, web_port=8080):
        self.patient_id = patient_id
        self.broker = broker
        self.port = port
        self.web_port = web_port
        self.session = session
        self.active_provider = active_provider
        self.input_name = self.session.get_inputs()[0].name if self.session else "ecg_input"

        self.department = self.room = self.patient_name = self.diagnosis = ""
        self.registration_id = None
        self.config = default_bed_chaos()
        self.workers = []

        self.latest_hr = None
        self.latest_spo2 = None
        self.latest_co2 = None
        self.latest_ibp = None
        self.latest_nibp = None
        self.latest_ecg_var = 0.04
        self.hr_plausible = False
        self.spo2_plausible = False
        self.co2_plausible = False
        self.ibp_plausible = False
        self.nibp_plausible = False
        self.latest_ecg_wave = []
        self.last_prob = 0.02
        self.last_arrhythmia = False
        self.last_inf_ms = 0.25
        self.last_offloaded = False
        self.last_relay_b = False
        self.packets_processed = 0
        self.force_offload = False
        self.local_bed_alarm = False
        self.sensor_conflict = False
        self.stop_event = threading.Event()

        self.sensor_links = {
            agent_id: {
                "latency_ms": 10.0,
                "loss_pct": 0.0,
                "last_seq": None,
                "last_rx_time": None,
                "rx": 0,
                "lost": 0,
            }
            for agent_id in SENSOR_AGENT_IDS
        }
        self.trust_trackers = {agent_id: AgentTrustTracker(agent_id) for agent_id in SENSOR_AGENT_IDS}
        self.latest_sensors = {
            agent_id: {"value": None, "reading": {}, "waveform": [], "seq": None, "sent_ts": None, "valid": False}
            for agent_id in SENSOR_AGENT_IDS
        }
        self.sensor_links["Agent_CO2"] = self.sensor_links["Agent_EtCO2"]
        self.lock = threading.RLock()
        self.client = mqtt_client(f"Jetson-{patient_id}", broker, port)

    def get_profile(self):
        with self.lock:
            return {
                "patient_id": self.patient_id,
                "bed_label": self.patient_id.replace("_", " "),
                "department": self.department,
                "room": self.room,
                "patient_name": self.patient_name,
                "diagnosis": self.diagnosis,
                "broker": self.broker,
                "web_port": self.web_port,
            }

    def get_config(self):
        with self.lock:
            return copy.deepcopy(self.config)

    def set_config(self, config):
        with self.lock:
            self.config = copy.deepcopy(config)

    def start(self, profile, registration_id):
        with self.lock:
            self.department = profile["department"]
            self.room = profile["room"]
            self.patient_name = profile["patient_name"]
            self.diagnosis = profile["diagnosis"]
            self.registration_id = registration_id
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect_async(self.broker, self.port, keepalive=60)
        self.client.loop_start()
        simulation = PatientSimulation(self.patient_id)
        self.workers = [SensorAgentThread(self.patient_id, aid, simulation, self.get_config,
                                          self.deliver_sample, self.stop_event) for aid in SENSOR_AGENT_IDS]
        self.workers.append(threading.Thread(target=self._run_summary_loop, daemon=True))
        for worker in self.workers:
            worker.start()

    def stop(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=2)
        if self.registration_id:
            self.client.disconnect()
            self.client.loop_stop()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            print(f"[{self.patient_id}] Mosquitto rejected the connection: {rc}", flush=True)
            return
        client.subscribe(f"hospital/{self.patient_id}/control/#")
        self.send_discovery_announcement()

    def send_discovery_announcement(self):
        self.client.publish(f"hospital/{self.patient_id}/registration", json.dumps({
            "patient_id": self.patient_id, "registration_id": self.registration_id,
            "department": self.department, "room": self.room, "patient_name": self.patient_name,
            "diagnosis": self.diagnosis, "exec_provider": self.active_provider,
        }), qos=1)

    def deliver_sample(self, packet):
        self.process_sensor(packet)
        if self.client.is_connected():
            topic = f"hospital/{self.patient_id}/sensors/{packet['agent_id']}"
            self.client.publish(topic, json.dumps(packet))

    def _track_link(self, agent_id, payload, now):
        if agent_id not in self.sensor_links:
            return
        info = self.sensor_links[agent_id]
        sent_ts = float(payload.get("sent_ts", now))
        lat_ms = max(0.5, (now - sent_ts) * 1000.0)
        info["latency_ms"] = round(0.7 * info["latency_ms"] + 0.3 * lat_ms, 1)

        seq = int(payload.get("seq", 0))
        if info["last_seq"] is not None and seq > info["last_seq"]:
            gap = seq - info["last_seq"] - 1
            if gap > 0:
                info["lost"] += gap
        info["last_seq"] = seq
        info["last_rx_time"] = now
        info["rx"] += 1
        total = max(1, info["rx"] + info["lost"])
        info["loss_pct"] = round((info["lost"] / total) * 100.0, 1)

    def get_sensor_snapshot(self):
        with self.lock:
            bed_chaos = self.config
            snapshot = {}
            for agent_id, sensor in self.latest_sensors.items():
                is_removed = bool(bed_chaos.get(agent_id, {}).get("removed", False))
                if agent_id == "Agent_EtCO2" and not is_removed:
                    is_removed = bool(bed_chaos.get("Agent_CO2", {}).get("removed", False))

                if is_removed:
                    snapshot[agent_id] = {
                        "value": None,
                        "reading": {},
                        "waveform": [],
                        "seq": sensor.get("seq"),
                        "sent_ts": sensor.get("sent_ts"),
                        "valid": False,
                        "removed": True,
                        "trust_score": 0.0,
                        "signal_quality": 0,
                        "connection_status": "REMOVED",
                        "reliability_level": "REMOVED",
                        "sensor_health": "REMOVED",
                        "age_ms": None,
                    }
                    continue

                tracker = self.trust_trackers[agent_id]
                reliability = tracker.evaluate_status()
                connected = reliability != "DISCONNECTED"
                valid = bool(tracker.validity_window and tracker.validity_window[-1] > 0)
                quality_score = min(tracker.s_l, tracker.s_j, tracker.s_p, tracker.s_d) if connected and valid else 0.0
                signal_quality = round(float(np.clip(quality_score, 0.0, 1.0)) * 100)
                snapshot[agent_id] = {
                    **sensor,
                    "removed": False,
                    "trust_score": round(tracker.trust_score * 100, 1),
                    "signal_quality": signal_quality,
                    "connection_status": "CONNECTED" if connected else "DISCONNECTED",
                    "reliability_level": reliability,
                    "sensor_health": "HEALTHY" if connected and valid else "DEGRADED" if connected else "DISCONNECTED",
                    "age_ms": round((time.time() - tracker.last_rx_time) * 1000) if tracker.last_rx_time else None,
                }
            return snapshot

    def _run_inference(self, ecg_wave, overload_gpu=False):
        t0 = time.perf_counter()
        x = np.array(ecg_wave, dtype=np.float32).reshape(1, 1, 250)

        if self.session is not None:
            try:
                logits = self.session.run(None, {self.input_name: x})[0]
                exp_scores = np.exp(logits - np.max(logits))
                probs = exp_scores / np.sum(exp_scores)
                anomaly_prob = float(probs[0][1])
            except Exception:
                peak_to_peak = float(np.ptp(x))
                anomaly_prob = 0.95 if peak_to_peak > 2.2 else 0.02
        else:
            peak_to_peak = float(np.ptp(x))
            anomaly_prob = 0.95 if peak_to_peak > 2.2 else 0.02

        if overload_gpu:
            time.sleep(0.065)

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return anomaly_prob, inference_ms

    def _on_message(self, client, userdata, msg):
        try:
            now = time.time()
            topic = msg.topic
            payload = json.loads(msg.payload.decode("utf-8"))

            if topic == f"hospital/{self.patient_id}/control/probe":
                self.send_discovery_announcement()
                return
            if topic == f"hospital/{self.patient_id}/control/admit":
                with self.lock:
                    if payload.get("department"):
                        self.department = str(payload["department"])
                    if payload.get("room"):
                        self.room = str(payload["room"])
                print(f"[EDGE BRAIN:{self.patient_id}] Node officially admitted by Nurse Station to {self.department} ({self.room}).")
                return

            if topic == f"hospital/{self.patient_id}/control/offload":
                with self.lock:
                    self.force_offload = bool(payload.get("offload", False))
                return

        except (TypeError, ValueError) as error:
            print(f"[{self.patient_id}] Invalid control message: {error}")

    def process_sensor(self, payload):
        try:
            now = time.time()
            agent_id = payload["agent_id"]
            bed_chaos = self.get_config()
            if bool(bed_chaos.get(agent_id, {}).get("removed", False)):
                return

            payload = {**payload, "agent_id": agent_id}

            wave_key = WAVEFORM_KEYS.get(agent_id)
            waveform = payload.get(wave_key, []) if wave_key else []
            if not isinstance(waveform, list):
                waveform = []
            reading = payload.get("reading", {})
            if not isinstance(reading, dict):
                reading = {}
            if agent_id in {"Agent_IBP", "Agent_NIBP"} and not reading:
                pressure = payload.get("value", {})
                if isinstance(pressure, dict):
                    reading = {
                        "systolic": pressure.get("systolic", pressure.get("sys")),
                        "diastolic": pressure.get("diastolic", pressure.get("dia")),
                        "map": pressure.get("map"),
                    }
            payload["reading"] = reading
            with self.lock:
                self._track_link(agent_id, payload, now)
                tracker = self.trust_trackers[agent_id]
                tracker.update(payload=payload, recv_ts=now, cross_sensor_conflict=self.sensor_conflict)
                self.latest_sensors[agent_id] = {
                    "value": payload.get("value"),
                    "reading": reading,
                    "waveform": waveform,
                    "seq": payload.get("seq"),
                    "sent_ts": payload.get("sent_ts"),
                    "waveform_end_ts": payload.get("waveform_end_ts", payload.get("sent_ts")),
                    "sample_rate_hz": payload.get("sample_rate_hz", SAMPLE_RATE),
                    "valid": bool(tracker.validity_window and tracker.validity_window[-1] > 0),
                }

                if agent_id == "Agent_HR":
                    hr_val = int(payload.get("value", 75))
                    self.latest_hr = hr_val
                    self.hr_plausible = 35 <= hr_val <= 210
                elif agent_id == "Agent_SpO2":
                    spo2_val = int(payload.get("value", 98))
                    self.latest_spo2 = spo2_val
                    self.spo2_plausible = 65 <= spo2_val <= 100
                elif agent_id == "Agent_EtCO2":
                    co2_value = reading.get("etco2_mmhg", payload.get("value", 38))
                    if isinstance(co2_value, dict):
                        co2_value = co2_value.get("value", co2_value.get("etco2_mmhg", 38))
                    self.latest_co2 = int(co2_value)
                    self.co2_plausible = 15 <= self.latest_co2 <= 75
                elif agent_id in {"Agent_IBP", "Agent_NIBP"}:
                    pressure = reading or payload.get("value", {})
                    if not isinstance(pressure, dict):
                        pressure = {}
                    systolic = int(pressure.get("systolic", pressure.get("sys", 0)))
                    diastolic = int(pressure.get("diastolic", pressure.get("dia", 0)))
                    mean = int(pressure.get("map", round((systolic + 2 * diastolic) / 3)))
                    normalized = {"systolic": systolic, "diastolic": diastolic, "map": mean}
                    legacy = {
                        "sys": systolic,
                        "dia": diastolic,
                        "map": mean,
                        "formatted": pressure.get("formatted", f"{systolic}/{diastolic} ({mean})"),
                    }
                    pressure_valid = 40 <= diastolic < systolic <= 250 and diastolic <= mean <= systolic
                    if agent_id == "Agent_IBP":
                        self.latest_ibp = legacy
                        self.ibp_plausible = pressure_valid
                        self.latest_sensors[agent_id]["reading"] = normalized
                    else:
                        self.latest_nibp = legacy
                        self.nibp_plausible = pressure_valid
                        self.latest_sensors[agent_id]["reading"] = normalized

            if agent_id in {"Agent_HR", "Agent_SpO2"}:
                self._check_plausibility()
                return

            if agent_id in {"Agent_Respiration", "Agent_EtCO2", "Agent_IBP", "Agent_NIBP"}:
                self._check_plausibility()
                return

            if agent_id == "Agent_ECG":
                ecg_wave = payload.get("ecg_wave", [])
                if len(ecg_wave) != 250:
                    return

                orin_chaos = bed_chaos.get("Orin_Link", {})
                overload_gpu = bool(orin_chaos.get("overload_gpu", False)) or self.force_offload
                use_relay_b = bool(orin_chaos.get("use_relay_b", False)) or overload_gpu

                anomaly_prob, inference_ms = self._run_inference(ecg_wave, overload_gpu=overload_gpu)
                var_val = float(np.var(ecg_wave))
                ptp_val = float(np.ptp(ecg_wave))
                is_arrhythmia = (var_val > 0.18 or ptp_val > 2.2) and (anomaly_prob >= 0.50 or var_val > 0.25)

                with self.lock:
                    self.packets_processed += 1
                    self.latest_ecg_wave = ecg_wave
                    self.latest_ecg_var = var_val
                    self.last_prob = anomaly_prob
                    self.last_arrhythmia = is_arrhythmia
                    self.last_inf_ms = inference_ms
                    self.last_offloaded = overload_gpu
                    self.last_relay_b = use_relay_b
                    self.local_bed_alarm = is_arrhythmia

                self._check_plausibility()

                if is_arrhythmia:
                    sub_route = "relay_b" if use_relay_b else "primary"
                    alert_topic = f"hospital/{self.patient_id}/alerts/{sub_route}"
                    alert_payload = {
                        "patient_id": self.patient_id,
                        "alert": "CRITICAL_ARRHYTHMIA_DETECTED",
                        "arrhythmia_prob": round(anomaly_prob, 4),
                        "inf_ms": round(inference_ms, 2),
                        "ecg_variance": round(var_val, 4),
                        "timestamp": time.time(),
                        "route": sub_route,
                    }
                    try:
                        self.client.publish(alert_topic, json.dumps(alert_payload), qos=1)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[EDGE BRAIN:{self.patient_id}] Message error: {e}")

    def _check_plausibility(self):
        with self.lock:
            hr = self.latest_hr
            spo2 = self.latest_spo2
            co2 = self.latest_co2
            ibp = self.latest_ibp
            arrhythmia = self.last_arrhythmia

        conflict = False
        if hr is not None and spo2 is not None:
            if not arrhythmia and spo2 >= 95 and (hr <= 0 or hr >= 240):
                conflict = True
            if spo2 <= 30 and hr > 60 and not arrhythmia:
                conflict = True
        if spo2 is not None and (spo2 < 50 or spo2 > 105):
            conflict = True
        if hr is not None and (hr < 30 or hr > 280):
            conflict = True
        if co2 is not None and (co2 < 10 or co2 > 85):
            conflict = True
        if isinstance(ibp, dict):
            s_v = int(ibp.get("sys", 120))
            d_v = int(ibp.get("dia", 80))
            if s_v <= d_v or s_v < 40 or s_v > 260:
                conflict = True

        with self.lock:
            self.sensor_conflict = conflict

    def _run_summary_loop(self):
        loop_cnt = 0
        while not self.stop_event.is_set():
            if self.stop_event.wait(0.40):
                break
            loop_cnt += 1
            if not self.client.is_connected():
                continue
            if loop_cnt % 10 == 0:
                self.send_discovery_announcement()

            sensors_snap = self.get_sensor_snapshot()
            with self.lock:
                summary = {
                    "patient_id": self.patient_id,
                    "inf_ms": round(self.last_inf_ms, 2),
                    "exec_provider": self.active_provider,
                    "arrhythmia_prob": 0.0 if sensors_snap["Agent_ECG"]["removed"] else round(self.last_prob, 4),
                    "local_bed_alarm": False if sensors_snap["Agent_ECG"]["removed"] else self.local_bed_alarm,
                    "sensor_conflict": self.sensor_conflict,
                    "latest_vitals": {
                        "hr": None if sensors_snap["Agent_HR"]["removed"] else self.latest_hr,
                        "spo2": None if sensors_snap["Agent_SpO2"]["removed"] else self.latest_spo2,
                        "co2": None if sensors_snap["Agent_EtCO2"]["removed"] else self.latest_co2,
                        "ibp": None if sensors_snap["Agent_IBP"]["removed"] else self.latest_ibp,
                        "nibp": None if sensors_snap["Agent_NIBP"]["removed"] else self.latest_nibp,
                        "respiration_rate": None if sensors_snap["Agent_Respiration"]["removed"] else self.latest_sensors["Agent_Respiration"]["value"],
                        "etco2_mmhg": None if sensors_snap["Agent_EtCO2"]["removed"] else self.latest_sensors["Agent_EtCO2"]["value"],
                        "temperature_c": None if sensors_snap["Agent_Temperature"]["removed"] else self.latest_sensors["Agent_Temperature"]["value"],
                        "ecg_variance": round(self.latest_ecg_var, 4),
                    },
                    "sensors": sensors_snap,
                    "ecg_wave": [] if sensors_snap["Agent_ECG"]["removed"] else self.latest_ecg_wave,
                    "routing_path": "relay_b" if self.last_relay_b else "primary",
                    "timestamp": time.time(),
                }

            try:
                self.client.publish(
                    f"hospital/{self.patient_id}/edge/summary",
                    json.dumps(summary),
                    qos=0,
                )
            except Exception:
                pass


def create_shared_onnx_session(model_path):
    if ort is None or not os.path.exists(model_path):
        return None, "CPUExecutionProvider"

    available = ort.get_available_providers()
    providers = []
    if "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    chosen_provider = session.get_providers()[0]
    return session, chosen_provider


def main():
    parser = argparse.ArgumentParser(description="Bedside GPU Edge Brain & Isolated Local Bedside Monitor")
    parser.add_argument("--patient-id", type=str, default="Bed_1", help="Primary Patient ID")
    parser.add_argument("--broker", type=str, default=broker_host(), help="Mosquitto hostname")
    parser.add_argument("--port", type=int, default=None, help="Mosquitto port")
    parser.add_argument("--web-port", type=int, default=8080, help="Base Local Bedside Monitor Web UI port")
    parser.add_argument(
        "--web-host",
        type=str,
        default="127.0.0.1",
        help="Bind address for bedside web UI (defaults to 127.0.0.1 so other Wi-Fi devices cannot view this bedside)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to ecg_anomaly.onnx")
    args = parser.parse_args()

    if args.web_host != "127.0.0.1":
        parser.error("Bedside pages are available only on local loopback.")
    args.port = args.port or broker_port(args.broker)
    if not valid_device_id(args.patient_id):
        parser.error("Choose a unique device ID using letters, numbers, underscores or hyphens.")
    session, active_provider = create_shared_onnx_session(args.model)
    brain = JetsonEdgeBrain(args.patient_id, args.broker, args.port, session, active_provider, args.web_port)
    server = start_bedside_web_server(brain, web_port=args.web_port, web_host=args.web_host)
    if server is None:
        raise SystemExit(f"Could not start {args.patient_id} on port {args.web_port}")
    print(f"[{args.patient_id}] Mosquitto {args.broker}:{args.port} · {active_provider}", flush=True)

    stop_event = threading.Event()

    def handle_sig(sig, frame):
        print("\n[INFO] Gracefully shutting down Bedside Edge Brain...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while not stop_event.is_set():
        time.sleep(0.5)

    server.shutdown()
    server.server_close()
    brain.stop()


if __name__ == "__main__":
    main()
