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
import random
import hashlib
import secrets
import argparse
import threading
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import paho.mqtt.client as mqtt
from patient_simulation import SAMPLE_RATE, SENSOR_AGENT_IDS, WAVEFORM_KEYS
from trust_evaluator import AgentTrustTracker

try:
    import onnxruntime as ort
except ImportError:
    ort = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAOS_FILE = os.path.join(BASE_DIR, "chaos.json")
DEFAULT_MODEL = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
BEDSIDE_HTML_FILE = os.path.join(BASE_DIR, "bedside_monitor.html")
BEDSIDE_SIM_HTML_FILE = os.path.join(BASE_DIR, "bedside_simulation.html")
CHAOS_LOCK = threading.Lock()

DEFAULT_BED_PROFILES = {
    "Bed_1": {
        "department": "ICU",
        "room": "Room 101",
        "patient_name": "Eleanor Vance (F, 64)",
        "diagnosis": "Acute Myocardial Infarction / Post-CABG",
        "wifi_ssid": "Hospital-CriticalCare-5G",
    },
    "Bed_2": {
        "department": "CCU",
        "room": "Room 201",
        "patient_name": "Arthur Pendelton (M, 71)",
        "diagnosis": "Severe ARDS & Sepsis Protocol",
        "wifi_ssid": "Hospital-CriticalCare-5G",
    },
    "Bed_3": {
        "department": "NCU",
        "room": "Room 301",
        "patient_name": "Sarah Jenkins (F, 42)",
        "diagnosis": "Traumatic Brain Injury / ICP Monitor",
        "wifi_ssid": "Hospital-CriticalCare-5G",
    },
}


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


def read_patient_chaos(patient_id="Bed_1"):
    if not os.path.exists(CHAOS_FILE):
        return default_bed_chaos()
    try:
        with CHAOS_LOCK:
            with open(CHAOS_FILE, "r") as f:
                data = json.load(f)
        if patient_id in data and isinstance(data[patient_id], dict):
            base = default_bed_chaos()
            for k, v in data[patient_id].items():
                if k in base and isinstance(v, dict):
                    base[k].update(v)
                elif k in base:
                    base[k] = v
            return base
        if "Agent_ECG" in data:
            base = default_bed_chaos()
            for k, v in data.items():
                if k in base and isinstance(v, dict):
                    base[k].update(v)
                elif k in base:
                    base[k] = v
            return base
        return default_bed_chaos()
    except Exception:
        return default_bed_chaos()


def write_patient_chaos(patient_id, bed_chaos):
    try:
        with CHAOS_LOCK:
            current = {}
            if os.path.exists(CHAOS_FILE):
                try:
                    with open(CHAOS_FILE, "r") as f:
                        current = json.load(f)
                except Exception:
                    current = {}
            if "Agent_ECG" in current:
                current = {"Bed_1": current}
            current[patient_id] = bed_chaos
            tmp_path = CHAOS_FILE + f".tmp.{random.randint(1000, 9999)}"
            with open(tmp_path, "w") as f:
                json.dump(current, f, indent=2)
            os.replace(tmp_path, CHAOS_FILE)
        return True
    except Exception:
        return False


def update_chaos_file(new_data, patient_id=None):
    if patient_id:
        cfg = read_patient_chaos(patient_id)
        for k, v in new_data.items():
            if k in cfg and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        write_patient_chaos(patient_id, cfg)
    else:
        for pid in ["Bed_1", "Bed_2", "Bed_3"]:
            cfg = read_patient_chaos(pid)
            for k, v in new_data.items():
                if k in cfg and isinstance(v, dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            write_patient_chaos(pid, cfg)


class BedsideAuthController:
    """Manages per-bedside registration (Wi-Fi, Bed ID, Ward, Room, Password) and session tokens."""

    def __init__(self, brain, web_port):
        self.brain = brain
        self.web_port = web_port
        self.cookie_name = f"bedside_session_{web_port}"
        self.lock = threading.Lock()
        self.registered = False
        self.password_salt = b""
        self.password_hash = b""
        self.sessions = set()

    def _hash_password(self, password, salt):
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)

    def register(self, payload):
        password = str(payload.get("password") or "").strip()
        if len(password) < 3:
            raise ValueError("Password must be at least 3 characters.")

        raw_bed = str(payload.get("patient_id") or self.brain.patient_id).strip()
        if not raw_bed:
            raw_bed = self.brain.patient_id
        # Normalize "1" or "Bed 1" -> "Bed_1"
        if raw_bed.isdigit():
            patient_id = f"Bed_{raw_bed}"
        else:
            patient_id = raw_bed.replace(" ", "_")

        department = str(payload.get("department") or self.brain.department).strip() or "ICU"
        room = str(payload.get("room") or self.brain.room).strip() or "Room 101"
        patient_name = str(payload.get("patient_name") or self.brain.patient_name).strip() or f"Patient {patient_id}"
        diagnosis = str(payload.get("diagnosis") or self.brain.diagnosis).strip() or "Observation & Standard Telemetry"
        wifi_ssid = str(payload.get("wifi_ssid") or self.brain.wifi_ssid).strip() or "Hospital-CriticalCare-5G"
        broker = str(payload.get("broker") or self.brain.broker).strip() or self.brain.broker

        salt = secrets.token_bytes(16)
        pw_hash = self._hash_password(password, salt)
        token = secrets.token_urlsafe(32)

        with self.lock:
            self.password_salt = salt
            self.password_hash = pw_hash
            self.registered = True
            self.sessions = {token}

        self.brain.update_registration(
            patient_id=patient_id,
            department=department,
            room=room,
            patient_name=patient_name,
            diagnosis=diagnosis,
            wifi_ssid=wifi_ssid,
            broker=broker,
        )
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
            if token:
                self.sessions.discard(token)
            else:
                self.sessions.clear()

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
            "profile": self.brain.get_profile(),
        }


def make_bedside_handler(brain, auth_ctrl, meta_info):
    class BedsideMonitorHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _get_session_token(self):
            raw_cookie = self.headers.get("Cookie", "")
            if raw_cookie:
                cookie = SimpleCookie()
                try:
                    cookie.load(raw_cookie)
                    if auth_ctrl.cookie_name in cookie:
                        return cookie[auth_ctrl.cookie_name].value
                except Exception:
                    pass
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header.removeprefix("Bearer ").strip()
            return None

        def _send_json(self, obj, code=200, set_cookie=None, clear_cookie=False):
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            if set_cookie:
                self.send_header(
                    "Set-Cookie",
                    f"{auth_ctrl.cookie_name}={set_cookie}; Path=/; HttpOnly; SameSite=Strict",
                )
            elif clear_cookie:
                self.send_header(
                    "Set-Cookie",
                    f"{auth_ctrl.cookie_name}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict",
                )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path_only = self.path.split("?", 1)[0]
            token = self._get_session_token()

            if path_only == "/api/auth/status":
                self._send_json(auth_ctrl.status_payload(token))
                return

            if self.path.startswith("/api/state"):
                if not auth_ctrl.is_authenticated(token):
                    self._send_json({"error": "Authentication required", "auth": auth_ctrl.status_payload(token)}, code=401)
                    return

                pid = brain.patient_id
                chaos = read_patient_chaos(pid)
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
                        "sensor_links": brain.sensor_links,
                        "chaos": chaos,
                    }
                self._send_json(
                    {
                        "broker": f"{brain.broker}:{brain.port}",
                        "provider": meta_info.get("provider", "CUDAExecutionProvider"),
                        "server_time": time.time(),
                        "patient_id": pid,
                        "profile": brain.get_profile(),
                        "bed": bed_state,
                        "beds": {pid: bed_state},
                    }
                )
            elif self.path.startswith("/api/chaos"):
                if not auth_ctrl.is_authenticated(token):
                    self._send_json({"error": "Authentication required"}, code=401)
                    return
                self._send_json(read_patient_chaos(brain.patient_id))
            elif path_only in ("/bedside_simulation.html", "/bedside_simulation", "/simulation"):
                body = load_bedside_sim_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = load_bedside_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                data = json.loads(raw)
            except Exception:
                self._send_json({"ok": False, "error": "Invalid JSON"}, code=400)
                return

            path_only = self.path.split("?", 1)[0]
            token = self._get_session_token()

            if path_only == "/api/auth/register":
                try:
                    new_token = auth_ctrl.register(data)
                    self._send_json(
                        {"ok": True, "token": new_token, **auth_ctrl.status_payload(new_token)},
                        set_cookie=new_token,
                    )
                except ValueError as err:
                    self._send_json({"ok": False, "error": str(err)}, code=400)
                return

            if path_only == "/api/auth/login":
                try:
                    new_token = auth_ctrl.login(data.get("password", ""))
                    self._send_json(
                        {"ok": True, "token": new_token, **auth_ctrl.status_payload(new_token)},
                        set_cookie=new_token,
                    )
                except ValueError as err:
                    self._send_json({"ok": False, "error": str(err)}, code=401)
                return

            if path_only == "/api/auth/logout":
                auth_ctrl.logout(token)
                self._send_json({"ok": True, **auth_ctrl.status_payload(None)}, clear_cookie=True)
                return

            if not auth_ctrl.is_authenticated(token):
                self._send_json({"ok": False, "error": "Authentication required"}, code=401)
                return

            # Strict isolation: always operate ONLY on this bedside server's bound patient_id
            pid = brain.patient_id

            if self.path.startswith("/api/chaos/reset"):
                clean = default_bed_chaos()
                write_patient_chaos(pid, clean)
                self._send_json({"ok": True, "chaos": clean})
            elif self.path.startswith("/api/chaos"):
                chaos_cfg = data.get("chaos", default_bed_chaos())
                write_patient_chaos(pid, chaos_cfg)
                self._send_json({"ok": True, "chaos": chaos_cfg})
            elif self.path.startswith("/api/manipulate") or self.path.startswith("/api/vitals"):
                if data.get("reset"):
                    clean = default_bed_chaos()
                    write_patient_chaos(pid, clean)
                    self._send_json({"ok": True, "chaos": clean})
                    return

                chaos = read_patient_chaos(pid)
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

                write_patient_chaos(pid, chaos)
                self._send_json({"ok": True, "chaos": chaos})
            else:
                self._send_json({"ok": False}, code=404)

    return BedsideMonitorHandler


def start_bedside_web_server(brain, meta_info, web_port=8080, web_host="127.0.0.1"):
    auth_ctrl = BedsideAuthController(brain, web_port)
    handler_cls = make_bedside_handler(brain, auth_ctrl, meta_info)
    try:
        server = ThreadingHTTPServer((web_host, web_port), handler_cls)
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

        defaults = DEFAULT_BED_PROFILES.get(patient_id, {})
        digits = "".join(filter(str.isdigit, patient_id)) or "1"
        self.department = defaults.get("department", "ICU")
        self.room = defaults.get("room", f"Room 10{digits}")
        self.patient_name = defaults.get("patient_name", f"Patient {patient_id}")
        self.diagnosis = defaults.get("diagnosis", "Observation & Standard Telemetry")
        self.wifi_ssid = defaults.get("wifi_ssid", "Hospital-CriticalCare-5G")

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
        self.lock = threading.Lock()
        self.client = None

    def get_profile(self):
        with self.lock:
            return {
                "patient_id": self.patient_id,
                "bed_label": self.patient_id.replace("_", " "),
                "department": self.department,
                "room": self.room,
                "patient_name": self.patient_name,
                "diagnosis": self.diagnosis,
                "wifi_ssid": self.wifi_ssid,
                "broker": self.broker,
                "web_port": self.web_port,
            }

    def update_registration(self, patient_id, department, room, patient_name, diagnosis, wifi_ssid, broker):
        with self.lock:
            old_pid = self.patient_id
            self.patient_id = patient_id
            self.department = department
            self.room = room
            self.patient_name = patient_name
            self.diagnosis = diagnosis
            self.wifi_ssid = wifi_ssid
            self.broker = broker

        if self.client and old_pid != patient_id:
            try:
                self.client.unsubscribe(f"hospital/{old_pid}/sensors/#")
                self.client.unsubscribe(f"hospital/{old_pid}/control/#")
                self.client.subscribe(f"hospital/{patient_id}/sensors/#")
                self.client.subscribe(f"hospital/{patient_id}/control/#")
            except Exception:
                pass
        self.send_discovery_announcement(auto_admit=True)

    def start(self):
        cid = f"Jetson_Edge_Brain_{self.patient_id}_{os.getpid()}_{random.randint(100, 999)}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        else:
            self.client = mqtt.Client(client_id=cid)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[EDGE BRAIN:{self.patient_id}] Warning: Could not connect to broker {self.broker}:{self.port}: {e}")

        self.pub_thread = threading.Thread(target=self._run_summary_loop, daemon=True)
        self.pub_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[EDGE BRAIN:{self.patient_id}] Connected to MQTT broker {self.broker}:{self.port}")
        self.client.subscribe(f"hospital/{self.patient_id}/sensors/#")
        self.client.subscribe(f"hospital/{self.patient_id}/control/#")
        self.client.subscribe("hospital/control/#")
        self.client.subscribe("hospital/discovery/probe")
        self.send_discovery_announcement()

    def send_discovery_announcement(self, auto_admit=False):
        if not self.client:
            return
        try:
            with self.lock:
                hr = self.latest_hr
                spo2 = self.latest_spo2
                co2 = self.latest_co2
                ibp = self.latest_ibp
                nibp = self.latest_nibp
                ecg_var = self.latest_ecg_var
                profile = {
                    "department": self.department,
                    "room": self.room,
                    "patient_name": self.patient_name,
                    "diagnosis": self.diagnosis,
                    "wifi_ssid": self.wifi_ssid,
                    "web_port": self.web_port,
                    "auto_admit": auto_admit,
                }
            announcement = {
                "patient_id": self.patient_id,
                "device_model": "NVIDIA Jetson Orin",
                "exec_provider": self.active_provider,
                "broker": f"{self.broker}:{self.port}",
                "timestamp": time.time(),
                "status": "ready",
                **profile,
                "latest_vitals": {
                    "hr": hr,
                    "spo2": spo2,
                    "co2": co2,
                    "ibp": ibp,
                    "nibp": nibp,
                    "ecg_variance": ecg_var,
                },
            }
            self.client.publish("hospital/discovery/announce", json.dumps(announcement), qos=1)
        except Exception:
            pass

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
        bed_chaos = read_patient_chaos(self.patient_id)
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

            if topic == "hospital/discovery/probe":
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

            if topic == f"hospital/{self.patient_id}/control/chaos":
                update_chaos_file(payload, patient_id=self.patient_id)
                return

            if topic == "hospital/control/chaos":
                update_chaos_file(payload)
                return

            topic_agent_id = payload.get("agent_id", "")
            agent_id = "Agent_EtCO2" if topic_agent_id == "Agent_CO2" else topic_agent_id
            if agent_id not in self.trust_trackers:
                return
            valid_topics = {f"hospital/{self.patient_id}/sensors/{agent_id}"}
            if agent_id == "Agent_EtCO2":
                valid_topics.add(f"hospital/{self.patient_id}/sensors/Agent_CO2")
            if topic not in valid_topics:
                return

            bed_chaos = read_patient_chaos(self.patient_id)
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
            time.sleep(0.40)
            loop_cnt += 1
            if loop_cnt % 10 == 0:
                self.send_discovery_announcement()

            if not self.client:
                continue

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
    parser.add_argument("--patients", nargs="+", default=None, help="Optional multi-patient list (e.g. Bed_1 Bed_2)")
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--web-port", type=int, default=8080, help="Base Local Bedside Monitor Web UI port")
    parser.add_argument(
        "--web-host",
        type=str,
        default="127.0.0.1",
        help="Bind address for bedside web UI (defaults to 127.0.0.1 so other Wi-Fi devices cannot view this bedside)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to ecg_anomaly.onnx")
    args = parser.parse_args()

    patient_list = args.patients if args.patients else [args.patient_id]
    session, active_provider = create_shared_onnx_session(args.model)

    print("=" * 66)
    print("NVIDIA Jetson Orin - Bedside GPU Edge Brain Active")
    print(f"Patient Nodes:     {patient_list}")
    print(f"MQTT Gateway:      {args.broker}:{args.port}")
    print(f"Bind Host:         {args.web_host} (isolated from external Wi-Fi peers)")
    print(f"HTML Template:     {BEDSIDE_HTML_FILE}")
    print(f"Inference Engine:  {active_provider}")
    print("=" * 66)

    engines = []
    web_servers = []
    for idx, pid in enumerate(patient_list):
        unit_port = args.web_port + idx
        brain = JetsonEdgeBrain(
            patient_id=pid,
            broker=args.broker,
            port=args.port,
            session=session,
            active_provider=active_provider,
            web_port=unit_port,
        )
        brain.start()
        engines.append(brain)

        meta = {
            "broker": f"{args.broker}:{args.port}",
            "provider": active_provider,
            "patient_id": pid,
            "web_port": unit_port,
        }
        srv = start_bedside_web_server(brain, meta, web_port=unit_port, web_host=args.web_host)
        if srv:
            web_servers.append(srv)

    stop_event = threading.Event()

    def handle_sig(sig, frame):
        print("\n[INFO] Gracefully shutting down Bedside Edge Brain...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while not stop_event.is_set():
        time.sleep(0.5)

    for brain in engines:
        brain.stop()
    for srv in web_servers:
        try:
            srv.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
