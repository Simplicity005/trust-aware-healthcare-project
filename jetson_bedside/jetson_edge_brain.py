#!/usr/bin/env python3
"""
Bedside GPU Edge Brain & Local Bedside Monitor (Runs on NVIDIA Jetson Orin)
Executes 1D-CNN ECG anomaly inference, triggers local bedside alarms,
validates cross-sensor physiological plausibility, streams bedside summaries to
the Nurse Station, and serves the Local Bedside Monitor from bedside_monitor.html
directly on the Jetson at http://localhost:8080 with local sensor fault controls.
"""

import os
import sys
import time
import json
import signal
import random
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import paho.mqtt.client as mqtt

try:
    import onnxruntime as ort
except ImportError:
    ort = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAOS_FILE = os.path.join(BASE_DIR, "chaos.json")
DEFAULT_MODEL = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
BEDSIDE_HTML_FILE = os.path.join(BASE_DIR, "bedside_monitor.html")
CHAOS_LOCK = threading.Lock()


def load_bedside_html():
    """Reads bedside_monitor.html from disk so HTML/CSS edits take effect on refresh."""
    if os.path.exists(BEDSIDE_HTML_FILE):
        with open(BEDSIDE_HTML_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "<!DOCTYPE html><html><body><h3>bedside_monitor.html not found</h3></body></html>"


def default_bed_chaos():
    return {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False, "condition": "normal"},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_spo2": 98},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_hr": 75},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False},
        "vitals": {
            "target_hr": 75,
            "target_spo2": 98,
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


def make_bedside_handler(engines_dict, meta_info):
    class BedsideMonitorHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send_json(self, obj, code=200):
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                beds_payload = {}
                for pid, eng in engines_dict.items():
                    with eng.lock:
                        beds_payload[pid] = {
                            "hr": eng.latest_hr,
                            "spo2": eng.latest_spo2,
                            "hr_valid": eng.hr_plausible,
                            "spo2_valid": eng.spo2_plausible,
                            "anomaly_prob": round(eng.last_prob, 4),
                            "arrhythmia": eng.last_arrhythmia,
                            "inference_ms": round(eng.last_inf_ms, 2),
                            "offloaded": eng.last_offloaded,
                            "relay_b": eng.last_relay_b,
                            "packets": eng.packets_processed,
                            "ecg_wave": eng.latest_ecg_wave[-250:] if eng.latest_ecg_wave else [],
                            "sensor_links": eng.sensor_links,
                            "chaos": read_patient_chaos(pid),
                        }
                self._send_json(
                    {
                        "broker": meta_info.get("broker", "127.0.0.1:1883"),
                        "provider": meta_info.get("provider", "CUDAExecutionProvider"),
                        "beds": beds_payload,
                    }
                )
            elif self.path.startswith("/api/chaos"):
                bed_id = next(iter(engines_dict.keys()), "Bed_1")
                if "bed=" in self.path:
                    bed_id = self.path.split("bed=", 1)[1].split("&")[0]
                self._send_json(read_patient_chaos(bed_id))
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
                self._send_json({"ok": False}, code=400)
                return

            if self.path.startswith("/api/chaos/reset"):
                pid = data.get("patient_id") or next(iter(engines_dict.keys()), "Bed_1")
                clean = default_bed_chaos()
                write_patient_chaos(pid, clean)
                self._send_json({"ok": True, "chaos": clean})
            elif self.path.startswith("/api/chaos"):
                pid = data.get("patient_id") or next(iter(engines_dict.keys()), "Bed_1")
                chaos_cfg = data.get("chaos", default_bed_chaos())
                write_patient_chaos(pid, chaos_cfg)
                self._send_json({"ok": True, "chaos": chaos_cfg})
            elif self.path.startswith("/api/manipulate") or self.path.startswith("/api/vitals"):
                pid = data.get("patient_id") or next(iter(engines_dict.keys()), "Bed_1")
                chaos = read_patient_chaos(pid)
                if "vitals" not in chaos:
                    chaos["vitals"] = {}

                if "target_hr" in data:
                    chaos["vitals"]["target_hr"] = int(data["target_hr"])
                    chaos.setdefault("Agent_HR", {})["target_hr"] = int(data["target_hr"])
                if "target_spo2" in data:
                    chaos["vitals"]["target_spo2"] = int(data["target_spo2"])
                    chaos.setdefault("Agent_SpO2", {})["target_spo2"] = int(data["target_spo2"])
                if "ecg_condition" in data:
                    chaos["vitals"]["ecg_condition"] = str(data["ecg_condition"])
                    chaos.setdefault("Agent_ECG", {})["condition"] = str(data["ecg_condition"])

                if "delay_ms" in data:
                    d_ms = float(data["delay_ms"])
                    for aid in ["Agent_ECG", "Agent_SpO2", "Agent_HR"]:
                        chaos.setdefault(aid, {})["delay_ms"] = d_ms
                if "drop_rate" in data:
                    dr = float(data["drop_rate"])
                    for aid in ["Agent_ECG", "Agent_SpO2", "Agent_HR"]:
                        chaos.setdefault(aid, {})["drop_rate"] = dr
                if "malicious" in data:
                    m = bool(data["malicious"])
                    for aid in ["Agent_ECG", "Agent_SpO2", "Agent_HR"]:
                        chaos.setdefault(aid, {})["malicious"] = m
                if "overload_gpu" in data:
                    ov = bool(data["overload_gpu"])
                    chaos.setdefault("Orin_Link", {})["overload_gpu"] = ov
                    chaos.setdefault("Orin_Link", {})["use_relay_b"] = ov

                write_patient_chaos(pid, chaos)
                self._send_json({"ok": True, "chaos": chaos})
            else:
                self._send_json({"ok": False}, code=404)

    return BedsideMonitorHandler


def start_bedside_web_server(engines_dict, meta_info, web_port=8080):
    handler_cls = make_bedside_handler(engines_dict, meta_info)
    try:
        server = ThreadingHTTPServer(("0.0.0.0", web_port), handler_cls)
        server.daemon_threads = True
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[BEDSIDE MONITOR] Clean Bento Bedside UI live at http://localhost:{web_port}")
        return server
    except Exception as e:
        print(f"[BEDSIDE MONITOR] Notice: Could not bind port {web_port} ({e})")
        return None


class JetsonEdgeBrain:
    def __init__(self, patient_id, broker, port, session, active_provider):
        self.patient_id = patient_id
        self.broker = broker
        self.port = port
        self.session = session
        self.active_provider = active_provider
        self.input_name = self.session.get_inputs()[0].name if self.session else "ecg_input"

        self.latest_hr = 75
        self.latest_spo2 = 98
        self.latest_ecg_var = 0.04
        self.hr_plausible = True
        self.spo2_plausible = True
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
            "Agent_ECG": {"latency_ms": 12.0, "loss_pct": 0.0, "last_seq": None, "rx": 0, "lost": 0},
            "Agent_SpO2": {"latency_ms": 10.0, "loss_pct": 0.0, "last_seq": None, "rx": 0, "lost": 0},
            "Agent_HR": {"latency_ms": 10.0, "loss_pct": 0.0, "last_seq": None, "rx": 0, "lost": 0},
        }
        self.lock = threading.Lock()
        self.client = None

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

    def send_discovery_announcement(self):
        if not self.client:
            return
        try:
            with self.lock:
                hr = self.latest_hr
                spo2 = self.latest_spo2
                ecg_var = self.latest_ecg_var
            announcement = {
                "patient_id": self.patient_id,
                "device_model": "NVIDIA Jetson Orin",
                "exec_provider": self.active_provider,
                "broker": f"{self.broker}:{self.port}",
                "timestamp": time.time(),
                "status": "ready",
                "latest_vitals": {"hr": hr, "spo2": spo2, "ecg_variance": ecg_var},
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
        info["rx"] += 1
        total = max(1, info["rx"] + info["lost"])
        info["loss_pct"] = round((info["lost"] / total) * 100.0, 1)

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
                print(f"[EDGE BRAIN:{self.patient_id}] Node officially admitted by Nurse Station to ICU Ward.")
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

            agent_id = payload.get("agent_id", "")
            with self.lock:
                self._track_link(agent_id, payload, now)

            if agent_id == "Agent_HR":
                hr_val = int(payload.get("value", 75))
                with self.lock:
                    self.latest_hr = hr_val
                    self.hr_plausible = 35 <= hr_val <= 210
                self._check_plausibility()
                return

            if agent_id == "Agent_SpO2":
                spo2_val = int(payload.get("value", 98))
                with self.lock:
                    self.latest_spo2 = spo2_val
                    self.spo2_plausible = 65 <= spo2_val <= 100
                self._check_plausibility()
                return

            if agent_id == "Agent_ECG":
                ecg_wave = payload.get("ecg_wave", [])
                if len(ecg_wave) != 250:
                    return

                bed_chaos = read_patient_chaos(self.patient_id)
                orin_chaos = bed_chaos.get("Orin_Link", {})
                overload_gpu = bool(orin_chaos.get("overload_gpu", False)) or self.force_offload
                use_relay_b = bool(orin_chaos.get("use_relay_b", False)) or overload_gpu

                anomaly_prob, inference_ms = self._run_inference(ecg_wave, overload_gpu=overload_gpu)
                is_arrhythmia = anomaly_prob > 0.50
                var_val = float(np.var(ecg_wave))

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
            arrhythmia = self.last_arrhythmia

        conflict = False
        if not arrhythmia and spo2 >= 95 and (hr <= 0 or hr >= 240):
            conflict = True
        if spo2 < 50 or spo2 > 105 or hr < 30 or hr > 280:
            conflict = True
        if spo2 <= 30 and hr > 60 and not arrhythmia:
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

            with self.lock:
                summary = {
                    "patient_id": self.patient_id,
                    "inf_ms": round(self.last_inf_ms, 2),
                    "exec_provider": self.active_provider,
                    "arrhythmia_prob": round(self.last_prob, 4),
                    "local_bed_alarm": self.local_bed_alarm,
                    "sensor_conflict": self.sensor_conflict,
                    "latest_vitals": {
                        "hr": self.latest_hr,
                        "spo2": self.latest_spo2,
                        "ecg_variance": round(self.latest_ecg_var, 4),
                    },
                    "ecg_wave": self.latest_ecg_wave,
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
    parser = argparse.ArgumentParser(description="Bedside GPU Edge Brain & Local Bedside Monitor")
    parser.add_argument("--patient-id", type=str, default="Bed_1", help="Primary Patient ID")
    parser.add_argument("--patients", nargs="+", default=None, help="Optional multi-patient list (e.g. Bed_1 Bed_2)")
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--web-port", type=int, default=8080, help="Local Bedside Monitor Web UI port")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to ecg_anomaly.onnx")
    args = parser.parse_args()

    patient_list = args.patients if args.patients else [args.patient_id]
    session, active_provider = create_shared_onnx_session(args.model)

    print("=" * 66)
    print("NVIDIA Jetson Orin - Bedside GPU Edge Brain Active")
    print(f"Patient Nodes:     {patient_list}")
    print(f"MQTT Gateway:      {args.broker}:{args.port}")
    print(f"Local Bedside Web: http://localhost:{args.web_port}")
    print(f"HTML Template:     {BEDSIDE_HTML_FILE}")
    print(f"Inference Engine:  {active_provider}")
    print("=" * 66)

    engines = {}
    for pid in patient_list:
        brain = JetsonEdgeBrain(
            patient_id=pid,
            broker=args.broker,
            port=args.port,
            session=session,
            active_provider=active_provider,
        )
        brain.start()
        engines[pid] = brain

    meta = {
        "broker": f"{args.broker}:{args.port}",
        "provider": active_provider,
        "patients": patient_list,
    }
    web_server = start_bedside_web_server(engines, meta, web_port=args.web_port)

    stop_event = threading.Event()

    def handle_sig(sig, frame):
        print("\n[INFO] Gracefully shutting down Bedside Edge Brain...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while not stop_event.is_set():
        time.sleep(0.5)

    for brain in engines.values():
        brain.stop()
    if web_server:
        try:
            web_server.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
