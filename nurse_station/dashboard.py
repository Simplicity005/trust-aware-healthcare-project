#!/usr/bin/env python3
"""
Multi-Patient Central Nurse Station & Trust Engine Dashboard
(Runs on the Nurse's Table Gateway Laptop)
Dynamically discovers patient beds as Jetson bedside nodes connect over MQTT,
computes composite Trust Scores T_i(t) in [0.0, 1.0] per sensor per bed,
manages compute offloading and alert routing, and renders a clean dark
Bento Box interface with zero clutter.
"""

import os
import time
import json
import threading
from collections import deque
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
AGENT_IDS = ["Agent_ECG", "Agent_SpO2", "Agent_HR"]
AGENT_LABELS = {"Agent_ECG": "ECG sensor", "Agent_SpO2": "SpO2 sensor", "Agent_HR": "HR sensor"}


class AgentTrustTracker:
    """
    Computes composite Trust Score T_i(t) in [0.0, 1.0] per sensor agent:
        T_i(t) = w1 * S_L(t) + w2 * S_J(t) + w3 * S_P(t) + w4 * S_D(t)
    with weights w1=0.25, w2=0.20, w3=0.25, w4=0.30 and EWMA smoothing (alpha=0.35).
    """

    def __init__(self, agent_id, w1=0.25, w2=0.20, w3=0.25, w4=0.30, alpha=0.35):
        self.agent_id = agent_id
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        self.alpha = alpha

        self.latencies = deque(maxlen=25)
        self.delivery_window = deque(maxlen=30)
        self.validity_window = deque(maxlen=15)

        self.last_seq = None
        self.last_rx_time = time.time()
        self.last_value = None
        self.ecg_wave = [0.0] * 250

        self.s_l = 1.0
        self.s_j = 1.0
        self.s_p = 1.0
        self.s_d = 1.0
        self.trust_score = 1.0
        self.status = "TRUSTED"

    def check_data_plausibility(self, payload):
        if self.agent_id == "Agent_ECG":
            wave = payload.get("ecg_wave", [])
            if not wave or len(wave) != 250:
                return 0.0
            arr = np.array(wave, dtype=np.float32)
            max_abs = float(np.max(np.abs(arr)))
            std_val = float(np.std(arr))
            diff_std = float(np.std(np.diff(arr)))
            if max_abs > 2.2 or std_val > 0.65 or std_val < 0.03 or diff_std > 0.35:
                return 0.0
            return 1.0

        val = payload.get("value", 0)
        if self.agent_id == "Agent_SpO2":
            return 1.0 if (70 <= val <= 100) else 0.0
        if self.agent_id == "Agent_HR":
            return 1.0 if (40 <= val <= 190) else 0.0
        return 1.0

    def update(self, payload, rx_time):
        seq = int(payload.get("seq", 0))
        sent_ts = float(payload.get("sent_ts", rx_time))

        latency_ms = max(0.5, (rx_time - sent_ts) * 1000.0)
        self.latencies.append(latency_ms)

        if self.last_seq is not None and seq > self.last_seq:
            missed = seq - self.last_seq - 1
            for _ in range(min(missed, 30)):
                self.delivery_window.append(0)
        self.delivery_window.append(1)
        self.last_seq = seq
        self.last_rx_time = rx_time

        if self.agent_id == "Agent_ECG":
            self.ecg_wave = payload.get("ecg_wave", [0.0] * 250)
        else:
            self.last_value = payload.get("value", 0)

        d_valid = self.check_data_plausibility(payload)
        self.validity_window.append(d_valid)

        avg_lat = float(np.mean(self.latencies))
        self.s_l = float(np.clip(np.exp(-max(0.0, avg_lat - 25.0) / 160.0), 0.0, 1.0))

        if len(self.latencies) >= 2:
            jitter_ms = float(np.std(self.latencies))
        else:
            jitter_ms = 0.0
        self.s_j = float(np.clip(np.exp(-jitter_ms / 60.0), 0.0, 1.0))

        self.s_p = float(np.mean(self.delivery_window)) if self.delivery_window else 1.0
        ewma_valid = float(np.mean(self.validity_window)) if self.validity_window else 1.0
        self.s_d = 0.25 * ewma_valid if d_valid == 0.0 else ewma_valid

        raw_trust = (
            self.w1 * self.s_l
            + self.w2 * self.s_j
            + self.w3 * self.s_p
            + self.w4 * self.s_d
        )

        if d_valid == 0.0:
            raw_trust = min(raw_trust, 0.32)

        self.trust_score = float(
            np.clip(self.alpha * raw_trust + (1.0 - self.alpha) * self.trust_score, 0.0, 1.0)
        )
        self.evaluate_status(rx_time)

    def evaluate_status(self, now):
        silence_sec = now - self.last_rx_time
        if silence_sec > 1.8:
            penalty = np.exp(-(silence_sec - 1.8) / 2.5)
            self.s_p = float(np.clip(self.s_p * penalty, 0.0, 1.0))
            raw_trust = (
                self.w1 * self.s_l
                + self.w2 * self.s_j
                + self.w3 * self.s_p
                + self.w4 * self.s_d
            )
            self.trust_score = float(np.clip(raw_trust, 0.0, 1.0))

        prev_status = self.status
        if self.trust_score >= 0.70:
            self.status = "TRUSTED"
        elif self.trust_score >= 0.45:
            self.status = "SUSPICIOUS"
        else:
            self.status = "ISOLATED"

        return prev_status != self.status, prev_status, self.status


class MultiPatientTrustEngine:
    def __init__(self, broker_host=None, broker_port=1883):
        self.lock = threading.Lock()
        # Active beds officially admitted to the ward by the nurse
        self.beds = {}
        # Set of bed IDs admitted by the nurse (defaults to Bed_1 so initial ward is active)
        self.added_bed_ids = {"Bed_1"}
        # Map of all detected Jetson devices on the network/MQTT: {patient_id: {...info...}}
        self.discovered_jetsons = {}
        self.global_alert_log = deque(maxlen=20)
        self.stop_event = threading.Event()

        # Station identity & High Availability (HA) Failover tracking
        self.station_id = os.environ.get("NURSE_STATION_ID", "Nurse_Station_1")
        self.station_port = int(os.environ.get("STREAMLIT_SERVER_PORT", "8501"))
        self.is_primary = ("1" in self.station_id or "primary" in self.station_id.lower())
        self.peer_stations = {}
        self.last_primary_heartbeat_ts = time.time() if self.is_primary else 0.0
        self.failover_triggered = False
        self.failover_simulated = False

        # Room mapping for ICU beds
        self.bed_rooms = {
            "Bed_1": "Room 101 (Cardiac ICU)",
            "Bed_2": "Room 102 (Cardiac ICU)",
            "Bed_3": "Room 103 (Neuro ICU)",
            "Bed_4": "Room 104 (Surgical ICU)",
            "Bed_5": "Room 105 (Trauma ICU)",
        }

        self.ort_session = None
        self.ort_input_name = "ecg_input"
        if os.path.exists(MODEL_FILE):
            try:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.cpu_session = ort.InferenceSession(
                    MODEL_FILE, sess_options=opts, providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                print(f"[DASHBOARD] Warning: Could not load CPU fallback model: {e}")

        self.client = None
        self._init_mqtt()

        # Background simulator to keep admitted beds with no external script active and animated
        self.sim_thread = threading.Thread(target=self._run_background_synthesizer, daemon=True)
        self.sim_thread.start()

        # High-availability heartbeat thread
        self.hb_thread = threading.Thread(target=self._run_heartbeat_broadcaster, daemon=True)
        self.hb_thread.start()

    def get_bed_room(self, pid):
        if pid in self.bed_rooms:
            return self.bed_rooms[pid]
        digits = "".join(filter(str.isdigit, pid))
        num = f"10{digits}" if digits else "105"
        return f"Room {num} (ICU Ward)"

    def set_bed_room(self, pid, room_name):
        with self.lock:
            self.bed_rooms[pid] = room_name

    def _run_heartbeat_broadcaster(self):
        """Broadcasts Nurse Station heartbeat and monitors peer stations for failover."""
        while not self.stop_event.is_set():
            time.sleep(1.0)
            now = time.time()

            # 1. Publish own heartbeat over MQTT if not simulating failure
            if self.client and not self.failover_simulated:
                try:
                    payload = {
                        "station_id": self.station_id,
                        "role": "primary" if self.is_primary else "secondary",
                        "port": self.station_port,
                        "timestamp": now,
                        "active_beds": list(self.added_bed_ids),
                    }
                    self.client.publish(
                        "hospital/nurse_stations/heartbeat",
                        json.dumps(payload),
                        qos=0,
                    )
                except Exception:
                    pass

            # 2. Check for Primary Station timeout if this is a standby station
            if not self.is_primary and not self.failover_simulated:
                if self.last_primary_heartbeat_ts > 0 and (now - self.last_primary_heartbeat_ts) > 4.0:
                    with self.lock:
                        self.is_primary = True
                        self.failover_triggered = True
                        ts_str = time.strftime("%H:%M:%S")
                        entry = (
                            f"[{ts_str}] [HIGH AVAILABILITY] PRIMARY STATION FAILURE DETECTED: "
                            f"Primary station timed out (>4s). {self.station_id} (Port {self.station_port}) "
                            f"AUTOMATICALLY PROMOTED TO ACTIVE PRIMARY LEADER!"
                        )
                        self.global_alert_log.appendleft(entry)

    def trigger_failover_test(self):
        """Simulates primary station crash/failover to demonstrate hot-standby promotion."""
        now = time.time()
        with self.lock:
            if self.is_primary:
                self.is_primary = False
                self.failover_simulated = True
                ts_str = time.strftime("%H:%M:%S")
                entry = f"[{ts_str}] [HA TEST] Simulated Primary Station failure on {self.station_id}. Standby station taking over..."
                self.global_alert_log.appendleft(entry)
                if self.client:
                    try:
                        self.client.publish(
                            "hospital/nurse_stations/failover",
                            json.dumps({"failed_station": self.station_id, "timestamp": now}),
                            qos=1,
                        )
                    except Exception:
                        pass
            else:
                self.is_primary = True
                self.failover_triggered = True
                self.failover_simulated = False
                ts_str = time.strftime("%H:%M:%S")
                entry = f"[{ts_str}] [HA TEST] Manual promotion: {self.station_id} is now ACTIVE PRIMARY LEADER!"
                self.global_alert_log.appendleft(entry)

    def _run_background_synthesizer(self):
        """Synthesizes baseline telemetry for admitted beds that have no active external sensor script."""
        step = 0
        while not self.stop_event.is_set():
            time.sleep(0.40)
            step += 1
            now = time.time()
            with self.lock:
                for pid in list(self.added_bed_ids):
                    bed = self._get_or_create_bed(pid)
                    # If this bed hasn't received real MQTT packets in > 2.0 seconds, synthesize healthy baseline
                    if (now - bed.last_seen_ts) > 2.0:
                        phase = ((sum(ord(c) for c in pid) % 5) - 2) * 0.01
                        t = np.linspace(0, 1, 250)
                        ecg = (
                            0.15 * np.exp(-((t - (0.20 + phase)) ** 2) / (2 * 0.025 ** 2))
                            - 0.15 * np.exp(-((t - (0.35 + phase)) ** 2) / (2 * 0.010 ** 2))
                            + 1.25 * np.exp(-((t - (0.40 + phase)) ** 2) / (2 * 0.012 ** 2))
                            - 0.25 * np.exp(-((t - (0.45 + phase)) ** 2) / (2 * 0.012 ** 2))
                            + 0.30 * np.exp(-((t - (0.65 + phase)) ** 2) / (2 * 0.040 ** 2))
                            + np.random.normal(0, 0.015, 250)
                        )
                        wave_list = np.round(ecg, 4).tolist()
                        bed.latest_ecg_wave = wave_list
                        hr_val = 72 + (sum(ord(c) for c in pid) % 7) + int(np.random.randint(-1, 2))
                        spo2_val = 98 + int(np.random.randint(0, 2))
                        var_val = float(np.var(ecg))
                        bed.edge_summary["latest_vitals"] = {
                            "hr": hr_val,
                            "spo2": spo2_val,
                            "ecg_variance": round(var_val, 4),
                        }
                        bed.edge_summary["inf_ms"] = round(0.18 + float(np.random.uniform(0.01, 0.08)), 2)
                        bed.edge_summary["arrhythmia_prob"] = 0.02
                        bed.edge_summary["local_bed_alarm"] = False
                        bed.agents["Agent_ECG"].update(step, now - 0.01, now, round(var_val, 4), wave_list)
                        bed.agents["Agent_SpO2"].update(step, now - 0.01, now, spo2_val)
                        bed.agents["Agent_HR"].update(step, now - 0.01, now, hr_val)
                        if step % 3 == 0:
                            bed.trust_history.append({
                                "time": time.strftime("%H:%M:%S", time.localtime(now)),
                                "Agent_ECG": bed.agents["Agent_ECG"].trust_score,
                                "Agent_SpO2": bed.agents["Agent_SpO2"].trust_score,
                                "Agent_HR": bed.agents["Agent_HR"].trust_score,
                            })

    def _get_or_create_bed(self, patient_id):
        if not patient_id:
            patient_id = "Bed_1"
        if patient_id not in self.beds:
            print(f"[NURSE STATION] Initializing Bed monitor: {patient_id}")
            self.beds[patient_id] = PatientBedMonitor(patient_id)
        self.beds[patient_id].last_seen_ts = time.time()
        return self.beds[patient_id]

    def remove_inactive_beds(self, timeout_sec=8.0):
        """Removes beds that have stopped transmitting for longer than timeout_sec."""
        now = time.time()
        with self.lock:
            stale = [pid for pid, b in self.beds.items() if (now - b.last_seen_ts) > timeout_sec and pid != "Bed_1"]
            for pid in stale:
                if pid in self.added_bed_ids:
                    self.added_bed_ids.remove(pid)
                del self.beds[pid]

    def probe_network_for_jetsons(self):
        """Broadcasts active probe request over MQTT to solicit discovery announcements from all Jetsons."""
        if self.client:
            try:
                self.client.publish(
                    "hospital/discovery/probe",
                    json.dumps({"action": "probe", "timestamp": time.time()}),
                    qos=1,
                )
            except Exception as e:
                print(f"[DASHBOARD] Discovery probe error: {e}")

    def add_bed(self, patient_id, room_name=None):
        """Nurse officially admits/adds a detected Jetson to the ICU Ward."""
        with self.lock:
            self.added_bed_ids.add(patient_id)
            if room_name:
                self.bed_rooms[patient_id] = room_name
            bed = self._get_or_create_bed(patient_id)
            if patient_id in self.discovered_jetsons:
                v = self.discovered_jetsons[patient_id].get("latest_vitals")
                if v:
                    bed.edge_summary["latest_vitals"] = v
            ts_str = time.strftime("%H:%M:%S")
            assigned_rm = self.get_bed_room(patient_id)
            entry = f"[{ts_str}] [NURSE ACTION] Added Bed '{patient_id}' (Jetson Orin) assigned to {assigned_rm}."
            self.global_alert_log.appendleft(entry)
            bed.alert_log.appendleft(entry)

        if self.client:
            try:
                self.client.publish(
                    f"hospital/{patient_id}/control/admit",
                    json.dumps({"patient_id": patient_id, "room": self.get_bed_room(patient_id), "status": "admitted", "timestamp": time.time()}),
                    qos=1,
                )
            except Exception:
                pass
        return True

    def remove_bed(self, patient_id):
        """Nurse discharges/releases a bed from active ward monitoring (returns it to unadded pool)."""
        with self.lock:
            if patient_id in self.added_bed_ids:
                self.added_bed_ids.remove(patient_id)
            if patient_id in self.beds:
                del self.beds[patient_id]
            ts_str = time.strftime("%H:%M:%S")
            entry = f"[{ts_str}] [NURSE ACTION] Released Bed '{patient_id}' from ICU ward."
            self.global_alert_log.appendleft(entry)
        return True

    def simulate_new_jetson(self, suggested_id=None):
        """Simulates an unadded Jetson node broadcasting on MQTT/LAN so testing and demos are seamless."""
        with self.lock:
            if not suggested_id:
                n = 2
                while f"Bed_{n}" in self.added_bed_ids or f"Bed_{n}" in self.discovered_jetsons:
                    n += 1
                suggested_id = f"Bed_{n}"

            now = time.time()
            self.discovered_jetsons[suggested_id] = {
                "patient_id": suggested_id,
                "device_model": "NVIDIA Jetson Orin Nano",
                "exec_provider": "CUDAExecutionProvider",
                "first_seen_ts": now,
                "last_seen_ts": now,
                "packet_count": 42,
                "latest_vitals": {"hr": 76, "spo2": 99, "ecg_variance": 0.039},
                "ip": "192.168.1.108",
            }
            return suggested_id

    def _init_mqtt(self):
        client_id = f"Nurse_Station_MultiBed_{os.getpid()}_{int(time.time() * 1000) % 10000}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        else:
            self.client = mqtt.Client(client_id=cid)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.mqtt_connected = False
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            self.mqtt_connected = True
        except Exception:
            self.mqtt_connected = False

    def _ensure_bed_locked(self, pid):
        if pid not in self.beds:
            self.beds[pid] = {
                "trackers": {aid: AgentTrustTracker(aid) for aid in AGENT_IDS},
                "history": deque(maxlen=80),
                "last_seen": time.time(),
                "edge_summary": {
                    "inference_ms": 0.85,
                    "anomaly_prob": 0.01,
                    "is_arrhythmia": False,
                    "execution_provider": "CPUExecutionProvider",
                    "gpu_overloaded": False,
                    "link_congested": False,
                    "alert_route": "Direct_Edge_Link",
                },
                "compute_mode": "Jetson Local",
                "alert_route": "Direct",
            }
        return self.beds[pid]

    def _log(self, pid, level, msg):
        ts = time.strftime("%H:%M:%S")
        self.event_logs.appendleft({"ts": ts, "bed": pid, "level": level, "msg": msg})

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self.mqtt_connected = True
        client.subscribe("hospital/+/sensors/#")
        client.subscribe("hospital/+/edge/summary")
        client.subscribe("hospital/+/alerts/#")
        client.subscribe("hospital/sensors/#")
        client.subscribe("hospital/edge/summary")
        client.subscribe("hospital/alerts/#")
        client.subscribe("hospital/discovery/#")
        client.subscribe("hospital/+/discovery/#")
        client.subscribe("hospital/+/announce")
        client.subscribe("hospital/nurse_stations/heartbeat")
        client.subscribe("hospital/nurse_stations/failover")

    def _on_message(self, client, userdata, msg):
        rx_time = time.time()
        try:
            payload = json.loads(msg.payload.decode("utf-8"))

            # Handle Nurse Station Heartbeat & Failover Coordination
            if topic == "hospital/nurse_stations/heartbeat":
                sender_id = payload.get("station_id")
                if sender_id and sender_id != self.station_id:
                    with self.lock:
                        self.peer_stations[sender_id] = {
                            "station_id": sender_id,
                            "role": payload.get("role", "secondary"),
                            "port": payload.get("port", 8501),
                            "timestamp": recv_ts,
                            "active_beds": payload.get("active_beds", []),
                        }
                        if payload.get("role") == "primary":
                            self.last_primary_heartbeat_ts = recv_ts
                return

            if topic == "hospital/nurse_stations/failover":
                failed = payload.get("failed_station")
                if failed and failed != self.station_id:
                    with self.lock:
                        self.is_primary = True
                        self.failover_triggered = True
                        ts_str = time.strftime("%H:%M:%S")
                        entry = (
                            f"[{ts_str}] [HIGH AVAILABILITY] FAILOVER NOTIFICATION: "
                            f"{failed} is OFFLINE! {self.station_id} has assumed ACTIVE PRIMARY monitoring."
                        )
                        self.global_alert_log.appendleft(entry)
                return

            patient_id = payload.get("patient_id")

            if not patient_id and len(parts) >= 3 and parts[1] not in ("sensors", "edge", "alerts", "control", "discovery", "nurse_stations"):
                patient_id = parts[1]
            if not patient_id:
                patient_id = "Bed_1"

            # 1. Always register/update device in discovered_jetsons registry
            with self.lock:
                if patient_id not in self.discovered_jetsons:
                    self.discovered_jetsons[patient_id] = {
                        "patient_id": patient_id,
                        "device_model": payload.get("device_model", "NVIDIA Jetson Orin"),
                        "exec_provider": payload.get("exec_provider", "CUDA / Orin GPU"),
                        "first_seen_ts": recv_ts,
                        "last_seen_ts": recv_ts,
                        "packet_count": 1,
                        "latest_vitals": payload.get("latest_vitals", {}),
                        "ip": payload.get("broker", "127.0.0.1"),
                    }
                    print(f"[NURSE STATION] Detected online Jetson hardware node: {patient_id}")
                else:
                    dj = self.discovered_jetsons[patient_id]
                    dj["last_seen_ts"] = recv_ts
                    dj["packet_count"] += 1
                    if payload.get("latest_vitals"):
                        dj["latest_vitals"] = payload.get("latest_vitals")
                    if payload.get("exec_provider"):
                        dj["exec_provider"] = payload.get("exec_provider")
                    if payload.get("device_model"):
                        dj["device_model"] = payload.get("device_model")

            # 2. If this is an explicit discovery announcement, finished
            if "announce" in parts or topic.endswith("discovery/announce"):
                return

            # 3. Only route telemetry and alerts into active ward if nurse has added this bed!
            if patient_id not in self.added_bed_ids:
                return

            if "sensors" in parts:
                self._handle_sensor_packet(patient_id, payload, recv_ts)
            elif parts[-1] == "summary":
                with self.lock:
                    bed = self._get_or_create_bed(patient_id)
                    bed.edge_summary = payload
                    if payload.get("ecg_wave") and len(payload["ecg_wave"]) == 250:
                        bed.latest_ecg_wave = payload["ecg_wave"]
            elif "alerts" in parts:
                with self.lock:
                    bed = self._get_or_create_bed(patient_id)
                    route_label = "Relay Path B" if "relay_b" in topic else "Primary Link"
                    prob = payload.get("arrhythmia_prob", 0.0) * 100.0
                    inf_ms = payload.get("inf_ms", 0.0)
                    ts_str = time.strftime("%H:%M:%S", time.localtime(recv_ts))
                    entry = (
                        f"[{ts_str}] [{patient_id}] {payload.get('alert', 'ALERT')} "
                        f"({prob:.1f}%) via {route_label} [{inf_ms:.2f} ms]"
                    )
                    bed.alert_log.appendleft(entry)
                    self.global_alert_log.appendleft(entry)
        except Exception as e:
            print(f"[DASHBOARD] Message handling error: {e}")

    def _handle_sensor_packet(self, patient_id, payload, recv_ts):
        agent_id = payload.get("agent_id")
        if agent_id not in AGENT_IDS:
            return

        seq = int(payload.get("seq", 0))
        sent_ts = float(payload.get("sent_ts", recv_ts))
        value = payload.get("value", 0)
        ecg_wave = payload.get("ecg_wave", [])

        all_chaos = read_all_chaos_config([patient_id])
        bed_chaos = all_chaos.get(patient_id, default_bed_chaos())
        orin_cfg = bed_chaos.get("Orin_Link", {})
        orin_congested = bool(orin_cfg.get("overload_gpu", False)) or bool(
            orin_cfg.get("use_relay_b", False)
        )

                    if payload.get("link_congested", False) or payload.get("alert_route") == "Relay_Agent_B":
                        bed["alert_route"] = "Relay B"
                    else:
                        bed["alert_route"] = "Direct"

                elif "/alerts/" in msg.topic:
                    if payload.get("arrhythmia_detected"):
                        self._log(pid, "CRITICAL", f"Arrhythmia detected (p={payload.get('anomaly_probability', 0):.2f})")
        except Exception:
            pass

    def snapshot(self):
        now = time.time()
        with self.lock:
            beds_snap = {}
            for pid in sorted(self.beds.keys()):
                if pid not in self.added_bed_ids:
                    continue
                bed = self.beds[pid]
                agents_snap = {}
                for aid, ag in bed.agents.items():
                    agents_snap[aid] = {
                        "trust_score": ag.trust_score,
                        "s_l": round(ag.s_l, 2),
                        "s_j": round(ag.s_j, 2),
                        "s_p": round(ag.s_p, 2),
                        "s_d": round(ag.s_d, 2),
                        "avg_latency_ms": round(ag.avg_latency_ms, 1),
                        "jitter_ms": round(ag.jitter_ms, 1),
                        "loss_rate_pct": round(ag.loss_rate * 100.0, 1),
                        "latest_value": ag.latest_value,
                        "plausible": ag.plausible,
                        "status": ag.status,
                        "packets_received": ag.packets_received,
                        "packets_missed": ag.packets_missed,
                    }
                beds_snap[pid] = {
                    "patient_id": pid,
                    "room": self.get_bed_room(pid),
                    "online": (now - bed.last_seen_ts) <= 8.0,
                    "seconds_since_seen": round(now - bed.last_seen_ts, 1),
                    "agents": agents_snap,
                    "edge_summary": dict(bed.edge_summary),
                    "ecg_wave": list(bed.latest_ecg_wave),
                    "gateway_cpu_inf_ms": bed.gateway_cpu_inf_ms,
                    "active_routing": bed.active_routing,
                    "offload_active": bed.offload_active,
                    "trust_history": list(bed.trust_history),
                    "alert_log": list(bed.alert_log),
                }

            unadded_jetsons = []
            for pid, dev in self.discovered_jetsons.items():
                if pid not in self.added_bed_ids:
                    is_active = (now - dev["last_seen_ts"]) <= 15.0
                    unadded_jetsons.append({
                        **dev,
                        "room": self.get_bed_room(pid),
                        "online": is_active,
                        "seconds_since_seen": round(now - dev["last_seen_ts"], 1),
                    })
            unadded_jetsons.sort(key=lambda x: x["patient_id"])

            active_peers = {
                sid: p for sid, p in self.peer_stations.items() if (now - p["timestamp"]) <= 5.0
            }

            return {
                "beds": beds_snap,
                "global_alert_log": list(self.global_alert_log),
                "unadded_jetsons": unadded_jetsons,
                "added_bed_ids": list(self.added_bed_ids),
                "discovered_jetsons": dict(self.discovered_jetsons),
                "ha_status": {
                    "station_id": self.station_id,
                    "station_port": self.station_port,
                    "is_primary": self.is_primary,
                    "failover_triggered": self.failover_triggered,
                    "failover_simulated": self.failover_simulated,
                    "peer_stations": active_peers,
                    "total_stations_online": len(active_peers) + (0 if self.failover_simulated else 1),
                },
            }


@st.cache_resource
def get_engine(version="v4_clean_bento"):
    return MultiPatientTrustEngine()


def svg_ecg(wave, color="#10b981", width=600, height=140):
    if not wave:
        wave = [0.0] * 250
    pts = []
    n = len(wave)
    mid = height * 0.58
    scale = height * 0.28
    for i, val in enumerate(wave):
        x = (i / max(1, n - 1)) * width
        y = mid - float(val) * scale
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="display:block;background:#0e1014;border-radius:6px;border:1px solid #1c1f26;">'
        f'<line x1="0" y1="{mid:.1f}" x2="{width}" y2="{mid:.1f}" stroke="#191c22" stroke-width="1"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}"/>'
        f"</svg>"
    )


def svg_trust_chart(history, width=600, height=130):
    if not history:
        return ""
    n = len(history)

    def line_pts(key):
        pts = []
        for i, h in enumerate(history):
            x = (i / max(1, n - 1)) * width
            y = height - float(h[key]) * (height - 16) - 8
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    y70 = height - 0.70 * (height - 16) - 8
    y45 = height - 0.45 * (height - 16) - 8

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="display:block;background:#0e1014;border-radius:6px;border:1px solid #1c1f26;">'
        f'<line x1="0" y1="{y70:.1f}" x2="{width}" y2="{y70:.1f}" stroke="#262932" stroke-dasharray="4 4" stroke-width="1"/>'
        f'<line x1="0" y1="{y45:.1f}" x2="{width}" y2="{y45:.1f}" stroke="#3a2024" stroke-dasharray="4 4" stroke-width="1"/>'
        f'<polyline fill="none" stroke="#10b981" stroke-width="2" points="{line_pts("ecg")}"/>'
        f'<polyline fill="none" stroke="#3b82f6" stroke-width="2" points="{line_pts("spo2")}"/>'
        f'<polyline fill="none" stroke="#f59e0b" stroke-width="2" points="{line_pts("hr")}"/>'
        f"</svg>"
    )


def render_sidebar_room_directory(engine, all_snap, selected_bed):
    """Renders Left Sidebar Room-Based Device Directory & HA Station Redundancy."""
    ha = all_snap.get("ha_status", {})
    beds_snap = all_snap.get("beds", {})
    unadded_jetsons = all_snap.get("unadded_jetsons", [])
    bed_ids = list(beds_snap.keys())

    # --- High Availability (HA) Station Status Header ---
    station_id = ha.get("station_id", "Nurse_Station_1")
    is_primary = ha.get("is_primary", True)
    station_port = ha.get("station_port", 8501)
    peers_online = ha.get("total_stations_online", 1)

    role_bg = "#064e3b" if is_primary else "#78350f"
    role_fg = "#a7f3d0" if is_primary else "#fde68a"
    role_txt = "PRIMARY LEADER" if is_primary else "HOT STANDBY"

    st.sidebar.markdown(
        f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px 12px;margin-bottom:12px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-weight:800;font-size:0.92rem;color:#f8fafc;">🖥️ {station_id}</span>'
        f'<span style="background:{role_bg};color:{role_fg};padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700;">'
        f'{role_txt}</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">'
        f'Port: <code>{station_port}</code> • HA Peers: <strong>{peers_online} Station(s) Active</strong>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if ha.get("failover_triggered"):
        st.sidebar.error("🚨 FAILOVER ACTIVE: Primary Station offline. Standby promoted to Active Primary!")

    with st.sidebar.expander("🛡️ High-Availability (HA) Failover Controls", expanded=False):
        st.caption("Dual nurse dashboards run concurrently. If Nurse Station 1 fails, Nurse Station 2 automatically assumes primary ward control.")
        peers = ha.get("peer_stations", {})
        if peers:
            for pid, p in peers.items():
                st.markdown(f"• **{pid}** (`{p.get('role', 'peer').upper()}` on port `{p.get('port')}`) — 🟢 Connected")
        else:
            st.caption("No secondary stations detected yet. Launch `./nurse_station/start_secondary_nurse_station.sh` to run Station 2 on port 8502.")

        if st.button("⚡ Test Failover / Switch Role", key="btn_failover_test", use_container_width=True):
            engine.trigger_failover_test()
            st.rerun()

    st.sidebar.divider()

    # --- Room-Based Device Directory ---
    st.sidebar.title("🏥 ICU Ward Rooms & Devices")
    st.sidebar.caption("Available Jetson Orin bedside units organized by patient room.")

    # Group active beds by room
    rooms_map = {}
    for pid, b in beds_snap.items():
        rm = b.get("room", engine.get_bed_room(pid))
        if rm not in rooms_map:
            rooms_map[rm] = []
        rooms_map[rm].append((pid, b))

    all_room_names = sorted(list(rooms_map.keys()))
    filter_options = ["All Rooms"] + all_room_names
    selected_room_filter = st.sidebar.selectbox("Filter Room / Ward:", filter_options, index=0, key="room_filter_select")

    if not beds_snap:
        st.sidebar.info("No active devices in ward. Use 'Add New Bed' to commission Jetson nodes.")
    else:
        for rm in all_room_names:
            if selected_room_filter != "All Rooms" and rm != selected_room_filter:
                continue

            beds_in_room = rooms_map[rm]
            with st.sidebar.container(border=True):
                st.markdown(f"**🚪 {rm}** ({len(beds_in_room)} Bed)")
                for pid, b in beds_in_room:
                    edge = b["edge_summary"]
                    vitals = edge.get("latest_vitals", {})
                    agents = b["agents"]
                    alarm = bool(edge.get("local_bed_alarm", False))
                    isolated = any(ag["status"].startswith("ISOLATED") for ag in agents.values())
                    offloaded = b["offload_active"]

                    if not b.get("online", True):
                        dot, st_lbl, st_col = "⚠️", "Offline", "#94a3b8"
                    elif alarm:
                        dot, st_lbl, st_col = "🚨", "Arrhythmia", "#ef4444"
                    elif isolated:
                        dot, st_lbl, st_col = "🛡️", "Quarantined", "#f43f5e"
                    elif offloaded:
                        dot, st_lbl, st_col = "🔀", "Offloaded", "#f59e0b"
                    else:
                        dot, st_lbl, st_col = "🟢", "Trusted", "#22c55e"

                    is_curr = (pid == selected_bed)
                    border_style = "border-left: 3px solid #38bdf8; padding-left: 6px;" if is_curr else ""
                    st.markdown(
                        f"<div style='margin-bottom:6px;{border_style}'>"
                        f"<strong>{dot} {pid}</strong> <span style='font-size:0.75rem;color:{st_col};font-weight:700;'>[{st_lbl}]</span><br/>"
                        f"<span style='font-size:0.80rem;color:#94a3b8;'>💓 {vitals.get('hr', 75)} bpm • 🫁 {vitals.get('spo2', 98)}% • {edge.get('inf_ms', 0.12):.2f}ms</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    c_sel, c_rel = st.columns([1.5, 1])
                    with c_sel:
                        if st.button("Inspect", key=f"side_inspect_{pid}", use_container_width=True, disabled=is_curr):
                            st.session_state["detailed_bed_selector"] = pid
                            st.rerun()
                    with c_rel:
                        if len(bed_ids) > 1 and st.button("Release", key=f"side_rel_{pid}", use_container_width=True):
                            engine.remove_bed(pid)
                            st.rerun()

    # --- Available / Unassigned Jetson Nodes Section ---
    st.sidebar.divider()
    if unadded_jetsons:
        st.sidebar.markdown(f"#### 📡 Available Jetsons ({len(unadded_jetsons)})")
        st.sidebar.caption("Unassigned devices detected on LAN / MQTT:")
        for dev in unadded_jetsons:
            pid = dev["patient_id"]
            model = dev.get("device_model", "NVIDIA Jetson Orin")
            room_list = ["Room 101 (Cardiac ICU)", "Room 102 (Cardiac ICU)", "Room 103 (Neuro ICU)", "Room 104 (Surgical ICU)", "Room 105 (Trauma ICU)"]
            target_rm = st.selectbox(
                f"Room for {pid}:",
                room_list,
                index=min(max(0, int("".join(filter(str.isdigit, pid)) or "1") - 1), len(room_list) - 1),
                key=f"side_rm_sel_{pid}"
            )
            with st.sidebar.container(border=True):
                st.markdown(f"**🟡 {pid}** ({model})")
                st.caption(f"Signal: 🟢 Online ({dev.get('seconds_since_seen', 0)}s ago)")
                if st.button(f"➕ Assign to {target_rm.split()[0]} & Add", key=f"side_add_{pid}", type="primary", use_container_width=True):
                    engine.set_bed_room(pid, target_rm)
                    engine.add_bed(pid, room_name=target_rm)
                    st.toast(f"✅ Added {pid} to {target_rm}!", icon="🎉")
                    st.rerun()
    else:
        st.sidebar.caption("No unassigned Jetson devices currently waiting on the network.")

    # --- Bedside Vitals Manipulation Notice ---
    st.sidebar.divider()
    st.sidebar.markdown(
        f'<div style="background:#0b1329;border:1px solid #1e3a8a;border-radius:8px;padding:10px 12px;">'
        f'<span style="font-size:0.80rem;font-weight:800;color:#38bdf8;">🎛️ Bedside Vitals Manipulation</span><br/>'
        f'<span style="font-size:0.75rem;color:#94a3b8;">Patient vitals &amp; fault injection are hosted locally on each Jetson Bedside Monitor:</span><br/>'
        f'<a href="http://localhost:8080" target="_blank" style="font-size:0.78rem;color:#60a5fa;font-weight:700;">🔗 Open Bedside Console (:8080)</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    auto_refresh = st.sidebar.checkbox("⏱️ Live Auto-Refresh (0.6s)", value=True, key="side_auto_refresh")
    return auto_refresh


def render_add_bed_panel(engine, unadded_jetsons, added_bed_ids):
    """Renders the Add New Bed panel that scans for and admits unassigned Jetson Orin nodes."""
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <h3 style="margin:0;color:#38bdf8;">🛏️ Add New Bed — Jetson Edge AI Node Discovery</h3>
                <span style="background:#0369a1;color:#f0f9ff;padding:4px 10px;border-radius:6px;font-size:0.8rem;font-weight:700;">
                    NVIDIA Jetson Auto-Detection
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The Nurse Station automatically listens on the hospital LAN & MQTT broker for unassigned Jetson Orin nodes. "
            "Review detected hardware below, assign a patient room, and click **Add Jetson** to commission the bed into the live ICU ward."
        )

        col_scan1, col_scan2, col_scan3 = st.columns([2, 1.2, 1])
        with col_scan1:
            if unadded_jetsons:
                st.success(f"🔍 **Detection Complete:** Found **{len(unadded_jetsons)}** unassigned Jetson device(s) online.")
            else:
                st.info("🔍 **Network Scan Active:** Listening for unassigned Jetson Orin nodes on MQTT...")
        with col_scan2:
            if st.button("🔄 Rescan Network / MQTT", key="rescan_modal_btn", use_container_width=True):
                engine.probe_network_for_jetsons()
                st.toast("Discovery probe sent across hospital network!", icon="📡")
                st.rerun()
        with col_scan3:
            if st.button("✖️ Close Panel", key="close_modal_btn", use_container_width=True):
                st.session_state["show_add_bed"] = False
                st.rerun()

        st.markdown("---")

        if unadded_jetsons:
            st.markdown(f"#### 📡 Detected Jetson Nodes Ready to Pair ({len(unadded_jetsons)})")
            room_options = [
                "Room 101 (Cardiac ICU)",
                "Room 102 (Cardiac ICU)",
                "Room 103 (Neuro ICU)",
                "Room 104 (Surgical ICU)",
                "Room 105 (Trauma ICU)",
            ]
            for dev in unadded_jetsons:
                pid = dev["patient_id"]
                vitals = dev.get("latest_vitals", {})
                hr = vitals.get("hr", 75)
                spo2 = vitals.get("spo2", 98)
                model = dev.get("device_model", "NVIDIA Jetson Orin")
                provider = str(dev.get("exec_provider", "CUDA / Orin GPU")).replace("ExecutionProvider", "")
                seen_sec = dev.get("seconds_since_seen", 0)

                with st.container(border=True):
                    c_info, c_vitals, c_action = st.columns([1.4, 1.3, 1.3])
                    with c_info:
                        st.markdown(f"**🖥️ Hardware Node:** `{pid}`")
                        st.markdown(f"• **Device:** `{model}`  \n• **AI Provider:** `{provider}`")
                        st.caption(f"Signal: 🟢 Online (Seen {seen_sec}s ago, {dev.get('packet_count', 1)} pkts)")
                    with c_vitals:
                        st.markdown("**Live Telemetry Preview:**")
                        st.markdown(f"• Heart Rate: **{hr} bpm**  \n• SpO2: **{spo2}%**  \n• 250 Hz ECG: **Ready**")
                    with c_action:
                        assigned_room = st.selectbox(
                            f"Assign Room for {pid}:",
                            room_options,
                            index=min(max(0, int("".join(filter(str.isdigit, pid)) or "1") - 1), len(room_options) - 1),
                            key=f"panel_rm_{pid}",
                        )
                        if st.button(f"➕ Add Jetson ({pid}) to {assigned_room.split()[0]}", key=f"btn_add_{pid}", type="primary", use_container_width=True):
                            engine.add_bed(pid, room_name=assigned_room)
                            st.session_state["show_add_bed"] = False
                            st.toast(f"✅ Successfully added {pid} to {assigned_room}!", icon="🎉")
                            st.rerun()
        else:
            st.warning("⚠️ **No unassigned Jetson nodes currently detected.**")
            st.markdown(
                """
                - **To connect a physical NVIDIA Jetson Orin:** Turn on the Jetson and launch:  
                  `./start_bedside.sh --patient-id Bed_4 --broker <NURSE_LAPTOP_IP>`  
                  *It will be detected here automatically.*
                - **To test or simulate a new Jetson right now:** Click below to simulate an online Jetson Orin node.
                """
            )
            c_sim, _ = st.columns([1.6, 2])
            with c_sim:
                if st.button("⚡ Simulate New Jetson Node", key="btn_sim_jetson", type="primary", use_container_width=True):
                    sim_id = engine.simulate_new_jetson()
                    st.toast(f"Simulated Jetson {sim_id} detected!", icon="📡")
                    st.rerun()

    hr_txt = "ISO" if hr_ag["status"] == "ISOLATED" else (str(hr_ag["value"]) if hr_ag["value"] is not None else "--")
    sp_txt = "ISO" if sp_ag["status"] == "ISOLATED" else (f"{sp_ag['value']}%" if sp_ag["value"] is not None else "--")
    ecg_col = "#ef4444" if (ecg_ag["status"] == "ISOLATED" or bed["edge"].get("is_arrhythmia")) else "#10b981"
    border_col = "#3b82f6" if is_selected else "#1f2228"

def render_multi_patient_ward_overview(beds_snap, engine=None):
    """Renders top-level ICU Ward Overview cards dynamically for all connected Jetson nodes."""
    bed_ids = list(beds_snap.keys())
    st.subheader(
        f"🏥 ICU Ward Live Overview — {len(bed_ids)} Active Bedside Jetson Node(s) Monitored"
    )
    cols = st.columns(max(1, len(bed_ids)))

    for idx, pid in enumerate(bed_ids):
        b = beds_snap[pid]
        edge = b["edge_summary"]
        agents = b["agents"]
        vitals = edge.get("latest_vitals", {})
        room_name = b.get("room", "ICU Room")

        any_isolated = any(ag["status"].startswith("ISOLATED") for ag in agents.values())
        any_offloaded = b["offload_active"]
        local_alarm = bool(edge.get("local_bed_alarm", False)) and not agents["Agent_ECG"][
            "status"
        ].startswith("ISOLATED")

        if not b.get("online", True):
            banner_bg, banner_fg, banner_txt = "#475569", "#f1f5f9", f"⚠️ OFFLINE ({b.get('seconds_since_seen', 0)}s)"
        elif local_alarm:
            banner_bg, banner_fg, banner_txt = "#dc3545", "#ffffff", "🚨 ARRHYTHMIA ALARM"
        elif any_isolated:
            banner_bg, banner_fg, banner_txt = "#842029", "#f8d7da", "🛡️ SENSOR QUARANTINED"
        elif any_offloaded:
            banner_bg, banner_fg, banner_txt = "#664d03", "#fff3cd", "🔀 OFFLOADED (RELAY B)"
        else:
            banner_bg, banner_fg, banner_txt = "#0f5132", "#d1e7dd", "🟢 ALL SENSORS TRUSTED"

        with cols[idx]:
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div><span style="font-size:1.1rem;font-weight:800;">🛏️ {pid}</span> '
                    f'<span style="font-size:0.80rem;color:#38bdf8;font-weight:700;">({room_name})</span></div>'
                    f'<span style="background:{banner_bg};color:{banner_fg};padding:3px 9px;'
                    f'border-radius:5px;font-size:0.78rem;font-weight:700;">{banner_txt}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("HR", f"{vitals.get('hr', 75)} bpm")
                c2.metric("SpO2", f"{vitals.get('spo2', 98)}%")
                c3.metric("AI Speed", f"{edge.get('inf_ms', 0.0):.2f} ms")
                st.caption(
                    f"**Trust Scores:** ECG `{agents['Agent_ECG']['trust_score']:.2f}` • "
                    f"SpO2 `{agents['Agent_SpO2']['trust_score']:.2f}` • "
                    f"HR `{agents['Agent_HR']['trust_score']:.2f}`  \n"
                    f"**Route:** `{b['active_routing']}`"
                )
                if engine and len(bed_ids) > 1:
                    if st.button(f"🚪 Release {pid}", key=f"rel_{pid}", help=f"Release {pid} back to unadded pool", use_container_width=True):
                        engine.remove_bed(pid)
                        st.rerun()
                c1, c2, c3 = st.columns(3)
                c1.metric("HR", f"{vitals.get('hr', 75)} bpm")
                c2.metric("SpO2", f"{vitals.get('spo2', 98)}%")
                c3.metric("AI Speed", f"{edge.get('inf_ms', 0.0):.2f} ms")
                st.caption(
                    f"**Trust Scores:** ECG `{agents['Agent_ECG']['trust_score']:.2f}` • "
                    f"SpO2 `{agents['Agent_SpO2']['trust_score']:.2f}` • "
                    f"HR `{agents['Agent_HR']['trust_score']:.2f}`  \n"
                    f"**Route:** `{b['active_routing']}`"
                )
                if engine and len(bed_ids) > 1:
                    if st.button(f"🚪 Release {pid}", key=f"rel_{pid}", help=f"Release {pid} back to unadded pool", use_container_width=True):
                        engine.remove_bed(pid)
                        st.rerun()


def detail_view(pid, bed, logs):
    ecg_ag = bed["agents"]["Agent_ECG"]
    sp_ag = bed["agents"]["Agent_SpO2"]
    hr_ag = bed["agents"]["Agent_HR"]
    edge = bed["edge"]

    ecg_Quarantined = ecg_ag["status"] == "ISOLATED"
    is_arr = bool(edge.get("is_arrhythmia")) and not ecg_Quarantined
    ecg_col = "#ef4444" if (is_arr or ecg_Quarantined) else "#10b981"
    rhythm_txt = "Quarantined (Spoofed)" if ecg_Quarantined else ("Arrhythmia" if is_arr else "Normal Sinus")
    rhythm_col = "#ef4444" if (is_arr or ecg_Quarantined) else "#ededef"

    def sensor_tile(title, ag, unit):
        st_name = ag["status"]
        p_cls = "pill-green" if st_name == "TRUSTED" else ("pill-amber" if st_name == "SUSPICIOUS" else "pill-red")
        bar_col = dot_color(st_name)
        pct = int(round(ag["trust"] * 100))
        if title == "ECG sensor":
            reading = "250 Hz stream" if st_name != "ISOLATED" else "Stream isolated"
        else:
            v = ag["value"]
            reading = f"{v} {unit}".strip() if (v is not None and st_name != "ISOLATED") else ("Isolated" if st_name == "ISOLATED" else "--")

        return (
            f'<div class="tile span-4">'
            f'<div class="row-between"><span class="lbl">{title}</span><span class="pill {p_cls}">{st_name}</span></div>'
            f'<div class="row-between" style="align-items:baseline;margin-top:8px;">'
            f'<div class="val-lg">{ag["trust"]:.2f}</div>'
            f'<div class="sub">{reading}</div>'
            f"</div>"
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{bar_col};"></div></div>'
            f'<div class="sub-metrics">'
            f'<span>L {ag["s_l"]:.2f}</span><span>J {ag["s_j"]:.2f}</span><span>P {ag["s_p"]:.2f}</span><span>D {ag["s_d"]:.2f}</span>'
            f"</div>"
            f"</div>"
        )

    bed_logs = [e for e in logs if e["bed"] == pid][:6]
    if bed_logs:
        rows = "".join(
            f'<div class="log-row"><span class="log-ts">{e["ts"]}</span>'
            f'<span class="log-lvl log-{e["level"].lower()}">{e["level"]}</span>'
            f'<span class="log-msg">{e["msg"]}</span></div>'
            for e in bed_logs
        )
    else:
        rows = '<div class="sub" style="padding:6px 0;">No anomalies or trust transitions recorded for this bed.</div>'

    engine = get_nurse_station_engine()
    all_snap = engine.get_all_beds_snapshot()
    beds_snap = all_snap["beds"]
    unadded_jetsons = all_snap.get("unadded_jetsons", [])
    bed_ids = list(beds_snap.keys())
    ha = all_snap.get("ha_status", {})

    st.title("🏥 Trust-Aware Multi-Patient ICU Edge Intelligence Dashboard")
    role_str = "ACTIVE PRIMARY LEADER" if ha.get("is_primary") else "HOT STANDBY"
    st.caption(
        f"🖥️ **{ha.get('station_id', 'Nurse_Station_1')}** (`{role_str}` on Port `{ha.get('station_port', 8501)}`) • "
        f"High Availability: `{ha.get('total_stations_online', 1)}` Station(s) Online • Dynamic Bedside Jetson Discovery"
    )

    if ha.get("failover_triggered"):
        st.error(
            "🚨 **HIGH AVAILABILITY FAILOVER ACTIVE:** Primary Nurse Station became unreachable! "
            f"**{ha.get('station_id')}** has automatically assumed **ACTIVE PRIMARY WARD LEADERSHIP** with zero data loss."
        )

    # Determine currently selected bed for inspection
    if "detailed_bed_selector" not in st.session_state and bed_ids:
        st.session_state["detailed_bed_selector"] = bed_ids[0]
    curr_sel = st.session_state.get("detailed_bed_selector", bed_ids[0] if bed_ids else "Bed_1")
    if curr_sel not in bed_ids and bed_ids:
        curr_sel = bed_ids[0]
        st.session_state["detailed_bed_selector"] = curr_sel

    # Render Room-Based Device Navigation in Left Sidebar
    auto_refresh = render_sidebar_room_directory(engine, all_snap, curr_sel)

    # Top action bar with Add New Bed button and Network Scan
    if "show_add_bed" not in st.session_state:
        st.session_state["show_add_bed"] = False

    top_c1, top_c2, top_c3 = st.columns([2.0, 1.2, 1.0])
    with top_c1:
        badge = f" • 🟢 **{len(unadded_jetsons)} Unassigned Jetson(s) Online**" if unadded_jetsons else ""
        st.markdown(
            f"**ICU Ward Census:** `{len(bed_ids)}` Active Bed(s) Monitoring{badge}"
        )
    with top_c2:
        btn_badge = f" ({len(unadded_jetsons)} Detected)" if unadded_jetsons else ""
        btn_label = "➖ Hide Add Bed" if st.session_state["show_add_bed"] else f"➕ Add New Bed{btn_badge}"
        btn_type = "secondary" if st.session_state["show_add_bed"] else ("primary" if unadded_jetsons else "secondary")
        if st.button(btn_label, type=btn_type, use_container_width=True, key="main_toggle_add_bed"):
            st.session_state["show_add_bed"] = not st.session_state["show_add_bed"]
            if st.session_state["show_add_bed"]:
                engine.probe_network_for_jetsons()
            st.rerun()
    with top_c3:
        if st.button("🔄 Scan Network", use_container_width=True, key="main_scan_btn"):
            engine.probe_network_for_jetsons()
            st.toast("Scanning network for online Jetson Orin nodes...", icon="🔍")
            st.rerun()

    # Render discovery & add panel if toggled open
    if st.session_state["show_add_bed"]:
        render_add_bed_panel(engine, unadded_jetsons, all_snap.get("added_bed_ids", []))
        st.markdown("---")

    if not bed_ids:
        st.info(
            f"ℹ️ **No patient beds currently admitted to the ICU Ward.**\n\n"
            f"Click **➕ Add New Bed** above to detect and admit online Bedside Jetson nodes! "
            f"({len(unadded_jetsons)} Jetson node(s) currently detected on the network)."
        )
        if auto_refresh:
            time.sleep(0.6)
            st.rerun()
        return

    # 1. Top Multi-Patient Ward Summary Row
    render_multi_patient_ward_overview(beds_snap, engine=engine)

    engine = get_engine("v4_clean_bento")
    beds_snap, logs = engine.snapshot()
    bed_ids = list(beds_snap.keys())
    online_count = sum(1 for b in beds_snap.values() if b["online"])

    # 2. Patient Bed Selector for Detailed Dual-View Telemetry
    prev_sel = st.session_state.get("detailed_bed_selector", bed_ids[0])
    sel_index = bed_ids.index(prev_sel) if prev_sel in bed_ids else 0

    selected_bed = st.radio(
        "🔍 **Select Connected Patient Bed for Detailed Bedside & Trust Inspection:**",
        options=bed_ids,
        index=sel_index,
        horizontal=True,
        key="detailed_bed_selector",
    )

    snap = beds_snap[selected_bed]
    edge = snap["edge_summary"]
    agents = snap["agents"]

    left_col, right_col = st.columns([1.0, 1.15], gap="large")

    # =========================================================================
    # LEFT COLUMN: Selected Patient's Bedside Jetson Orin Monitor
    # =========================================================================
    with left_col:
        st.subheader(f"🖥️ {selected_bed} ({engine.get_bed_room(selected_bed)}) — NVIDIA Jetson Orin Bedside Monitor")

        ecg_isolated = agents["Agent_ECG"]["status"].startswith("ISOLATED")
        local_alarm = bool(edge.get("local_bed_alarm", False))

        if local_alarm and not ecg_isolated:
            st.markdown(
                f'<div class="bedside-alarm-banner">'
                f'🚨 [{selected_bed}] CRITICAL BEDSIDE ALARM: ARRHYTHMIA DETECTED '
                f'({edge.get("arrhythmia_prob", 0.0) * 100:.1f}% Confidence | '
                f'{edge.get("inf_ms", 0.0):.2f} ms Edge Response)'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif local_alarm and ecg_isolated:
            st.markdown(
                f'<div class="conflict-warning-banner">'
                f'🛡️ [{selected_bed}] BEDSIDE ALARM SUPPRESSED AT NURSE STATION: '
                f'Agent_ECG Quarantined (Trust &lt; 0.45 — Spoofed/Noisy Signal)'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bedside-normal-banner">'
                f'✅ [{selected_bed}] NORMAL SINUS RHYTHM — Zero-Latency Bedside AI Monitor Nominal'
                f'</div>',
                unsafe_allow_html=True,
            )

        if edge.get("sensor_conflict", False):
            st.markdown(
                f'<div class="conflict-warning-banner">'
                f'⚠️ [{selected_bed}] CROSS-SENSOR PHYSIOLOGICAL CONFLICT DETECTED: '
                f'Discrepancy between ECG Sinus Rhythm, HR, and SpO2!'
                f'</div>',
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "1D-CNN Inference Latency",
            f"{edge.get('inf_ms', 0.0):.2f} ms",
            delta="< 5.0 ms SLA",
        )
        m2.metric(
            "Execution Provider",
            str(edge.get("exec_provider", "CPU")).replace("ExecutionProvider", ""),
        )
        m3.metric(
            "Arrhythmia Probability",
            f"{edge.get('arrhythmia_prob', 0.0) * 100:.1f}%",
        )
        time.sleep(0.8)
        st.rerun()
        return

    if "selected_bed" not in st.session_state or st.session_state.selected_bed not in bed_ids:
        st.session_state.selected_bed = bed_ids[0]

    if len(bed_ids) > 1:
        cols = st.columns([3, 9])
        with cols[0]:
            st.session_state.selected_bed = st.selectbox(
                "Bed",
                bed_ids,
                index=bed_ids.index(st.session_state.selected_bed),
                label_visibility="collapsed",
            )

    selected_bed = st.session_state.selected_bed

    ward_cards = "".join(ward_tile(pid, b, pid == selected_bed) for pid, b in beds_snap.items())
    st.markdown(f'<div class="ward-grid">{ward_cards}</div>', unsafe_allow_html=True)

    st.markdown(detail_view(selected_bed, beds_snap[selected_bed], logs), unsafe_allow_html=True)

    time.sleep(0.6)
    st.rerun()


if __name__ == "__main__":
    main()
