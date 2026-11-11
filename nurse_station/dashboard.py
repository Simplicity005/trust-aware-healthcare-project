#!/usr/bin/env python3
"""
Multi-Patient Central Nurse Station, Mathematical Trust Engine & Live Dashboard
(Runs on the Nurse's Table Gateway Laptop)
Dynamically discovers and adds patient beds as Bedside Jetson Orin units connect
over MQTT (no hardcoded bed count!), computes real-time composite Trust Scores
T_i(t) in [0.0, 1.0] per sensor per bed, manages dynamic compute offloading &
alert rerouting, quarantines Byzantine sensors, and provides a per-patient Live
Chaos Controller over MQTT.
"""

import os
import time
import json
import threading
from collections import deque
import numpy as np
import pandas as pd
import onnxruntime as ort
import paho.mqtt.client as mqtt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAOS_FILE = os.path.join(BASE_DIR, "chaos.json")
BEDSIDE_CHAOS_FILE = os.path.join(os.path.dirname(BASE_DIR), "jetson_bedside", "chaos.json")
MODEL_FILE = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
AGENT_IDS = ["Agent_ECG", "Agent_SpO2", "Agent_HR"]


def default_bed_chaos():
    return {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False},
    }


def read_all_chaos_config(active_bed_ids=None):
    """Reads multi-patient chaos.json dynamically for whatever beds are active."""
    data = {}
    if os.path.exists(CHAOS_FILE):
        try:
            with open(CHAOS_FILE, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                if "Agent_ECG" in loaded:
                    data["Bed_1"] = loaded
                else:
                    data = loaded
        except Exception:
            data = {}

    if active_bed_ids:
        for bed in active_bed_ids:
            if bed not in data:
                data[bed] = default_bed_chaos()
    return data


def write_all_chaos_config(config):
    """Writes updated chaos configuration atomically to nurse_station/chaos.json (and jetson_bedside/chaos.json if local)."""
    for target_file in [CHAOS_FILE, BEDSIDE_CHAOS_FILE]:
        target_dir = os.path.dirname(target_file)
        if os.path.isdir(target_dir):
            try:
                tmp_path = target_file + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(config, f, indent=2)
                os.replace(tmp_path, target_file)
            except Exception:
                pass


class AgentTrustTracker:
    """
    Mathematical Trust Engine for a single autonomous medical sensor agent.
    Maintains a sliding window of N = 20 packets and computes:
      - Latency Score (S_L):      max(0.0, 1.0 - avg_latency_ms / 400.0)
      - Jitter Score (S_J):       max(0.0, 1.0 - (max_L - min_L) / 200.0)
      - Packet Loss Score (S_P):  max(0.0, 1.0 - 2.5 * loss_rate)
      - Plausibility Score (S_D): Asymmetric EWMA decay (-0.35 on violation, +0.08 on valid)
      - Composite Trust (T_i):    0.25*S_L + 0.15*S_J + 0.25*S_P + 0.35*S_D
    """

    def __init__(self, agent_id, window_size=20):
        self.agent_id = agent_id
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.delivery_window = deque(maxlen=window_size)
        self.last_seq = None
        self.last_recv_ts = None

        self.s_l = 1.0
        self.s_j = 1.0
        self.s_p = 1.0
        self.s_d = 1.0
        self.trust_score = 1.0

        self.avg_latency_ms = 10.0
        self.recent_latency_ms = 10.0
        self.jitter_ms = 0.0
        self.loss_rate = 0.0
        self.latest_value = 75 if agent_id == "Agent_HR" else (98 if agent_id == "Agent_SpO2" else 0.0)
        self.latest_ecg_wave = []
        self.plausible = True
        self.status = "TRUSTED (Orin GPU)"
        self.packets_received = 0
        self.packets_missed = 0

    def update(self, seq, sent_ts, recv_ts, value, ecg_wave=None, cross_sensor_conflict=False):
        self.packets_received += 1
        self.last_recv_ts = recv_ts

        # 1. Latency & Jitter calculation
        latency_ms = max(1.0, (recv_ts - sent_ts) * 1000.0)
        self.latencies.append(latency_ms)
        self.avg_latency_ms = float(np.mean(self.latencies))
        self.recent_latency_ms = float(np.mean(list(self.latencies)[-3:]))
        if len(self.latencies) >= 2:
            self.jitter_ms = float(np.max(self.latencies) - np.min(self.latencies))
        else:
            self.jitter_ms = 0.0

        self.s_l = max(0.0, 1.0 - (self.avg_latency_ms / 400.0))
        self.s_j = max(0.0, 1.0 - (self.jitter_ms / 200.0))

        # 2. Sequence gap & Packet Loss Score calculation
        missed = 0
        if self.last_seq is not None and seq > self.last_seq:
            gap = seq - self.last_seq - 1
            if gap > 0:
                missed = gap
                self.packets_missed += missed
        self.last_seq = seq
        self.delivery_window.append((1, missed))

        total_recv = sum(r for r, _ in self.delivery_window)
        total_miss = sum(m for _, m in self.delivery_window)
        total_expected = total_recv + total_miss
        self.loss_rate = (total_miss / total_expected) if total_expected > 0 else 0.0
        self.s_p = max(0.0, 1.0 - 2.5 * self.loss_rate)

        # 3. Physiological Plausibility Score (S_D) with asymmetric decay
        is_plausible = True
        if self.agent_id == "Agent_HR":
            self.latest_value = value
            if value < 40 or value > 220:
                is_plausible = False
        elif self.agent_id == "Agent_SpO2":
            self.latest_value = value
            if value < 70 or value > 100:
                is_plausible = False
        elif self.agent_id == "Agent_ECG":
            if ecg_wave and len(ecg_wave) == 250:
                self.latest_ecg_wave = ecg_wave
                wave_np = np.array(ecg_wave, dtype=np.float32)
                variance = float(np.var(wave_np))
                peak = float(np.max(np.abs(wave_np)))
                self.latest_value = round(variance, 4)
                if variance > 0.35 or peak > 2.5:
                    is_plausible = False

        if cross_sensor_conflict and not is_plausible:
            is_plausible = False

        self.plausible = is_plausible
        if not is_plausible:
            self.s_d = max(0.0, self.s_d - 0.35)
        else:
            self.s_d = min(1.0, self.s_d + 0.08)

        # 4. Composite Trust Score T_i(t)
        self.trust_score = round(
            0.25 * self.s_l + 0.15 * self.s_j + 0.25 * self.s_p + 0.35 * self.s_d,
            4,
        )
        return self.trust_score

    def evaluate_status(self, orin_link_congested=False):
        if self.trust_score < 0.45:
            self.status = "ISOLATED (Quarantined)"
        elif (
            self.trust_score < 0.70
            or self.avg_latency_ms > 180.0
            or self.recent_latency_ms > 180.0
            or orin_link_congested
        ):
            self.status = "OFFLOADED (Gateway CPU Fallback)"
        else:
            self.status = "TRUSTED (Orin GPU)"
        return self.status


class PatientBedMonitor:
    """Maintains all sensor trust trackers, bedside AI summary, and routing state for one patient bed."""

    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.last_seen_ts = time.time()
        self.agents = {aid: AgentTrustTracker(aid) for aid in AGENT_IDS}
        self.edge_summary = {
            "patient_id": patient_id,
            "inf_ms": 0.30,
            "exec_provider": "CPUExecutionProvider",
            "arrhythmia_prob": 0.02,
            "local_bed_alarm": False,
            "sensor_conflict": False,
            "overload_gpu": False,
            "latest_vitals": {"hr": 75, "spo2": 98, "ecg_variance": 0.04},
            "routing_path": "primary",
            "timestamp": time.time(),
        }
        self.latest_ecg_wave = self._default_ecg_wave()
        self.gateway_cpu_inf_ms = 0.0
        self.active_routing = "Primary Direct Link"
        self.offload_active = False
        self.last_offload_published = None
        self.trust_history = deque(maxlen=60)
        self.alert_log = deque(maxlen=12)

    @staticmethod
    def _default_ecg_wave():
        t = np.linspace(0, 1, 250)
        ecg = (
            0.15 * np.exp(-((t - 0.20) ** 2) / (2 * 0.025 ** 2))
            - 0.15 * np.exp(-((t - 0.35) ** 2) / (2 * 0.010 ** 2))
            + 1.25 * np.exp(-((t - 0.40) ** 2) / (2 * 0.012 ** 2))
            - 0.25 * np.exp(-((t - 0.45) ** 2) / (2 * 0.012 ** 2))
            + 0.30 * np.exp(-((t - 0.65) ** 2) / (2 * 0.040 ** 2))
        )
        return np.round(ecg, 4).tolist()


class NurseStationEngine:
    """
    Dynamic Multi-Patient MQTT listener, Trust Engine coordinator, and Gateway CPU Fallback engine.
    Starts with zero hardcoded beds and registers beds dynamically as Jetsons connect!
    """

    def __init__(self, broker="127.0.0.1", port=1883):
        self.broker = broker
        self.port = port
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

        self.cpu_session = None
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
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        else:
            self.client = mqtt.Client(client_id=client_id)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[DASHBOARD] MQTT connect error: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
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
        recv_ts = time.time()
        try:
            topic = msg.topic
            parts = topic.split("/")
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

        with self.lock:
            bed = self._get_or_create_bed(patient_id)
            conflict = bool(bed.edge_summary.get("sensor_conflict", False))
            tracker = bed.agents[agent_id]
            tracker.update(
                seq=seq,
                sent_ts=sent_ts,
                recv_ts=recv_ts,
                value=value,
                ecg_wave=ecg_wave,
                cross_sensor_conflict=conflict,
            )

            if agent_id == "Agent_ECG" and ecg_wave and len(ecg_wave) == 250:
                bed.latest_ecg_wave = ecg_wave
                if bed.offload_active and self.cpu_session is not None:
                    t0 = time.perf_counter()
                    wave_np = np.array(ecg_wave, dtype=np.float32).reshape(1, 1, 250)
                    _ = self.cpu_session.run(None, {"ecg_input": wave_np})
                    bed.gateway_cpu_inf_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            any_offloaded = False
            for aid, ag in bed.agents.items():
                st_badge = ag.evaluate_status(orin_link_congested=orin_congested)
                if st_badge.startswith("OFFLOADED"):
                    any_offloaded = True
                elif st_badge.startswith("ISOLATED") and ag.packets_received % 5 == 1:
                    ts_str = time.strftime("%H:%M:%S", time.localtime(recv_ts))
                    msg_str = (
                        f"[{ts_str}] [{patient_id}] QUARANTINED {aid} "
                        f"(Trust={ag.trust_score:.2f} < 0.45) — Muted clinical alerts"
                    )
                    if not bed.alert_log or bed.alert_log[0] != msg_str:
                        bed.alert_log.appendleft(msg_str)
                        self.global_alert_log.appendleft(msg_str)

            bed.offload_active = any_offloaded or orin_congested
            bed.active_routing = (
                "Relay Path B (Rerouted)" if bed.offload_active else "Primary Direct Link"
            )

            if agent_id == "Agent_ECG":
                bed.trust_history.append(
                    {
                        "time": time.strftime("%H:%M:%S", time.localtime(recv_ts)),
                        "Agent_ECG": bed.agents["Agent_ECG"].trust_score,
                        "Agent_SpO2": bed.agents["Agent_SpO2"].trust_score,
                        "Agent_HR": bed.agents["Agent_HR"].trust_score,
                    }
                )

            offload_changed = bed.last_offload_published != bed.offload_active
            current_offload = bed.offload_active
            bed.last_offload_published = current_offload

        if offload_changed and self.client:
            try:
                self.client.publish(
                    f"hospital/{patient_id}/control/offload",
                    json.dumps({"patient_id": patient_id, "offload": current_offload}),
                    qos=0,
                )
            except Exception:
                pass

    def publish_patient_chaos(self, patient_id, bed_chaos_cfg, full_chaos_cfg):
        """Publishes updated chaos settings over MQTT to the specific patient bed's Jetson."""
        if self.client:
            try:
                self.client.publish(
                    f"hospital/{patient_id}/control/chaos",
                    json.dumps(bed_chaos_cfg),
                    qos=1,
                    retain=True,
                )
                self.client.publish(
                    "hospital/control/chaos",
                    json.dumps(full_chaos_cfg),
                    qos=1,
                    retain=True,
                )
            except Exception as e:
                print(f"[DASHBOARD] Error publishing chaos config: {e}")

    def get_all_beds_snapshot(self):
        """Returns a thread-safe snapshot of all dynamically discovered patient beds."""
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
def get_nurse_station_engine(version="v2_dynamic_beds"):
    broker = os.environ.get("MQTT_BROKER", "127.0.0.1")
    port = int(os.environ.get("MQTT_PORT", "1883"))
    return NurseStationEngine(broker=broker, port=port)


def render_status_badge(status_str):
    if status_str.startswith("TRUSTED"):
        bg, fg, border = "#0f5132", "#d1e7dd", "#198754"
    elif status_str.startswith("OFFLOADED"):
        bg, fg, border = "#664d03", "#fff3cd", "#ffc107"
    else:
        bg, fg, border = "#842029", "#f8d7da", "#dc3545"

    return (
        f'<div style="background-color:{bg};color:{fg};border:1px solid {border};'
        f'padding:6px 10px;border-radius:6px;font-weight:700;text-align:center;'
        f'font-size:0.86rem;margin-top:4px;margin-bottom:8px;">{status_str}</div>'
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


def main():
    st.set_page_config(
        page_title="Multi-Patient Trust-Aware Smart Healthcare ICU Monitor",
        page_icon="🏥",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        @keyframes pulse-red {
            0% { background-color: #842029; box-shadow: 0 0 8px #dc3545; }
            50% { background-color: #dc3545; box-shadow: 0 0 22px #ff4d4d; }
            100% { background-color: #842029; box-shadow: 0 0 8px #dc3545; }
        }
        .bedside-alarm-banner {
            animation: pulse-red 1.0s infinite;
            color: #ffffff;
            padding: 14px 18px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 1.15rem;
            text-align: center;
            margin-bottom: 12px;
            border: 2px solid #ffccd5;
        }
        .bedside-normal-banner {
            background-color: #0f5132;
            color: #d1e7dd;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.02rem;
            text-align: center;
            margin-bottom: 12px;
            border: 1px solid #198754;
        }
        .conflict-warning-banner {
            background-color: #664d03;
            color: #fff3cd;
            padding: 10px 14px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 12px;
            border: 1px solid #ffc107;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    st.divider()

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

        st.markdown(f"**Live 250 Hz Bedside ECG Waveform ({selected_bed})**")
        ecg_df = pd.DataFrame({"ECG (mV)": snap["ecg_wave"]})
        st.line_chart(ecg_df, height=240)

        v1, v2, v3 = st.columns(3)
        vitals = edge.get("latest_vitals", {})
        v1.metric("Bedside Heart Rate", f"{vitals.get('hr', 75)} bpm")
        v2.metric("Bedside SpO2", f"{vitals.get('spo2', 98)} %")
        v3.metric("ECG Wave Variance", f"{vitals.get('ecg_variance', 0.04):.4f}")

        st.markdown("**ICU Ward Emergency & Security Event Log**")
        if all_snap["global_alert_log"]:
            for entry in all_snap["global_alert_log"][:6]:
                st.code(entry, language="text")
        else:
            st.info("No emergency alerts or security quarantines triggered yet.")

    # =========================================================================
    # RIGHT COLUMN: Selected Patient's Multi-Agent Trust & Routing Engine
    # =========================================================================
    with right_col:
        st.subheader(f"🛡️ {selected_bed} — Multi-Agent Trust & Routing Engine")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            route_icon = "🔀" if snap["offload_active"] else "🟢"
            st.metric("Active Alert Routing Path", f"{route_icon} {snap['active_routing']}")
        with r_col2:
            if snap["offload_active"]:
                cpu_ms = snap["gateway_cpu_inf_ms"]
                offload_txt = (
                    f"OFFLOADED (Gateway CPU: {cpu_ms:.2f} ms)"
                    if cpu_ms > 0
                    else "OFFLOADED (Gateway CPU Fallback)"
                )
            else:
                offload_txt = "PRIMARY (Jetson Orin Edge GPU)"
            st.metric("Compute Offloading State", offload_txt)

        card_cols = st.columns(3)
        units = {
            "Agent_ECG": ("Var", ""),
            "Agent_SpO2": ("SpO2", "%"),
            "Agent_HR": ("HR", " bpm"),
        }

        for idx, agent_id in enumerate(AGENT_IDS):
            ag = agents[agent_id]
            label_prefix, unit_suffix = units[agent_id]
            with card_cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{agent_id}**")
                    st.markdown(render_status_badge(ag["status"]), unsafe_allow_html=True)
                    st.metric(
                        label="Composite Trust Score T_i(t)",
                        value=f"{ag['trust_score']:.2f}",
                        delta=f"{ag['trust_score'] * 100:.0f}% Trust",
                    )
                    if ag["status"].startswith("ISOLATED"):
                        st.markdown(f"**Reading:** ~~`{ag['latest_value']}{unit_suffix}`~~ *(Muted)*")
                    else:
                        st.markdown(f"**Reading ({label_prefix}):** `{ag['latest_value']}{unit_suffix}`")
                    st.caption(
                        f"Latency: **{ag['avg_latency_ms']:.1f} ms** | "
                        f"Jitter: **{ag['jitter_ms']:.1f} ms**\n\n"
                        f"Loss: **{ag['loss_rate_pct']:.1f}%** | "
                        f"S_L={ag['s_l']:.2f} S_J={ag['s_j']:.2f} S_P={ag['s_p']:.2f} S_D={ag['s_d']:.2f}"
                    )

        st.markdown(f"**Real-Time Multi-Agent Trust Trajectory ({selected_bed}) $T_i(t) \\in [0.0, 1.0]$**")
        if snap["trust_history"]:
            hist_df = pd.DataFrame(snap["trust_history"]).set_index("time")
            st.line_chart(hist_df[["Agent_ECG", "Agent_SpO2", "Agent_HR"]], height=230)
        else:
            empty_df = pd.DataFrame(
                {"Agent_ECG": [1.0], "Agent_SpO2": [1.0], "Agent_HR": [1.0]}
            )
            st.line_chart(empty_df, height=230)

        st.caption(
            "**Thresholds:** `TRUSTED ≥ 0.70` (Green / Orin GPU) • "
            "`OFFLOADED 0.45–0.70 or Latency > 180 ms` (Yellow / Gateway CPU & Relay B) • "
            "`ISOLATED < 0.45` (Red / Byzantine Sensor Quarantined)"
        )

    if auto_refresh:
        time.sleep(0.6)
        st.rerun()


if __name__ == "__main__":
    main()
