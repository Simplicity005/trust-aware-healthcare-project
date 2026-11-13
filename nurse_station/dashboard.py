#!/usr/bin/env python3
"""
Multi-Patient Central Nurse Station & Trust Engine Dashboard
(Runs on the Nurse's Table Gateway Laptop)

Features:
  - Dynamically discovers and admits patient beds as Bedside Jetson Orin nodes connect over MQTT.
  - Computes real-time composite Trust Scores T_i(t) in [0.0, 1.0] per sensor per bed.
  - Quarantines Byzantine/spoofed sensors with automated clinical alert suppression.
  - Manages dynamic compute offloading to Gateway CPU and rerouting via Relay Agent B.
  - Room-based Device Directory in the Left Sidebar.
  - High Availability (HA) dual nurse station leader-election and automatic failover.
  - Clean Bento Box dark UI with zero emojis.
"""

import os
import re
import sys
import time
import json
import random
import threading
from collections import deque
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
HTML_TEMPLATE_FILE = os.path.join(BASE_DIR, "dashboard.html")
AGENT_IDS = ["Agent_ECG", "Agent_SpO2", "Agent_HR"]
AGENT_LABELS = {
    "Agent_ECG": "ECG Sensor",
    "Agent_SpO2": "SpO2 Sensor",
    "Agent_HR": "Heart Rate Sensor",
}


class AgentTrustTracker:
    """
    Computes composite Trust Score T_i(t) in [0.0, 1.0] per sensor agent:
        T_i(t) = w1 * S_L(t) + w2 * S_J(t) + w3 * S_P(t) + w4 * S_D(t)
    with weights w1=0.25, w2=0.20, w3=0.25, w4=0.30 and EWMA smoothing.
    """

    def __init__(self, agent_id, window_size=25, w1=0.25, w2=0.20, w3=0.25, w4=0.30, alpha=0.35):
        self.agent_id = agent_id
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        self.alpha = alpha

        self.latencies = deque(maxlen=window_size)
        self.delivery_window = deque(maxlen=max(window_size, 30))
        self.validity_window = deque(maxlen=min(window_size, 15))

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
        self.packets_received = 0
        self.packets_missed = 0

    def check_data_plausibility(self, value=0, ecg_wave=None, cross_sensor_conflict=False):
        if cross_sensor_conflict:
            return 0.10

        if self.agent_id == "Agent_ECG":
            if not ecg_wave or len(ecg_wave) != 250:
                return 0.0
            arr = np.array(ecg_wave, dtype=np.float32)
            max_abs = float(np.max(np.abs(arr)))
            std_val = float(np.std(arr))
            diff_std = float(np.std(np.diff(arr)))
            if max_abs > 2.2 or std_val > 0.65 or std_val < 0.03 or diff_std > 0.35:
                return 0.0
            return 1.0

        if self.agent_id == "Agent_SpO2":
            val = int(value or 0)
            return 1.0 if (70 <= val <= 100) else 0.0

        if self.agent_id == "Agent_HR":
            val = int(value or 0)
            return 1.0 if (40 <= val <= 190) else 0.0

        return 1.0

    def update(
        self,
        seq=None,
        sent_ts=None,
        recv_ts=None,
        value=None,
        ecg_wave=None,
        cross_sensor_conflict=False,
        payload=None,
    ):
        rx_time = recv_ts if recv_ts is not None else time.time()

        if payload is not None and isinstance(payload, dict):
            seq = int(payload.get("seq", 0))
            sent_ts = float(payload.get("sent_ts", rx_time))
            value = payload.get("value", 0)
            ecg_wave = payload.get("ecg_wave", [])
        else:
            seq = int(seq or 0)
            sent_ts = float(sent_ts if sent_ts is not None else rx_time)
            value = value if value is not None else 0
            ecg_wave = ecg_wave or []

        self.packets_received += 1
        latency_ms = max(0.5, (rx_time - sent_ts) * 1000.0)
        self.latencies.append(latency_ms)

        if self.last_seq is not None and seq > self.last_seq:
            missed = seq - self.last_seq - 1
            if missed > 0:
                self.packets_missed += missed
                for _ in range(min(missed, 30)):
                    self.delivery_window.append(0)
        self.delivery_window.append(1)
        self.last_seq = seq
        self.last_rx_time = rx_time

        if self.agent_id == "Agent_ECG" and ecg_wave:
            self.ecg_wave = ecg_wave
            self.last_value = round(float(np.var(ecg_wave)), 4)
        else:
            self.last_value = value

        d_valid = self.check_data_plausibility(value, ecg_wave, cross_sensor_conflict)
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
        self.evaluate_status()
        return self.trust_score

    def evaluate_status(self, orin_link_congested=False):
        now = time.time()
        silence_sec = now - self.last_rx_time
        if silence_sec > 2.0:
            penalty = np.exp(-(silence_sec - 2.0) / 3.0)
            self.s_p = float(np.clip(self.s_p * penalty, 0.0, 1.0))
            raw_trust = (
                self.w1 * self.s_l
                + self.w2 * self.s_j
                + self.w3 * self.s_p
                + self.w4 * self.s_d
            )
            self.trust_score = float(np.clip(raw_trust, 0.0, 1.0))

        if self.trust_score < 0.45:
            self.status = "ISOLATED"
        elif orin_link_congested:
            self.status = "OFFLOADED"
        elif self.trust_score < 0.70:
            self.status = "SUSPICIOUS"
        else:
            self.status = "TRUSTED"

        return self.status


class PatientBedMonitor:
    """Maintains sensor trust trackers, bedside AI summaries, and telemetry history for one bed."""

    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.last_seen_ts = time.time()
        self.agents = {aid: AgentTrustTracker(aid) for aid in AGENT_IDS}
        self.edge_summary = {
            "patient_id": patient_id,
            "inf_ms": 0.28,
            "exec_provider": "CUDA / Orin GPU",
            "arrhythmia_prob": 0.02,
            "local_bed_alarm": False,
            "sensor_conflict": False,
            "latest_vitals": {"hr": 75, "spo2": 98, "ecg_variance": 0.04},
            "routing_path": "primary",
            "timestamp": time.time(),
        }
        self.latest_ecg_wave = self._default_ecg_wave()
        self.gateway_cpu_inf_ms = 0.0
        self.active_routing = "Primary Direct Link"
        self.offload_active = False
        self.trust_history = deque(maxlen=60)
        self.alert_log = deque(maxlen=15)

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


class MultiPatientTrustEngine:
    def __init__(self, broker_host=None, broker_port=1883):
        self.lock = threading.Lock()
        self.broker_host = broker_host or os.environ.get("MQTT_BROKER", "127.0.0.1")
        self.broker_port = int(os.environ.get("MQTT_PORT", broker_port))

        self.beds = {}
        self.added_bed_ids = {"Bed_1"}
        self.discovered_jetsons = {}
        self.global_alert_log = deque(maxlen=30)
        self.stop_event = threading.Event()

        # High Availability configuration
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

        # CPU fallback ONNX inference
        self.cpu_session = None
        self.cpu_input_name = "ecg_input"
        if os.path.exists(MODEL_FILE):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                self.cpu_session = ort.InferenceSession(
                    MODEL_FILE, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.cpu_input_name = self.cpu_session.get_inputs()[0].name
            except Exception as e:
                print(f"[DASHBOARD] CPU fallback load warning: {e}")

        # Initialize Bed_1 so initial display is active
        self._get_or_create_bed("Bed_1")

        # MQTT Connection
        self.client = None
        self._init_mqtt()

        # Background simulator for admitted beds without physical hardware
        self.sim_thread = threading.Thread(target=self._run_background_synthesizer, daemon=True)
        self.sim_thread.start()

        # HA Heartbeat thread
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

    def _init_mqtt(self):
        cid = f"NurseStation_{self.station_id}_{os.getpid()}_{random.randint(100, 999)}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        else:
            self.client = mqtt.Client(client_id=cid)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            print(f"[DASHBOARD] Connected to MQTT broker {self.broker_host}:{self.broker_port}")
        except Exception as e:
            print(f"[DASHBOARD] MQTT connect notice: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        client.subscribe("hospital/+/sensors/#")
        client.subscribe("hospital/+/edge/summary")
        client.subscribe("hospital/+/alerts/#")
        client.subscribe("hospital/discovery/announce")
        client.subscribe("hospital/nurse_stations/heartbeat")
        client.subscribe("hospital/nurse_stations/failover")
        self.probe_network_for_jetsons()

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode("utf-8"))
            recv_ts = time.time()
            parts = topic.split("/")

            # 1. High Availability Heartbeats
            if topic == "hospital/nurse_stations/heartbeat":
                sid = payload.get("station_id")
                if sid and sid != self.station_id:
                    with self.lock:
                        self.peer_stations[sid] = {
                            "station_id": sid,
                            "role": payload.get("role", "secondary"),
                            "port": payload.get("port", 8502),
                            "timestamp": recv_ts,
                        }
                        if payload.get("role") == "primary":
                            self.last_primary_heartbeat_ts = recv_ts
                return

            if topic == "hospital/nurse_stations/failover":
                failed_station = payload.get("failed_station")
                if not self.is_primary and failed_station != self.station_id:
                    with self.lock:
                        self.is_primary = True
                        self.failover_triggered = True
                        entry = (
                            f"[{time.strftime('%H:%M:%S')}] [HIGH AVAILABILITY] Failover received: "
                            f"{self.station_id} is now ACTIVE PRIMARY LEADER."
                        )
                        self.global_alert_log.appendleft(entry)
                return

            # 2. Jetson Edge Discovery announcements
            if topic == "hospital/discovery/announce":
                patient_id = payload.get("patient_id", "Unknown_Jetson")
                with self.lock:
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
                return

            # 3. Patient Bed Telemetry
            if len(parts) >= 3 and parts[0] == "hospital":
                patient_id = parts[1]

                # Update discovery registry
                with self.lock:
                    if patient_id not in self.discovered_jetsons:
                        self.discovered_jetsons[patient_id] = {
                            "patient_id": patient_id,
                            "device_model": "NVIDIA Jetson Orin",
                            "exec_provider": "CUDA / Orin GPU",
                            "first_seen_ts": recv_ts,
                            "last_seen_ts": recv_ts,
                            "packet_count": 1,
                            "latest_vitals": {},
                            "ip": "127.0.0.1",
                        }
                    else:
                        dev = self.discovered_jetsons[patient_id]
                        dev["last_seen_ts"] = recv_ts
                        dev["packet_count"] += 1

                # If nurse has not admitted this bed yet, do not mix into active ICU ward
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
                        if payload.get("routing_path") == "relay_b":
                            bed.active_routing = "Relay Path B (Rerouted)"
                            bed.offload_active = True
                        else:
                            bed.active_routing = "Primary Direct Link"
                            bed.offload_active = False
                elif "alerts" in parts:
                    with self.lock:
                        bed = self._get_or_create_bed(patient_id)
                        route_label = "Relay Path B" if "relay_b" in topic else "Primary Link"
                        prob = payload.get("arrhythmia_prob", 0.0) * 100.0
                        inf_ms = payload.get("inf_ms", 0.0)
                        ts_str = time.strftime("%H:%M:%S", time.localtime(recv_ts))
                        entry = (
                            f"[{ts_str}] [{patient_id}] CRITICAL ARRHYTHMIA DETECTED "
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

        with self.lock:
            bed = self._get_or_create_bed(patient_id)
            conflict = bool(bed.edge_summary.get("sensor_conflict", False))
            tracker = bed.agents[agent_id]
            tracker.update(payload=payload, recv_ts=recv_ts, cross_sensor_conflict=conflict)

            if agent_id == "Agent_ECG":
                ecg_wave = payload.get("ecg_wave", [])
                if ecg_wave and len(ecg_wave) == 250:
                    bed.latest_ecg_wave = ecg_wave
                    if bed.offload_active and self.cpu_session is not None:
                        t0 = time.perf_counter()
                        wave_np = np.array(ecg_wave, dtype=np.float32).reshape(1, 1, 250)
                        _ = self.cpu_session.run(None, {self.cpu_input_name: wave_np})
                        bed.gateway_cpu_inf_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                bed.trust_history.append(
                    {
                        "time": time.strftime("%H:%M:%S", time.localtime(recv_ts)),
                        "Agent_ECG": bed.agents["Agent_ECG"].trust_score,
                        "Agent_SpO2": bed.agents["Agent_SpO2"].trust_score,
                        "Agent_HR": bed.agents["Agent_HR"].trust_score,
                    }
                )

            bed.last_seen_ts = recv_ts

    def _get_or_create_bed(self, patient_id):
        if not patient_id:
            patient_id = "Bed_1"
        if patient_id not in self.beds:
            self.beds[patient_id] = PatientBedMonitor(patient_id)
        self.beds[patient_id].last_seen_ts = time.time()
        return self.beds[patient_id]

    def _run_heartbeat_broadcaster(self):
        while not self.stop_event.is_set():
            time.sleep(1.0)
            now = time.time()

            # 1. Publish own heartbeat over MQTT
            if self.client and not self.failover_simulated:
                try:
                    payload = {
                        "station_id": self.station_id,
                        "role": "primary" if self.is_primary else "secondary",
                        "port": self.station_port,
                        "timestamp": now,
                        "active_beds": list(self.added_bed_ids),
                    }
                    self.client.publish("hospital/nurse_stations/heartbeat", json.dumps(payload), qos=0)
                except Exception:
                    pass

            # 2. Check for Primary Station timeout if this is a standby station
            if not self.is_primary and not self.failover_simulated:
                if self.last_primary_heartbeat_ts > 0 and (now - self.last_primary_heartbeat_ts) > 4.0:
                    with self.lock:
                        self.is_primary = True
                        self.failover_triggered = True
                        entry = (
                            f"[{time.strftime('%H:%M:%S')}] [HIGH AVAILABILITY] PRIMARY STATION TIMEOUT (>4.0s). "
                            f"{self.station_id} AUTOMATICALLY PROMOTED TO ACTIVE PRIMARY LEADER!"
                        )
                        self.global_alert_log.appendleft(entry)

    def trigger_failover_test(self):
        now = time.time()
        with self.lock:
            if self.is_primary:
                self.is_primary = False
                self.failover_simulated = True
                entry = f"[{time.strftime('%H:%M:%S')}] [HA TEST] Simulated Primary Station failure on {self.station_id}."
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
                entry = f"[{time.strftime('%H:%M:%S')}] [HA TEST] Manual override: {self.station_id} promoted to PRIMARY LEADER."
                self.global_alert_log.appendleft(entry)

    def _run_background_synthesizer(self):
        step = 0
        while not self.stop_event.is_set():
            time.sleep(0.40)
            step += 1
            now = time.time()
            with self.lock:
                for pid in list(self.added_bed_ids):
                    bed = self._get_or_create_bed(pid)
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
                        bed.edge_summary["inf_ms"] = round(0.22 + float(np.random.uniform(0.01, 0.06)), 2)
                        bed.edge_summary["arrhythmia_prob"] = 0.02
                        bed.edge_summary["local_bed_alarm"] = False
                        bed.agents["Agent_ECG"].update(
                            seq=step, sent_ts=now - 0.01, recv_ts=now, value=round(var_val, 4), ecg_wave=wave_list
                        )
                        bed.agents["Agent_SpO2"].update(seq=step, sent_ts=now - 0.01, recv_ts=now, value=spo2_val)
                        bed.agents["Agent_HR"].update(seq=step, sent_ts=now - 0.01, recv_ts=now, value=hr_val)
                        if step % 3 == 0:
                            bed.trust_history.append(
                                {
                                    "time": time.strftime("%H:%M:%S", time.localtime(now)),
                                    "Agent_ECG": bed.agents["Agent_ECG"].trust_score,
                                    "Agent_SpO2": bed.agents["Agent_SpO2"].trust_score,
                                    "Agent_HR": bed.agents["Agent_HR"].trust_score,
                                }
                            )

    def probe_network_for_jetsons(self):
        if self.client:
            try:
                self.client.publish(
                    "hospital/discovery/probe",
                    json.dumps({"action": "probe", "timestamp": time.time()}),
                    qos=1,
                )
            except Exception:
                pass

    def admit_jetson(self, patient_id, room_name=None):
        with self.lock:
            self.added_bed_ids.add(patient_id)
            if room_name:
                self.bed_rooms[patient_id] = room_name
            self._get_or_create_bed(patient_id)
            if self.client:
                try:
                    self.client.publish(
                        f"hospital/{patient_id}/control/admit",
                        json.dumps({"admitted": True, "room": room_name or self.get_bed_room(patient_id)}),
                        qos=1,
                    )
                except Exception:
                    pass
            entry = f"[{time.strftime('%H:%M:%S')}] [WARD ADMISSION] Jetson node {patient_id} admitted to {self.get_bed_room(patient_id)}."
            self.global_alert_log.appendleft(entry)

    def release_jetson(self, patient_id):
        with self.lock:
            if patient_id in self.added_bed_ids:
                self.added_bed_ids.remove(patient_id)
            if patient_id in self.beds:
                del self.beds[patient_id]
            entry = f"[{time.strftime('%H:%M:%S')}] [WARD RELEASE] Bed {patient_id} released from active monitoring."
            self.global_alert_log.appendleft(entry)

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
                        "trust": ag.trust_score,
                        "s_l": round(ag.s_l, 2),
                        "s_j": round(ag.s_j, 2),
                        "s_p": round(ag.s_p, 2),
                        "s_d": round(ag.s_d, 2),
                        "value": ag.last_value,
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
                    "edge": dict(bed.edge_summary),
                    "ecg_wave": list(bed.latest_ecg_wave),
                    "gateway_cpu_inf_ms": bed.gateway_cpu_inf_ms,
                    "compute_mode": "Gateway CPU (Offloaded)" if bed.offload_active else "Jetson Local",
                    "alert_route": bed.active_routing,
                    "history": list(bed.trust_history),
                    "alert_log": list(bed.alert_log),
                }

            unadded_jetsons = []
            for pid, dev in self.discovered_jetsons.items():
                if pid not in self.added_bed_ids:
                    is_active = (now - dev["last_seen_ts"]) <= 15.0
                    unadded_jetsons.append(
                        {
                            **dev,
                            "room": self.get_bed_room(pid),
                            "online": is_active,
                            "seconds_since_seen": round(now - dev["last_seen_ts"], 1),
                        }
                    )
            unadded_jetsons.sort(key=lambda x: x["patient_id"])

            active_peers = {
                sid: p for sid, p in self.peer_stations.items() if (now - p["timestamp"]) <= 5.0
            }

            return {
                "beds": beds_snap,
                "global_alert_log": list(self.global_alert_log),
                "unadded_jetsons": unadded_jetsons,
                "added_bed_ids": list(self.added_bed_ids),
                "ha": {
                    "station_id": self.station_id,
                    "station_port": self.station_port,
                    "is_primary": self.is_primary,
                    "failover_triggered": self.failover_triggered,
                    "failover_simulated": self.failover_simulated,
                    "peers": active_peers,
                    "total_stations_online": len(active_peers) + (0 if self.failover_simulated else 1),
                },
            }


@st.cache_resource
def get_engine():
    return MultiPatientTrustEngine()


def load_html_templates():
    """Reads dashboard.html from disk on every render so HTML/CSS edits apply immediately."""
    if not os.path.exists(HTML_TEMPLATE_FILE):
        return "", {}
    with open(HTML_TEMPLATE_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    style_match = re.search(r"(<style[^>]*>.*?</style>)", raw, flags=re.DOTALL | re.IGNORECASE)
    css_block = style_match.group(1) if style_match else ""

    templates = {}
    for m in re.finditer(
        r'<template\s+id=["\']([^"\']+)["\']\s*>(.*?)</template>',
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        tid = m.group(1)
        body = re.sub(r"<!--.*?-->", "", m.group(2), flags=re.DOTALL)
        compact = "\n".join(line.strip() for line in body.splitlines() if line.strip())
        templates[tid] = compact

    return css_block, templates


def render_template(templates, name, **kwargs):
    html = templates.get(name, "")
    for k, v in kwargs.items():
        html = html.replace(f"{{{{{k}}}}}", str(v))
    return html


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
        f'style="display:block;background:#08090b;border-radius:6px;border:1px solid #1a1d24;">'
        f'<line x1="0" y1="{mid:.1f}" x2="{width}" y2="{mid:.1f}" stroke="#161820" stroke-width="1"/>'
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
            val = float(h.get(key, 1.0))
            y = height - val * (height - 18) - 9
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    y70 = height - 0.70 * (height - 18) - 9
    y45 = height - 0.45 * (height - 18) - 9

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="display:block;background:#08090b;border-radius:6px;border:1px solid #1a1d24;">'
        f'<line x1="0" y1="{y70:.1f}" x2="{width}" y2="{y70:.1f}" stroke="#22c55e" stroke-dasharray="4 4" stroke-width="1"/>'
        f'<line x1="0" y1="{y45:.1f}" x2="{width}" y2="{y45:.1f}" stroke="#ef4444" stroke-dasharray="4 4" stroke-width="1"/>'
        f'<polyline fill="none" stroke="#10b981" stroke-width="2" points="{line_pts("Agent_ECG")}"/>'
        f'<polyline fill="none" stroke="#3b82f6" stroke-width="2" points="{line_pts("Agent_SpO2")}"/>'
        f'<polyline fill="none" stroke="#f59e0b" stroke-width="2" points="{line_pts("Agent_HR")}"/>'
        f"</svg>"
    )


def status_pill_info(bed):
    if not bed["online"]:
        return "OFFLINE", "pill-muted"
    agents = bed["agents"]
    isolated = [a for a, d in agents.items() if d["status"] == "ISOLATED"]
    suspicious = [a for a, d in agents.items() if d["status"] == "SUSPICIOUS"]
    if bed["edge"].get("local_bed_alarm") and agents["Agent_ECG"]["status"] != "ISOLATED":
        return "ALARM", "pill-red"
    if isolated:
        return "ISOLATED", "pill-red"
    if suspicious or bed["compute_mode"] != "Jetson Local":
        return "DEGRADED" if suspicious else "OFFLOADED", "pill-amber"
    return "STABLE", "pill-green"


def dot_color(status):
    if status == "TRUSTED":
        return "#10b981"
    if status == "SUSPICIOUS":
        return "#f59e0b"
    return "#ef4444"


def ward_tile(pid, bed, is_selected, templates=None):
    if templates is None:
        _, templates = load_html_templates()
    label, pill_cls = status_pill_info(bed)
    hr_ag = bed["agents"]["Agent_HR"]
    sp_ag = bed["agents"]["Agent_SpO2"]
    ecg_ag = bed["agents"]["Agent_ECG"]

    hr_txt = "ISO" if hr_ag["status"] == "ISOLATED" else (str(hr_ag["value"]) if hr_ag["value"] is not None else "--")
    sp_txt = "ISO" if sp_ag["status"] == "ISOLATED" else (f"{sp_ag['value']}%" if sp_ag["value"] is not None else "--")
    ecg_col = "#ef4444" if (ecg_ag["status"] == "ISOLATED" or bed["edge"].get("local_bed_alarm")) else "#10b981"
    border_col = "#3b82f6" if is_selected else "#1f2228"

    return render_template(
        templates,
        "ward_tile",
        border_col=border_col,
        pid=pid,
        room_short=bed["room"].split()[0],
        pill_cls=pill_cls,
        label=label,
        hr_txt=hr_txt,
        sp_txt=sp_txt,
        compute_mode=bed["compute_mode"],
        ecg_svg=svg_ecg(bed["ecg_wave"], color=ecg_col, width=360, height=54),
        ecg_dot=dot_color(ecg_ag["status"]),
        ecg_trust=f"{ecg_ag['trust']:.2f}",
        spo2_dot=dot_color(sp_ag["status"]),
        spo2_trust=f"{sp_ag['trust']:.2f}",
        hr_dot=dot_color(hr_ag["status"]),
        hr_trust=f"{hr_ag['trust']:.2f}",
    )


def detail_view(pid, bed, logs, templates=None):
    if templates is None:
        _, templates = load_html_templates()
    ecg_ag = bed["agents"]["Agent_ECG"]
    sp_ag = bed["agents"]["Agent_SpO2"]
    hr_ag = bed["agents"]["Agent_HR"]
    edge = bed["edge"]

    ecg_quarantined = ecg_ag["status"] == "ISOLATED"
    is_arr = bool(edge.get("local_bed_alarm")) and not ecg_quarantined
    ecg_col = "#ef4444" if (is_arr or ecg_quarantined) else "#10b981"
    rhythm_txt = (
        "Quarantined (Byzantine Signal)"
        if ecg_quarantined
        else ("Ventricular Arrhythmia" if is_arr else "Normal Sinus Rhythm")
    )
    rhythm_col = "#ef4444" if (is_arr or ecg_quarantined) else "#ededef"

    def build_sensor_tile(title, ag, unit):
        st_name = ag["status"]
        p_cls = "pill-green" if st_name == "TRUSTED" else ("pill-amber" if st_name == "SUSPICIOUS" else "pill-red")
        bar_col = dot_color(st_name)
        pct = int(round(ag["trust"] * 100))
        if "ECG" in title:
            reading = "250 Hz Stream Nominal" if st_name != "ISOLATED" else "Stream Quarantined"
        else:
            v = ag["value"]
            reading = (
                f"{v} {unit}".strip()
                if (v is not None and st_name != "ISOLATED")
                else ("Quarantined" if st_name == "ISOLATED" else "--")
            )

        return render_template(
            templates,
            "sensor_tile",
            title=title,
            p_cls=p_cls,
            st_name=st_name,
            trust_val=f"{ag['trust']:.2f}",
            reading=reading,
            pct=pct,
            bar_col=bar_col,
            s_l=f"{ag['s_l']:.2f}",
            s_j=f"{ag['s_j']:.2f}",
            s_p=f"{ag['s_p']:.2f}",
            s_d=f"{ag['s_d']:.2f}",
        )

    sensor_tiles_html = (
        build_sensor_tile("ECG Sensor (Agent_ECG)", ecg_ag, "")
        + build_sensor_tile("SpO2 Sensor (Agent_SpO2)", sp_ag, "%")
        + build_sensor_tile("Heart Rate Sensor (Agent_HR)", hr_ag, "bpm")
    )

    bed_logs = [e for e in logs if pid in e][:6]
    if bed_logs:
        log_rows_html = "".join(f'<div class="log-row"><span class="log-msg">{entry}</span></div>' for entry in bed_logs)
    else:
        log_rows_html = '<div class="sub" style="padding:6px 0;">No anomaly alerts or quarantine events for this patient bed.</div>'

    return render_template(
        templates,
        "detail_view",
        pid=pid,
        room=bed["room"],
        ecg_svg=svg_ecg(bed["ecg_wave"], color=ecg_col, width=720, height=150),
        rhythm_col=rhythm_col,
        rhythm_txt=rhythm_txt,
        arrhythmia_prob=f"{edge.get('arrhythmia_prob', 0.0) * 100:.1f}",
        inf_ms=f"{edge.get('inf_ms', 0.0):.2f}",
        compute_mode=bed["compute_mode"],
        alert_route=bed["alert_route"],
        sensor_tiles_html=sensor_tiles_html,
        trust_chart_svg=svg_trust_chart(bed["history"], width=720, height=130),
        log_rows_html=log_rows_html,
    )


def render_sidebar_room_directory(engine, all_snap, selected_bed, templates):
    ha = all_snap["ha"]
    role_cls = "pill-green" if ha["is_primary"] else "pill-amber"
    role_name = "PRIMARY LEADER" if ha["is_primary"] else "HOT STANDBY"

    st.sidebar.markdown(
        render_template(
            templates,
            "sidebar_header",
            station_id=ha["station_id"],
            role_cls=role_cls,
            role_name=role_name,
            station_port=ha["station_port"],
            total_stations_online=ha["total_stations_online"],
        ),
        unsafe_allow_html=True,
    )

    if ha["failover_triggered"]:
        st.sidebar.markdown(render_template(templates, "sidebar_failover_banner"), unsafe_allow_html=True)

    with st.sidebar.expander("High Availability (HA) Failover Controls", expanded=False):
        st.markdown(
            f"**Station:** `{ha['station_id']}`  \n"
            f"**Role:** `{role_name}`  \n"
            f"**Port:** `{ha['station_port']}`"
        )
        if ha["peers"]:
            st.markdown("**Discovered Peer Stations:**")
            for sid, p in ha["peers"].items():
                st.markdown(f"- `{sid}` ({p.get('role', 'peer').upper()} on port `{p.get('port')}`)")
        else:
            st.caption("No secondary peer stations detected on LAN.")

        if st.button("Test Failover / Switch Role", key="btn_ha_toggle", use_container_width=True):
            engine.trigger_failover_test()
            st.rerun()

    st.sidebar.markdown('<div class="sec-label">ICU Room Directory</div>', unsafe_allow_html=True)

    rooms = {}
    for pid, b in all_snap["beds"].items():
        rm = b["room"]
        if rm not in rooms:
            rooms[rm] = []
        rooms[rm].append((pid, b))

    all_rooms_list = ["All Rooms"] + sorted(rooms.keys())
    selected_room_filter = st.sidebar.selectbox("Filter Rooms", options=all_rooms_list, index=0)

    for rm, bed_list in sorted(rooms.items()):
        if selected_room_filter != "All Rooms" and rm != selected_room_filter:
            continue

        st.sidebar.markdown(f'<div class="room-header">{rm} ({len(bed_list)} Bed)</div>', unsafe_allow_html=True)
        for pid, b in bed_list:
            st_lbl, pill_cls = status_pill_info(b)
            v = b["edge"].get("latest_vitals", {})
            hr_v = v.get("hr", 75)
            sp_v = v.get("spo2", 98)

            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.markdown(
                    render_template(
                        templates,
                        "sidebar_bed_row",
                        pid=pid,
                        pill_cls=pill_cls,
                        st_lbl=st_lbl,
                        hr_v=hr_v,
                        sp_v=sp_v,
                    ),
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("View", key=f"sel_btn_{pid}", use_container_width=True):
                    st.session_state["selected_bed"] = pid
                    st.rerun()

    unadded = all_snap["unadded_jetsons"]
    if unadded:
        st.sidebar.markdown(
            f'<div class="sec-label" style="color:#38bdf8;">Unassigned Jetson Nodes ({len(unadded)})</div>',
            unsafe_allow_html=True,
        )
        for dev in unadded:
            pid = dev["patient_id"]
            st.sidebar.markdown(
                render_template(
                    templates,
                    "sidebar_unassigned_row",
                    pid=pid,
                    exec_provider=dev.get("exec_provider", "CUDA"),
                ),
                unsafe_allow_html=True,
            )
            target_rm = dev.get("room", "Room 101 (Cardiac ICU)")
            if st.sidebar.button(f"+ Admit {pid}", key=f"side_admit_{pid}", use_container_width=True):
                engine.admit_jetson(pid, target_rm)
                st.session_state["selected_bed"] = pid
                st.rerun()

    st.sidebar.markdown(render_template(templates, "sidebar_bedside_link"), unsafe_allow_html=True)


def render_admission_modal(engine, unadded_jetsons, added_bed_ids, templates):
    st.markdown(render_template(templates, "admission_banner"), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"**Discovered Nodes Ready to Pair:** `{len(unadded_jetsons)}` unassigned device(s) online.")
    with c2:
        if st.button("Rescan Network / Probe", key="rescan_modal_btn", use_container_width=True):
            engine.probe_network_for_jetsons()
            st.rerun()
    with c3:
        if st.button("Close Panel", key="close_admission_btn", use_container_width=True):
            st.session_state["show_add_bed"] = False
            st.rerun()

    if unadded_jetsons:
        st.markdown("---")
        for dev in unadded_jetsons:
            pid = dev["patient_id"]
            model = dev.get("device_model", "NVIDIA Jetson Orin")
            provider = dev.get("exec_provider", "CUDA / Orin GPU")
            vitals = dev.get("latest_vitals", {})
            hr = vitals.get("hr", 75)
            spo2 = vitals.get("spo2", 98)

            col_dev, col_room, col_act = st.columns([4, 4, 3], gap="medium")
            with col_dev:
                st.markdown(
                    f"**Hardware Node:** `{pid}`  \n"
                    f"Model: `{model}` | AI Engine: `{provider}`  \n"
                    f"Vitals Snapshot: **HR {hr} bpm** | **SpO2 {spo2}%**"
                )
            with col_room:
                room_opts = [
                    "Room 101 (Cardiac ICU)",
                    "Room 102 (Cardiac ICU)",
                    "Room 103 (Neuro ICU)",
                    "Room 104 (Surgical ICU)",
                    "Room 105 (Trauma ICU)",
                ]
                assigned_room = st.selectbox(
                    "Assign Room",
                    options=room_opts,
                    key=f"rm_select_{pid}",
                    label_visibility="collapsed",
                )
            with col_act:
                if st.button(f"+ Admit {pid}", key=f"admit_btn_{pid}", type="primary", use_container_width=True):
                    engine.admit_jetson(pid, assigned_room)
                    st.session_state["selected_bed"] = pid
                    st.session_state["show_add_bed"] = False
                    st.rerun()
    else:
        st.markdown(render_template(templates, "admission_empty"), unsafe_allow_html=True)
        sim_id = f"Bed_{len(added_bed_ids) + 1}"
        if st.button(f"+ Simulate & Detect New Jetson Node ({sim_id})", key="btn_sim_jetson"):
            engine.admit_jetson(sim_id, "Room 102 (Cardiac ICU)")
            st.session_state["selected_bed"] = sim_id
            st.session_state["show_add_bed"] = False
            st.rerun()


def main():
    st.set_page_config(
        page_title="Trust-Aware ICU Nurse Station",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    css_block, templates = load_html_templates()
    st.markdown(css_block, unsafe_allow_html=True)

    engine = get_engine()
    all_snap = engine.snapshot()
    beds_snap = all_snap["beds"]
    logs = all_snap["global_alert_log"]
    unadded_jetsons = all_snap["unadded_jetsons"]
    added_bed_ids = all_snap["added_bed_ids"]
    ha = all_snap["ha"]

    if "selected_bed" not in st.session_state or st.session_state["selected_bed"] not in beds_snap:
        st.session_state["selected_bed"] = next(iter(beds_snap.keys()), "Bed_1")

    selected_bed = st.session_state["selected_bed"]
    render_sidebar_room_directory(engine, all_snap, selected_bed, templates)

    add_bed_badge = f" ({len(unadded_jetsons)} Detected)" if unadded_jetsons else ""
    role_label = "[PRIMARY LEADER]" if ha["is_primary"] else "[HOT STANDBY]"

    top_c1, top_c2 = st.columns([8, 4])
    with top_c1:
        st.markdown(
            render_template(
                templates,
                "topbar",
                station_id=ha["station_id"],
                role_label=role_label,
                station_port=ha["station_port"],
                bed_count=len(beds_snap),
            ),
            unsafe_allow_html=True,
        )
    with top_c2:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if "show_add_bed" not in st.session_state:
                st.session_state["show_add_bed"] = False
            add_lbl = "- Close Panel" if st.session_state["show_add_bed"] else f"+ Add New Bed{add_bed_badge}"
            if st.button(add_lbl, key="toggle_add_bed", use_container_width=True):
                st.session_state["show_add_bed"] = not st.session_state["show_add_bed"]
                st.rerun()
        with btn_c2:
            if st.button("Rescan LAN", key="top_rescan_btn", use_container_width=True):
                engine.probe_network_for_jetsons()
                st.rerun()

    if st.session_state.get("show_add_bed", False):
        render_admission_modal(engine, unadded_jetsons, added_bed_ids, templates)

    st.markdown('<div class="sec-label">ICU Multi-Patient Ward Overview</div>', unsafe_allow_html=True)
    if not beds_snap:
        st.info("No active patient beds admitted. Click '+ Add New Bed' above to detect and admit Jetson bedside nodes.")
    else:
        ward_html = "".join(ward_tile(pid, b, pid == selected_bed, templates=templates) for pid, b in beds_snap.items())
        st.markdown(f'<div class="ward-grid">{ward_html}</div>', unsafe_allow_html=True)

    bed_keys = list(beds_snap.keys())
    if len(bed_keys) > 1:
        sel_c1, sel_c2 = st.columns([3, 9])
        with sel_c1:
            cur_idx = bed_keys.index(selected_bed) if selected_bed in bed_keys else 0
            new_sel = st.selectbox(
                "Inspect Bed",
                options=bed_keys,
                index=cur_idx,
                label_visibility="collapsed",
                key="bed_select_dropdown",
            )
            if new_sel != selected_bed:
                st.session_state["selected_bed"] = new_sel
                st.rerun()

    st.markdown(
        f'<div class="sec-label">Detailed Telemetry &amp; Trust Inspection &mdash; {selected_bed} ({engine.get_bed_room(selected_bed)})</div>',
        unsafe_allow_html=True,
    )
    if selected_bed in beds_snap:
        st.markdown(detail_view(selected_bed, beds_snap[selected_bed], logs, templates=templates), unsafe_allow_html=True)

    time.sleep(0.6)
    st.rerun()


if __name__ == "__main__":
    main()
