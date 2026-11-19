#!/usr/bin/env python3
"""
Multi-patient Nurse Station trust engine
(Runs on the Nurse's Table Gateway Laptop)

Features:
  - Dynamically discovers and admits patient beds as Bedside Jetson Orin nodes connect over MQTT.
  - Computes real-time composite Trust Scores T_i(t) in [0.0, 1.0] per sensor per bed.
  - Quarantines Byzantine/spoofed sensors with automated clinical alert suppression.
  - Manages dynamic compute offloading to Gateway CPU and rerouting via Relay Agent B.
  - High Availability (HA) dual nurse station leader-election and automatic failover.
"""

import os
import sys
import time
import json
import random
import threading
from collections import deque
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None
import paho.mqtt.client as mqtt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEDSIDE_DIR = os.path.join(os.path.dirname(BASE_DIR), "jetson_bedside")
if BEDSIDE_DIR not in sys.path:
    sys.path.insert(0, BEDSIDE_DIR)
from patient_simulation import PatientSimulation, SAMPLE_RATE, WINDOW_SAMPLES

MODEL_FILE = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
AGENT_IDS = ["Agent_ECG", "Agent_SpO2", "Agent_HR", "Agent_CO2", "Agent_IBP", "Agent_NIBP"]
AGENT_LABELS = {
    "Agent_ECG": "ECG Sensor",
    "Agent_SpO2": "SpO2 Sensor",
    "Agent_HR": "Heart Rate Sensor",
    "Agent_CO2": "EtCO2 Capnography",
    "Agent_IBP": "Invasive Arterial BP",
    "Agent_NIBP": "Non-Invasive Cuff BP",
}
HOSPITAL_DEPARTMENTS = [
    ("ICU", "Intensive Care Unit"),
    ("CCU", "Coronary Care Unit"),
    ("NCU", "Neurological Care Unit"),
    ("NICU", "Neonatal Intensive Care Unit"),
    ("PICU", "Pediatric Intensive Care Unit"),
    ("SICU", "Surgical Intensive Care Unit"),
    ("Trauma / ER", "Emergency & Trauma"),
    ("General Ward", "General & Step-Down Ward"),
]
BEDSIDE_CHAOS_FILE = os.path.join(os.path.dirname(BASE_DIR), "jetson_bedside", "chaos.json")
CHAOS_FILE = os.environ.get("NURSE_CHAOS_FILE") or (
    BEDSIDE_CHAOS_FILE if os.path.isfile(BEDSIDE_CHAOS_FILE) else os.path.join(BASE_DIR, "chaos.json")
)

DEFAULT_PATIENTS = {
    "Bed_1": {
        "name": "Eleanor Vance (F, 64)",
        "diagnosis": "Acute Myocardial Infarction / Post-CABG",
        "admitted": "Nov 17, 08:30",
    },
    "Bed_2": {
        "name": "Arthur Pendelton (M, 71)",
        "diagnosis": "Severe ARDS & Sepsis Protocol",
        "admitted": "Nov 17, 11:15",
    },
    "Bed_3": {
        "name": "Sarah Jenkins (F, 42)",
        "diagnosis": "Traumatic Brain Injury / ICP Monitor",
        "admitted": "Nov 17, 14:00",
    },
    "Bed_4": {
        "name": "Michael Chang (M, 55)",
        "diagnosis": "Post-Cardiac Arrest / Hypothermia Protocol",
        "admitted": "Nov 18, 02:45",
    },
    "Bed_5": {
        "name": "Elena Rostova (F, 29)",
        "diagnosis": "Major Polytrauma & Hemorrhagic Shock",
        "admitted": "Nov 18, 06:10",
    },
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

        if self.agent_id == "Agent_CO2":
            val = int(value or 0)
            return 1.0 if (15 <= val <= 75) else 0.0

        if self.agent_id in ("Agent_IBP", "Agent_NIBP"):
            if isinstance(value, dict):
                sys_v = int(value.get("sys", 120))
                dia_v = int(value.get("dia", 80))
            elif isinstance(value, str) and "/" in value:
                try:
                    sys_v = int(value.split("/")[0])
                    dia_v = int(value.split("/")[1].split()[0])
                except Exception:
                    sys_v, dia_v = 120, 80
            else:
                sys_v, dia_v = int(value or 120), 80
            return 1.0 if (50 <= sys_v <= 250 and 25 <= dia_v <= 140 and sys_v > dia_v) else 0.0

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
        self.last_mqtt_ts = 0.0
        self.simulation = PatientSimulation(patient_id)
        self.agents = {aid: AgentTrustTracker(aid) for aid in AGENT_IDS}
        self.edge_summary = {
            "patient_id": patient_id,
            "inf_ms": 0.28,
            "exec_provider": "CUDA / Orin GPU",
            "arrhythmia_prob": 0.02,
            "local_bed_alarm": False,
            "sensor_conflict": False,
            "latest_vitals": {
                "hr": 75,
                "spo2": 98,
                "co2": 38,
                "ibp": {"sys": 120, "dia": 80, "map": 93, "formatted": "120/80 (93)"},
                "nibp": {"sys": 120, "dia": 80, "map": 93, "formatted": "120/80"},
                "ecg_variance": 0.04,
            },
            "routing_path": "primary",
            "timestamp": time.time(),
        }
        initial_frame = self.simulation.sample({})
        self.latest_ecg_wave = initial_frame.waveforms.get("Agent_ECG") or self._default_ecg_wave()
        self.latest_ecg_end_ts = initial_frame.end_ts
        self.latest_ecg_rate = SAMPLE_RATE
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
        self.lock = threading.RLock()
        self.broker_host = broker_host or os.environ.get("MQTT_BROKER", "127.0.0.1")
        self.broker_port = int(os.environ.get("MQTT_PORT", broker_port))

        self.beds = {}
        self.added_bed_ids = {"Bed_1", "Bed_2", "Bed_3"}
        self.bed_patients = dict(DEFAULT_PATIENTS)
        self.discovered_jetsons = {}
        self.global_alert_log = deque(maxlen=30)
        self.stop_event = threading.Event()

        # High Availability configuration
        self.station_id = os.environ.get("NURSE_STATION_ID", "Nurse_Station_1")
        self.station_port = int(os.environ.get("NURSE_STATION_PORT", "8501"))
        self.is_primary = ("1" in self.station_id or "primary" in self.station_id.lower())
        self.peer_stations = {}
        self.last_primary_heartbeat_ts = time.time() if self.is_primary else 0.0
        self.failover_triggered = False
        self.failover_simulated = False

        # Department & Room mapping for hospital beds
        self.bed_departments = {
            "Bed_1": "ICU",
            "Bed_2": "CCU",
            "Bed_3": "NCU",
            "Bed_4": "SICU",
            "Bed_5": "Trauma / ER",
        }
        self.bed_rooms = {
            "Bed_1": "Room 101",
            "Bed_2": "Room 201",
            "Bed_3": "Room 301",
            "Bed_4": "Room 401",
            "Bed_5": "Room 501",
        }
        # Keep the display number with the bed when its patient is released.
        self.bed_numbers = {pid: int(pid.removeprefix("Bed_")) for pid in self.bed_rooms}

        # CPU fallback ONNX inference
        self.cpu_session = None
        self.cpu_input_name = "ecg_input"
        if ort is not None and os.path.exists(MODEL_FILE):
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

        # Initialize default beds so initial ward display is populated
        for b in ["Bed_1", "Bed_2", "Bed_3"]:
            self._get_or_create_bed(b)

        # MQTT Connection
        self.client = None
        self._init_mqtt()

        # Background simulator for admitted beds without physical hardware
        self.sim_thread = threading.Thread(target=self._run_background_synthesizer, daemon=True)
        self.sim_thread.start()

        # HA Heartbeat thread
        self.hb_thread = threading.Thread(target=self._run_heartbeat_broadcaster, daemon=True)
        self.hb_thread.start()

    def close(self):
        self.stop_event.set()
        if self.client:
            self.client.disconnect()
            self.client.loop_stop()
        self.sim_thread.join()
        self.hb_thread.join()

    def get_patient_info(self, pid):
        if pid in self.bed_patients:
            return self.bed_patients[pid]
        return {
            "name": f"Patient {pid}",
            "diagnosis": "Observation & Standard Telemetry",
            "admitted": "Nov 18, 08:00",
        }

    def set_patient_info(self, pid, name=None, diagnosis=None, admitted=None):
        with self.lock:
            if pid not in self.bed_patients:
                self.bed_patients[pid] = self.get_patient_info(pid)
            if name:
                self.bed_patients[pid]["name"] = name
            if diagnosis:
                self.bed_patients[pid]["diagnosis"] = diagnosis
            if admitted:
                self.bed_patients[pid]["admitted"] = admitted

    def get_bed_department(self, pid):
        if pid in self.bed_departments:
            return self.bed_departments[pid]
        return "ICU"

    def set_bed_department(self, pid, department):
        with self.lock:
            self.bed_departments[pid] = department

    def get_bed_room(self, pid):
        if pid in self.bed_rooms:
            return self.bed_rooms[pid]
        digits = "".join(filter(str.isdigit, pid))
        num = f"10{digits}" if digits else "105"
        return f"Room {num}"

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
            self.client.connect_async(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[DASHBOARD] MQTT connect notice: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            print(f"[DASHBOARD] MQTT connection rejected: {rc}")
            return
        print(f"[DASHBOARD] Connected to MQTT broker {self.broker_host}:{self.broker_port}")
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
                        bed = self._get_or_create_bed(patient_id)
                        bed.last_seen_ts = recv_ts
                        bed.last_mqtt_ts = recv_ts
                        bed.edge_summary = payload
                        ecg_sensor = (payload.get("sensors") or {}).get("Agent_ECG") or {}
                        summary_wave = ecg_sensor.get("waveform") or payload.get("ecg_wave")
                        summary_ts = float(
                            ecg_sensor.get("waveform_end_ts")
                            or ecg_sensor.get("sent_ts")
                            or payload.get("timestamp")
                            or recv_ts
                        )
                        if isinstance(summary_wave, list) and len(summary_wave) == 250 and summary_ts >= bed.latest_ecg_end_ts:
                            bed.latest_ecg_wave = summary_wave
                            bed.latest_ecg_end_ts = summary_ts
                            bed.latest_ecg_rate = int(ecg_sensor.get("sample_rate_hz") or SAMPLE_RATE)
                        if payload.get("routing_path") == "relay_b":
                            bed.active_routing = "Relay Path B (Rerouted)"
                            bed.offload_active = True
                        else:
                            bed.active_routing = "Primary Direct Link"
                            bed.offload_active = False
                    elif "alerts" in parts:
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
        if agent_id == "Agent_EtCO2":
            agent_id = "Agent_CO2"
        if agent_id not in AGENT_IDS:
            return

        reading = payload.get("reading")
        if (
            agent_id in {"Agent_IBP", "Agent_NIBP"}
            and not isinstance(payload.get("value"), dict)
            and isinstance(reading, dict)
            and {"systolic", "diastolic"} <= reading.keys()
        ):
            payload = {
                **payload,
                "value": {"sys": reading["systolic"], "dia": reading["diastolic"], "map": reading.get("map")},
            }

        with self.lock:
            bed = self._get_or_create_bed(patient_id)
            conflict = bool(bed.edge_summary.get("sensor_conflict", False))
            tracker = bed.agents[agent_id]
            tracker.update(payload=payload, recv_ts=recv_ts, cross_sensor_conflict=conflict)

            if agent_id == "Agent_ECG":
                ecg_wave = payload.get("ecg_wave", [])
                if ecg_wave and len(ecg_wave) == 250:
                    bed.latest_ecg_wave = ecg_wave
                    bed.latest_ecg_end_ts = float(
                        payload.get("waveform_end_ts") or payload.get("sent_ts") or recv_ts
                    )
                    bed.latest_ecg_rate = int(payload.get("sample_rate_hz") or SAMPLE_RATE)
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
                        "Agent_CO2": bed.agents["Agent_CO2"].trust_score,
                        "Agent_IBP": bed.agents["Agent_IBP"].trust_score,
                        "Agent_NIBP": bed.agents["Agent_NIBP"].trust_score,
                    }
                )

            bed.last_seen_ts = recv_ts
            bed.last_mqtt_ts = recv_ts

    def _get_or_create_bed(self, patient_id):
        if not patient_id:
            patient_id = "Bed_1"
        if patient_id not in self.beds:
            self.beds[patient_id] = PatientBedMonitor(patient_id)
        self.beds[patient_id].last_seen_ts = time.time()
        return self.beds[patient_id]

    def _run_heartbeat_broadcaster(self):
        while not self.stop_event.wait(1.0):
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
        last_logged_alarm = {}
        while not self.stop_event.wait(0.40):
            step += 1
            now = time.time()

            # Read chaos.json if present to keep bedside and nurse station 100% synchronized
            chaos_data = {}
            if os.path.exists(CHAOS_FILE):
                try:
                    with open(CHAOS_FILE, "r", encoding="utf-8") as f:
                        chaos_data = json.load(f)
                except Exception:
                    pass

            with self.lock:
                for pid in list(self.added_bed_ids):
                    bed = self._get_or_create_bed(pid)
                    # If live physical MQTT packets arrived within 2.0 seconds, let physical hardware drive
                    if (now - bed.last_mqtt_ts) <= 2.0:
                        continue

                    # Otherwise, synchronize from chaos.json (bedside simulator state)
                    bed_cfg = chaos_data.get(pid, {})
                    vitals_cfg = bed_cfg.get("vitals", {})
                    ecg_condition = vitals_cfg.get("ecg_condition") or bed_cfg.get("Agent_ECG", {}).get("condition", "normal")

                    orin_cfg = bed_cfg.get("Orin_Link", {})
                    use_relay_b = orin_cfg.get("use_relay_b", False)
                    overload_gpu = orin_cfg.get("overload_gpu", False)

                    frame = bed.simulation.sample(bed_cfg)
                    is_arrhythmia = (ecg_condition == "arrhythmia")
                    if is_arrhythmia:
                        arrhythmia_prob = 0.98
                        local_bed_alarm = True
                        if now - last_logged_alarm.get(pid, 0) > 10.0:
                            entry = (
                                f"[{time.strftime('%H:%M:%S')}] [{pid}] CRITICAL ARRHYTHMIA DETECTED (98.0%) "
                                f"via Primary Link [0.28 ms]"
                            )
                            bed.alert_log.appendleft(entry)
                            self.global_alert_log.appendleft(entry)
                            last_logged_alarm[pid] = now
                    else:
                        arrhythmia_prob = 0.02
                        local_bed_alarm = False

                    # Sensor Byzantine spoofing from chaos.json
                    ecg_mal = bed_cfg.get("Agent_ECG", {}).get("malicious", False) or (ecg_condition == "malicious")
                    sp_mal = bed_cfg.get("Agent_SpO2", {}).get("malicious", False)
                    hr_mal = bed_cfg.get("Agent_HR", {}).get("malicious", False)
                    co2_mal = bed_cfg.get("Agent_CO2", {}).get("malicious", False) or bed_cfg.get("Agent_EtCO2", {}).get("malicious", False)
                    ibp_mal = bed_cfg.get("Agent_IBP", {}).get("malicious", False)
                    nibp_mal = bed_cfg.get("Agent_NIBP", {}).get("malicious", False)

                    if ecg_mal:
                        times = frame.end_ts + np.arange(1 - WINDOW_SAMPLES, 1) / SAMPLE_RATE
                        noise = 2.1 * np.sin(2 * np.pi * 27 * times) + 1.65 * np.sin(2 * np.pi * 43 * times)
                        wave_list = np.round(noise, 4).tolist()
                    else:
                        wave_list = frame.waveforms["Agent_ECG"]
                    bed.latest_ecg_wave = wave_list
                    bed.latest_ecg_end_ts = frame.end_ts
                    bed.latest_ecg_rate = SAMPLE_RATE

                    hr_val = 240 if hr_mal else int(frame.readings["Agent_HR"])
                    spo2_val = 42 if sp_mal else int(frame.readings["Agent_SpO2"])
                    co2_val = 95 if co2_mal else int(frame.readings["Agent_EtCO2"])
                    ibp_reading = frame.readings["Agent_IBP"]
                    sys_val = 310 if ibp_mal else int(ibp_reading["systolic"])
                    dia_val = 20 if ibp_mal else int(ibp_reading["diastolic"])
                    map_val = int(round((sys_val + 2 * dia_val) / 3))

                    cuff_reading = frame.readings.get("Agent_NIBP") or ibp_reading
                    nibp_sys = 290 if nibp_mal else int(cuff_reading["systolic"])
                    nibp_dia = 25 if nibp_mal else int(cuff_reading["diastolic"])
                    nibp_map = int(round((nibp_sys + 2 * nibp_dia) / 3))

                    ibp_val = {"sys": sys_val, "dia": dia_val, "map": map_val, "formatted": f"{sys_val}/{dia_val} ({map_val})"}
                    nibp_val = {"sys": nibp_sys, "dia": nibp_dia, "map": nibp_map, "formatted": f"{nibp_sys}/{nibp_dia}"}
                    var_val = float(np.var(wave_list))

                    if use_relay_b:
                        bed.active_routing = "Relay Path B (Rerouted)"
                        bed.offload_active = True
                        bed.gateway_cpu_inf_ms = 4.2
                    else:
                        bed.active_routing = "Primary Direct Link"
                        bed.offload_active = False
                        bed.gateway_cpu_inf_ms = 0.0

                    inf_val = 18.5 if overload_gpu else round(0.22 + float(np.random.uniform(0.01, 0.06)), 2)

                    bed.edge_summary["latest_vitals"] = {
                        "hr": hr_val,
                        "spo2": spo2_val,
                        "co2": co2_val,
                        "ibp": ibp_val,
                        "nibp": nibp_val,
                        "ecg_variance": round(var_val, 4),
                    }
                    bed.edge_summary["inf_ms"] = inf_val
                    bed.edge_summary["arrhythmia_prob"] = arrhythmia_prob
                    bed.edge_summary["local_bed_alarm"] = local_bed_alarm

                    bed.agents["Agent_ECG"].update(
                        seq=step, sent_ts=now - 0.01, recv_ts=now, value=round(var_val, 4), ecg_wave=wave_list
                    )
                    bed.agents["Agent_SpO2"].update(seq=step, sent_ts=now - 0.01, recv_ts=now, value=spo2_val)
                    bed.agents["Agent_HR"].update(seq=step, sent_ts=now - 0.01, recv_ts=now, value=hr_val)
                    bed.agents["Agent_CO2"].update(seq=step, sent_ts=now - 0.01, recv_ts=now, value=co2_val)
                    bed.agents["Agent_IBP"].update(seq=step, sent_ts=now - 0.01, recv_ts=now, value=ibp_val)
                    bed.agents["Agent_NIBP"].update(seq=step, sent_ts=now - 0.01, recv_ts=now, value=nibp_val)
                    bed.last_seen_ts = now

                    if step % 3 == 0:
                        bed.trust_history.append(
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(now)),
                                "Agent_ECG": bed.agents["Agent_ECG"].trust_score,
                                "Agent_SpO2": bed.agents["Agent_SpO2"].trust_score,
                                "Agent_HR": bed.agents["Agent_HR"].trust_score,
                                "Agent_CO2": bed.agents["Agent_CO2"].trust_score,
                                "Agent_IBP": bed.agents["Agent_IBP"].trust_score,
                                "Agent_NIBP": bed.agents["Agent_NIBP"].trust_score,
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

    def admit_jetson(self, patient_id, department=None, room_name=None, patient_name=None, diagnosis=None):
        with self.lock:
            if patient_id not in self.bed_numbers:
                suffix = patient_id.removeprefix("Bed_")
                number = int(suffix) if suffix.isdecimal() else None
                used = set(self.bed_numbers.values())
                if number is None or number in used:
                    number = next(n for n in range(1, len(used) + 2) if n not in used)
                self.bed_numbers[patient_id] = number
            changed_department = department and department != self.get_bed_department(patient_id)
            self.added_bed_ids.add(patient_id)
            if department:
                self.bed_departments[patient_id] = department
            if room_name:
                self.bed_rooms[patient_id] = room_name
            elif patient_id not in self.bed_rooms or changed_department:
                dept_idx = next(
                    (i + 1 for i, (code, _) in enumerate(HOSPITAL_DEPARTMENTS) if code == self.get_bed_department(patient_id)),
                    1,
                )
                digits = "".join(filter(str.isdigit, patient_id)) or str(len(self.added_bed_ids))
                self.bed_rooms[patient_id] = f"Room {dept_idx}0{digits}"
            if patient_name or diagnosis:
                cur = self.get_patient_info(patient_id)
                self.bed_patients[patient_id] = {
                    "name": patient_name or cur.get("name", f"Patient {patient_id}"),
                    "diagnosis": diagnosis or cur.get("diagnosis", "Observation & Standard Telemetry"),
                    "admitted": cur.get("admitted", time.strftime("Nov %d, %H:%M")),
                }
            self._get_or_create_bed(patient_id)
            dept_lbl = self.get_bed_department(patient_id)
            rm_lbl = self.get_bed_room(patient_id)
            p_lbl = self.get_patient_info(patient_id)["name"]
            if self.client:
                try:
                    self.client.publish(
                        f"hospital/{patient_id}/control/admit",
                        json.dumps({"admitted": True, "department": dept_lbl, "room": rm_lbl}),
                        qos=1,
                    )
                except Exception:
                    pass
            entry = f"[{time.strftime('%H:%M:%S')}] [WARD ADMISSION] {p_lbl} ({patient_id}) admitted to {dept_lbl} ({rm_lbl})."
            self.global_alert_log.appendleft(entry)

    def admit_simulated_jetson(self, department, room_name=None):
        with self.lock:
            reserved = self.added_bed_ids | self.discovered_jetsons.keys()
            index = 1
            while f"Bed_{index}" in reserved:
                index += 1
            patient_id = f"Bed_{index}"
            self.admit_jetson(patient_id, department=department, room_name=room_name)
            return patient_id

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
        chaos_data = {}
        if os.path.exists(CHAOS_FILE):
            try:
                with open(CHAOS_FILE, "r", encoding="utf-8") as f:
                    chaos_data = json.load(f)
            except Exception:
                pass
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
                pinfo = self.get_patient_info(pid)
                beds_snap[pid] = {
                    "patient_id": pid,
                    "patient_name": pinfo.get("name", f"Patient {pid}"),
                    "diagnosis": pinfo.get("diagnosis", "Observation & Telemetry"),
                    "admitted": pinfo.get("admitted", "Nov 18, 08:00"),
                    "department": self.get_bed_department(pid),
                    "room_only": self.get_bed_room(pid),
                    "room": f"{self.get_bed_department(pid)} · {self.get_bed_room(pid)}",
                    "online": (now - bed.last_seen_ts) <= 8.0,
                    "seconds_since_seen": round(now - bed.last_seen_ts, 1),
                    "agents": agents_snap,
                    "edge": dict(bed.edge_summary),
                    "ecg_wave": list(bed.latest_ecg_wave),
                    "ecg_end_ts": bed.latest_ecg_end_ts,
                    "ecg_rate": bed.latest_ecg_rate,
                    "chaos": chaos_data.get(pid, {}),
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
                            "department": self.get_bed_department(pid),
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
                "server_time": now,
                "beds": beds_snap,
                "bed_directory": [
                    {"patient_id": pid, "number": str(number), "department": self.get_bed_department(pid),
                     "room": self.get_bed_room(pid)}
                    for pid, number in sorted(self.bed_numbers.items(), key=lambda item: item[1])
                ],
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
