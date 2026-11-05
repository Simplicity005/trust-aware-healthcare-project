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
        # Dynamically populated as Jetson bedside nodes publish over MQTT
        self.beds = {}
        self.global_alert_log = deque(maxlen=15)

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

    def _get_or_create_bed(self, patient_id):
        if not patient_id:
            patient_id = "Bed_1"
        if patient_id not in self.beds:
            print(f"[NURSE STATION] Discovered new Bedside Jetson node: {patient_id}")
            self.beds[patient_id] = PatientBedMonitor(patient_id)
        self.beds[patient_id].last_seen_ts = time.time()
        return self.beds[patient_id]

    def remove_inactive_beds(self, timeout_sec=8.0):
        """Removes beds that have stopped transmitting for longer than timeout_sec."""
        now = time.time()
        with self.lock:
            stale = [pid for pid, b in self.beds.items() if (now - b.last_seen_ts) > timeout_sec]
            for pid in stale:
                del self.beds[pid]

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

    def _on_message(self, client, userdata, msg):
        recv_ts = time.time()
        try:
            topic = msg.topic
            parts = topic.split("/")
            payload = json.loads(msg.payload.decode("utf-8"))
            patient_id = payload.get("patient_id")

            if not patient_id and len(parts) >= 3 and parts[1] not in ("sensors", "edge", "alerts", "control"):
                patient_id = parts[1]
            if not patient_id:
                patient_id = "Bed_1"

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
            return {
                "beds": beds_snap,
                "global_alert_log": list(self.global_alert_log),
            }


@st.cache_resource
def get_nurse_station_engine():
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


def render_sidebar_chaos_controller(engine, bed_ids):
    """Renders per-patient Sidebar Chaos Controller dynamically for active Jetson beds."""
    st.sidebar.title("⚡ Live Fault Injection (Chaos Engine)")
    st.sidebar.caption(
        "Beds are discovered automatically as Jetsons connect. Select any active bed below to inject faults."
    )

    if not bed_ids:
        st.sidebar.info("⏳ Waiting for at least 1 Bedside Jetson node to connect...")
        auto_refresh = st.sidebar.checkbox("⏱️ Live Auto-Refresh (0.6s)", value=True)
        return auto_refresh

    all_chaos = read_all_chaos_config(bed_ids)

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("🔄 Reset Faults", use_container_width=True):
            baseline = {bed: default_bed_chaos() for bed in bed_ids}
            write_all_chaos_config(baseline)
            for bed in bed_ids:
                engine.publish_patient_chaos(bed, baseline[bed], baseline)
            st.rerun()
    with col_b:
        if st.button("🗑️ Prune Offline", use_container_width=True):
            engine.remove_inactive_beds(timeout_sec=6.0)
            st.rerun()

    st.sidebar.divider()

    target_bed = st.sidebar.selectbox(
        f"🛏️ Target Patient Bed ({len(bed_ids)} Connected)",
        options=bed_ids,
        index=0,
        key="chaos_target_bed_select",
    )

    bed_cfg = all_chaos.get(target_bed, default_bed_chaos())
    new_bed_cfg = {}

    agent_labels = {
        "Agent_ECG": "🫀 Agent_ECG (Waveform Sensor)",
        "Agent_SpO2": "🫁 Agent_SpO2 (Pulse Oximeter)",
        "Agent_HR": "💓 Agent_HR (Heart Rate Monitor)",
    }

    for agent_id in AGENT_IDS:
        st.sidebar.subheader(f"{agent_labels[agent_id]} [{target_bed}]")
        cfg = bed_cfg.get(agent_id, {"delay_ms": 10, "drop_rate": 0.0, "malicious": False})

        delay_ms = st.sidebar.slider(
            f"{agent_id} Delay (ms)",
            min_value=5,
            max_value=400,
            value=int(cfg.get("delay_ms", 10)),
            step=5,
            key=f"delay_{target_bed}_{agent_id}",
        )
        drop_rate = st.sidebar.slider(
            f"{agent_id} Packet Drop Rate",
            min_value=0.0,
            max_value=0.6,
            value=float(cfg.get("drop_rate", 0.0)),
            step=0.05,
            key=f"drop_{target_bed}_{agent_id}",
        )
        malicious = st.sidebar.checkbox(
            f"Inject Spoofed/Malicious Vitals ({agent_id})",
            value=bool(cfg.get("malicious", False)),
            key=f"mal_{target_bed}_{agent_id}",
        )
        new_bed_cfg[agent_id] = {
            "delay_ms": delay_ms,
            "drop_rate": drop_rate,
            "malicious": malicious,
        }
        st.sidebar.markdown("---")

    st.sidebar.subheader(f"🖧 {target_bed} Orin Link & Compute")
    orin_cfg = bed_cfg.get("Orin_Link", {"overload_gpu": False, "use_relay_b": False})
    sim_congest = st.sidebar.checkbox(
        f"Simulate {target_bed} Orin Link Congestion / Offload",
        value=bool(orin_cfg.get("overload_gpu", False) or orin_cfg.get("use_relay_b", False)),
        key=f"orin_congest_{target_bed}",
    )
    new_bed_cfg["Orin_Link"] = {
        "overload_gpu": sim_congest,
        "use_relay_b": sim_congest,
    }

    if new_bed_cfg != bed_cfg:
        all_chaos[target_bed] = new_bed_cfg
        write_all_chaos_config(all_chaos)
        engine.publish_patient_chaos(target_bed, new_bed_cfg, all_chaos)

    st.sidebar.divider()
    auto_refresh = st.sidebar.checkbox("⏱️ Live Auto-Refresh (0.6s)", value=True)
    return auto_refresh


def render_multi_patient_ward_overview(beds_snap):
    """Renders top-level ICU Ward Overview cards dynamically for all connected Jetson nodes."""
    bed_ids = list(beds_snap.keys())
    st.subheader(
        f"🏥 ICU Ward Live Overview — {len(bed_ids)} Active Bedside Jetson Node(s) Connected"
    )
    cols = st.columns(max(1, len(bed_ids)))

    for idx, pid in enumerate(bed_ids):
        b = beds_snap[pid]
        edge = b["edge_summary"]
        agents = b["agents"]
        vitals = edge.get("latest_vitals", {})

        any_isolated = any(ag["status"].startswith("ISOLATED") for ag in agents.values())
        any_offloaded = b["offload_active"]
        local_alarm = bool(edge.get("local_bed_alarm", False)) and not agents["Agent_ECG"][
            "status"
        ].startswith("ISOLATED")

        if not b["online"]:
            banner_bg, banner_fg, banner_txt = "#475569", "#f1f5f9", f"⚠️ OFFLINE ({b['seconds_since_seen']}s)"
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
                    f'<span style="font-size:1.1rem;font-weight:800;">🛏️ {pid}</span>'
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

    st.title("🏥 Trust-Aware Multi-Patient ICU Edge Intelligence Dashboard")
    st.caption(
        "Central Nurse Station • Dynamic Bedside Jetson Discovery • Real-Time Trust Scoring, Offloading & Sensor Quarantine"
    )

    engine = get_nurse_station_engine()
    all_snap = engine.get_all_beds_snapshot()
    beds_snap = all_snap["beds"]
    bed_ids = list(beds_snap.keys())

    auto_refresh = render_sidebar_chaos_controller(engine, bed_ids)

    if not bed_ids:
        st.info(
            "📡 **Listening for Bedside Jetson Orin nodes on MQTT Broker...**\n\n"
            "No patient beds are hardcoded! As soon as you start `./start_bedside.sh --patient-id <Bed_Name> --broker <LAPTOP_IP>` "
            "on one or more Jetsons, each patient bed will automatically appear here in real time."
        )
        if auto_refresh:
            time.sleep(0.6)
            st.rerun()
        return

    # 1. Top Multi-Patient Ward Summary Row
    render_multi_patient_ward_overview(beds_snap)

    st.divider()

    # 2. Patient Bed Selector for Detailed Dual-View Telemetry
    selected_bed = st.radio(
        "🔍 **Select Connected Patient Bed for Detailed Bedside & Trust Inspection:**",
        options=bed_ids,
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
        st.subheader(f"🖥️ {selected_bed} — NVIDIA Jetson Orin Bedside Monitor")

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
