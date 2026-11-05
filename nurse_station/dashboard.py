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
        self.broker_host = broker_host or os.environ.get("MQTT_BROKER", "127.0.0.1")
        self.broker_port = int(os.environ.get("MQTT_PORT", broker_port))
        self.beds = {}
        self.event_logs = deque(maxlen=60)

        self.ort_session = None
        self.ort_input_name = "ecg_input"
        if os.path.exists(MODEL_FILE):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self.ort_session = ort.InferenceSession(
                MODEL_FILE, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.ort_input_name = self.ort_session.get_inputs()[0].name

        cid = f"NurseStation_Engine_{os.getpid()}_{ int(time.time() * 1000) % 10000 }"
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

    def _on_message(self, client, userdata, msg):
        rx_time = time.time()
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            parts = msg.topic.split("/")
            pid = payload.get("patient_id") or (parts[1] if len(parts) > 2 else "Bed_1")
            if pid == "control":
                return

            with self.lock:
                bed = self._ensure_bed_locked(pid)
                bed["last_seen"] = rx_time

                if "/sensors/" in msg.topic:
                    aid = payload.get("agent_id")
                    if aid in bed["trackers"]:
                        tr = bed["trackers"][aid]
                        prev_status = tr.status
                        tr.update(payload, rx_time)
                        if prev_status != tr.status:
                            lvl = "CRITICAL" if tr.status == "ISOLATED" else ("WARN" if tr.status == "SUSPICIOUS" else "INFO")
                            self._log(pid, lvl, f"{AGENT_LABELS.get(aid, aid)} {prev_status} -> {tr.status} (T={tr.trust_score:.2f})")

                        if aid == "Agent_ECG" and bed["edge_summary"].get("gpu_overloaded", False):
                            if self.ort_session is not None and tr.status != "ISOLATED":
                                t0 = time.perf_counter()
                                x = np.array(tr.ecg_wave, dtype=np.float32).reshape(1, 1, 250)
                                logits = self.ort_session.run(None, {self.ort_input_name: x})[0]
                                exp_s = np.exp(logits - np.max(logits))
                                prob = float((exp_s / np.sum(exp_s))[0][1])
                                inf_ms = (time.perf_counter() - t0) * 1000.0
                                bed["edge_summary"]["inference_ms"] = round(inf_ms, 2)
                                bed["edge_summary"]["anomaly_prob"] = round(prob, 4)
                                bed["edge_summary"]["is_arrhythmia"] = prob > 0.50
                                bed["edge_summary"]["execution_provider"] = "Gateway CPU (Offloaded)"

                elif msg.topic.endswith("/edge/summary"):
                    bed["edge_summary"].update(payload)
                    if payload.get("gpu_overloaded", False) or payload.get("inference_ms", 0.0) > 25.0:
                        bed["compute_mode"] = "Offloaded to Gateway"
                    else:
                        bed["compute_mode"] = "Jetson Local"

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
            out = {}
            for pid, bed in sorted(self.beds.items()):
                online = (now - bed["last_seen"]) < 6.0
                for aid, tr in bed["trackers"].items():
                    changed, old_s, new_s = tr.evaluate_status(now)
                    if changed:
                        lvl = "CRITICAL" if new_s == "ISOLATED" else ("WARN" if new_s == "SUSPICIOUS" else "INFO")
                        self._log(pid, lvl, f"{AGENT_LABELS.get(aid, aid)} {old_s} -> {new_s} (T={tr.trust_score:.2f})")

                ecg_t = bed["trackers"]["Agent_ECG"].trust_score
                spo2_t = bed["trackers"]["Agent_SpO2"].trust_score
                hr_t = bed["trackers"]["Agent_HR"].trust_score

                if not bed["history"] or (now - bed["history"][-1]["ts"]) >= 0.45:
                    bed["history"].append({"ts": now, "ecg": ecg_t, "spo2": spo2_t, "hr": hr_t})

                if bed["edge_summary"].get("link_congested", False) or ecg_t < 0.65:
                    bed["alert_route"] = "Relay B"
                else:
                    bed["alert_route"] = "Direct"

                out[pid] = {
                    "online": online,
                    "last_seen_ago": round(now - bed["last_seen"], 1),
                    "compute_mode": bed["compute_mode"],
                    "alert_route": bed["alert_route"],
                    "edge": dict(bed["edge_summary"]),
                    "history": list(bed["history"]),
                    "agents": {
                        aid: {
                            "trust": round(tr.trust_score, 3),
                            "s_l": round(tr.s_l, 2),
                            "s_j": round(tr.s_j, 2),
                            "s_p": round(tr.s_p, 2),
                            "s_d": round(tr.s_d, 2),
                            "status": tr.status,
                            "value": tr.last_value,
                            "ecg_wave": list(tr.ecg_wave),
                        }
                        for aid, tr in bed["trackers"].items()
                    },
                }
            return out, list(self.event_logs)

    def remove_offline(self):
        now = time.time()
        with self.lock:
            stale = [pid for pid, b in self.beds.items() if (now - b["last_seen"]) >= 6.0]
            for pid in stale:
                del self.beds[pid]


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


def status_pill_info(bed):
    if not bed["online"]:
        return "OFFLINE", "pill-muted"
    isolated = [a for a, d in bed["agents"].items() if d["status"] == "ISOLATED"]
    suspicious = [a for a, d in bed["agents"].items() if d["status"] == "SUSPICIOUS"]
    if bed["edge"].get("is_arrhythmia") and bed["agents"]["Agent_ECG"]["status"] != "ISOLATED":
        return "ALARM", "pill-red"
    if isolated:
        return "SENSOR ISOLATED", "pill-red"
    if suspicious or bed["compute_mode"] != "Jetson Local":
        return "DEGRADED" if suspicious else "OFFLOADED", "pill-amber"
    return "STABLE", "pill-green"


def dot_color(status):
    if status == "TRUSTED":
        return "#10b981"
    if status == "SUSPICIOUS":
        return "#f59e0b"
    return "#ef4444"


def ward_tile(pid, bed, is_selected):
    label, pill_cls = status_pill_info(bed)
    hr_ag = bed["agents"]["Agent_HR"]
    sp_ag = bed["agents"]["Agent_SpO2"]
    ecg_ag = bed["agents"]["Agent_ECG"]

    hr_txt = "ISO" if hr_ag["status"] == "ISOLATED" else (str(hr_ag["value"]) if hr_ag["value"] is not None else "--")
    sp_txt = "ISO" if sp_ag["status"] == "ISOLATED" else (f"{sp_ag['value']}%" if sp_ag["value"] is not None else "--")
    ecg_col = "#ef4444" if (ecg_ag["status"] == "ISOLATED" or bed["edge"].get("is_arrhythmia")) else "#10b981"
    border_col = "#3b82f6" if is_selected else "#1f2228"

    return (
        f'<div class="tile" style="border-color:{border_col};">'
        f'<div class="row-between">'
        f'<span class="bed-id">{pid}</span>'
        f'<span class="pill {pill_cls}">{label}</span>'
        f"</div>"
        f'<div class="ward-vitals">'
        f'<div><div class="lbl">HR</div><div class="val-sm">{hr_txt} <span class="unit">bpm</span></div></div>'
        f'<div><div class="lbl">SpO2</div><div class="val-sm">{sp_txt}</div></div>'
        f'<div><div class="lbl">Mode</div><div class="val-xs">{bed["compute_mode"]}</div></div>'
        f"</div>"
        f'<div style="margin:10px 0 8px 0;">{svg_ecg(ecg_ag["ecg_wave"], color=ecg_col, width=360, height=54)}</div>'
        f'<div class="trust-dots">'
        f'<span><i class="dot" style="background:{dot_color(ecg_ag["status"])}"></i>ECG {ecg_ag["trust"]:.2f}</span>'
        f'<span><i class="dot" style="background:{dot_color(sp_ag["status"])}"></i>SpO2 {sp_ag["trust"]:.2f}</span>'
        f'<span><i class="dot" style="background:{dot_color(hr_ag["status"])}"></i>HR {hr_ag["trust"]:.2f}</span>'
        f"</div>"
        f"</div>"
    )


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

    return (
        f'<div class="bento">'
        # Row 1: ECG Waveform (8) + Inference & Rhythm (4)
        f'<div class="tile span-8">'
        f'<div class="row-between"><span class="lbl">ECG Lead II &mdash; {pid}</span><span class="sub">250 Hz</span></div>'
        f'<div style="margin-top:10px;">{svg_ecg(ecg_ag["ecg_wave"], color=ecg_col, width=720, height=150)}</div>'
        f"</div>"
        f'<div class="tile span-4">'
        f'<div class="lbl">Rhythm &amp; Edge Inference</div>'
        f'<div class="val-md" style="color:{rhythm_col};margin-top:6px;">{rhythm_txt}</div>'
        f'<div class="kv-list">'
        f'<div class="kv"><span>Arrhythmia risk</span><span>{edge.get("anomaly_prob", 0.0) * 100:.1f}%</span></div>'
        f'<div class="kv"><span>Inference latency</span><span>{edge.get("inference_ms", 0.0):.2f} ms</span></div>'
        f'<div class="kv"><span>Compute state</span><span>{bed["compute_mode"]}</span></div>'
        f'<div class="kv"><span>Alert path</span><span>{bed["alert_route"]}</span></div>'
        f"</div>"
        f"</div>"
        # Row 2: Three Sensor Trust Tiles (4 + 4 + 4)
        + sensor_tile("ECG sensor", ecg_ag, "")
        + sensor_tile("SpO2 sensor", sp_ag, "%")
        + sensor_tile("HR sensor", hr_ag, "bpm")
        # Row 3: Trust History (8) + Recent Events (4)
        + f'<div class="tile span-8">'
        f'<div class="row-between">'
        f'<span class="lbl">Trust score timeline</span>'
        f'<span class="legend">'
        f'<span><i class="dot" style="background:#10b981"></i>ECG</span>'
        f'<span><i class="dot" style="background:#3b82f6"></i>SpO2</span>'
        f'<span><i class="dot" style="background:#f59e0b"></i>HR</span>'
        f"</span>"
        f"</div>"
        f'<div style="margin-top:10px;">{svg_trust_chart(bed["history"], width=720, height=130)}</div>'
        f"</div>"
        f'<div class="tile span-4">'
        f'<div class="lbl">Events</div>'
        f'<div class="log-list">{rows}</div>'
        f"</div>"
        f"</div>"
    )


BENTO_CSS = """
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"], header, footer, #MainMenu {
    display: none !important;
  }
  .stApp {
    background-color: #0a0a0c;
    color: #ededef;
  }
  .block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2rem !important;
    max-width: 1320px !important;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    margin-bottom: 16px;
    border-bottom: 1px solid #1f2228;
  }
  .title {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #ededef;
  }
  .meta {
    font-size: 12px;
    color: #7c828d;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .sec-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7c828d;
    margin: 14px 0 10px 0;
  }
  .ward-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }
  .bento {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 12px;
  }
  .tile {
    background: #121316;
    border: 1px solid #1f2228;
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .span-8 { grid-column: span 8; }
  .span-4 { grid-column: span 4; }
  .span-12 { grid-column: span 12; }
  @media (max-width: 960px) {
    .span-8, .span-4 { grid-column: span 12; }
  }
  .row-between {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .bed-id {
    font-size: 15px;
    font-weight: 600;
    color: #ededef;
  }
  .lbl {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #7c828d;
  }
  .sub {
    font-size: 12px;
    color: #7c828d;
    font-variant-numeric: tabular-nums;
  }
  .val-lg {
    font-size: 30px;
    font-weight: 600;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
  }
  .val-md {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .val-sm {
    font-size: 20px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
  }
  .val-xs {
    font-size: 13px;
    font-weight: 500;
    color: #ededef;
    margin-top: 6px;
  }
  .unit {
    font-size: 11px;
    color: #7c828d;
    font-weight: 400;
  }
  .ward-vitals {
    display: grid;
    grid-template-columns: 1fr 1fr 1.2fr;
    gap: 8px;
    margin-top: 12px;
  }
  .pill {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 999px;
  }
  .pill-green { color: #10b981; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }
  .pill-amber { color: #f59e0b; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); }
  .pill-red { color: #ef4444; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); }
  .pill-muted { color: #7c828d; background: #181a1f; border: 1px solid #262932; }
  .trust-dots {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #7c828d;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .legend {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: #7c828d;
  }
  .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 5px;
  }
  .bar-bg {
    width: 100%;
    height: 5px;
    background: #1c1f26;
    border-radius: 999px;
    overflow: hidden;
    margin: 10px 0 8px 0;
  }
  .bar-fill {
    height: 100%;
    border-radius: 999px;
  }
  .sub-metrics {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #7c828d;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .kv-list {
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .kv {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #7c828d;
    padding-top: 6px;
    border-top: 1px solid #1b1e24;
  }
  .kv span:last-child {
    color: #ededef;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .log-list {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .log-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    padding-top: 6px;
    border-top: 1px solid #1b1e24;
  }
  .log-ts {
    color: #7c828d;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .log-lvl {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 4px;
  }
  .log-critical { color: #ef4444; background: rgba(239, 68, 68, 0.12); }
  .log-warn { color: #f59e0b; background: rgba(245, 158, 11, 0.12); }
  .log-info { color: #10b981; background: rgba(16, 185, 129, 0.12); }
  .log-msg { color: #ededef; }
</style>
"""


def main():
    st.set_page_config(page_title="Nurse Station", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(BENTO_CSS, unsafe_allow_html=True)

    engine = get_engine("v4_clean_bento")
    beds_snap, logs = engine.snapshot()
    bed_ids = list(beds_snap.keys())
    online_count = sum(1 for b in beds_snap.values() if b["online"])

    st.markdown(
        f'<div class="topbar">'
        f'<div class="title">Central Nurse Station</div>'
        f'<div class="meta">Broker {engine.broker_host}:{engine.broker_port} &nbsp;|&nbsp; {online_count} active bed(s)</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if not bed_ids:
        st.markdown(
            '<div class="tile" style="padding:32px;text-align:center;">'
            '<div class="val-md">Waiting for bedside nodes</div>'
            '<div class="sub" style="margin-top:8px;">'
            "Start a Jetson bedside node pointed at this broker: "
            "<code>./start_bedside.sh --patient-id Bed_1 --broker &lt;LAPTOP_IP&gt;</code>"
            "</div></div>",
            unsafe_allow_html=True,
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
