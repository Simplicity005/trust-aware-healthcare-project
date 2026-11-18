#!/usr/bin/env python3
"""Bedside sensor streamers for seven patient-monitoring channels."""

import os
import time
import json
import random
import signal
import argparse
import threading
import numpy as np
import paho.mqtt.client as mqtt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAOS_FILE = os.path.join(BASE_DIR, "chaos.json")
CHAOS_LOCK = threading.Lock()


def default_bed_chaos():
    settings = {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_CO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_co2": 38},
        "Agent_Respiration": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_EtCO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_IBP": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_sys": 120, "target_dia": 80},
        "Agent_NIBP": {"delay_ms": 15, "drop_rate": 0.0, "malicious": False, "target_sys": 120, "target_dia": 80},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False},
        "vitals": {
            "target_hr": 75,
            "target_spo2": 98,
            "target_co2": 38,
            "target_etco2": 38,
            "target_sys": 120,
            "target_dia": 80,
            "target_respiration": 16,
            "ecg_condition": "normal",
        },
    }
    return settings


def read_patient_chaos(patient_id="Bed_1"):
    """Reads chaos settings for a specific patient bed with fallback."""
    if not os.path.exists(CHAOS_FILE):
        return default_bed_chaos()
    try:
        with CHAOS_LOCK:
            with open(CHAOS_FILE, "r") as f:
                data = json.load(f)
        if patient_id in data and isinstance(data[patient_id], dict):
            return data[patient_id]
        if "Agent_ECG" in data:
            return data
        return default_bed_chaos()
    except Exception:
        return default_bed_chaos()


def update_chaos_file(new_data, patient_id=None):
    """Updates chaos.json safely when an MQTT chaos control packet arrives."""
    try:
        with CHAOS_LOCK:
            current = {}
            if os.path.exists(CHAOS_FILE):
                try:
                    with open(CHAOS_FILE, "r") as f:
                        current = json.load(f)
                except Exception:
                    current = {}
            if patient_id and "Agent_ECG" in new_data:
                current[patient_id] = new_data
            else:
                current.update(new_data)
            tmp_path = CHAOS_FILE + f".tmp.{random.randint(1000, 9999)}"
            with open(tmp_path, "w") as f:
                json.dump(current, f, indent=2)
            os.replace(tmp_path, CHAOS_FILE)
    except Exception:
        pass


def smooth_vital(base, timestamp, phase_offset, amplitude, lower, upper):
    variation = amplitude * (
        0.65 * np.sin(timestamp * 0.055 + phase_offset)
        + 0.35 * np.sin(timestamp * 0.019 + phase_offset * 1.7)
    )
    return int(np.clip(round(base + variation), lower, upper))


def generate_ecg_wave(condition="normal", phase_offset=0.0, heart_rate=75, start_time=None):
    """Generate a 250 Hz PQRST waveform whose beat spacing follows heart rate."""
    t = np.arange(250, dtype=np.float64) / 250.0
    if condition == "malicious":
        wave = np.random.uniform(-2.5, 2.5, 250) + np.sin(2 * np.pi * 35 * t) * 1.5
        return np.round(np.clip(wave, -4.0, 4.0), 4).tolist()

    if condition == "arrhythmia":
        # Ventricular Tachycardia / Acute Arrhythmia pattern (broad erratic QRS complexes)
        c = phase_offset
        pvc1 = 1.75 * np.exp(-((t - (0.22 + c)) ** 2) / (2 * 0.035 ** 2))
        pvc2 = -1.50 * np.exp(-((t - (0.42 + c)) ** 2) / (2 * 0.040 ** 2))
        pvc3 = 1.85 * np.exp(-((t - (0.62 + c)) ** 2) / (2 * 0.035 ** 2))
        pvc4 = -1.35 * np.exp(-((t - (0.82 + c)) ** 2) / (2 * 0.040 ** 2))
        noise = np.random.normal(0, 0.03, 250)
        ecg = pvc1 + pvc2 + pvc3 + pvc4 + noise
        return np.round(ecg, 4).tolist()

    epoch = time.time() if start_time is None else start_time
    phase_origin = (epoch * heart_rate / 60.0) % 1.0
    beat_phase = (t * heart_rate / 60.0 + phase_origin + phase_offset) % 1.0

    def wave(center, width, amplitude):
        distance = (beat_phase - center + 0.5) % 1.0 - 0.5
        return amplitude * np.exp(-(distance ** 2) / (2 * width ** 2))

    p_wave = wave(0.18, 0.025, 0.15)
    q_wave = wave(0.34, 0.010, -0.15)
    r_wave = wave(0.37, 0.012, 1.25)
    s_wave = wave(0.40, 0.012, -0.25)
    t_wave = wave(0.64, 0.040, 0.30)
    noise = np.random.normal(0, 0.02, 250)
    ecg = p_wave + q_wave + r_wave + s_wave + t_wave + noise
    return np.round(ecg, 4).tolist()


class SensorAgentThread(threading.Thread):
    def __init__(self, patient_id, agent_id, primary_broker, backup_broker, port, stop_event):
        super().__init__(daemon=True)
        self.patient_id = patient_id
        self.agent_id = agent_id
        self.primary_broker = primary_broker
        self.backup_broker = backup_broker
        self.port = port
        self.stop_event = stop_event
        self.seq = 0
        self.current_broker = primary_broker
        self.client = None
        bed_hash = sum(ord(ch) for ch in patient_id) % 5
        self.hr_offset = bed_hash * 2 - 4
        self.phase_offset = (bed_hash - 2) * 0.01

    def _init_client(self):
        try:
            cid = f"{self.patient_id}_{self.agent_id}_{os.getpid()}_{random.randint(1000, 9999)}"
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
            else:
                self.client = mqtt.Client(client_id=cid)
            self.client.connect(self.current_broker, self.port, keepalive=60)
            self.client.loop_start()
            print(f"[{self.patient_id}:{self.agent_id}] Connected to broker {self.current_broker}:{self.port}")
            return True
        except Exception as e:
            print(f"[{self.patient_id}:{self.agent_id}] Failed to connect to {self.current_broker}: {e}")
            return False

    def _failover_broker(self):
        next_broker = (
            self.backup_broker if self.current_broker == self.primary_broker else self.primary_broker
        )
        print(
            f"[{self.patient_id}:{self.agent_id}] Broker failover: {self.current_broker} -> {next_broker}"
        )
        self.current_broker = next_broker
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        return self._init_client()

    def run(self):
        if not self._init_client():
            self._failover_broker()

        topic = f"hospital/{self.patient_id}/sensors/{self.agent_id}"

        while not self.stop_event.is_set():
            loop_start = time.time()
            self.seq += 1

            bed_chaos = read_patient_chaos(self.patient_id)
            agent_chaos = bed_chaos.get(
                self.agent_id, {"delay_ms": 10, "drop_rate": 0.0, "malicious": False}
            )
            if self.agent_id == "Agent_EtCO2":
                agent_chaos = {
                    **bed_chaos.get("Agent_CO2", {}),
                    **agent_chaos,
                }

            drop_rate = float(agent_chaos.get("drop_rate", 0.0))
            delay_ms = float(agent_chaos.get("delay_ms", 10.0))
            malicious = bool(agent_chaos.get("malicious", False))

            if random.random() < drop_rate:
                time.sleep(0.4)
                continue

            sent_ts = time.time()

            delay_sec = max(0.001, delay_ms / 1000.0)
            jitter_sec = random.uniform(0.0, 0.3 * delay_sec)
            time.sleep(delay_sec + jitter_sec)

            # Check for direct bedside vitals overrides set on the Jetson
            vitals_cfg = bed_chaos.get("vitals", {})
            target_hr = vitals_cfg.get("target_hr") or bed_chaos.get("Agent_HR", {}).get("target_hr")
            target_spo2 = vitals_cfg.get("target_spo2") or bed_chaos.get("Agent_SpO2", {}).get("target_spo2")
            target_co2 = vitals_cfg.get("target_co2") or bed_chaos.get("Agent_CO2", {}).get("target_co2")
            target_etco2 = vitals_cfg.get("target_etco2") or target_co2 or bed_chaos.get("Agent_EtCO2", {}).get("target_etco2")
            target_sys = vitals_cfg.get("target_sys") or bed_chaos.get("Agent_IBP", {}).get("target_sys") or bed_chaos.get("Agent_NIBP", {}).get("target_sys")
            target_dia = vitals_cfg.get("target_dia") or bed_chaos.get("Agent_IBP", {}).get("target_dia") or bed_chaos.get("Agent_NIBP", {}).get("target_dia")
            ecg_condition = vitals_cfg.get("ecg_condition") or bed_chaos.get("Agent_ECG", {}).get("condition", "normal")

            value = 0
            ecg_wave = []
            reading = {}
            waveforms = {}
            base_hr = int(target_hr) if target_hr is not None else 75 + self.hr_offset
            heart_rate = smooth_vital(
                base_hr, sent_ts, self.phase_offset, 2.0 if target_hr is not None else 5.0, 35, 210
            )
            base_respiration = float(vitals_cfg.get("target_respiration", 16))
            respiration_rate = smooth_vital(
                base_respiration, sent_ts, self.phase_offset + 0.7, 1.5, 8, 35
            )

            if self.agent_id == "Agent_ECG":
                cond = "malicious" if malicious else ecg_condition
                ecg_wave = generate_ecg_wave(
                    condition=cond,
                    phase_offset=self.phase_offset,
                    heart_rate=heart_rate,
                    start_time=sent_ts - 1.0,
                )
                value = 0
            elif self.agent_id == "Agent_SpO2":
                if malicious:
                    value = random.choice([0, 15, 145])
                else:
                    base_spo2 = int(target_spo2) if target_spo2 is not None else 98
                    value = smooth_vital(base_spo2, sent_ts, self.phase_offset + 0.3, 0.8, 70, 100)
                t = np.linspace(0, 1, 250, endpoint=False)
                phase = (t * heart_rate / 60.0 + (sent_ts - 1.0) * heart_rate / 60.0 + self.phase_offset) % 1.0
                pulse = np.maximum(0.0, np.sin(2 * np.pi * phase - 0.45)) ** 3
                waveforms["pleth_wave"] = np.round(0.72 * pulse - 0.12, 4).tolist()
            elif self.agent_id == "Agent_HR":
                if malicious:
                    value = random.choice([0, -40, 340])
                else:
                    value = heart_rate
            elif self.agent_id == "Agent_Respiration":
                value = respiration_rate
                if malicious:
                    value = random.choice([0, 55])
                reading = {"respiration_rate": value}
                t = np.linspace(0, 1, 250, endpoint=False)
                phase = (t * value / 60.0 + (sent_ts - 1.0) * value / 60.0 + self.phase_offset) % 1.0
                waveforms["respiration_wave"] = np.round(
                    0.7 * np.sin(2 * np.pi * phase), 4
                ).tolist()
            elif self.agent_id in {"Agent_EtCO2", "Agent_CO2"}:
                target = float(target_etco2 if target_etco2 is not None else 38)
                value = smooth_vital(target, sent_ts, self.phase_offset + 1.1, 1.5, 15, 60)
                if malicious:
                    value = random.choice([0, 75])
                reading = {"etco2_mmhg": value}
                phase = (
                    np.linspace(0, 1, 250, endpoint=False) * respiration_rate / 60.0
                    + (sent_ts - 1.0) * respiration_rate / 60.0
                    + self.phase_offset
                ) % 1.0
                capnogram = np.select(
                    [phase < 0.08, phase < 0.18, phase < 0.68, phase < 0.78],
                    [phase / 0.08, 1.0, 1.0, 1.0 - (phase - 0.68) / 0.10],
                    default=0.0,
                )
                waveforms["etco2_wave"] = np.round(capnogram * value, 2).tolist()
            elif self.agent_id == "Agent_IBP":
                systolic = smooth_vital(int(target_sys or 120), sent_ts, self.phase_offset, 3.0, 60, 220)
                diastolic = smooth_vital(int(target_dia or 80), sent_ts, self.phase_offset + 0.8, 2.0, 30, 130)
                if malicious:
                    systolic, diastolic = 0, 0
                mean = int(round((systolic + 2 * diastolic) / 3))
                value = {
                    "sys": systolic,
                    "dia": diastolic,
                    "map": mean,
                    "formatted": f"{systolic}/{diastolic} ({mean})",
                }
                reading = {"systolic": systolic, "diastolic": diastolic, "map": mean}
                phase = (
                    np.linspace(0, 1, 250, endpoint=False) * heart_rate / 60.0
                    + (sent_ts - 1.0) * heart_rate / 60.0
                    + self.phase_offset
                ) % 1.0
                pulse = np.select(
                    [phase < 0.12, phase < 0.35, phase < 0.62],
                    [phase / 0.12, 1.0 - (phase - 0.12) / 0.23 * 0.45, 0.55],
                    default=0.0,
                )
                waveforms["ibp_wave"] = np.round(diastolic + pulse * (systolic - diastolic), 2).tolist()
            elif self.agent_id == "Agent_NIBP":
                systolic = smooth_vital(int(target_sys or 120), sent_ts, self.phase_offset + 0.3, 2.0, 60, 220)
                diastolic = smooth_vital(int(target_dia or 80), sent_ts, self.phase_offset + 1.1, 1.5, 30, 130)
                if malicious:
                    systolic, diastolic = 290, 180
                value = int(round((systolic + 2 * diastolic) / 3))
                reading = {"systolic": systolic, "diastolic": diastolic, "map": value}
            payload = {
                "patient_id": self.patient_id,
                "agent_id": self.agent_id,
                "seq": self.seq,
                "sent_ts": sent_ts,
                "value": value,
                "ecg_wave": ecg_wave,
                "reading": reading,
                **waveforms,
            }

            for _ in range(2):
                if self.client:
                    try:
                        res = self.client.publish(topic, json.dumps(payload), qos=0)
                        if res.rc == mqtt.MQTT_ERR_SUCCESS:
                            break
                        self._failover_broker()
                    except Exception:
                        self._failover_broker()
                else:
                    self._failover_broker()

            elapsed = time.time() - loop_start
            time.sleep(max(0.05, 0.40 - elapsed))

        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Bedside Multi-Agent Medical Sensor Streamer")
    parser.add_argument(
        "--patient-id", type=str, default="Bed_1", help="Patient/Bed identifier for this Jetson"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional list of multiple patient IDs to simulate (e.g. Bed_1 Bed_2 Bed_3)",
    )
    parser.add_argument("--primary-broker", type=str, default="127.0.0.1", help="Primary MQTT broker IP")
    parser.add_argument("--backup-broker", type=str, default="127.0.0.1", help="Backup MQTT broker IP")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    args = parser.parse_args()

    patient_list = args.patients if args.patients else [args.patient_id]
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n[INFO] Stopping bedside sensor agents...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    agent_names = [
        "Agent_ECG",
        "Agent_SpO2",
        "Agent_HR",
        "Agent_Respiration",
        "Agent_EtCO2",
        "Agent_IBP",
        "Agent_NIBP",
    ]
    threads = []

    print("=" * 64)
    print("Bedside Multi-Agent Sensor Streamer Active")
    print(f"Patient Beds:   {patient_list}")
    print(f"Primary Broker: {args.primary_broker}:{args.port}")
    print(f"Backup Broker:  {args.backup_broker}:{args.port}")
    print("=" * 64)

    for pid in patient_list:
        for agent_id in agent_names:
            t = SensorAgentThread(
                pid, agent_id, args.primary_broker, args.backup_broker, args.port, stop_event
            )
            t.start()
            threads.append(t)

    while not stop_event.is_set():
        time.sleep(0.5)

    for t in threads:
        t.join(timeout=2.0)
    print("[INFO] All bedside sensor agent threads terminated cleanly.")


if __name__ == "__main__":
    main()
