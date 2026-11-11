#!/usr/bin/env python3
"""
Bedside Autonomous Multi-Agent Sensor Streamer (Runs on NVIDIA Jetson Orin)
Simulates Agent_ECG, Agent_SpO2, and Agent_HR for one or more patient beds,
streaming vitals over MQTT with automatic broker failover and local bedside fault configuration.
"""

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
    return {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False},
    }


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


def generate_ecg_wave(condition="normal", phase_offset=0.0):
    """Generates a 1-second 250 Hz ECG waveform based on bedside clinical condition."""
    t = np.linspace(0, 1, 250)
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

    # Normal sinus rhythm (PQRST waveform) with slight patient-specific offset
    c = phase_offset
    p_wave = 0.15 * np.exp(-((t - (0.20 + c)) ** 2) / (2 * 0.025 ** 2))
    q_wave = -0.15 * np.exp(-((t - (0.35 + c)) ** 2) / (2 * 0.010 ** 2))
    r_wave = 1.25 * np.exp(-((t - (0.40 + c)) ** 2) / (2 * 0.012 ** 2))
    s_wave = -0.25 * np.exp(-((t - (0.45 + c)) ** 2) / (2 * 0.012 ** 2))
    t_wave = 0.30 * np.exp(-((t - (0.65 + c)) ** 2) / (2 * 0.040 ** 2))
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
            ecg_condition = vitals_cfg.get("ecg_condition") or bed_chaos.get("Agent_ECG", {}).get("condition", "normal")

            value = 0
            ecg_wave = []

            if self.agent_id == "Agent_ECG":
                cond = "malicious" if malicious else ecg_condition
                ecg_wave = generate_ecg_wave(condition=cond, phase_offset=self.phase_offset)
                value = 0
            elif self.agent_id == "Agent_SpO2":
                if malicious:
                    value = random.choice([0, 15, 145])
                elif target_spo2 is not None:
                    value = int(np.clip(int(target_spo2) + random.randint(0, 1), 40, 100))
                else:
                    value = random.randint(96, 100)
            elif self.agent_id == "Agent_HR":
                if malicious:
                    value = random.choice([0, -40, 340])
                elif target_hr is not None:
                    value = int(np.clip(int(target_hr) + random.randint(-1, 1), 30, 240))
                else:
                    value = random.randint(68, 88) + self.hr_offset

            payload = {
                "patient_id": self.patient_id,
                "agent_id": self.agent_id,
                "seq": self.seq,
                "sent_ts": sent_ts,
                "value": value,
                "ecg_wave": ecg_wave,
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

    agent_names = ["Agent_ECG", "Agent_SpO2", "Agent_HR"]
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
