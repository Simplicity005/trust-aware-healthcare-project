#!/usr/bin/env python3
"""Independent MQTT workers for the bedside sensors."""

import os
import time
import json
import random
import signal
import argparse
import threading
import numpy as np
import paho.mqtt.client as mqtt

from patient_simulation import PatientSimulation, SAMPLE_RATE, SENSOR_AGENT_IDS, WINDOW_SAMPLES, WAVEFORM_KEYS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAOS_FILE = os.path.join(BASE_DIR, "chaos.json")
CHAOS_LOCK = threading.Lock()
PUBLISH_INTERVAL_SECONDS = 0.4


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
        "Agent_Temperature": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
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


class SensorAgentThread(threading.Thread):
    def __init__(self, patient_id, agent_id, simulation, primary_broker, backup_broker, port, stop_event):
        super().__init__(name=f"{patient_id}:{agent_id}", daemon=True)
        self.patient_id = patient_id
        self.agent_id = agent_id
        self.simulation = simulation
        self.primary_broker = primary_broker
        self.backup_broker = backup_broker
        self.port = port
        self.stop_event = stop_event
        self.seq = 0
        self.current_broker = primary_broker
        self.client = None

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

    def _packet(self, frame, config, malicious, sent_ts):
        agent = self.agent_id
        value = frame.readings.get(agent, 0)
        wave = frame.waveforms.get(agent)
        reading = {}

        if agent == "Agent_ECG":
            condition = config.get("vitals", {}).get("ecg_condition") or config.get("Agent_ECG", {}).get("condition")
            if malicious or condition == "malicious":
                times = frame.end_ts + np.arange(1 - WINDOW_SAMPLES, 1) / SAMPLE_RATE
                noise = 2.1 * np.sin(2 * np.pi * 27 * times) + 1.65 * np.sin(2 * np.pi * 43 * times)
                wave = np.round(noise, 4).tolist()
        elif agent == "Agent_HR" and malicious:
            value = random.choice([0, -40, 340])
        elif agent == "Agent_SpO2" and malicious:
            value = random.choice([0, 15, 145])
        elif agent == "Agent_Respiration":
            if malicious:
                value = random.choice([0, 55])
            reading = {"respiration_rate": value}
        elif agent == "Agent_EtCO2":
            if malicious:
                value = random.choice([0, 75])
                wave = [sample * value / frame.readings[agent] for sample in wave]
            reading = {"etco2_mmhg": value}
        elif agent == "Agent_Temperature":
            if malicious:
                value = random.choice([0, 65])
            reading = {"temperature_c": value}
        elif agent in {"Agent_IBP", "Agent_NIBP"}:
            reading = value
            if malicious:
                reading = (
                    {"systolic": 0, "diastolic": 0, "map": 0} if agent == "Agent_IBP"
                    else {"systolic": 290, "diastolic": 180, "map": 217}
                )
            if agent == "Agent_IBP":
                sys, dia, mean = reading["systolic"], reading["diastolic"], reading["map"]
                value = {"sys": sys, "dia": dia, "map": mean, "formatted": f"{sys}/{dia} ({mean})"}
                if malicious:
                    wave = [0] * WINDOW_SAMPLES
            else:
                value = reading["map"]

        packet = {
            "patient_id": self.patient_id,
            "agent_id": agent,
            "seq": self.seq,
            "sent_ts": sent_ts,
            "value": value,
            "reading": reading,
            "ecg_wave": [],
        }
        if wave is not None:
            packet[WAVEFORM_KEYS[agent]] = wave
            packet["sample_rate_hz"] = SAMPLE_RATE
            packet["waveform_end_ts"] = frame.end_ts
        return packet

    def run(self):
        if not self._init_client():
            self._failover_broker()

        topic = f"hospital/{self.patient_id}/sensors/{self.agent_id}"
        try:
            while not self.stop_event.is_set():
                loop_start = time.monotonic()
                self.seq += 1
                config = read_patient_chaos(self.patient_id)
                settings = config.get(self.agent_id, {})
                if self.agent_id == "Agent_EtCO2":
                    settings = {**config.get("Agent_CO2", {}), **settings}

                if bool(settings.get("removed", False)):
                    self.stop_event.wait(PUBLISH_INTERVAL_SECONDS)
                    continue

                if random.random() >= float(settings.get("drop_rate", 0.0)):
                    frame = self.simulation.sample(config)
                    sent_ts = time.time()
                    payload = self._packet(frame, config, bool(settings.get("malicious", False)), sent_ts)
                    delay_sec = max(0.001, float(settings.get("delay_ms", 10.0)) / 1000.0)
                    if self.stop_event.wait(delay_sec * random.uniform(1.0, 1.3)):
                        break
                    message = json.dumps(payload)
                    for _ in range(2):
                        try:
                            if self.client and self.client.publish(topic, message, qos=0).rc == mqtt.MQTT_ERR_SUCCESS:
                                break
                        except (OSError, RuntimeError):
                            pass
                        self._failover_broker()

                self.stop_event.wait(max(0.05, PUBLISH_INTERVAL_SECONDS - (time.monotonic() - loop_start)))
        finally:
            if self.client:
                self.client.disconnect()
                self.client.loop_stop()


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

    threads = []

    print("=" * 64)
    print("Bedside Multi-Agent Sensor Streamer Active")
    print(f"Patient Beds:   {patient_list}")
    print(f"Primary Broker: {args.primary_broker}:{args.port}")
    print(f"Backup Broker:  {args.backup_broker}:{args.port}")
    print("=" * 64)

    for pid in patient_list:
        simulation = PatientSimulation(pid)
        for agent_id in SENSOR_AGENT_IDS:
            t = SensorAgentThread(
                pid, agent_id, simulation, args.primary_broker, args.backup_broker, args.port, stop_event
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
