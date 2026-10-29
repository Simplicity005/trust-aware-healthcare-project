#!/usr/bin/env python3
"""
Autonomous Multi-Agent Sensor Streamer with MQTT Broker Failover & Chaos Injection
Simulates Agent_ECG, Agent_SpO2, and Agent_HR streaming vitals over MQTT.
"""

import os
import sys
import time
import json
import random
import signal
import argparse
import threading
import numpy as np
import paho.mqtt.client as mqtt

CHAOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chaos.json")

def read_chaos_config():
    """Reads chaos.json safely with fallback to default settings."""
    default_config = {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False}
    }
    if not os.path.exists(CHAOS_FILE):
        return default_config
    try:
        with open(CHAOS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return default_config

def generate_ecg_wave(malicious=False):
    """Generates a 1-second 250 Hz ECG waveform."""
    t = np.linspace(0, 1, 250)
    if malicious:
        # High-amplitude erratic chaotic noise
        wave = np.random.uniform(-2.5, 2.5, 250) + np.sin(2 * np.pi * 35 * t) * 1.5
        return np.round(np.clip(wave, -4.0, 4.0), 4).tolist()
    
    # Normal sinus rhythm (PQRST waveform)
    p_wave = 0.15 * np.exp(-((t - 0.20) ** 2) / (2 * 0.025 ** 2))
    q_wave = -0.15 * np.exp(-((t - 0.35) ** 2) / (2 * 0.010 ** 2))
    r_wave = 1.25 * np.exp(-((t - 0.40) ** 2) / (2 * 0.012 ** 2))
    s_wave = -0.25 * np.exp(-((t - 0.45) ** 2) / (2 * 0.012 ** 2))
    t_wave = 0.30 * np.exp(-((t - 0.65) ** 2) / (2 * 0.040 ** 2))
    noise = np.random.normal(0, 0.02, 250)
    ecg = p_wave + q_wave + r_wave + s_wave + t_wave + noise
    return np.round(ecg, 4).tolist()

class SensorAgentThread(threading.Thread):
    def __init__(self, agent_id, primary_broker, backup_broker, port, stop_event):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.primary_broker = primary_broker
        self.backup_broker = backup_broker
        self.port = port
        self.stop_event = stop_event
        self.seq = 0
        self.current_broker = primary_broker
        self.client = None

    def _init_client(self):
        try:
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"{self.agent_id}_{os.getpid()}_{random.randint(1000, 9999)}"
                )
            else:
                self.client = mqtt.Client(
                    client_id=f"{self.agent_id}_{os.getpid()}_{random.randint(1000, 9999)}"
                )
            self.client.connect(self.current_broker, self.port, keepalive=60)
            self.client.loop_start()
            print(f"[{self.agent_id}] Connected to broker {self.current_broker}:{self.port}")
            return True
        except Exception as e:
            print(f"[{self.agent_id}] Failed to connect to {self.current_broker}: {e}")
            return False

    def _failover_broker(self):
        """Switches broker between primary and backup on failure."""
        next_broker = (
            self.backup_broker if self.current_broker == self.primary_broker else self.primary_broker
        )
        print(f"[{self.agent_id}] Initiating broker failover: {self.current_broker} -> {next_broker}")
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

        topic = f"hospital/sensors/{self.agent_id}"

        while not self.stop_event.is_set():
            loop_start = time.time()
            self.seq += 1

            chaos = read_chaos_config()
            agent_chaos = chaos.get(self.agent_id, {"delay_ms": 10, "drop_rate": 0.0, "malicious": False})

            drop_rate = float(agent_chaos.get("drop_rate", 0.0))
            delay_ms = float(agent_chaos.get("delay_ms", 10.0))
            malicious = bool(agent_chaos.get("malicious", False))

            # Simulated packet drop: increment seq without publishing
            if random.random() < drop_rate:
                # Sequence gap occurs
                time.sleep(0.4)
                continue

            # Simulated network latency and jitter (0 to 30% of delay)
            delay_sec = max(0.001, delay_ms / 1000.0)
            jitter_sec = random.uniform(0.0, 0.3 * delay_sec)
            time.sleep(delay_sec + jitter_sec)

            # Generate sensor payload
            value = 0
            ecg_wave = []

            if self.agent_id == "Agent_ECG":
                ecg_wave = generate_ecg_wave(malicious=malicious)
                value = 0
            elif self.agent_id == "Agent_SpO2":
                if malicious:
                    value = random.choice([0, 15, 145])
                else:
                    value = random.randint(96, 100)
            elif self.agent_id == "Agent_HR":
                if malicious:
                    value = random.choice([0, -40, 340])
                else:
                    value = random.randint(65, 92)

            payload = {
                "agent_id": self.agent_id,
                "seq": self.seq,
                "sent_ts": time.time(),
                "value": value,
                "ecg_wave": ecg_wave
            }

            # Publish with failover handling
            published = False
            for _ in range(2):
                if self.client:
                    try:
                        res = self.client.publish(topic, json.dumps(payload), qos=0)
                        if res.rc == mqtt.MQTT_ERR_SUCCESS:
                            published = True
                            break
                        else:
                            print(f"[{self.agent_id}] Publish failed with code {res.rc}, triggering failover...")
                            self._failover_broker()
                    except Exception as e:
                        print(f"[{self.agent_id}] Socket error: {e}, triggering failover...")
                        self._failover_broker()
                else:
                    self._failover_broker()

            # Maintain ~0.4s interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.05, 0.40 - elapsed)
            time.sleep(sleep_time)

        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        print(f"[{self.agent_id}] Agent thread stopped.")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Medical Sensor Streamer")
    parser.add_argument("--primary-broker", type=str, default="127.0.0.1", help="Primary MQTT broker address")
    parser.add_argument("--backup-broker", type=str, default="127.0.0.1", help="Backup MQTT broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    args = parser.parse_args()

    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n[INFO] Stopping sensor agents...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    agent_names = ["Agent_ECG", "Agent_SpO2", "Agent_HR"]
    threads = []

    print("=" * 60)
    print("Multi-Agent Sensor Streamer Active")
    print(f"Primary Broker: {args.primary_broker}:{args.port}")
    print(f"Backup Broker:  {args.backup_broker}:{args.port}")
    print(f"Active Agents:  {agent_names}")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    for agent_id in agent_names:
        thread = SensorAgentThread(agent_id, args.primary_broker, args.backup_broker, args.port, stop_event)
        thread.start()
        threads.append(thread)

    while not stop_event.is_set():
        time.sleep(0.5)

    for thread in threads:
        thread.join(timeout=2.0)
    print("[INFO] All sensor agent threads terminated cleanly.")

if __name__ == "__main__":
    main()
