"""Independent bedside sensor workers, started by a registered Jetson."""
import random
import threading
import time

import numpy as np
from patient_simulation import SAMPLE_RATE, WINDOW_SAMPLES, WAVEFORM_KEYS

PUBLISH_INTERVAL_SECONDS = 0.4


class SensorAgentThread(threading.Thread):
    def __init__(self, patient_id, agent_id, simulation, get_config, publish, stop_event):
        super().__init__(name=f"{patient_id}:{agent_id}", daemon=True)
        self.patient_id = patient_id
        self.agent_id = agent_id
        self.simulation = simulation
        self.get_config = get_config
        self.publish = publish
        self.stop_event = stop_event
        self.seq = 0

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
                value = random.choice([0, 95])
                base = frame.readings[agent] or 38.0
                wave = [round(sample * value / base, 4) for sample in wave]
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
        while not self.stop_event.is_set():
            started = time.monotonic()
            config = self.get_config()
            settings = config.get(self.agent_id, {})
            if not settings.get("removed"):
                self.seq += 1
                if random.random() >= float(settings.get("drop_rate", 0)):
                    frame = self.simulation.sample(config)
                    packet = self._packet(frame, config, bool(settings.get("malicious")), time.time())
                    delay = max(0, float(settings.get("delay_ms", 0))) / 1000
                    if self.stop_event.wait(delay * random.uniform(1, 1.3)):
                        break
                    self.publish(packet)
            self.stop_event.wait(max(0.05, PUBLISH_INTERVAL_SECONDS - (time.monotonic() - started)))
