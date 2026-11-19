"""Continuous synthetic vital signs shared by a bed's independent sensor workers."""

from dataclasses import dataclass
import threading
import time

import numpy as np

SAMPLE_RATE = 250
WINDOW_SAMPLES = SAMPLE_RATE
SETTLE_SECONDS = 1.5
CUFF_INTERVAL_SECONDS = 15
SENSOR_AGENT_IDS = (
    "Agent_ECG", "Agent_SpO2", "Agent_HR", "Agent_Respiration",
    "Agent_EtCO2", "Agent_IBP", "Agent_NIBP", "Agent_Temperature",
)
WAVEFORM_KEYS = {
    "Agent_ECG": "ecg_wave",
    "Agent_SpO2": "pleth_wave",
    "Agent_Respiration": "respiration_wave",
    "Agent_EtCO2": "etco2_wave",
    "Agent_IBP": "ibp_wave",
}


@dataclass(frozen=True)
class PatientFrame:
    end_ts: float
    readings: dict
    waveforms: dict


def shaped_cycle(phase, knots, levels):
    """Interpolate the landmarks of a periodic pulse with smooth transitions."""
    knots = np.asarray(knots)
    levels = np.asarray(levels)
    left = np.searchsorted(knots, phase, side="right") - 1
    blend = (phase - knots[left]) / (knots[left + 1] - knots[left])
    blend = blend * blend * (3 - 2 * blend)
    return levels[left] + (levels[left + 1] - levels[left]) * blend


def ecg_trace(phase, seconds, arrhythmia):
    def wave(center, width, amplitude):
        distance = (phase - center + 0.5) % 1.0 - 0.5
        return amplitude * np.exp(-0.5 * (distance / width) ** 2)

    if arrhythmia:
        return wave(0.29, 0.083, 1.75) + wave(0.52, 0.10, -1.18) + wave(0.73, 0.11, 0.24)
    baseline = 0.025 * np.sin(2 * np.pi * seconds / 4.7) + 0.006 * np.sin(2 * np.pi * seconds * 29)
    return (
        baseline + wave(0.15, 0.026, 0.13)
        + wave(0.265, 0.010, -0.15) + wave(0.30, 0.013, 1.17) + wave(0.334, 0.013, -0.24)
        + wave(0.58, 0.057, 0.27)
    )


def pressure_reading(systolic, diastolic):
    sys = round(float(systolic))
    dia = round(float(diastolic))
    return {"systolic": sys, "diastolic": dia, "map": round((sys + 2 * dia) / 3)}


class PatientSimulation:
    def __init__(self, patient_id):
        self.lock = threading.Lock()
        self.started_at = time.monotonic()
        self.started_at_unix = time.time()
        patient_number = sum((index + 1) * ord(ch) for index, ch in enumerate(patient_id))
        self.heart_phase = (patient_number % 37) / 37
        self.breath_phase = (patient_number % 29) / 29
        self.offset = self.heart_phase * 2 * np.pi
        self.last_index = -WINDOW_SAMPLES
        self.levels = None
        self.frame = None
        self.last_cuff_index = -CUFF_INTERVAL_SECONDS * SAMPLE_RATE
        self.cuff = None
        self.windows = {key: np.empty(0) for key in WAVEFORM_KEYS}

    def sample(self, config):
        with self.lock:
            last = int((time.monotonic() - self.started_at) * SAMPLE_RATE)
            if last <= self.last_index:
                return self.frame
            return self._advance(last, config)

    def _advance(self, last, config):
        vitals = config.get("vitals", {})
        hr = config.get("Agent_HR", {})
        spo2 = config.get("Agent_SpO2", {})
        co2 = config.get("Agent_EtCO2", {})
        legacy_co2 = config.get("Agent_CO2", {})
        ibp = config.get("Agent_IBP", {})
        cuff = config.get("Agent_NIBP", {})
        ecg = config.get("Agent_ECG", {})
        targets = np.clip(np.asarray([
            vitals.get("target_hr") or hr.get("target_hr", 75),
            vitals.get("target_respiration") or 16,
            vitals.get("target_spo2") or spo2.get("target_spo2", 98),
            vitals.get("target_etco2") or vitals.get("target_co2")
            or co2.get("target_etco2") or legacy_co2.get("target_co2", 38),
            vitals.get("target_sys") or ibp.get("target_sys") or cuff.get("target_sys", 120),
            vitals.get("target_dia") or ibp.get("target_dia") or cuff.get("target_dia", 80),
        ], dtype=float), [35, 8, 70, 15, 60, 30], [210, 35, 100, 75, 220, 130])
        previous = targets if self.levels is None else self.levels

        # Only the latest second is sent to consumers. The integrated phase still
        # advances through any elapsed time when a worker resumes after a delay.
        indices = np.arange(max(self.last_index + 1, last - WINDOW_SAMPLES + 1), last + 1)
        seconds = indices / SAMPLE_RATE
        elapsed = (indices - self.last_index) / SAMPLE_RATE
        remaining = np.exp(-elapsed / SETTLE_SECONDS)
        levels = targets[:, None] + (previous - targets)[:, None] * remaining
        integrated = targets[:2, None] * elapsed + (
            (previous - targets)[:2, None] * SETTLE_SECONDS * (1 - remaining)
        )
        previous_time = self.last_index / SAMPLE_RATE
        heart_frequency = 2 * np.pi / 12
        breath_frequency = 2 * np.pi / 19
        heart = self.heart_phase + (
            integrated[0] + 1.8 * (
                np.cos(previous_time * heart_frequency + self.offset)
                - np.cos(seconds * heart_frequency + self.offset)
            ) / heart_frequency
        ) / 60
        breath = self.breath_phase + (
            integrated[1] + 0.7 * (
                np.cos(previous_time * breath_frequency + self.offset)
                - np.cos(seconds * breath_frequency + self.offset)
            ) / breath_frequency
        ) / 60
        heart_rate = levels[0] + 1.8 * np.sin(seconds * heart_frequency + self.offset)
        respiratory_rate = levels[1] + 0.7 * np.sin(seconds * breath_frequency + self.offset)
        oxygen = np.clip(levels[2] + 0.45 * np.sin(seconds * 0.44 + self.offset), 70, 100)
        carbon_dioxide = levels[3] + 0.65 * np.sin(seconds * 0.31 + self.offset)
        systolic = levels[4] + 2.0 * np.sin(2 * np.pi * breath)
        diastolic = np.minimum(levels[5] + 1.0 * np.sin(2 * np.pi * breath), systolic - 10.0)

        breath_phase = breath % 1.0
        arterial = shaped_cycle((heart - 0.30 - 0.105 * heart_rate / 60) % 1.0,
                                (0, 0.10, 0.34, 0.39, 0.46, 1), (0, 1, 0.53, 0.42, 0.50, 0))
        pleth = shaped_cycle((heart - 0.30 - 0.185 * heart_rate / 60) % 1.0,
                             (0, 0.14, 0.36, 0.44, 0.55, 1), (0, 1, 0.56, 0.36, 0.42, 0))
        respiration = shaped_cycle(breath_phase, (0, 0.37, 1), (-0.46, 0.60, -0.46))
        capnogram = shaped_cycle(breath_phase, (0, 0.38, 0.49, 0.87, 0.94, 1), (0, 0, 0.80, 1, 0, 0))
        condition = vitals.get("ecg_condition") or ecg.get("condition", "normal")
        waves = {
            "Agent_ECG": ecg_trace(heart % 1.0, seconds, condition == "arrhythmia"),
            "Agent_SpO2": -0.10 + pleth * (0.64 + 0.035 * np.sin(2 * np.pi * breath)),
            "Agent_Respiration": respiration,
            "Agent_EtCO2": np.clip(carbon_dioxide, 15, 75) * capnogram,
            "Agent_IBP": diastolic + (systolic - diastolic) * arterial,
        }
        waveforms = {}
        for agent, wave in waves.items():
            window = np.concatenate((self.windows[agent], wave))[-WINDOW_SAMPLES:]
            self.windows[agent] = window
            waveforms[agent] = np.round(window, 4).tolist()

        if last - self.last_cuff_index >= CUFF_INTERVAL_SECONDS * SAMPLE_RATE:
            self.cuff = pressure_reading(systolic[-1] + 2, diastolic[-1] - 1)
            self.last_cuff_index = last
        self.last_index = last
        self.heart_phase = heart[-1] % 1.0
        self.breath_phase = breath[-1] % 1.0
        self.levels = levels[:, -1]
        self.frame = PatientFrame(self.started_at_unix + last / SAMPLE_RATE, {
            "Agent_HR": round(float(np.clip(heart_rate[-1], 35, 210))),
            "Agent_Respiration": round(float(np.clip(respiratory_rate[-1], 8, 35))),
            "Agent_SpO2": round(float(oxygen[-1])),
            "Agent_EtCO2": round(float(np.clip(carbon_dioxide[-1], 15, 75))),
            "Agent_IBP": pressure_reading(systolic[-1], diastolic[-1]),
            "Agent_NIBP": self.cuff,
            "Agent_Temperature": round(36.7 + 0.18 * np.sin(seconds[-1] * 0.04 + self.offset), 1),
        }, waveforms)
        return self.frame
