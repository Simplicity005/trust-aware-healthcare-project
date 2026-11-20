"""Shared sensor trust evaluator used by bedside and nurse-station monitors."""

import time
from collections import deque

import numpy as np


class AgentTrustTracker:
    """Compute the project composite trust score for one streaming sensor."""

    def __init__(self, agent_id, window_size=25, w1=0.25, w2=0.20, w3=0.25, w4=0.30, alpha=0.35):
        self.agent_id = agent_id
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        self.alpha = alpha
        self.latencies = deque(maxlen=window_size)
        self.delivery_window = deque(maxlen=max(window_size, 30))
        self.validity_window = deque(maxlen=min(window_size, 15))
        self.last_seq = None
        self.last_rx_time = 0.0
        self.last_value = None
        self.ecg_wave = [0.0] * 250
        self.s_l = 1.0
        self.s_j = 1.0
        self.s_p = 1.0
        self.s_d = 1.0
        self.trust_score = 1.0
        self.status = "DISCONNECTED"
        self.packets_received = 0
        self.packets_missed = 0

    def check_data_plausibility(
        self, value=0, ecg_wave=None, cross_sensor_conflict=False, reading=None, waveform=None
    ):
        if self.agent_id == "Agent_ECG":
            if not ecg_wave or len(ecg_wave) != 250:
                return 0.0
            arr = np.array(ecg_wave, dtype=np.float32)
            max_abs = float(np.max(np.abs(arr)))
            std_val = float(np.std(arr))
            diff_std = float(np.std(np.diff(arr)))
            if max_abs > 2.2 or std_val > 1.0 or std_val < 0.03 or diff_std > 0.35:
                return 0.0
            return 0.10 if cross_sensor_conflict else 1.0
        if self.agent_id in {"Agent_SpO2", "Agent_Respiration", "Agent_EtCO2", "Agent_IBP"}:
            if not isinstance(waveform, list) or len(waveform) != 250:
                return 0.0
            arr = np.asarray(waveform, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                return 0.0
            peak = float(np.max(arr))
            # A one-second capnogram window can lie entirely within inspiration
            # or the expiratory plateau.
            flat_capnogram = self.agent_id == "Agent_EtCO2" and (peak < 0.01 or 15 <= peak <= 78)
            if float(np.std(arr)) < 0.01 and not flat_capnogram:
                return 0.0
            if self.agent_id == "Agent_SpO2":
                if float(np.min(arr)) < -0.5 or float(np.max(arr)) > 1.0:
                    return 0.0
            elif self.agent_id == "Agent_Respiration":
                if float(np.max(np.abs(arr))) > 1.5:
                    return 0.0
            elif self.agent_id == "Agent_EtCO2":
                if float(np.min(arr)) < 0 or float(np.max(arr)) > 80:
                    return 0.0
            elif float(np.min(arr)) < 20 or float(np.max(arr)) > 250:
                return 0.0

        own_valid = 1.0
        if self.agent_id == "Agent_SpO2":
            own_valid = 1.0 if 70 <= int(value or 0) <= 100 else 0.0
        elif self.agent_id == "Agent_HR":
            own_valid = 1.0 if 40 <= int(value or 0) <= 190 else 0.0
        elif self.agent_id == "Agent_Respiration":
            own_valid = 1.0 if 8 <= int(value or 0) <= 35 else 0.0
        elif self.agent_id == "Agent_EtCO2":
            own_valid = 1.0 if 15 <= int(value or 0) <= 75 else 0.0
        elif self.agent_id == "Agent_Temperature":
            own_valid = 1.0 if 20 <= float(value or 0) <= 50 else 0.0
        elif self.agent_id in {"Agent_IBP", "Agent_NIBP"}:
            reading = reading if isinstance(reading, dict) else {}
            try:
                systolic = float(reading["systolic"])
                diastolic = float(reading["diastolic"])
                mean = float(reading["map"])
            except (KeyError, TypeError, ValueError):
                return 0.0
            valid_pressure = 40 <= diastolic < systolic <= 250
            valid_mean = diastolic <= mean <= systolic
            own_valid = 1.0 if valid_pressure and valid_mean else 0.0

        if own_valid == 0.0:
            return 0.0
        return 0.10 if cross_sensor_conflict else own_valid

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
            reading = payload.get("reading", {})
            waveform_key = {
                "Agent_SpO2": "pleth_wave",
                "Agent_Respiration": "respiration_wave",
                "Agent_EtCO2": "etco2_wave",
                "Agent_IBP": "ibp_wave",
            }.get(self.agent_id)
            waveform = payload.get(waveform_key, []) if waveform_key else []
        else:
            seq = int(seq or 0)
            sent_ts = float(sent_ts if sent_ts is not None else rx_time)
            value = value if value is not None else 0
            ecg_wave = ecg_wave or []
            reading = {}
            waveform = []

        self.packets_received += 1
        self.latencies.append(max(0.5, (rx_time - sent_ts) * 1000.0))
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

        data_valid = self.check_data_plausibility(
            value, ecg_wave, cross_sensor_conflict, reading, waveform
        )
        self.validity_window.append(data_valid)
        avg_latency = float(np.mean(self.latencies))
        self.s_l = float(np.clip(np.exp(-max(0.0, avg_latency - 25.0) / 160.0), 0.0, 1.0))
        jitter_ms = float(np.std(self.latencies)) if len(self.latencies) >= 2 else 0.0
        self.s_j = float(np.clip(np.exp(-jitter_ms / 60.0), 0.0, 1.0))
        self.s_p = float(np.mean(self.delivery_window)) if self.delivery_window else 1.0
        ewma_valid = float(np.mean(self.validity_window)) if self.validity_window else 1.0
        self.s_d = 0.25 * ewma_valid if data_valid == 0.0 else ewma_valid

        raw_trust = self.w1 * self.s_l + self.w2 * self.s_j + self.w3 * self.s_p + self.w4 * self.s_d
        if data_valid == 0.0:
            raw_trust = min(raw_trust, 0.32)
        if self.packets_received == 1:
            self.trust_score = float(np.clip(raw_trust, 0.0, 1.0))
        else:
            self.trust_score = float(
                np.clip(self.alpha * raw_trust + (1.0 - self.alpha) * self.trust_score, 0.0, 1.0)
            )
        self.evaluate_status()
        return self.trust_score

    def evaluate_status(self, orin_link_congested=False):
        if self.last_rx_time == 0.0:
            self.trust_score = 0.0
            self.status = "DISCONNECTED"
            return self.status

        silence_sec = time.time() - self.last_rx_time
        if silence_sec > 2.0:
            penalty = float(np.clip(np.exp(-(silence_sec - 2.0) / 3.0), 0.0, 1.0))
            self.s_p = min(self.s_p, penalty)
            self.s_d = min(self.s_d, penalty)
            raw_trust = self.w1 * self.s_l + self.w2 * self.s_j + self.w3 * self.s_p + self.w4 * self.s_d
            self.trust_score = float(np.clip(min(self.trust_score, raw_trust), 0.0, 1.0))
            self.status = "DISCONNECTED"
            return self.status

        if self.trust_score < 0.45:
            self.status = "ISOLATED"
        elif orin_link_congested:
            self.status = "OFFLOADED"
        elif self.trust_score < 0.70:
            self.status = "SUSPICIOUS"
        else:
            self.status = "TRUSTED"
        return self.status
