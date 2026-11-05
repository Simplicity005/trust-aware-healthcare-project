#!/usr/bin/env python3
"""
Bedside GPU Hero Agent & Local ICU Monitor (Runs on NVIDIA Jetson Orin)
Executes <5ms 1D-CNN ECG anomaly inference, triggers local bedside alarms,
validates cross-sensor physiological plausibility, and streams bedside summaries
for one or more patient beds (--patient-id Bed_1 or --patients Bed_1 Bed_2 Bed_3).
"""

import os
import time
import json
import signal
import random
import argparse
import threading
import numpy as np
import paho.mqtt.client as mqtt

try:
    import onnxruntime as ort
except ImportError:
    ort = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAOS_FILE = os.path.join(BASE_DIR, "chaos.json")
DEFAULT_MODEL = os.path.join(BASE_DIR, "ecg_anomaly.onnx")
CHAOS_LOCK = threading.Lock()


def read_patient_chaos(patient_id="Bed_1"):
    """Reads chaos settings for a specific patient bed safely."""
    default_cfg = {"Orin_Link": {"overload_gpu": False, "use_relay_b": False}}
    if not os.path.exists(CHAOS_FILE):
        return default_cfg
    try:
        with CHAOS_LOCK:
            with open(CHAOS_FILE, "r") as f:
                data = json.load(f)
        if patient_id in data and isinstance(data[patient_id], dict):
            return data[patient_id]
        if "Orin_Link" in data:
            return data
        return default_cfg
    except Exception:
        return default_cfg


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
            tmp_path = CHAOS_FILE + f".tmp.brain.{random.randint(1000, 9999)}"
            with open(tmp_path, "w") as f:
                json.dump(current, f, indent=2)
            os.replace(tmp_path, CHAOS_FILE)
    except Exception:
        pass


class BedsideEdgeBrain:
    """Manages bedside AI inference and local alarm state for one patient bed."""

    def __init__(self, patient_id, broker="127.0.0.1", port=1883, session=None, active_provider="CPUExecutionProvider", stop_event=None):
        self.patient_id = patient_id
        self.broker = broker
        self.port = port
        self.session = session
        self.active_provider = active_provider
        self.stop_event = stop_event or threading.Event()
        self.lock = threading.Lock()

        # Telemetry state for this bed
        self.latest_hr = 75
        self.latest_spo2 = 98
        self.latest_ecg_wave = []
        self.latest_ecg_var = 0.04
        self.inf_ms = 0.0
        self.arrhythmia_prob = 0.0
        self.local_bed_alarm = False
        self.sensor_conflict = False
        self.force_offload = False
        self.client = None

        self._init_mqtt()

    def _init_mqtt(self):
        client_id = f"Jetson_Brain_{self.patient_id}_{os.getpid()}_{random.randint(1000, 9999)}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        else:
            self.client = mqtt.Client(client_id=client_id)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[EDGE BRAIN:{self.patient_id}] Connected to MQTT broker {self.broker}:{self.port} (rc={rc})")
        self.client.subscribe(f"hospital/{self.patient_id}/sensors/#")
        self.client.subscribe(f"hospital/{self.patient_id}/control/offload")
        self.client.subscribe(f"hospital/{self.patient_id}/control/chaos")
        self.client.subscribe("hospital/control/chaos")

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode("utf-8"))

            if topic == f"hospital/{self.patient_id}/control/offload":
                with self.lock:
                    self.force_offload = bool(payload.get("offload", False))
                return

            if topic == f"hospital/{self.patient_id}/control/chaos":
                update_chaos_file(payload, patient_id=self.patient_id)
                return

            if topic == "hospital/control/chaos":
                update_chaos_file(payload)
                return

            if topic.startswith(f"hospital/{self.patient_id}/sensors/"):
                agent_id = payload.get("agent_id")
                if agent_id == "Agent_ECG":
                    self._process_ecg(payload)
                elif agent_id == "Agent_SpO2":
                    with self.lock:
                        self.latest_spo2 = payload.get("value", 98)
                    self._check_plausibility()
                elif agent_id == "Agent_HR":
                    with self.lock:
                        self.latest_hr = payload.get("value", 75)
                    self._check_plausibility()
        except Exception as e:
            print(f"[EDGE BRAIN:{self.patient_id}] Error processing {msg.topic}: {e}")

    def _process_ecg(self, payload):
        ecg_wave = payload.get("ecg_wave", [])
        if not ecg_wave or len(ecg_wave) != 250:
            return

        wave_np = np.array(ecg_wave, dtype=np.float32).reshape(1, 1, 250)
        variance = float(np.var(wave_np))

        t0 = time.perf_counter()
        if self.session is not None:
            outputs = self.session.run(None, {"ecg_input": wave_np})
            logits = outputs[0]
        else:
            # Lightweight 1D-CNN NumPy inference fallback for older JetPack Python 3.8
            kernel = np.array([0.1, 0.25, -0.5, 1.0, -0.5, 0.25, 0.1], dtype=np.float32)
            conv_feat = np.convolve(wave_np.flatten(), kernel, mode="same")
            energy = float(np.mean(np.maximum(0.0, conv_feat)))
            logits = np.array([[1.2 - energy * 2.5, energy * 2.5 - 0.8]], dtype=np.float32)
        t1 = time.perf_counter()
        inf_ms = (t1 - t0) * 1000.0

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        arrhythmia_prob = float(probs[0, 1])

        if variance > 0.35 or np.max(np.abs(wave_np)) > 2.5:
            arrhythmia_prob = max(arrhythmia_prob, 0.96)

        is_alarm = arrhythmia_prob > 0.70

        with self.lock:
            self.latest_ecg_wave = ecg_wave
            self.latest_ecg_var = variance
            self.inf_ms = inf_ms
            self.arrhythmia_prob = arrhythmia_prob
            self.local_bed_alarm = is_alarm

        if is_alarm:
            self._trigger_bedside_alert(arrhythmia_prob, inf_ms, variance)

        self._check_plausibility()

    def _check_plausibility(self):
        with self.lock:
            hr = self.latest_hr
            spo2 = self.latest_spo2
            arrhythmia = self.arrhythmia_prob > 0.70

        conflict = False
        if not arrhythmia and spo2 >= 95 and (hr <= 0 or hr >= 240):
            conflict = True
        if spo2 < 50 or spo2 > 105 or hr < 30 or hr > 280:
            conflict = True
        if spo2 <= 30 and hr > 60 and not arrhythmia:
            conflict = True

        with self.lock:
            self.sensor_conflict = conflict

    def _trigger_bedside_alert(self, prob, inf_ms, variance):
        chaos = read_patient_chaos(self.patient_id)
        orin_cfg = chaos.get("Orin_Link", {})
        use_relay_b = (
            bool(orin_cfg.get("use_relay_b", False))
            or bool(orin_cfg.get("overload_gpu", False))
            or self.force_offload
        )

        sub_route = "relay_b" if use_relay_b else "primary"
        alert_topic = f"hospital/{self.patient_id}/alerts/{sub_route}"

        alert_payload = {
            "patient_id": self.patient_id,
            "alert": "CRITICAL_ARRHYTHMIA_DETECTED",
            "arrhythmia_prob": round(prob, 4),
            "inf_ms": round(inf_ms, 2),
            "ecg_variance": round(variance, 4),
            "timestamp": time.time(),
            "route": sub_route,
        }

        try:
            self.client.publish(alert_topic, json.dumps(alert_payload), qos=1)
            print(
                f"[EDGE ALARM:{self.patient_id}] Bedside Alarm ({prob*100:.1f}%) via {alert_topic} in {inf_ms:.2f}ms"
            )
        except Exception as e:
            print(f"[EDGE BRAIN:{self.patient_id}] Failed to dispatch alert: {e}")

    def run_summary_publisher(self):
        summary_topic = f"hospital/{self.patient_id}/edge/summary"

        while not self.stop_event.is_set():
            time.sleep(0.40)
            chaos = read_patient_chaos(self.patient_id)
            orin_cfg = chaos.get("Orin_Link", {})
            overload_gpu = bool(orin_cfg.get("overload_gpu", False))
            use_relay_b = (
                bool(orin_cfg.get("use_relay_b", False))
                or overload_gpu
                or self.force_offload
            )
            routing_path = "relay_b" if use_relay_b else "primary"

            with self.lock:
                summary = {
                    "patient_id": self.patient_id,
                    "timestamp": time.time(),
                    "inf_ms": round(self.inf_ms, 2),
                    "exec_provider": self.active_provider,
                    "arrhythmia_prob": round(self.arrhythmia_prob, 4),
                    "local_bed_alarm": self.local_bed_alarm,
                    "sensor_conflict": self.sensor_conflict,
                    "overload_gpu": overload_gpu,
                    "latest_vitals": {
                        "hr": self.latest_hr,
                        "spo2": self.latest_spo2,
                        "ecg_variance": round(self.latest_ecg_var, 4),
                    },
                    "ecg_wave": self.latest_ecg_wave,
                    "routing_path": routing_path,
                }

            try:
                self.client.publish(summary_topic, json.dumps(summary), qos=0)
            except Exception as e:
                print(f"[EDGE BRAIN:{self.patient_id}] Error publishing summary: {e}")

    def shutdown(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


def create_shared_onnx_session(model_path):
    """Loads ONNX model with priority: TensorRT -> CUDA -> CPU (with NumPy fallback if ort unavailable)."""
    if ort is None or not os.path.exists(model_path):
        print("[EDGE BRAIN] Using built-in Jetson NumPy 1D-CNN engine.")
        return None, "Jetson_NumPy_1DCNN"

    try:
        available_providers = ort.get_available_providers()
        print(f"[EDGE BRAIN] Available execution providers: {available_providers}")

        provider_priority = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        selected_providers = [p for p in provider_priority if p in available_providers]
        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Explicitly set thread counts to prevent pthread_setaffinity_np warnings on Jetson ARM cores
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        session = ort.InferenceSession(model_path, sess_options=opts, providers=selected_providers)
        active_provider = session.get_providers()[0]
        print(f"[EDGE BRAIN] Active Execution Provider: {active_provider}")
        return session, active_provider
    except Exception as e:
        print(f"[EDGE BRAIN] ONNX Runtime fallback triggered ({e}); using built-in 1D-CNN engine.")
        return None, "Jetson_1DCNN_Engine"


def main():
    parser = argparse.ArgumentParser(description="Bedside Edge Brain (GPU Hero Agent)")
    parser.add_argument(
        "--patient-id", type=str, default="Bed_1", help="Patient/Bed identifier for this Jetson"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional list of multiple patient IDs to run (e.g. Bed_1 Bed_2 Bed_3)",
    )
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT broker IP address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to ecg_anomaly.onnx")
    args = parser.parse_args()

    patient_list = args.patients if args.patients else [args.patient_id]
    stop_event = threading.Event()

    session, active_provider = create_shared_onnx_session(args.model)

    brains = []
    threads = []
    for pid in patient_list:
        brain = BedsideEdgeBrain(
            patient_id=pid,
            broker=args.broker,
            port=args.port,
            session=session,
            active_provider=active_provider,
            stop_event=stop_event,
        )
        t = threading.Thread(target=brain.run_summary_publisher, daemon=True)
        t.start()
        brains.append(brain)
        threads.append(t)

    def signal_handler(sig, frame):
        print("\n[INFO] Stopping Bedside Edge Brain(s)...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 64)
    print("Bedside Edge Brain Active (ICU Bed Monitor)")
    print(f"Patient Beds:     {patient_list}")
    print(f"MQTT Broker:      {args.broker}:{args.port}")
    print(f"Active Provider:  {active_provider}")
    print("=" * 64)

    while not stop_event.is_set():
        time.sleep(0.5)

    for t in threads:
        t.join(timeout=2.0)
    for b in brains:
        b.shutdown()
    print("[EDGE BRAIN] Terminated cleanly.")


if __name__ == "__main__":
    main()

