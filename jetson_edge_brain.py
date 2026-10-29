#!/usr/bin/env python3
"""
Bedside GPU Hero Agent & Local ICU Monitor (NVIDIA Jetson Orin / Edge Brain)
Executes 1D-CNN ECG anomaly inference, triggers zero-latency bedside alarms,
validates multi-sensor plausibility, and publishes verified summaries over MQTT.
"""

import os
import sys
import time
import json
import signal
import argparse
import threading
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt

CHAOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chaos.json")

def read_chaos_config():
    """Reads chaos.json safely with fallback."""
    default_config = {
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False}
    }
    if not os.path.exists(CHAOS_FILE):
        return default_config
    try:
        with open(CHAOS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return default_config

class BedsideEdgeBrain:
    def __init__(self, broker="127.0.0.1", port=1883, model_path="ecg_anomaly.onnx"):
        self.broker = broker
        self.port = port
        self.model_path = model_path
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # Telemetry State
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

        # Initialize ONNX Runtime Session with Hardware Acceleration Fallback
        self._init_inference_engine()

        # Initialize MQTT Client
        self._init_mqtt()

    def _init_inference_engine(self):
        """Loads ONNX model with priority: TensorRT -> CUDA -> CPU."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found! Run make_model.py first.")

        available_providers = ort.get_available_providers()
        print(f"[EDGE BRAIN] Available execution providers: {available_providers}")

        provider_priority = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        selected_providers = [p for p in provider_priority if p in available_providers]

        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=selected_providers)

        active_provider = self.session.get_providers()[0]
        self.active_provider = active_provider
        print(f"[EDGE BRAIN] Inference engine initialized successfully.")
        print(f"[EDGE BRAIN] Active Execution Provider: {self.active_provider}")

    def _init_mqtt(self):
        """Initializes MQTT connection and subscriptions."""
        client_id = f"Jetson_Edge_Brain_{os.getpid()}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        else:
            self.client = mqtt.Client(client_id=client_id)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[EDGE BRAIN] Connected to MQTT broker at {self.broker}:{self.port} (rc={rc})")
        # Subscribe to sensor telemetry and central control offload topic
        self.client.subscribe("hospital/sensors/#")
        self.client.subscribe("hospital/control/offload")

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode("utf-8"))

            if topic == "hospital/control/offload":
                with self.lock:
                    self.force_offload = bool(payload.get("offload", False))
                return

            if topic.startswith("hospital/sensors/"):
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
            print(f"[EDGE BRAIN] Error processing message from {msg.topic}: {e}")

    def _process_ecg(self, payload):
        """Performs 1D-CNN Arrhythmia inference and bedside alarm trigger."""
        ecg_wave = payload.get("ecg_wave", [])
        if not ecg_wave or len(ecg_wave) != 250:
            return

        wave_np = np.array(ecg_wave, dtype=np.float32).reshape(1, 1, 250)
        variance = float(np.var(wave_np))

        # Measure high-precision inference latency in milliseconds
        t0 = time.perf_counter()
        outputs = self.session.run(None, {"ecg_input": wave_np})
        t1 = time.perf_counter()
        inf_ms = (t1 - t0) * 1000.0

        logits = outputs[0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        arrhythmia_prob = float(probs[0, 1])

        # Flag anomaly if model outputs high probability OR waveform variance exceeds normal sinus bounds
        if variance > 0.35 or np.max(np.abs(wave_np)) > 2.5:
            arrhythmia_prob = max(arrhythmia_prob, 0.96)

        # Bedside immediate alarm condition (under 5ms response time)
        is_alarm = arrhythmia_prob > 0.70

        with self.lock:
            self.latest_ecg_wave = ecg_wave
            self.latest_ecg_var = variance
            self.inf_ms = inf_ms
            self.arrhythmia_prob = arrhythmia_prob
            self.local_bed_alarm = is_alarm

        # Trigger immediate alert if alarm condition is met
        if is_alarm:
            self._trigger_bedside_alert(arrhythmia_prob, inf_ms, variance)

        self._check_plausibility()

    def _check_plausibility(self):
        """Evaluates cross-sensor physiological plausibility."""
        with self.lock:
            hr = self.latest_hr
            spo2 = self.latest_spo2
            arrhythmia = self.arrhythmia_prob > 0.70

        conflict = False

        # Rule 1: Normal ECG sinus rhythm & good oxygen, but HR impossible (e.g. <= 0 or >= 240)
        if not arrhythmia and spo2 >= 95 and (hr <= 0 or hr >= 240):
            conflict = True

        # Rule 2: Clinically impossible vital extremes
        if spo2 < 50 or spo2 > 105 or hr < 30 or hr > 280:
            conflict = True

        # Rule 3: Extreme SpO2 desaturation with completely normal HR and resting ECG
        if spo2 <= 30 and hr > 60 and not arrhythmia:
            conflict = True

        with self.lock:
            self.sensor_conflict = conflict

    def _trigger_bedside_alert(self, prob, inf_ms, variance):
        """Dispatches immediate emergency alert over appropriate network route."""
        chaos = read_chaos_config()
        use_relay_b = chaos.get("Orin_Link", {}).get("use_relay_b", False) or self.force_offload

        alert_topic = "hospital/alerts/relay_b" if use_relay_b else "hospital/alerts/primary"

        alert_payload = {
            "alert": "CRITICAL_ARRHYTHMIA_DETECTED",
            "arrhythmia_prob": round(prob, 4),
            "inf_ms": round(inf_ms, 2),
            "ecg_variance": round(variance, 4),
            "timestamp": time.time(),
            "route": "relay_b" if use_relay_b else "primary"
        }

        try:
            self.client.publish(alert_topic, json.dumps(alert_payload), qos=1)
            print(f"[EDGE ALARM] Triggered Bedside Alarm ({prob*100:.1f}%) via {alert_topic} in {inf_ms:.2f}ms")
        except Exception as e:
            print(f"[EDGE BRAIN] Failed to dispatch alert: {e}")

    def run_summary_publisher(self):
        """Publishes unified bedside telemetry and inference summary every 0.4 seconds."""
        summary_topic = "hospital/edge/summary"

        while not self.stop_event.is_set():
            time.sleep(0.40)
            chaos = read_chaos_config()
            use_relay_b = chaos.get("Orin_Link", {}).get("use_relay_b", False) or self.force_offload
            routing_path = "relay_b" if use_relay_b else "primary"

            with self.lock:
                summary = {
                    "timestamp": time.time(),
                    "inf_ms": round(self.inf_ms, 2),
                    "exec_provider": self.active_provider,
                    "arrhythmia_prob": round(self.arrhythmia_prob, 4),
                    "local_bed_alarm": self.local_bed_alarm,
                    "sensor_conflict": self.sensor_conflict,
                    "latest_vitals": {
                        "hr": self.latest_hr,
                        "spo2": self.latest_spo2,
                        "ecg_variance": round(self.latest_ecg_var, 4)
                    },
                    "routing_path": routing_path
                }

            try:
                self.client.publish(summary_topic, json.dumps(summary), qos=0)
            except Exception as e:
                print(f"[EDGE BRAIN] Error publishing summary: {e}")

    def start(self):
        publisher_thread = threading.Thread(target=self.run_summary_publisher, daemon=True)
        publisher_thread.start()

        print("=" * 60)
        print("Bedside Edge Brain Active (ICU Bed Monitor)")
        print(f"MQTT Broker:      {self.broker}:{self.port}")
        print(f"Model Path:       {self.model_path}")
        print(f"Active Provider:  {self.active_provider}")
        print("Listening for sensor telemetry on hospital/sensors/#...")
        print("Press Ctrl+C to terminate.")
        print("=" * 60)

        while not self.stop_event.is_set():
            time.sleep(0.5)

        publisher_thread.join(timeout=2.0)
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        print("[EDGE BRAIN] Terminated cleanly.")

def main():
    parser = argparse.ArgumentParser(description="Bedside Edge Brain (GPU Hero Agent)")
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--model", type=str, default="ecg_anomaly.onnx", help="Path to ecg_anomaly.onnx model")
    args = parser.parse_args()

    brain = BedsideEdgeBrain(broker=args.broker, port=args.port, model_path=args.model)

    def signal_handler(sig, frame):
        print("\n[INFO] Stopping Bedside Edge Brain...")
        brain.stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    brain.start()

if __name__ == "__main__":
    main()
