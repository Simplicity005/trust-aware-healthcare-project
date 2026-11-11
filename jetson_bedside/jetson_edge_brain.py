#!/usr/bin/env python3
"""
Bedside GPU Hero Agent & Local ICU Monitor (Runs on NVIDIA Jetson Orin)
Executes <5ms 1D-CNN ECG anomaly inference, triggers local bedside alarms,
validates cross-sensor physiological plausibility, streams bedside summaries to
the Nurse Station, AND serves a real-time Local Bedside ICU Monitor Dashboard
directly on the Jetson at http://localhost:8080 (zero extra dependencies).
"""

import os
import time
import json
import signal
import random
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


BEDSIDE_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Jetson Orin — Local Bedside ICU Monitor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    padding: 18px 24px;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 12px;
    margin-bottom: 16px;
  }
  .title-block h1 {
    font-size: 1.45rem;
    color: #f8fafc;
    letter-spacing: 0.5px;
  }
  .title-block p {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 3px;
  }
  .badge-row {
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .pill {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #38bdf8;
  }
  .banner {
    padding: 14px 18px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 1.15rem;
    text-align: center;
    margin-bottom: 14px;
    transition: all 0.2s ease;
  }
  .banner-normal {
    background: #064e3b;
    color: #a7f3d0;
    border: 1px solid #10b981;
  }
  @keyframes flash-red {
    0% { background: #7f1d1d; box-shadow: 0 0 10px #ef4444; }
    50% { background: #dc2626; box-shadow: 0 0 28px #f87171; }
    100% { background: #7f1d1d; box-shadow: 0 0 10px #ef4444; }
  }
  .banner-alarm {
    animation: flash-red 0.8s infinite;
    color: #ffffff;
    border: 2px solid #fca5a5;
  }
  .banner-conflict {
    background: #78350f;
    color: #fde68a;
    border: 1px solid #f59e0b;
    padding: 10px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    text-align: center;
    margin-bottom: 14px;
    display: none;
  }
  .main-grid {
    display: grid;
    grid-template-columns: 2.3fr 1fr;
    gap: 18px;
  }
  .panel {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 16px;
  }
  .panel-title {
    font-size: 0.9rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
  }
  canvas {
    width: 100%;
    height: 320px;
    background: #050811;
    border: 1px solid #1e293b;
    border-radius: 8px;
    display: block;
  }
  .vitals-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .vital-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .vital-label {
    font-size: 0.82rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 700;
  }
  .vital-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 3px;
  }
  .vital-val {
    font-size: 2.2rem;
    font-weight: 900;
  }
  .hr-color { color: #22c55e; }
  .spo2-color { color: #38bdf8; }
  .ai-color { color: #facc15; }
  .prob-color { color: #f472b6; }
  .preset-btn {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.80rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.1s ease, filter 0.1s ease;
  }
  .preset-btn:hover {
    filter: brightness(1.2);
    transform: translateY(-1px);
  }
  .footer-bar {
    margin-top: 16px;
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #64748b;
    border-top: 1px solid #1e293b;
    padding-top: 10px;
  }
</style>
</head>
<body>
  <div class="header">
    <div class="title-block">
      <h1>🖥️ NVIDIA Jetson Orin — Local Bedside ICU Monitor (<span id="bedId">Bed_1</span>)</h1>
      <p>Zero-Latency On-Device 1D-CNN ECG Arrhythmia Detection (&lt;5 ms SLA) &amp; Bedside Vitals Controller</p>
    </div>
    <div class="badge-row">
      <div class="pill">Provider: <span id="provider">CPU</span></div>
      <div class="pill">Route: <span id="route">PRIMARY</span></div>
    </div>
  </div>

  <div id="alarmBanner" class="banner banner-normal">
    ✅ NORMAL SINUS RHYTHM — Zero-Latency Bedside AI Monitor Nominal
  </div>

  <div id="conflictBanner" class="banner-conflict">
    ⚠️ CROSS-SENSOR PHYSIOLOGICAL CONFLICT DETECTED: Discrepancy between ECG Sinus Rhythm, HR, and SpO2!
  </div>

  <div class="main-grid">
    <div class="panel">
      <div class="panel-title">
        <span>🫀 Live 250 Hz Bedside ECG Waveform (Lead II — 1.0s Window)</span>
        <span>Variance: <strong id="ecgVar">0.0400</strong></span>
      </div>
      <canvas id="ecgCanvas" width="900" height="320"></canvas>
    </div>

    <div class="vitals-stack">
      <div class="vital-card">
        <div>
          <div class="vital-label">💓 Heart Rate</div>
          <div class="vital-sub">Agent_HR (Normal: 60–100)</div>
        </div>
        <div class="vital-val hr-color"><span id="hrVal">75</span> <span style="font-size:1rem;">BPM</span></div>
      </div>

      <div class="vital-card">
        <div>
          <div class="vital-label">🫁 Oxygen Saturation</div>
          <div class="vital-sub">Agent_SpO2 (Normal: 95–100%)</div>
        </div>
        <div class="vital-val spo2-color"><span id="spo2Val">98</span> <span style="font-size:1rem;">%</span></div>
      </div>

      <div class="vital-card">
        <div>
          <div class="vital-label">⚡ 1D-CNN Inference Time</div>
          <div class="vital-sub">Bedside SLA Target: &lt; 5.0 ms</div>
        </div>
        <div class="vital-val ai-color"><span id="infMs">0.12</span> <span style="font-size:1rem;">ms</span></div>
      </div>

      <div class="vital-card">
        <div>
          <div class="vital-label">🧠 Arrhythmia Risk</div>
          <div class="vital-sub">1D-CNN Softmax Output</div>
        </div>
        <div class="vital-val prob-color"><span id="probVal">2.0</span> <span style="font-size:1rem;">%</span></div>
      </div>
    </div>
  </div>

  <!-- Bedside Patient Vitals & Sensor Manipulation Console -->
  <div class="panel" style="margin-top: 16px; border: 1px solid #0284c7;">
    <div class="panel-title">
      <span style="color:#38bdf8; font-weight:800; font-size:0.95rem;">🎛️ Bedside Patient Vitals &amp; Sensor Manipulation Console</span>
      <span id="manipStatus" style="font-size:0.80rem; color:#10b981;">● Hosted Directly on Jetson Hardware</span>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 10px;">
      <!-- Column 1: Patient Target Vitals -->
      <div>
        <h4 style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; margin-bottom:10px;">Patient Target Vitals (Bedside Override)</h4>
        
        <div style="margin-bottom:14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
            <span>💓 Target Heart Rate:</span>
            <span><strong id="targetHrLabel" style="color:#22c55e;">75</strong> BPM</span>
          </div>
          <input type="range" id="targetHr" min="30" max="220" value="75" step="1" oninput="sendManipulation()" style="width:100%; cursor:pointer;">
        </div>

        <div style="margin-bottom:14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
            <span>🫁 Target SpO2 Saturation:</span>
            <span><strong id="targetSpo2Label" style="color:#38bdf8;">98</strong>%</span>
          </div>
          <input type="range" id="targetSpo2" min="50" max="100" value="98" step="1" oninput="sendManipulation()" style="width:100%; cursor:pointer;">
        </div>

        <div style="margin-bottom:14px;">
          <span style="font-size:0.85rem; display:block; margin-bottom:6px;">🫀 ECG Cardiac Waveform Rhythm:</span>
          <select id="ecgCondition" onchange="sendManipulation()" style="width:100%; background:#0f172a; color:#f8fafc; border:1px solid #334155; padding:8px 10px; border-radius:6px; font-size:0.85rem;">
            <option value="normal">🟢 Normal Sinus Rhythm (Nominal PQRST Wave)</option>
            <option value="arrhythmia">🚨 Ventricular Arrhythmia (Triggers &lt;5ms 1D-CNN Alarm)</option>
            <option value="malicious">🛡️ Byzantine Spoofed / Erratic Signal (Triggers Isolation)</option>
          </select>
        </div>
      </div>

      <!-- Column 2: Sensor Latency, Drop Rate & Compute -->
      <div>
        <h4 style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; margin-bottom:10px;">Network Transit Delay &amp; Fault Injection</h4>

        <div style="margin-bottom:14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
            <span>⏱️ Sensor Delay (ms):</span>
            <span><strong id="delayLabel" style="color:#facc15;">10</strong> ms</span>
          </div>
          <input type="range" id="delayMs" min="5" max="400" value="10" step="5" oninput="sendManipulation()" style="width:100%; cursor:pointer;">
        </div>

        <div style="margin-bottom:14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
            <span>📉 Packet Drop Rate:</span>
            <span><strong id="dropLabel" style="color:#f87171;">0.00</strong></span>
          </div>
          <input type="range" id="dropRate" min="0.0" max="0.6" value="0.0" step="0.05" oninput="sendManipulation()" style="width:100%; cursor:pointer;">
        </div>

        <div style="display:flex; gap:16px; margin-top:12px;">
          <label style="font-size:0.84rem; display:flex; align-items:center; gap:6px; cursor:pointer;">
            <input type="checkbox" id="maliciousSpoof" onchange="sendManipulation()"> Inject Byzantine Spoofing
          </label>
          <label style="font-size:0.84rem; display:flex; align-items:center; gap:6px; cursor:pointer;">
            <input type="checkbox" id="overloadGpu" onchange="sendManipulation()"> Simulate GPU Congestion (Relay B)
          </label>
        </div>
      </div>
    </div>

    <!-- Clinical Presets Toolbar -->
    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #1e293b; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
      <span style="font-size:0.82rem; color:#94a3b8; font-weight:700;">Clinical Presets:</span>
      <button onclick="applyPreset('healthy')" class="preset-btn" style="background:#064e3b; color:#a7f3d0; border:1px solid #10b981;">🟢 Normal Sinus</button>
      <button onclick="applyPreset('arrhythmia')" class="preset-btn" style="background:#7f1d1d; color:#fecaca; border:1px solid #ef4444;">🚨 Ventricular Arrhythmia</button>
      <button onclick="applyPreset('spoofed')" class="preset-btn" style="background:#701a75; color:#f5d0fe; border:1px solid #c026d3;">🛡️ Byzantine Spoofed</button>
      <button onclick="applyPreset('lag')" class="preset-btn" style="background:#78350f; color:#fde68a; border:1px solid #f59e0b;">⏱️ Wi-Fi Network Lag</button>
      <button onclick="applyPreset('congested')" class="preset-btn" style="background:#1e293b; color:#38bdf8; border:1px solid #38bdf8;">🔀 Orin Offload (Relay B)</button>
      <button onclick="applyPreset('reset')" class="preset-btn" style="background:#334155; color:#f1f5f9; border:1px solid #64748b;">🔄 Reset to Baseline</button>
    </div>
  </div>

  <div class="footer-bar">
    <span>MQTT Nurse Station Broker: <strong id="brokerInfo">127.0.0.1:1883</strong></span>
    <span>Local Bedside Refresh: <strong>250 ms (Direct Memory Read)</strong></span>
  </div>

<script>
const canvas = document.getElementById('ecgCanvas');
const ctx = canvas.getContext('2d');

function drawGrid() {
  ctx.fillStyle = '#050811';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#0f291e';
  ctx.lineWidth = 1;
  const step = 30;
  for (let x = 0; x < canvas.width; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }
}

function drawECG(wave, isAlarm) {
  drawGrid();
  if (!wave || wave.length === 0) return;

  ctx.strokeStyle = isAlarm ? '#ef4444' : '#22c55e';
  ctx.lineWidth = 2.6;
  ctx.shadowColor = isAlarm ? '#ef4444' : '#22c55e';
  ctx.shadowBlur = 8;
  ctx.beginPath();

  const midY = canvas.height * 0.62;
  const scaleY = canvas.height * 0.32;
  for (let i = 0; i < wave.length; i++) {
    const x = (i / (wave.length - 1)) * canvas.width;
    const y = midY - wave[i] * scaleY;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

async function pollState() {
  try {
    const res = await fetch('/api/state');
    const s = await res.json();
    document.getElementById('bedId').textContent = s.patient_id;
    document.getElementById('provider').textContent = s.exec_provider.replace('ExecutionProvider', '');
    document.getElementById('route').textContent = s.routing_path === 'relay_b' ? 'RELAY PATH B (OFFLOADED)' : 'PRIMARY DIRECT';
    document.getElementById('hrVal').textContent = s.latest_vitals.hr;
    document.getElementById('spo2Val').textContent = s.latest_vitals.spo2;
    document.getElementById('infMs').textContent = s.inf_ms.toFixed(2);
    document.getElementById('probVal').textContent = (s.arrhythmia_prob * 100).toFixed(1);
    document.getElementById('ecgVar').textContent = s.latest_vitals.ecg_variance.toFixed(4);
    document.getElementById('brokerInfo').textContent = s.broker;

    const banner = document.getElementById('alarmBanner');
    if (s.local_bed_alarm) {
      banner.className = 'banner banner-alarm';
      banner.innerHTML = '🚨 CRITICAL BEDSIDE ALARM: ARRHYTHMIA DETECTED (' +
        (s.arrhythmia_prob * 100).toFixed(1) + '% Confidence | ' + s.inf_ms.toFixed(2) + ' ms Edge Response)';
    } else {
      banner.className = 'banner banner-normal';
      banner.innerHTML = '✅ NORMAL SINUS RHYTHM — Zero-Latency Bedside AI Monitor Nominal';
    }

    const cBanner = document.getElementById('conflictBanner');
    cBanner.style.display = s.sensor_conflict ? 'block' : 'none';

    drawECG(s.ecg_wave, s.local_bed_alarm);
  } catch (e) {}
}

async function sendManipulation() {
  const targetHr = parseInt(document.getElementById('targetHr').value);
  const targetSpo2 = parseInt(document.getElementById('targetSpo2').value);
  const ecgCondition = document.getElementById('ecgCondition').value;
  const delayMs = parseInt(document.getElementById('delayMs').value);
  const dropRate = parseFloat(document.getElementById('dropRate').value);
  const malicious = document.getElementById('maliciousSpoof').checked;
  const overloadGpu = document.getElementById('overloadGpu').checked;

  document.getElementById('targetHrLabel').textContent = targetHr;
  document.getElementById('targetSpo2Label').textContent = targetSpo2;
  document.getElementById('delayLabel').textContent = delayMs;
  document.getElementById('dropLabel').textContent = dropRate.toFixed(2);

  const payload = {
    target_hr: targetHr,
    target_spo2: targetSpo2,
    ecg_condition: ecgCondition,
    delay_ms: delayMs,
    drop_rate: dropRate,
    malicious: malicious,
    overload_gpu: overloadGpu,
    use_relay_b: overloadGpu
  };

  try {
    const res = await fetch('/api/manipulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      const st = document.getElementById('manipStatus');
      st.textContent = '✅ Bedside vitals applied directly on Jetson';
      st.style.color = '#22c55e';
      setTimeout(() => { st.textContent = '● Hosted Directly on Jetson Hardware'; st.style.color = '#10b981'; }, 2000);
    }
  } catch (e) {}
}

function applyPreset(name) {
  if (name === 'healthy') {
    document.getElementById('targetHr').value = 75;
    document.getElementById('targetSpo2').value = 98;
    document.getElementById('ecgCondition').value = 'normal';
    document.getElementById('delayMs').value = 10;
    document.getElementById('dropRate').value = 0.0;
    document.getElementById('maliciousSpoof').checked = false;
    document.getElementById('overloadGpu').checked = false;
  } else if (name === 'arrhythmia') {
    document.getElementById('targetHr').value = 145;
    document.getElementById('targetSpo2').value = 94;
    document.getElementById('ecgCondition').value = 'arrhythmia';
    document.getElementById('delayMs').value = 12;
    document.getElementById('dropRate').value = 0.0;
    document.getElementById('maliciousSpoof').checked = false;
    document.getElementById('overloadGpu').checked = false;
  } else if (name === 'spoofed') {
    document.getElementById('targetHr').value = 190;
    document.getElementById('targetSpo2').value = 65;
    document.getElementById('ecgCondition').value = 'malicious';
    document.getElementById('delayMs').value = 25;
    document.getElementById('dropRate').value = 0.15;
    document.getElementById('maliciousSpoof').checked = true;
    document.getElementById('overloadGpu').checked = false;
  } else if (name === 'lag') {
    document.getElementById('targetHr').value = 78;
    document.getElementById('targetSpo2').value = 97;
    document.getElementById('ecgCondition').value = 'normal';
    document.getElementById('delayMs').value = 240;
    document.getElementById('dropRate').value = 0.35;
    document.getElementById('maliciousSpoof').checked = false;
    document.getElementById('overloadGpu').checked = false;
  } else if (name === 'congested') {
    document.getElementById('targetHr').value = 75;
    document.getElementById('targetSpo2').value = 98;
    document.getElementById('ecgCondition').value = 'normal';
    document.getElementById('delayMs').value = 15;
    document.getElementById('dropRate').value = 0.0;
    document.getElementById('maliciousSpoof').checked = false;
    document.getElementById('overloadGpu').checked = true;
  } else if (name === 'reset') {
    document.getElementById('targetHr').value = 75;
    document.getElementById('targetSpo2').value = 98;
    document.getElementById('ecgCondition').value = 'normal';
    document.getElementById('delayMs').value = 10;
    document.getElementById('dropRate').value = 0.0;
    document.getElementById('maliciousSpoof').checked = false;
    document.getElementById('overloadGpu').checked = false;
  }
  sendManipulation();
}

setInterval(pollState, 250);
pollState();
</script>
</body>
</html>
"""


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

    def __init__(
        self,
        patient_id,
        broker="127.0.0.1",
        port=1883,
        session=None,
        active_provider="CPUExecutionProvider",
        stop_event=None,
    ):
        self.patient_id = patient_id
        self.broker = broker
        self.port = port
        self.session = session
        self.active_provider = active_provider
        self.stop_event = stop_event or threading.Event()
        self.lock = threading.Lock()

        self.latest_hr = 75
        self.latest_spo2 = 98
        self.latest_ecg_wave = []
        self.latest_ecg_var = 0.04
        self.inf_ms = 0.12
        self.arrhythmia_prob = 0.02
        self.local_bed_alarm = False
        self.sensor_conflict = False
        self.force_offload = False
        self.routing_path = "primary"
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
        self.client.subscribe("hospital/discovery/probe")
        self.client.subscribe(f"hospital/{self.patient_id}/control/admit")
        self.send_discovery_announcement()

    def send_discovery_announcement(self):
        """Announces this Jetson Orin node to the Nurse Station for auto-detection."""
        if not self.client:
            return
        try:
            with self.lock:
                hr = self.latest_hr
                spo2 = self.latest_spo2
                ecg_var = self.latest_ecg_var
            announcement = {
                "patient_id": self.patient_id,
                "device_model": "NVIDIA Jetson Orin",
                "exec_provider": self.active_provider,
                "broker": f"{self.broker}:{self.port}",
                "timestamp": time.time(),
                "status": "ready",
                "latest_vitals": {"hr": hr, "spo2": spo2, "ecg_variance": ecg_var},
            }
            self.client.publish("hospital/discovery/announce", json.dumps(announcement), qos=1)
        except Exception:
            pass

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode("utf-8"))

            if topic == "hospital/discovery/probe":
                self.send_discovery_announcement()
                return

            if topic == f"hospital/{self.patient_id}/control/admit":
                print(f"[EDGE BRAIN:{self.patient_id}] Officially admitted by Nurse Station to ICU Ward!")
                return

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

    def get_local_state_dict(self):
        with self.lock:
            return {
                "patient_id": self.patient_id,
                "broker": f"{self.broker}:{self.port}",
                "timestamp": time.time(),
                "inf_ms": round(self.inf_ms, 2),
                "exec_provider": self.active_provider,
                "arrhythmia_prob": round(self.arrhythmia_prob, 4),
                "local_bed_alarm": self.local_bed_alarm,
                "sensor_conflict": self.sensor_conflict,
                "latest_vitals": {
                    "hr": self.latest_hr,
                    "spo2": self.latest_spo2,
                    "ecg_variance": round(self.latest_ecg_var, 4),
                },
                "ecg_wave": self.latest_ecg_wave,
                "routing_path": self.routing_path,
            }

    def run_summary_publisher(self):
        summary_topic = f"hospital/{self.patient_id}/edge/summary"
        loop_cnt = 0

        while not self.stop_event.is_set():
            time.sleep(0.40)
            loop_cnt += 1
            if loop_cnt % 10 == 0:
                self.send_discovery_announcement()

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
                self.routing_path = routing_path
                summary = {
                    "patient_id": self.patient_id,
                    "device_model": "NVIDIA Jetson Orin",
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


def start_local_bedside_web_server(primary_brain, port=8080):
    """Starts a zero-dependency local HTTP Bedside ICU Monitor dashboard on the Jetson."""

    class BedsideHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/state":
                state = primary_brain.get_local_state_dict()
                body = json.dumps(state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path in ("/api/chaos", "/api/vitals"):
                chaos = read_patient_chaos(primary_brain.patient_id)
                body = json.dumps(chaos).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                body = BEDSIDE_HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            if self.path in ("/api/manipulate", "/api/chaos", "/api/vitals"):
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                try:
                    data = json.loads(post_body.decode("utf-8"))
                    patient_id = primary_brain.patient_id

                    new_cfg = {
                        "Agent_ECG": {
                            "delay_ms": int(data.get("delay_ms", 12)),
                            "drop_rate": float(data.get("drop_rate", 0.0)),
                            "malicious": bool(data.get("malicious", False)) or (data.get("ecg_condition") == "malicious"),
                            "condition": data.get("ecg_condition", "normal"),
                        },
                        "Agent_SpO2": {
                            "delay_ms": int(data.get("delay_ms", 10)),
                            "drop_rate": float(data.get("drop_rate", 0.0)),
                            "malicious": bool(data.get("malicious", False)),
                            "target_spo2": int(data.get("target_spo2", 98)),
                        },
                        "Agent_HR": {
                            "delay_ms": int(data.get("delay_ms", 10)),
                            "drop_rate": float(data.get("drop_rate", 0.0)),
                            "malicious": bool(data.get("malicious", False)),
                            "target_hr": int(data.get("target_hr", 75)),
                        },
                        "Orin_Link": {
                            "overload_gpu": bool(data.get("overload_gpu", False)),
                            "use_relay_b": bool(data.get("use_relay_b", False)),
                        },
                        "vitals": {
                            "target_hr": int(data.get("target_hr", 75)),
                            "target_spo2": int(data.get("target_spo2", 98)),
                            "ecg_condition": data.get("ecg_condition", "normal"),
                        },
                    }
                    update_chaos_file(new_cfg, patient_id=patient_id)

                    with primary_brain.lock:
                        primary_brain.latest_hr = int(data.get("target_hr", primary_brain.latest_hr))
                        primary_brain.latest_spo2 = int(data.get("target_spo2", primary_brain.latest_spo2))
                        if data.get("ecg_condition") == "arrhythmia":
                            primary_brain.arrhythmia_prob = 0.94
                            primary_brain.local_bed_alarm = True
                        elif data.get("ecg_condition") == "normal":
                            primary_brain.arrhythmia_prob = 0.02
                            primary_brain.local_bed_alarm = False
                        primary_brain.force_offload = bool(data.get("use_relay_b", False)) or bool(data.get("overload_gpu", False))

                    if primary_brain.client:
                        try:
                            primary_brain.client.publish(
                                f"hospital/{patient_id}/control/chaos",
                                json.dumps(new_cfg),
                                qos=1,
                                retain=True,
                            )
                        except Exception:
                            pass

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "patient_id": patient_id, "config": new_cfg}).encode("utf-8"))
                    return
                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                    return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass  # Keep terminal output clean

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), BedsideHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[BEDSIDE DASHBOARD] Local ICU Monitor live at: http://localhost:{port}")
        return server
    except Exception as e:
        print(f"[BEDSIDE DASHBOARD] Note: Port {port} busy ({e})")
        return None


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
    parser = argparse.ArgumentParser(description="Bedside Edge Brain & Local ICU Monitor")
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
    parser.add_argument("--web-port", type=int, default=8080, help="Local Bedside Dashboard HTTP port")
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

    # Start Local Bedside ICU Monitor Web Server on the Jetson (default http://localhost:8080)
    web_server = start_local_bedside_web_server(brains[0], port=args.web_port)

    def signal_handler(sig, frame):
        print("\n[INFO] Stopping Bedside Edge Brain(s)...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 64)
    print("Bedside Edge Brain Active (ICU Bed Monitor)")
    print(f"Patient Beds:       {patient_list}")
    print(f"MQTT Broker:        {args.broker}:{args.port}")
    print(f"Active Provider:    {active_provider}")
    print(f"Bedside Dashboard:  http://localhost:{args.web_port}")
    print("=" * 64)

    while not stop_event.is_set():
        time.sleep(0.5)

    if web_server:
        web_server.shutdown()
    for t in threads:
        t.join(timeout=2.0)
    for b in brains:
        b.shutdown()
    print("[EDGE BRAIN] Terminated cleanly.")


if __name__ == "__main__":
    main()
