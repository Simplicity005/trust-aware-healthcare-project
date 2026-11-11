#!/usr/bin/env python3
"""
Bedside GPU Edge Brain & Local Bedside Monitor (Runs on NVIDIA Jetson Orin)
Executes 1D-CNN ECG anomaly inference, triggers local bedside alarms,
validates cross-sensor physiological plausibility, streams bedside summaries to
the Nurse Station, and serves a clean Bento-grid Local Bedside Monitor
directly on the Jetson at http://localhost:8080 with local sensor fault controls.
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


def default_bed_chaos():
    return {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False},
    }


def read_patient_chaos(patient_id="Bed_1"):
    if not os.path.exists(CHAOS_FILE):
        return default_bed_chaos()
    try:
        with CHAOS_LOCK:
            with open(CHAOS_FILE, "r") as f:
                data = json.load(f)
        if patient_id in data and isinstance(data[patient_id], dict):
            base = default_bed_chaos()
            for k, v in data[patient_id].items():
                if k in base and isinstance(v, dict):
                    base[k].update(v)
            return base
        if "Agent_ECG" in data:
            base = default_bed_chaos()
            for k, v in data.items():
                if k in base and isinstance(v, dict):
                    base[k].update(v)
            return base
        return default_bed_chaos()
    except Exception:
        return default_bed_chaos()


def write_patient_chaos(patient_id, bed_chaos):
    try:
        with CHAOS_LOCK:
            current = {}
            if os.path.exists(CHAOS_FILE):
                try:
                    with open(CHAOS_FILE, "r") as f:
                        current = json.load(f)
                except Exception:
                    current = {}
            if "Agent_ECG" in current:
                current = {"Bed_1": current}
            current[patient_id] = bed_chaos
            tmp_path = CHAOS_FILE + f".tmp.{random.randint(1000, 9999)}"
            with open(tmp_path, "w") as f:
                json.dump(current, f, indent=2)
            os.replace(tmp_path, CHAOS_FILE)
        return True
    except Exception:
        return False


BEDSIDE_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bedside Monitor</title>
<style>
  :root {
    --bg: #0a0a0c;
    --card: #121316;
    --border: #1f2228;
    --text: #ededef;
    --muted: #7c828d;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --blue: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    padding: 20px 24px;
    line-height: 1.4;
    -webkit-font-smoothing: antialiased;
  }
  .container {
    max-width: 1320px;
    margin: 0 auto;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 16px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .topbar-left {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }
  .bed-title {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .bed-meta {
    font-size: 12px;
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .topbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .bed-select {
    display: none;
    background: var(--card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
  }
  .pill {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text);
  }
  .pill.normal { color: var(--green); border-color: rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.08); }
  .pill.warn { color: var(--amber); border-color: rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.08); }
  .pill.danger { color: var(--red); border-color: rgba(239, 68, 68, 0.35); background: rgba(239, 68, 68, 0.12); }

  .bento {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 12px;
  }
  .tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .span-8-2 { grid-column: span 8; grid-row: span 2; }
  .span-4 { grid-column: span 4; }
  .span-3 { grid-column: span 3; }
  .span-12 { grid-column: span 12; }

  @media (max-width: 960px) {
    .span-8-2, .span-4, .span-3 { grid-column: span 12; }
  }

  .label {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .vital-num {
    font-size: 44px;
    font-weight: 600;
    letter-spacing: -0.03em;
    font-variant-numeric: tabular-nums;
    margin: 6px 0 2px 0;
  }
  .vital-unit {
    font-size: 13px;
    color: var(--muted);
    font-weight: 400;
    margin-left: 6px;
  }
  .sub {
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .stat-val {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 8px 0 4px 0;
    font-variant-numeric: tabular-nums;
  }
  canvas {
    width: 100%;
    height: 190px;
    display: block;
    margin-top: 10px;
  }

  /* Sensor & Local Controls Table */
  .ctrl-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .btn {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
    cursor: pointer;
  }
  .btn:hover { border-color: var(--muted); }
  .sensor-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }
  @media (max-width: 960px) {
    .sensor-grid { grid-template-columns: 1fr; }
  }
  .sensor-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }
  .sensor-box-title {
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
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
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
  }
  input[type="range"] {
    width: 100%;
    accent-color: var(--blue);
    cursor: pointer;
  }
  .toggle-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: var(--muted);
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }
  input[type="checkbox"] {
    width: 15px;
    height: 15px;
    accent-color: var(--red);
    cursor: pointer;
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
    </div>

    <div class="bento">
      <!-- ECG Waveform (8 cols x 2 rows) -->
      <div class="tile span-8-2">
        <div class="label">
          <span>ECG Lead II</span>
          <span id="ecgMeta" class="sub">250 Hz</span>
        </div>
        <canvas id="ecgCanvas" width="820" height="190"></canvas>
      </div>

      <!-- Heart Rate (4 cols) -->
      <div class="tile span-4">
        <div class="label">
          <span>Heart rate</span>
          <span id="hrLink" class="sub">0 ms</span>
        </div>
        <div>
          <span class="vital-num" id="hrVal" style="color: var(--green);">--</span>
          <span class="vital-unit">bpm</span>
        </div>
        <div class="sub" id="hrSub">60 - 100 bpm</div>
      </div>

      <!-- SpO2 (4 cols) -->
      <div class="tile span-4">
        <div class="label">
          <span>SpO2</span>
          <span id="spo2Link" class="sub">0 ms</span>
        </div>
        <div>
          <span class="vital-num" id="spo2Val" style="color: var(--blue);">--</span>
          <span class="vital-unit">%</span>
        </div>
        <div class="sub" id="spo2Sub">95 - 100 %</div>
      </div>

      <!-- 4 Compact Status Tiles (3 cols each) -->
      <div class="tile span-3">
        <div class="label"><span>Rhythm</span></div>
        <div class="stat-val" id="rhythmVal">Normal Sinus</div>
        <div class="sub" id="rhythmSub">1D-CNN classifier</div>
      </div>

      <div class="tile span-3">
        <div class="label"><span>Arrhythmia risk</span></div>
        <div class="stat-val" id="probVal">0.0%</div>
        <div class="sub">Threshold 50.0%</div>
      </div>

      <div class="tile span-3">
        <div class="label"><span>Inference</span></div>
        <div class="stat-val" id="infVal">0.00 ms</div>
        <div class="sub" id="provSub">CPUExecutionProvider</div>
      </div>

      <div class="tile span-3">
        <div class="label"><span>Alert route</span></div>
        <div class="stat-val" id="routeVal">Direct</div>
        <div class="sub" id="pktSub">0 packets</div>
      </div>

      <!-- Local Sensor Link Telemetry & Fault Injection (12 cols) -->
      <div class="tile span-12">
        <div class="ctrl-header">
          <div class="label"><span>Sensors and fault injection</span></div>
          <button class="btn" onclick="resetChaos()">Reset all</button>
        </div>

        <div class="sensor-grid">
          <!-- ECG Sensor Controls -->
          <div class="sensor-box">
            <div class="sensor-box-title">
              <span>ECG sensor</span>
              <span class="sensor-metrics" id="m_ecg">12 ms | 0% loss</span>
            </div>
            <div class="field">
              <div class="field-top"><span>Delay</span><span id="lbl_ecg_delay">12 ms</span></div>
              <input type="range" id="ecg_delay" min="5" max="450" step="5" value="12" oninput="pushChaos()">
            </div>
            <div class="field">
              <div class="field-top"><span>Drop rate</span><span id="lbl_ecg_drop">0%</span></div>
              <input type="range" id="ecg_drop" min="0" max="0.9" step="0.05" value="0" oninput="pushChaos()">
            </div>
            <div class="toggle-row">
              <span>Spoof waveform</span>
              <input type="checkbox" id="ecg_mal" onchange="pushChaos()">
            </div>
          </div>

          <!-- SpO2 Sensor Controls -->
          <div class="sensor-box">
            <div class="sensor-box-title">
              <span>SpO2 sensor</span>
              <span class="sensor-metrics" id="m_spo2">10 ms | 0% loss</span>
            </div>
            <div class="field">
              <div class="field-top"><span>Delay</span><span id="lbl_spo2_delay">10 ms</span></div>
              <input type="range" id="spo2_delay" min="5" max="450" step="5" value="10" oninput="pushChaos()">
            </div>
            <div class="field">
              <div class="field-top"><span>Drop rate</span><span id="lbl_spo2_drop">0%</span></div>
              <input type="range" id="spo2_drop" min="0" max="0.9" step="0.05" value="0" oninput="pushChaos()">
            </div>
            <div class="toggle-row">
              <span>Spoof reading</span>
              <input type="checkbox" id="spo2_mal" onchange="pushChaos()">
            </div>
          </div>

          <!-- HR Sensor Controls -->
          <div class="sensor-box">
            <div class="sensor-box-title">
              <span>HR sensor</span>
              <span class="sensor-metrics" id="m_hr">10 ms | 0% loss</span>
            </div>
            <div class="field">
              <div class="field-top"><span>Delay</span><span id="lbl_hr_delay">10 ms</span></div>
              <input type="range" id="hr_delay" min="5" max="450" step="5" value="10" oninput="pushChaos()">
            </div>
            <div class="field">
              <div class="field-top"><span>Drop rate</span><span id="lbl_hr_drop">0%</span></div>
              <input type="range" id="hr_drop" min="0" max="0.9" step="0.05" value="0" oninput="pushChaos()">
            </div>
            <div class="toggle-row">
              <span>Spoof reading</span>
              <input type="checkbox" id="hr_mal" onchange="pushChaos()">
            </div>
          </div>

          <!-- Jetson Node Controls -->
          <div class="sensor-box">
            <div class="sensor-box-title">
              <span>Edge node</span>
              <span class="sensor-metrics" id="m_node">Active</span>
            </div>
            <div class="toggle-row" style="border-top: none; margin-top: 0; padding-top: 2px;">
              <span>GPU load spike</span>
              <input type="checkbox" id="overload_gpu" onchange="pushChaos()">
            </div>
            <div class="toggle-row">
              <span>Link congestion</span>
              <input type="checkbox" id="use_relay_b" onchange="pushChaos()">
            </div>
          </div>
        </div>
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
let selectedBed = null;
let lastWave = [];
let editingChaos = false;
let editTimer = null;
const canvas = document.getElementById('ecgCanvas');
const ctx = canvas.getContext('2d');

function drawECG(wave, isAlert) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#121316';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = '#191b20';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += 38) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  if (!wave || wave.length === 0) return;

  ctx.strokeStyle = isAlert ? '#ef4444' : '#10b981';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const mid = h * 0.58;
  const scale = h * 0.28;
  for (let i = 0; i < wave.length; i++) {
    const x = (i / (wave.length - 1)) * w;
    const y = mid - wave[i] * scale;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function switchBed(bedId) {
  selectedBed = bedId;
  loadChaosForBed(bedId);
}

function updateLabels() {
  document.getElementById('lbl_ecg_delay').textContent = document.getElementById('ecg_delay').value + ' ms';
  document.getElementById('lbl_ecg_drop').textContent = Math.round(document.getElementById('ecg_drop').value * 100) + '%';
  document.getElementById('lbl_spo2_delay').textContent = document.getElementById('spo2_delay').value + ' ms';
  document.getElementById('lbl_spo2_drop').textContent = Math.round(document.getElementById('spo2_drop').value * 100) + '%';
  document.getElementById('lbl_hr_delay').textContent = document.getElementById('hr_delay').value + ' ms';
  document.getElementById('lbl_hr_drop').textContent = Math.round(document.getElementById('hr_drop').value * 100) + '%';
}

function applyChaosToInputs(c) {
  if (!c) return;
  const ecg = c.Agent_ECG || {};
  const spo2 = c.Agent_SpO2 || {};
  const hr = c.Agent_HR || {};
  const link = c.Orin_Link || {};

  document.getElementById('ecg_delay').value = ecg.delay_ms ?? 12;
  document.getElementById('ecg_drop').value = ecg.drop_rate ?? 0;
  document.getElementById('ecg_mal').checked = !!ecg.malicious;

  document.getElementById('spo2_delay').value = spo2.delay_ms ?? 10;
  document.getElementById('spo2_drop').value = spo2.drop_rate ?? 0;
  document.getElementById('spo2_mal').checked = !!spo2.malicious;

  document.getElementById('hr_delay').value = hr.delay_ms ?? 10;
  document.getElementById('hr_drop').value = hr.drop_rate ?? 0;
  document.getElementById('hr_mal').checked = !!hr.malicious;

  document.getElementById('overload_gpu').checked = !!link.overload_gpu;
  document.getElementById('use_relay_b').checked = !!link.use_relay_b;
  updateLabels();
}

async function loadChaosForBed(bedId) {
  try {
    const res = await fetch('/api/chaos?bed=' + encodeURIComponent(bedId));
    const data = await res.json();
    applyChaosToInputs(data);
  } catch (e) {}
}

async function pushChaos() {
  editingChaos = true;
  if (editTimer) clearTimeout(editTimer);
  editTimer = setTimeout(() => { editingChaos = false; }, 1200);
  updateLabels();
  if (!selectedBed) return;

  const payload = {
    patient_id: selectedBed,
    chaos: {
      Agent_ECG: {
        delay_ms: Number(document.getElementById('ecg_delay').value),
        drop_rate: Number(document.getElementById('ecg_drop').value),
        malicious: document.getElementById('ecg_mal').checked
      },
      Agent_SpO2: {
        delay_ms: Number(document.getElementById('spo2_delay').value),
        drop_rate: Number(document.getElementById('spo2_drop').value),
        malicious: document.getElementById('spo2_mal').checked
      },
      Agent_HR: {
        delay_ms: Number(document.getElementById('hr_delay').value),
        drop_rate: Number(document.getElementById('hr_drop').value),
        malicious: document.getElementById('hr_mal').checked
      },
      Orin_Link: {
        overload_gpu: document.getElementById('overload_gpu').checked,
        use_relay_b: document.getElementById('use_relay_b').checked
      }
    }
  };
  try {
    await fetch('/api/chaos', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
  } catch (e) {}
}

async function resetChaos() {
  if (!selectedBed) return;
  try {
    const res = await fetch('/api/chaos/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({patient_id: selectedBed})
    });
    const data = await res.json();
    applyChaosToInputs(data.chaos);
  } catch (e) {}
}

async function pollState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    const beds = Object.keys(data.beds || {});
    if (beds.length === 0) return;

    const sel = document.getElementById('bedSelect');
    if (beds.length > 1) {
      sel.style.display = 'inline-block';
      if (sel.options.length !== beds.length) {
        sel.innerHTML = beds.map(b => `<option value="${b}">${b}</option>`).join('');
      }
    }
    if (!selectedBed || !data.beds[selectedBed]) {
      selectedBed = beds[0];
      loadChaosForBed(selectedBed);
    }

    const b = data.beds[selectedBed];
    document.getElementById('bedTitle').textContent = selectedBed;
    document.getElementById('bedMeta').textContent = `Broker ${data.broker} | ${data.provider}`;

    const hrEl = document.getElementById('hrVal');
    hrEl.textContent = b.hr !== null ? b.hr : '--';
    const hrBad = b.hr !== null && (b.hr < 50 || b.hr > 130);
    hrEl.style.color = hrBad ? '#ef4444' : '#10b981';
    document.getElementById('hrSub').textContent = b.hr_valid ? 'Plausible (60 - 100 bpm)' : 'Out of range';

    const spEl = document.getElementById('spo2Val');
    spEl.textContent = b.spo2 !== null ? b.spo2 : '--';
    const spBad = b.spo2 !== null && (b.spo2 < 90 || b.spo2 > 100);
    spEl.style.color = spBad ? '#ef4444' : '#3b82f6';
    document.getElementById('spo2Sub').textContent = b.spo2_valid ? 'Plausible (95 - 100 %)' : 'Out of range';

    const isAlert = b.arrhythmia || !b.hr_valid || !b.spo2_valid;
    const pill = document.getElementById('alarmPill');
    if (b.arrhythmia) {
      pill.className = 'pill danger';
      pill.textContent = 'ARRHYTHMIA ALARM';
    } else if (!b.hr_valid || !b.spo2_valid) {
      pill.className = 'pill warn';
      pill.textContent = 'SENSOR FAULT';
    } else if (b.offloaded) {
      pill.className = 'pill warn';
      pill.textContent = 'OFFLOADED';
    } else {
      pill.className = 'pill normal';
      pill.textContent = 'NORMAL';
    }

    const rEl = document.getElementById('rhythmVal');
    rEl.textContent = b.arrhythmia ? 'Arrhythmia' : 'Normal Sinus';
    rEl.style.color = b.arrhythmia ? '#ef4444' : '#ededef';

    document.getElementById('probVal').textContent = (b.anomaly_prob * 100).toFixed(1) + '%';
    document.getElementById('infVal').textContent = b.inference_ms.toFixed(2) + ' ms';
    document.getElementById('provSub').textContent = b.offloaded ? 'Offloaded to Gateway' : data.provider;
    document.getElementById('routeVal').textContent = b.relay_b ? 'Relay B' : 'Direct';
    document.getElementById('pktSub').textContent = b.packets + ' packets';

    const links = b.sensor_links || {};
    const ecgL = links.Agent_ECG || {latency_ms: 0, loss_pct: 0};
    const spL = links.Agent_SpO2 || {latency_ms: 0, loss_pct: 0};
    const hrL = links.Agent_HR || {latency_ms: 0, loss_pct: 0};

    document.getElementById('ecgMeta').textContent = `250 Hz | ${ecgL.latency_ms.toFixed(0)} ms`;
    document.getElementById('spo2Link').textContent = `${spL.latency_ms.toFixed(0)} ms`;
    document.getElementById('hrLink').textContent = `${hrL.latency_ms.toFixed(0)} ms`;

    document.getElementById('m_ecg').textContent = `${ecgL.latency_ms.toFixed(0)} ms | ${ecgL.loss_pct.toFixed(0)}% loss`;
    document.getElementById('m_spo2').textContent = `${spL.latency_ms.toFixed(0)} ms | ${spL.loss_pct.toFixed(0)}% loss`;
    document.getElementById('m_hr').textContent = `${hrL.latency_ms.toFixed(0)} ms | ${hrL.loss_pct.toFixed(0)}% loss`;
    document.getElementById('m_node').textContent = b.offloaded ? 'Offloaded' : 'Local GPU';

    if (!editingChaos && b.chaos) {
      applyChaosToInputs(b.chaos);
    }

    if (b.ecg_wave && b.ecg_wave.length > 0) {
      lastWave = b.ecg_wave;
    }
    drawECG(lastWave, b.arrhythmia);
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


def make_bedside_handler(engines_dict, meta_info):
    class BedsideMonitorHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send_json(self, obj, code=200):
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                beds_payload = {}
                for pid, eng in engines_dict.items():
                    with eng.lock:
                        beds_payload[pid] = {
                            "hr": eng.latest_hr,
                            "spo2": eng.latest_spo2,
                            "hr_valid": eng.hr_plausible,
                            "spo2_valid": eng.spo2_plausible,
                            "anomaly_prob": round(eng.last_prob, 4),
                            "arrhythmia": eng.last_arrhythmia,
                            "inference_ms": round(eng.last_inf_ms, 2),
                            "offloaded": eng.last_offloaded,
                            "relay_b": eng.last_relay_b,
                            "packets": eng.packets_processed,
                            "ecg_wave": eng.latest_ecg_wave[-250:] if eng.latest_ecg_wave else [],
                            "sensor_links": eng.sensor_links,
                            "chaos": read_patient_chaos(pid),
                        }
                self._send_json(
                    {
                        "broker": meta_info["broker"],
                        "provider": meta_info["provider"],
                        "beds": beds_payload,
                    }
                )
            elif self.path.startswith("/api/chaos"):
                bed_id = next(iter(engines_dict.keys()), "Bed_1")
                if "bed=" in self.path:
                    bed_id = self.path.split("bed=", 1)[1].split("&")[0]
                self._send_json(read_patient_chaos(bed_id))
            else:
                body = BEDSIDE_HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                data = json.loads(raw)
            except Exception:
                self._send_json({"ok": False}, code=400)
                return

            if self.path.startswith("/api/chaos/reset"):
                pid = data.get("patient_id") or next(iter(engines_dict.keys()), "Bed_1")
                clean = default_bed_chaos()
                write_patient_chaos(pid, clean)
                self._send_json({"ok": True, "chaos": clean})
            elif self.path.startswith("/api/chaos"):
                pid = data.get("patient_id") or next(iter(engines_dict.keys()), "Bed_1")
                chaos_cfg = data.get("chaos", default_bed_chaos())
                write_patient_chaos(pid, chaos_cfg)
                self._send_json({"ok": True, "chaos": chaos_cfg})
            else:
                self._send_json({"ok": False}, code=404)

    return BedsideMonitorHandler


def start_bedside_web_server(engines_dict, meta_info, web_port=8080):
    handler_cls = make_bedside_handler(engines_dict, meta_info)
    try:
        server = ThreadingHTTPServer(("0.0.0.0", web_port), handler_cls)
        server.daemon_threads = True
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[BEDSIDE MONITOR] Live at http://localhost:{web_port}")
        return server
    except Exception as e:
        print(f"[BEDSIDE MONITOR] Could not bind port {web_port}: {e}")
        return None


class JetsonEdgeBrain:
    def __init__(self, patient_id, broker, port, session, active_provider):
        self.patient_id = patient_id
        self.broker = broker
        self.port = port
        self.session = session
        self.active_provider = active_provider
        self.input_name = self.session.get_inputs()[0].name if self.session else "ecg_input"

        self.latest_hr = None
        self.latest_spo2 = None
        self.hr_plausible = True
        self.spo2_plausible = True
        self.latest_ecg_wave = []
        self.last_prob = 0.0
        self.last_arrhythmia = False
        self.last_inf_ms = 0.0
        self.last_offloaded = False
        self.last_relay_b = False
        self.packets_processed = 0
        self.sensor_links = {
            "Agent_ECG": {"latency_ms": 12.0, "loss_pct": 0.0, "last_seq": None, "rx": 0, "lost": 0},
            "Agent_SpO2": {"latency_ms": 10.0, "loss_pct": 0.0, "last_seq": None, "rx": 0, "lost": 0},
            "Agent_HR": {"latency_ms": 10.0, "loss_pct": 0.0, "last_seq": None, "rx": 0, "lost": 0},
        }
        self.lock = threading.Lock()
        self.client = None

    def start(self):
        cid = f"Jetson_Edge_Brain_{self.patient_id}_{os.getpid()}_{random.randint(100, 999)}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        else:
            self.client = mqtt.Client(client_id=cid)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass

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
            now = time.time()
            payload = json.loads(msg.payload.decode("utf-8"))
            agent_id = payload.get("agent_id", "")

            if topic == "hospital/discovery/probe":
                self.send_discovery_announcement()
                return

            if topic == f"hospital/{self.patient_id}/control/admit":
                print(f"[EDGE BRAIN:{self.patient_id}] Officially admitted by Nurse Station to ICU Ward!")
                return

            if topic == f"hospital/{self.patient_id}/control/offload":
                with self.lock:
                    self.latest_hr = hr_val
                    self.hr_plausible = 35 <= hr_val <= 210
                return

            if agent_id == "Agent_SpO2":
                spo2_val = int(payload.get("value", 98))
                with self.lock:
                    self.latest_spo2 = spo2_val
                    self.spo2_plausible = 65 <= spo2_val <= 100
                return

            if agent_id == "Agent_ECG":
                ecg_wave = payload.get("ecg_wave", [])
                if len(ecg_wave) != 250:
                    return

                bed_chaos = read_patient_chaos(self.patient_id)
                orin_chaos = bed_chaos.get("Orin_Link", {})
                overload_gpu = bool(orin_chaos.get("overload_gpu", False))
                use_relay_b = bool(orin_chaos.get("use_relay_b", False))

                anomaly_prob, inference_ms = self._run_inference(ecg_wave, overload_gpu=overload_gpu)
                is_arrhythmia = anomaly_prob > 0.50

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
                    "seq": payload.get("seq", 0),
                    "inference_ms": round(inference_ms, 3),
                    "anomaly_prob": round(anomaly_prob, 4),
                    "is_arrhythmia": is_arrhythmia,
                    "execution_provider": self.active_provider,
                    "gpu_overloaded": overload_gpu,
                    "link_congested": use_relay_b,
                    "alert_route": "Relay_Agent_B" if use_relay_b else "Direct_Edge_Link",
                }
                self.client.publish(
                    f"hospital/{self.patient_id}/edge/summary", json.dumps(summary_msg), qos=0
                )

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
    if ort is None or not os.path.exists(model_path):
        return None, "NumPyFallback"

    avail = ort.get_available_providers()
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    selected_providers = [p for p in preferred if p in avail] or avail

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        session = ort.InferenceSession(model_path, sess_options=opts, providers=selected_providers)
        active_provider = session.get_providers()[0]
    except Exception:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        active_provider = "CPUExecutionProvider"

    dummy = np.zeros((1, 1, 250), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    for _ in range(3):
        session.run(None, {input_name: dummy})

    return session, active_provider


def main():
    parser = argparse.ArgumentParser(description="Jetson Bedside GPU Edge Brain & Local Monitor")
    parser.add_argument("--patient-id", type=str, default="Bed_1", help="Patient/Bed ID for this Jetson")
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional list of patient IDs to run on this node",
    )
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT Broker IP")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker Port")
    parser.add_argument("--web-port", type=int, default=8080, help="Local Bedside Monitor HTTP Port")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to ONNX model")
    args = parser.parse_args()

    patient_list = args.patients if args.patients else [args.patient_id]
    session, active_provider = create_shared_onnx_session(args.model)

    engines = {}
    for pid in patient_list:
        eng = JetsonEdgeBrain(pid, args.broker, args.port, session, active_provider)
        eng.start()
        engines[pid] = eng

    meta_info = {"broker": f"{args.broker}:{args.port}", "provider": active_provider}
    web_server = start_bedside_web_server(engines, meta_info, web_port=args.web_port)

    stop_event = threading.Event()

    def sig_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    while not stop_event.is_set():
        time.sleep(0.5)

    if web_server:
        web_server.shutdown()
    for eng in engines.values():
        eng.stop()


if __name__ == "__main__":
    main()
