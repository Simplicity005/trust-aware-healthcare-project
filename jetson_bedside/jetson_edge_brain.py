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
  .sensor-metrics {
    font-size: 11px;
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .field {
    margin-top: 8px;
  }
  .field-top {
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
  <div class="container">
    <div class="topbar">
      <div class="topbar-left">
        <div class="bed-title" id="bedTitle">Bed_1</div>
        <div class="bed-meta" id="bedMeta">Jetson Bedside Node</div>
      </div>
      <div class="topbar-right">
        <select id="bedSelect" class="bed-select" onchange="switchBed(this.value)"></select>
        <div id="alarmPill" class="pill normal">NORMAL</div>
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

setInterval(pollState, 350);
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
        client.subscribe(f"hospital/{self.patient_id}/sensors/#")

    def _track_link(self, agent_id, payload, now):
        if agent_id not in self.sensor_links:
            return
        info = self.sensor_links[agent_id]
        sent_ts = float(payload.get("sent_ts", now))
        lat_ms = max(0.5, (now - sent_ts) * 1000.0)
        info["latency_ms"] = round(0.7 * info["latency_ms"] + 0.3 * lat_ms, 1)

        seq = int(payload.get("seq", 0))
        if info["last_seq"] is not None and seq > info["last_seq"]:
            gap = seq - info["last_seq"] - 1
            if gap > 0:
                info["lost"] += gap
        info["last_seq"] = seq
        info["rx"] += 1
        total = max(1, info["rx"] + info["lost"])
        info["loss_pct"] = round((info["lost"] / total) * 100.0, 1)

    def _run_inference(self, ecg_wave, overload_gpu=False):
        t0 = time.perf_counter()
        x = np.array(ecg_wave, dtype=np.float32).reshape(1, 1, 250)

        if self.session is not None:
            logits = self.session.run(None, {self.input_name: x})[0]
            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / np.sum(exp_scores)
            anomaly_prob = float(probs[0][1])
        else:
            peak_to_peak = float(np.ptp(x))
            anomaly_prob = 0.95 if peak_to_peak > 2.2 else 0.02

        if overload_gpu:
            time.sleep(0.065)

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return anomaly_prob, inference_ms

    def _on_message(self, client, userdata, msg):
        try:
            now = time.time()
            payload = json.loads(msg.payload.decode("utf-8"))
            agent_id = payload.get("agent_id", "")

            with self.lock:
                self._track_link(agent_id, payload, now)

            if agent_id == "Agent_HR":
                hr_val = int(payload.get("value", 75))
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

                with self.lock:
                    self.packets_processed += 1
                    self.latest_ecg_wave = ecg_wave
                    self.last_prob = anomaly_prob
                    self.last_arrhythmia = is_arrhythmia
                    self.last_inf_ms = inference_ms
                    self.last_offloaded = overload_gpu
                    self.last_relay_b = use_relay_b
                    hr_snap = self.latest_hr
                    spo2_snap = self.latest_spo2
                    hr_ok = self.hr_plausible
                    spo2_ok = self.spo2_plausible

                if is_arrhythmia or not hr_ok or not spo2_ok:
                    alert_msg = {
                        "patient_id": self.patient_id,
                        "timestamp": time.time(),
                        "seq": payload.get("seq", 0),
                        "arrhythmia_detected": is_arrhythmia,
                        "anomaly_probability": round(anomaly_prob, 4),
                        "inference_ms": round(inference_ms, 3),
                        "execution_provider": self.active_provider,
                        "hr": hr_snap,
                        "spo2": spo2_snap,
                        "hr_plausible": hr_ok,
                        "spo2_plausible": spo2_ok,
                        "routed_via": "Relay_Agent_B" if use_relay_b else "Direct_Edge_Link",
                    }
                    alert_topic = (
                        f"hospital/{self.patient_id}/alerts/relay_b"
                        if use_relay_b
                        else f"hospital/{self.patient_id}/alerts/critical"
                    )
                    self.client.publish(alert_topic, json.dumps(alert_msg), qos=0)

                summary_msg = {
                    "patient_id": self.patient_id,
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

        except Exception:
            pass


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
