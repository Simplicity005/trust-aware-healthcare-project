"use client";

import Link from "next/link";
import { useEffect, useRef, type ReactNode } from "react";
import { type Bed, bedStatus, formatReading, sensors, wardURL } from "./telemetry";

export function Badge({ children, tone = "muted" }: { children: ReactNode; tone?: string }) {
  return <span className={`badge ${tone}`}><span className="status-dot" />{children}</span>;
}

type RingBuffer = {
  rate: number;
  end: number;
  samples: Float32Array;
  stamps: Float64Array;
};

const SWEEP_SECONDS = 6;
const PLAYBACK_DELAY = 0.8;
const ECG_MIN = -1.35;
const ECG_MAX = 1.55;

function interpolateSample(buffer: RingBuffer, time: number) {
  const index = time * buffer.rate;
  const sample = Math.floor(index);
  const len = buffer.samples.length;
  const slot = ((sample % len) + len) % len;
  const next = (slot + 1) % len;
  if (buffer.stamps[slot] !== sample) return NaN;
  if (buffer.stamps[next] !== sample + 1) return index - sample < 0.5 ? buffer.samples[slot] : NaN;
  const value = buffer.samples[slot];
  return value + (buffer.samples[next] - value) * (index - sample);
}

function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number, seconds: number) {
  ctx.fillStyle = "#030604";
  ctx.fillRect(0, 0, width, height);
  ctx.lineWidth = 1;
  for (let step = 0; step <= seconds * 5; step++) {
    const x = (step * width) / (seconds * 5);
    ctx.strokeStyle = step % 5 === 0 ? "#19251e" : "#0d1510";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  ctx.strokeStyle = "#152019";
  for (let row = 1; row < 5; row++) {
    const y = (row * height) / 5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

export function Waveform({ bed }: { bed: Bed }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bufferRef = useRef<RingBuffer | null>(null);
  const clockRef = useRef<{ offset?: number; sweepOrigin?: number }>({});
  const ecgRemoved = Boolean(bed.agents.Agent_ECG.removed);
  const red = ecgRemoved || bed.agents.Agent_ECG.status === "ISOLATED" || Boolean(bed.edge.local_bed_alarm);
  const redRef = useRef(red);
  redRef.current = red;

  useEffect(() => {
    if (ecgRemoved || !bed.online) {
      bufferRef.current = null;
      clockRef.current = {};
      return;
    }
    const wave = bed.ecg_wave;
    if (!Array.isArray(wave) || !wave.length) return;
    const rate = Number.isInteger(bed.ecg_rate) && (bed.ecg_rate ?? 0) > 0 ? (bed.ecg_rate as number) : 250;
    const timestamp = Number.isFinite(bed.ecg_end_ts) && (bed.ecg_end_ts ?? 0) > 0
      ? (bed.ecg_end_ts as number)
      : Date.now() / 1000;

    const end = Math.round(timestamp * rate);
    const previous = bufferRef.current;
    if (previous && previous.rate === rate && end <= previous.end) return;

    const nowSec = performance.now() / 1000;
    const offset = timestamp - nowSec;
    const clock = clockRef.current;
    if (clock.offset === undefined || Math.abs(offset - clock.offset) > 2) {
      clock.offset = offset;
      clock.sweepOrigin = Math.floor(timestamp - PLAYBACK_DELAY);
    } else {
      clock.offset += Math.max(-0.015, Math.min(0.015, (offset - clock.offset) * 0.15));
    }

    let buffer = bufferRef.current;
    if (!buffer || buffer.rate !== rate) {
      const capacity = (SWEEP_SECONDS + 3) * rate;
      buffer = {
        rate,
        end: 0,
        samples: new Float32Array(capacity),
        stamps: new Float64Array(capacity),
      };
      bufferRef.current = buffer;
    }

    const len = buffer.samples.length;
    const first = end - wave.length + 1;

    for (let sample = Math.max(first, buffer.end + 1); sample <= end; sample++) {
      const slot = ((sample % len) + len) % len;
      const value = wave[sample - first];
      buffer.samples[slot] = Number.isFinite(value) ? value : NaN;
      buffer.stamps[slot] = sample;
    }
    buffer.end = Math.max(buffer.end, end);
  }, [bed.ecg_wave, bed.ecg_end_ts, bed.ecg_rate, bed.online, ecgRemoved]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const { target, contentRect } of entries) {
        const ratio = window.devicePixelRatio || 1;
        (target as HTMLCanvasElement).width = Math.max(1, Math.round(contentRect.width * ratio));
        (target as HTMLCanvasElement).height = Math.max(1, Math.round(contentRect.height * ratio));
      }
    });
    resizeObserver.observe(canvas);

    let frameId = 0;
    const render = (now: number) => {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.width / ratio;
      const height = canvas.height / ratio;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      drawGrid(ctx, width, height, SWEEP_SECONDS);

      const buffer = bufferRef.current;
      const { offset, sweepOrigin } = clockRef.current;
      if (buffer && offset !== undefined && sweepOrigin !== undefined) {
        const time = now / 1000 + offset - PLAYBACK_DELAY;
        const sweptSeconds = (((time - sweepOrigin) % SWEEP_SECONDS) + SWEEP_SECONDS) % SWEEP_SECONDS;
        const start = time - sweptSeconds;
        const head = (sweptSeconds / SWEEP_SECONDS) * width;
        const gap = 14;

        ctx.strokeStyle = redRef.current ? "#ff5d5d" : "#45e985";
        ctx.lineWidth = 1.7;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        let drawing = false;
        for (let x = 0; x <= width; x += 0.75) {
          const ahead = x > head ? x - head : x + width - head;
          if (ahead < gap) {
            drawing = false;
            continue;
          }
          const sampleTime = start + (x / width) * SWEEP_SECONDS - (x > head ? SWEEP_SECONDS : 0);
          const value = interpolateSample(buffer, sampleTime);
          if (!Number.isFinite(value)) {
            drawing = false;
            continue;
          }
          const y = height * (0.91 - (0.76 * (value - ECG_MIN)) / (ECG_MAX - ECG_MIN));
          if (drawing) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
          drawing = true;
        }
        ctx.stroke();

        const leading = interpolateSample(buffer, time);
        if (Number.isFinite(leading)) {
          const y = height * (0.91 - (0.76 * (leading - ECG_MIN)) / (ECG_MAX - ECG_MIN));
          ctx.fillStyle = ctx.strokeStyle;
          ctx.beginPath();
          ctx.arc(head, y, 1.7, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      frameId = window.requestAnimationFrame(render);
    };

    frameId = window.requestAnimationFrame(render);
    return () => {
      resizeObserver.disconnect();
      window.cancelAnimationFrame(frameId);
    };
  }, []);

  return (
    <div className={`wave-grid ${red ? "wave-alert" : ""}`}>
      <canvas ref={canvasRef} aria-label="Live bedside ECG waveform" />
    </div>
  );
}

export function EventLog({ title, entries }: { title: string; entries: string[] }) {
  return (
    <section className="panel log">
      <div className="section-heading"><h2>{title}</h2><span className="secondary">Latest first</span></div>
      {entries.length ? <ul>{entries.map((entry, index) => {
        const match = entry.match(/^\[([^\]]+)\]\s*(.*)$/);
        return <li key={`${entry}-${index}`}>{match && <time>{match[1]}</time>}<span>{match?.[2] || entry}</span></li>;
      })}</ul> : <p className="empty">No events recorded.</p>}
    </section>
  );
}

export function BedCard({ bed, department }: { bed: Bed; department: string }) {
  const status = bedStatus(bed);
  const vitals = sensors.slice(1);
  return (
    <Link className="panel bed-card" href={wardURL(department, bed.patient_id)} prefetch={false}>
      <div className="card-heading">
        <div>
          <p className="room-label">{bed.department} · {bed.room_only}</p>
          <h3>{bed.patient_name}</h3>
        </div>
        <Badge tone={status.tone}>{status.label}</Badge>
      </div>
      <p className="diagnosis">{bed.diagnosis}</p>
      <Waveform bed={bed} />
      <dl className="vital-grid">
        {vitals.map((sensor) => {
          const agent = bed.agents[sensor.id];
          const isRem = Boolean(agent.removed);
          const receiving = bed.online && agent.packets_received > 0;
          const pct = Math.round(agent.trust * 100);
          const tone = isRem || agent.status === "ISOLATED" ? "red" : agent.status === "SUSPICIOUS" ? "amber" : "green";
          return (
            <div key={sensor.id}>
              <dt>
                <span>{sensor.short}</span>
                {(receiving || isRem) && <small className={`vital-trust ${tone}`} title={isRem ? `${sensor.label} removed` : `${sensor.label} trust: ${pct}%`}>{isRem ? "0%" : `${pct}%`}</small>}
              </dt>
              <dd className={isRem || agent.status === "ISOLATED" ? "red-text" : ""}>
                {isRem ? "Removed" : !receiving ? "—" : agent.status === "ISOLATED" ? "Isolated" : formatReading(agent.value)}
              </dd>
            </div>
          );
        })}
      </dl>
    </Link>
  );
}

export function BedPanel({ bed, onRelease, releasing, logs }: {
  bed: Bed;
  onRelease: () => void;
  releasing: boolean;
  logs: string[];
}) {
  const status = bedStatus(bed);
  const ecg = bed.agents.Agent_ECG;
  const ecgRemoved = Boolean(ecg.removed);
  const ecgPct = Math.round(ecg.trust * 100);
  const ecgTone = ecgRemoved || ecg.status === "ISOLATED" ? "red" : ecg.status === "SUSPICIOUS" ? "amber" : "green";
  const hasECG = bed.online && bed.ecg_wave.length > 0;
  const rhythm = !bed.online ? "Bedside offline" : ecgRemoved ? "ECG sensor removed" : !hasECG ? "Waiting for ECG" : ecg.status === "ISOLATED" ? "Signal unreliable" :
    bed.edge.local_bed_alarm ? "Ventricular arrhythmia" : "Receiving ECG";
  const vitals = sensors.slice(1);

  return (
    <div className="bed-detail">
      {bed.emergency.length > 0 && <div className="emergency-detail" role="alert">
        <strong>Active clinical emergency</strong>
        <p>{bed.emergency.join(" · ")}</p>
      </div>}
      <section className="panel detail-header">
        <div>
          <p className="room-label">{bed.room_only} · {bed.department} · {bed.patient_id}</p>
          <h2>{bed.patient_name}</h2>
          <p>{bed.diagnosis && `${bed.diagnosis} · `}Admitted {bed.admitted}</p>
        </div>
        <Badge tone={status.tone}>{status.label}</Badge>
      </section>
      <section className="panel chart-panel">
        <div className="section-heading">
          <h2>ECG lead II {(hasECG || ecgRemoved) && <small className={`vital-trust ${ecgTone}`}>{ecgRemoved ? "Removed" : `${ecgPct}%`}</small>}</h2>
          <span className="secondary">{rhythm}</span>
        </div>
        <Waveform bed={bed} />
      </section>
      <section className="panel detail-vitals-panel">
        <dl className="vital-grid detail-vital-grid">
          {vitals.map((sensor) => {
            const agent = bed.agents[sensor.id];
            const isRem = Boolean(agent.removed);
            const receiving = bed.online && agent.packets_received > 0;
            const pct = Math.round(agent.trust * 100);
            const tone = isRem || agent.status === "ISOLATED" ? "red" : agent.status === "SUSPICIOUS" ? "amber" : "green";
            const reading = formatReading(agent.value);
            return (
              <div key={sensor.id}>
                <dt>
                  <span>{sensor.short}</span>
                  {(receiving || isRem) && <small className={`vital-trust ${tone}`}>{isRem ? "0%" : `${pct}%`}</small>}
                </dt>
                <dd className={isRem || agent.status === "ISOLATED" ? "red-text" : ""}>
                  {isRem ? "Removed" : !receiving ? "—" : agent.status === "ISOLATED" ? "Isolated" : reading}
                  {receiving && !isRem && agent.status !== "ISOLATED" && reading !== "—" && sensor.unit && <small className="vital-unit"> {sensor.unit}</small>}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>
      <EventLog title="Bed activity" entries={logs} />
      <div className="detail-actions">
        <button type="button" className="button danger" onClick={onRelease} disabled={releasing}>Release bed</button>
      </div>
    </div>
  );
}
