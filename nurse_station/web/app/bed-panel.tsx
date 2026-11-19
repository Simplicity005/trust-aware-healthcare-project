import Link from "next/link";
import type { ReactNode } from "react";
import { type Bed, bedStatus, formatReading, sensors, wardURL } from "./telemetry";

export function Badge({ children, tone = "muted" }: { children: ReactNode; tone?: string }) {
  return <span className={`badge ${tone}`}><span className="status-dot" />{children}</span>;
}

function Waveform({ bed }: { bed: Bed }) {
  const signal = bed.ecg_wave;
  const extent = Math.max(1, ...signal.map(Math.abs));
  const points = signal.map((sample, index) => {
    const x = index * 600 / Math.max(1, signal.length - 1);
    return `${x.toFixed(1)},${(56 - sample * 46 / extent).toFixed(1)}`;
  }).join(" ");
  const red = bed.agents.Agent_ECG.status === "ISOLATED" || bed.edge.local_bed_alarm;
  return (
    <div className={`wave-grid ${red ? "wave-alert" : ""}`}>
      <svg aria-label="Latest ECG waveform" role="img" viewBox="0 0 600 112" preserveAspectRatio="none">
        <polyline points={points} fill="none" stroke="currentColor" strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  );
}

function TrustHistory({ bed }: { bed: Bed }) {
  const history = bed.history;
  return (
    <section className="panel chart-panel">
      <div className="section-heading"><h2>Sensor trust history</h2><span className="secondary">Latest {history.length} samples</span></div>
      {history.length > 1 ? <>
        <div className="history-chart">
          <div className="axis-labels"><span>1.0</span><span>0.5</span><span>0.0</span></div>
          <svg role="img" aria-label="Recent trust scores for all six sensors" viewBox="0 0 600 104" preserveAspectRatio="none">
            {[2, 52, 102].map((y) => <line key={y} x1="0" x2="600" y1={y} y2={y} stroke="#293a4c" />)}
            {sensors.map((sensor) => <polyline key={sensor.id}
              points={history.map((sample, index) => `${600 * index / (history.length - 1)},${102 - 100 * sample[sensor.id]}`).join(" ")}
              stroke={sensor.color} vectorEffect="non-scaling-stroke" fill="none" strokeWidth="1.6" />)}
          </svg>
        </div>
        <div className="history-legend secondary">{sensors.map((sensor) => <span key={sensor.id}>
          <i style={{ background: sensor.color }} />{sensor.short}
        </span>)}</div>
      </> : <p className="empty">Waiting for trust samples…</p>}
    </section>
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
    <article className="panel bed-card">
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
          return <div key={sensor.id}>
            <dt>{sensor.short}</dt>
            <dd className={agent.status === "ISOLATED" ? "red-text" : ""}>
              {agent.status === "ISOLATED" ? "Isolated" : formatReading(agent.value)}
            </dd>
          </div>;
        })}
      </dl>
      <dl className="trust-strip" aria-label="Sensor trust scores">
        {sensors.map((sensor) => {
          const agent = bed.agents[sensor.id];
          const tone = agent.status === "ISOLATED" ? "red" : agent.status === "SUSPICIOUS" ? "amber" : "green";
          return <div key={sensor.id} title={agent.status}><dt>{sensor.short}</dt>
            <dd><span className={`status-dot ${tone}`} />{agent.trust.toFixed(2)}</dd></div>;
        })}
      </dl>
      <div className="card-footer">
        <span className="secondary">{bed.compute_mode}</span>
        <Link className="button button-small" href={wardURL(department, bed.patient_id)} prefetch={false}>Open bed</Link>
      </div>
    </article>
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
  const rhythm = ecg.status === "ISOLATED" ? "Signal unreliable" :
    bed.edge.local_bed_alarm ? "Ventricular arrhythmia" : "Normal sinus rhythm";
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
          <p>{bed.diagnosis} · Admitted {bed.admitted}</p>
        </div>
        <Badge tone={status.tone}>{status.label}</Badge>
      </section>
      <div className="detail-grid">
        <section className="panel chart-panel">
          <div className="section-heading"><h2>ECG lead II</h2><span className="secondary">{rhythm}</span></div>
          <Waveform bed={bed} />
          <div className="chart-footer"><span>Arrhythmia risk <strong>{((bed.edge.arrhythmia_prob ?? 0) * 100).toFixed(1)}%</strong></span></div>
        </section>
        <section className="panel routing">
          <div className="section-heading"><h2>Compute &amp; routing</h2></div>
          <dl>
            <div><dt>Compute</dt><dd>{bed.compute_mode}</dd></div>
            {bed.edge.exec_provider && <div><dt>Provider</dt><dd>{bed.edge.exec_provider}</dd></div>}
            <div><dt>Edge inference</dt><dd>{(bed.edge.inf_ms ?? 0).toFixed(2)} ms</dd></div>
            {bed.gateway_cpu_inf_ms > 0 && <div><dt>Gateway fallback</dt><dd>{bed.gateway_cpu_inf_ms.toFixed(2)} ms</dd></div>}
            <div><dt>Alert path</dt><dd>{bed.alert_route}</dd></div>
          </dl>
        </section>
      </div>
      <section>
        <div className="section-heading standalone-heading"><h2>Sensor trust</h2></div>
        <div className="sensor-grid">
          {sensors.map((sensor) => {
            const agent = bed.agents[sensor.id];
            const tone = agent.status === "ISOLATED" ? "red" : agent.status === "SUSPICIOUS" ? "amber" : "green";
            const reading = formatReading(agent.value);
            return <article className="panel sensor-card" key={sensor.id}>
              <div className="card-heading"><h3>{sensor.label}</h3><Badge tone={tone}>{agent.status.toLowerCase()}</Badge></div>
              <div className="sensor-reading">
                {agent.status === "ISOLATED" ? "Signal excluded" : sensor.id === "Agent_ECG" ? "Signal available" : reading}
                {agent.status !== "ISOLATED" && reading !== "—" && sensor.unit && <small>{sensor.unit}</small>}
                <strong>{agent.trust.toFixed(2)}</strong>
              </div>
              <progress className={tone} aria-label={`${sensor.label} trust score`} value={agent.trust} max={1} />
              <dl className="sensor-components">
                <div><dt>Latency</dt><dd>{agent.s_l.toFixed(2)}</dd></div>
                <div><dt>Jitter</dt><dd>{agent.s_j.toFixed(2)}</dd></div>
                <div><dt>Delivery</dt><dd>{agent.s_p.toFixed(2)}</dd></div>
                <div><dt>Validity</dt><dd>{agent.s_d.toFixed(2)}</dd></div>
              </dl>
              <p className="secondary">{agent.packets_received} received · {agent.packets_missed} missed</p>
            </article>;
          })}
        </div>
      </section>
      <TrustHistory bed={bed} />
      <EventLog title="Bed activity" entries={logs} />
      <div className="detail-actions">
        <button type="button" className="button danger" onClick={onRelease} disabled={releasing}>Release bed</button>
      </div>
    </div>
  );
}
