"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState, type FormEvent } from "react";
import { Badge } from "../bed-panel";
import { requestAction, useTelemetry } from "../dashboard";
import { bedStatus, formatReading, sensors, type BedChaos, wardURL } from "../telemetry";

type SimState = {
  target_hr: number;
  target_spo2: number;
  target_co2: number;
  target_respiration: number;
  target_sys: number;
  target_dia: number;
  ecg_condition: string;
  delay_ms: number;
  drop_rate: number;
  overload_gpu: boolean;
  faults: Record<string, boolean>;
  removed: Record<string, boolean>;
};

const DEFAULT_SIM: SimState = {
  target_hr: 75,
  target_spo2: 98,
  target_co2: 38,
  target_respiration: 16,
  target_sys: 120,
  target_dia: 80,
  ecg_condition: "normal",
  delay_ms: 10,
  drop_rate: 0,
  overload_gpu: false,
  faults: {
    Agent_ECG: false,
    Agent_HR: false,
    Agent_SpO2: false,
    Agent_CO2: false,
    Agent_IBP: false,
    Agent_NIBP: false,
  },
  removed: {
    Agent_ECG: false,
    Agent_HR: false,
    Agent_SpO2: false,
    Agent_CO2: false,
    Agent_IBP: false,
    Agent_NIBP: false,
  },
};

function simFromChaos(chaos?: BedChaos): SimState {
  if (!chaos) return DEFAULT_SIM;
  const v = chaos.vitals || {};
  return {
    target_hr: v.target_hr ?? chaos.Agent_HR?.target_hr ?? 75,
    target_spo2: v.target_spo2 ?? chaos.Agent_SpO2?.target_spo2 ?? 98,
    target_co2: v.target_co2 ?? v.target_etco2 ?? chaos.Agent_CO2?.target_co2 ?? 38,
    target_respiration: v.target_respiration ?? 16,
    target_sys: v.target_sys ?? chaos.Agent_IBP?.target_sys ?? 120,
    target_dia: v.target_dia ?? chaos.Agent_IBP?.target_dia ?? 80,
    ecg_condition: v.ecg_condition ?? chaos.Agent_ECG?.condition ?? "normal",
    delay_ms: chaos.Agent_ECG?.delay_ms ?? 10,
    drop_rate: chaos.Agent_ECG?.drop_rate ?? 0,
    overload_gpu: Boolean(chaos.Orin_Link?.overload_gpu),
    faults: {
      Agent_ECG: Boolean(chaos.Agent_ECG?.malicious),
      Agent_HR: Boolean(chaos.Agent_HR?.malicious),
      Agent_SpO2: Boolean(chaos.Agent_SpO2?.malicious),
      Agent_CO2: Boolean(chaos.Agent_CO2?.malicious ?? chaos.Agent_EtCO2?.malicious),
      Agent_IBP: Boolean(chaos.Agent_IBP?.malicious),
      Agent_NIBP: Boolean(chaos.Agent_NIBP?.malicious),
    },
    removed: {
      Agent_ECG: Boolean(chaos.Agent_ECG?.removed),
      Agent_HR: Boolean(chaos.Agent_HR?.removed),
      Agent_SpO2: Boolean(chaos.Agent_SpO2?.removed),
      Agent_CO2: Boolean(chaos.Agent_CO2?.removed ?? chaos.Agent_EtCO2?.removed),
      Agent_IBP: Boolean(chaos.Agent_IBP?.removed),
      Agent_NIBP: Boolean(chaos.Agent_NIBP?.removed),
    },
  };
}

function SimulationContent() {
  const params = useSearchParams();
  const { snapshot, connection } = useTelemetry();
  const allBeds = Object.values(snapshot?.beds || {});
  const initialBed = params.get("bed") || "";
  const [selectedBedId, setSelectedBedId] = useState(initialBed);
  const [loadedBedId, setLoadedBedId] = useState("");
  const [sim, setSim] = useState<SimState>(DEFAULT_SIM);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const activeBed = allBeds.find((b) => b.patient_id === selectedBedId) || allBeds[0];

  useEffect(() => {
    if (!activeBed) return;
    if (selectedBedId !== activeBed.patient_id) {
      setSelectedBedId(activeBed.patient_id);
    }
    if (loadedBedId !== activeBed.patient_id) {
      setLoadedBedId(activeBed.patient_id);
      setSim(simFromChaos(activeBed.chaos));
    }
  }, [activeBed, selectedBedId, loadedBedId]);

  function flashNotice(text: string) {
    setNotice(text);
    window.setTimeout(() => setNotice((prev) => (prev === text ? "" : prev)), 2000);
  }

  async function applySimulation(next: SimState, targetId?: string) {
    const pid = targetId || activeBed?.patient_id;
    if (!pid) return;
    setSim(next);
    setError("");
    try {
      await requestAction("manipulate", {
        patient_id: pid,
        target_hr: next.target_hr,
        target_spo2: next.target_spo2,
        target_co2: next.target_co2,
        target_respiration: next.target_respiration,
        target_sys: next.target_sys,
        target_dia: next.target_dia,
        ecg_condition: next.ecg_condition,
        delay_ms: next.delay_ms,
        drop_rate: next.drop_rate,
        overload_gpu: next.overload_gpu,
        use_relay_b: next.overload_gpu,
        sensor_faults: {
          ...next.faults,
          Agent_ECG: next.faults.Agent_ECG || next.ecg_condition === "malicious",
        },
        removed_sensors: next.removed,
      });
      flashNotice("Updated simulation");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Failed to apply simulation parameters.");
    }
  }

  async function resetBed() {
    if (!activeBed) return;
    setBusy(true);
    setError("");
    try {
      await requestAction("manipulate", { patient_id: activeBed.patient_id, reset: true });
      setSim(DEFAULT_SIM);
      flashNotice("Reset to defaults");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Failed to reset bed parameters.");
    } finally {
      setBusy(false);
    }
  }

  function applyPreset(name: string) {
    const presets: Record<string, Partial<SimState>> = {
      healthy: {
        target_hr: 75,
        target_spo2: 98,
        target_co2: 38,
        target_respiration: 16,
        target_sys: 120,
        target_dia: 80,
        ecg_condition: "normal",
        delay_ms: 10,
        drop_rate: 0,
        overload_gpu: false,
        faults: { ...DEFAULT_SIM.faults },
        removed: { ...DEFAULT_SIM.removed },
      },
      arrhythmia: {
        target_hr: 142,
        target_spo2: 93,
        target_co2: 36,
        target_respiration: 22,
        target_sys: 110,
        target_dia: 70,
        ecg_condition: "arrhythmia",
      },
      hypoxia: {
        target_hr: 108,
        target_spo2: 84,
        target_co2: 48,
        target_respiration: 24,
        target_sys: 105,
        target_dia: 65,
        ecg_condition: "normal",
      },
      hypercapnia: {
        target_hr: 94,
        target_spo2: 91,
        target_co2: 62,
        target_respiration: 10,
        target_sys: 130,
        target_dia: 85,
        ecg_condition: "normal",
      },
      hypotension: {
        target_hr: 124,
        target_spo2: 93,
        target_co2: 28,
        target_respiration: 22,
        target_sys: 72,
        target_dia: 44,
        ecg_condition: "normal",
      },
      hypertension: {
        target_hr: 95,
        target_spo2: 98,
        target_co2: 41,
        target_respiration: 18,
        target_sys: 205,
        target_dia: 115,
        ecg_condition: "normal",
      },
    };
    const patch = presets[name];
    if (!patch) return;
    void applySimulation({ ...sim, ...patch });
  }

  async function triggerAlarm(active: boolean) {
    if (!activeBed) return;
    setBusy(true);
    setError("");
    try {
      await requestAction("alarm", { patient_id: activeBed.patient_id, active });
      const updated = active
        ? { ...sim, ecg_condition: "arrhythmia", target_hr: 165, target_spo2: 82 }
        : { ...sim, ecg_condition: "normal", target_hr: 75, target_spo2: 98 };
      setSim(updated);
      flashNotice(active ? "Emergency alarm triggered" : "Emergency alarm cleared");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Alarm update failed.");
    } finally {
      setBusy(false);
    }
  }

  async function triggerFailover() {
    setBusy(true);
    setError("");
    try {
      await requestAction("failover", {});
      flashNotice("Toggled station HA role");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Failover command failed.");
    } finally {
      setBusy(false);
    }
  }

  async function simulateNewBed(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const department = String(form.get("department"));
    const room = String(form.get("room")).trim() || null;
    setBusy(true);
    setError("");
    try {
      const result = await requestAction("simulate", { department, room });
      if (result.patient_id) {
        setSelectedBedId(result.patient_id);
        setLoadedBedId("");
        flashNotice(`Admitted simulated ${result.patient_id}`);
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not simulate a new bed.");
    } finally {
      setBusy(false);
    }
  }

  const connectionLabel = connection === "live" ? "Live telemetry" : connection === "connecting" ? "Connecting" : "Reconnecting";

  return (
    <main className="main sim-page">
      <header className="topbar">
        <div>
          <p className="room-label">Nurse Station Scenario &amp; Fault Controller</p>
          <h1>Simulation Control Center</h1>
        </div>
        <div className="header-actions">
          {notice && <Badge tone="green">{notice}</Badge>}
          <span role="status"><Badge tone={connection === "live" ? "green" : "amber"}>{connectionLabel}</Badge></span>
          {activeBed && (
            <Link className="button" href={wardURL(activeBed.department, activeBed.patient_id)} prefetch={false}>
              Open {activeBed.patient_id} view
            </Link>
          )}
          <Link className="button button-primary" href="/" prefetch={false}>
            Back to Nurse Station
          </Link>
        </div>
      </header>

      {error && (
        <div role="alert" className="error-box error-action">
          <span>{error}</span>
          <button className="button button-small" onClick={() => setError("")}>Dismiss</button>
        </div>
      )}

      {!snapshot && <div className="panel empty" role="status">Connecting to Nurse Station telemetry…</div>}

      {snapshot && (
        <div className="sim-layout">
          <section className="panel sim-section">
            <div className="section-heading">
              <div>
                <h2>Target Patient Bed</h2>
                {activeBed && <p className="secondary">{activeBed.patient_name} · {activeBed.department} · {activeBed.room_only} · {activeBed.diagnosis}</p>}
              </div>
              <div className="header-actions">
                {activeBed && <Badge tone={bedStatus(activeBed).tone}>{bedStatus(activeBed).label}</Badge>}
                <label className="inline-label">
                  <span>Select bed</span>
                  <select
                    value={activeBed?.patient_id || ""}
                    disabled={!allBeds.length}
                    onChange={(event) => {
                      setSelectedBedId(event.target.value);
                      setLoadedBedId("");
                    }}
                  >
                    {!allBeds.length && <option value="">No active beds</option>}
                    {allBeds.map((bed) => (
                      <option key={bed.patient_id} value={bed.patient_id}>
                        {bed.patient_id} · {bed.room_only} · {bed.patient_name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>

            {activeBed && (
              <dl className="vital-grid sim-live-grid">
                {sensors.map((sensor) => {
                  const agent = activeBed.agents[sensor.id];
                  const isRem = Boolean(agent.removed);
                  const pct = Math.round(agent.trust * 100);
                  const tone = isRem || agent.status === "ISOLATED" ? "red" : agent.status === "SUSPICIOUS" ? "amber" : "green";
                  const reading = sensor.id === "Agent_ECG"
                    ? (activeBed.edge.local_bed_alarm ? "Arrhythmia" : "Normal sinus")
                    : formatReading(agent.value);
                  return (
                    <div key={sensor.id}>
                      <dt>
                        <span>{sensor.short}</span>
                        <small className={`vital-trust ${tone}`}>{isRem ? "0%" : `${pct}%`}</small>
                      </dt>
                      <dd className={isRem || agent.status === "ISOLATED" ? "red-text" : ""}>
                        {isRem ? "Removed" : agent.status === "ISOLATED" ? "Isolated" : reading}
                      </dd>
                    </div>
                  );
                })}
              </dl>
            )}
          </section>

          {activeBed && (
            <>
              <section className="panel sim-section">
                <div className="section-heading">
                  <h2>Clinical Scenario &amp; Emergency Presets</h2>
                </div>
                <div className="header-actions">
                  <button type="button" className="button" onClick={() => applyPreset("healthy")}>Healthy baseline</button>
                  <button type="button" className="button danger" onClick={() => applyPreset("arrhythmia")}>Ventricular arrhythmia</button>
                  <button type="button" className="button" onClick={() => applyPreset("hypoxia")}>Critical hypoxia (Low O₂)</button>
                  <button type="button" className="button" onClick={() => applyPreset("hypercapnia")}>Severe hypercapnia (High CO₂)</button>
                  <button type="button" className="button danger" onClick={() => applyPreset("hypotension")}>Hypotensive shock</button>
                  <button type="button" className="button" onClick={() => applyPreset("hypertension")}>Hypertensive crisis</button>
                  <button type="button" className="button danger" disabled={busy} onClick={() => void triggerAlarm(true)}>Trigger emergency alarm</button>
                  <button type="button" className="button" disabled={busy} onClick={() => void triggerAlarm(false)}>Clear alarm</button>
                  <button type="button" className="button" disabled={busy} onClick={() => void resetBed()}>Reset defaults</button>
                </div>
              </section>

              <section className="panel sim-section">
                <div className="section-heading">
                  <h2>Physiological Vital Sign Parameters</h2>
                </div>
                <div className="sim-controls-grid">
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Heart rate</span><strong>{sim.target_hr} bpm</strong></div>
                    <input type="range" min={40} max={200} value={sim.target_hr}
                      onChange={(e) => void applySimulation({ ...sim, target_hr: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Oxygen saturation (SpO₂)</span><strong>{sim.target_spo2} %</strong></div>
                    <input type="range" min={70} max={100} value={sim.target_spo2}
                      onChange={(e) => void applySimulation({ ...sim, target_spo2: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>End-Tidal CO₂</span><strong>{sim.target_co2} mmHg</strong></div>
                    <input type="range" min={15} max={75} value={sim.target_co2}
                      onChange={(e) => void applySimulation({ ...sim, target_co2: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Respiration rate</span><strong>{sim.target_respiration} br/min</strong></div>
                    <input type="range" min={8} max={35} value={sim.target_respiration}
                      onChange={(e) => void applySimulation({ ...sim, target_respiration: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Systolic BP</span><strong>{sim.target_sys} mmHg</strong></div>
                    <input type="range" min={60} max={220} value={sim.target_sys}
                      onChange={(e) => void applySimulation({ ...sim, target_sys: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Diastolic BP</span><strong>{sim.target_dia} mmHg</strong></div>
                    <input type="range" min={30} max={130} value={sim.target_dia}
                      onChange={(e) => void applySimulation({ ...sim, target_dia: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Cardiac rhythm condition</span></div>
                    <select value={sim.ecg_condition}
                      onChange={(e) => void applySimulation({ ...sim, ecg_condition: e.target.value })}>
                      <option value="normal">Normal sinus rhythm</option>
                      <option value="arrhythmia">Ventricular arrhythmia</option>
                      <option value="malicious">Irregular / artifact signal</option>
                    </select>
                  </div>
                </div>
              </section>

              <section className="panel sim-section">
                <div className="section-heading">
                  <h2>Network, Sensor Trust &amp; Edge Compute Faults</h2>
                </div>
                <div className="sim-controls-grid">
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Sensor network delay</span><strong>{sim.delay_ms} ms</strong></div>
                    <input type="range" min={0} max={500} step={5} value={sim.delay_ms}
                      onChange={(e) => void applySimulation({ ...sim, delay_ms: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Sensor packet drop rate</span><strong>{Math.round(sim.drop_rate * 100)}%</strong></div>
                    <input type="range" min={0} max={0.9} step={0.05} value={sim.drop_rate}
                      onChange={(e) => void applySimulation({ ...sim, drop_rate: Number(e.target.value) })} />
                  </div>
                  <div className="sim-box">
                    <div className="sim-box-head"><span>Edge GPU offload &amp; routing</span></div>
                    <label className="sim-checkbox">
                      <input type="checkbox" checked={sim.overload_gpu}
                        onChange={(e) => void applySimulation({ ...sim, overload_gpu: e.target.checked })} />
                      <span>Overload Jetson GPU / Reroute via Relay B</span>
                    </label>
                  </div>
                </div>
                <div className="sim-subhead">Remove / Disconnect Sensors (Unplugs Sensor Feed)</div>
                <div className="sim-fault-grid">
                  {sensors.map((sensor) => (
                    <label key={`rem-${sensor.id}`} className="sim-checkbox sim-fault-card">
                      <input
                        type="checkbox"
                        checked={Boolean(sim.removed[sensor.id])}
                        onChange={(e) => {
                          const nextRemoved = { ...sim.removed, [sensor.id]: e.target.checked };
                          void applySimulation({ ...sim, removed: nextRemoved });
                        }}
                      />
                      <span>Remove {sensor.short} ({sensor.label})</span>
                    </label>
                  ))}
                </div>
                <div className="sim-subhead">Per-Sensor Spoofing / Fault Injection (Tests Trust Score Degradation &amp; Isolation)</div>
                <div className="sim-fault-grid">
                  {sensors.map((sensor) => (
                    <label key={sensor.id} className="sim-checkbox sim-fault-card">
                      <input
                        type="checkbox"
                        checked={Boolean(sim.faults[sensor.id])}
                        onChange={(e) => {
                          const nextFaults = { ...sim.faults, [sensor.id]: e.target.checked };
                          void applySimulation({ ...sim, faults: nextFaults });
                        }}
                      />
                      <span>Spoof {sensor.short} ({sensor.label})</span>
                    </label>
                  ))}
                </div>
              </section>
            </>
          )}

          <div className="detail-grid">
            <section className="panel sim-section">
              <div className="section-heading">
                <h2>Simulate New Bedside Device</h2>
              </div>
              <p className="secondary">Spawn a new simulated patient bed feed in any hospital department.</p>
              <form className="sim-admit-form" onSubmit={(e) => void simulateNewBed(e)}>
                <label>Department
                  <select name="department" defaultValue={snapshot.departments[0]?.[0]} required>
                    {snapshot.departments.map(([code, name]) => <option key={code} value={code}>{name}</option>)}
                  </select>
                </label>
                <label>Room <span className="secondary">(optional)</span>
                  <input name="room" placeholder="Auto-assigned" maxLength={60} />
                </label>
                <button className="button button-primary" disabled={busy} type="submit">
                  + Simulate new bed
                </button>
              </form>
            </section>

            <section className="panel sim-section">
              <div className="section-heading">
                <h2>High Availability (HA) Failover Simulation</h2>
                <Badge tone={snapshot.ha.failover_simulated ? "red" : snapshot.ha.is_primary ? "green" : "amber"}>
                  {snapshot.ha.failover_simulated ? "Failure simulated" : snapshot.ha.is_primary ? "Primary leader" : "Hot standby"}
                </Badge>
              </div>
              <p className="secondary">
                Station: <strong>{snapshot.ha.station_id.replaceAll("_", " ")}</strong> ·{" "}
                {snapshot.ha.total_stations_online} {snapshot.ha.total_stations_online === 1 ? "station" : "stations"} online
              </p>
              {snapshot.ha.failover_triggered && <p className="secondary">This station has been promoted to primary leader.</p>}
              <div style={{ marginTop: "0.75rem" }}>
                <button type="button" className="button" disabled={busy} onClick={() => void triggerFailover()}>
                  Test failover / switch station role
                </button>
              </div>
            </section>
          </div>
        </div>
      )}
    </main>
  );
}

export default function SimulationPage() {
  return (
    <Suspense fallback={<main className="loading">Opening Simulation Control Center…</main>}>
      <SimulationContent />
    </Suspense>
  );
}
