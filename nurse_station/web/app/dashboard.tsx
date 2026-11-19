"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import Admission, { type Action } from "./admission";
import { Badge, BedCard, BedPanel, EventLog } from "./bed-panel";
import { bedStatus, formatReading, type Bed, type Device, type Snapshot, wardURL } from "./telemetry";

const requestAction: Action = async (operation, data) => {
  const response = await fetch(`/api/${operation}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) {
    const detail = typeof result.detail === "string" ? result.detail : "Check the values and try again.";
    throw new Error(detail);
  }
  return result;
};
const probe = () => requestAction("probe", {});

function useTelemetry() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [connection, setConnection] = useState<"connecting" | "live" | "reconnecting">("connecting");

  useEffect(() => {
    const events = new EventSource("/api/events");
    let lastMessage = 0;
    events.onmessage = (event) => {
      try {
        setSnapshot(JSON.parse(event.data));
        setConnection("live");
        lastMessage = Date.now();
      } catch {
        setConnection("reconnecting");
      }
    };
    events.onerror = () => setConnection("reconnecting");
    const watchdog = window.setInterval(() => {
      if (lastMessage && Date.now() - lastMessage > 3500) setConnection("reconnecting");
    }, 1000);
    return () => {
      events.close();
      window.clearInterval(watchdog);
    };
  }, []);

  return { snapshot, connection };
}

function Directory({ snapshot, beds, selected, openAdmission, runAction, busy }: {
  snapshot: Snapshot;
  beds: Bed[];
  selected?: Bed;
  openAdmission: () => void;
  runAction: (operation: string, data: Record<string, string | boolean>) => void;
  busy: boolean;
}) {
  const [alarmBed, setAlarmBed] = useState("");
  const alarmTarget = beds.find((bed) => bed.patient_id === alarmBed)?.patient_id || beds[0]?.patient_id || "";
  const { ha } = snapshot;

  return <>
    <div className="station-card">
      <span className="secondary">Station role</span>
      <strong>{ha.station_id.replaceAll("_", " ")}</strong>
      <Badge tone={ha.failover_simulated ? "red" : ha.is_primary ? "green" : "amber"}>
        {ha.failover_simulated ? "Failure simulated" : ha.is_primary ? "Primary leader" : "Hot standby"}
      </Badge>
      <span className="secondary">{ha.total_stations_online} {ha.total_stations_online === 1 ? "station" : "stations"} online</span>
    </div>
    <div className="sidebar-directory">
      <h2>Room directory</h2>
      {snapshot.departments.map(([department, name]) => {
        const rooms = beds.filter((bed) => bed.department === department);
        if (!rooms.length) return null;
        return <div key={department} className="room-group">
          <Link className="department-link" href={wardURL(department)} prefetch={false} title={name}>{department}</Link>
          {rooms.map((bed) => {
            const status = bedStatus(bed);
            return <Link key={bed.patient_id} className="room-link" aria-current={selected?.patient_id === bed.patient_id ? "page" : undefined}
              href={wardURL(department, bed.patient_id)} prefetch={false}>
              <span className={`status-dot ${status.tone}`} title={status.label} />
              <span>{bed.room_only}<small>{bed.patient_name}</small></span>
            </Link>;
          })}
        </div>;
      })}
      {!beds.length && <p className="secondary">No active beds.</p>}
      {snapshot.unadded_jetsons.length > 0 && <button className="button sidebar-button" onClick={openAdmission}>
        Unassigned devices ({snapshot.unadded_jetsons.length})
      </button>}
    </div>
    <div className="station-tools">
      <h2>Station controls</h2>
      <details>
        <summary>High availability</summary>
        <div className="tool-content">
          {ha.failover_triggered && <p className="secondary">This station has been promoted to primary.</p>}
          {Object.values(ha.peers).map((peer) => <div className="secondary" key={peer.station_id}>
            <a href={`//${typeof window !== "undefined" ? window.location.hostname : "localhost"}:${peer.port}`}>{peer.station_id.replaceAll("_", " ")}</a> · {peer.role}
          </div>)}
          {!Object.keys(ha.peers).length && <p className="secondary">No peer stations detected.</p>}
          <button className="button sidebar-button" disabled={busy} onClick={() => runAction("failover", {})}>Test failover / switch role</button>
        </div>
      </details>
      <details>
        <summary>Emergency simulator</summary>
        <div className="tool-content">
          <label>Target bed
            <select value={alarmTarget} disabled={!beds.length} onChange={(event) => setAlarmBed(event.target.value)}>
              {!beds.length && <option value="">No active beds</option>}
              {beds.map((bed) => <option key={bed.patient_id} value={bed.patient_id}>{bed.patient_id} · {bed.room_only}</option>)}
            </select>
          </label>
          <button className="button danger sidebar-button" disabled={busy || !alarmTarget}
            onClick={() => runAction("alarm", { patient_id: alarmTarget, active: true })}>Trigger alarm</button>
          <button className="button sidebar-button" disabled={busy || !alarmTarget}
            onClick={() => runAction("alarm", { patient_id: alarmTarget, active: false })}>Clear alarm</button>
        </div>
      </details>
    </div>
  </>;
}

export default function Dashboard() {
  const params = useSearchParams();
  const { snapshot, connection } = useTelemetry();
  const [admission, setAdmission] = useState(false);
  const [scanDeadline, setScanDeadline] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const allBeds = Object.values(snapshot?.beds || {});
  const selectedId = params.get("view") === "detail" ? params.get("bed") : null;
  const selected = selectedId && snapshot && Object.hasOwn(snapshot.beds, selectedId)
    ? snapshot.beds[selectedId] : undefined;
  const requestedDepartment = params.get("dept");
  const department = snapshot?.departments.find(([code]) => code === requestedDepartment)?.[0] || "";
  const visibleBeds = department ? allBeds.filter((bed) => bed.department === department) : allBeds;
  const emergencies = allBeds.filter((bed) => bed.emergency.length > 0);

  const toggleScan = useCallback((active: boolean) => setScanDeadline(active ? Date.now() + 150_000 : null), []);
  function openAdmission() {
    setAdmission(true);
    toggleScan(true);
  }
  function closeAdmission() {
    setAdmission(false);
    toggleScan(false);
  }

  async function runAction(
    operation: string,
    data: Record<string, string | boolean | null>,
    onSuccess?: (result: { patient_id?: string }) => void,
  ) {
    setBusy(true);
    setError("");
    try {
      const result = await requestAction(operation, data);
      onSuccess?.(result);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The station command failed.");
    } finally {
      setBusy(false);
    }
  }

  function admit(device: Device | undefined, assignedDepartment: string, room: string | null) {
    const assignment = { department: assignedDepartment, room };
    const data = device ? { ...assignment, patient_id: device.patient_id } : assignment;
    void runAction(device ? "admit" : "simulate", data, ({ patient_id }) => {
      if (patient_id) window.history.pushState(null, "", wardURL(assignedDepartment, patient_id));
      closeAdmission();
    });
  }

  function release(bed: Bed) {
    if (!window.confirm(`Release ${bed.patient_id} from active monitoring?`)) return;
    void runAction("release", { patient_id: bed.patient_id }, () => window.history.pushState(null, "", wardURL(department)));
  }

  const connectionLabel = connection === "live" ? "Live telemetry" : connection === "connecting" ? "Connecting" : "Reconnecting";

  return (
    <div className="dashboard">
      <a className="skip-link" href="#content">Skip to dashboard</a>
      <aside className="sidebar">
        <Link className="brand" href="/" prefetch={false}>
          <span className="brand-icon" aria-hidden="true">+</span>
          <span>Nurse Station<small>Trust-Aware Healthcare</small></span>
        </Link>
        {snapshot && <Directory snapshot={snapshot} beds={allBeds} selected={selected}
          openAdmission={openAdmission} runAction={runAction} busy={busy} />}
      </aside>
      <main id="content" className="main">
        <header className="topbar">
          <div>
            <p className="room-label">{selected ? `${selected.department} · ${selected.room_only}` : "Central monitoring"}</p>
            <h1>{selected ? "Bed dashboard" : department ? `${department} overview` : "Ward overview"}</h1>
          </div>
          <div className="header-actions">
            <span role="status"><Badge tone={connection === "live" ? "green" : "amber"}>{connectionLabel}</Badge></span>
            <button className="button" disabled={!snapshot} onClick={openAdmission}>{scanDeadline ? "Rescan now" : "Rescan LAN"}</button>
            <button className="button button-primary" disabled={!snapshot} onClick={admission ? closeAdmission : openAdmission}>
              {admission ? "Close admission" : `+ Add new bed${snapshot?.unadded_jetsons.length ? ` (${snapshot.unadded_jetsons.length})` : ""}`}
            </button>
          </div>
        </header>

        {connection === "reconnecting" && <p className="connection-warning" role="status">
          {snapshot ? "Connection interrupted. Displayed readings may be out of date." : "Waiting for the Nurse Station service. Reconnecting automatically."}
        </p>}
        {error && <div role="alert" className="error-box error-action"><span>{error}</span>
          <button className="button button-small" onClick={() => setError("")}>Dismiss</button>
        </div>}

        {snapshot && <>
          {admission && <Admission snapshot={snapshot} scanDeadline={scanDeadline} onScan={toggleScan} probe={probe}
            onClose={closeAdmission} onAdmit={admit} busy={busy} />}
          <nav className="departments" aria-label="Hospital departments">
            <Link className="filter" aria-current={!department && !selectedId ? "page" : undefined} href="/" prefetch={false}>All wards <span>{allBeds.length}</span></Link>
            {snapshot.departments.map(([code, name]) => <Link className="filter" key={code} title={name} prefetch={false}
              aria-current={code === (selected?.department || department) ? "page" : undefined} href={wardURL(code)}>
              {code} <span>{allBeds.filter((bed) => bed.department === code).length}</span>
            </Link>)}
          </nav>

          {selectedId ? <>
            <div className="detail-nav">
              <Link className="button" href={wardURL(department)} prefetch={false}>Back to {department || "all wards"}</Link>
              <label>Switch bed
                <select value={selected?.patient_id || ""} onChange={(event) => {
                  const bed = snapshot.beds[event.target.value];
                  if (bed) window.history.pushState(null, "", wardURL(bed.department, bed.patient_id));
                }}>
                  {!selected && <option value="">Select a bed</option>}
                  {allBeds.map((bed) => <option key={bed.patient_id} value={bed.patient_id}>{bed.room_only} · {bed.patient_name}</option>)}
                </select>
              </label>
            </div>
            {selected ? <BedPanel key={selected.patient_id} bed={selected} releasing={busy} onRelease={() => release(selected)}
              logs={[...new Set([...selected.alert_log, ...snapshot.global_alert_log.filter((entry) => entry.includes(`[${selected.patient_id}]`) || entry.includes(`(${selected.patient_id})`))])]} />
              : <p className="panel empty">This bed is unavailable. Select an active bed or return to the ward.</p>}
          </> : <>
            <section className="ward-summary" aria-label="Ward status">
              <div className="panel metric"><span>Monitored beds</span><strong>{visibleBeds.length}</strong></div>
              <div className="panel metric"><span>Bedside feeds online</span><strong className="mint-text">{visibleBeds.filter((bed) => bed.online).length}</strong></div>
              <div className="panel metric"><span>Active emergencies</span><strong className={emergencies.length ? "red-text" : ""}>{emergencies.length}</strong></div>
            </section>
            <section aria-labelledby="bed-overview">
              <div className="section-heading standalone-heading"><h2 id="bed-overview">Bed telemetry</h2></div>
              <div className="bed-grid">{visibleBeds.map((bed) => <BedCard bed={bed} department={department} key={bed.patient_id} />)}</div>
              {!visibleBeds.length && <p className="panel empty">No beds are assigned to this ward.</p>}
            </section>
            <EventLog title="Station activity" entries={snapshot.global_alert_log} />
          </>}
          <div className="emergency-stack" aria-label="Emergency alerts">
            {emergencies.filter((bed) => bed.patient_id !== selected?.patient_id).map((bed) => {
              const { hr, spo2, co2, ibp } = bed.edge.latest_vitals || {};
              return <article key={bed.patient_id} className="emergency-toast">
                <p role="alert" className="room-label">Emergency · {bed.room_only} · {bed.department}</p>
                <h2>{bed.patient_name}</h2>
                <p>{bed.emergency.join(" · ")}</p>
                <p className="emergency-vitals">HR {formatReading(hr)} · SpO₂ {formatReading(spo2)} · EtCO₂ {formatReading(co2)} · BP {formatReading(ibp)}</p>
                <Link className="button button-small" href={wardURL(bed.department, bed.patient_id)} prefetch={false}>Open emergency bed</Link>
              </article>;
            })}
          </div>
        </>}
        {!snapshot && <div className="panel empty" role="status">Waiting for live ward telemetry…</div>}
      </main>
    </div>
  );
}
