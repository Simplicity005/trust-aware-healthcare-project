import { useEffect, useState, type FormEvent } from "react";
import { Badge } from "./bed-panel";
import { formatReading, type Device, type Snapshot } from "./telemetry";

export type Action = (operation: string, data: Record<string, string | number | boolean | Record<string, boolean> | null>) => Promise<{ patient_id?: string }>;

function AssignmentForm({ device, departments, busy, onSubmit }: {
  device: Device;
  departments: Snapshot["departments"];
  busy: boolean;
  onSubmit: (device: Device, department: string, room: string | null) => void;
}) {
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    onSubmit(device, String(form.get("department")), String(form.get("room")).trim() || null);
  }

  return (
    <form className="assignment" onSubmit={submit}>
      <div className="device-info">
        <div className="card-heading"><h3>{device.patient_id}</h3>
          <Badge tone={device.online ? "green" : "muted"}>{device.online ? "Online" : "Last seen " + device.seconds_since_seen + "s ago"}</Badge>
        </div>
        <p>{device.device_model} · {device.exec_provider}</p>
        <p className="secondary">
          HR {formatReading(device.latest_vitals?.hr)} · SpO₂ {formatReading(device.latest_vitals?.spo2)} ·
          {" "}EtCO₂ {formatReading(device.latest_vitals?.co2)} · BP {formatReading(device.latest_vitals?.ibp)}
        </p>
      </div>
      <label>Department
        <select name="department" defaultValue={device.department || departments[0]?.[0]} required>
          {departments.map(([code, name]) => <option key={code} value={code}>{name}</option>)}
        </select>
      </label>
      <label>Room <span className="secondary">(optional)</span>
        <input name="room" placeholder={device.room || "Auto-assigned"} maxLength={60} />
      </label>
      <button className="button button-primary" disabled={busy} type="submit">
        Admit bed
      </button>
    </form>
  );
}

export default function Admission({ snapshot, scanDeadline, onScan, onClose, onAdmit, busy, probe }: {
  snapshot: Snapshot;
  scanDeadline: number | null;
  onScan: (active: boolean) => void;
  onClose: () => void;
  onAdmit: (device: Device, department: string, room: string | null) => void;
  busy: boolean;
  probe: () => Promise<{ patient_id?: string }>;
}) {
  const [remaining, setRemaining] = useState(150);
  const [scanError, setScanError] = useState("");

  useEffect(() => {
    if (!scanDeadline) return;
    let canceled = false;
    const updateTime = () => {
      const seconds = Math.max(0, Math.ceil((scanDeadline - Date.now()) / 1000));
      setRemaining(seconds);
      if (seconds === 0) onScan(false);
    };
    const sendProbe = async () => {
      if (Date.now() >= scanDeadline) return;
      try {
        await probe();
        if (!canceled) setScanError("");
      } catch (error) {
        if (!canceled) setScanError(error instanceof Error ? error.message : "The scan could not reach the station.");
      }
    };
    updateTime();
    void sendProbe();
    const clock = window.setInterval(updateTime, 1000);
    const scan = window.setInterval(sendProbe, 4000);
    return () => {
      canceled = true;
      window.clearInterval(clock);
      window.clearInterval(scan);
    };
  }, [scanDeadline, probe, onScan]);

  const count = snapshot.unadded_jetsons.length;
  return (
    <section className="panel admission" aria-labelledby="admission-title">
      <div className="section-heading"><h2 id="admission-title">Add a bedside device</h2><button className="button button-small" onClick={onClose}>Close panel</button></div>
      <div className="scan">
        <div>
          <Badge tone={scanDeadline ? "blue" : "muted"}>{scanDeadline ? "Detecting devices" : "Detection stopped"}</Badge>
          {scanDeadline && <span className="scan-timer">{Math.floor(remaining / 60)}:{String(remaining % 60).padStart(2, "0")} remaining</span>}
        </div>
        <button className="button button-small" onClick={() => onScan(!scanDeadline)}>{scanDeadline ? "Stop detection" : "Restart detection"}</button>
      </div>
      {scanError && <p role="status" className="error-box">{scanError}</p>}
      <p className="secondary">{count ? `${count} unassigned ${count === 1 ? "device" : "devices"} found.` : "No unassigned devices found."} Confirm the bedside is running and connected to the same MQTT broker.</p>
      {snapshot.unadded_jetsons.map((device) => <AssignmentForm key={device.patient_id} device={device}
        departments={snapshot.departments} busy={busy} onSubmit={onAdmit} />)}
    </section>
  );
}
