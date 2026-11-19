import type { FormEvent } from "react";
import { Badge } from "./bed-panel";
import type { Device, Snapshot } from "./telemetry";

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

export default function Admission({ snapshot, onAdmit, busy }: {
  snapshot: Snapshot;
  onAdmit: (device: Device, department: string, room: string | null) => void;
  busy: boolean;
}) {
  return (
    <section className="panel admission" aria-labelledby="admission-title">
      <div className="section-heading"><h2 id="admission-title">Readmit a released bed</h2></div>
      {snapshot.unadded_jetsons.map((device) => <AssignmentForm key={device.patient_id} device={device}
        departments={snapshot.departments} busy={busy} onSubmit={onAdmit} />)}
    </section>
  );
}
