import Link from "next/link";
import { bedStatus, type Bed, type Snapshot, wardURL } from "./telemetry";

function occupiedBedStatus(bed: Bed, live: boolean) {
  if (!live) return { label: "Connection interrupted", tone: "amber" };
  const status = bedStatus(bed);
  if (bed.online && bed.emergency.length) return status;
  if (status.tone !== "green") return { ...status, tone: "amber" };
  return status;
}

export default function RoomDirectory({ snapshot, selectedId, live }: {
  snapshot: Snapshot;
  selectedId?: string;
  live: boolean;
}) {
  return <nav aria-label="Room directory">
    {snapshot.departments.map(([department, name]) => {
      const slots = snapshot.bed_directory.filter((slot) => slot.department === department);
      return <details key={department} className="room-group" open={slots.length > 0}>
        <summary title={name}>{department}</summary>
        <div className="room-numbers">
          {slots.map((slot) => {
            const bed = snapshot.beds[slot.patient_id];
            const location = `Bed ${slot.number} · ${department} · ${slot.room}`;
            if (!bed) {
              const label = `${location} · Unoccupied`;
              return <button key={slot.patient_id} className="bed-number" type="button" disabled
                title={label} aria-label={label}>{slot.number}</button>;
            }
            const status = occupiedBedStatus(bed, live);
            const label = `${location} · ${bed.patient_name} · ${status.label}`;
            return <Link key={slot.patient_id} className={`bed-number ${status.tone}`}
              href={wardURL(department, slot.patient_id)} prefetch={false}
              aria-current={selectedId === slot.patient_id ? "page" : undefined}
              title={label} aria-label={label}>{slot.number}</Link>;
          })}
          {!slots.length && <span className="secondary">No beds configured.</span>}
        </div>
      </details>;
    })}
  </nav>;
}
