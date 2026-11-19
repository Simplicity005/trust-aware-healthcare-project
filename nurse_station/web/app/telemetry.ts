export const sensors = [
  { id: "Agent_ECG", label: "ECG lead II", short: "ECG", unit: "", color: "#6de5b9" },
  { id: "Agent_HR", label: "Heart rate", short: "HR", unit: "bpm", color: "#f4a2bb" },
  { id: "Agent_SpO2", label: "Oxygen saturation", short: "SpO₂", unit: "%", color: "#77bcff" },
  { id: "Agent_CO2", label: "Capnography", short: "EtCO₂", unit: "mmHg", color: "#f5c371" },
  { id: "Agent_IBP", label: "Invasive pressure", short: "IBP", unit: "mmHg", color: "#b8a2fa" },
  { id: "Agent_NIBP", label: "Cuff pressure", short: "NIBP", unit: "mmHg", color: "#8da4bc" },
] as const;

export type SensorId = (typeof sensors)[number]["id"];
type Pressure = { sys?: number; dia?: number; formatted?: string };
type Reading = number | string | Pressure | null;

type Sensor = {
  trust: number;
  status: "TRUSTED" | "SUSPICIOUS" | "ISOLATED";
  removed?: boolean;
  value: Reading;
  s_l: number;
  s_j: number;
  s_p: number;
  s_d: number;
  packets_received: number;
  packets_missed: number;
};

type Vitals = { hr?: Reading; spo2?: Reading; co2?: Reading; ibp?: Reading };

export type AgentChaos = {
  delay_ms?: number;
  drop_rate?: number;
  malicious?: boolean;
  removed?: boolean;
  condition?: string;
  target_hr?: number;
  target_spo2?: number;
  target_co2?: number;
  target_sys?: number;
  target_dia?: number;
};

export type BedChaos = {
  vitals?: {
    target_hr?: number;
    target_spo2?: number;
    target_co2?: number;
    target_etco2?: number;
    target_respiration?: number;
    target_sys?: number;
    target_dia?: number;
    ecg_condition?: string;
  };
  Orin_Link?: {
    overload_gpu?: boolean;
    use_relay_b?: boolean;
  };
  Agent_ECG?: AgentChaos;
  Agent_HR?: AgentChaos;
  Agent_SpO2?: AgentChaos;
  Agent_CO2?: AgentChaos;
  Agent_EtCO2?: AgentChaos;
  Agent_IBP?: AgentChaos;
  Agent_NIBP?: AgentChaos;
};

export type Bed = {
  patient_id: string;
  patient_name: string;
  diagnosis: string;
  admitted: string;
  department: string;
  room_only: string;
  online: boolean;
  seconds_since_seen: number;
  agents: Record<SensorId, Sensor>;
  edge: {
    latest_vitals?: Vitals;
    local_bed_alarm?: boolean;
    arrhythmia_prob?: number;
    inf_ms?: number;
    exec_provider?: string;
  };
  ecg_wave: number[];
  ecg_end_ts?: number;
  ecg_rate?: number;
  chaos?: BedChaos;
  history: (Record<SensorId, number> & { time: string })[];
  compute_mode: string;
  gateway_cpu_inf_ms: number;
  alert_route: string;
  alert_log: string[];
  emergency: string[];
};

export type Device = {
  patient_id: string;
  device_model: string;
  exec_provider: string;
  department: string;
  room: string;
  online: boolean;
  seconds_since_seen: number;
  latest_vitals: Vitals;
};

export type Snapshot = {
  server_time?: number;
  beds: Record<string, Bed>;
  bed_directory: { patient_id: string; number: string; department: string; room: string }[];
  departments: [string, string][];
  unadded_jetsons: Device[];
  global_alert_log: string[];
  ha: {
    station_id: string;
    station_port: number;
    is_primary: boolean;
    failover_triggered: boolean;
    failover_simulated: boolean;
    total_stations_online: number;
    peers: Record<string, { station_id: string; role: string; port: number }>;
  };
};

export function bedStatus(bed: Bed) {
  if (!bed.online) return { label: "Offline", tone: "muted" };
  if (bed.emergency.length) return { label: "Emergency", tone: "red" };
  const agents = Object.values(bed.agents);
  if (agents.some((agent) => agent.status === "ISOLATED")) return { label: "Sensor isolated", tone: "red" };
  if (agents.some((agent) => agent.status === "SUSPICIOUS")) return { label: "Degraded", tone: "amber" };
  if (bed.compute_mode !== "Jetson Local") return { label: "Offloaded", tone: "amber" };
  return { label: "Stable", tone: "green" };
}

export function formatReading(value?: Reading) {
  if (value == null) return "—";
  if (typeof value !== "object") return String(value);
  return value.formatted || (value.sys != null && value.dia != null ? `${value.sys}/${value.dia}` : "—");
}

export function wardURL(department = "", bed = "") {
  const params = new URLSearchParams();
  if (department) params.set("dept", department);
  if (bed) {
    params.set("bed", bed);
    params.set("view", "detail");
  }
  return params.size ? `/?${params}` : "/";
}
