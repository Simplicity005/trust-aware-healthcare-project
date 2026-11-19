"""Local HTTP and live telemetry interface for the Nurse Station."""

import asyncio
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, StringConstraints

from engine import CHAOS_FILE, HOSPITAL_DEPARTMENTS, MultiPatientTrustEngine

BedId = Annotated[str, StringConstraints(min_length=1, max_length=100, pattern=r"^[^/+#\x00-\x1f\x7f]+$")]
RoomName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]
DEPARTMENT_CODES = {code for code, _ in HOSPITAL_DEPARTMENTS}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = MultiPatientTrustEngine()
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.engine.close)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class BedAction(BaseModel):
    patient_id: BedId


class Assignment(BaseModel):
    department: str
    room: RoomName | None = None


class Admission(Assignment, BedAction):
    pass


class AlarmAction(BedAction):
    active: bool


class ManipulateAction(BedAction):
    target_hr: int | None = None
    target_spo2: int | None = None
    target_co2: int | None = None
    target_respiration: int | None = None
    target_sys: int | None = None
    target_dia: int | None = None
    ecg_condition: str | None = None
    delay_ms: float | None = None
    drop_rate: float | None = None
    malicious: bool | None = None
    sensor_faults: dict[str, bool] | None = None
    overload_gpu: bool | None = None
    use_relay_b: bool | None = None
    reset: bool = False


def validate_department(department):
    if department not in DEPARTMENT_CODES:
        raise HTTPException(422, "Unknown department.")


def numeric(value, default):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return default


def emergency_reasons(bed):
    if not bed["online"]:
        return []
    edge = bed["edge"]
    vitals = edge.get("latest_vitals") or {}
    if not isinstance(vitals, dict):
        vitals = {}
    hr = numeric(vitals.get("hr"), 75)
    spo2 = numeric(vitals.get("spo2"), 98)
    co2 = numeric(vitals.get("co2"), 38)
    pressure = vitals.get("ibp")
    sys_bp = numeric(pressure.get("sys"), 120) if isinstance(pressure, dict) else 120
    probability = numeric(edge.get("arrhythmia_prob"), 0)
    reasons = []
    if edge.get("local_bed_alarm") or probability >= 0.70:
        reasons.append(f"Ventricular Arrhythmia ({probability * 100:.1f}% risk)")
    if spo2 <= 85:
        reasons.append(f"Critical Hypoxia (SpO₂ {spo2}%)")
    if hr >= 160:
        reasons.append(f"Extreme Tachycardia (HR {hr} bpm)")
    elif hr <= 40:
        reasons.append(f"Severe Bradycardia (HR {hr} bpm)")
    if sys_bp <= 75:
        reasons.append(f"Hypotensive Shock (Sys {sys_bp} mmHg)")
    elif sys_bp >= 200:
        reasons.append(f"Hypertensive Crisis (Sys {sys_bp} mmHg)")
    if co2 >= 60:
        reasons.append(f"Severe Hypercapnia (EtCO₂ {co2} mmHg)")
    return reasons


def telemetry_json(engine):
    data = engine.snapshot()
    data["departments"] = HOSPITAL_DEPARTMENTS
    for bed in data["beds"].values():
        bed["emergency"] = emergency_reasons(bed)
        # The chart already reads the dedicated ECG field.
        bed["edge"].pop("ecg_wave", None)
    return json.dumps(data, separators=(",", ":"))


ALL_SENSOR_AGENTS = (
    "Agent_ECG", "Agent_SpO2", "Agent_HR", "Agent_CO2",
    "Agent_Respiration", "Agent_EtCO2", "Agent_IBP", "Agent_NIBP", "Agent_Temperature",
)


def default_bed_chaos():
    return {
        "Agent_ECG": {"delay_ms": 12, "drop_rate": 0.0, "malicious": False, "condition": "normal"},
        "Agent_SpO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_spo2": 98},
        "Agent_HR": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_hr": 75},
        "Agent_CO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_co2": 38},
        "Agent_Respiration": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Agent_EtCO2": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_etco2": 38},
        "Agent_IBP": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False, "target_sys": 120, "target_dia": 80},
        "Agent_NIBP": {"delay_ms": 15, "drop_rate": 0.0, "malicious": False, "target_sys": 120, "target_dia": 80},
        "Agent_Temperature": {"delay_ms": 10, "drop_rate": 0.0, "malicious": False},
        "Orin_Link": {"overload_gpu": False, "use_relay_b": False},
        "vitals": {
            "target_hr": 75,
            "target_spo2": 98,
            "target_co2": 38,
            "target_etco2": 38,
            "target_respiration": 16,
            "target_sys": 120,
            "target_dia": 80,
            "ecg_condition": "normal",
        },
    }


def write_chaos_config(config):
    path = Path(CHAOS_FILE)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with output:
                json.dump(config, output, indent=2)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def set_simulated_alarm(patient_id, active):
    path = Path(CHAOS_FILE)
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    bed = config.setdefault(patient_id, default_bed_chaos())
    condition = "arrhythmia" if active else "normal"
    bed.setdefault("vitals", {}).update(
        ecg_condition=condition,
        target_hr=165 if active else 75,
        target_spo2=82 if active else 98,
    )
    bed.setdefault("Agent_ECG", {})["condition"] = condition
    write_chaos_config(config)


def apply_bed_manipulation(body: ManipulateAction, engine):
    path = Path(CHAOS_FILE)
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if body.reset:
        bed = default_bed_chaos()
        config[body.patient_id] = bed
    else:
        bed = config.setdefault(body.patient_id, default_bed_chaos())
        vitals = bed.setdefault("vitals", {})
        if body.target_hr is not None:
            vitals["target_hr"] = int(body.target_hr)
            bed.setdefault("Agent_HR", {})["target_hr"] = int(body.target_hr)
        if body.target_spo2 is not None:
            vitals["target_spo2"] = int(body.target_spo2)
            bed.setdefault("Agent_SpO2", {})["target_spo2"] = int(body.target_spo2)
        if body.target_co2 is not None:
            vitals["target_co2"] = int(body.target_co2)
            vitals["target_etco2"] = int(body.target_co2)
            bed.setdefault("Agent_CO2", {})["target_co2"] = int(body.target_co2)
            bed.setdefault("Agent_EtCO2", {})["target_etco2"] = int(body.target_co2)
        if body.target_respiration is not None:
            vitals["target_respiration"] = int(body.target_respiration)
        if body.target_sys is not None:
            vitals["target_sys"] = int(body.target_sys)
            bed.setdefault("Agent_IBP", {})["target_sys"] = int(body.target_sys)
            bed.setdefault("Agent_NIBP", {})["target_sys"] = int(body.target_sys)
        if body.target_dia is not None:
            vitals["target_dia"] = int(body.target_dia)
            bed.setdefault("Agent_IBP", {})["target_dia"] = int(body.target_dia)
            bed.setdefault("Agent_NIBP", {})["target_dia"] = int(body.target_dia)
        if body.ecg_condition is not None:
            vitals["ecg_condition"] = str(body.ecg_condition)
            bed.setdefault("Agent_ECG", {})["condition"] = str(body.ecg_condition)
        if body.delay_ms is not None:
            for aid in ALL_SENSOR_AGENTS:
                bed.setdefault(aid, {})["delay_ms"] = float(body.delay_ms)
        if body.drop_rate is not None:
            for aid in ALL_SENSOR_AGENTS:
                bed.setdefault(aid, {})["drop_rate"] = float(body.drop_rate)
        if body.malicious is not None:
            for aid in ALL_SENSOR_AGENTS:
                bed.setdefault(aid, {})["malicious"] = bool(body.malicious)
        if body.sensor_faults:
            for aid, is_mal in body.sensor_faults.items():
                if aid in ALL_SENSOR_AGENTS:
                    bed.setdefault(aid, {})["malicious"] = bool(is_mal)
                    if aid == "Agent_CO2":
                        bed.setdefault("Agent_EtCO2", {})["malicious"] = bool(is_mal)
                    elif aid == "Agent_EtCO2":
                        bed.setdefault("Agent_CO2", {})["malicious"] = bool(is_mal)
        if body.overload_gpu is not None or body.use_relay_b is not None:
            ov = bool(body.overload_gpu if body.overload_gpu is not None else body.use_relay_b)
            bed.setdefault("Orin_Link", {})["overload_gpu"] = ov
            bed.setdefault("Orin_Link", {})["use_relay_b"] = ov

    write_chaos_config(config)
    if engine.client:
        try:
            engine.client.publish(f"hospital/{body.patient_id}/control/chaos", json.dumps(bed), qos=1)
        except Exception:
            pass
    return bed


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/snapshot")
def snapshot(request: Request):
    return Response(
        telemetry_json(request.app.state.engine),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/events")
async def events(request: Request):
    async def updates():
        while not await request.is_disconnected():
            data = await asyncio.to_thread(telemetry_json, request.app.state.engine)
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.35)

    return StreamingResponse(
        updates(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/probe")
def probe(request: Request):
    request.app.state.engine.probe_network_for_jetsons()
    return {"ok": True}


@app.post("/admit")
def admit(body: Admission, request: Request):
    validate_department(body.department)
    engine = request.app.state.engine
    with engine.lock:
        if body.patient_id in engine.added_bed_ids:
            return {"patient_id": body.patient_id}
        if body.patient_id not in engine.discovered_jetsons:
            raise HTTPException(404, "This bedside node has not been discovered.")
        engine.admit_jetson(body.patient_id, department=body.department, room_name=body.room)
    return {"patient_id": body.patient_id}


@app.post("/simulate")
def simulate(body: Assignment, request: Request):
    validate_department(body.department)
    patient_id = request.app.state.engine.admit_simulated_jetson(body.department, body.room)
    return {"patient_id": patient_id}


@app.post("/release")
def release(body: BedAction, request: Request):
    engine = request.app.state.engine
    with engine.lock:
        if body.patient_id in engine.added_bed_ids:
            engine.release_jetson(body.patient_id)
    return {"ok": True}


@app.post("/alarm")
def alarm(body: AlarmAction, request: Request):
    engine = request.app.state.engine
    with engine.lock:
        if body.patient_id not in engine.added_bed_ids:
            raise HTTPException(404, "This bedside node is not admitted.")
        try:
            set_simulated_alarm(body.patient_id, body.active)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(503, "Could not update the alarm simulator configuration.") from exc
    return {"ok": True}


@app.post("/failover")
def failover(request: Request):
    request.app.state.engine.trigger_failover_test()
    return {"ok": True}


@app.post("/manipulate")
def manipulate(body: ManipulateAction, request: Request):
    engine = request.app.state.engine
    with engine.lock:
        if body.patient_id not in engine.added_bed_ids:
            raise HTTPException(404, "This bedside node is not admitted.")
        try:
            chaos = apply_bed_manipulation(body, engine)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(503, "Could not update the simulator configuration.") from exc
    return {"ok": True, "chaos": chaos}
