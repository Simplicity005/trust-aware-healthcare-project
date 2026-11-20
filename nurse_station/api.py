"""Local HTTP and live telemetry interface for the Nurse Station."""

import asyncio
from contextlib import asynccontextmanager
import json
import math
from typing import Optional
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, StringConstraints

from engine import HOSPITAL_DEPARTMENTS, MultiPatientTrustEngine

BedId = Annotated[str, StringConstraints(min_length=1, max_length=100, pattern=r"^[^/+#\x00-\x1f\x7f]+$")]
RoomName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]
DEPARTMENT_CODES = {code for code, _ in HOSPITAL_DEPARTMENTS}


async def run_in_thread(fn, *args):
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(fn, *args)
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = MultiPatientTrustEngine()
    try:
        yield
    finally:
        await run_in_thread(app.state.engine.close)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class BedAction(BaseModel):
    patient_id: BedId


class Assignment(BaseModel):
    department: str
    room: Optional[RoomName] = None


class Admission(Assignment, BedAction):
    pass


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
            data = await run_in_thread(telemetry_json, request.app.state.engine)
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


@app.post("/release")
def release(body: BedAction, request: Request):
    engine = request.app.state.engine
    with engine.lock:
        if body.patient_id in engine.added_bed_ids:
            engine.release_jetson(body.patient_id)
    return {"ok": True}


@app.post("/failover")
def failover(request: Request):
    request.app.state.engine.trigger_failover_test()
    return {"ok": True}
