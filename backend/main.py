from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    AutoSOSCheck,
    ChatMessage,
    CompanionRequest,
    GuardianCreate,
    GuardianNotify,
    Report,
    RouteSegment,
    ScoreRouteRequest,
    ShareRequest,
    SOSEvent,
)
from database import db, create_document, get_documents

app = FastAPI(title="SafeRoutes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utilities

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def score_segment(seg: RouteSegment, time_of_day: str, mode: str) -> float:
    s = seg.signals
    weights = {
        "fastest": dict(light=0.05, cctv=0.05, police=0.05, crowd=0.05, crime=0.05, reports=0.05, speed=0.7),
        "safest": dict(light=0.2, cctv=0.2, police=0.2, crowd=0.1, crime=0.2, reports=0.1, speed=0.0),
        "balanced": dict(light=0.15, cctv=0.15, police=0.15, crowd=0.1, crime=0.15, reports=0.1, speed=0.2),
        "night_safe": dict(light=0.25, cctv=0.15, police=0.15, crowd=0.1, crime=0.25, reports=0.1, speed=0.0),
        "female_friendly": dict(light=0.15, cctv=0.25, police=0.2, crowd=0.1, crime=0.2, reports=0.1, speed=0.0),
    }.get(mode, {
        "light":0.15,"cctv":0.15,"police":0.15,"crowd":0.1,"crime":0.15,"reports":0.1,"speed":0.2
    })

    # Base safety
    crime_inv = 1 - s.crime_index
    base = (
        s.streetlight_intensity * weights["light"] +
        s.cctv_density * weights["cctv"] +
        s.police_proximity * weights["police"] +
        s.crowd_density * weights["crowd"] +
        crime_inv * weights["crime"] +
        s.community_reports_safety * weights["reports"]
    )
    # Time of day modifier
    if time_of_day == "night":
        base *= 0.9
    elif time_of_day == "dawn_dusk":
        base *= 0.95

    # Normalize to 0..1 then scale to 0..100
    safety = max(0.0, min(1.0, base)) * 100
    return round(safety, 1)


@app.post("/api/safety/score-segment")
async def api_score_segment(payload: Dict):
    seg = RouteSegment(**payload.get("segment"))
    time_of_day = payload.get("time_of_day", "day")
    mode = payload.get("mode", "balanced")
    return {"safety_score": score_segment(seg, time_of_day, mode)}


@app.post("/api/routes/score")
async def api_score_route(body: ScoreRouteRequest):
    seg_scores = []
    total_dist = 0.0
    total_time_hours = 0.0
    for seg in body.segments:
        ss = score_segment(seg, body.time_of_day, body.mode)
        seg_scores.append({"segment_id": seg.segment_id, "safety_score": ss})
        total_dist += seg.distance_m
        total_time_hours += (seg.distance_m / 1000) / max(5.0, seg.avg_speed_kmh)

    avg_safety = round(sum(s["safety_score"] for s in seg_scores) / max(1, len(seg_scores)), 1)

    # ETA adjustments per mode
    eta_minutes = total_time_hours * 60
    if body.mode in ("safest", "night_safe", "female_friendly"):
        eta_minutes *= 1.15
    elif body.mode == "fastest":
        eta_minutes *= 0.9

    # Persist a trip record
    trip_id = create_document("trip", {
        "mode": body.mode,
        "distance_km": round(total_dist/1000, 2),
        "eta_minutes": round(eta_minutes),
        "average_safety_score": avg_safety,
    })

    return {
        "trip_id": trip_id,
        "mode": body.mode,
        "eta_minutes": round(eta_minutes),
        "average_safety_score": avg_safety,
        "segment_scores": seg_scores,
    }


# Companions
@app.post("/api/companions/request")
async def create_companion_request(body: CompanionRequest):
    rid = create_document("companionrequest", body.model_dump())
    return {"request_id": rid}


@app.get("/api/companions/match")
async def match_companions(user_uid: str = Query(...)):
    requests = get_documents("companionrequest", {"active": True})
    me = [r for r in requests if r.get("user_uid") == user_uid]
    if not me:
        raise HTTPException(404, "Create a companion request first")
    mine = me[0]
    candidates = [r for r in requests if r.get("user_uid") != user_uid and r.get("gender") == mine.get("gender")]
    out = []
    for r in candidates:
        o = r.get("origin", {})
        d = r.get("destination", {})
        score = 1.0 / (1 + (haversine_m(mine["origin"]["lat"], mine["origin"]["lon"], o["lat"], o["lon"]) +
                             haversine_m(mine["destination"]["lat"], mine["destination"]["lon"], d["lat"], d["lon"])) / 1000)
        out.append({
            "request_id": r.get("_id"),
            "user_uid": r.get("user_uid"),
            "distance_to_origin_m": int(haversine_m(mine["origin"]["lat"], mine["origin"]["lon"], o["lat"], o["lon"])) ,
            "distance_to_destination_m": int(haversine_m(mine["destination"]["lat"], mine["destination"]["lon"], d["lat"], d["lon"])) ,
            "score": round(score, 3)
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:10]


# Chat threads (simple stub) - stored as messages
@app.post("/api/companions/chat/send")
async def send_chat(body: ChatMessage):
    mid = create_document("chatmessage", {**body.model_dump(), "sent_at": datetime.utcnow()})
    return {"message_id": mid}


@app.get("/api/companions/chat/thread")
async def get_thread(thread_id: str):
    msgs = get_documents("chatmessage", {"thread_id": thread_id}, limit=100)
    msgs.sort(key=lambda m: m.get("sent_at", datetime.utcnow()))
    return {"messages": msgs}


# Live sharing & guardians
@app.post("/api/location/share")
async def share_location(body: ShareRequest):
    text = (
        f"Live trip • ETA {body.eta_minutes} min • Battery {body.battery_level}%\n"
        f"Companion: none • Route: {body.route_id}\n"
        f"Open link: https://saferoutes.app/share/{body.user_uid}/{body.route_id}"
    )
    sid = create_document("share", {**body.model_dump(), "text": text})
    return {"share_id": sid, "text": text}


@app.post("/api/guardians")
async def add_guardian(body: GuardianCreate):
    gid = create_document("guardian", body.model_dump())
    return {"guardian_id": gid}


@app.post("/api/guardians/notify")
async def notify_guardians(body: GuardianNotify):
    nid = create_document("guardian_notify", {**body.model_dump(), "sent_at": datetime.utcnow()})
    return {"notified": True, "notification_id": nid}


# SOS
@app.post("/api/sos/trigger")
async def sos_trigger(body: SOSEvent):
    eid = create_document("sos", {**body.model_dump(), "triggered_at": datetime.utcnow(), "actions": ["call_police", "alarm", "live_share", "recording"]})
    return {"sos_id": eid, "actions": ["call_police", "alarm", "live_share", "recording"]}


@app.post("/api/sos/auto-check")
async def sos_auto_check(body: AutoSOSCheck):
    reasons = []
    if body.risk_level >= 0.8:
        reasons.append("high_risk")
    if body.is_stationary_minutes >= 5 and body.risk_level >= 0.7:
        reasons.append("stationary_high_risk")
    if body.fall_detected:
        reasons.append("fall")
    if body.heart_rate and body.hr_baseline and body.heart_rate >= body.hr_baseline + 40:
        reasons.append("hr_spike")
    return {"should_trigger": len(reasons) > 0, "reasons": reasons}


# Alerts
@app.get("/api/alerts")
async def alerts(lat: float, lon: float, time_of_day: str = "day"):
    hints = []
    if time_of_day == "night":
        hints.append({"type": "dark_zone", "message": "Low lighting ahead, prefer lit streets."})
    recent_reports = get_documents("report", limit=20)
    events = [{"type": "report", "message": f"Recent {r.get('category')} nearby"} for r in recent_reports]
    return {"alerts": hints + events}


# Reports
@app.post("/api/reports")
async def create_report(body: Report):
    # Basic spam filter: require category and location
    if body.category not in {"dark_spot", "harassment", "suspicious_activity", "hazard", "other"}:
        raise HTTPException(400, "invalid category")
    rid = create_document("report", body.model_dump())
    return {"report_id": rid}


@app.post("/api/reports/vote")
async def vote_report(report_id: str, vote: int = Query(..., ge=-1, le=1)):
    # Placeholder: write a vote record
    vid = create_document("report_vote", {"report_id": report_id, "vote": vote})
    return {"vote_id": vid}


# Trips
@app.get("/api/trips")
async def get_trips(user_uid: str):
    trips = get_documents("trip", limit=20)
    return {"trips": trips}


@app.get("/test")
async def test():
    # Touch DB
    create_document("_ping", {"ok": True, "ts": datetime.utcnow()})
    return {"status": "ok"}
