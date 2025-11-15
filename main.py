import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

import json
from urllib.parse import urlencode
from urllib.request import urlopen, Request

app = FastAPI(title="Safety Navigation Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models (requests/responses)
# -----------------------------
class Location(BaseModel):
    lat: float
    lon: float

class SegmentSignals(BaseModel):
    streetlight_intensity: float = 0.0
    cctv_density: float = 0.0
    police_proximity: float = 0.0
    crowd_density: float = 0.0
    crime_index: float = 0.0  # higher is worse
    community_reports_safety: float = 0.0

class RouteSegment(BaseModel):
    segment_id: str
    start: Location
    end: Location
    distance_m: float
    avg_speed_kmh: float = 30.0
    signals: SegmentSignals

class ScoreSegmentRequest(BaseModel):
    segment: RouteSegment
    time_of_day: Literal["day", "night", "dawn_dusk"] = "day"
    mode: Literal["safest", "fastest", "balanced", "night_safe", "female_friendly"] = "balanced"

class ScoreSegmentResponse(BaseModel):
    safety_score: float  # 0-100
    factors: Dict[str, float]

class ScoreRouteRequest(BaseModel):
    segments: List[RouteSegment]
    time_of_day: Literal["day", "night", "dawn_dusk"] = "day"
    mode: Literal["safest", "fastest", "balanced", "night_safe", "female_friendly"] = "balanced"

class SegmentScore(BaseModel):
    segment_id: str
    safety_score: float

class ScoreRouteResponse(BaseModel):
    total_distance_m: float
    eta_minutes: float
    average_safety_score: float
    segment_scores: List[SegmentScore]
    mode: str

class CompanionRequest(BaseModel):
    user_uid: str
    gender: Literal["female", "male", "non-binary", "prefer-not-to-say"]
    origin: Location
    destination: Location
    earliest_departure: datetime
    latest_departure: datetime
    active: bool = True

class MatchResult(BaseModel):
    request_id: str
    user_uid: str
    distance_to_origin_m: float
    distance_to_destination_m: float
    score: float

class ChatMessage(BaseModel):
    thread_id: str
    sender_uid: str
    receiver_uid: str
    content: str

class ShareLocationRequest(BaseModel):
    user_uid: str
    companion_uid: Optional[str] = None
    route_id: Optional[str] = None
    eta_minutes: Optional[float] = None
    platform: Literal["whatsapp", "telegram", "sms", "generic"] = "generic"
    battery_level: Optional[int] = None

class Guardian(BaseModel):
    user_uid: str
    name: str
    contact_type: Literal["phone", "email", "whatsapp", "telegram"]
    contact_value: str
    auto_updates: bool = True

class SOSTriggerRequest(BaseModel):
    user_uid: str
    location: Location
    companion_uid: Optional[str] = None
    triggered_by: Literal["manual", "fall_detection", "heart_rate", "persistent_risk"] = "manual"

class AutoSOSCheck(BaseModel):
    risk_level: float = Field(0, ge=0, le=1)
    is_stationary_minutes: float = 0.0
    fall_detected: bool = False
    heart_rate: Optional[int] = None
    hr_baseline: Optional[int] = None

class ReportCreate(BaseModel):
    reporter_uid: Optional[str] = None
    category: Literal["dark_spot", "harassment", "suspicious_activity", "hazard", "other"]
    location: Location
    description: Optional[str] = None
    photo_url: Optional[str] = None

class ReportVote(BaseModel):
    report_id: str
    upvote: bool = True

class RewardsUpdate(BaseModel):
    user_uid: str
    points: int
    reason: str

# Routing models
class RoutePlanRequest(BaseModel):
    start: Location
    end: Location
    mode: Literal["safest", "fastest", "balanced", "night_safe", "female_friendly"] = "balanced"
    time_of_day: Literal["day", "night", "dawn_dusk"] = "day"

class RouteGeometry(BaseModel):
    coordinates: List[List[float]]  # [lat, lon] pairs

class AlternativeRoute(BaseModel):
    geometry: RouteGeometry
    distance_m: float
    duration_s: float
    eta_minutes: float
    segment_scores: List[SegmentScore]
    average_safety_score: float

class RoutePlanResponse(BaseModel):
    chosen: AlternativeRoute
    alternatives: List[AlternativeRoute]
    mode: str

# -----------------------------
# Utilities
# -----------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    from math import radians, cos, sin, asin, sqrt
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def score_segment(signals: SegmentSignals, time_of_day: str, mode: str) -> Tuple[float, Dict[str, float]]:
    # Base weights for signals (sum to ~1)
    weights = {
        "streetlight_intensity": 0.18,
        "cctv_density": 0.16,
        "police_proximity": 0.14,
        "crowd_density": 0.14,
        "crime_index": 0.2,  # negative contribution
        "community_reports_safety": 0.18,
    }

    # Mode adjustments
    if mode == "night_safe":
        weights["streetlight_intensity"] += 0.08
        weights["crime_index"] += 0.05
    if mode == "female_friendly":
        weights["police_proximity"] += 0.05
        weights["cctv_density"] += 0.04
    if mode == "fastest":
        # slightly reduce safety impact for speed preference
        for k in weights:
            weights[k] *= 0.95

    # Time-of-day modifiers
    time_mod = 1.0
    if time_of_day == "night":
        time_mod = 0.9
    elif time_of_day == "dawn_dusk":
        time_mod = 0.95

    # Normalize inputs and compute score
    safe_components = {
        "streetlight_intensity": max(0.0, min(1.0, signals.streetlight_intensity)),
        "cctv_density": max(0.0, min(1.0, signals.cctv_density)),
        "police_proximity": max(0.0, min(1.0, signals.police_proximity)),
        "crowd_density": max(0.0, min(1.0, signals.crowd_density)),
        # invert crime index: assume 0 (good) to 1 (bad)
        "crime_index": 1.0 - max(0.0, min(1.0, signals.crime_index)),
        "community_reports_safety": max(0.0, min(1.0, signals.community_reports_safety)),
    }

    # Night penalty for low light
    if time_of_day == "night":
        low_light_penalty = (1 - safe_components["streetlight_intensity"]) * 0.1
    else:
        low_light_penalty = 0.0

    weighted = sum(safe_components[k] * weights[k] for k in weights)
    base_score = max(0.0, min(1.0, (weighted - low_light_penalty)))
    score_0_100 = round(base_score * 100 * time_mod, 2)
    return score_0_100, {**safe_components, "low_light_penalty": round(low_light_penalty, 3)}


def osrm_fetch_routes(start: Location, end: Location) -> Dict[str, Any]:
    # Use OSRM public demo server (note: rate limited, for demo)
    base = "https://router.project-osrm.org/route/v1/driving/"
    coords = f"{start.lon},{start.lat};{end.lon},{end.lat}"
    params = {
        "alternatives": "true",
        "geometries": "geojson",
        "overview": "full",
        "steps": "true",
    }
    url = base + coords + "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "SafeRoutes/0.2"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != "Ok" or not data.get("routes"):
        raise HTTPException(status_code=502, detail="Routing service unavailable")
    return data


def signals_from_osrm_step(step: Dict[str, Any]) -> SegmentSignals:
    # Heuristic mapping based on step properties (demo purposes)
    name = (step.get("name") or "").lower()
    mode = step.get("mode") or "driving"
    distance = float(step.get("distance") or 0.0)

    # Base assumptions
    streetlight = 0.6
    cctv = 0.5
    police = 0.5
    crowd = 0.5
    crime = 0.3
    community = 0.6

    # Adjustments by road context
    if any(k in name for k in ["highway", "express", "motorway"]):
        crowd -= 0.1
        cctv -= 0.05
        streetlight -= 0.05
    if any(k in name for k in ["main", "market", "station", "plaza", "cp "]):
        crowd += 0.15
        cctv += 0.1
    if any(k in name for k in ["park", "trail", "footpath", "lane"]):
        streetlight -= 0.1
        crime += 0.05
    if distance > 800:
        police -= 0.05
        community -= 0.05

    # Clamp 0..1
    clamp = lambda x: max(0.0, min(1.0, x))
    return SegmentSignals(
        streetlight_intensity=clamp(streetlight),
        cctv_density=clamp(cctv),
        police_proximity=clamp(police),
        crowd_density=clamp(crowd),
        crime_index=clamp(crime),
        community_reports_safety=clamp(community),
    )


def plan_routes_with_safety(start: Location, end: Location, mode: str, time_of_day: str) -> RoutePlanResponse:
    data = osrm_fetch_routes(start, end)
    routes = data.get("routes", [])
    alternatives: List[AlternativeRoute] = []

    for idx, r in enumerate(routes):
        geometry = r.get("geometry", {})
        coords_lonlat: List[List[float]] = geometry.get("coordinates", [])  # [lon, lat]
        coords_latlon = [[latlon[1], latlon[0]] for latlon in coords_lonlat]
        distance_m = float(r.get("distance", 0.0))
        duration_s = float(r.get("duration", 0.0))

        # Build segments from steps across legs
        segment_scores: List[SegmentScore] = []
        seg_id = 1
        for leg in r.get("legs", []):
            for step in leg.get("steps", []):
                step_dist = float(step.get("distance", 0.0))
                s_sig = signals_from_osrm_step(step)
                # Use step speed from duration
                step_dur = float(step.get("duration", 1.0))
                avg_speed_kmh = max(5.0, (step_dist/1000.0) / max(0.000277, step_dur/3600.0))
                # Score
                score, _ = score_segment(s_sig, time_of_day, mode)
                segment_scores.append(SegmentScore(segment_id=str(seg_id), safety_score=score))
                seg_id += 1

        avg_score = round(sum(s.safety_score for s in segment_scores) / max(1, len(segment_scores)), 2)
        eta_minutes = round(duration_s/60.0 * (1.1 if mode in ("safest", "night_safe", "female_friendly") else (0.95 if mode=="fastest" else 1.0)), 1)

        alternatives.append(AlternativeRoute(
            geometry=RouteGeometry(coordinates=coords_latlon),
            distance_m=round(distance_m, 2),
            duration_s=round(duration_s, 1),
            eta_minutes=eta_minutes,
            segment_scores=segment_scores,
            average_safety_score=avg_score,
        ))

    # Choose route by mode
    chosen: AlternativeRoute
    if mode == "fastest":
        chosen = min(alternatives, key=lambda a: a.duration_s)
    elif mode in ("safest", "night_safe", "female_friendly"):
        chosen = max(alternatives, key=lambda a: a.average_safety_score)
    else:  # balanced
        # rank by combined normalized metrics
        by_time = min(a.duration_s for a in alternatives) or 1.0
        by_safety = max(a.average_safety_score for a in alternatives) or 1.0
        def rank(a: AlternativeRoute):
            return (a.duration_s/by_time)*0.5 + (by_safety/max(1.0, a.average_safety_score))*0.5
        chosen = min(alternatives, key=rank)

    return RoutePlanResponse(chosen=chosen, alternatives=alternatives, mode=mode)


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Safety Navigation API running"}


@app.get("/schema")
def get_schema_summary():
    # Minimal schema listing for inspector tools
    return {
        "collections": [
            "userprofile", "routesegment", "report", "trip", "companionrequest",
            "chatmessage", "guardian", "sosevent", "reward", "badge"
        ]
    }


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


# Safety scoring
@app.post("/api/safety/score-segment", response_model=ScoreSegmentResponse)
def api_score_segment(payload: ScoreSegmentRequest):
    score, factors = score_segment(payload.segment.signals, payload.time_of_day, payload.mode)
    return ScoreSegmentResponse(safety_score=score, factors=factors)


@app.post("/api/routes/score", response_model=ScoreRouteResponse)
def api_score_route(payload: ScoreRouteRequest):
    total_distance = sum(s.distance_m for s in payload.segments)
    # ETA baseline using avg speed per segment
    total_hours = sum((s.distance_m / 1000.0) / max(5.0, s.avg_speed_kmh) for s in payload.segments)
    eta_min = total_hours * 60

    seg_scores: List[SegmentScore] = []
    for s in payload.segments:
        sc, _ = score_segment(s.signals, payload.time_of_day, payload.mode)
        seg_scores.append(SegmentScore(segment_id=s.segment_id, safety_score=sc))

    avg_score = round(sum(s.safety_score for s in seg_scores) / max(1, len(seg_scores)), 2)

    # Mode influence on eta (safer routes may be longer)
    if payload.mode in ("safest", "night_safe", "female_friendly"):
        eta_min *= 1.1
    elif payload.mode == "fastest":
        eta_min *= 0.95

    return ScoreRouteResponse(
        total_distance_m=round(total_distance, 2),
        eta_minutes=round(eta_min, 1),
        average_safety_score=avg_score,
        segment_scores=seg_scores,
        mode=payload.mode,
    )


@app.post("/api/routes/plan", response_model=RoutePlanResponse)
def api_plan_route(req: RoutePlanRequest):
    return plan_routes_with_safety(req.start, req.end, req.mode, req.time_of_day)


# Companion matching
@app.post("/api/companions/request")
def create_companion_request(req: CompanionRequest):
    doc = req.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    rid = db["companionrequest"].insert_one(doc).inserted_id
    return {"request_id": str(rid)}


@app.get("/api/companions/match", response_model=List[MatchResult])
def find_matches(user_uid: str = Query(...), max_origin_distance_m: int = 800, time_window_min: int = 30):
    my_req = db["companionrequest"].find_one({"user_uid": user_uid, "active": True})
    if not my_req:
        raise HTTPException(status_code=404, detail="Active request not found for user")

    now = datetime.now(timezone.utc)
    q = {
        "active": True,
        "user_uid": {"$ne": user_uid},
        "earliest_departure": {"$lte": now + timedelta(minutes=time_window_min)},
        "latest_departure": {"$gte": now - timedelta(minutes=time_window_min)},
        "gender": my_req.get("gender")  # same gender
    }
    candidates = list(db["companionrequest"].find(q).limit(50))

    results: List[MatchResult] = []
    for c in candidates:
        d_origin = haversine_m(my_req["origin"]["lat"], my_req["origin"]["lon"], c["origin"]["lat"], c["origin"]["lon"])
        d_dest = haversine_m(my_req["destination"]["lat"], my_req["destination"]["lon"], c["destination"]["lat"], c["destination"]["lon"]) \
            if c.get("destination") else 1e9
        if d_origin <= max_origin_distance_m:
            # simple score: closer origins and destinations get higher score
            score = max(0.0, 1.0 - (d_origin + 0.5 * d_dest) / 5000.0)
            results.append(MatchResult(
                request_id=str(c["_id"]),
                user_uid=c["user_uid"],
                distance_to_origin_m=round(d_origin, 1),
                distance_to_destination_m=round(d_dest, 1),
                score=round(score, 3),
            ))
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:10]


# Chat (simple, not realtime)
@app.post("/api/companions/chat/send")
def send_message(msg: ChatMessage):
    doc = msg.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    mid = db["chatmessage"].insert_one(doc).inserted_id
    return {"message_id": str(mid)}


@app.get("/api/companions/chat/thread")
def get_thread(thread_id: str):
    msgs = list(db["chatmessage"].find({"thread_id": thread_id}).sort("created_at", 1))
    for m in msgs:
        m["_id"] = str(m["_id"])  # serialize
    return {"messages": msgs}


# Location sharing
@app.post("/api/location/share")
def share_location(req: ShareLocationRequest):
    base_url = os.getenv("PUBLIC_FRONTEND_URL", "https://example.com")
    share_link = f"{base_url}/share?uid={req.user_uid}&route={req.route_id or ''}"
    pieces = [
        "Live location share",
        f"ETA: {req.eta_minutes} min" if req.eta_minutes else None,
        f"Battery: {req.battery_level}%" if req.battery_level is not None else None,
        f"Companion: {req.companion_uid}" if req.companion_uid else None,
        f"Link: {share_link}",
    ]
    text = " | ".join([p for p in pieces if p])
    return {
        "platform": req.platform,
        "text": text,
        "link": share_link,
    }


# Guardians
@app.post("/api/guardians")
def add_guardian(g: Guardian):
    gid = db["guardian"].insert_one({**g.model_dump(), "created_at": datetime.now(timezone.utc)}).inserted_id
    return {"guardian_id": str(gid)}


@app.post("/api/guardians/notify")
def guardian_notify(user_uid: str, message: str):
    # In production integrate with SMS/WhatsApp/Email providers.
    doc = {
        "user_uid": user_uid,
        "message": message,
        "created_at": datetime.now(timezone.utc),
        "type": "guardian_update"
    }
    db["notifications"].insert_one(doc)
    return {"status": "queued"}


# SOS
@app.post("/api/sos/trigger")
def sos_trigger(req: SOSTriggerRequest):
    event = {
        **req.model_dump(),
        "created_at": datetime.now(timezone.utc),
        "media_url": None,
        "handled": False,
    }
    eid = db["sosevent"].insert_one(event).inserted_id
    # Simulate actions
    actions = {
        "call_police": True,
        "send_live_location": True,
        "alarm": True,
        "record_media": True,
        "upload_to_cloud": True,
        "link_companion": bool(req.companion_uid),
        "event_id": str(eid)
    }
    return {"status": "triggered", "actions": actions}


@app.post("/api/sos/auto-check")
def sos_auto_check(check: AutoSOSCheck):
    triggers = []
    if check.fall_detected:
        triggers.append("fall_detection")
    if check.risk_level >= 0.8 and check.is_stationary_minutes >= 5:
        triggers.append("persistent_risk")
    if check.heart_rate and check.hr_baseline:
        if check.heart_rate > check.hr_baseline * 1.5:
            triggers.append("heart_rate")
    return {
        "should_trigger": len(triggers) > 0,
        "reasons": triggers
    }


# Alerts
@app.get("/api/alerts")
def get_alerts(lat: float, lon: float, time_of_day: Literal["day", "night", "dawn_dusk"] = "day"):
    # Heuristic alerts demo; in production, derive from live feeds and reports
    nearby_reports = list(db["report"].find({}).sort("created_at", -1).limit(50))
    alerts: List[Dict[str, Any]] = []
    for r in nearby_reports:
        d = haversine_m(lat, lon, r["location"]["lat"], r["location"]["lon"]) if r.get("location") else 1e9
        if d < 600:
            alerts.append({
                "type": r.get("category", "incident"),
                "message": f"Recent {r.get('category')} reported nearby ({int(d)}m)",
                "distance_m": int(d),
                "time": r.get("created_at")
            })
    if time_of_day == "night":
        alerts.append({"type": "dark_zone", "message": "Low-light area ahead. Prefer well-lit streets."})
    return {"alerts": alerts[:5]}


# Community reports
@app.post("/api/reports")
def create_report(r: ReportCreate):
    # Simple spam filter: block if description has banned patterns or repeated chars
    desc = (r.description or "").lower()
    banned = ["http://", "https://", "buy now", "free $$$"]
    if any(b in desc for b in banned) or (len(set(desc)) <= 3 and len(desc) > 12):
        raise HTTPException(status_code=400, detail="Report flagged as spam")
    rid = db["report"].insert_one({**r.model_dump(), "upvotes": 0, "downvotes": 0, "created_at": datetime.now(timezone.utc)}).inserted_id
    return {"report_id": str(rid)}


@app.post("/api/reports/vote")
def vote_report(v: ReportVote):
    from bson import ObjectId
    inc = {"upvotes": 1} if v.upvote else {"downvotes": 1}
    db["report"].update_one({"_id": ObjectId(v.report_id)}, {"$inc": inc})
    return {"status": "ok"}


# Rewards
@app.post("/api/rewards")
def add_reward(rw: RewardsUpdate):
    rid = db["reward"].insert_one({**rw.model_dump(), "created_at": datetime.now(timezone.utc)}).inserted_id
    return {"reward_id": str(rid)}


# Trip history (basic)
class TripLog(BaseModel):
    user_uid: str
    origin: Location
    destination: Location
    route_id: str
    mode: str
    distance_km: float
    eta_minutes: float
    safety_score: float


@app.post("/api/trips")
def log_trip(t: TripLog):
    tid = db["trip"].insert_one({**t.model_dump(), "created_at": datetime.now(timezone.utc)}).inserted_id
    return {"trip_id": str(tid)}


@app.get("/api/trips")
def get_trips(user_uid: str):
    trips = list(db["trip"].find({"user_uid": user_uid}).sort("created_at", -1).limit(20))
    for t in trips:
        t["_id"] = str(t["_id"])  # serialize
    return {"trips": trips}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
