from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class LatLon(BaseModel):
    lat: float
    lon: float


class Signals(BaseModel):
    streetlight_intensity: float = Field(ge=0, le=1)
    cctv_density: float = Field(ge=0, le=1)
    police_proximity: float = Field(ge=0, le=1)
    crowd_density: float = Field(ge=0, le=1)
    crime_index: float = Field(ge=0, le=1)
    community_reports_safety: float = Field(ge=0, le=1)


class RouteSegment(BaseModel):
    segment_id: str
    start: LatLon
    end: LatLon
    distance_m: float
    avg_speed_kmh: float
    signals: Signals


class ScoreSegmentRequest(BaseModel):
    segment: RouteSegment
    time_of_day: str = Field(description="day|night|dawn_dusk")
    mode: str = Field(description="fastest|safest|balanced|night_safe|female_friendly")


class ScoreRouteRequest(BaseModel):
    segments: List[RouteSegment]
    time_of_day: str
    mode: str


class CompanionRequest(BaseModel):
    user_uid: str
    gender: str
    origin: LatLon
    destination: LatLon
    earliest_departure: datetime
    latest_departure: datetime
    active: bool = True


class ChatMessage(BaseModel):
    thread_id: str
    sender_uid: str
    text: str


class Report(BaseModel):
    reporter_uid: str
    category: str
    description: Optional[str] = None
    location: LatLon


class ShareRequest(BaseModel):
    user_uid: str
    route_id: str
    eta_minutes: int
    battery_level: int
    platform: str = "generic"


class GuardianCreate(BaseModel):
    user_uid: str
    guardian_uid: Optional[str] = None
    phone: Optional[str] = None


class GuardianNotify(BaseModel):
    user_uid: str
    message: str


class SOSEvent(BaseModel):
    user_uid: str
    location: LatLon
    triggered_by: str


class AutoSOSCheck(BaseModel):
    risk_level: float
    is_stationary_minutes: int = 0
    fall_detected: bool = False
    heart_rate: Optional[int] = None
    hr_baseline: Optional[int] = None
