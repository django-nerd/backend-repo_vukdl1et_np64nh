"""
Database Schemas for Safety Navigation App

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).

Collections:
- userprofile
- routeSegment
- report
- trip
- companionrequest
- chatmessage
- guardianguardian
- sosevent
- reward
- badge
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

# Core domain
class Location(BaseModel):
    lat: float
    lon: float

class UserProfile(BaseModel):
    uid: str = Field(..., description="External auth user id (Firebase or similar)")
    name: str
    email: EmailStr
    gender: Literal["female", "male", "non-binary", "prefer-not-to-say"] = "prefer-not-to-say"
    verified: bool = False
    rating: float = 0.0
    reviews_count: int = 0
    photo_url: Optional[str] = None
    created_at: Optional[datetime] = None

class RouteSegment(BaseModel):
    segment_id: str
    start: Location
    end: Location
    distance_m: float
    avg_speed_kmh: float = 30.0
    # Safety signals (0-1 normalized where applicable)
    streetlight_intensity: float = 0.0
    cctv_density: float = 0.0
    police_proximity: float = 0.0  # inverse distance normalized
    crowd_density: float = 0.0
    crime_index: float = 0.0       # higher is worse, will be inverted in score
    community_reports_safety: float = 0.0  # net upvotes on safety

class Report(BaseModel):
    reporter_uid: Optional[str] = None
    category: Literal["dark_spot", "harassment", "suspicious_activity", "hazard", "other"]
    location: Location
    description: Optional[str] = None
    photo_url: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0
    created_at: Optional[datetime] = None

class Trip(BaseModel):
    user_uid: str
    origin: Location
    destination: Location
    mode: Literal["fastest", "safest", "balanced", "night_safe", "female_friendly"]
    route_id: str
    eta_minutes: float
    distance_km: float
    safety_score: float
    companion_uid: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

class CompanionRequest(BaseModel):
    user_uid: str
    gender: Literal["female", "male", "non-binary", "prefer-not-to-say"]
    origin: Location
    destination: Location
    earliest_departure: datetime
    latest_departure: datetime
    active: bool = True

class ChatMessage(BaseModel):
    thread_id: str
    sender_uid: str
    receiver_uid: str
    content: str
    created_at: Optional[datetime] = None

class Guardian(BaseModel):
    user_uid: str
    name: str
    contact_type: Literal["phone", "email", "whatsapp", "telegram"]
    contact_value: str
    auto_updates: bool = True

class SOSEvent(BaseModel):
    user_uid: str
    location: Location
    triggered_by: Literal["manual", "fall_detection", "heart_rate", "persistent_risk"]
    media_url: Optional[str] = None
    handled: bool = False

class Reward(BaseModel):
    user_uid: str
    points: int = 0
    reason: str

class Badge(BaseModel):
    user_uid: str
    code: str
    label: str
    earned_at: Optional[datetime] = None
