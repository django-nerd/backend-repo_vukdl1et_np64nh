import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection

DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "safe_routes")

_client = MongoClient(DATABASE_URL)
_db = _client[DATABASE_NAME]


def db():
    return _db


def _collection(name: str) -> Collection:
    return _db[name]


def create_document(collection_name: str, data: Dict[str, Any]) -> str:
    now = datetime.utcnow()
    payload = {**data, "created_at": now, "updated_at": now}
    res = _collection(collection_name).insert_one(payload)
    return str(res.inserted_id)


def get_documents(collection_name: str, filter_dict: Optional[Dict[str, Any]] = None, limit: int = 50) -> List[Dict[str, Any]]:
    filter_dict = filter_dict or {}
    cursor = _collection(collection_name).find(filter_dict).sort("created_at", -1).limit(limit)
    results: List[Dict[str, Any]] = []
    for d in cursor:
        d["_id"] = str(d.get("_id"))
        results.append(d)
    return results
