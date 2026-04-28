"""
events/schemas.py

Defines the message structure for every topic in the system.
All events share a common envelope: topic, event_id, timestamp, payload.
Individual payload schemas are defined per topic.
"""

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Any

# Base envelope — every event must have these four fields

@dataclass
class Event:
    topic: str
    event_id: str
    timestamp: str
    payload: dict

    @staticmethod
    def create(topic: str, payload: dict) -> "Event":
        return Event(
            topic=topic,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

# Topics — single source of truth for topic names

class Topics:
    IMAGE_SUBMITTED       = "image.submitted"
    INFERENCE_COMPLETED   = "inference.completed"
    ANNOTATION_STORED     = "annotation.stored"
    EMBEDDING_CREATED     = "embedding.created"
    ANNOTATION_CORRECTED  = "annotation.corrected"
    QUERY_SUBMITTED       = "query.submitted"
    QUERY_COMPLETED       = "query.completed"

# Payload schemas (dataclasses for clarity; serialized to dict for Redis)

@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: BoundingBox
    reviewer_notes: Optional[str] = None


# image.submitted
@dataclass
class ImageSubmittedPayload:
    image_id: str
    filename: str
    storage_path: str
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# inference.completed
@dataclass
class InferenceCompletedPayload:
    image_id: str
    objects: List[dict]        # list of DetectedObject dicts
    embedding: List[float]     # mock vector
    inferred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# annotation.stored
@dataclass
class AnnotationStoredPayload:
    image_id: str
    annotation_doc_id: str
    objects: List[dict]
    embedding: List[float]
    stored_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# embedding.created
@dataclass
class EmbeddingCreatedPayload:
    image_id: str
    faiss_index_id: str
    stored_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# annotation.corrected
@dataclass
class AnnotationCorrectedPayload:
    image_id: str
    annotation_doc_id: str
    corrected_objects: List[dict]
    corrected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# query.submitted
@dataclass
class QuerySubmittedPayload:
    query_id: str
    query_text: str
    top_k: int
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# query.completed
@dataclass
class QueryResult:
    image_id: str
    score: float


@dataclass
class QueryCompletedPayload:
    query_id: str
    results: List[dict]        # list of QueryResult dicts
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
