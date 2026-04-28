"""
events/validation.py

Validates that incoming events conform to the expected envelope and payload.
Malformed events are rejected with a descriptive error — they never crash a service.
"""

from typing import Tuple


REQUIRED_ENVELOPE_FIELDS = {"topic", "event_id", "timestamp", "payload"}

REQUIRED_PAYLOAD_FIELDS = {
    "image.submitted":      {"image_id", "filename", "storage_path"},
    "inference.completed":  {"image_id", "objects", "embedding"},
    "annotation.stored":    {"image_id", "annotation_doc_id", "objects", "embedding"},
    "embedding.created":    {"image_id", "faiss_index_id"},
    "annotation.corrected": {"image_id", "annotation_doc_id", "corrected_objects"},
    "query.submitted":      {"query_id", "query_text", "top_k"},
    "query.completed":      {"query_id", "results"},
}


def validate_event(event: dict) -> Tuple[bool, str]:
    """
    Returns (True, "") if valid.
    Returns (False, reason) if invalid.
    Never raises — always returns a result.
    """
    # Check envelope
    missing = REQUIRED_ENVELOPE_FIELDS - set(event.keys())
    if missing:
        return False, f"Missing envelope fields: {missing}"

    topic = event.get("topic")
    payload = event.get("payload")

    if not isinstance(payload, dict):
        return False, "Payload must be a dict"

    if topic not in REQUIRED_PAYLOAD_FIELDS:
        return False, f"Unknown topic: {topic}"

    # Check payload fields for this topic
    required = REQUIRED_PAYLOAD_FIELDS[topic]
    missing_payload = required - set(payload.keys())
    if missing_payload:
        return False, f"Missing payload fields for {topic}: {missing_payload}"

    return True, ""
