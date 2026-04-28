"""
tests/unit/test_events.py

Tests event envelope validation, payload structure, and schema contracts.
No broker or service required. 
"""

import pytest
from events.schemas import Event, Topics
from events.validation import validate_event


# Envelope tests                                                       

def test_valid_event_has_required_envelope_fields():
    event = Event.create(
        topic=Topics.IMAGE_SUBMITTED,
        payload={
            "image_id": "abc123",
            "filename": "dog.jpg",
            "storage_path": "/images/dog.jpg",
        }
    )
    d = event.to_dict()
    assert "topic" in d
    assert "event_id" in d
    assert "timestamp" in d
    assert "payload" in d


def test_event_id_is_unique():
    e1 = Event.create(Topics.IMAGE_SUBMITTED, {"image_id": "a", "filename": "a.jpg", "storage_path": "/a"})
    e2 = Event.create(Topics.IMAGE_SUBMITTED, {"image_id": "b", "filename": "b.jpg", "storage_path": "/b"})
    assert e1.event_id != e2.event_id


def test_missing_envelope_field_fails_validation():
    event = {"topic": "image.submitted", "event_id": "123", "payload": {}}
    # Missing 'timestamp'
    valid, reason = validate_event(event)
    assert not valid
    assert "timestamp" in reason


def test_missing_payload_field_fails_validation():
    event = {
        "topic": "image.submitted",
        "event_id": "abc",
        "timestamp": "2026-04-09T00:00:00Z",
        "payload": {"image_id": "x"},  # missing filename and storage_path
    }
    valid, reason = validate_event(event)
    assert not valid


def test_unknown_topic_fails_validation():
    event = {
        "topic": "unknown.topic",
        "event_id": "abc",
        "timestamp": "2026-04-09T00:00:00Z",
        "payload": {},
    }
    valid, reason = validate_event(event)
    assert not valid
    assert "Unknown topic" in reason


def test_non_dict_payload_fails_validation():
    event = {
        "topic": "image.submitted",
        "event_id": "abc",
        "timestamp": "2026-04-09T00:00:00Z",
        "payload": "this is not a dict",
    }
    valid, reason = validate_event(event)
    assert not valid


# Per-topic payload tests                                              

@pytest.mark.parametrize("topic,payload", [
    (Topics.IMAGE_SUBMITTED, {"image_id": "x", "filename": "x.jpg", "storage_path": "/x"}),
    (Topics.INFERENCE_COMPLETED, {"image_id": "x", "objects": [], "embedding": [0.1, 0.2]}),
    (Topics.ANNOTATION_STORED, {"image_id": "x", "annotation_doc_id": "d1", "objects": [], "embedding": []}),
    (Topics.EMBEDDING_CREATED, {"image_id": "x", "faiss_index_id": "1"}),
    (Topics.ANNOTATION_CORRECTED, {"image_id": "x", "annotation_doc_id": "d1", "corrected_objects": []}),
    (Topics.QUERY_SUBMITTED, {"query_id": "q1", "query_text": "cat", "top_k": 3}),
    (Topics.QUERY_COMPLETED, {"query_id": "q1", "results": []}),
])
def test_valid_payload_passes_validation(topic, payload):
    event = Event.create(topic=topic, payload=payload)
    valid, reason = validate_event(event.to_dict())
    assert valid, f"Expected valid but got: {reason}"
