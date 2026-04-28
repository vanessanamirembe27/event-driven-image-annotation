"""
tests/unit/test_event_generator.py

Tests for EventGenerator in isolation using MockBroker.
The generator can be tested without a live broker by mocking publish().
"""

import pytest
from broker.redis_broker import MockBroker
from services.event_generator import EventGenerator
from events.schemas import Topics


@pytest.fixture
def gen():
    broker = MockBroker()
    return EventGenerator(broker), broker


def test_emit_publishes_to_broker(gen):
    generator, broker = gen
    generator.emit(Topics.IMAGE_SUBMITTED, {
        "image_id": "g-001",
        "filename": "g.jpg",
        "storage_path": "/images/g.jpg",
    })
    assert len(broker.get_published(Topics.IMAGE_SUBMITTED)) == 1


def test_emit_raises_on_invalid_payload(gen):
    generator, _ = gen
    with pytest.raises(ValueError):
        generator.emit(Topics.IMAGE_SUBMITTED, {"broken": True})


def test_emitted_event_has_all_envelope_fields(gen):
    generator, broker = gen
    generator.emit(Topics.IMAGE_SUBMITTED, {
        "image_id": "g-002",
        "filename": "g.jpg",
        "storage_path": "/images/g.jpg",
    })
    event = broker.published[0]
    assert "topic" in event
    assert "event_id" in event
    assert "timestamp" in event
    assert "payload" in event


def test_replay_upload_scenario_fires_image_submitted(gen):
    generator, broker = gen
    events = generator.replay_upload_scenario("replay-img")
    assert len(events) == 1
    assert events[0]["topic"] == Topics.IMAGE_SUBMITTED
    assert events[0]["payload"]["image_id"] == "replay-img"


def test_replay_search_scenario_fires_query_submitted(gen):
    generator, broker = gen
    events = generator.replay_search_scenario("fluffy dog", top_k=5)
    assert len(events) == 1
    assert events[0]["topic"] == Topics.QUERY_SUBMITTED
    assert events[0]["payload"]["query_text"] == "fluffy dog"
    assert events[0]["payload"]["top_k"] == 5


def test_inject_duplicate_publishes_same_event_twice(gen):
    generator, broker = gen
    generator.inject_duplicate(Topics.IMAGE_SUBMITTED, {
        "image_id": "dup-g",
        "filename": "dup.jpg",
        "storage_path": "/images/dup.jpg",
    })
    published = broker.get_published(Topics.IMAGE_SUBMITTED)
    assert len(published) == 2
    # Both should have the same event_id (true duplicate)
    assert published[0]["event_id"] == published[1]["event_id"]


def test_inject_malformed_publishes_broken_event(gen):
    generator, broker = gen
    generator.inject_malformed(Topics.IMAGE_SUBMITTED)
    published = broker.get_published(Topics.IMAGE_SUBMITTED)
    assert len(published) == 1
    assert published[0]["payload"] == {"broken": True}


def test_generator_works_without_live_broker():
    """
    EventGenerator can be tested by injecting any object with a publish() method.
    No Redis connection needed.
    """
    published_calls = []

    class SpyBroker:
        def publish(self, topic, event):
            published_calls.append((topic, event))

    gen = EventGenerator(SpyBroker())
    gen.emit(Topics.QUERY_SUBMITTED, {
        "query_id": "spy-q",
        "query_text": "test",
        "top_k": 3,
    })
    assert len(published_calls) == 1
    assert published_calls[0][0] == Topics.QUERY_SUBMITTED
