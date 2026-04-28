"""
tests/unit/test_broker.py

Tests the MockBroker in isolation — no services, no Redis.
Covers publish/subscribe, fault injection, and deterministic replay.
"""

from broker.redis_broker import MockBroker
from events.schemas import Event, Topics


def make_event(topic=Topics.IMAGE_SUBMITTED):
    return Event.create(topic=topic, payload={
        "image_id": "test-id",
        "filename": "test.jpg",
        "storage_path": "/images/test.jpg",
    }).to_dict()


# Basic pub/sub                                                        

def test_subscriber_receives_published_event():
    broker = MockBroker()
    received = []
    broker.subscribe(Topics.IMAGE_SUBMITTED, lambda e: received.append(e))
    event = make_event()
    broker.publish(Topics.IMAGE_SUBMITTED, event)
    assert len(received) == 1
    assert received[0]["event_id"] == event["event_id"]


def test_subscriber_on_different_topic_does_not_receive_event():
    broker = MockBroker()
    received = []
    broker.subscribe(Topics.INFERENCE_COMPLETED, lambda e: received.append(e))
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    assert len(received) == 0


def test_multiple_subscribers_all_receive():
    broker = MockBroker()
    r1, r2 = [], []
    broker.subscribe(Topics.IMAGE_SUBMITTED, lambda e: r1.append(e))
    broker.subscribe(Topics.IMAGE_SUBMITTED, lambda e: r2.append(e))
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    assert len(r1) == 1
    assert len(r2) == 1


def test_published_events_are_recorded():
    broker = MockBroker()
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    assert len(broker.published) == 2


def test_get_published_filters_by_topic():
    broker = MockBroker()
    broker.publish(Topics.IMAGE_SUBMITTED, make_event(Topics.IMAGE_SUBMITTED))
    broker.publish(Topics.QUERY_SUBMITTED, make_event(Topics.QUERY_SUBMITTED))
    assert len(broker.get_published(Topics.IMAGE_SUBMITTED)) == 1
    assert len(broker.get_published(Topics.QUERY_SUBMITTED)) == 1


# Fault injection                                                      

def test_dropped_message_not_delivered():
    broker = MockBroker()
    received = []
    broker.subscribe(Topics.IMAGE_SUBMITTED, lambda e: received.append(e))
    broker.inject_drop(Topics.IMAGE_SUBMITTED)
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    assert len(received) == 0


def test_duplicate_message_delivered_twice():
    broker = MockBroker()
    received = []
    broker.subscribe(Topics.IMAGE_SUBMITTED, lambda e: received.append(e))
    broker.inject_duplicate(Topics.IMAGE_SUBMITTED)
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    assert len(received) == 2


def test_clear_faults_restores_normal_delivery():
    broker = MockBroker()
    received = []
    broker.subscribe(Topics.IMAGE_SUBMITTED, lambda e: received.append(e))
    broker.inject_drop(Topics.IMAGE_SUBMITTED)
    broker.clear_faults()
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    assert len(received) == 1


def test_reset_clears_all_state():
    broker = MockBroker()
    received = []
    broker.subscribe(Topics.IMAGE_SUBMITTED, lambda e: received.append(e))
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    broker.reset()
    assert len(broker.published) == 0
    broker.publish(Topics.IMAGE_SUBMITTED, make_event())
    assert len(received) == 1   # subscriber was cleared on reset
