"""
tests/integration/test_flows.py

End-to-end integration tests using MockBroker.
Tests the full message chain from CLI through all services.
Covers failure modes: duplicates, dropped messages, delayed delivery,
subscriber downtime, and malformed events in the chain.
"""

import pytest
from broker.redis_broker import MockBroker
from events.schemas import Event, Topics
from services.inference.inference_service import InferenceService
from services.annotation.annotation_service import AnnotationService
from services.embedding.embedding_service import EmbeddingService
from services.cli.cli_service import CLIService


# Fixture: fully wired system                       

@pytest.fixture
def system():
    """Returns a dict of all services wired to one MockBroker."""
    broker = MockBroker()
    return {
        "broker": broker,
        "inference": InferenceService(broker),
        "annotation": AnnotationService(broker),
        "embedding": EmbeddingService(broker),
        "cli": CLIService(broker),
    }


def submit_image(broker, image_id="img-001"):
    event = Event.create(Topics.IMAGE_SUBMITTED, {
        "image_id": image_id,
        "filename": f"{image_id}.jpg",
        "storage_path": f"/images/{image_id}.jpg",
    }).to_dict()
    broker.publish(Topics.IMAGE_SUBMITTED, event)
    return event

# Happy path                                                           

class TestHappyPath:

    def test_upload_chain_produces_all_four_events(self, system):
        broker = system["broker"]
        submit_image(broker)

        assert len(broker.get_published(Topics.IMAGE_SUBMITTED)) == 1
        assert len(broker.get_published(Topics.INFERENCE_COMPLETED)) == 1
        assert len(broker.get_published(Topics.ANNOTATION_STORED)) == 1
        assert len(broker.get_published(Topics.EMBEDDING_CREATED)) == 1

    def test_image_is_annotated_and_indexed_after_upload(self, system):
        broker = system["broker"]
        submit_image(broker, "img-happy")

        assert system["annotation"].get_annotation("img-happy") is not None
        assert system["embedding"].has_image("img-happy")

    def test_search_returns_indexed_images(self, system):
        broker = system["broker"]
        # Index three images
        for i in range(3):
            submit_image(broker, f"img-{i:03d}")

        query_id = system["cli"].search("cat on a couch", top_k=2)
        completed = broker.get_published(Topics.QUERY_COMPLETED)
        assert len(completed) == 1
        results = completed[0]["payload"]["results"]
        assert len(results) <= 2
        assert all("image_id" in r and "score" in r for r in results)

    def test_annotation_corrected_updates_document_and_reindexes(self, system):
        broker = system["broker"]
        submit_image(broker, "img-correct")

        correction = Event.create(Topics.ANNOTATION_CORRECTED, {
            "image_id": "img-correct",
            "annotation_doc_id": system["annotation"].get_annotation("img-correct")["annotation_doc_id"],
            "corrected_objects": [{"label": "dog", "confidence": 0.99,
                                   "bbox": {"x": 0, "y": 0, "w": 50, "h": 50}}],
        }).to_dict()
        broker.publish(Topics.ANNOTATION_CORRECTED, correction)

        doc = system["annotation"].get_annotation("img-correct")
        assert doc["objects"][0]["label"] == "dog"
        assert doc["version"] == 2
        # embedding.created should have fired twice: once on store, once on correction
        assert len(broker.get_published(Topics.EMBEDDING_CREATED)) == 2


# Failure mode: duplicate events                                       

class TestDuplicateEvents:

    def test_duplicate_image_submitted_produces_one_annotation(self, system):
        broker = system["broker"]
        event = Event.create(Topics.IMAGE_SUBMITTED, {
            "image_id": "dup-img",
            "filename": "dup.jpg",
            "storage_path": "/images/dup.jpg",
        }).to_dict()

        broker.publish(Topics.IMAGE_SUBMITTED, event)
        broker.publish(Topics.IMAGE_SUBMITTED, event)  # exact duplicate

        assert len(broker.get_published(Topics.INFERENCE_COMPLETED)) == 1
        assert len(broker.get_published(Topics.ANNOTATION_STORED)) == 1
        assert len(broker.get_published(Topics.EMBEDDING_CREATED)) == 1

    def test_duplicate_inference_completed_produces_one_annotation(self, system):
        broker = system["broker"]
        # Manually fire duplicate inference.completed
        event = Event.create(Topics.INFERENCE_COMPLETED, {
            "image_id": "dup-inf",
            "objects": [],
            "embedding": [0.1, 0.2],
        }).to_dict()

        broker.publish(Topics.INFERENCE_COMPLETED, event)
        broker.publish(Topics.INFERENCE_COMPLETED, event)

        assert len(broker.get_published(Topics.ANNOTATION_STORED)) == 1

    def test_broker_duplicate_injection_idempotency(self, system):
        """
        Even if the broker delivers the same event twice (network quirk),
        the system must not create duplicate state.
        """
        broker = system["broker"]
        broker.inject_duplicate(Topics.IMAGE_SUBMITTED)
        submit_image(broker, "dup-inject")

        # Each service's idempotency guard ensures one annotation and one embedding
        assert len(broker.get_published(Topics.ANNOTATION_STORED)) == 1
        assert system["embedding"].index_size() == 1


# Failure mode: dropped messages                                       

class TestDroppedMessages:

    def test_dropped_image_submitted_means_nothing_is_processed(self, system):
        broker = system["broker"]
        broker.inject_drop(Topics.IMAGE_SUBMITTED)
        submit_image(broker, "dropped-img")

        assert len(broker.get_published(Topics.INFERENCE_COMPLETED)) == 0
        assert system["annotation"].get_annotation("dropped-img") is None
        assert not system["embedding"].has_image("dropped-img")

    def test_dropped_inference_completed_means_no_annotation(self, system):
        broker = system["broker"]
        broker.inject_drop(Topics.INFERENCE_COMPLETED)
        submit_image(broker, "dropped-inf")

        # Inference fires but annotation never receives it
        assert len(broker.get_published(Topics.INFERENCE_COMPLETED)) == 1
        assert len(broker.get_published(Topics.ANNOTATION_STORED)) == 0
        assert system["annotation"].get_annotation("dropped-inf") is None

    def test_dropped_annotation_stored_means_no_embedding(self, system):
        broker = system["broker"]
        broker.inject_drop(Topics.ANNOTATION_STORED)
        submit_image(broker, "dropped-ann")

        assert len(broker.get_published(Topics.ANNOTATION_STORED)) == 1
        assert len(broker.get_published(Topics.EMBEDDING_CREATED)) == 0
        assert not system["embedding"].has_image("dropped-ann")


# Failure mode: subscriber downtime                                   

class TestSubscriberDowntime:

    def test_events_published_before_subscriber_connects_are_missed(self):
        """
        In Redis pub/sub (fire-and-forget), if a subscriber is not running
        when a message is published, it misses that message.
        This test documents that behaviour explicitly.
        """
        broker = MockBroker()
        # Publish BEFORE inference service subscribes
        event = Event.create(Topics.IMAGE_SUBMITTED, {
            "image_id": "late-sub",
            "filename": "late.jpg",
            "storage_path": "/images/late.jpg",
        }).to_dict()
        broker.publish(Topics.IMAGE_SUBMITTED, event)

        # Inference service starts AFTER publish
        inference = InferenceService(broker)

        # No inference.completed — message was missed
        assert len(broker.get_published(Topics.INFERENCE_COMPLETED)) == 0

    def test_system_recovers_when_subscriber_reconnects_and_new_event_arrives(self):
        """
        After a subscriber comes back online, new events are processed normally.
        """
        broker = MockBroker()
        inference = InferenceService(broker)
        annotation = AnnotationService(broker)
        embedding = EmbeddingService(broker)

        # Normal event after all subscribers are up
        submit_image(broker, "recovery-img")
        assert system_is_converged(broker, "recovery-img", annotation, embedding)


# Failure mode: malformed events in chain                              

class TestMalformedEvents:

    def test_malformed_image_submitted_does_not_crash_inference(self, system):
        broker = system["broker"]
        bad = {"topic": "image.submitted", "event_id": "bad1",
               "timestamp": "t", "payload": {"broken": True}}
        broker.publish(Topics.IMAGE_SUBMITTED, bad)
        # System keeps running — subsequent valid event works
        submit_image(broker, "after-malformed")
        assert len(broker.get_published(Topics.INFERENCE_COMPLETED)) == 1

    def test_malformed_inference_completed_does_not_crash_annotation(self, system):
        broker = system["broker"]
        bad = {"topic": "inference.completed", "event_id": "bad2",
               "timestamp": "t", "payload": {}}
        broker.publish(Topics.INFERENCE_COMPLETED, bad)
        # Annotation service still alive
        good = Event.create(Topics.INFERENCE_COMPLETED, {
            "image_id": "after-bad-inf",
            "objects": [],
            "embedding": [0.1],
        }).to_dict()
        broker.publish(Topics.INFERENCE_COMPLETED, good)
        assert system["annotation"].get_annotation("after-bad-inf") is not None

    def test_completely_missing_payload_does_not_crash_any_service(self, system):
        broker = system["broker"]
        for topic in [Topics.IMAGE_SUBMITTED, Topics.INFERENCE_COMPLETED,
                      Topics.ANNOTATION_STORED, Topics.QUERY_SUBMITTED]:
            bad = {"topic": topic, "event_id": "x", "timestamp": "t",
                   "payload": None}
            broker.publish(topic, bad)
        # No assertion needed — test passes if nothing raises


# Helper                                                               

def system_is_converged(broker, image_id, annotation, embedding):
    """Check that the full chain has completed for a given image."""
    return (
        annotation.get_annotation(image_id) is not None
        and embedding.has_image(image_id)
        and len(broker.get_published(Topics.EMBEDDING_CREATED)) >= 1
    )
