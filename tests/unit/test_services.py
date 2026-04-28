"""
tests/unit/test_services.py

Per-service unit tests using MockBroker.
Covers: normal flow, idempotency, malformed event robustness,
        duplicate events, dropped messages, and correction handling.
"""

import pytest
from broker.redis_broker import MockBroker
from events.schemas import Event, Topics
from services.inference.inference_service import InferenceService
from services.annotation.annotation_service import AnnotationService
from services.embedding.embedding_service import EmbeddingService
from services.cli.cli_service import CLIService



# Helpers                                                             

def image_submitted_event(image_id="img-001"):
    return Event.create(Topics.IMAGE_SUBMITTED, {
        "image_id": image_id,
        "filename": "test.jpg",
        "storage_path": "/images/test.jpg",
    }).to_dict()

def inference_completed_event(image_id="img-001"):
    return Event.create(Topics.INFERENCE_COMPLETED, {
        "image_id": image_id,
        "objects": [{"label": "cat", "confidence": 0.9,
                     "bbox": {"x": 0, "y": 0, "w": 50, "h": 50},
                     "reviewer_notes": None}],
        "embedding": [0.1, 0.2, 0.3],
    }).to_dict()

def annotation_stored_event(image_id="img-001", doc_id="doc-001"):
    return Event.create(Topics.ANNOTATION_STORED, {
        "image_id": image_id,
        "annotation_doc_id": doc_id,
        "objects": [{"label": "cat", "confidence": 0.9,
                     "bbox": {"x": 0, "y": 0, "w": 50, "h": 50}}],
        "embedding": [0.1, 0.2, 0.3],
    }).to_dict()

def annotation_corrected_event(image_id="img-001", doc_id="doc-001"):
    return Event.create(Topics.ANNOTATION_CORRECTED, {
        "image_id": image_id,
        "annotation_doc_id": doc_id,
        "corrected_objects": [{"label": "dog", "confidence": 0.95,
                                "bbox": {"x": 0, "y": 0, "w": 50, "h": 50}}],
    }).to_dict()

def query_submitted_event(query_id="q-001", query_text="cat", top_k=3):
    return Event.create(Topics.QUERY_SUBMITTED, {
        "query_id": query_id,
        "query_text": query_text,
        "top_k": top_k,
    }).to_dict()


# InferenceService                                                    

class TestInferenceService:

    def setup_method(self):
        self.broker = MockBroker()
        self.service = InferenceService(self.broker)

    def test_publishes_inference_completed_on_valid_event(self):
        self.broker.publish(Topics.IMAGE_SUBMITTED, image_submitted_event())
        published = self.broker.get_published(Topics.INFERENCE_COMPLETED)
        assert len(published) == 1
        assert published[0]["payload"]["image_id"] == "img-001"
        assert "objects" in published[0]["payload"]
        assert "embedding" in published[0]["payload"]

    def test_idempotency_duplicate_event_not_processed_twice(self):
        event = image_submitted_event()
        self.broker.publish(Topics.IMAGE_SUBMITTED, event)
        self.broker.publish(Topics.IMAGE_SUBMITTED, event)  # same event_id
        published = self.broker.get_published(Topics.INFERENCE_COMPLETED)
        assert len(published) == 1

    def test_malformed_event_does_not_crash_service(self):
        bad_event = {"topic": "image.submitted", "event_id": "x", "timestamp": "t",
                     "payload": {}}  # missing image_id, filename, storage_path
        # Should not raise
        self.broker.publish(Topics.IMAGE_SUBMITTED, bad_event)
        # No inference.completed should be published
        assert len(self.broker.get_published(Topics.INFERENCE_COMPLETED)) == 0

    def test_dropped_message_means_no_inference_completed(self):
        self.broker.inject_drop(Topics.IMAGE_SUBMITTED)
        self.broker.publish(Topics.IMAGE_SUBMITTED, image_submitted_event())
        assert len(self.broker.get_published(Topics.INFERENCE_COMPLETED)) == 0


# AnnotationService                                                    

class TestAnnotationService:

    def setup_method(self):
        self.broker = MockBroker()
        self.service = AnnotationService(self.broker)

    def test_stores_document_and_publishes_annotation_stored(self):
        self.broker.publish(Topics.INFERENCE_COMPLETED, inference_completed_event())
        assert self.service.get_annotation("img-001") is not None
        published = self.broker.get_published(Topics.ANNOTATION_STORED)
        assert len(published) == 1

    def test_annotation_stored_carries_embedding_forward(self):
        self.broker.publish(Topics.INFERENCE_COMPLETED, inference_completed_event())
        published = self.broker.get_published(Topics.ANNOTATION_STORED)
        assert "embedding" in published[0]["payload"]

    def test_idempotency_duplicate_does_not_create_two_documents(self):
        event = inference_completed_event()
        self.broker.publish(Topics.INFERENCE_COMPLETED, event)
        self.broker.publish(Topics.INFERENCE_COMPLETED, event)
        # Only one annotation.stored should have been published
        assert len(self.broker.get_published(Topics.ANNOTATION_STORED)) == 1

    def test_malformed_event_does_not_crash(self):
        bad = {"topic": "inference.completed", "event_id": "x",
               "timestamp": "t", "payload": {"image_id": "img-001"}}
        self.broker.publish(Topics.INFERENCE_COMPLETED, bad)
        assert self.service.get_annotation("img-001") is None

    def test_correction_updates_document(self):
        self.broker.publish(Topics.INFERENCE_COMPLETED, inference_completed_event())
        self.broker.publish(Topics.ANNOTATION_CORRECTED, annotation_corrected_event())
        doc = self.service.get_annotation("img-001")
        assert doc["objects"][0]["label"] == "dog"
        assert doc["version"] == 2

    def test_correction_on_unknown_image_does_not_crash(self):
        self.broker.publish(Topics.ANNOTATION_CORRECTED, annotation_corrected_event("unknown-img"))


# EmbeddingService                                                     

class TestEmbeddingService:

    def setup_method(self):
        self.broker = MockBroker()
        self.service = EmbeddingService(self.broker)

    def test_indexes_embedding_and_publishes_embedding_created(self):
        self.broker.publish(Topics.ANNOTATION_STORED, annotation_stored_event())
        assert self.service.has_image("img-001")
        published = self.broker.get_published(Topics.EMBEDDING_CREATED)
        assert len(published) == 1

    def test_idempotency_duplicate_annotation_stored(self):
        event = annotation_stored_event()
        self.broker.publish(Topics.ANNOTATION_STORED, event)
        self.broker.publish(Topics.ANNOTATION_STORED, event)
        assert self.service.index_size() == 1
        assert len(self.broker.get_published(Topics.EMBEDDING_CREATED)) == 1

    def test_malformed_event_does_not_crash(self):
        bad = {"topic": "annotation.stored", "event_id": "x",
               "timestamp": "t", "payload": {}}
        self.broker.publish(Topics.ANNOTATION_STORED, bad)
        assert self.service.index_size() == 0

    def test_query_returns_results_up_to_top_k(self):
        # Index 5 images first
        for i in range(5):
            self.broker.publish(Topics.ANNOTATION_STORED,
                                annotation_stored_event(f"img-{i:03d}", f"doc-{i:03d}"))
        self.broker.publish(Topics.QUERY_SUBMITTED, query_submitted_event(top_k=3))
        completed = self.broker.get_published(Topics.QUERY_COMPLETED)
        assert len(completed) == 1
        assert len(completed[0]["payload"]["results"]) <= 3

    def test_query_with_empty_index_returns_empty_results(self):
        self.broker.publish(Topics.QUERY_SUBMITTED, query_submitted_event())
        completed = self.broker.get_published(Topics.QUERY_COMPLETED)
        assert completed[0]["payload"]["results"] == []

    def test_correction_triggers_reindex(self):
        self.broker.publish(Topics.ANNOTATION_STORED, annotation_stored_event())
        old_embedding = self.service._index["img-001"][:]
        self.broker.publish(Topics.ANNOTATION_CORRECTED, annotation_corrected_event())
        new_embedding = self.service._index["img-001"]
        # Embeddings should differ after re-indexing
        assert old_embedding != new_embedding


# CLIService                                                           

class TestCLIService:

    def setup_method(self):
        self.broker = MockBroker()
        self.cli = CLIService(self.broker)

    def test_search_publishes_query_submitted(self):
        self.cli.search("fluffy cat", top_k=3)
        published = self.broker.get_published(Topics.QUERY_SUBMITTED)
        assert len(published) == 1
        assert published[0]["payload"]["query_text"] == "fluffy cat"
        assert published[0]["payload"]["top_k"] == 3

    def test_cli_receives_query_completed_and_caches_results(self):
        query_id = self.cli.search("dog on couch", top_k=2)
        # Simulate embedding service responding
        result_event = Event.create(Topics.QUERY_COMPLETED, {
            "query_id": query_id,
            "results": [
                {"image_id": "img-001", "score": 0.95},
                {"image_id": "img-002", "score": 0.88},
            ],
        }).to_dict()
        self.broker.publish(Topics.QUERY_COMPLETED, result_event)
        results = self.cli.get_results(query_id)
        assert results is not None
        assert len(results) == 2

    def test_cli_does_not_crash_on_malformed_query_completed(self):
        bad = {"topic": "query.completed", "event_id": "x",
               "timestamp": "t", "payload": {}}
        self.broker.publish(Topics.QUERY_COMPLETED, bad)


# System guarantee: eventual consistency end-to-end                   

class TestSystemGuarantees:

    def test_full_upload_chain_converges(self):
        """
        Verify that publishing image.submitted causes annotation and
        embedding to be stored — eventual consistency through the chain.
        """
        broker = MockBroker()
        inference = InferenceService(broker)
        annotation = AnnotationService(broker)
        embedding = EmbeddingService(broker)

        event = image_submitted_event("chain-test")
        broker.publish(Topics.IMAGE_SUBMITTED, event)

        # Chain: image.submitted -> inference.completed -> annotation.stored -> embedding.created
        assert annotation.get_annotation("chain-test") is not None
        assert embedding.has_image("chain-test")
        assert len(broker.get_published(Topics.EMBEDDING_CREATED)) == 1

    def test_duplicate_image_submitted_produces_one_annotation(self):
        broker = MockBroker()
        InferenceService(broker)
        annotation = AnnotationService(broker)
        EmbeddingService(broker)

        event = image_submitted_event("dup-test")
        broker.publish(Topics.IMAGE_SUBMITTED, event)
        broker.publish(Topics.IMAGE_SUBMITTED, event)   # duplicate

        assert len(broker.get_published(Topics.ANNOTATION_STORED)) == 1
        assert len(broker.get_published(Topics.EMBEDDING_CREATED)) == 1
