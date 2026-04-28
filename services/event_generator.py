"""
services/event_generator.py

The Event Generator is a service that produces simulated events
for testing, replay, and fault injection without needing a real image,
a real user, or a live AI model.

Supports:
  - Publishing any event with a controlled payload
  - Deterministic replay of a fixed scenario
  - Fault injection: duplicates, drops, delayed delivery
  - Contract testing: verify a topic's message structure
"""

import time
import logging
from events.schemas import Event, Topics
from events.validation import validate_event

logger = logging.getLogger(__name__)


class EventGenerator:

    def __init__(self, broker):
        self._broker = broker

    # Single event publishing                                              

    def emit(self, topic: str, payload: dict) -> dict:
        """Build, validate, and publish one event. Returns the event dict."""
        event = Event.create(topic=topic, payload=payload)
        valid, reason = validate_event(event.to_dict())
        if not valid:
            raise ValueError(f"EventGenerator refused to emit invalid event: {reason}")
        self._broker.publish(topic, event.to_dict())
        logger.info(f"[EventGenerator] Emitted {topic} ({event.event_id})")
        return event.to_dict()

    # Scenario replay                                                      

    def replay_upload_scenario(self, image_id: str = "replay-001",
                                filename: str = "replay.jpg") -> list:
        """
        Fire the full upload chain with deterministic payloads.
        Returns list of all emitted events in order.
        Useful for integration tests and demos without real images.
        """
        events = []

        events.append(self.emit(Topics.IMAGE_SUBMITTED, {
            "image_id": image_id,
            "filename": filename,
            "storage_path": f"/images/{image_id}.jpg",
        }))

        return events

    def replay_search_scenario(self, query_text: str = "cat on a couch",
                                top_k: int = 3) -> list:
        """Fire a search query scenario."""
        import uuid
        events = []
        events.append(self.emit(Topics.QUERY_SUBMITTED, {
            "query_id": str(uuid.uuid4()),
            "query_text": query_text,
            "top_k": top_k,
        }))
        return events

    def replay_correction_scenario(self, image_id: str,
                                    annotation_doc_id: str) -> list:
        """Fire an annotation correction for an existing image."""
        events = []
        events.append(self.emit(Topics.ANNOTATION_CORRECTED, {
            "image_id": image_id,
            "annotation_doc_id": annotation_doc_id,
            "corrected_objects": [
                {"label": "corrected_label", "confidence": 0.99,
                 "bbox": {"x": 0, "y": 0, "w": 100, "h": 100},
                 "reviewer_notes": "Manually corrected by reviewer"},
            ],
        }))
        return events

    # Fault injection helpers                                              

    def inject_duplicate(self, topic: str, payload: dict) -> None:
        """Publish the same event twice to test idempotency."""
        event = Event.create(topic=topic, payload=payload).to_dict()
        self._broker.publish(topic, event)
        self._broker.publish(topic, event)
        logger.info(f"[EventGenerator] Injected duplicate on {topic}")

    def inject_malformed(self, topic: str) -> None:
        """Publish a structurally broken event to test robustness."""
        bad = {"topic": topic, "event_id": "malformed-001",
               "timestamp": "bad-timestamp", "payload": {"broken": True}}
        self._broker.publish(topic, bad)
        logger.info(f"[EventGenerator] Injected malformed event on {topic}")

    def inject_delayed(self, topic: str, payload: dict,
                        delay_seconds: float = 0.5) -> None:
        """Publish an event after a delay to simulate late delivery."""
        time.sleep(delay_seconds)
        self.emit(topic, payload)
        logger.info(f"[EventGenerator] Injected delayed event on {topic} "
                    f"after {delay_seconds}s")
