"""
services/inference/inference_service.py

Responsibility:
    Subscribe to image.submitted. 
    Simulate object detection by producing fake but realistic detected objects and a mock embedding vector.
    Publish inference.completed.

Owns: nothing (stateless)
Publishes: inference.completed
Subscribes: image.submitted
"""

import random
import logging
from datetime import datetime, timezone

from events.schemas import Event, Topics
from events.validation import validate_event

logger = logging.getLogger(__name__)

SIMULATED_LABELS = ["cat", "dog", "couch", "chair", "table", "person",
                    "bicycle", "car", "plant", "laptop"]
EMBEDDING_DIM = 128


class InferenceService:

    def __init__(self, broker):
        self._broker = broker
        self._seen_event_ids = set()   # idempotency guard
        broker.subscribe(Topics.IMAGE_SUBMITTED, self._handle)

    def _handle(self, event: dict) -> None:
        # Robustness: validate before processing 
        valid, reason = validate_event(event)
        if not valid:
            logger.error(f"[InferenceService] Rejected malformed event: {reason}")
            return

        #  skip duplicate events 
        event_id = event["event_id"]
        if event_id in self._seen_event_ids:
            logger.warning(f"[InferenceService] Duplicate event {event_id}, skipping")
            return
        self._seen_event_ids.add(event_id)

        payload = event["payload"]
        image_id = payload["image_id"]
        logger.info(f"[InferenceService] Processing image {image_id}")

        # Simulate detection 
        objects = self._simulate_detection()
        embedding = self._simulate_embedding()

        result = Event.create(
            topic=Topics.INFERENCE_COMPLETED,
            payload={
                "image_id": image_id,
                "objects": objects,
                "embedding": embedding,
                "inferred_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        self._broker.publish(Topics.INFERENCE_COMPLETED, result.to_dict())
        logger.info(f"[InferenceService] Published inference.completed for {image_id}")

    def _simulate_detection(self) -> list:
        """Produce 1–4 fake detected objects with random labels and bboxes."""
        n = random.randint(1, 4)
        objects = []
        for _ in range(n):
            objects.append({
                "label": random.choice(SIMULATED_LABELS),
                "confidence": round(random.uniform(0.70, 0.99), 2),
                "bbox": {
                    "x": random.randint(0, 100),
                    "y": random.randint(0, 100),
                    "w": random.randint(20, 200),
                    "h": random.randint(20, 200),
                },
                "reviewer_notes": None,
            })
        return objects

    def _simulate_embedding(self) -> list:
        """Return a random unit-normalised float vector."""
        vec = [random.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
        magnitude = sum(v ** 2 for v in vec) ** 0.5
        return [round(v / magnitude, 6) for v in vec]
