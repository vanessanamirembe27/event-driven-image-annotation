"""
services/annotation/annotation_service.py

Responsibility:
    Subscribe to inference.completed
    Write a nested annotation document to the document store. 
    Publish annotation.stored.
    Handle annotation.corrected by updating the existing document.

Owns: document store (simulated in-memory dict standing in for MongoDB)
Publishes: annotation.stored
Subscribes: inference.completed, annotation.corrected
"""

import uuid
import logging
from datetime import datetime, timezone

from events.schemas import Event, Topics
from events.validation import validate_event
from typing import Optional, List, Any

logger = logging.getLogger(__name__)


class AnnotationService:

    def __init__(self, broker):
        self._broker = broker
        self._db: dict[str, dict] = {}    # image_id -> annotation document
        self._seen_event_ids = set()       # idempotency guard

        broker.subscribe(Topics.INFERENCE_COMPLETED, self._handle_inference)
        broker.subscribe(Topics.ANNOTATION_CORRECTED, self._handle_correction)

    # Handlers                                                             

    def _handle_inference(self, event: dict) -> None:
        valid, reason = validate_event(event)
        if not valid:
            logger.error(f"[AnnotationService] Rejected malformed event: {reason}")
            return

        event_id = event["event_id"]
        if event_id in self._seen_event_ids:
            logger.warning(f"[AnnotationService] Duplicate event {event_id}, skipping")
            return
        self._seen_event_ids.add(event_id)

        payload = event["payload"]
        image_id = payload["image_id"]

        # if document already exists for this image, skip write
        if image_id in self._db:
            logger.warning(f"[AnnotationService] Document for {image_id} already exists, skipping")
            return

        doc_id = str(uuid.uuid4())
        document = {
            "annotation_doc_id": doc_id,
            "image_id": image_id,
            "objects": payload["objects"],
            "embedding": payload["embedding"],
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }

        # Write to document store
        self._db[image_id] = document
        logger.info(f"[AnnotationService] Stored annotation {doc_id} for image {image_id}")

        # Publish annotation.stored — carry embedding forward so
        # Embedding Service does not need to query this DB
        result = Event.create(
            topic=Topics.ANNOTATION_STORED,
            payload={
                "image_id": image_id,
                "annotation_doc_id": doc_id,
                "objects": payload["objects"],
                "embedding": payload["embedding"],
                "stored_at": document["stored_at"],
            }
        )
        self._broker.publish(Topics.ANNOTATION_STORED, result.to_dict())

    def _handle_correction(self, event: dict) -> None:
        valid, reason = validate_event(event)
        if not valid:
            logger.error(f"[AnnotationService] Rejected malformed correction event: {reason}")
            return

        event_id = event["event_id"]
        if event_id in self._seen_event_ids:
            logger.warning(f"[AnnotationService] Duplicate correction {event_id}, skipping")
            return
        self._seen_event_ids.add(event_id)

        payload = event["payload"]
        image_id = payload["image_id"]

        if image_id not in self._db:
            logger.error(f"[AnnotationService] Cannot correct unknown image {image_id}")
            return

        # Update document in place, bump version
        self._db[image_id]["objects"] = payload["corrected_objects"]
        self._db[image_id]["version"] += 1
        self._db[image_id]["corrected_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"[AnnotationService] Corrected annotation for {image_id}")

    # Query interface (internal — only called by this service's own tests) 

    def get_annotation(self, image_id: str) -> Optional[dict]:
        return self._db.get(image_id)

    def all_image_ids(self) -> list:
        return list(self._db.keys())
