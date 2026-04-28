"""
services/embedding/embedding_service.py

Responsibility:
    Subscribe to annotation.stored and index the embedding vector into FAISS.
    Subscribe to annotation.corrected and re-index with updated embedding.
    Subscribe to query.submitted and run similarity search, publishing query.completed.

Owns: FAISS vector index (simulated with an in-memory dict until FAISS is integrated)
Publishes: embedding.created, query.completed
Subscribes: annotation.stored, annotation.corrected, query.submitted
"""

import logging
import random
from datetime import datetime, timezone

from events.schemas import Event, Topics
from events.validation import validate_event

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, broker):
        self._broker = broker
        # Simulated FAISS index: image_id -> embedding vector
        # Replace with real faiss.IndexFlatL2 when FAISS is integrated
        self._index: dict[str, list[float]] = {}
        self._seen_event_ids = set()

        broker.subscribe(Topics.ANNOTATION_STORED, self._handle_stored)
        broker.subscribe(Topics.ANNOTATION_CORRECTED, self._handle_corrected)
        broker.subscribe(Topics.QUERY_SUBMITTED, self._handle_query)

    # Handlers                                                             

    def _handle_stored(self, event: dict) -> None:
        valid, reason = validate_event(event)
        if not valid:
            logger.error(f"[EmbeddingService] Rejected malformed event: {reason}")
            return

        event_id = event["event_id"]
        if event_id in self._seen_event_ids:
            logger.warning(f"[EmbeddingService] Duplicate event {event_id}, skipping")
            return
        self._seen_event_ids.add(event_id)

        payload = event["payload"]
        image_id = payload["image_id"]

        # overwrite is safe — same image gets same vector

        self._index[image_id] = payload["embedding"]
        faiss_id = str(len(self._index))
        logger.info(f"[EmbeddingService] Indexed embedding for {image_id} (faiss_id={faiss_id})")

        result = Event.create(
            topic=Topics.EMBEDDING_CREATED,
            payload={
                "image_id": image_id,
                "faiss_index_id": faiss_id,
                "stored_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._broker.publish(Topics.EMBEDDING_CREATED, result.to_dict())

    def _handle_corrected(self, event: dict) -> None:
        """Re-index after an annotation correction."""
        valid, reason = validate_event(event)
        if not valid:
            logger.error(f"[EmbeddingService] Rejected malformed correction: {reason}")
            return

        event_id = event["event_id"]
        if event_id in self._seen_event_ids:
            logger.warning(f"[EmbeddingService] Duplicate correction {event_id}, skipping")
            return
        self._seen_event_ids.add(event_id)

        payload = event["payload"]
        image_id = payload["image_id"]

        if image_id not in self._index:
            logger.warning(f"[EmbeddingService] No existing index entry for {image_id}, skipping re-index")
            return

        # Re-index: simulate a corrected embedding by re-randomising
        # In production this would use the updated embedding from the payload

        new_embedding = self._simulate_embedding(len(self._index[image_id]))
        self._index[image_id] = new_embedding
        faiss_id = str(list(self._index.keys()).index(image_id))
        logger.info(f"[EmbeddingService] Re-indexed embedding for {image_id}")

        result = Event.create(
            topic=Topics.EMBEDDING_CREATED,
            payload={
                "image_id": image_id,
                "faiss_index_id": faiss_id,
                "stored_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._broker.publish(Topics.EMBEDDING_CREATED, result.to_dict())

    def _handle_query(self, event: dict) -> None:
        valid, reason = validate_event(event)
        if not valid:
            logger.error(f"[EmbeddingService] Rejected malformed query: {reason}")
            return

        payload = event["payload"]
        query_id = payload["query_id"]
        top_k = int(payload["top_k"])

        # Simulate similarity search
        # In production: encode query_text to embedding, call faiss.search()

        results = self._simulate_search(top_k)
        logger.info(f"[EmbeddingService] Query {query_id} — returning {len(results)} results")

        result = Event.create(
            topic=Topics.QUERY_COMPLETED,
            payload={
                "query_id": query_id,
                "results": results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._broker.publish(Topics.QUERY_COMPLETED, result.to_dict())

    # Simulation helpers                                                   

    def _simulate_search(self, top_k: int) -> list:
        """Return top_k image_ids from the index with simulated scores."""
        candidates = list(self._index.keys())
        random.shuffle(candidates)
        top = candidates[:top_k]
        return [
            {"image_id": img_id, "score": round(random.uniform(0.75, 0.99), 4)}
            for img_id in top
        ]

    def _simulate_embedding(self, dim: int) -> list:
        vec = [random.gauss(0, 1) for _ in range(dim)]
        magnitude = sum(v ** 2 for v in vec) ** 0.5
        return [round(v / magnitude, 6) for v in vec]

    # Internal access (for tests only)                                    

    def index_size(self) -> int:
        return len(self._index)

    def has_image(self, image_id: str) -> bool:
        return image_id in self._index
