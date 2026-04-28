"""
services/cli/cli_service.py

Responsibility:
    User-facing entry point. Handles upload and search commands.
    Publishes image.submitted (via UploadService) and query.submitted.
    Subscribes to query.completed and renders results.

Owns: nothing
Publishes: image.submitted (delegated to UploadService), query.submitted
Subscribes: query.completed

Important: CLI never calls MongoDB or FAISS directly.
           All data access goes through the broker.
"""

import uuid
import logging
from datetime import datetime, timezone

from events.schemas import Event, Topics
from events.validation import validate_event
from services.upload.upload_service import UploadService
from typing import Optional

logger = logging.getLogger(__name__)


class CLIService:

    def __init__(self, broker):
        self._broker = broker
        self._upload_service = UploadService(broker)
        self._pending_queries: dict[str, list] = {}   # query_id -> results when ready

        broker.subscribe(Topics.QUERY_COMPLETED, self._handle_query_completed)


    # Commands                                                             

    def upload(self, source_path: str, filename: str) -> str:
        """Upload an image. Returns image_id."""
        image_id = self._upload_service.upload(source_path, filename)
        print(f"[CLI] Image submitted: {image_id}")
        return image_id

    def search(self, query_text: str, top_k: int = 5) -> str:
        """Submit a search query. Returns query_id."""
        query_id = str(uuid.uuid4())

        event = Event.create(
            topic=Topics.QUERY_SUBMITTED,
            payload={
                "query_id": query_id,
                "query_text": query_text,
                "top_k": top_k,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        valid, reason = validate_event(event.to_dict())
        if not valid:
            logger.error(f"[CLI] Will not publish malformed query: {reason}")
            raise ValueError(reason)

        self._broker.publish(Topics.QUERY_SUBMITTED, event.to_dict())
        print(f"[CLI] Query submitted: '{query_text}' (top_k={top_k}, id={query_id})")
        return query_id

    # Subscriber                                                           

    def _handle_query_completed(self, event: dict) -> None:
        valid, reason = validate_event(event)
        if not valid:
            logger.error(f"[CLI] Rejected malformed query.completed: {reason}")
            return

        payload = event["payload"]
        query_id = payload["query_id"]
        results = payload["results"]

        self._pending_queries[query_id] = results
        self._render_results(query_id, results)

    def _render_results(self, query_id: str, results: list) -> None:
        print(f"\n[CLI] Results for query {query_id}:")
        if not results:
            print("  No matching images found.")
            return
        for i, r in enumerate(results, 1):
            print(f"  {i}. image_id={r['image_id']}  score={r['score']:.4f}")
        print()

    def get_results(self, query_id: str) -> Optional[list]:
        """Return cached results for a query_id (used in tests)."""
        return self._pending_queries.get(query_id)
