"""
services/upload/upload_service.py

Responsibility:
    Accept an image from the CLI, save it to the image store and publish image.submitted. 
    
Owns: image file store (data/images/)
Publishes: image.submitted
Subscribes: nothing
"""

import uuid
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone

from events.schemas import Event, Topics
from events.validation import validate_event

logger = logging.getLogger(__name__)

IMAGE_STORE = Path("data/images")


class UploadService:

    def __init__(self, broker):
        self._broker = broker
        IMAGE_STORE.mkdir(parents=True, exist_ok=True)

    def upload(self, source_path: str, filename: str) -> str:
        """
        Save an image and publish image.submitted.
        Returns the image_id.
        """
        image_id = str(uuid.uuid4())
        dest = IMAGE_STORE / f"{image_id}_{filename}"

        # Save image to store
        shutil.copy2(source_path, dest)
        storage_path = str(dest)
        logger.info(f"Saved image {image_id} to {storage_path}")

        # Build and publish event
        event = Event.create(
            topic=Topics.IMAGE_SUBMITTED,
            payload={
                "image_id": image_id,
                "filename": filename,
                "storage_path": storage_path,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        valid, reason = validate_event(event.to_dict())
        if not valid:
            logger.error(f"Will not publish malformed event: {reason}")
            raise ValueError(f"Malformed event: {reason}")

        self._broker.publish(Topics.IMAGE_SUBMITTED, event.to_dict())
        logger.info(f"Published image.submitted for {image_id}")
        return image_id
