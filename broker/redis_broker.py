"""
broker/redis_broker.py

Thin abstraction over Redis pub/sub.
Services call publish() and subscribe() — they never import redis directly.
This means tests can inject a MockBroker without a live Redis instance.
"""

import json
import threading
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class RedisBroker:
    """
    Production broker backed by Redis pub/sub.
    """

    def __init__(self, host: str = "localhost", port: int = 6379):
        import redis
        self._client = redis.Redis(host=host, port=port, decode_responses=True)
        self._pubsub = self._client.pubsub()

    def publish(self, topic: str, event: dict) -> None:
        message = json.dumps(event)
        self._client.publish(topic, message)
        logger.debug(f"Published to {topic}: {event.get('event_id')}")

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        """
        Subscribes to a topic. Calls handler(event_dict) for each message.
        Runs the listener in a background daemon thread.
        """
        self._pubsub.subscribe(topic)

        def _listen():
            for raw in self._pubsub.listen():
                if raw["type"] == "message":
                    try:
                        event = json.loads(raw["data"])
                        handler(event)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message on {topic}: {e}")
                    except Exception as e:
                        logger.error(f"Handler error on {topic}: {e}")

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()
        logger.info(f"Subscribed to {topic}")


class MockBroker:
    """
    In-memory broker for unit testing.
    No Redis required. Supports mocking, deterministic replay, and fault injection.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self.published: list[dict] = []          # all published events, in order
        self._drop_topics: set[str] = set()      # topics to silently drop (fault injection)
        self._duplicate_topics: set[str] = set() # topics to publish twice (fault injection)

    def publish(self, topic: str, event: dict) -> None:
        self.published.append(event)

        if topic in self._drop_topics:
            logger.debug(f"[MockBroker] Dropping message on {topic} (fault injection)")
            return

        handlers = self._subscribers.get(topic, [])
        for handler in handlers:
            handler(event)

        if topic in self._duplicate_topics:
            logger.debug(f"[MockBroker] Duplicating message on {topic} (fault injection)")
            for handler in handlers:
                handler(event)

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    # --- Fault injection helpers ---

    def inject_drop(self, topic: str) -> None:
        """Messages on this topic will be silently dropped."""
        self._drop_topics.add(topic)

    def inject_duplicate(self, topic: str) -> None:
        """Messages on this topic will be delivered twice."""
        self._duplicate_topics.add(topic)

    def clear_faults(self) -> None:
        self._drop_topics.clear()
        self._duplicate_topics.clear()

    def reset(self) -> None:
        self.published.clear()
        self._subscribers.clear()
        self.clear_faults()

    def get_published(self, topic: str) -> list:
        return [e for e in self.published if e.get("topic") == topic]
