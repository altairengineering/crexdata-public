import json
import logging
from typing import Any, Dict, Optional

from confluent_kafka import Consumer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class KafkaConsumer:
    """Consume JSON messages from a single Kafka topic and return decoded Python objects."""


    def __init__(self, topic: str, config: Optional[Dict[str, Any]] = None, logger=None) -> None:
        self.logger = logging.getLogger(__name__) if logger is None else logger
        config = dict(config) if config is not None else {}

        self.topic = topic
        self.broker = config.get("bootstrap.servers", "localhost:9092")
        config.setdefault("bootstrap.servers", self.broker)
        config.setdefault("enable.auto.commit", True)

        self.consumer = Consumer(config)
        self.consumer.subscribe([self.topic])

        self.logger.info(
            f"KafkaConsumer subscribed to topic='{self.topic}' "
            f"on broker='{self.broker}'"
        )

    def receive(self, timeout: float = 1.0):
        """Poll for a JSON message and return its decoded payload, or None on failure."""
        
        while True:
            msg = self.consumer.poll(timeout=timeout)

            if msg is None:
                continue

            if msg.error():
                self.logger.error(f"Kafka error: {msg.error()}")
                continue

            raw = msg.value()
            if raw is None:
                self.logger.warning("Received empty message, skipping")
                continue

            try:
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                data = json.loads(text)
                return data
            except Exception as e:
                self.logger.error(f"Failed to decode JSON message: {e}")
                return None

    def close(self) -> None:
        """Close the Kafka consumer cleanly."""
        if self.consumer is not None:
            self.consumer.close()
            self.logger.info("KafkaConsumer closed")
