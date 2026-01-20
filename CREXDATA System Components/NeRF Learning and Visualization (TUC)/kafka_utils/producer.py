import json
import logging
from typing import Any, Dict, Optional

from confluent_kafka import Producer, KafkaException

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class KafkaProducer:
    """
    Minimal Kafka producer for JSON messages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None) -> None:
        self.logger = logging.getLogger(__name__) if logger is None else logger
        config = dict(config) if config is not None else {}
        self.broker = config.get("bootstrap.servers", "localhost:9092")
        config.setdefault("bootstrap.servers", self.broker)
        self.producer = Producer(config)

    def send(self, topic: str, message: Any) -> None:
        """
        Send a single JSON-encoded message to Kafka.
        """
        try:
            payload = json.dumps(message).encode("utf-8")
            self.producer.produce(topic, value=payload)
            self.producer.flush()
            self.logger.info(f"✅ JSON message sent to topic '{topic}'")
        except KafkaException as e:
            self.logger.error(f"❌ Failed to send message to {topic}: {e}")
            raise

    def close(self) -> None:
        self.producer.flush()
