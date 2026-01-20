import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from confluent_kafka.admin import AdminClient, NewTopic
DEFAULT_KAFKA_BROKER = "localhost:9092"

class KafkaTopicManager:
    """Create Kafka topics if they do not already exist."""

    def __init__(self, broker: str = DEFAULT_KAFKA_BROKER, logger=None):
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.broker = broker
        self.admin = AdminClient({"bootstrap.servers": self.broker})

    def create_topic(
        self,
        topic_name: str,
        num_partitions: int = 1,
        replication_factor: int = 1,
    ):
        existing_topics = self.admin.list_topics(timeout=5).topics

        if topic_name in existing_topics:
            return

        topic = NewTopic(
            topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
        )
        fs = self.admin.create_topics([topic])
        try:
            fs[topic_name].result()
            self.logger.info(f"Topic '{topic_name}' created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to create topic '{topic_name}': {e}")
