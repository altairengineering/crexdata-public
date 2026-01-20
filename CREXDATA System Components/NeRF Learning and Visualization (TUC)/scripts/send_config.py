#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kafka_utils import KafkaProducer, KafkaTopicManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(file_path):

    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading configuration: {e}")
        return None


def main(args):
    config_path, broker, topic = args.configPath, args.broker, args.topic
    topic_manager = KafkaTopicManager(broker)
    kafka_producer = KafkaProducer({"bootstrap.servers": broker})

    topic_manager.create_topic(topic)
    config = load_config(config_path)

    if config is None:
        logging.error("Aborting: Configuration could not be loaded.")
        return

    try:
        kafka_producer.send(topic=topic, message=config)
        logging.info(f"Sent config from {config_path} to topic '{topic}'")
    finally:
        kafka_producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send NeRF job config to Kafka.")
    parser.add_argument(
        "--configPath", type=str, required=True, help="Path to JSON config file."
    )
    parser.add_argument(
        "--broker",
        type=str,
        default="localhost:9092",
        help="Kafka broker",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="nerfConfigs",  # <<< align with mediator
        help="Topic to send the json formatted NeRF configuration.",
    )
    args = parser.parse_args()
    main(args)
