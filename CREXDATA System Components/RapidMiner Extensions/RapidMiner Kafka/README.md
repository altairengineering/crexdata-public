# Kafka Consumer for RapidMiner

A custom RapidMiner operator that reads messages from Apache Kafka topics with flexible consumption modes.

## Overview

This operator enables RapidMiner to consume messages from Kafka brokers without authentication. It supports multiple reading strategies (reading all existing messages, listening for new messages, or reading the last N messages) and returns the data as a structured DataFrame for further processing in RapidMiner workflows.

## Features

- **Multiple Read Modes**: Choose how you want to consume messages
- **Partition-Aware**: Automatically handles multiple topic partitions
- **Timeout Control**: Configurable timeout for message consumption
- **Message Metadata**: Captures timestamp, partition, offset, topic, and key information
- **Error Handling**: Robust error handling with detailed logging

## Installation

1. Ensure you have the Kafka Python client installed:

   ```bash
   pip install kafka-python pandas
   ```

2. Place the extension in your RapidMiner extensions directory

## Parameters

| Parameter         | Type    | Default        | Description                                                            |
| ----------------- | ------- | -------------- | ---------------------------------------------------------------------- |
| `TOPIC`           | String  | Required       | Kafka topic name to consume from                                       |
| `READ_MODE`       | String  | `all_existing` | How to read messages: `all_existing`, `new_only`, or `last_n_messages` |
| `MAX_MESSAGES`    | Integer | 0              | Maximum number of messages to read (0 = unlimited)                     |
| `TIMEOUT_SECONDS` | Integer | 10             | Timeout in seconds for message consumption                             |

The Kafka broker configuration is provided via the RapidMiner Connection input.

## Read Modes

### `all_existing`

Reads **all messages** from the beginning of the topic across all partitions. Useful for full data extraction or reprocessing historical data.

- Starts from the earliest message (offset 0)
- Processes all partitions sequentially
- Respects `MAX_MESSAGES` limit if set

### `new_only`

Listens for **new messages only** starting from the latest offset. Useful for real-time data streaming or continuous monitoring.

- Skips all existing messages
- Waits for new messages to arrive
- Auto-commits offsets (consumer group tracking)

### `last_n_messages`

Reads the **last N messages** across all partitions. Useful for recent data snapshots or windowed analysis.

- Calculates appropriate start offset based on `MAX_MESSAGES`
- Sorts messages by timestamp
- Returns the most recent messages

## Output

Returns a pandas DataFrame with the following columns:

| Column      | Type    | Description                                           |
| ----------- | ------- | ----------------------------------------------------- |
| `message`   | String  | The message content (JSON stringified)                |
| `timestamp` | String  | ISO format timestamp of when the message was produced |
| `partition` | Integer | Kafka partition where the message resides             |
| `offset`    | Integer | Message offset within its partition                   |
| `topic`     | String  | Topic name                                            |

## Usage Example

### Read All Existing Messages

```
TOPIC: "sensor-data"
READ_MODE: "all_existing"
MAX_MESSAGES: 0
TIMEOUT_SECONDS: 10
```

This will read all historical messages from the sensor-data topic.

### Listen for New Messages (Max 1000)

```
TOPIC: "alerts"
READ_MODE: "new_only"
MAX_MESSAGES: 1000
TIMEOUT_SECONDS: 30
```

This will wait up to 30 seconds for new alert messages, stopping after 1000 messages.

### Get Last 500 Messages

```
TOPIC: "events"
READ_MODE: "last_n_messages"
MAX_MESSAGES: 500
TIMEOUT_SECONDS: 10
```

This will retrieve the 500 most recent events from the topic.

## Performance Considerations

- **Large Topics**: For topics with millions of messages, use `MAX_MESSAGES` to limit reads
- **Timeout Settings**: Increase `TIMEOUT_SECONDS` when reading large volumes of messages
- **Partitions**: The operator efficiently handles multi-partition topics in parallel
- **Memory**: Be cautious with `all_existing` mode on very large topics; consider pagination

## Dependencies

- `kafka-python` >= 2.0.0
- `pandas` >= 1.0.0
- Python 3.6+
