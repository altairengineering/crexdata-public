import json
import os


script_directory = os.path.dirname(os.path.abspath(__file__))
# Relative path to the tmp directory
tmp_dir = '../../metrics/tmp'
comb_dir = os.path.normpath(os.path.join(script_directory, f'{tmp_dir}/combinations'))


def get_test_hyper_parameters(comb_id, proc_id):
    with open(f'{comb_dir}/{comb_id}.json', 'r') as f:
        all_combinations = json.load(f)
    return all_combinations[proc_id]


def kafka_get_test_hyper_parameters(topic='FedL', bootstrap_servers='localhost:9092', group_id='fda'):
    from confluent_kafka import Consumer, KafkaError

    # Consumer example
    c = Consumer({
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'latest'
    })

    c.subscribe([topic])

    while True:
        msg = c.poll(1.0)
        if not msg:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(msg.error())
                break
        val = msg.value().decode("utf-8")

        # Parse the JSON string back into a Python data structure
        combinations = json.loads(val)

        c.close()

        # Signal the parent process before returning
        fifo_path = os.environ['FIFO_PATH']  # Get the FIFO path from the environment variable
        with open(fifo_path, 'w') as fifo:
            fifo.write('Received Kafka message')  # Send the signal to the parent

        return combinations[0]

