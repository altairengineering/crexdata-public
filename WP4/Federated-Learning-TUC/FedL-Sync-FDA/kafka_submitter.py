import argparse
import subprocess
from confluent_kafka import Consumer, KafkaError
import os

processes = []


def gpu_id_gen(n_gpus, use_all_gpus):
    """
    A generator that yields the GPU ID to use.
    """
    i = 0
    while True:
        if use_all_gpus:
            yield ','.join(str(i) for i in range(n_gpus))
        else:
            yield i % n_gpus
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int,
                        help="The number of available GPUs in this machine. (We will only use one per simulation)")
    parser.add_argument('--kafka_topic', type=str, help="The Kafka topic to read hyper-parameters from.")
    parser.add_argument('--kafka_server', type=str, help="The bootstrap server <IP>:<PORT>.")
    # Add the gpu_mem argument
    parser.add_argument('--gpu_mem', type=int, default=-1,
                        help="The GPU memory to be used. If not provided we let TensorFlow dynamically allocate.")
    parser.add_argument('--use_all_gpus', action='store_true',
                        help="Use all available GPUs. Defaults to False if not given.")
    args = parser.parse_args()

    gpu_id_generator = gpu_id_gen(args.n_gpus, args.use_all_gpus)

    sim_id = 0

    while True:
        sim_id += 1

        fifo_path = f"/tmp/my_fifo_{sim_id}"

        # Create a FIFO (named pipe)
        try:
            os.mkfifo(fifo_path)
        except FileExistsError:
            pass  # The FIFO already exists

        # Construct the command
        cmd = [
            'python', '-u', '-m', 'fdavg.main', f'--gpu_id={next(gpu_id_generator)}', f'--sim_id={sim_id}',
            f'--gpu_mem={args.gpu_mem}', '--kafka'
        ]

        with open(f'metrics/tmp/local_out/server_sim{sim_id}.out', 'w') as stdout_file:
            with open(f'metrics/tmp/local_out/server_sim{sim_id}.err', 'w') as stderr_file:
                process = subprocess.Popen(
                    cmd, stdout=stdout_file, stderr=stderr_file, env=dict(os.environ, FIFO_PATH=fifo_path)
                )
                processes.append(process)

        # Open the FIFO for reading and block until the child process writes to it
        with open(fifo_path, 'r') as fifo:
            message = fifo.read()
            print(f'Received hyper-parameters from Kafka! FedL workflow has started with ID: {sim_id}')

        # Cleanup
        os.remove(fifo_path)

