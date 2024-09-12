import argparse
import subprocess
import signal

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


def signal_handler(signum, frame):
    """
    Signal handler to kill all child processes.
    """
    global processes  # Explicitly state that we're using the global variable

    for proc in processes:
        if not proc.poll():  # Check if the process is still running
            proc.kill()
    print("All processes terminated. Exiting.")
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int,
                        help="The number of available GPUs in this machine. (We will only use one per simulation)")
    parser.add_argument('--n_sims', type=int, help="The number of simulations to run in parallel. One in each GPU.")
    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json.")
    # Add the gpu_mem argument
    parser.add_argument('--gpu_mem', type=int, default=-1,
                        help="The GPU memory to be used. If not provided we let TensorFlow dynamically allocate.")
    parser.add_argument('--use_all_gpus', action='store_true',
                        help="Use all available GPUs. Defaults to False if not given.")
    args = parser.parse_args()

    gpu_id_generator = gpu_id_gen(args.n_gpus, args.use_all_gpus)

    for sim_id in range(args.n_sims):
        # Ask the user whether to start a new process or quit
        user_response = input(
            f"Press 'Enter' to start process {sim_id + 1}/{args.n_sims} or 'q' to terminate all and quit: "
        ).strip().lower()

        # If the user enters 'q', terminate all running processes and exit
        if user_response == "q":
            print("Terminating all processes based on user input.")
            for p in processes:
                if not p.poll():  # Check if the process is still running
                    p.kill()
            print("All processes terminated. Exiting.")
            exit()

        # Construct the command
        cmd = [
            'python', '-u', '-m', 'fdavg.main', f'--comb_file_id={args.comb_file_id}',
            f'--sim_id={sim_id}', f'--gpu_id={next(gpu_id_generator)}',
            f'--gpu_mem={args.gpu_mem}'
        ]

        with open(f'metrics/tmp/local_out/c{args.comb_file_id}_sim{sim_id}.out', 'w') as stdout_file:
            with open(f'metrics/tmp/local_out/c{args.comb_file_id}_sim{sim_id}.err', 'w') as stderr_file:
                print(f"Running: {' '.join(cmd)}")
                process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
                processes.append(process)

        print()

    print(f"The specified {args.n_sims} processes have all been started.")
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for all child processes to complete
    for p in processes:
        p.wait()

    print("All processes terminated. Exiting.")

