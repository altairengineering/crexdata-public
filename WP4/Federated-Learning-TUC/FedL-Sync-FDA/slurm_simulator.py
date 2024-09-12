import argparse
import subprocess
import os

processes = []


def gpu_id_gen(n_gpus):
    """
    A generator that yields the GPU ID to use.
    """
    i = 0
    while True:
        yield i % n_gpus
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int,
                        help="The number of available GPUs in this machine. (We will only use one per simulation)",
                        required=True)
    parser.add_argument('--n_sims', type=int, help="The number of simulations to run in parallel. One in each GPU.",
                        required=True)
    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json.",
                        required=True)
    # Add the gpu_mem argument
    parser.add_argument('--gpu_mem', type=int, default=-1,
                        help="The GPU memory to be used. If not provided we let TensorFlow dynamically allocate.")
    parser.add_argument('--starting_sim_id_in_submit', type=int,
                        help="The starting simulation ID (index in the combinations file). "
                             "The first simulation will have this ID.",
                        required=True)
    args = parser.parse_args()

    slurm_procid = int(os.environ.get('SLURM_PROCID'))

    starting_sim_id = args.starting_sim_id_in_submit + slurm_procid * args.n_sims

    gpu_id_generator = gpu_id_gen(args.n_gpus)

    for sim_id in range(starting_sim_id, starting_sim_id + args.n_sims):
        # Construct the command
        cmd = [
            'python', '-u', '-m', 'fdavg.main', f'--comb_file_id={args.comb_file_id}',
            f'--sim_id={sim_id}', f'--gpu_id={next(gpu_id_generator)}',
            f'--gpu_mem={args.gpu_mem}', '--slurm'
        ]
        
        process = subprocess.Popen(cmd)
        processes.append(process)

        print()

    print(f"The specified {args.n_sims} processes have all been started.")

    # Wait for all child processes to complete
    for p in processes:
        p.wait()

    print("All processes terminated. Exiting.")

