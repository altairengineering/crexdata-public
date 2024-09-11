import subprocess
import argparse
import os


slurm_template = """#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name={job_name}
#SBATCH --output=metrics/tmp/slurm_out/{job_name}.out   # Stdout %j expands to jobId, %a is array task index
#SBATCH --error=metrics/tmp/slurm_out/{job_name}.err   # Stderr %j expands to jobId, %a is array task index
#SBATCH --ntasks={n_tasks}   # Number of tasks requested
#SBATCH --nodes={n_nodes}   # Number of nodes requested
#SBATCH --ntasks-per-node=1   # Tasks per node
#SBATCH --cpus-per-task={cpus_per_task}    # Threads per task
#SBATCH --time={walltime}   # walltime
#SBATCH --mem={mem}    # memory per NODE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --account=pa240202

## LOAD MODULES ##
module purge            # clean up loaded modules
module load gnu/8
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load python/3.8.13
module load tftorch/270-191

export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
srun python -m slurm_simulator --n_gpus={n_gpus} --comb_file_id={comb_file_id} --gpu_mem={gpu_mem} --starting_sim_id_in_submit={starting_sim_id_in_submit} --n_sims={n_sims}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sims', type=int, help="Total number of simulations.",
                        required=True)
    parser.add_argument('--comb_file_id', type=int, help="The combinations prefix, i.e., <PREFIX>.json.",
                        required=True)
    # Add the gpu_mem argument
    parser.add_argument('--gpu_mem', type=int, default=-1,
                        help="The GPU memory to be used. If not provided we let TensorFlow dynamically allocate.")
    parser.add_argument('--gpus_per_node', type=int, default=2, help="Number of GPUs per Node.")
    parser.add_argument('--sims_per_gpu', type=int, default=4, help="Number of simulations per GPU.")
    parser.add_argument('--mem_per_sim', type=int, default=1024*10, help="Memory (MB) per simulation.")
    parser.add_argument('--cpus_per_sim', type=int, default=2, help="CPUs per simulation.")
    parser.add_argument('--nodes_per_submit', type=int, default=2, help="Nodes per job.")
    parser.add_argument('--walltime', type=str, default="24:00:00", help="Walltime.")
    args = parser.parse_args()

    if args.n_sims % (args.nodes_per_submit * args.sims_per_gpu * args.gpus_per_node != 0):
        raise ValueError("The number of simulations must be a multiple of "
                         "nodes_per_submit * sims_per_gpu * gpus_per_node.")

    sims_per_submit = args.nodes_per_submit * args.sims_per_gpu * args.gpus_per_node
    num_of_submits = args.n_sims // sims_per_submit

    for i in range(num_of_submits):
        slurm_script = slurm_template.format(
            job_name=f"c{args.comb_file_id}_s{i}",
            n_tasks=args.nodes_per_submit,
            n_nodes=args.nodes_per_submit,
            cpus_per_task=args.cpus_per_sim * args.sims_per_gpu * args.gpus_per_node,
            walltime=args.walltime,
            mem=args.mem_per_sim * args.sims_per_gpu * args.gpus_per_node,
            gpus_per_node=args.gpus_per_node,
            # arguments of the slurm_simulator
            n_gpus=args.gpus_per_node,
            comb_file_id=args.comb_file_id,
            gpu_mem=args.gpu_mem,
            starting_sim_id_in_submit=i*sims_per_submit,
            n_sims=args.sims_per_gpu * args.gpus_per_node
        )

        # Save the Slurm script content to a temporary file
        with open("tmp_slurm_script.slurm", "w") as f:
            f.write(slurm_script)

        # Submit the Slurm script using sbatch
        result = subprocess.run(["sbatch", "tmp_slurm_script.slurm"], capture_output=True, text=True)
        print(result)
        print(f"Submitted slurm job {i + 1}/{num_of_submits} for combination file {args.comb_file_id}.json.")
        print()

    # Remove the temporary file
    os.remove("tmp_slurm_script.slurm")



