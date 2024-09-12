#!/bin/sh
#SBATCH --job-name=PHYSIBOSSCOVID # Job name
#SBATCH --nodes=6
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH -vvv
#SBATCH --mem=50gb                  # Total memory limit
#SBATCH --time=1:00:00              # Time limit hrs:min:sec
#SBATCH --output=log_%j.log     # Standard output and error log

export PATH="/mnt/beegfs/common/apps/gcc/gcc-11.1/bin":$PATH
export PATH="/mnt/beegfs/common/apps/python/python-3.9.5/bin/":$PATH
export LD_LIBRARY_PATH="/mnt/beegfs/common/apps/gcc/gcc-11.1/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/mnt/beegfs/common/apps/gcc/gcc-11.1/lib64":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/mnt/beegfs/common/apps/python/python-3.9.5/lib64/":$LD_LIBRARY_PATH
export OMP_NUM_THREADS=20

make clean; make -j

srun -n 12 ./run_replicate.sh __SETTINGS__

cd scripts/Replicates
python3 ParseReplicates.py --folder __SETTINGS__
python3 ParseReplicatesDat.py --folder __SETTINGS__
python3 ParseReplicatesStates.py --folder __SETTINGS__
python3 plotReplicates.py --folder __SETTINGS__

