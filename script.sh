#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-23:50:20
#SBATCH --job-name=Qopt
#SBATCH --error=job.%J.err_node_48
#SBATCH --output=job.%J.out_node_48
#SBATCH --partition=highmemory

# Activate the virtual environment
source /home/apps/DL/DL-CondaPy3.7/etc/profile.d/conda.sh
conda activate qutip

# Run your Python script
python3 open_dicke_Liou_gen_eval.py