#!/bin/bash
#SBATCH -N2
#SBATCH --job-name SVD
#SBATCH --nodes 1
#SBATCH --tasks-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 64gb
#SBATCH --time 72:00:00
#SBATCH --gpus v100:1

source /etc/profile.d/modules.sh
module add gcc/12.3.0 
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate SVD

python eval.py --method $1