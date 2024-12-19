#!/bin/bash
#SBATCH --job-name SVD_inference
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 64gb
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition mri2020

source /etc/profile.d/modules.sh
module add gcc/12.3.0 
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate SVD

list=("3" "6" "9")
for element in "${list[@]}"; do
    python inference_svd.py --scene $1 --model_path $2 --output_path $3 --num_views $element --base_folders '/home/xi9/code/DATASET/DL3DV-960P-2K-Randominit' --ref_folders '/scratch/xi9/Large-DATASET/DL3DV-10K/2K'
done