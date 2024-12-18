#!/bin/bash
#SBATCH -N2
#SBATCH --job-name SVD
#SBATCH --nodes 1
#SBATCH --tasks-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node a100:2
#SBATCH --mem 64gb
#SBATCH --time 24:00:00
#SBATCH --gpus a100:2
#SBATCH --constraint gpu_a100_80gb
#SBATCH --partition mri2020

source /etc/profile.d/modules.sh
module add gcc/12.3.0
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate SVD

HOSTNAME=$(hostname)

accelerate launch --num_processes 2 --num_machines 1 train_svd.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --num_frames=26 \
    --num_workers=4 \
    --width=512 \
    --height=512 \
    --checkpointing_steps=10000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=1000 \
    --conditioning_dropout_prob=0.1 \
    --base_folder='/scratch/xi9/DATASET/DL3DV-960P-2K-Randominit' \
    --ref_folders='/scratch/xi9/Large-DATASET/DL3DV-10K/2K' \
    --output_dir=./model_outputs/DL3DV_fullres_camP-2K-revised-100k-test
    
