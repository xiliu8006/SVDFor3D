#!/bin/bash

source /etc/profile.d/modules.sh
module add gcc/12.3.0 
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate SVD
python --version
gcc --version
cd /scratch/xi9/code/SVDFor3DGS

HOSTNAME = $(hostname)


accelerate launch --num_processes 2 --num_machines 1 train_svd.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --num_frames=25 \
    --num_workers=4 \
    --width=512 \
    --height=512 \
    --checkpointing_steps=10000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=500 \
    --conditioning_dropout_prob=0.1 \
    --output_dir=./outputs_diff_ratio_l1_loss_single_machine
