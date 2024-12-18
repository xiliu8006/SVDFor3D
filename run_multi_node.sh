#!/bin/bash
#SBATCH --job-name SVD
#SBATCH --nodes 8
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node a100:2
#SBATCH --mem 64gb
#SBATCH --time 72:00:00
#SBATCH --gpus a100:16
#SBATCH --constraint gpu_a100_80gb

source /etc/profile.d/modules.sh
module add gcc/12.3.0 
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate SVD
python --version
gcc --version
cd /scratch/xi9/code/SVDFor3DGS

export NCCL_DEBUG=INFO

export GPUS_PER_NODE=2

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export LAUNCHER=" \
    torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$UID \
    "
export SCRIPT="train_svd.py"

export SCRIPT_ARGS=" \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --num_frames=20 \
    --num_workers=4 \
    --width=512 \
    --height=512 \
    --checkpointing_steps=10000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=500 \
    --conditioning_dropout_prob=0.1 \
    --output_dir=./outputs_diff_ratio_l1_loss \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 

