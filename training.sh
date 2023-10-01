#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --mem=8000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2

module use /opt/insy/modulefiles
#module load cuda/11.0
module load miniconda/3.7 

CUDA_LAUNCH_BLOCKING=1 python3 train.py \
     --use-cuda \
    --timestep 10 \
    --iters -1 \
    --epochs 100 \
    --lr-steps 51  \
    --dataset imagenetvid \
    --data-dir ../ILSVRC2015\
