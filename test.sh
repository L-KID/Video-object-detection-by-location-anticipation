#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --mem=16000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

module use /opt/insy/modulefiles
#module load cuda/11.0
module load miniconda/3.7

python3 train.py \
    --use-cuda \
    --resume \
    --timestep 10 \
    --iters -1 \
    --epochs 20 \
    --dataset imagenetvid \
    --data-dir ../ILSVRC2015 \
