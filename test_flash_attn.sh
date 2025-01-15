#!/bin/bash
#SBATCH --job-name=test_flash_attn
#SBATCH --output=logs/test_flash_attn_%j.out
#SBATCH --error=logs/test_flash_attn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --partition=c23g

# Activate conda environment
source /home/ba214121/miniconda-h100/etc/profile.d/conda.sh
conda activate base

# Run test
python test_flash_attn.py 