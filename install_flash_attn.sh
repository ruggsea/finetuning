#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --output=logs/install_flash_attn_%j.out
#SBATCH --error=logs/install_flash_attn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=c23g

# Activate conda environment
source /home/ba214121/miniconda-h100/etc/profile.d/conda.sh
conda activate base

# Install Flash Attention
pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print('Flash Attention version:', flash_attn.__version__)" 