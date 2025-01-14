#!/bin/zsh
#SBATCH --job-name=llama3_instruct_finetune
#SBATCH --output=logs/llama3_instruct_finetune_%j.out
#SBATCH --error=logs/llama3_instruct_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=c23g

# Create logs directory
mkdir -p logs

# Activate conda environment
source /home/ba214121/miniconda-h100/etc/profile.d/conda.sh
conda activate base

# Source the rc file based on the shell
if [ -n "$ZSH_VERSION" ]; then
    source ~/.zshrc
elif [ -n "$BASH_VERSION" ]; then
    source ~/.bashrc
fi

# Verify HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    exit 1
fi

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0

# Set proper OpenMP threads (as integer)
export OMP_NUM_THREADS=8

# Set up Hugging Face cache directories in HPCWORK
export TRANSFORMERS_CACHE="$HPCWORK/.cache/huggingface"
export HF_HOME="$HPCWORK/.huggingface"

# Create necessary cache directories
mkdir -p $HPCWORK/.cache/huggingface
mkdir -p $HPCWORK/.huggingface

# Run the finetuning script
python finetune_llama3_instruct.py 