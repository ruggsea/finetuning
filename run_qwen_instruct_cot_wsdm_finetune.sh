#!/bin/zsh
#SBATCH --job-name=qwen_instruct_finetune
#SBATCH --output=logs/qwen_instruct_finetune_%j.out
#SBATCH --error=logs/qwen_instruct_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --partition=c23g
#SBATCH --mem=256G

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

# Set proper OpenMP threads
export OMP_NUM_THREADS=32

# Set up Hugging Face cache directories in HPCWORK
export TRANSFORMERS_CACHE="$HPCWORK/.cache/huggingface"
export HF_HOME="$HPCWORK/.huggingface"

# Create necessary cache directories
mkdir -p $HPCWORK/.cache/huggingface
mkdir -p $HPCWORK/.huggingface

# Run the finetuning script
python finetune_qwen_instruct_cot_wsdm.py 