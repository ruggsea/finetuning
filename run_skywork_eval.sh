#!/bin/zsh
#SBATCH --job-name=skywork_eval
#SBATCH --output=logs/skywork_eval_%j.out
#SBATCH --error=logs/skywork_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=c23g

# Create logs directory
mkdir -p logs

# Activate conda environment
source /home/ba214121/miniconda-h100/etc/profile.d/conda.sh
conda activate base

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0

# Set up Hugging Face cache directories in HPCWORK
export TRANSFORMERS_CACHE="$HPCWORK/.cache/huggingface"
export HF_HOME="$HPCWORK/.huggingface"

# Create necessary cache directories
mkdir -p $HPCWORK/.cache/huggingface
mkdir -p $HPCWORK/.huggingface

# Check if model argument is provided
if [ -z "$MODEL" ]; then
    echo "Error: MODEL must be set"
    echo "Example usage: MODEL=Skywork/Skywork-Critic-Llama-3.1-70B sbatch run_skywork_eval.sh"
    exit 1
fi

# Run the evaluation script with provided model
python skywork_eval.py --model "$MODEL" 