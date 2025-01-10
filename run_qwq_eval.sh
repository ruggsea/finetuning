#!/bin/zsh
#SBATCH --job-name=qwq_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=c23g

# Create logs directory
mkdir -p logs

# Set job name with method for better identification
if [ ! -z "$METHOD" ]; then
    #SBATCH --job-name=qwq_eval_${METHOD}
    #SBATCH --output=logs/qwq_eval_%j_${METHOD}.out
    #SBATCH --error=logs/qwq_eval_%j_${METHOD}.err
else
    #SBATCH --output=logs/qwq_eval_%j.out
    #SBATCH --error=logs/qwq_eval_%j.err
fi

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
export CUDA_VISIBLE_DEVICES=0,1

# Set up Hugging Face cache directories in HPCWORK
export TRANSFORMERS_CACHE="$HPCWORK/.cache/huggingface"
export HF_HOME="$HPCWORK/.huggingface"

# Create necessary cache directories
mkdir -p $HPCWORK/.cache/huggingface
mkdir -p $HPCWORK/.huggingface

# Check if required arguments are provided
if [ -z "$MODEL" ] || [ -z "$METHOD" ]; then
    echo "Error: MODEL and METHOD must be set"
    echo "Example usage: MODEL=Qwen/QwQ-32B-Preview METHOD=simple sbatch run_qwq_eval.sh"
    exit 1
fi

# Validate METHOD argument
if [[ ! "$METHOD" =~ ^(simple|rationale|both)$ ]]; then
    echo "Error: METHOD must be one of: simple, rationale, both"
    exit 1
fi

# Run the evaluation script with provided arguments
python qwq_baseline_eval.py --model "$MODEL" --method "$METHOD" 