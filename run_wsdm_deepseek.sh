#!/bin/bash
#SBATCH --job-name=wsdm_deepseek
#SBATCH --output=logs/wsdm_deepseek_%j.out
#SBATCH --error=logs/wsdm_deepseek_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=c23g

# source the rc file based on the shell
if [ -n "$ZSH_VERSION" ]; then
    source ~/.zshrc
elif [ -n "$BASH_VERSION" ]; then
    source ~/.bashrc
fi

# make sure the OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

# Activate your conda environment if needed
# conda activate your_env_name

# Run the script
conda activate base
python wsdm_deepseek.py 