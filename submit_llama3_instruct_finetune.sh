#!/bin/bash

# Create timestamp for this job
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit job
sbatch \
    --output=logs/llama3_instruct_finetune_%j_${TIMESTAMP}.out \
    --error=logs/llama3_instruct_finetune_%j_${TIMESTAMP}.err \
    run_llama3_instruct_finetune.sh

echo "Submitted Llama 3 instruct finetuning job" 