#!/bin/bash

# Create timestamp for this job
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit job
sbatch \
    --output=logs/llama70b_instruct_finetune_%j_${TIMESTAMP}.out \
    --error=logs/llama70b_instruct_finetune_%j_${TIMESTAMP}.err \
    run_llama70b_instruct_cot_wsdm_finetune.sh

echo "Submitted Llama 70B instruct finetuning job" 