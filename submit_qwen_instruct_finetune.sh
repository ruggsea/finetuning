#!/bin/bash

# Create timestamp for this job
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit job
sbatch \
    --output=logs/qwen_instruct_finetune_%j_${TIMESTAMP}.out \
    --error=logs/qwen_instruct_finetune_%j_${TIMESTAMP}.err \
    run_qwen_instruct_cot_wsdm_finetune.sh

echo "Submitted Qwen 72B instruct finetuning job" 