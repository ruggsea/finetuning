#!/bin/bash

# Create array of Skywork models
MODELS=("Skywork/Skywork-Critic-Llama-3.1-70B" "Skywork/Skywork-Critic-Llama-3.1-8B")

# Create timestamp for this batch of jobs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Submit jobs
for model in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $model | cut -d'/' -f2)
    
    # Set GPU count and time limit based on model size
    if [[ $MODEL_NAME == *"70B"* ]]; then
        GPU_COUNT=4
        TIME_LIMIT="16:00:00"  # Double time for 70B models
    else
        GPU_COUNT=2
        TIME_LIMIT="04:00:00"
    fi
    
    # Submit job with specific model
    sbatch \
        --output=logs/skywork_eval_${MODEL_NAME}_%j_${TIMESTAMP}.out \
        --error=logs/skywork_eval_${MODEL_NAME}_%j_${TIMESTAMP}.err \
        --gres=gpu:${GPU_COUNT} \
        --time=${TIME_LIMIT} \
        --export=MODEL="$model" \
        run_skywork_eval.sh
done 