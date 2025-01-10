#!/bin/bash

# Create array of Qwen models
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)

# Create array of methods
METHODS=("simple" "rationale")

# Create timestamp for this batch of jobs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Submit jobs
for model in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $model | cut -d'/' -f2)
    MODEL_SIZE=$(echo $MODEL_NAME | cut -d'-' -f2)
    
    for method in "${METHODS[@]}"; do
        # Set GPU count and time limit based on model size and method
        if [[ $MODEL_SIZE == "14B" ]]; then
            GPU_COUNT=2
            TIME_LIMIT="08:00:00"
        elif [[ $MODEL_SIZE == "7B" ]]; then
            GPU_COUNT=2
            TIME_LIMIT="06:00:00"
        else
            GPU_COUNT=1
            TIME_LIMIT="04:00:00"
        fi
        
        # Increase time limit for rationale method
        if [[ $method == "rationale" ]]; then
            TIME_LIMIT="12:00:00"
        fi
        
        # Submit job with specific model and method
        sbatch \
            --output=logs/qwen_eval_${MODEL_NAME}_${method}_%j_${TIMESTAMP}.out \
            --error=logs/qwen_eval_${MODEL_NAME}_${method}_%j_${TIMESTAMP}.err \
            --gres=gpu:${GPU_COUNT} \
            --time=${TIME_LIMIT} \
            --export=MODEL="$model",METHOD="$method" \
            run_qwen_eval.sh
    done
done 