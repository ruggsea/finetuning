#!/bin/bash

# Create array of QwQ models
MODELS=("Qwen/QwQ-32B-Preview")

# Create timestamp for this batch of jobs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create array of methods
METHODS=("simple" "rationale")

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit jobs
for model in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $model | cut -d'/' -f2)
    
    for method in "${METHODS[@]}"; do
        # Set GPU count and time limit based on method
        if [ "$method" == "simple" ]; then
            TIME_LIMIT="08:00:00"
        else
            TIME_LIMIT="12:00:00"
        fi
        
        # Submit job with specific model and method
        sbatch \
            --job-name=qwq_eval_${MODEL_NAME}_${method} \
            --output=logs/qwq_eval_${MODEL_NAME}_${method}_%j_${TIMESTAMP}.out \
            --error=logs/qwq_eval_${MODEL_NAME}_${method}_%j_${TIMESTAMP}.err \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=8 \
            --gres=gpu:2 \
            --time=${TIME_LIMIT} \
            --partition=c23g \
            --export=ALL,MODEL="$model",METHOD="$method" \
            run_qwq_eval.sh
            
        echo "Submitted ${method} evaluation job for ${MODEL_NAME}"
    done
done 
