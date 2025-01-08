#!/bin/bash

# Create array of model configurations
MODELS=("meta-llama/Meta-Llama-3.1-70B-Instruct" "meta-llama/Meta-Llama-3.1-8B-Instruct")
TEMPS=(0.1 0.3 0.7 1.0)
METHODS=("simple" "rationale")

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
        GPU_COUNT=1
        TIME_LIMIT="08:00:00"
    fi
    
    for temp in "${TEMPS[@]}"; do
        for method in "${METHODS[@]}"; do
            # Submit job with specific model, temperature and method
            sbatch \
                --output=logs/wsdm_eval_${MODEL_NAME}_temp${temp}_${method}_%j_${TIMESTAMP}.out \
                --error=logs/wsdm_eval_${MODEL_NAME}_temp${temp}_${method}_%j_${TIMESTAMP}.err \
                --gres=gpu:${GPU_COUNT} \
                --time=${TIME_LIMIT} \
                --export=MODEL="$model",TEMPERATURE="$temp",METHOD="$method" \
                run_wsdm_eval.sh
        done
    done
done 