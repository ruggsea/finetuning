#!/bin/bash

# Activate the vllm Conda environment
source ~/.zshrc
conda activate vllm

HF_HOME=/hpcwork/ba214121/.cache/huggingface
TRANFORMERS_CACHE=/hpcwork/ba214121/.cache/huggingface
HF_DATASETS_CACHE=/hpcwork/ba214121/.cache/huggingface

echo $HF_DATASETS_CACHE
# Serve the model using vLLM's command-line interface
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-Large-Instruct-2407 \
    --tensor-parallel-size 4 \
    --dtype half \
    --max-num-batched-tokens 4096 \
    --host 0.0.0.0 \
    --port 8000
