# LLM Finetuning Scripts

A collection of scripts I use for finetuning LLMs and running evaluations on RWTH's HPC cluster. Includes stuff for both regular LLM evaluation and Skywork preference models.

## What's in here?

### Main stuff
- Finetuning notebooks and scripts for custom LLMs
- Scripts to generate synthetic datasets using vLLM
- Evaluation scripts for comparing model outputs (WSDM, Skywork, DeepSeek)

### Finetuning Scripts & Notebooks
- `finetuning_llama.ipynb` - Main notebook for Llama finetuning on custom data
- `chat_finetuning.ipynb` - Finetunes models for chat/conversation
- `finetune_llama3_chat_sep.py` - Script version of chat finetuning
- `merge_peft.py` - Merges PEFT adapters with base models after training

### Dataset Processing
- `multi_turn_dataset_processing.ipynb` - Processes chat data into training format
- `dataset_processing.ipynb` - Preps Stanford Encyclopedia of Philosophy data for finetuning
- `test_generation_callback.py` - Callback utilities for data generation

### Evaluation Scripts
- `wsdm_baseline_eval.py` - Evaluates Llama models
  - A/B testing with optional CoT explanations
  - Temperature control for response diversity
- `wsdm_deepseek.py` - Same but for DeepSeek models
- `skywork_eval.py` - For Skywork preference models
- `qwq_baseline_eval.py` - Additional baseline evaluations
- `evaluate_phil.py` - Philosophy question testing
- `evaluate_checkpoints_with_lm_harness.py` - Checkpoint testing with LM harness

### Server Scripts
- `serve_llama3_vllm.sh` - vLLM OpenAI-compatible server for Llama 3 70B
- `serve_llama3_athene_vllm.sh` - Same but for Nexusflow/Athene-70B
- `serve_mistral_large_vllm.sh` - Serves Mistral-Large-Instruct-2407 model

### SLURM Job Scripts
- `run_wsdm_eval.sh` - Runs WSDM evals on cluster
- `run_wsdm_deepseek.sh` - Runs DeepSeek evals
- `run_skywork_eval.sh` - Runs Skywork evals

### Job Runners
- `submit_jobs.sh` - Launches multiple WSDM evals with different settings
- `submit_skywork_jobs.sh` - Launches Skywork eval jobs

### Templates & Prompts
- `multi_turn_gen_prompt.txt` - Template for generating philosophy tutoring conversations