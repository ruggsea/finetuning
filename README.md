# LLM Finetuning Scripts

A collection of scripts I use for finetuning LLMs and running evaluations on RWTH's HPC cluster. Includes stuff for both regular LLM evaluation and Skywork preference models.

## What's in here?

### Main stuff
- Finetuning notebooks for training custom LLMs
- Scripts to generate synthetic datasets using vLLM (way faster than regular inference)
- Evaluation scripts for comparing model outputs and running preference modeling

### Finetuning Notebooks
- `finetuning_llama.ipynb` - Main notebook for Llama finetuning on custom data
- `chat_finetuning.ipynb` - Finetunes models for chat/conversation
- `multi_turn_dataset_processing.ipynb` - Processes chat data into training format

### Dataset & Evaluation Notebooks
- `dataset_processing.ipynb` - Preps data for finetuning, cleans things up, was used for the SEP semisynthetic dataset
- `testing_vllm_inference.ipynb` - Playing around with vLLM inference
- `evaluate_phil_test.ipynb` - Testing models on philosophy questions
- `wsdm.ipynb` - Analyzes evaluation results, makes some nice plots

### Python Scripts
- `wsdm_baseline_eval.py` - Runs evaluations using regular LLMs (Llama 3.1)
  - Can do quick A/B evaluation or generate explanations (CoT style) for two answers to a prompt
  - Lets you play with different temperatures

- `skywork_eval.py` - Special script for Skywork preference models
  - Zero-shot preference stuff
  - Uses their special template
  - Outputs clean A/B decisions

- `evaluate_checkpoints_with_lm_harness.py` - Quick way to test model checkpoints
- `evaluate_phil.py` - Tests models on philosophy questions
- `merge_peft.py` - Merges PEFT adapters with base models after training

### Server Scripts
- `serve_llama3_vllm.sh` - Spins up a vLLM OpenAI-compatible server for Llama 3 70B
- `serve_llama3_athene_vllm.sh` - Same but for Nexusflow/Athene-70B
- `serve_mistral_large_vllm.sh` - Serves Mistral-Large-Instruct-2407 model

### Evaluation Scripts
- `run_wsdm_eval.sh` - Runs the WSDM eval script on the cluster
- `run_skywork_eval.sh` - Same but for Skywork models

### Job Runners
- `submit_jobs.sh` - Launches a bunch of WSDM evals with different settings
- `submit_skywork_jobs.sh` - Launches Skywork eval jobs

### Templates & Prompts
- `multi_turn_gen_prompt.txt` - Template for generating philosophy tutoring conversations between a professor and a student

