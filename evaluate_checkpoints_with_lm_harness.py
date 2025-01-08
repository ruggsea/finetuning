import os
import subprocess
import json
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the path to your LoRA checkpoints
lora_checkpoint_dir = "./llama3.1-8B-Instruct-stanford-philosophy-chat_finetune"

# Define the base model path or name
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with your actual base model

# Get all LoRA checkpoint directories
lora_checkpoint_dirs = [os.path.join(lora_checkpoint_dir, d) for d in os.listdir(lora_checkpoint_dir) if d.startswith("checkpoint-")]
lora_checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
print("LoRA checkpoint directories found:", lora_checkpoint_dirs)

# Function to run lm-eval command
def run_lm_eval(checkpoint):
    checkpoint_name=checkpoint.split("-")[-1]
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={base_model_name},peft={checkpoint}",
        "--tasks", "mmlu_pro_philosophy",
        "--num_fewshot", "5",
        "--device", "cuda:0",
        "--wandb_args", f"project=lm-eval-harness-integration,name=checkpoint_{checkpoint_name}_{current_time}",
    ]
    
    print(f"Running evaluation for checkpoint: {checkpoint}")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Evaluation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during evaluation: {e}")
        print("Standard output:", e.stdout)
        print("Standard error:", e.stderr)
        return False

# Evaluate each LoRA checkpoint
for checkpoint in lora_checkpoint_dirs:
    success = run_lm_eval(checkpoint)
    checkpoint_name=checkpoint.split("-")[-1]
    # if success:
    #     # Read and print results
    #     with open(f"{checkpoint_name}_results.json", 'r') as f:
    #         results = json.load(f)
    #     print(f"Results for {checkpoint}:")
    #     print(json.dumps(results, indent=2))
    
    print("-" * 50)

print("All evaluations complete!")
