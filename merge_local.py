import os
os.environ["HF_HOME"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["WANDB_DIR"] = "/hpcwork/ba214121/.cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from trl import setup_chat_format
import torch
import shutil

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Set up chat format
print("Setting up chat format...")
model, tokenizer = setup_chat_format(base_model, tokenizer)

# Load and merge adapter
print("Loading and merging adapter...")
model = PeftModel.from_pretrained(
    model,
    "/hpcwork/ba214121/.cache/huggingface/checkpoints/llama3-8b-multi-turn-chat-sep",
    device_map="auto"
)
model = model.merge_and_unload()

# Save locally
print("Saving merged model locally...")
tmp_dir = "/hpcwork/ba214121/.cache/huggingface/merged_model"
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)
model.save_pretrained(tmp_dir)
tokenizer.save_pretrained(tmp_dir)

# Test the merged model
print("\nTesting merged model...")
messages = [{"role": "user", "content": "Define ethics in one sentence."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("\nPrompt:")
print(prompt)

gen_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

outputs = model.generate(
    **tokenizer(prompt, return_tensors="pt").to(model.device),
    generation_config=gen_config
)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\nResponse with special tokens visible:")
print(response)

# Cleanup
shutil.rmtree(tmp_dir) 