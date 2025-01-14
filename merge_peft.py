import os
os.environ["HF_HOME"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["WANDB_DIR"] = "/hpcwork/ba214121/.cache"

# Example usage:
# python merge_peft.py --base_model=meta-llama/Llama-2-7b-hf --peft_model=./qlora-out --hub_id=alpaca-qlora

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import torch
import shutil
from huggingface_hub import HfApi
from trl import setup_chat_format

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--hub_id", type=str)

    return parser.parse_args()

def main():
    args = get_args()

    print(f"[1/6] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # setup chat format
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)


    print(f"[2/6] Loading adapter: {args.peft_model}")
    model = PeftModel.from_pretrained(base_model, args.peft_model, device_map="auto")
    
    print("[3/6] Merge base model and adapter")
    model = model.merge_and_unload()
    
    # Add generation config
    print("[4/6] Setting up generation config")
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end_of_text_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    
    # Update both model config and generation config
    model.config.eos_token_id = [im_end_token_id, end_of_text_token_id]
    model.config.forced_eos_token_id = end_of_text_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    generation_config = GenerationConfig(
        eos_token_id=[im_end_token_id, end_of_text_token_id],
        forced_eos_token_id=end_of_text_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    model.generation_config = generation_config
    
    # Create a temporary directory for saving
    tmp_dir = "/hpcwork/ba214121/.cache/huggingface/merged_model"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    print(f"[4/6] Saving model and tokenizer locally in {tmp_dir}")
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    
    print(f"[5/6] Creating repository: {args.hub_id}")
    api = HfApi()
    api.create_repo(repo_id=args.hub_id, exist_ok=True, private=True)
    
    print(f"[6/6] Uploading to Hugging Face Hub: {args.hub_id}")
    model.push_to_hub(args.hub_id, use_auth_token=True)
    tokenizer.push_to_hub(args.hub_id, use_auth_token=True)
    
    print("Merged model uploaded to Hugging Face Hub!")
    
    # Cleanup
    shutil.rmtree(tmp_dir)

if __name__ == "__main__" :
    main()