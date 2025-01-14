#!/usr/bin/env python3

# Set environment variables first
import os
os.environ["HF_HOME"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["WANDB_DIR"] = "/hpcwork/ba214121/.cache"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    GenerationConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb
from datetime import datetime
from trl.models.utils import unwrap_model_for_generation
import pandas as pd
import shutil
from peft import PeftModel
from huggingface_hub import HfApi

def _generate_completion(model, tokenizer, prompt, generation_config, accelerator=None):
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move to device and ensure correct dtype
    inputs = {k: v.to(dtype=torch.long if k == 'input_ids' else torch.bfloat16, device=model.device) for k, v in inputs.items()}

    # Generate with unwrapped model
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        unwrapped_model.eval()
        unwrapped_model = unwrapped_model.to(dtype=torch.bfloat16)
        
        with torch.no_grad():
            outputs = unwrapped_model.generate(
                **inputs,
                generation_config=generation_config,
            )
    
    # Decode and return only the generated part
    completion = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return completion

class GenerationCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, steps=100):
        print("\nInitializing GenerationCallback...")
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.steps = steps
        self.test_prompts = [
            "Who are you?",
            "What can you help me with?",
            "Are you an expert in Philosophy?",
        ]
        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
        )
        self._last_logged_step = -1
        print("GenerationCallback initialized successfully")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self._last_logged_step:
            return

        if state.global_step % self.steps == 0:
            print(f"\nStep {state.global_step}: Running generation tests...")
            
            try:
                records = []
                
                for test_prompt in self.test_prompts:
                    # Format test message directly with chat template
                    test_messages = [{"role": "user", "content": test_prompt}]
                    prompt = self.tokenizer.apply_chat_template(
                        test_messages,
                        tokenize=False
                    )

                    response = _generate_completion(
                        model=self.trainer.model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        generation_config=self.generation_config,
                        accelerator=self.trainer.accelerator
                    )
                    
                    records.append({
                        "prompt": test_prompt,
                        "response": response,
                        "step": state.global_step,
                        "epoch": state.epoch
                    })
                    
                    print(f"\nPrompt: {test_prompt}")
                    print(f"Response: {response}")
                
                predictions_df = pd.DataFrame.from_records(records)
                generation_table = wandb.Table(dataframe=predictions_df)
                
                wandb.log({
                    "generation_tests": generation_table,
                    "step": state.global_step
                })
                
                self._last_logged_step = state.global_step
                
            except Exception as e:
                print(f"Generation failed with error: {str(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                print("Stopping training due to generation failure")
                control.should_training_stop = True
                raise e
                
            finally:
                self.trainer.model.train()

def main():
    # Model and training parameters
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-instruct"  # Changed to instruct model
    OUTPUT_DIR = "llama3-8b-instruct-SEP-chat"
    HUB_MODEL_ID = "ruggsea/Llama3.1-Instruct-SEP-Chat"

    # Setup environment variables for caching
    cache_dir = os.path.expanduser("/hpcwork/ba214121/.cache/huggingface")
    checkpoint_dir = os.path.join(cache_dir, "checkpoints", OUTPUT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["WANDB_DIR"] = os.path.dirname(cache_dir)

    # Load dataset
    dataset = load_dataset("json", data_files="multi_turn_gen_conversations.jsonl", split="train")

    # Convert dataset to text format using chat template
    def format_chat(example):
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in example["conversation"]]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        }

    # Initialize wandb
    wandb.init(project="llama3-instruct-chat")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        # Load tokenizer first since we need it for dataset processing
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        # Process dataset
        processed_dataset = dataset.map(
            format_chat,
            remove_columns=['prompt', 'conversation'],
            batched=False
        )
        
        # Split into train/eval
        split_dataset = processed_dataset.train_test_split(test_size=0.05)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of validation examples: {len(eval_dataset)}")

        # Load base model
        print(f"Loading base model: {BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Prepare model for training
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        peft_config = LoraConfig(
                lora_alpha=128,
                lora_dropout=0.05,
                r=256,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM", 
        )

        # Training arguments
        training_args = SFTConfig(
            output_dir=checkpoint_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            learning_rate=2e-5,
            bf16=True,
            logging_steps=10,
            logging_dir=os.path.join(cache_dir, "logs"),
            save_strategy="epoch",
            eval_strategy="steps",
            eval_steps=500,
            do_eval=True,
            report_to="wandb",
            run_name=f"llama3-instruct-chat-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
            warmup_ratio=0.03,
            group_by_length=True,
            push_to_hub=False,  # We'll handle this manually after merging
            hub_model_id=HUB_MODEL_ID,
            packing=True,
        )

        # Initialize trainer with generation callback
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=lambda x: x["text"],
            peft_config=peft_config,
        )
        
        generation_callback = GenerationCallback(trainer, tokenizer, steps=300)
        trainer.add_callback(generation_callback)

        # Train
        print("Starting training...")
        model.config.use_cache = False
        trainer.train()
        
        # Save the adapter weights
        print("Saving adapter weights...")
        trainer.save_model()
        
        print("Training completed. Starting model merging process...")
        
        # Load the base model again for merging (in fp16 this time)
        print("Loading base model for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load the trained adapter
        print("Loading and merging adapter...")
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        
        # Merge and unload
        print("Merging weights...")
        model = model.merge_and_unload()
        

        # Create a temporary directory for saving
        print("Saving merged model...")
        tmp_dir = os.path.join(cache_dir, "merged_model")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        
        # Save locally first
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        
        # Upload to Hub
        print(f"Uploading merged model to Hub: {HUB_MODEL_ID}")
        api = HfApi()
        api.create_repo(repo_id=HUB_MODEL_ID, exist_ok=True, private=True)
        
        model.push_to_hub(HUB_MODEL_ID, use_auth_token=True)
        tokenizer.push_to_hub(HUB_MODEL_ID, use_auth_token=True)
        
        print("Model successfully merged and uploaded!")
        
        # Cleanup
        shutil.rmtree(tmp_dir)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 