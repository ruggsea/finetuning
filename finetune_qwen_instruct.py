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
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from trl.models.utils import unwrap_model_for_generation
import wandb
from datetime import datetime
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
            "Prompt: What is machine learning?\n\nResponse A: ```Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed.```\n\nResponse B: ```Machine learning uses statistical techniques to allow computers to find patterns in data.```\n\n",
            "Prompt: How does photosynthesis work?\n\nResponse A: ```Plants convert sunlight into energy```\n\nResponse B: ```Photosynthesis is the process where plants convert light energy into chemical energy using chlorophyll, water, and carbon dioxide to produce glucose and oxygen.```\n\n",
        ]
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
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
                    # Format test message with chat template
                    test_messages = [
                        {
                            "role": "system",
                            "content": """Let's think step by step to judge which response is better for the given prompt. Please keep your thoughts clear and concise and at max around 300 words. The output should be in the following format:\n```## Rationale: <Your reasoning>\n## Winner: <model_a or model_b>```\n\n"""
                        },
                        {
                            "role": "user",
                            "content": test_prompt
                        }
                    ]
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
    BASE_MODEL = "Qwen/Qwen1.5-72B-Chat"
    OUTPUT_DIR = "qwen-72b-cot-wsdm"
    HUB_MODEL_ID = "ruggsea/Qwen72B-CoT-WSDM"

    # Setup environment variables for caching
    cache_dir = os.path.expanduser("/hpcwork/ba214121/.cache/huggingface")
    checkpoint_dir = os.path.join(cache_dir, "checkpoints", OUTPUT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["WANDB_DIR"] = os.path.dirname(cache_dir)

    # Initialize wandb
    wandb.init(project="wsdm_cot_finetune")

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

        # Load dataset
        dataset = load_dataset("ruggsea/wsdm2024-deepseek-cot", split="train")
        
        def format_chat(example):
            messages = [
                {
                    "role": "system",
                    "content": """Let's think step by step to judge which response is better for the given prompt. Please keep your thoughts clear and concise and at max around 300 words. The output should be in the following format:\n```## Rationale: <Your reasoning>\n## Winner: <model_a or model_b>```\n\n"""
                },
                {
                    "role": "user",
                    "content": f"Prompt: {example['prompt']}\n\nResponse A: ```{example['response_a']}```\n\nResponse B: ```{example['response_b']}```\n\n"
                },
                {
                    "role": "assistant",
                    "content": f"## Rationale: {example['rationale']}\n## Winner: {example['winner']}"
                }
            ]
            return {
                "text": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False
                )
            }
        
        # Process dataset
        processed_dataset = dataset.map(
            format_chat,
            remove_columns=dataset.column_names,
            num_proc=8
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
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        )

        # Training arguments
        training_args = SFTConfig(
            output_dir=checkpoint_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            logging_dir=os.path.join(cache_dir, "logs"),
            save_strategy="epoch",
            eval_strategy="steps",
            eval_steps=500,
            do_eval=True,
            report_to="wandb",
            run_name=f"qwen-72b-cot-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
            warmup_ratio=0.03,
            group_by_length=True,
            push_to_hub=False,  # We'll handle this manually after merging
            hub_model_id=HUB_MODEL_ID,
            max_seq_length=4096,
            packing=False,
            weight_decay=0.01,
            max_grad_norm=0.3,
            lr_scheduler_type="linear",
            seed=3407,
        )

        # Initialize trainer with generation callback
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            peft_config=peft_config,
            data_collator=DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer),
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
            trust_remote_code=True,
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