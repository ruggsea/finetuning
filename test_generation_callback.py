#!/usr/bin/env python3

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    GenerationConfig,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig, setup_chat_format
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
import wandb
from trl.models.utils import unwrap_model_for_generation


def _generate_completion(model, tokenizer, prompt, generation_config, accelerator=None):
    print("\nStarting generation...")
    print(f"Prompt: {prompt}")
    print(f"Model device: {model.device}")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Move to device and ensure correct dtype
    inputs = {k: v.to(dtype=torch.long if k == 'input_ids' else torch.bfloat16, device=model.device) for k, v in inputs.items()}
    print("Moved inputs to device with correct dtype")

    # Generate with unwrapped model
    print("Unwrapping model for generation...")
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        print("Model unwrapped successfully")
        unwrapped_model.eval()
        print("Model set to eval mode")
        
        # Convert model to bfloat16
        unwrapped_model = unwrapped_model.to(dtype=torch.bfloat16)
        print("Converted model to bfloat16")
        
        with torch.no_grad():
            print("Starting generation...")
            outputs = unwrapped_model.generate(
                **inputs,
                generation_config=generation_config,
            )
            print("Generation completed")
    
    # Decode and return only the generated part
    completion = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"Generated completion: {completion}")
    return completion

class GenerationCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, steps=5):  # Reduced steps for testing
        print("\nInitializing GenerationCallback...")
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.steps = steps
        self.test_prompt = "Once upon a time"
        self.generation_config = GenerationConfig(
            max_new_tokens=20,  # Reduced for testing
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        self._last_logged_step = -1
        print("GenerationCallback initialized successfully")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self._last_logged_step:
            return

        if state.global_step % self.steps == 0:
            print(f"\nStep {state.global_step}: Attempting generation...")
            print(f"Trainer state: model={hasattr(self.trainer, 'model')}, accelerator={hasattr(self.trainer, 'accelerator')}")
            
            try:
                # Generate completion
                print("Starting generation process...")
                response = _generate_completion(
                    model=self.trainer.model,
                    tokenizer=self.tokenizer,
                    prompt=self.test_prompt,
                    generation_config=self.generation_config,
                    accelerator=self.trainer.accelerator
                )
                print("Generation completed successfully")
                
                # Print
                print(f"\nStep {state.global_step} generation test:")
                print(f"Prompt: {self.test_prompt}")
                print(f"Response: {response}\n")
                
                # Update last logged step
                self._last_logged_step = state.global_step
                print("Updated last logged step")
                
            except Exception as e:
                print(f"Generation failed with error: {str(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                
            finally:
                if hasattr(self.trainer, "model"):
                    print("Setting model back to train mode")
                    self.trainer.model.train()
                    print("Model set back to train mode")

def main():
    # Create a larger dummy dataset with longer conversations
    data = [
        {"messages": [
            {"role": "user", "content": "Tell me about artificial intelligence."},
            {"role": "assistant", "content": "Artificial Intelligence (AI) is a broad field of computer science focused on creating intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, understanding natural language, and perception. AI systems can be trained on large amounts of data to recognize patterns and make decisions."},
            {"role": "user", "content": "What are the main types of AI?"},
            {"role": "assistant", "content": "There are two main types of AI: Narrow AI (or Weak AI) which is designed for specific tasks, and General AI (or Strong AI) which would have human-like general intelligence. Currently, all AI systems are Narrow AI. Examples include image recognition, language models, and game-playing AI."}
        ]},
        {"messages": [
            {"role": "user", "content": "How does machine learning work?"},
            {"role": "assistant", "content": "Machine learning works by identifying patterns in data. The system learns from examples rather than being explicitly programmed. It uses algorithms to parse data, learn from it, and make informed decisions based on what it has learned."},
            {"role": "user", "content": "Can you give me an example?"},
            {"role": "assistant", "content": "A common example is email spam detection. The system learns from millions of emails labeled as spam or not spam. It identifies patterns in these emails like certain words, sender patterns, and formatting that are common in spam. Then it uses these patterns to classify new emails."}
        ]},
        {"messages": [
            {"role": "user", "content": "What is deep learning?"},
            {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. These networks are inspired by the human brain and can automatically learn representations from data without manual feature engineering."},
            {"role": "user", "content": "Where is it used?"},
            {"role": "assistant", "content": "Deep learning is used in many applications including image and speech recognition, natural language processing, autonomous vehicles, medical diagnosis, and game playing. For example, the image recognition in your smartphone's camera uses deep learning."}
        ]},
    ] * 100  # Create 300 samples by repeating 100 times
    dataset = Dataset.from_list(data)
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(eval_dataset)}")

    try:
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Set CUDA device
        torch.cuda.set_device(0)  # Use first GPU
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM-135M",
            quantization_config=bnb_config,
            device_map={"": 0},  # Use GPU 0
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM-135M",
            trust_remote_code=True,
        )
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        # Prepare model for training
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        # set up chat format
        model, tokenizer = setup_chat_format(model, tokenizer)
        
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
            output_dir="gpt2-test",
            max_steps=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            learning_rate=2e-5,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            evaluation_strategy="no",
            do_eval=False,
            report_to="none",
            warmup_ratio=0.0,
            group_by_length=True,
            packing=True,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
            ddp_find_unused_parameters=False,  # Disable DDP
            local_rank=-1,  # Disable distributed training
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
        
        # Add generation callback
        generation_callback = GenerationCallback(trainer, tokenizer, steps=5)
        trainer.add_callback(generation_callback)

        # Train
        model.config.use_cache = False
        trainer.train()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 