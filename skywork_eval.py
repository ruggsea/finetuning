import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import os
import logging
import time
import wandb
import argparse

MODELS = [
    "Skywork/Skywork-Critic-Llama-3.1-70B",
    "Skywork/Skywork-Critic-Llama-3.1-8B"
]

from huggingface_hub import login

# Get API key from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face API token.")
login(token=hf_token)

MAX_RETRIES = 3
MAX_NEW_TOKENS = 2048
MAX_INPUT_LENGTH = 32000

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/skywork_eval_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

class SkyworkEvaluator:
    def __init__(self, model_name, run):
        self.model_name = model_name
        self.run = run
        logging.info(f"Initializing Skywork model {model_name}")
        
        self.run.config.update({
            "model_name": model_name,
            "max_retries": MAX_RETRIES,
        })
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    
    def evaluate_pair(self, row, retry_count=0):
        # Using Skywork's official prompt template
        prompt_template = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{input}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
"""
        
        user_message = prompt_template.format(
            input=row['prompt'],
            response_a=row['response_a'],
            response_b=row['response_b']
        )
        
        conversation = [{"role": "user", "content": user_message}]
        
        try:
            inputs = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            input_length = inputs.shape[1]
            if input_length > MAX_INPUT_LENGTH:
                logging.warning(f"Skipping row {row['id']}: Input length {input_length} exceeds maximum")
                return None
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Extract prediction based on Skywork's format
            if "[[A]]" in response:
                prediction = "A"
            elif "[[B]]" in response:
                prediction = "B"
            else:
                prediction = None
                logging.warning(f"Invalid response format for ID {row['id']}: '{response}'")
                if retry_count < MAX_RETRIES:
                    time.sleep(1)
                    return self.evaluate_pair(row, retry_count + 1)
                else:
                    logging.error(f"Max retries reached for ID {row['id']}")
            
            return prediction, response
            
        except Exception as e:
            logging.error(f"Error in evaluation for ID {row['id']}: {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(1)
                return self.evaluate_pair(row, retry_count + 1)
            return None, str(e)

def evaluate_batch(evaluator, df, batch_size=100):
    results = []
    correct = 0
    total = 0
    failed_predictions = 0
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_results = []
        
        for _, row in batch.iterrows():
            prediction, rationale = evaluator.evaluate_pair(row)
            true_label = 'A' if row['winner'] == 'model_a' else 'B'
            
            if prediction is not None:
                correct += (prediction == true_label)
                total += 1
            else:
                failed_predictions += 1
            
            batch_results.append({
                'id': row['id'],
                'prediction': prediction,
                'true_label': true_label,
                'correct': prediction == true_label if prediction else None,
                'rationale': rationale
            })
        
        results.extend(batch_results)
        
        # Log batch metrics
        accuracy_so_far = correct/total if total > 0 else 0
        evaluator.run.log({
            "batch": i//batch_size + 1,
            "running_accuracy": accuracy_so_far,
            "failed_predictions": failed_predictions,
            "processed_examples": total
        })
        
        logging.info(f"Batch {i//batch_size + 1} completed: "
                    f"Accuracy so far: {accuracy_so_far:.4f}, "
                    f"Failed predictions: {failed_predictions}")
    
    accuracy = correct / total if total > 0 else 0
    
    # Log final metrics
    evaluator.run.log({
        "final_accuracy": accuracy,
        "total_failed_predictions": failed_predictions,
        "total_processed": total
    })
    
    logging.info(f"Evaluation completed: "
                f"Final accuracy: {accuracy:.4f}, "
                f"Total failed predictions: {failed_predictions}")
    
    return accuracy, results

def parse_args():
    parser = argparse.ArgumentParser(description='Skywork Critic Evaluation Script')
    parser.add_argument('--model', type=str, choices=MODELS,
                       help='Skywork model to evaluate')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()
    logging.info("Starting Skywork evaluation script")
    
    wandb.init(
        project="wsdm-2024",
        entity="ruggsea",
        name=f"skywork_{args.model.split('/')[-1]}",
        config={
            "model": args.model,
            "max_retries": MAX_RETRIES,
            "batch_size": 100
        }
    )
    
    os.makedirs('results', exist_ok=True)
    
    try:
        train_df = pd.read_parquet("wsdm/train.parquet")
        logging.info(f"Loaded dataset with {len(train_df)} rows")
        
        models_to_evaluate = [args.model] if args.model else MODELS
        
        for model_name in models_to_evaluate:
            logging.info(f"\nStarting evaluation for model: {model_name}")
            evaluator = SkyworkEvaluator(model_name, wandb.run)
            
            accuracy, results = evaluate_batch(evaluator, train_df)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"results/skywork_{model_name.split('/')[-1]}_{timestamp}.json"
            
            results_data = {
                'model': model_name,
                'accuracy': accuracy,
                'results': results
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            wandb.save(results_file)
            logging.info(f"Saved results to {results_file}")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        wandb.alert(
            title="Skywork Evaluation Failed",
            text=f"Error: {str(e)}",
            level=wandb.AlertLevel.ERROR
        )
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 