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
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct"
]

from huggingface_hub import login
import os

# Get API key from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face API token.")
login(token=hf_token)

MAX_RETRIES = 3
MAX_NEW_TOKENS = 4096
MAX_INPUT_LENGTH = 32000

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/qwen_eval_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

class QwenEvaluator:
    def __init__(self, model_name, run):
        self.model_name = model_name
        self.run = run
        logging.info(f"Initializing model {model_name}")
        
        # Extract model size for group name
        self.size = model_name.split('-')[1].lower()  # e.g., "3b", "7b", "14b"
        
        # Log model initialization
        self.run.config.update({
            "model_name": model_name,
            "model_size": self.size,
            "max_retries": MAX_RETRIES,
        })
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
    def simple_classify(self, row, retry_count=0):
        # Format as chat messages
        messages = [
            {
                "role": "system", 
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            },
            {
                "role": "user", 
                "content": f"""Given these two responses to the prompt: "{row['prompt']}"
Response A: {row['response_a']}
Response B: {row['response_b']}
Which response is better? Answer with just A or B."""
            }
        ]
        
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            input_length = inputs.input_ids.shape[1]
            if input_length > MAX_INPUT_LENGTH:
                logging.warning(f"Skipping row {row['id']}: Input length {input_length} exceeds maximum {MAX_INPUT_LENGTH}")
                return None, None
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            
            # Get only the new tokens
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response_length = len(generated_ids[0])
            
            # Look for A or B in different formats
            prediction = None
            
            # Check for Final Answer format
            if "Final Answer" in response:
                final_answer_part = response.split("Final Answer")[-1].strip()
                if 'A' in final_answer_part[:10] and 'B' not in final_answer_part[:10]:
                    prediction = 'A'
                elif 'B' in final_answer_part[:10] and 'A' not in final_answer_part[:10]:
                    prediction = 'B'
            
            # Check for LaTeX format
            elif "\\boxed" in response:
                if '\\boxed{\\text{A}}' in response or '\\boxed{A}' in response:
                    prediction = 'A'
                elif '\\boxed{\\text{B}}' in response or '\\boxed{B}' in response:
                    prediction = 'B'
            
            # Standard format (fallback)
            elif 'A' in response and 'B' not in response:
                prediction = 'A'
            elif 'B' in response and 'A' not in response:
                prediction = 'B'
            
            if prediction is None:
                logging.warning(f"Invalid simple classification response for ID {row['id']}: '{response}'")
                if retry_count < MAX_RETRIES:
                    logging.info(f"Retrying simple classification for ID {row['id']} (attempt {retry_count + 1})")
                    time.sleep(1)
                    return self.simple_classify(row, retry_count + 1)
                else:
                    logging.error(f"Max retries reached for simple classification ID {row['id']}")
            
            return prediction, response_length
            
        except Exception as e:
            logging.error(f"Error in simple classification for ID {row['id']}: {str(e)}")
            if retry_count < MAX_RETRIES:
                logging.info(f"Retrying due to error for ID {row['id']} (attempt {retry_count + 1})")
                time.sleep(1)
                return self.simple_classify(row, retry_count + 1)
            return None, None

    def rationale_classify(self, row, retry_count=0):
        # Format as chat messages
        messages = [
            {
                "role": "system", 
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"""Given these two responses to the prompt: "{row['prompt']}"
Response A: {row['response_a']}
Response B: {row['response_b']}
Explain step by step which response is better and why, then conclude with "Therefore, Response [A/B] is better." """
            }
        ]
        
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            input_length = inputs.input_ids.shape[1]
            if input_length > MAX_INPUT_LENGTH:
                logging.warning(f"Skipping row {row['id']}: Input length {input_length} exceeds maximum {MAX_INPUT_LENGTH}")
                return None, None, None
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            
            # Get only the new tokens
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response_length = len(generated_ids[0])
            
            # Look for conclusion in different formats
            prediction = None
            
            # Check for standard "Therefore" format
            if "Therefore, Response A is better" in response:
                prediction = "A"
            elif "Therefore, Response B is better" in response:
                prediction = "B"
            
            # Check for Final Answer format
            elif "Final Answer" in response:
                final_answer_part = response.split("Final Answer")[-1].strip()
                if 'A' in final_answer_part[:10] and 'B' not in final_answer_part[:10]:
                    prediction = 'A'
                elif 'B' in final_answer_part[:10] and 'A' not in final_answer_part[:10]:
                    prediction = 'B'
            
            # Check for LaTeX format
            elif "\\boxed" in response:
                if '\\boxed{\\text{A}}' in response or '\\boxed{A}' in response:
                    prediction = 'A'
                elif '\\boxed{\\text{B}}' in response or '\\boxed{B}' in response:
                    prediction = 'B'
            
            if prediction is None:
                logging.warning(f"Invalid rationale classification response for ID {row['id']}: '{response}'")
                if retry_count < MAX_RETRIES:
                    logging.info(f"Retrying rationale classification for ID {row['id']} (attempt {retry_count + 1})")
                    time.sleep(1)
                    return self.rationale_classify(row, retry_count + 1)
                else:
                    logging.error(f"Max retries reached for rationale classification ID {row['id']}")
            
            return prediction, response_length, response
            
        except Exception as e:
            logging.error(f"Error in rationale classification for ID {row['id']}: {str(e)}")
            if retry_count < MAX_RETRIES:
                logging.info(f"Retrying due to error for ID {row['id']} (attempt {retry_count + 1})")
                time.sleep(1)
                return self.rationale_classify(row, retry_count + 1)
            return None, None, str(e)

def evaluate_batch(evaluator, df, method='simple', batch_size=100):
    results = []
    correct = 0
    total = 0
    failed_predictions = 0
    total_tokens = 0
    valid_responses = 0
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_results = []
        batch_tokens = 0
        batch_valid = 0
        
        for _, row in batch.iterrows():
            if method == 'simple':
                prediction, response_length = evaluator.simple_classify(row)
                rationale = None
            else:
                prediction, response_length, rationale = evaluator.rationale_classify(row)
                
            true_label = 'A' if row['winner'] == 'model_a' else 'B'
            
            if prediction is not None:
                correct += (prediction == true_label)
                total += 1
                if response_length is not None:
                    total_tokens += response_length
                    batch_tokens += response_length
                    valid_responses += 1
                    batch_valid += 1
            else:
                failed_predictions += 1
            
            batch_results.append({
                'id': row['id'],
                'prediction': prediction,
                'true_label': true_label,
                'correct': prediction == true_label if prediction else None,
                'response_length': response_length,
                'rationale': rationale
            })
            
        results.extend(batch_results)
        
        # Calculate metrics
        accuracy_so_far = correct/total if total > 0 else 0
        avg_tokens_so_far = total_tokens/valid_responses if valid_responses > 0 else 0
        batch_avg_tokens = batch_tokens/batch_valid if batch_valid > 0 else 0
        
        # Log batch metrics to wandb under qwen group
        evaluator.run.log({
            f"qwen/{evaluator.size}/{method}/batch": i//batch_size + 1,
            f"qwen/{evaluator.size}/{method}/running_accuracy": accuracy_so_far,
            f"qwen/{evaluator.size}/{method}/failed_predictions": failed_predictions,
            f"qwen/{evaluator.size}/{method}/processed_examples": total,
            f"qwen/{evaluator.size}/{method}/avg_response_tokens": avg_tokens_so_far,
            f"qwen/{evaluator.size}/{method}/batch_avg_response_tokens": batch_avg_tokens
        })
        
        logging.info(f"Batch {i//batch_size + 1} completed: "
                    f"Accuracy so far: {accuracy_so_far:.4f}, "
                    f"Failed predictions: {failed_predictions}, "
                    f"Avg tokens: {avg_tokens_so_far:.1f}")
    
    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / valid_responses if valid_responses > 0 else 0
    
    # Log final metrics under qwen group
    evaluator.run.log({
        f"qwen/{evaluator.size}/{method}/final_accuracy": accuracy,
        f"qwen/{evaluator.size}/{method}/total_failed_predictions": failed_predictions,
        f"qwen/{evaluator.size}/{method}/total_processed": total,
        f"qwen/{evaluator.size}/{method}/final_avg_response_tokens": avg_tokens
    })
    
    logging.info(f"Evaluation completed: "
                f"Final accuracy: {accuracy:.4f}, "
                f"Total failed predictions: {failed_predictions}, "
                f"Average response tokens: {avg_tokens:.1f}")
    return accuracy, results

def parse_args():
    parser = argparse.ArgumentParser(description='Qwen2.5 Evaluation Script')
    parser.add_argument('--model', type=str, choices=MODELS,
                       help='Model to evaluate')
    parser.add_argument('--method', type=str, choices=['simple', 'rationale', 'both'],
                       default='both',
                       help='Evaluation method (simple, rationale, or both)')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    setup_logging()
    logging.info("Starting evaluation script")
    
    # Initialize wandb with model size group
    model_size = args.model.split('-')[1].lower() if args.model else "all"
    wandb.init(
        project="wsdm-2024",
        entity="ruggsea",
        name=f"qwen_{model_size}_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": args.model,
            "method": args.method,
            "max_retries": MAX_RETRIES,
            "batch_size": 100
        },
        group="qwen"  # Group all Qwen runs together
    )
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Load dataset
        train_df = pd.read_parquet("wsdm/train.parquet")
        logging.info(f"Loaded dataset with {len(train_df)} rows")
        
        # Store all results
        all_results = {}
        
        # Use specified model or all models
        models_to_evaluate = [args.model] if args.model else MODELS
        
        for model_name in models_to_evaluate:
            all_results[model_name] = {}
            logging.info(f"\nStarting evaluation for model: {model_name}")
            
            evaluator = QwenEvaluator(model_name, wandb.run)
            
            # Run evaluations based on method argument
            if args.method in ['simple', 'both']:
                logging.info("Starting simple classification...")
                simple_acc, simple_results = evaluate_batch(evaluator, train_df, method='simple')
                all_results[model_name]['simple'] = {
                    'accuracy': simple_acc,
                    'results': simple_results
                }
            
            if args.method in ['rationale', 'both']:
                logging.info("Starting rationale-based classification...")
                rationale_acc, rationale_results = evaluate_batch(evaluator, train_df, method='rationale')
                all_results[model_name]['rationale'] = {
                    'accuracy': rationale_acc,
                    'results': rationale_results
                }
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = model_name.split('-')[1].lower()
            results_file = f"results/qwen_{model_size}_{args.method}_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(all_results[model_name], f, indent=2)
            
            # Log file to wandb
            wandb.save(results_file)
            logging.info(f"Saved results to {results_file}")
        
        # Create and log summary
        summary_data = []
        for model in all_results:
            model_size = model.split('-')[1].lower()
            model_data = {
                'model': model,
                'size': model_size,
            }
            if 'simple' in all_results[model]:
                model_data['simple_accuracy'] = all_results[model]['simple']['accuracy']
            if 'rationale' in all_results[model]:
                model_data['rationale_accuracy'] = all_results[model]['rationale']['accuracy']
            summary_data.append(model_data)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"results/qwen_summary_{args.method}_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Log summary to wandb
        wandb.log({"qwen/summary": wandb.Table(dataframe=summary_df)})
        
        logging.info(f"Saved summary to {summary_file}")
        print("\nEvaluation complete! Summary:")
        print(summary_df)
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        wandb.alert(
            title="Qwen Evaluation Failed",
            text=f"Error: {str(e)}",
            level=wandb.AlertLevel.ERROR
        )
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 