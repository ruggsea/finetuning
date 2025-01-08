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
import re  # Add to imports if not already there
import argparse

MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]

from huggingface_hub import login
import os

# Get API key from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face API token.")
login(token=hf_token)

TEMPERATURES = [0.1, 0.3, 0.7, 1.0]
MAX_RETRIES = 3
MAX_NEW_TOKENS = 4096  # Very large but reasonable limit
MAX_INPUT_LENGTH = 32000  # Maximum input length we'll accept

# Setup logging
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    
    # File handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/eval_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

class ModelEvaluator:
    def __init__(self, model_name, temperature, run):
        self.model_name = model_name
        self.temperature = temperature
        self.run = run  # wandb run object
        logging.info(f"Initializing model {model_name} with temperature {temperature}")
        
        # Log model initialization
        self.run.config.update({
            "model_name": model_name,
            "temperature": temperature,
            "max_retries": MAX_RETRIES,
        })
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
                "role": "user", 
                "content": f"""Given these two responses to the prompt: "{row['prompt']}"
Response A: {row['response_a']}
Response B: {row['response_b']}
Which response is better? Answer with just A or B."""
            }
        ]
        
        try:
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            )
            
            input_length = inputs.shape[1]
            if input_length > MAX_INPUT_LENGTH:
                logging.warning(f"Skipping row {row['id']}: Input length {input_length} exceeds maximum {MAX_INPUT_LENGTH}")
                return None
            
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            
            # Decode the full response and look for A or B in the entire text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Find the actual response after the prompt
            prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            actual_response = response[len(prompt_text):].strip()
            
            # Use regex to find A or B, including cases with periods or other punctuation
            a_match = re.search(r'\bA\.?\b', actual_response)
            b_match = re.search(r'\bB\.?\b', actual_response)
            
            prediction = None
            if a_match and not b_match:
                prediction = 'A'
            elif b_match and not a_match:
                prediction = 'B'
            
            if prediction is None:
                logging.warning(f"Invalid simple classification response for ID {row['id']}: '{actual_response}'")
                if retry_count < MAX_RETRIES:
                    logging.info(f"Retrying simple classification for ID {row['id']} (attempt {retry_count + 1})")
                    time.sleep(1)
                    return self.simple_classify(row, retry_count + 1)
                else:
                    logging.error(f"Max retries reached for simple classification ID {row['id']}")
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error in simple classification for ID {row['id']}: {str(e)}")
            if retry_count < MAX_RETRIES:
                logging.info(f"Retrying due to error for ID {row['id']} (attempt {retry_count + 1})")
                time.sleep(1)
                return self.simple_classify(row, retry_count + 1)
            return None
    
    def rationale_classify(self, row, retry_count=0):
        # Format as chat messages
        messages = [
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
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            )
            
            input_length = inputs.shape[1]
            if input_length > MAX_INPUT_LENGTH:
                logging.warning(f"Skipping row {row['id']}: Input length {input_length} exceeds maximum {MAX_INPUT_LENGTH}")
                return None, f"Skipped: Input length {input_length} exceeds maximum"
            
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            if "Response A is better" in response:
                prediction = "A"
            elif "Response B is better" in response:
                prediction = "B"
            else:
                prediction = None
                logging.warning(f"Invalid rationale classification response for ID {row['id']}: '{response}'")
                if retry_count < MAX_RETRIES:
                    logging.info(f"Retrying rationale classification for ID {row['id']} (attempt {retry_count + 1})")
                    time.sleep(1)
                    return self.rationale_classify(row, retry_count + 1)
                else:
                    logging.error(f"Max retries reached for rationale classification ID {row['id']}")
            
            return prediction, response
            
        except Exception as e:
            logging.error(f"Error in rationale classification for ID {row['id']}: {str(e)}")
            if retry_count < MAX_RETRIES:
                logging.info(f"Retrying due to error for ID {row['id']} (attempt {retry_count + 1})")
                time.sleep(1)
                return self.rationale_classify(row, retry_count + 1)
            return None, str(e)

def evaluate_batch(evaluator, df, method='simple', batch_size=100):
    results = []
    correct = 0
    total = 0
    failed_predictions = 0
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_results = []
        
        for _, row in batch.iterrows():
            if method == 'simple':
                prediction = evaluator.simple_classify(row)
                rationale = None
            else:
                prediction, rationale = evaluator.rationale_classify(row)
                
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
        
        # Log batch metrics to wandb
        accuracy_so_far = correct/total if total > 0 else 0
        evaluator.run.log({
            f"{method}/batch": i//batch_size + 1,
            f"{method}/running_accuracy": accuracy_so_far,
            f"{method}/failed_predictions": failed_predictions,
            f"{method}/processed_examples": total
        })
        
        logging.info(f"Batch {i//batch_size + 1} completed: "
                    f"Accuracy so far: {accuracy_so_far:.4f}, "
                    f"Failed predictions: {failed_predictions}")
    
    accuracy = correct / total if total > 0 else 0
    
    # Log final metrics
    evaluator.run.log({
        f"{method}/final_accuracy": accuracy,
        f"{method}/total_failed_predictions": failed_predictions,
        f"{method}/total_processed": total
    })
    
    logging.info(f"Evaluation completed: "
                f"Final accuracy: {accuracy:.4f}, "
                f"Total failed predictions: {failed_predictions}")
    return accuracy, results

def parse_args():
    parser = argparse.ArgumentParser(description='WSDM Evaluation Script')
    parser.add_argument('--model', type=str, choices=MODELS,
                       help='Model to evaluate')
    parser.add_argument('--temperature', type=float,
                       help='Temperature for generation')
    parser.add_argument('--method', type=str, choices=['simple', 'rationale', 'both'],
                       default='both',
                       help='Evaluation method (simple, rationale, or both)')
    
    args = parser.parse_args()
    
    # Validate temperature after parsing
    if args.temperature is not None and args.temperature not in TEMPERATURES:
        parser.error(f"Temperature must be one of {TEMPERATURES}")
    
    return args

def main():
    args = parse_args()
    setup_logging()
    logging.info("Starting evaluation script")
    
    # Initialize wandb with a more descriptive name
    wandb.init(
        project="wsdm-2024",
        entity="ruggsea",
        name=f"{args.model.split('/')[-1]}_temp{args.temperature}_{args.method}",  # More descriptive run name
        config={
            "model": args.model,
            "temperature": args.temperature,
            "method": args.method,
            "max_retries": MAX_RETRIES,
            "batch_size": 100
        }
    )
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Load dataset
        train_df = pd.read_parquet("wsdm/train.parquet")
        logging.info(f"Loaded dataset with {len(train_df)} rows")
        
        # Store all results
        all_results = {}
        
        # Use either specified model or all models
        models_to_evaluate = [args.model] if args.model else MODELS
        temps_to_evaluate = [args.temperature] if args.temperature else TEMPERATURES
        
        for model_name in models_to_evaluate:
            all_results[model_name] = {}
            logging.info(f"\nStarting evaluation for model: {model_name}")
            
            for temp in temps_to_evaluate:
                # Remove the nested wandb.init since we already initialized it above
                logging.info(f"Evaluating temperature: {temp}")
                evaluator = ModelEvaluator(model_name, temp, wandb.run)  # Use the existing run
                
                # Run evaluations based on method argument
                if args.method in ['simple', 'both']:
                    logging.info("Starting simple classification...")
                    simple_acc, simple_results = evaluate_batch(evaluator, train_df, method='simple')
                else:
                    simple_acc, simple_results = None, None
                
                if args.method in ['rationale', 'both']:
                    logging.info("Starting rationale-based classification...")
                    rationale_acc, rationale_results = evaluate_batch(evaluator, train_df, method='rationale')
                else:
                    rationale_acc, rationale_results = None, None
                
                all_results[model_name][temp] = {
                    'simple': {
                        'accuracy': simple_acc,
                        'results': simple_results
                    } if simple_acc is not None else None,
                    'rationale': {
                        'accuracy': rationale_acc,
                        'results': rationale_results
                    } if rationale_acc is not None else None
                }
                
                # Save results and log to wandb
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"results/eval_{model_name.split('/')[-1]}_{temp}_{timestamp}.json"
                
                results_data = {
                    'model': model_name,
                    'temperature': temp,
                    'simple_accuracy': simple_acc,
                    'rationale_accuracy': rationale_acc,
                    'simple_results': simple_results,
                    'rationale_results': rationale_results
                }
                
                with open(results_file, 'w') as f:
                    json.dump(results_data, f, indent=2)
                
                # Log file to wandb
                wandb.save(results_file)
                logging.info(f"Saved results to {results_file}")
        
        # Create and log summary
        summary_data = []
        for model in all_results:
            for temp in all_results[model]:
                summary_data.append({
                    'model': model,
                    'temperature': temp,
                    'simple_accuracy': all_results[model][temp]['simple']['accuracy'],
                    'rationale_accuracy': all_results[model][temp]['rationale']['accuracy']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"results/summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Log summary to wandb
        wandb.log({"summary": wandb.Table(dataframe=summary_df)})
        
        logging.info(f"Saved summary to {summary_file}")
        print("\nEvaluation complete! Summary:")
        print(summary_df)
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        wandb.alert(
            title="Evaluation Failed",
            text=f"Error: {str(e)}",
            level=wandb.AlertLevel.ERROR
        )
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 