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
from vllm import LLM, SamplingParams

MODELS = [
    "Qwen/QwQ-32B-Preview",
]

from huggingface_hub import login
import os

# Get API key from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face API token.")
login(token=hf_token)

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
            logging.FileHandler(f'logs/qwq_eval_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

class QwQEvaluator:
    def __init__(self, model_name, run):
        self.model_name = model_name
        self.run = run
        logging.info(f"Initializing model {model_name}")
        
        # Log model initialization
        self.run.config.update({
            "model_name": model_name,
            "max_retries": MAX_RETRIES,
        })
        
        # Initialize vLLM engine instead of transformers
        self.llm = LLM(
            model=model_name,
            dtype="float16",
            trust_remote_code=True
        )
        
        # We still need tokenizer for length calculations
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _batch_generate(self, prompts, max_tokens=MAX_NEW_TOKENS):
        """Helper method to handle batch generation with vLLM"""
        try:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=1.0  # Changed from 0.0 to 1.0 for more natural responses
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
        except Exception as e:
            logging.error(f"vLLM generation error: {str(e)}")
            return [""] * len(prompts)

    def simple_classify(self, rows, retry_count=0):
        """Modified to handle batch of rows"""
        try:
            # Format all prompts in batch
            prompts = []
            for row in rows:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
                    },
                    {
                        "role": "user",
                        "content": f"""Given these two responses to the prompt: "{row['prompt']}"
Response A: {row['response_a']}
Response B: {row['response_b']}
Which response is better? Answer with just A or B."""
                    }
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)

            # Generate all responses at once
            responses = self._batch_generate(prompts)
            
            results = []
            for i, response in enumerate(responses):
                prediction = None
                response_length = len(self.tokenizer.encode(response))
                
                # Extract prediction using existing logic
                if "Final Answer" in response:
                    final_answer_part = response.split("Final Answer")[-1].strip()
                    if 'A' in final_answer_part[:50] and 'B' not in final_answer_part[:50]:
                        prediction = 'A'
                    elif 'B' in final_answer_part[:50] and 'A' not in final_answer_part[:50]:
                        prediction = 'B'
                elif '\\boxed' in response:
                    if '\\boxed{A}' in response or '\\boxed{\\text{A}}' in response:
                        prediction = 'A'
                    elif '\\boxed{B}' in response or '\\boxed{\\text{B}}' in response:
                        prediction = 'B'
                elif 'A' in response and 'B' not in response:
                    prediction = 'A'
                elif 'B' in response and 'A' not in response:
                    prediction = 'B'
                
                results.append((prediction, response_length))
                
            return results

        except Exception as e:
            logging.error(f"Error in batch simple classification: {str(e)}")
            if retry_count < MAX_RETRIES:
                logging.info(f"Retrying batch (attempt {retry_count + 1})")
                time.sleep(1)
                return self.simple_classify(rows, retry_count + 1)
            return [(None, None)] * len(rows)

    def rationale_classify(self, rows, retry_count=0):
        """Modified to handle batch of rows"""
        try:
            # Format all prompts in batch
            prompts = []
            for row in rows:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
                    },
                    {
                        "role": "user",
                        "content": f"""Given these two responses to the prompt: "{row['prompt']}"
Response A: {row['response_a']}
Response B: {row['response_b']}
Explain step by step which response is better and why, then conclude with "Therefore, Response [A/B] is better." """
                    }
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)

            # Generate all responses at once
            responses = self._batch_generate(prompts)
            
            results = []
            for i, response in enumerate(responses):
                prediction = None
                response_length = len(self.tokenizer.encode(response))
                
                # Extract prediction using existing logic
                if "Therefore, Response A is better" in response:
                    prediction = "A"
                elif "Therefore, Response B is better" in response:
                    prediction = "B"
                elif "Final Answer" in response:
                    final_answer_part = response.split("Final Answer")[-1].strip()
                    if 'A' in final_answer_part[:50] and 'B' not in final_answer_part[:50]:
                        prediction = 'A'
                    elif 'B' in final_answer_part[:50] and 'A' not in final_answer_part[:50]:
                        prediction = 'B'
                elif '\\boxed' in response:
                    if '\\boxed{A}' in response or '\\boxed{\\text{A}}' in response:
                        prediction = 'A'
                    elif '\\boxed{B}' in response or '\\boxed{\\text{B}}' in response:
                        prediction = 'B'
                
                results.append((prediction, response_length, response))
                
            return results

        except Exception as e:
            logging.error(f"Error in batch rationale classification: {str(e)}")
            if retry_count < MAX_RETRIES:
                logging.info(f"Retrying batch (attempt {retry_count + 1})")
                time.sleep(1)
                return self.rationale_classify(rows, retry_count + 1)
            return [(None, None, str(e))] * len(rows)

def evaluate_batch(evaluator, df, method='simple', batch_size=100):
    results = []
    correct = 0
    total = 0
    failed_predictions = 0
    total_tokens = 0
    valid_responses = 0
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        # Get batch predictions
        if method == 'simple':
            batch_predictions = evaluator.simple_classify(batch.to_dict('records'))
            batch_results = [
                {
                    'id': row['id'],
                    'prediction': pred,
                    'true_label': 'A' if row['winner'] == 'model_a' else 'B',
                    'correct': pred == ('A' if row['winner'] == 'model_a' else 'B') if pred else None,
                    'response_length': length,
                    'rationale': None
                }
                for (pred, length), row in zip(batch_predictions, batch.to_dict('records'))
            ]
        else:
            batch_predictions = evaluator.rationale_classify(batch.to_dict('records'))
            batch_results = [
                {
                    'id': row['id'],
                    'prediction': pred,
                    'true_label': 'A' if row['winner'] == 'model_a' else 'B',
                    'correct': pred == ('A' if row['winner'] == 'model_a' else 'B') if pred else None,
                    'response_length': length,
                    'rationale': rationale
                }
                for (pred, length, rationale), row in zip(batch_predictions, batch.to_dict('records'))
            ]
        
        # Process batch results
        batch_tokens = 0
        batch_valid = 0
        
        for result in batch_results:
            if result['prediction'] is not None:
                correct += result['correct']
                total += 1
                if result['response_length'] is not None:
                    total_tokens += result['response_length']
                    batch_tokens += result['response_length']
                    valid_responses += 1
                    batch_valid += 1
            else:
                failed_predictions += 1
        
        results.extend(batch_results)
        
        # Calculate and log metrics (rest of the function remains the same)
        accuracy_so_far = correct/total if total > 0 else 0
        avg_tokens_so_far = total_tokens/valid_responses if valid_responses > 0 else 0
        batch_avg_tokens = batch_tokens/batch_valid if batch_valid > 0 else 0
        
        # Log batch metrics to wandb under qwq group
        evaluator.run.log({
            f"qwq/{method}/batch": i//batch_size + 1,
            f"qwq/{method}/running_accuracy": accuracy_so_far,
            f"qwq/{method}/failed_predictions": failed_predictions,
            f"qwq/{method}/processed_examples": total,
            f"qwq/{method}/avg_response_tokens": avg_tokens_so_far,
            f"qwq/{method}/batch_avg_response_tokens": batch_avg_tokens
        })
        
        logging.info(f"Batch {i//batch_size + 1} completed: "
                    f"Accuracy so far: {accuracy_so_far:.4f}, "
                    f"Failed predictions: {failed_predictions}, "
                    f"Avg tokens: {avg_tokens_so_far:.1f}")
    
    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / valid_responses if valid_responses > 0 else 0
    
    # Log final metrics under qwq group
    evaluator.run.log({
        f"qwq/{method}/final_accuracy": accuracy,
        f"qwq/{method}/total_failed_predictions": failed_predictions,
        f"qwq/{method}/total_processed": total,
        f"qwq/{method}/final_avg_response_tokens": avg_tokens
    })
    
    logging.info(f"Evaluation completed: "
                f"Final accuracy: {accuracy:.4f}, "
                f"Total failed predictions: {failed_predictions}, "
                f"Average response tokens: {avg_tokens:.1f}")
    return accuracy, results

def parse_args():
    parser = argparse.ArgumentParser(description='QwQ Evaluation Script')
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
    
    # Initialize wandb
    wandb.init(
        project="wsdm-2024",
        entity="ruggsea",
        name=f"qwq_{args.model.split('/')[-1]}_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": args.model,
            "method": args.method,
            "max_retries": MAX_RETRIES,
            "batch_size": 100
        },
        group="qwq"  # Group all QwQ runs together
    )
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Load dataset
        train_df = pd.read_parquet("wsdm/train.parquet")
        logging.info(f"Loaded dataset with {len(train_df)} rows")
        
        # Store all results
        all_results = {}
        
        # Use specified model
        models_to_evaluate = [args.model] if args.model else MODELS
        
        for model_name in models_to_evaluate:
            all_results[model_name] = {}
            logging.info(f"\nStarting evaluation for model: {model_name}")
            
            evaluator = QwQEvaluator(model_name, wandb.run)
            
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
            results_file = f"results/qwq_{model_name.split('/')[-1]}_{args.method}_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(all_results[model_name], f, indent=2)
            
            # Log file to wandb
            wandb.save(results_file)
            logging.info(f"Saved results to {results_file}")
        
        # Create and log summary
        summary_data = []
        for model in all_results:
            model_data = {
                'model': model,
            }
            if 'simple' in all_results[model]:
                model_data['simple_accuracy'] = all_results[model]['simple']['accuracy']
            if 'rationale' in all_results[model]:
                model_data['rationale_accuracy'] = all_results[model]['rationale']['accuracy']
            summary_data.append(model_data)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"results/qwq_summary_{args.method}_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Log summary to wandb
        wandb.log({"qwq/summary": wandb.Table(dataframe=summary_df)})
        
        logging.info(f"Saved summary to {summary_file}")
        print("\nEvaluation complete! Summary:")
        print(summary_df)
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        wandb.alert(
            title="QwQ Evaluation Failed",
            text=f"Error: {str(e)}",
            level=wandb.AlertLevel.ERROR
        )
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 