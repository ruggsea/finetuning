import pandas as pd
import json
import os
import aiohttp
import asyncio
import re
from tqdm.asyncio import tqdm_asyncio
from collections import defaultdict
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, Repository
import tempfile

results_buffer = defaultdict(dict)
moderation_failed_ids = set()  # Track IDs that fail moderation
buffer_size = 500
API_KEY = os.getenv("OPENAI_API_KEY")
REPO_ID = "ruggsea/wsdm2024-deepseek-cot-dataset"
MAX_RETRIES = 3

def clean_json_string(s):
    json_match = re.search(r'\{.*\}', s, re.DOTALL)
    if json_match:
        s = json_match.group(0)
    
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            s = re.sub(r':\s*"([^"]*)"', lambda m: ': "' + m.group(1).replace('\n', ' ').replace('  ', ' ') + '"', s)
            s = s.replace('\n', ' ').replace('\\', '\\\\').strip()
            return json.loads(s)
        except:
            rationale = re.search(r'"rationale":\s*"([^"]*)"', s)
            if rationale:
                return {"rationale": rationale.group(1)}
            return {"rationale": "Failed to parse response"}

async def generate_response(session, semaphore, row):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                winner = 'A' if row['winner'] == 'model_a' else 'B'
                formatted_prompt = f"""Given these two responses to the prompt: "{row['prompt']}"
Response A:\n{row['response_a']}\nResponse B:\n{row['response_b']}\nThe better response was: Response {winner}
Explain step by step why Response {winner} was better than Response {'A' if winner == 'B' else 'B'}.
IMPORTANT: Your response must be in valid JSON format with a single field called "rationale"."""

                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that explains preferences between responses."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    "temperature": 1.0,
                    "max_tokens": 8192
                }
                
                async with session.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        error_json = json.loads(error_text)
                        
                        # Check if it's a content moderation error
                        if (response.status == 400 and 
                            error_json.get("error", {}).get("message") == "Content Exists Risk"):
                            print(f"Content moderation failed for ID {row['id']}, skipping...")
                            moderation_failed_ids.add(row['id'])
                            return None
                            
                        print(f"Error {response.status} for ID {row['id']} (Attempt {attempt + 1}/{MAX_RETRIES}): {error_text}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None
                
            except Exception as e:
                print(f"Request error for ID {row['id']} (Attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
        
        return None

async def process_task(i, row, session, semaphore, pbar):
    try:
        response_content = await generate_response(session, semaphore, row)
        if response_content:
            parsed_response = clean_json_string(response_content)
            row_dict = row.to_dict()
            row_dict['rationale'] = str(parsed_response.get('rationale', 'Failed to extract rationale'))
            results_buffer[row['id']] = row_dict
            #print(f"Successfully processed ID {row['id']}")
            
            if len(results_buffer) >= buffer_size:
                print(f"Buffer full ({buffer_size} items), saving to disk...")
                await save_results('wsdm_deepseek/semisynthetic_train_deepseek.parquet')
                print("Save complete")
    except Exception as e:
        print(f"Error processing row {i} (ID {row['id']}): {str(e)}")
        row_dict = row.to_dict()
        row_dict['rationale'] = f"Error: {str(e)}"
        results_buffer[row['id']] = row_dict
    
    pbar.update(1)

async def save_results(cache_path):
    if not results_buffer:
        return
        
    try:
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            df = df[~df['id'].isin(results_buffer.keys())]
        else:
            df = pd.DataFrame()
        
        new_df = pd.DataFrame.from_records(list(results_buffer.values()))
        result_df = pd.concat([df, new_df], ignore_index=True) if not df.empty else new_df
        
        result_df['rationale'] = result_df['rationale'].astype(str)
        result_df.to_parquet(cache_path)
        results_buffer.clear()
    except Exception as e:
        print(f"Error saving results: {str(e)}")

async def upload_to_hub(result_df):
    print("Preparing dataset for upload...")
    
    # First, load and merge with existing dataset
    try:
        existing_df = load_dataset("agokrani/wsdm2024-8k-synthetic-cot")["train"].to_pandas()
        print(f"Loaded {len(existing_df)} examples from existing dataset")
        
        # Combine datasets
        combined_df = pd.concat([existing_df, result_df], ignore_index=True)
        print(f"Combined dataset has {len(combined_df)} examples")
    except Exception as e:
        print(f"Error loading existing dataset: {str(e)}")
        combined_df = result_df
    
    # Convert to HF Dataset
    dataset = Dataset.from_pandas(combined_df)
    
    # Push to hub
    dataset.push_to_hub(REPO_ID, private=False)
    
    # Clone repo to add README
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Repository(
            local_dir=tmp_dir,
            clone_from=REPO_ID,
            use_auth_token=True
        )
        
        # Create README content
        readme_content = """# WSDM 2024 Deepseek Chain-of-Thought Dataset

This dataset contains Chain-of-Thought (CoT) explanations generated using the Deepseek V3 model through their official API. The explanations describe why one response is better than another in a preference comparison task.

## Dataset Details

- The dataset combines examples from `agokrani/wsdm2024-8k-synthetic-cot` with new explanations generated using Deepseek.
- Each example contains:
  - A prompt
  - Two responses (A and B)
  - The winner (better response)
  - A rationale explaining why the winning response is better

## Generation Details

- Model: Deepseek Chat V3
- Temperature: 1.0
- Max Tokens: 8192
- Format: JSON with rationale field

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("ruggsea/wsdm2024-deepseek-cot-dataset")
```

## License

This dataset follows the same license as the original WSDM 2024 competition dataset.
"""
        
        # Write README
        with open(os.path.join(tmp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Commit and push
        repo.push_to_hub(commit_message="Add dataset documentation")
    
    print("Dataset uploaded and documented successfully!")

async def main():
    os.makedirs('wsdm_deepseek', exist_ok=True)
    train_df = pd.read_parquet("wsdm/train.parquet")
    cache_path = 'wsdm_deepseek/semisynthetic_train_deepseek.parquet'
    moderation_failed_path = 'wsdm_deepseek/moderation_failed_ids.json'
    
    # Load previously failed moderation IDs if they exist
    if os.path.exists(moderation_failed_path):
        with open(moderation_failed_path, 'r') as f:
            moderation_failed_ids.update(json.load(f))
    
    if os.path.exists(cache_path):
        cached_df = pd.read_parquet(cache_path)
        processed_ids = set(cached_df['id'].tolist())
        processed_count = len(processed_ids)
        print(f"Loaded {processed_count} cached entries")
    else:
        processed_ids = set()
        processed_count = 0

    semaphore = asyncio.Semaphore(1000)
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        pbar = tqdm_asyncio(total=len(train_df))
        pbar.update(processed_count)
            
        for i in range(len(train_df)):
            row = train_df.iloc[i]
            if row['id'] not in processed_ids and row['id'] not in moderation_failed_ids:
                tasks.append(process_task(i, row, session, semaphore, pbar))
        
        await asyncio.gather(*tasks)
        
        if results_buffer:
            await save_results(cache_path)
        
        pbar.close()
    
    # Save moderation failed IDs
    if moderation_failed_ids:
        with open(moderation_failed_path, 'w') as f:
            json.dump(list(moderation_failed_ids), f)
        print(f"Saved {len(moderation_failed_ids)} moderation failed IDs to {moderation_failed_path}")
    
    # Filter out error cases
    result_df = pd.read_parquet(cache_path)
    result_df = result_df[~result_df['rationale'].str.contains("Error|Failed")]
    result_df.to_parquet(cache_path)
    print("Processing complete!")
    
    # Upload to HuggingFace Hub
    print("Uploading to HuggingFace Hub...")
    await upload_to_hub(result_df)

if __name__ == "__main__":
    asyncio.run(main()) 