import os
os.environ["HF_HOME"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["WANDB_DIR"] = "/hpcwork/ba214121/.cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "ruggsea/Llama3.1-SEP-Chat",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("ruggsea/Llama3.1-SEP-Chat")

# Test generation
print("\nTesting generation...")
messages = [{"role": "user", "content": "Say hi briefly."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("\nPrompt:", prompt.strip(), "\n")

# Get token IDs for special tokens
im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
end_of_text_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

gen_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=[im_end_token_id, end_of_text_token_id],
    pad_token_id=tokenizer.pad_token_id,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    forced_eos_token_id=[im_end_token_id, end_of_text_token_id]
)

outputs = model.generate(
    **tokenizer(prompt, return_tensors="pt").to(model.device),
    generation_config=gen_config
)

# Show clean response first
print("\n" + "="*80)
print("CLEAN RESPONSE:")
print("="*80)
clean_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Clean up any strange characters at the end
clean_response = clean_response.strip().rstrip('\ufffd')
print(clean_response)

# Show response with special tokens for debugging
print("\n" + "="*80)
print("RESPONSE WITH SPECIAL TOKENS:")
print("="*80)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
response = response.strip().rstrip('\ufffd')
print(response)
print("="*80) 