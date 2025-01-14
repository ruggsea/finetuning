from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("ruggsea/Llama3.1-SEP-Chat")

print("\nCurrent format:")
print("Special tokens:", tokenizer.special_tokens_map)
print("Chat template:", tokenizer.chat_template)

# Test with a simple message
messages = [{"role": "user", "content": "Hi"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

print("\nExample prompt:")
print(prompt) 