import os
os.environ["HF_HOME"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/hpcwork/ba214121/.cache/huggingface"
os.environ["WANDB_DIR"] = "/hpcwork/ba214121/.cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Show special tokens in output")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "ruggsea/Llama3.1-8B-SEP-Chat",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "ruggsea/Llama3.1-8B-SEP-Chat",
        trust_remote_code=True
    )

    # Get special token IDs
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end_of_text_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    
    # Setup generation config
    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    # Chat loop
    messages = []
    print("\nChat started! Type 'quit' to exit.")
    print("Debug mode:", "ON" if args.debug else "OFF")
    print("-----------------------------------")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        # Add user message and create prompt
        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        if args.debug:
            print("\nPrompt with special tokens:")
            print(prompt)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            generation_config=gen_config
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=not args.debug)
        
        # Add assistant message and print response
        messages.append({"role": "assistant", "content": response})
        print("\nAssistant" + (" (with special tokens):" if args.debug else ":"), response)

if __name__ == "__main__":
    main() 