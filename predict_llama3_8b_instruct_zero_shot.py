#!/usr/bin/env python3

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_model_and_tokenizer():
    """
    Loads the original LLaMA-3.1-8B-Instruct model and its tokenizer.
    """
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Ensure pad token is defined; if not, use the eos token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def generate_predictions(model, tokenizer, test_file, max_length=512, max_new_tokens=256):
    """
    Iterates over each test example (formatted as JSONL) and generates a prediction.
    Each test example should contain:
      {
         "paper_id": "<paper_id>",
         "messages": [
             {"role": "system", "content": "<system prompt>"},
             {"role": "user", "content": "<instruction prompt>"}
         ]
      }
    The prompt is constructed as:
      "System message: <system prompt>\nUser instruction: <instruction prompt>\nAnswer:"
    The model output is parsed as JSON and stored under the key "review".
    """
    predictions = []
    # Count total lines to display a progress bar.
    with open(test_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())
    with open(test_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Generating predictions"):
            if not line.strip():
                continue
            example = json.loads(line)
            paper_id = example.get("paper_id", "")
            messages = example.get("messages", [])
            system_prompt = ""
            instruction_prompt = ""
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    instruction_prompt = content
            # Construct the prompt.
            prompt = f"System message: {system_prompt}\nUser instruction: {instruction_prompt}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = inputs.to(model.device)
            # Generate prediction with explicit pad_token_id to avoid repeated warnings.
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            # Attempt to parse the output as JSON.
            try:
                review = json.loads(output_text)
            except Exception as e:
                print(f"Warning: Failed to parse JSON for paper {paper_id}: {e}")
                review = output_text  # fallback to raw text
            predictions.append({
                "paper_id": paper_id,
                "review": review
            })
    return predictions

def main():
    test_file = "data/test_data/zero_shot/test_data_2024_summary_prompts_one_sample.jsonl"
    model, tokenizer = load_model_and_tokenizer()
    predictions = generate_predictions(model, tokenizer, test_file)
    
    # Print predictions.
    for pred in predictions:
        print(f"Paper ID: {pred['paper_id']}")
        print(f"Review:\n{json.dumps(pred['review'], indent=2)}\n{'='*80}\n")
    
    # Save predictions to a JSONL file.
    with open("test_data_2024_summary_prompts_llama3_results.jsonl", "w", encoding="utf-8") as out_f:
        for pred in predictions:
            out_f.write(json.dumps(pred) + "\n")

if __name__ == "__main__":
    main()
