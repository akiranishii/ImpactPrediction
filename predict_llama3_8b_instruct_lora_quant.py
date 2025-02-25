import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm  # Import tqdm for progress tracking

def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return model, tokenizer

def generate_predictions(model, tokenizer, test_file, max_length=512, max_new_tokens=256):
    predictions = []
    with open(test_file, "r", encoding="utf-8") as f:
        # Optionally count total lines for progress bar
        total_lines = sum(1 for line in f if line.strip())
        f.seek(0)
        for line in tqdm(f, total=total_lines, desc="Generating predictions"):
            if not line.strip():
                continue
            example = json.loads(line)
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
            prompt = f"System message: {system_prompt}\nUser instruction: {instruction_prompt}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = inputs.to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predictions.append({
                "paper_id": example.get("paper_id", ""),
                "prompt": prompt,
                "prediction": output_text
            })
    return predictions

def main():
    model_dir = "models/lora-llama8b-instruct-finetuned"
    test_file = "data/test_data/zero_shot/test_data_2024_summary_prompts.jsonl"
    
    model, tokenizer = load_model_and_tokenizer(model_dir)
    predictions = generate_predictions(model, tokenizer, test_file)
    
    for pred in predictions:
        print(f"Paper ID: {pred['paper_id']}")
        print(f"Prompt:\n{pred['prompt']}")
        print(f"Prediction:\n{pred['prediction']}\n{'='*80}\n")
    
    with open("test_data_2024_summary_prompts_predictions.jsonl", "w", encoding="utf-8") as out_f:
        for pred in predictions:
            out_f.write(json.dumps(pred) + "\n")

if __name__ == "__main__":
    main()
