#!/usr/bin/env python3

import json
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel
from bert_score import BERTScorer

def main():
    # ------------------------------------------------------------------
    # 1. Load Model + Tokenizer
    # ------------------------------------------------------------------
    model_path = "../Llama-3.2-1B"  # Adjust to your local model path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",       # use GPU if available
        torch_dtype=torch.float16
    )
    # Load your LoRA fine-tuned adapter
    lora_path = "../lora-llama-finetuned"
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Load Test Data
    # ------------------------------------------------------------------
    test_file = "./instruction_finetuning_data_test_subset.jsonl"   # JSONL with { "paper_id", "prompt", "response" }
    test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    # ------------------------------------------------------------------
    # 3. Initialize BERTScore
    # ------------------------------------------------------------------
    # BERTScore can use large base models, but "roberta-large" is a common choice for English
    scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)

    # We'll store references (gold) and candidates (model outputs) for batch scoring
    references = []
    candidates = []
    results = []

    # ------------------------------------------------------------------
    # 4. Generate & Compare
    # ------------------------------------------------------------------
    for entry in test_data:
        paper_id = entry.get("paper_id", "")
        prompt_text = entry.get("prompt", "")
        gold_response = entry.get("response", "")  # the true label

        # Format your prompt as done during training
        prompt_formatted = f"Instruction: {prompt_text}\nInput:\nAnswer:"
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_tokens = model.generate(**inputs, max_new_tokens=256)
        model_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # We'll store pairs for later BERTScore computation
        references.append(gold_response)
        candidates.append(model_output)

        results.append({
            "paper_id": paper_id,
            "prompt": prompt_text,
            "gold_response": gold_response,
            "model_output": model_output
        })

    # ------------------------------------------------------------------
    # 5. Compute BERTScore
    # ------------------------------------------------------------------
    # The 'score' method expects lists of strings
    # returns precision, recall, f1 for each pair
    P, R, F = scorer.score(candidates, references)

    # Convert each to a float
    # (BERTScore returns torch tensors)
    P = [float(p) for p in P]
    R = [float(r) for r in R]
    F = [float(f) for f in F]

    # Calculate averages
    avg_precision = sum(P) / len(P) if P else 0.0
    avg_recall = sum(R) / len(R) if R else 0.0
    avg_f1 = sum(F) / len(F) if F else 0.0

    # ------------------------------------------------------------------
    # 6. Print or Save the Results
    # ------------------------------------------------------------------
    print("=== BERTScore Results ===")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")

    # Optionally, print pairwise F1 for each example
    # for idx, item in enumerate(results):
    #     print("="*80)
    #     print(f"Paper ID: {item['paper_id']}")
    #     print(f"Prompt:\n{item['prompt']}")
    #     print(f"\nGold:\n{item['gold_response']}")
    #     print(f"\nModel:\n{item['model_output']}")
    #     print(f"\nBERTScore F1: {F[idx]:.4f}")

    # Save all results + BERTScore to a JSON
    # We'll attach the P, R, and F scores for each entry
    final_output = []
    for idx, item in enumerate(results):
        final_output.append({
            "paper_id": item["paper_id"],
            "prompt": item["prompt"],
            "gold_response": item["gold_response"],
            "model_output": item["model_output"],
            "bert_p": P[idx],
            "bert_r": R[idx],
            "bert_f1": F[idx]
        })

    with open("test_eval_with_bertscore.json", "w", encoding="utf-8") as out_f:
        json.dump({
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "results": final_output
        }, out_f, indent=2)

    print("\nEvaluation complete. Detailed results saved to 'test_eval_with_bertscore.json'.")

if __name__ == "__main__":
    main()
