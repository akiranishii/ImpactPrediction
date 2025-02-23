#!/usr/bin/env python3

import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import  Trainer, TrainingArguments, AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

class InstructionDataset(Dataset):
    """
    A Dataset for reading prompt–response pairs from a JSON/JSONL file where each entry has:
      {
        "paper_id": "...",
        "prompt": "...",
        "response": "..."
      }
    We train a causal language model (LLaMA) so that the model produces `response` given `prompt`.
    """
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Extract your fields
        # 'paper_id' is unused for training, so we ignore it
        prompt_text = example.get("prompt", "")
        response_text = example.get("response", "")

        # Construct an instruction-style prompt:
        # e.g. "Instruction: <prompt>\nAnswer: <response>"
        # You can adapt formatting to your preference:
        #   - Alpaca-like: "Instruction: {prompt}\nInput:\nAnswer: {response}"
        #   - Or any other style
        full_prompt = f"Instruction: {prompt_text}\nAnswer:"
        full_text = full_prompt + response_text

        # Tokenize the combined text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        # We only want the model to learn from the 'response' portion
        # so we "mask out" (set to -100) the prompt portion in the labels.
        # 1) Tokenize just the prompt to get its length
        with self.tokenizer.as_target_tokenizer():
            prompt_ids = self.tokenizer(
                full_prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"].squeeze()

        prompt_len = (prompt_ids != self.tokenizer.pad_token_id).sum().item()

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # ignore the prompt tokens in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    # ----------------------------
    # 1. Load your local dataset
    # ----------------------------
    dataset_file = "./instruction_finetuning_data_subset.jsonl"  # path to the  data file
    data = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # ----------------------------
    # 2. Prepare the tokenizer (use AutoTokenizer for LLaMA)
    # ----------------------------
    model_path = "../Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

    # LLaMA often doesn't define a pad_token, so we set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = 1024  # Increase if your prompts+responses are longer
    full_dataset = InstructionDataset(data, tokenizer, max_length)

    # Optionally split dataset into train/validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # ----------------------------
    # 3. Load LLaMA Model Locally
    # ----------------------------
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # ----------------------------
    # 4. Setup LoRA (PEFT)
    # ----------------------------
    peft_config = LoraConfig(
        r=8,               # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)

    # ----------------------------
    # 5. Training Arguments
    # ----------------------------
    training_args = TrainingArguments(
        output_dir="../lora-llama-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        warmup_steps=50,
        learning_rate=1e-4,
        fp16=True, #default is False and uses fp32 for cpu
        logging_dir="../logs",
        save_total_limit=2
    )

    # ----------------------------
    # 6. Trainer
    # ----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # ----------------------------
    # 7. Train
    # ----------------------------
    trainer.train()

    # ----------------------------
    # 8. Save LoRA Adapters
    # ----------------------------
    trainer.save_model("../lora-llama-finetuned")
    print("Training complete. Model saved in '../lora-llama-finetuned'.")

if __name__ == "__main__":
    main()
