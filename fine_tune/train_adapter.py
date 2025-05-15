import argparse
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def train_adapter(base_model: str, data_path: str, output_dir: str):
    # ─── Setup tokenizer & model ─────────────────────────────────────────────
    print("🔄 Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # GPT-2 has no pad token by default—use EOS as pad
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    print("✅ Model loaded.")

    # ─── Configure LoRA ──────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention & output proj
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    print("✅ PEFT LoRA adapter injected.")

    # ─── Load & preprocess dataset ───────────────────────────────────────────
    print(f"🔄 Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files={"train": data_path}, split="train")
    print(f"✅ Dataset loaded with {len(dataset)} examples.")

    def tokenize_fn(examples):
        # Combine instruction & response into one text sequence
        texts = [
            f"Instruction: {ins}\nResponse: {res}"
            for ins, res in zip(examples["instruction"], examples["response"])
        ]
        return tokenizer(texts, padding=True, truncation=True)

    print("🔄 Tokenizing and batching dataset...")
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("✅ Tokenization complete.")

    # ─── Training arguments & data collator ──────────────────────────────────
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ─── Trainer setup & launch ──────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"✅ Training complete. Adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapter on JSONL data")
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="HuggingFace model name (e.g. gpt2, gpt2-medium)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSONL file with 'instruction' and 'response' fields",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fine_tune/output",
        help="Directory to save the trained adapter",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_adapter(args.base_model, args.data, args.output_dir)
