# ==================================================
# === export/save.py ===
"""Utility to save LoRA adapters locally or to Hugging Face."""
from pathlib import Path

def save_lora(model, tokenizer, output_dir: str = "gemma-3"):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapters saved to {output_dir}")

def push_lora(model, tokenizer, repo_id: str, token: str):
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)
    print(f"LoRA adapters pushed to {repo_id}")