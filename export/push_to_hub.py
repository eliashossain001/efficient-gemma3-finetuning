# === scripts/push_to_hub.py ===
"""Standalone script to push LoRA or GGUF model to Hugging Face Hub."""
import argparse
from model.loader import load_model_and_tokenizer
from export.save import push_lora
from export.gguf_export import push_gguf

parser = argparse.ArgumentParser(description="Push fine-tuned models to Hugging Face Hub")
parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., username/model-name)")
parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
parser.add_argument("--type", choices=["lora", "gguf"], required=True, help="Type of model to push")
parser.add_argument("--quant", type=str, default="Q8_0", help="GGUF quantization type (Q8_0, F16, BF16)")
args = parser.parse_args()

model, tokenizer, _ = load_model_and_tokenizer()

if args.type == "lora":
    push_lora(model, tokenizer, args.repo_id, args.token)
elif args.type == "gguf":
    push_gguf(model, args.repo_id, args.token, quantization_type=args.quant)
    
    
# Follow this command to push this script: python -m export.push_to_hub --repo_id EliasHossain/gemma3-lora --token PASTE-THE-TOKEN-HERE --type lora

