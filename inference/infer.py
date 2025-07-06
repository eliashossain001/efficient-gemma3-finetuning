# === inference/infer.py ===
"""Simple inference helper for finetuned Gemma 3 model."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import TextStreamer
from model.loader import load_model_and_tokenizer
import torch

def generate(prompt: str, max_new_tokens: int = 64):
    model, tokenizer, system_prompt = load_model_and_tokenizer()
    model.eval()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )

if __name__ == "__main__":
    generate("What is the square root of 144?")