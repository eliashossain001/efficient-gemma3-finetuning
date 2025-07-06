# === export/gguf_export.py ===
"""Export merged finetuned model to GGUF/llama.cpp format."""

def save_gguf(model, output_dir: str = "gemma-3-finetune", quantization_type: str = "Q8_0"):
    model.save_pretrained_gguf(output_dir, quantization_type=quantization_type)
    print(f"GGUF model saved to {output_dir} (quant={quantization_type})")

def push_gguf(model, repo_id: str, token: str, quantization_type: str = "Q8_0"):
    model.push_to_hub_gguf(
        output_dir=repo_id,
        quantization_type=quantization_type,
        repo_id=repo_id,
        token=token
    )
    print(f"GGUF model pushed to {repo_id} (quant={quantization_type})")