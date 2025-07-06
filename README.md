
#  Gemma 3 Fine-Tuning with Unsloth + GRPO

This project demonstrates an efficient and modular pipeline for fine-tuning Google's Gemma 3 (1B) model using [Unsloth](https://github.com/unslothai/unsloth) and GRPO (Generalized Reinforcement Preference Optimization). It includes training, inference, and deployment scripts—ready for integration into production or research workflows.

This repo supports training on custom datasets in GSM8K-style format (`question` + `#### answer`); simply swap out the dataset loader. It uses parameter-efficient fine-tuning (LoRA) via Unsloth, making it lightweight and suitable for limited compute environments.

---

## 💡 Key Features

-  Efficient fine-tuning with LoRA adapters via Unsloth  
-  Supports GGUF export for LLaMA.cpp deployment  
-  Reward-based optimization using regex and ratio matching  
-  Clean VS Code-friendly modular structure  
-  Push-to-HuggingFace automation script  
-  Lightweight inference wrapper with system prompts

---

## 📁 Project Structure

```
gemma_finetune/
├── configs/                # YAML config for training
├── data/                   # Dataset preprocessing logic
├── export/                 # Save & push LoRA/GGUF models
├── inference/              # Run inference from finetuned model
├── model/                  # Model and tokenizer loader
├── rewards/                # Reward scoring functions
├── train_module/           # Training loop using GRPO
├── outputs/                # (Ignored) Training checkpoints
├── main.py                 # Entrypoint for training
├── requirements.txt        # Environment dependencies
└── README.md               # This file
```

---

## 🦜 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Fine-tune the model

```bash
python main.py
```

---

### 3. Inference

```bash
python inference/infer.py
```

---

### 4. Push to Hugging Face Hub

```bash
python -m export.push_to_hub --repo_id <your_username/repo_name> --token <hf_token> --type lora
# or for GGUF
python -m export.push_to_hub --repo_id <your_username/repo_name> --token <hf_token> --type gguf --quant Q8_0
```

---

## 📆 Model Details

- **Base**: `unsloth/gemma-3-1b-it`
- **Sequence Length**: 1024 tokens
- **LoRA Config**: `r=8`, `alpha=8`, `dropout=0`
- **Optimizer**: `AdamW (torch_fused)`
- **Reward Functions**:
  - Exact match using regex format
  - Approximate match using token patterns
  - Numeric and ratio-based answer correctness

---

## ⛔️ .gitignore Note

To keep the repo lightweight, large files like checkpoints and compiled caches are excluded:

```bash
outputs/
unsloth_compiled_cache/
__pycache__/
*.bin
*.pt
```

---

## 🧠 Credit

Developed using:
- [Unsloth](https://github.com/unslothai/unsloth)
- [Transformers](https://github.com/huggingface/transformers)
- [TRL](https://github.com/huggingface/trl)
- [Gemma by Google](https://ai.google.dev/gemma)

---

## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD in Progress | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
