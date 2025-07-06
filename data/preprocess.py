# === data/preprocess.py ===
from datasets import load_dataset

def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def load_and_prepare_dataset(system_prompt):
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    return dataset
