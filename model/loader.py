# === model/loader.py ===
from unsloth import FastModel

def load_model_and_tokenizer():
    max_seq_length = 1024
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-1b-it",
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = f"""You are given a problem.\nThink about the problem and provide your working out.\nPlace it between {reasoning_start} and {reasoning_end}.\nThen, provide your solution between {solution_start}{solution_end}"""

    return model, tokenizer, system_prompt