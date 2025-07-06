from model.loader import load_model_and_tokenizer
from data.preprocess import load_and_prepare_dataset
from rewards.scoring import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)
from train_module.train import train_model


def main():
    # Load model & tokenizer
    model, tokenizer, system_prompt = load_model_and_tokenizer()

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(system_prompt)

    # Train the model
    train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
    )


if __name__ == "__main__":
    main()