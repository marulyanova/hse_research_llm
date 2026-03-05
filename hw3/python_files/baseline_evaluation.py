from typing import Tuple
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import set_seed, format_prompt, SYSTEM_PROMPT
from prepare_dataset import get_train_val_data
from verifier import verify_solution


def evaluate_task(model, tokenizer, task: dict, config: dict) -> int:
    """
    Генерирует config['num_samples'] решений и возвращает количество корректных решений
    """

    # Передадим явно название функции из тесткейсов
    required_func_name = task["test_list"][0][7:].split("(")[0]
    prompt_text = (
        task["prompt"] + f"Give the function a name like this: {required_func_name}."
    )
    test_list = task["test_list"]
    test_imports = "\n".join(task["test_imports"])
    messages = format_prompt(prompt_text, SYSTEM_PROMPT)

    correct_count = 0

    for _ in range(config["num_samples"]):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(config["device"])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=config["temperature"],
                top_k=config["top_k"],
                max_new_tokens=config["max_new_tokens"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        if verify_solution(generated, test_list, test_imports):
            correct_count += 1

    return correct_count


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    CONFIG = {
        "num_samples": 10,
        "device": device,
        "temperature": 1.0,
        "top_k": 50,
        "max_new_tokens": 1024,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    print(f"Запуск с параметрами: {CONFIG}")

    model = AutoModelForCausalLM.from_pretrained(CONFIG["model"], device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])

    train_data, val_data = get_train_val_data()

    train_hard_idx, val_hard_idx = [], []
    data_for_curves_train, data_for_curves_val = [], []
    train_length, val_length = len(train_data["prompt"]), len(val_data["prompt"])

    print("Начало baseline train оценки...")
    for i in tqdm(range(train_length)):
        task = {
            "prompt": train_data["prompt"][i],
            "test_list": train_data["test_list"][i],
            "test_imports": train_data["test_imports"][i],
        }
        correct_count = evaluate_task(model, tokenizer, task, CONFIG)
        data_for_curves_train.append(correct_count)

        if correct_count == 0:
            print(
                f"{i}/{train_length} пример подходит, ни одного правильного ответа из {CONFIG['num_samples']}"
            )
            train_hard_idx.append(i)

    print("\n\nНачало baseline val оценки...")
    for i in tqdm(range(val_length)):
        task = {
            "prompt": val_data["prompt"][i],
            "test_list": val_data["test_list"][i],
            "test_imports": val_data["test_imports"][i],
        }
        correct_count = evaluate_task(model, tokenizer, task, CONFIG)
        data_for_curves_val.append(correct_count)

        if correct_count == 0:
            print(
                f"{i}/{val_length} пример подходит, ни одного правильного ответа из {CONFIG['num_samples']}"
            )
            val_hard_idx.append(i)

    print(
        f"\n\nBaseline оценка завершена. Найдено train_hard_idx: {len(train_hard_idx)}, val_hard_idx: {len(val_hard_idx)}"
    )

    data = {"train_hard_idx": train_hard_idx, "val_hard_idx": val_hard_idx}
    with open("hard_idx.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    data = {
        "data_for_curves_train": data_for_curves_train,
        "data_for_curves_val": data_for_curves_val,
    }
    with open("data_for_curves.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
