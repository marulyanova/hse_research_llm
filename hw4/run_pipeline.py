import json
import torch
from llm_agent import LLMAgent
from episode_runner import evaluate_model_batched
from utils import set_seed
import time
import random


def main():
    """Загрузка модели, оценка на Train, Val"""

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed()
    agent = LLMAgent("Qwen/Qwen2.5-0.5B-Instruct", device, logging_flag=True)

    print("device", agent.device)

    with open("train.json") as f:
        train_dataset = json.load(f)

    with open("val.json") as f:
        val_dataset = json.load(f)

    train_dataset = random.sample(train_dataset, 250)
    val_dataset = random.sample(val_dataset, 50)

    # metrics_train = evaluate_model_batched(agent, train_dataset)
    metrics_val = evaluate_model_batched(agent, val_dataset)

    # print(metrics_train)
    print(metrics_val)
    print(f"Execution time: {time.time() - start}")


if __name__ == "__main__":
    main()
