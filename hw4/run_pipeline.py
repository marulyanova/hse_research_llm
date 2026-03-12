import json
import torch
from appointment_env import BookingEnv
from llm_agent import LLMAgent
from episode_runner import evaluate_model
from utils import set_seed


def main():
    """Загрузка модели, оценка на Train, Val"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed()
    env = BookingEnv()
    agent = LLMAgent("Qwen/Qwen2.5-0.5B-Instruct", device)

    print(agent.device)

    with open("train.json") as f:
        train_dataset = json.load(f)

    with open("val.json") as f:
        val_dataset = json.load(f)

    train_dataset = train_dataset[:1]
    val_dataset = val_dataset[:1]

    metrics_train = evaluate_model(agent, env, train_dataset)
    metrics_val = evaluate_model(agent, env, val_dataset)

    print(metrics_train)
    print(metrics_val)


if __name__ == "__main__":
    main()
