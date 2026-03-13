import torch
import json
import random
from datasets import Dataset

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

from utils import set_seed
from appointment_env import BookingEnv
from episode_runner import run_parallel_episodes, format_history
from verifier import TrajectoryVerifier

set_seed()

CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_seq_length": 256,
    "max_new_tokens": 128,
    "num_generations": 8,
    "temperature": 0.8,
    "top_k": 40,
    "lr": 2e-5,
    "batch_size": 32,
    "epochs": 3,
}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model"],
    max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=16,
    gpu_memory_utilization=0.85,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=33,
)

grpo_config = GRPOConfig(
    output_dir="./grpo_booking",
    num_generations=CONFIG["num_generations"],
    max_completion_length=CONFIG["max_new_tokens"],
    temperature=CONFIG["temperature"],
    top_k=CONFIG["top_k"],
    learning_rate=CONFIG["lr"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=2,
    num_train_epochs=CONFIG["epochs"],
    beta=0.1,
    epsilon=0.2,
    logging_steps=5,
    save_steps=50,
    bf16=False,
    fp16=True,
    remove_unused_columns=False,
    log_completions=True,
)

verifier = TrajectoryVerifier()


# def env_reward(prompts, completions, dataset_row, **kwargs):
#     scenarios = []
#     for row in dataset_row:
#         scenarios.append(row)
#     results = run_parallel_episodes(agent, scenarios, verifier)
#     rewards = [r["total_reward"] for r in results]
#     return rewards


class RLAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def act_batch(self, observations):
        prompts = observations
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=True,
                temperature=CONFIG["temperature"],
                top_k=CONFIG["top_k"],
                pad_token_id=self.tokenizer.pad_token_id,
            )

        responses = []
        for i in range(len(prompts)):
            input_len = inputs["attention_mask"][i].sum().item()
            generated_tokens = outputs[i][input_len:]
            text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            responses.append(text)

        return responses


def make_reward_function(train_data):
    def env_reward(prompts, completions, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs["processing_class"]
        agent = RLAgent(model, tokenizer)
        scenarios = kwargs["dataset"]

        rewards = []
        for completion, scenario in zip(completions, scenarios):
            env = BookingEnv()
            obs = env.reset(scenario)
            actions = [completion]
            obs, reward, done, info = env.step(completion)
            history = [("OBS", obs), ("ACTION", completion)]
            step = 0
            while not done and step < 8:
                prompt = format_history(history)
                action = agent.act_batch([prompt])[0]
                actions.append(action)
                obs, r, done, info = env.step(action)
                reward += r
                history.append(("ACTION", action))
                history.append(("OBS", obs))
                step += 1
            result = verifier.verify_trajectory(BookingEnv(), scenario, actions)
            rewards.append(result["total_reward"])
        return rewards

    return env_reward


def grpo_train_loop(train_data):
    train_dataset = Dataset.from_list(train_data)
    reward_fn = make_reward_function(train_data)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
    )

    trainer.train()


def main():
    with open("train.json") as f:
        train_dataset = json.load(f)

    grpo_train_loop(train_dataset)


if __name__ == "__main__":
    main()
