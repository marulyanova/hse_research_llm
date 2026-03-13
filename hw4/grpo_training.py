import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from appointment_env import BookingEnv
from verifier import TrajectoryVerifier
from episode_runner import format_history
from utils import set_seed

set_seed()

CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_seq_length": 2048,
    "max_new_tokens": 128,
    "num_generations": 4,
    "temperature": 0.8,
    "top_k": 40,
    "lr": 2e-5,
    "batch_size": 4,
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


class RLAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def act_batch(self, observations):
        inputs = self.tokenizer(
            observations,
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
        for i, prompt in enumerate(observations):
            input_len = inputs["attention_mask"][i].sum().item()
            generated_tokens = outputs[i][input_len:]
            text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            responses.append(text)
        return responses


def prepare_grpo_dataset(data):
    rows = []
    for row in data:
        first_msg = row["user_messages"][0]
        prompt = f"""You are a helpful assistant for booking sport classes.

User: {first_msg}

Respond with either text or:

TOOL_CALL {{"name": "...", "args": {{...}}}}
"""
        rows.append({"prompt": prompt, "scenario": row})
    return Dataset.from_list(rows)


def make_reward_function(model, tokenizer, verifier, dataset):
    agent = RLAgent(model, tokenizer)

    def env_reward(prompts, completions, **kwargs):
        rewards = []
        for completion, row in zip(completions, dataset):
            scenario = row["scenario"]
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
    train_dataset = prepare_grpo_dataset(train_data)
    reward_fn = make_reward_function(model, tokenizer, verifier, train_dataset)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
    )

    trainer.train()
