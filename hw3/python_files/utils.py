import random
import numpy as np
import torch
import os


def set_seed(seed: int = 33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


SYSTEM_PROMPT = """You are a helpful assistant that writes Python code.
Given a problem description, implement a function that solves it.
Return only the code. Avoid explanations, markdown formatting and test cases."""


def format_prompt(prompt_text: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Problem: {prompt_text}\n\nImplement the function in Python:",
        },
    ]
    return messages
