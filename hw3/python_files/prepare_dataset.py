from datasets import load_dataset
from utils import set_seed


def get_train_val_data(train_size: int = 100, val_size: int = 30):
    set_seed()

    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    return train_dataset[:train_size], val_dataset[:val_size]
