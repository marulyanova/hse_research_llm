import random
import numpy as np
import torch
import os


# Воспроизводимость экспериментов
def set_seed(seed: int = 33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


set_seed()
