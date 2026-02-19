import gymnasium as gym
import torch
import numpy as np
from policy import PolicyNet

from utils import set_seed

MODEL_PATH = "models/model_bc.pth"
SEED = 33
EPISODES = 10

set_seed(SEED)

env = gym.make("CartPole-v1", render_mode="human")
env.reset(seed=SEED)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = PolicyNet(obs_dim, action_dim=n_actions, hidden_dim=64)
policy.load_state_dict(torch.load(MODEL_PATH))
policy.eval()

for ep in range(EPISODES):
    obs, _ = env.reset(seed=SEED + ep)

    # измененный угол шеста, чтобы показать, что BC не обобщается на новые состояния
    # obs[2] += np.random.uniform(-1000000, 1000000)

    done = False
    total_reward = 0
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        logits = policy(obs_t)

        # добавляем шум к наблюдениям, чтобы показать, что BC не обобщается на новые состояния
        # obs_t_noisy = obs_t + torch.randn_like(obs_t) * 0.1
        # logits = policy(obs_t_noisy)

        action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {ep}: reward = {total_reward}")
env.close()
