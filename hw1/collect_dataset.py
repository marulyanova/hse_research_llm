import gymnasium as gym
import torch
from torch.distributions import Categorical
import numpy as np
from policy import PolicyNet
import pandas as pd
from tqdm import tqdm

MODEL_PATH = "models/expert_pg_rloo_0.05.pth"
DATA_PATH = "data/expert_dataset_10.csv"
N_EPISODES = 200
SEED = 33

env = gym.make("CartPole-v1")
env.reset(seed=SEED)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = PolicyNet(obs_dim, action_dim=n_actions, hidden_dim=64)
policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
policy.eval()

dataset = []
rewards = []
for ep in tqdm(range(N_EPISODES)):
    obs, _ = env.reset(seed=SEED + ep)
    done = False
    total_reward = 0
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        logits = policy(obs_t)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = torch.argmax(probs).item()
        dataset.append(np.concatenate([obs, [action]]))

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    rewards.append(total_reward)
    print(f"Episode {ep}: total reward = {total_reward}")
env.close()

columns = ["x", "x_dot", "theta", "theta_dot", "action"]
df = pd.DataFrame(dataset, columns=columns)

df = df.sample(len(df) // 10)
df.to_csv(DATA_PATH, index=False)

print(f"Average reward over {N_EPISODES} episodes: {np.mean(rewards)}")
print(f"Dataset {DATA_PATH}, size: {len(df)} samples")
