import warnings
import argparse
from utils import set_seed

warnings.filterwarnings("ignore")

import gymnasium as gym

import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from collections import deque

from policy import PolicyNet, ValueFunction
from loss import (
    discounted_returns,
    vanilla_policy_gradient_loss,
    policy_gradient_loss_with_baseline,
    policy_gradient_loss_with_rloo,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)

    # "vanilla", "pg_baseline_mean", "pg_baseline_vf", "pg_rloo"
    parser.add_argument("--loss_type", type=str, default="vanilla")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--entropy_reg", action="store_true")
    parser.add_argument("--entropy_reg_coef", type=float, default=0.01)
    parser.add_argument("--save_prefix", type=str, default="default")
    parser.add_argument("--batch_size", type=int, default=8)  # для RLOO
    return parser.parse_args()


def main():

    args = parse_args()
    NUM_EPOCHS = args.n_epochs
    GAMMA = args.gamma
    LEARNING_RATE = args.lr
    HIDDEN_DIM = args.hidden_dim
    LOSS = args.loss_type
    SEED = args.seed
    BATCH_SIZE = args.batch_size

    set_seed(SEED)
    device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"

    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, hidden_dim=HIDDEN_DIM, action_dim=n_actions).to(device)
    if LOSS == "pg_baseline_vf":
        value_function = ValueFunction(obs_dim, hidden_dim=HIDDEN_DIM).to(device)
        vf_optimizer = optim.AdamW(value_function.parameters(), lr=LEARNING_RATE)

    optimizer = optim.AdamW(policy.parameters(), lr=LEARNING_RATE)
    avg_rewards = []
    reward_history = deque(maxlen=100)
    entropy_values = []

    for epoch in tqdm(range(NUM_EPOCHS)):

        returns_batch = []
        log_probs_batch = []
        obs_batch_list = []
        entropy_batch = []
        episode_rewards = []

        for _ in range(BATCH_SIZE):
            obs, _ = env.reset()
            done = False

            ep_obs = []
            ep_log_probs = []
            ep_rewards = []
            ep_entropies = []

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                ep_obs.append(obs_t.unsqueeze(0))

                logits = policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

                ep_entropies.append(dist.entropy())
                ep_log_probs.append(dist.log_prob(action))

                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                ep_rewards.append(reward)

            ep_obs_tensor = torch.cat(ep_obs, dim=0)
            ep_logp_tensor = torch.stack(ep_log_probs)
            ep_returns = torch.tensor(
                discounted_returns(ep_rewards, gamma=GAMMA), dtype=torch.float32
            )
            ep_entropy_tensor = (
                torch.stack(ep_entropies)
                if len(ep_entropies) > 0
                else torch.tensor([0.0])
            )

            obs_batch_list.append(ep_obs_tensor)
            log_probs_batch.append(ep_logp_tensor)
            returns_batch.append(ep_returns)
            entropy_batch.append(ep_entropy_tensor)
            episode_rewards.append(sum(ep_rewards))

        obs_batch = torch.cat(obs_batch_list, dim=0)
        all_log_probs = torch.cat(log_probs_batch, dim=0)
        all_returns = torch.cat(returns_batch, dim=0)

        if LOSS == "vanilla":
            loss = vanilla_policy_gradient_loss(all_returns, all_log_probs)

        elif LOSS == "pg_baseline_mean":
            for r in episode_rewards:
                reward_history.append(r)
            baseline = torch.tensor(np.mean(reward_history), dtype=torch.float32).to(
                device
            )
            loss = policy_gradient_loss_with_baseline(
                all_returns, all_log_probs, baseline
            )

        elif LOSS == "pg_baseline_vf":
            vf_optimizer.zero_grad()
            values = value_function(obs_batch).squeeze(-1)
            vf_loss = nn.MSELoss()(values, all_returns)
            vf_loss.backward()
            vf_optimizer.step()

            advantages = all_returns - values.detach()
            loss = -torch.mean(advantages * all_log_probs)

        elif LOSS == "pg_rloo":
            loss = policy_gradient_loss_with_rloo(returns_batch, log_probs_batch)

        else:
            raise ValueError("Invalid loss type")

        if args.entropy_reg:
            mean_entropy = torch.cat(entropy_batch).mean()
            loss -= args.entropy_reg_coef * mean_entropy

        mean_entropy = (
            torch.cat(entropy_batch).mean().item() if args.entropy_reg else 0.0
        )
        entropy_values.append(mean_entropy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"avgR_batch: {avg_reward:.1f} | "
                f"bestR_overall: {np.max(avg_rewards):.1f} | "
                f"loss: {loss.item():.3f}"
            )

    env.close()

    print("Best reward:", np.max(avg_rewards))

    df = pd.DataFrame(
        {
            "epoch": np.arange(len(avg_rewards)),
            "reward": avg_rewards,
            "entropy": entropy_values,
        }
    )
    df.to_csv(f"logs/{args.save_prefix}.csv", index=False)


if __name__ == "__main__":
    main()
