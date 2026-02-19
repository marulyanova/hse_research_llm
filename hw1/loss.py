import torch

device = "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"


def discounted_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns


def vanilla_policy_gradient_loss(rewards, log_probs):
    rewards = rewards.to(device)
    log_probs = log_probs.to(device)
    loss = -torch.mean(rewards * log_probs)
    return loss


def policy_gradient_loss_with_baseline(rewards, log_probs, baseline):
    rewards = rewards.to(device)
    log_probs = log_probs.to(device)
    baseline = baseline.to(device)
    advantages = rewards - baseline
    loss = -torch.mean(advantages * log_probs)
    return loss


def policy_gradient_loss_with_rloo(returns_batch, log_probs_batch):

    returns_batch = [ret.to(device) for ret in returns_batch]
    log_probs_batch = [lp.to(device) for lp in log_probs_batch]

    N = len(returns_batch)
    if N < 2:
        total_loss = sum(
            vanilla_policy_gradient_loss(ret, lp)
            for ret, lp in zip(returns_batch, log_probs_batch)
        )
        return total_loss / N

    losses = []

    episode_means = [ret.mean().detach() for ret in returns_batch]

    for i in range(N):
        G_i = returns_batch[i]
        logp_i = log_probs_batch[i]

        other_means = [episode_means[j] for j in range(N) if j != i]
        baseline_i = torch.stack(other_means).mean()

        advantages = G_i - baseline_i
        loss_i = -torch.mean(advantages * logp_i)
        losses.append(loss_i)

    return torch.stack(losses).mean()
