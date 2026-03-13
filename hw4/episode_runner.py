from verifier import TrajectoryVerifier
from appointment_env import BookingEnv
from tqdm import tqdm
import torch


def format_history(history):
    """Форматирование истории для Agent"""

    text = ""
    for role, content in history:
        if role == "OBS":
            text += f"\nObservation: {content}\n"
        elif role == "ACTION":
            text += f"\nAgent: {content}\n"

    return text


# def run_episode(env, agent, data, max_steps=10):
#     """Прогоняет 1 эпизод"""

#     obs = env.reset(data)
#     actions = []
#     history = []
#     history.append(("OBS", obs))
#     done = False

#     while not done and len(actions) < max_steps:
#         prompt = format_history(history)
#         action = agent.act(prompt)
#         actions.append(action)
#         history.append(("ACTION", action))
#         obs, reward, done, info = env.step(action)
#         history.append(("OBS", obs))

#     return actions


# def evaluate_model(agent, env, dataset):
#     """Получает actions с прогонки episode, проверяет траекторию, отдает метрики"""

#     verifier = TrajectoryVerifier()
#     metrics = []

#     for _, data in tqdm(enumerate(dataset)):
#         actions = run_episode(env, agent, data)
#         result = verifier.verify_trajectory(BookingEnv(), data, actions)
#         metrics.append(result)

#     success_rate = sum(m["success"] for m in metrics) / len(metrics)
#     avg_reward = sum(m["total_reward"] for m in metrics) / len(metrics)

#     print("Success rate:", success_rate)
#     print("Avg reward:", avg_reward)

#     return metrics


def evaluate_model_batched(agent, dataset, batch_size=32):
    """
    Запускает оценку параллельно для ускорения
    batch_size - сколько эпизодов прогонять одновременно
    """

    verifier = TrajectoryVerifier()
    metrics = []

    for i in range(0, len(dataset), batch_size):
        chunk_data = dataset[i : min(i + batch_size, len(dataset))]
        chunk_metrics = run_parallel_episodes(agent, chunk_data, verifier)
        metrics.extend(chunk_metrics)

    success_rate = sum(m["success"] for m in metrics) / len(metrics)
    avg_reward = sum(m["total_reward"] for m in metrics) / len(metrics)

    print("Success rate:", success_rate)
    print("Avg reward:", avg_reward)
    return metrics


def run_parallel_episodes(agent, data_samples, verifier):
    """
    Прогоняет группу эпизодов параллельно, синхронизируя шаги.
    """

    episodes = []
    for data in data_samples:
        env = BookingEnv()
        obs = env.reset(data)
        episodes.append(
            {
                "env": env,
                "data": data,
                "history": [("OBS", obs)],
                "actions": [],
                "done": False,
            }
        )

    max_steps = 10

    for step in range(max_steps):
        active_indices = [i for i, ep in enumerate(episodes) if not ep["done"]]
        if not active_indices:
            break

        prompts_to_process = []
        for idx in active_indices:
            history_text = format_history(episodes[idx]["history"])
            prompts_to_process.append(history_text)

        actions_batch = agent.act_batch(prompts_to_process)

        for j, idx in enumerate(active_indices):
            action = actions_batch[j]
            episodes[idx]["actions"].append(action)
            episodes[idx]["history"].append(("ACTION", action))

            obs, reward, done, info = episodes[idx]["env"].step(action)
            episodes[idx]["history"].append(("OBS", obs))
            episodes[idx]["done"] = done

    results = []
    for ep in episodes:
        result = verifier.verify_trajectory(BookingEnv(), ep["data"], ep["actions"])
        results.append(result)

    return results
