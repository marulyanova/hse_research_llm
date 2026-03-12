from verifier import TrajectoryVerifier
from appointment_env import BookingEnv
from tqdm import tqdm


def format_history(history):
    """Форматирование истории для Agent"""

    text = ""
    for role, content in history:
        if role == "OBS":
            text += f"\nObservation: {content}\n"
        elif role == "ACTION":
            text += f"\nAgent: {content}\n"

    return text


def run_episode(env, agent, data, max_steps=10):
    """Прогоняет 1 эпизод"""

    obs = env.reset(data)
    actions = []
    history = []
    history.append(("OBS", obs))
    done = False

    while not done and len(actions) < max_steps:
        prompt = format_history(history)
        action = agent.act(prompt)
        actions.append(action)
        history.append(("ACTION", action))
        obs, reward, done, info = env.step(action)
        history.append(("OBS", obs))

    return actions


def evaluate_model(agent, env, dataset):
    """Получает actions с прогонки episode, проверяет траекторию, отдает метрики"""

    verifier = TrajectoryVerifier()
    metrics = []

    for _, data in tqdm(enumerate(dataset)):
        actions = run_episode(env, agent, data)
        result = verifier.verify_trajectory(BookingEnv(), data, actions)
        metrics.append(result)

    success_rate = sum(m["success"] for m in metrics) / len(metrics)
    avg_reward = sum(m["total_reward"] for m in metrics) / len(metrics)

    print("Success rate:", success_rate)
    print("Avg reward:", avg_reward)

    return metrics
