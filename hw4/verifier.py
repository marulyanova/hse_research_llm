from typing import List, Dict, Any, Optional
from data import Data


class TrajectoryVerifier:

    def verify_trajectory(self, env, data, actions, max_steps=None):
        """Прогоняет 1 траекторию по заданным данным, выполняя actions, возвращает метрики"""

        obs = env.reset(data)

        total_reward = 0
        steps = 0
        tool_calls = 0
        policy_violations = 0
        invalid_actions = 0
        info_trace = []

        for action in actions:

            if max_steps and steps >= max_steps:
                break

            if action.startswith("TOOL_CALL"):
                tool_calls += 1

            obs, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1

            if info.get("policy_violation"):
                policy_violations += 1

            if info.get("invalid_action"):
                invalid_actions += 1

            info_trace.append(info)

            if done:
                break

        success = info_trace[-1].get("success", False) if info_trace else False

        return {
            "success": success,
            "total_reward": total_reward,
            "steps": steps,
            "tool_calls": tool_calls,
            "policy_violations": policy_violations,
            "invalid_actions": invalid_actions,
            "info_trace": info_trace,
        }
