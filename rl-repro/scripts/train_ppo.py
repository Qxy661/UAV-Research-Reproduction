"""
PPO 训练脚本 — 四旋翼悬停任务
CPU 友好：小网络 (64-64)、少 episode (500)
"""

import sys
import os
import json
import time
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.hover_env import HoverEnv, HoverEnvWind
from algorithms.ppo import PPO


def train_ppo(
    n_episodes=500,
    max_steps_per_episode=500,
    hidden_dim=64,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_epsilon=0.2,
    n_epochs=10,
    batch_size=64,
    wind_enabled=False,
    wind_speed=3.0,
    save_dir="results/ppo_training",
    seed=42,
):
    """训练 PPO 智能体"""

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建环境
    if wind_enabled:
        env = HoverEnvWind(wind_speed=wind_speed)
        env_name = f"hover_wind_{wind_speed}ms"
    else:
        env = HoverEnv()
        env_name = "hover_no_wind"

    # 设置随机种子
    np.random.seed(seed)
    env.reset(seed=seed)

    # 创建 PPO 智能体
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        lam=lam,
        clip_epsilon=clip_epsilon,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # 训练记录
    episode_rewards = []
    episode_lengths = []
    pos_errors = []
    training_log = []

    print(f"开始 PPO 训练: {env_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Network: {hidden_dim}-{hidden_dim}")
    print(f"  Device: CPU")
    print()

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        # 收集轨迹
        rollout = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

        for step in range(max_steps_per_episode):
            # 选择动作
            obs_tensor = np.expand_dims(obs, axis=0)
            action, log_prob, value = agent.network.get_action(
                __import__("torch").FloatTensor(obs_tensor)
            )

            action_np = action.numpy().flatten()
            log_prob_np = log_prob.numpy().item()
            value_np = value.numpy().item()

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # 存储
            rollout["obs"].append(obs)
            rollout["actions"].append(action_np)
            rollout["rewards"].append(reward)
            rollout["log_probs"].append(log_prob_np)
            rollout["values"].append(value_np)
            rollout["dones"].append(float(done))

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                break

        # 计算 GAE
        advantages, returns = agent.compute_gae(
            np.array(rollout["rewards"]),
            np.array(rollout["values"]),
            np.array(rollout["dones"]),
        )

        # 更新策略
        update_rollout = {
            "obs": np.array(rollout["obs"]),
            "actions": np.array(rollout["actions"]),
            "log_probs": np.array(rollout["log_probs"]),
            "returns": returns,
            "advantages": advantages,
        }

        loss = agent.update(update_rollout)

        # 记录
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        pos_errors.append(info.get("pos_error", 0))

        # 日志
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_error = np.mean(pos_errors[-10:])
            elapsed = time.time() - start_time

            log_msg = (
                f"Episode {episode+1:4d}/{n_episodes} | "
                f"Reward: {avg_reward:8.2f} | "
                f"Length: {avg_length:6.1f} | "
                f"Pos Error: {avg_error:6.3f}m | "
                f"Time: {elapsed:6.1f}s"
            )
            print(log_msg)

            training_log.append({
                "episode": episode + 1,
                "avg_reward": float(avg_reward),
                "avg_length": float(avg_length),
                "avg_pos_error": float(avg_error),
                "elapsed": elapsed,
            })

    # 保存结果
    agent.save(os.path.join(save_dir, "ppo_model.pt"))

    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    with open(os.path.join(save_dir, "episode_rewards.json"), "w") as f:
        json.dump([float(r) for r in episode_rewards], f)

    total_time = time.time() - start_time
    print(f"\nPPO 训练完成!")
    print(f"  总时间: {total_time:.1f}s")
    print(f"  最终奖励 (最后10ep): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  最终位置误差: {np.mean(pos_errors[-10:]):.3f}m")
    print(f"  结果保存至: {save_dir}")

    return episode_rewards, training_log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO 训练四旋翼悬停")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--wind", action="store_true", help="启用风扰")
    parser.add_argument("--wind-speed", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppo(
        n_episodes=args.episodes,
        wind_enabled=args.wind,
        wind_speed=args.wind_speed,
        seed=args.seed,
    )
