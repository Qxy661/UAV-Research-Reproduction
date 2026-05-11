"""SAC 训练脚本 — 四旋翼悬停（CPU 友好）"""

import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from envs.hover_env import HoverEnv, HoverEnvWind
from algorithms.sac import SAC


def train_sac(n_episodes=500, wind_enabled=False, wind_speed=3.0, seed=42, save_dir="results/sac_training"):
    os.makedirs(save_dir, exist_ok=True)
    env = HoverEnvWind(wind_speed=wind_speed) if wind_enabled else HoverEnv(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = SAC(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=128, lr=3e-4, gamma=0.99,
                tau=0.005, buffer_size=50000, batch_size=128)

    episode_rewards, pos_errors, training_log = [], [], []
    total_steps = 0
    warmup_steps = 1000
    start_time = time.time()

    for episode in range(n_episodes):
        obs = env.reset(seed=seed + episode)
        episode_reward = 0

        for step in range(500):
            total_steps += 1
            if total_steps < warmup_steps:
                action = np.random.uniform(-1, 1, size=env.act_dim)
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.buffer.add(obs, action, reward, next_obs, float(done))

            if total_steps >= warmup_steps:
                agent.update()

            episode_reward += reward
            obs = next_obs
            if done:
                break

        episode_rewards.append(episode_reward)
        pos_errors.append(info.get("pos_error", 0))

        if (episode + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            avg_e = np.mean(pos_errors[-50:])
            elapsed = time.time() - start_time
            print(f"SAC Ep {episode+1:4d}/{n_episodes} | Steps: {total_steps:6d} | Reward: {avg_r:8.2f} | PosErr: {avg_e:.4f}m | Alpha: {agent.alpha:.4f} | {elapsed:.1f}s")
            training_log.append({"episode": episode+1, "total_steps": total_steps, "avg_reward": float(avg_r), "avg_pos_error": float(avg_e), "alpha": agent.alpha, "elapsed": elapsed})

    agent.save(os.path.join(save_dir, "sac_model.pt"))
    with open(os.path.join(save_dir, "episode_rewards.json"), "w") as f:
        json.dump([float(r) for r in episode_rewards], f)
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"SAC 完成: 最终奖励={np.mean(episode_rewards[-50:]):.2f}, 误差={np.mean(pos_errors[-50:]):.4f}m, Alpha={agent.alpha:.4f}, 耗时={time.time()-start_time:.1f}s")
    return episode_rewards


if __name__ == "__main__":
    train_sac()
