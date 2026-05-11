"""PPO 训练脚本 — 四旋翼悬停（CPU 友好）"""

import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from envs.hover_env import HoverEnv, HoverEnvWind
from algorithms.ppo import PPO


def train_ppo(n_episodes=500, wind_enabled=False, wind_speed=3.0, seed=42, save_dir="results/ppo_training"):
    os.makedirs(save_dir, exist_ok=True)
    env = HoverEnvWind(wind_speed=wind_speed) if wind_enabled else HoverEnv(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = PPO(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=64, lr=3e-4, gamma=0.99, lam=0.95,
                clip_epsilon=0.2, n_epochs=10, batch_size=64)

    episode_rewards, pos_errors, training_log = [], [], []
    start_time = time.time()

    for episode in range(n_episodes):
        obs = env.reset(seed=seed + episode)
        episode_reward = 0
        rollout = {"obs": [], "actions": [], "rewards": [], "log_probs": [], "values": [], "dones": []}

        for step in range(500):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, value = agent.network.get_action(obs_t)
            action_np = action.numpy().flatten()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            rollout["obs"].append(obs)
            rollout["actions"].append(action_np)
            rollout["rewards"].append(reward)
            rollout["log_probs"].append(log_prob.item())
            rollout["values"].append(value.item())
            rollout["dones"].append(float(done))

            episode_reward += reward
            obs = next_obs
            if done:
                break

        advantages, returns = agent.compute_gae(
            np.array(rollout["rewards"]), np.array(rollout["values"]), np.array(rollout["dones"]))

        agent.update({
            "obs": np.array(rollout["obs"]), "actions": np.array(rollout["actions"]),
            "log_probs": np.array(rollout["log_probs"]), "returns": returns, "advantages": advantages,
        })

        episode_rewards.append(episode_reward)
        pos_errors.append(info.get("pos_error", 0))

        if (episode + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            avg_e = np.mean(pos_errors[-50:])
            elapsed = time.time() - start_time
            print(f"PPO Ep {episode+1:4d}/{n_episodes} | Reward: {avg_r:8.2f} | PosErr: {avg_e:.4f}m | {elapsed:.1f}s")
            training_log.append({"episode": episode+1, "avg_reward": float(avg_r), "avg_pos_error": float(avg_e), "elapsed": elapsed})

    agent.save(os.path.join(save_dir, "ppo_model.pt"))
    with open(os.path.join(save_dir, "episode_rewards.json"), "w") as f:
        json.dump([float(r) for r in episode_rewards], f)
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"PPO 完成: 最终奖励={np.mean(episode_rewards[-50:]):.2f}, 误差={np.mean(pos_errors[-50:]):.4f}m, 耗时={time.time()-start_time:.1f}s")
    return episode_rewards


if __name__ == "__main__":
    train_ppo()
