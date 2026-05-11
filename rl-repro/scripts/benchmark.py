"""统一基准评估 — PID vs SAC/PPO 在同一 hover_env 中对比

计算统一指标：位置 RMSE、姿态 RMSE、平均奖励、成功率
支持风扰测试
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.hover_env import HoverEnv, HoverEnvWind
from algorithms.pid_controller import PIDController


def load_sac_model(model_path, obs_dim=8, act_dim=4, hidden_dim=256):
    """加载 SAC 模型"""
    import torch
    from algorithms.sac import SAC
    agent = SAC(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    agent.load(model_path)
    agent.actor.eval()
    return agent


def run_pid_episode(env, n_episodes=100):
    """PID 控制器评估"""
    all_rewards = []
    all_z_errors = []
    all_att_errors = []
    all_trajectories = []

    for ep in range(n_episodes):
        obs = env.reset(seed=42 + ep)
        pid = PIDController(m=env.m, g=env.g, dt=env.dt)
        ep_reward = 0
        ep_z_errors = []
        ep_att_errors = []
        ep_z轨迹 = []

        for step in range(env.max_steps):
            action = pid.step(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_z_errors.append(info["z_err"])
            ep_att_errors.append(info["att_err"])
            ep_z轨迹.append(env.state[0])  # z position

            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        all_z_errors.append(np.mean(ep_z_errors))
        all_att_errors.append(np.mean(ep_att_errors))
        all_trajectories.append(ep_z轨迹)

    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_z_error": np.mean(all_z_errors),
        "mean_att_error": np.mean(all_att_errors),
        "success_rate": np.mean([r > 500 for r in all_rewards]),
        "trajectories": all_trajectories,
    }


def run_sac_episode(env, model_path, n_episodes=100):
    """SAC 模型评估"""
    import torch
    agent = load_sac_model(model_path)

    all_rewards = []
    all_z_errors = []
    all_att_errors = []
    all_trajectories = []

    for ep in range(n_episodes):
        obs = env.reset(seed=42 + ep)
        ep_reward = 0
        ep_z_errors = []
        ep_att_errors = []
        ep_z轨迹 = []

        for step in range(env.max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.actor.sample(obs_t)
            action = action.squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_z_errors.append(info["z_err"])
            ep_att_errors.append(info["att_err"])
            ep_z轨迹.append(env.state[0])

            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        all_z_errors.append(np.mean(ep_z_errors))
        all_att_errors.append(np.mean(ep_att_errors))
        all_trajectories.append(ep_z轨迹)

    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_z_error": np.mean(all_z_errors),
        "mean_att_error": np.mean(all_att_errors),
        "success_rate": np.mean([r > 500 for r in all_rewards]),
        "trajectories": all_trajectories,
    }


def run_wind_test(model_path=None, wind_speeds=[1.0, 3.0, 5.0], n_episodes=50):
    """风扰鲁棒性测试"""
    results = {"PID": {}, "SAC": {}}

    for ws in wind_speeds:
        env = HoverEnvWind(wind_speed=ws)

        pid_res = run_pid_episode(env, n_episodes=n_episodes)
        results["PID"][f"wind_{ws}"] = {
            "mean_reward": pid_res["mean_reward"],
            "mean_z_error": pid_res["mean_z_error"],
        }

        if model_path and os.path.exists(model_path):
            try:
                sac_res = run_sac_episode(env, model_path, n_episodes=n_episodes)
                results["SAC"][f"wind_{ws}"] = {
                    "mean_reward": sac_res["mean_reward"],
                    "mean_z_error": sac_res["mean_z_error"],
                }
            except Exception:
                pass  # SAC model incompatible, skip

    return results


def print_table(headers, rows):
    """打印格式化表格"""
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def format_row(row):
        return "|" + "|".join(f" {str(row[i]):^{col_widths[i]-2}} " for i in range(len(row))) + "|"

    print(sep)
    print(format_row(headers))
    print(sep)
    for row in rows:
        print(format_row(row))
    print(sep)


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 优先使用新训练的 6DOF 模型
    sac_path = os.path.join(base, "results", "sac_training_6dof", "sac_model.pt")
    if not os.path.exists(sac_path):
        sac_path = os.path.join(base, "results", "sac_training", "sac_model.pt")

    print("=" * 60)
    print("统一基准评估 — PID vs SAC (hover_env)")
    print("=" * 60)

    # === 无风测试 ===
    print("\n[1] 无风悬停测试 (100 episodes)")
    env = HoverEnv(seed=42)

    pid_res = run_pid_episode(env, n_episodes=100)
    print(f"  PID: reward={pid_res['mean_reward']:.1f}±{pid_res['std_reward']:.1f}, "
          f"z_err={pid_res['mean_z_error']:.4f}m, att_err={pid_res['mean_att_error']:.4f}, "
          f"success={pid_res['success_rate']*100:.0f}%")

    sac_res = None
    if os.path.exists(sac_path):
        try:
            sac_res = run_sac_episode(env, sac_path, n_episodes=100)
            print(f"  SAC: reward={sac_res['mean_reward']:.1f}±{sac_res['std_reward']:.1f}, "
                  f"z_err={sac_res['mean_z_error']:.4f}m, att_err={sac_res['mean_att_error']:.4f}, "
                  f"success={sac_res['success_rate']*100:.0f}%")
        except Exception as e:
            print(f"  SAC: 加载失败 ({e})，跳过")
    else:
        print("  SAC: 模型文件不存在，跳过")

    # 对比表
    print("\n[2] 对比统计")
    headers = ["指标", "PID", "SAC", "优胜"]
    rows = []
    if sac_res:
        for metric, label in [("mean_reward", "平均奖励"), ("mean_z_error", "位置误差(m)"),
                               ("mean_att_error", "姿态误差"), ("success_rate", "成功率")]:
            pid_val = pid_res[metric]
            sac_val = sac_res[metric]
            if metric in ["mean_z_error", "mean_att_error"]:
                winner = "PID" if pid_val < sac_val else "SAC"
            else:
                winner = "PID" if pid_val > sac_val else "SAC"
            rows.append([label, f"{pid_val:.4f}", f"{sac_val:.4f}", winner])
        print_table(headers, rows)

    # === 风扰测试 ===
    print("\n[3] 风扰鲁棒性测试")
    wind_speeds = [1.0, 3.0, 5.0]
    wind_results = run_wind_test(sac_path if os.path.exists(sac_path) else None,
                                  wind_speeds=wind_speeds, n_episodes=50)

    headers = ["风速(m/s)", "PID奖励", "PID误差", "SAC奖励", "SAC误差"]
    rows = []
    for i, ws in enumerate(wind_speeds):
        pid_r = wind_results["PID"][f"wind_{ws}"]
        sac_r = wind_results["SAC"].get(f"wind_{ws}", {"mean_reward": "N/A", "mean_z_error": "N/A"})
        rows.append([
            f"{ws:.0f}",
            f"{pid_r['mean_reward']:.1f}",
            f"{pid_r['mean_z_error']:.4f}",
            f"{sac_r['mean_reward']:.1f}" if isinstance(sac_r['mean_reward'], float) else "N/A",
            f"{sac_r['mean_z_error']:.4f}" if isinstance(sac_r['mean_z_error'], float) else "N/A",
        ])
    print_table(headers, rows)

    # 保存结果
    save_dir = os.path.join(base, "results")
    save_data = {
        "no_wind": {"PID": pid_res},
        "wind": wind_results,
    }
    if sac_res:
        save_data["no_wind"]["SAC"] = sac_res

    # 移除不可 JSON 序列化的 trajectories
    for method in save_data["no_wind"]:
        save_data["no_wind"][method].pop("trajectories", None)

    with open(os.path.join(save_dir, "benchmark_results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n结果已保存到 {os.path.join(save_dir, 'benchmark_results.json')}")


if __name__ == "__main__":
    main()
