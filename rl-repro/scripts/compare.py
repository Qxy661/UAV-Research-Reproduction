"""
PPO vs SAC 对比分析脚本
生成对比图表和统计报告
"""

import json
import os
import numpy as np


def load_training_log(log_path):
    """加载训练日志"""
    with open(log_path, "r") as f:
        return json.load(f)


def compute_metrics(rewards, window=10):
    """计算训练指标"""
    rewards = np.array(rewards)

    # 收敛速度（达到阈值的 episode）
    threshold = 0.8 * np.max(rewards[-100:])  # 最后 100ep 最大值的 80%
    converged_ep = np.where(rewards > threshold)[0]
    convergence_speed = converged_ep[0] if len(converged_ep) > 0 else len(rewards)

    # 最终性能（最后 100ep 平均）
    final_performance = np.mean(rewards[-100:])

    # 训练稳定性（最后 100ep 标准差）
    training_stability = np.std(rewards[-100:])

    # 样本效率（达到 80% 最终性能所需 episode）
    target = 0.8 * final_performance
    reached = np.where(rewards > target)[0]
    sample_efficiency = reached[0] if len(reached) > 0 else len(rewards)

    return {
        "convergence_speed": int(convergence_speed),
        "final_performance": float(final_performance),
        "training_stability": float(training_stability),
        "sample_efficiency": int(sample_efficiency),
    }


def compare_algorithms(ppo_dir, sac_dir, output_dir="results/comparison"):
    """对比 PPO 和 SAC"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    ppo_rewards = json.load(open(os.path.join(ppo_dir, "episode_rewards.json")))
    sac_rewards = json.load(open(os.path.join(sac_dir, "episode_rewards.json")))

    ppo_log = load_training_log(os.path.join(ppo_dir, "training_log.json"))
    sac_log = load_training_log(os.path.join(sac_dir, "training_log.json"))

    # 计算指标
    ppo_metrics = compute_metrics(ppo_rewards)
    sac_metrics = compute_metrics(sac_rewards)

    # 生成对比报告
    report = {
        "PPO": ppo_metrics,
        "SAC": sac_metrics,
        "comparison": {
            "convergence_winner": "SAC" if sac_metrics["convergence_speed"] < ppo_metrics["convergence_speed"] else "PPO",
            "performance_winner": "SAC" if sac_metrics["final_performance"] > ppo_metrics["final_performance"] else "PPO",
            "stability_winner": "SAC" if sac_metrics["training_stability"] < ppo_metrics["training_stability"] else "PPO",
            "efficiency_winner": "SAC" if sac_metrics["sample_efficiency"] < ppo_metrics["sample_efficiency"] else "PPO",
        }
    }

    # 保存报告
    with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 打印报告
    print("=" * 60)
    print("PPO vs SAC 对比分析报告")
    print("=" * 60)

    print(f"\n{'指标':<20} {'PPO':<15} {'SAC':<15} {'优胜':<10}")
    print("-" * 60)

    metrics_names = {
        "convergence_speed": "收敛速度 (ep)",
        "final_performance": "最终性能",
        "training_stability": "训练稳定性 (std)",
        "sample_efficiency": "样本效率 (ep)",
    }

    for key, name in metrics_names.items():
        ppo_val = ppo_metrics[key]
        sac_val = sac_metrics[key]
        winner = report["comparison"][f"{key.split('_')[0]}_winner"] if f"{key.split('_')[0]}_winner" in report["comparison"] else ""

        if key == "training_stability":
            winner = "SAC" if sac_val < ppo_val else "PPO"
        elif key in ["convergence_speed", "sample_efficiency"]:
            winner = "SAC" if sac_val < ppo_val else "PPO"
        else:
            winner = "SAC" if sac_val > ppo_val else "PPO"

        print(f"{name:<20} {ppo_val:<15.3f} {sac_val:<15.3f} {winner:<10}")

    print("\n" + "=" * 60)
    print(f"综合优胜: {sum(1 for v in report['comparison'].values() if v == 'SAC') > 2 and 'SAC' or 'PPO'}")
    print("=" * 60)

    # 生成 ASCII 图表
    print("\n训练曲线对比 (ASCII):")
    print("-" * 60)

    # 简化绘图
    n_points = 20
    ppo_smooth = np.convolve(ppo_rewards, np.ones(25)/25, mode='valid')
    sac_smooth = np.convolve(sac_rewards, np.ones(25)/25, mode='valid')

    ppo_sampled = ppo_smooth[::len(ppo_smooth)//n_points][:n_points]
    sac_sampled = sac_smooth[::len(sac_smooth)//n_points][:n_points]

    max_val = max(max(ppo_sampled), max(sac_sampled))
    min_val = min(min(ppo_sampled), min(sac_sampled))
    height = 15

    for row in range(height, -1, -1):
        val = min_val + (max_val - min_val) * row / height
        line = f"{val:8.1f} |"
        for i in range(n_points):
            ppo_h = int((ppo_sampled[i] - min_val) / (max_val - min_val) * height)
            sac_h = int((sac_sampled[i] - min_val) / (max_val - min_val) * height)

            if ppo_h >= row and sac_h >= row:
                line += "X"  # 重叠
            elif ppo_h >= row:
                line += "P"  # PPO
            elif sac_h >= row:
                line += "S"  # SAC
            else:
                line += " "
        print(line)

    print(" " * 9 + "+" + "-" * n_points)
    print(" " * 9 + " " + "Episodes →")

    print("\n图例: P=PPO, S=SAC, X=重叠")
    print(f"\n结果已保存至: {output_dir}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO vs SAC 对比分析")
    parser.add_argument("--ppo-dir", default="results/ppo_training")
    parser.add_argument("--sac-dir", default="results/sac_training")
    parser.add_argument("--output-dir", default="results/comparison")
    args = parser.parse_args()

    compare_algorithms(args.ppo_dir, args.sac_dir, args.output_dir)
