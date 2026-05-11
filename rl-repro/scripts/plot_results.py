"""训练曲线可视化脚本"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def smooth(data, window=50):
    """滑动平均"""
    return [sum(data[max(0,i-window+1):i+1]) / (i - max(0,i-window+1) + 1) for i in range(len(data))]


def plot_ascii(rewards, title="Training Curve", width=60, height=20):
    """ASCII 训练曲线"""
    smoothed = smooth(rewards, 50)
    max_val = max(smoothed)
    min_val = min(smoothed)
    val_range = max_val - min_val if max_val != min_val else 1

    # 采样
    n = len(smoothed)
    step = max(1, n // width)
    sampled = smoothed[::step][:width]

    print(f"\n{title}")
    print(f"  min={min_val:.1f}, max={max_val:.1f}, final={smoothed[-1]:.1f}")
    print("  " + "-" * (width + 10))

    for row in range(height, -1, -1):
        threshold = min_val + val_range * row / height
        line = f"  {threshold:7.1f} |"
        for val in sampled:
            if val >= threshold:
                line += "#"
            else:
                line += " "
        print(line)

    print("  " + " " * 9 + "+" + "-" * width)
    print("  " + " " * 9 + "Episodes →")


def analyze_results(ppo_path, sac_path):
    """分析并对比 PPO/SAC 结果"""
    with open(ppo_path) as f:
        ppo_rewards = json.load(f)
    with open(sac_path) as f:
        sac_rewards = json.load(f)

    # 统计
    ppo_final = sum(ppo_rewards[-50:]) / 50
    sac_final = sum(sac_rewards[-50:]) / 50
    ppo_max = max(ppo_rewards)
    sac_max = max(sac_rewards)

    print("=" * 60)
    print("PPO vs SAC 训练对比")
    print("=" * 60)

    plot_ascii(ppo_rewards, "PPO Training Curve")
    plot_ascii(sac_rewards, "SAC Training Curve")

    print("\n" + "=" * 60)
    print("对比统计:")
    print(f"  {'指标':<20} {'PPO':<15} {'SAC':<15} {'优胜':<10}")
    print("  " + "-" * 55)
    print(f"  {'最终奖励(50ep)':<20} {ppo_final:<15.2f} {sac_final:<15.2f} {'PPO' if ppo_final > sac_final else 'SAC':<10}")
    print(f"  {'最大奖励':<20} {ppo_max:<15.2f} {sac_max:<15.2f} {'PPO' if ppo_max > sac_max else 'SAC':<10}")
    print(f"  {'训练episodes':<20} {len(ppo_rewards):<15} {len(sac_rewards):<15}")
    print("=" * 60)

    return {"ppo_final": ppo_final, "sac_final": sac_final, "ppo_max": ppo_max, "sac_max": sac_max}


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ppo_path = os.path.join(base, "results", "ppo_training", "episode_rewards.json")
    sac_path = os.path.join(base, "results", "sac_training", "episode_rewards.json")

    if os.path.exists(ppo_path) and os.path.exists(sac_path):
        analyze_results(ppo_path, sac_path)
    else:
        print("训练结果文件不存在，请先运行训练脚本")
