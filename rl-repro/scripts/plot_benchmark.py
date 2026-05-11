"""高质量可视化 — 训练曲线 + PID vs SAC 对比 + 风扰鲁棒性"""

import json, os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib 不可用，跳过图表生成")


def smooth(data, window=50):
    """滑动平均"""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training_curves(ppo_path, sac_path, save_dir):
    """绘制 PPO/SAC 训练曲线"""
    if not HAS_MPL:
        return

    with open(ppo_path) as f:
        ppo_rewards = json.load(f)
    with open(sac_path) as f:
        sac_rewards = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PPO
    ax = axes[0]
    episodes = np.arange(len(ppo_rewards))
    ax.plot(episodes, ppo_rewards, alpha=0.3, color='blue', linewidth=0.5)
    if len(ppo_rewards) > 50:
        smoothed = smooth(ppo_rewards, 50)
        ax.plot(np.arange(49, len(ppo_rewards)), smoothed, color='blue', linewidth=2, label='PPO (smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('PPO Training Curve (6DOF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SAC
    ax = axes[1]
    episodes = np.arange(len(sac_rewards))
    ax.plot(episodes, sac_rewards, alpha=0.3, color='red', linewidth=0.5)
    if len(sac_rewards) > 50:
        smoothed = smooth(sac_rewards, 50)
        ax.plot(np.arange(49, len(sac_rewards)), smoothed, color='red', linewidth=2, label='SAC (smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('SAC Training Curve (6DOF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {path}")


def plot_pid_vs_sac(save_dir):
    """绘制 PID vs SAC 对比图"""
    if not HAS_MPL:
        return

    # 从 benchmark 结果读取数据
    benchmark_path = os.path.join(save_dir, 'benchmark_results.json')
    if not os.path.exists(benchmark_path):
        print("benchmark_results.json 不存在，跳过对比图")
        return

    with open(benchmark_path) as f:
        data = json.load(f)

    # 无风对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 位置误差对比
    ax = axes[0]
    methods = []
    z_errors = []
    if 'PID' in data.get('no_wind', {}):
        methods.append('PID')
        z_errors.append(data['no_wind']['PID']['mean_z_error'])
    if 'SAC' in data.get('no_wind', {}):
        methods.append('SAC')
        z_errors.append(data['no_wind']['SAC']['mean_z_error'])

    if methods:
        colors = ['#2196F3', '#FF5722']
        bars = ax.bar(methods, z_errors, color=colors[:len(methods)])
        ax.set_ylabel('Position Error (m)')
        ax.set_title('No-Wind: Position Error Comparison')
        for bar, val in zip(bars, z_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}m', ha='center', va='bottom')
        ax.grid(True, alpha=0.3, axis='y')

    # 风扰对比
    ax = axes[1]
    wind_data = data.get('wind', {})
    wind_speeds = []
    pid_errors = []
    sac_errors = []

    for key in sorted(wind_data.get('PID', {}).keys()):
        ws = float(key.split('_')[1])
        wind_speeds.append(ws)
        pid_errors.append(wind_data['PID'][key]['mean_z_error'])
        if key in wind_data.get('SAC', {}):
            sac_errors.append(wind_data['SAC'][key]['mean_z_error'])

    x = np.arange(len(wind_speeds))
    width = 0.35
    ax.bar(x - width/2, pid_errors, width, label='PID', color='#2196F3')
    if sac_errors:
        ax.bar(x + width/2, sac_errors, width, label='SAC', color='#FF5722')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Wind Disturbance: Position Error')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ws:.0f}' for ws in wind_speeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(save_dir, 'pid_vs_sac.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PID vs SAC 对比图已保存: {path}")


def plot_wind_robustness(save_dir):
    """绘制风扰鲁棒性图"""
    if not HAS_MPL:
        return

    benchmark_path = os.path.join(save_dir, 'benchmark_results.json')
    if not os.path.exists(benchmark_path):
        return

    with open(benchmark_path) as f:
        data = json.load(f)

    wind_data = data.get('wind', {})
    if not wind_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color, marker in [('PID', '#2196F3', 'o'), ('SAC', '#FF5722', 's')]:
        if method not in wind_data:
            continue
        speeds = []
        errors = []
        for key in sorted(wind_data[method].keys()):
            speeds.append(float(key.split('_')[1]))
            errors.append(wind_data[method][key]['mean_z_error'])
        ax.plot(speeds, errors, color=color, marker=marker, linewidth=2, markersize=8, label=method)

    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Wind Robustness Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'wind_robustness.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"风扰鲁棒性图已保存: {path}")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base, 'results', 'figures')
    os.makedirs(save_dir, exist_ok=True)

    ppo_path = os.path.join(base, 'results', 'ppo_training', 'episode_rewards.json')
    sac_path = os.path.join(base, 'results', 'sac_training', 'episode_rewards.json')

    print("=" * 50)
    print("生成可视化图表")
    print("=" * 50)

    plot_training_curves(ppo_path, sac_path, save_dir)
    plot_pid_vs_sac(os.path.join(base, 'results'))
    plot_wind_robustness(os.path.join(base, 'results'))

    print(f"\n所有图表已保存到: {save_dir}")


if __name__ == "__main__":
    main()
