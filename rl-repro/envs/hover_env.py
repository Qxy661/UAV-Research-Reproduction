"""
简化四旋翼悬停环境（CPU 友好）
不依赖 Gym-PyBullet-Drones，使用点质量模型

状态空间 (12维):
  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

动作空间 (4维，连续):
  [f, tau_phi, tau_theta, tau_psi]，归一化到 [-1, 1]

奖励设计:
  - 位置误差惩罚：-||pos - pos_ref||
  - 姿态惩罚：-||euler||
  - 控制量惩罚：-0.01 * ||u||
  - 坠毁惩罚：-100（高度 < 0 或 姿态 > 60°）
  - 成功奖励：+100（悬停误差 < 0.1m 持续 100 步）
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HoverEnv(gym.Env):
    """简化四旋翼悬停环境"""

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()

        # 物理参数（Crazyflie 2.1）
        self.m = 0.027          # 质量 [kg]
        self.g = 9.81           # 重力加速度
        self.J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])  # 惯性矩阵
        self.l = 0.046          # 机臂长度
        self.Ct = 3.1582e-10    # 推力系数
        self.k_drag = 0.01      # 拖拽系数

        # 仿真参数
        self.dt = 0.02          # 仿真步长 [s] (50Hz)
        self.max_steps = 500    # 最大步数
        self.pos_ref = np.array([0.0, 0.0, 1.0])  # 悬停目标

        # 风扰配置
        self.wind_enabled = False
        self.wind_speed = 0.0

        # 状态/动作空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # 内部状态
        self.state = np.zeros(12)
        self.step_count = 0
        self.hover_count = 0  # 连续悬停计数

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始状态（小幅偏移）
        self.state = np.zeros(12)
        self.state[0:3] = self.np_random.uniform(-0.5, 0.5, size=3)
        self.state[2] = self.np_random.uniform(0.5, 1.5)  # z > 0
        self.state[6:9] = self.np_random.uniform(-0.1, 0.1, size=3)  # 小角度
        self.step_count = 0
        self.hover_count = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        # 动作反归一化
        action = np.clip(action, -1.0, 1.0)
        f_max = 4 * self.Ct * self.omega_max**2
        tau_max = self.l * self.Ct * self.omega_max**2

        f = (action[0] + 1) / 2 * f_max          # [0, f_max]
        tau = action[1:4] * tau_max                # [-tau_max, tau_max]

        # 动力学更新（简化的欧拉积分）
        x = self.state.copy()
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]

        # 旋转矩阵（简化）
        R = self._euler_to_rotation(phi, theta, psi)

        # 平移动力学
        thrust_body = np.array([0, 0, f])
        thrust_world = R @ thrust_body
        gravity = np.array([0, 0, -self.m * self.g])
        drag = -self.k_drag * x[3:6]

        # 风扰
        if self.wind_enabled:
            wind = self.np_random.uniform(-self.wind_speed, self.wind_speed, size=3)
        else:
            wind = np.zeros(3)

        accel = (thrust_world + gravity + drag + wind) / self.m

        # 旋转动力学
        omega = np.array([p, q, r])
        tau_total = tau - np.cross(omega, self.J @ omega)
        omega_dot = np.linalg.solve(self.J, tau_total)

        # 欧拉角速率
        W = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        euler_dot = W @ omega

        # 更新状态
        x[3:6] += accel * self.dt
        x[0:3] += x[3:6] * self.dt
        x[9:12] += omega_dot * self.dt
        x[6:9] += euler_dot * self.dt

        self.state = x
        self.step_count += 1

        # 计算奖励
        pos_error = np.linalg.norm(x[0:3] - self.pos_ref)
        att_error = np.linalg.norm(x[6:9])
        ctrl_penalty = 0.01 * np.linalg.norm(action)

        reward = -pos_error - 0.1 * att_error - ctrl_penalty

        # 悬停成功奖励
        if pos_error < 0.1:
            self.hover_count += 1
            if self.hover_count >= 100:
                reward += 100
        else:
            self.hover_count = 0

        # 终止条件
        terminated = False
        truncated = False

        # 坠毁检测
        if x[2] < 0 or abs(x[6]) > np.deg2rad(60) or abs(x[7]) > np.deg2rad(60):
            reward -= 100
            terminated = True

        # 超时
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "pos_error": pos_error,
            "att_error": att_error,
            "hover_count": self.hover_count,
        }

        return self.state.astype(np.float32), reward, terminated, truncated, info

    def _euler_to_rotation(self, phi, theta, psi):
        """欧拉角到旋转矩阵（ZYX 顺序）"""
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cr, sr = np.cos(psi), np.sin(psi)

        R = np.array([
            [cr*ct, cr*st*sp - sr*cp, cr*st*cp + sr*sp],
            [sr*ct, sr*st*sp + cr*cp, sr*st*cp - cr*sp],
            [-st, ct*sp, ct*cp]
        ])
        return R

    @property
    def omega_max(self):
        return 2500  # 最大电机转速


class HoverEnvWind(HoverEnv):
    """带风扰的悬停环境"""

    def __init__(self, wind_speed=3.0, **kwargs):
        super().__init__(**kwargs)
        self.wind_enabled = True
        self.wind_speed = wind_speed
