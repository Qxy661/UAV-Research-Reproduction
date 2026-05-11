"""
6DOF 简化悬停环境（姿态+高度联合控制）
状态 (8维): [z_err, vz, φ, θ, ψ, p, q, r]
动作 (4维): [thrust, τ_φ, τ_θ, τ_psi] 归一化到 [-1,1]
"""

import numpy as np


class HoverEnv:
    def __init__(self, seed=None):
        # 物理参数
        self.m = 0.027
        self.g = 9.81
        self.Jx = 1.4e-5
        self.Jy = 1.4e-5
        self.Jz = 2.17e-5
        self.l = 0.046
        self.Ct = 3.1582e-10
        self.omega_max = 2500

        # 仿真参数
        self.dt = 0.02
        self.max_steps = 500
        self.z_ref = 1.0
        self.att_ref = np.array([0.0, 0.0, 0.0])  # 目标姿态

        # 空间维度
        self.obs_dim = 8
        self.act_dim = 4

        # 风扰
        self.wind_enabled = False
        self.wind_speed = 0.0
        self.rng = np.random.RandomState(seed)

        # 状态: [z, vz, φ, θ, ψ, p, q, r]
        self.state = np.zeros(8)
        self.step_count = 0
        self.hover_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.state = np.zeros(8)
        # 初始位置靠近目标
        self.state[0] = self.z_ref + self.rng.uniform(-0.2, 0.2)
        self.state[0] = max(0.3, self.state[0])
        # 小角度初始姿态
        self.state[2:5] = self.rng.uniform(-0.05, 0.05, size=3)
        self.step_count = 0
        self.hover_count = 0
        return self._get_obs()

    def _get_obs(self):
        z, vz = self.state[0], self.state[1]
        phi, theta, psi = self.state[2], self.state[3], self.state[4]
        p, q, r = self.state[5], self.state[6], self.state[7]
        return np.array([
            self.z_ref - z,  # 高度误差
            vz,              # 垂直速度
            phi,             # 滚转角
            theta,           # 俯仰角
            psi,             # 偏航角
            p,               # 滚转角速率
            q,               # 俯仰角速率
            r,               # 偏航角速率
        ], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        z, vz = self.state[0], self.state[1]
        phi, theta, psi = self.state[2], self.state[3], self.state[4]
        p, q, r = self.state[5], self.state[6], self.state[7]

        # 推力映射
        f_hover = self.m * self.g
        f = f_hover * (1.0 + 0.5 * action[0])  # ±50% 悬停推力
        tau_phi = action[1] * 1e-4
        tau_theta = action[2] * 1e-4
        tau_psi = action[3] * 1e-4

        # 风扰
        if self.wind_enabled:
            wind_z = self.rng.uniform(-self.wind_speed, self.wind_speed)
        else:
            wind_z = 0.0

        # 高度动力学
        az = (f * np.cos(phi) * np.cos(theta) - self.m * self.g + wind_z) / self.m
        vz = vz + az * self.dt
        z = z + vz * self.dt

        # 姿态动力学（简化）
        dp = (tau_phi - (self.Jz - self.Jy) * q * r) / self.Jx * self.dt
        dq = (tau_theta - (self.Jx - self.Jz) * p * r) / self.Jy * self.dt
        dr = (tau_psi - (self.Jy - self.Jx) * p * q) / self.Jz * self.dt
        p = p + dp
        q = q + dq
        r = r + dr

        # 姿态运动学
        phi = phi + p * self.dt
        theta = theta + q * self.dt
        psi = psi + r * self.dt

        # 限幅
        z = np.clip(z, 0.0, 3.0)
        phi = np.clip(phi, -np.pi/3, np.pi/3)
        theta = np.clip(theta, -np.pi/3, np.pi/3)

        if z <= 0.01:
            vz = max(0, vz)

        self.state = np.array([z, vz, phi, theta, psi, p, q, r])
        self.step_count += 1

        # 奖励函数
        z_err = abs(z - self.z_ref)
        att_err = phi**2 + theta**2
        vel_penalty = abs(vz) * 0.1
        rate_penalty = (p**2 + q**2 + r**2) * 0.01
        ctrl_penalty = np.sum(action**2) * 0.001

        # 分层奖励
        if z_err < 0.05 and att_err < 0.01:
            reward = 10.0  # 精确悬停 + 稳定姿态
            self.hover_count += 1
            if self.hover_count >= 100:
                reward += 50.0
        elif z_err < 0.1 and att_err < 0.05:
            reward = 5.0
            self.hover_count = 0
        elif z_err < 0.3:
            reward = 1.0 - z_err * 5
            self.hover_count = 0
        else:
            reward = -z_err * 5
            self.hover_count = 0

        reward -= vel_penalty + rate_penalty + ctrl_penalty

        # 终止条件
        terminated = False
        if z <= 0.01 and self.step_count > 10:
            reward -= 50
            terminated = True
        if abs(phi) > np.pi/2 or abs(theta) > np.pi/2:
            reward -= 50
            terminated = True

        truncated = self.step_count >= self.max_steps
        info = {"z_err": z_err, "pos_error": z_err, "att_err": att_err, "z": z, "hover_count": self.hover_count}
        return self._get_obs(), reward, terminated, truncated, info


class HoverEnvWind(HoverEnv):
    def __init__(self, wind_speed=1.0, **kwargs):
        super().__init__(**kwargs)
        self.wind_enabled = True
        self.wind_speed = wind_speed
