"""
简化悬停环境 — 点质量模型（最终版）
状态: [z_error, vz] (2维)
动作: [thrust_cmd] (1维，[-1,1])
"""

import numpy as np


class HoverEnv:
    def __init__(self, seed=None):
        self.m = 0.027
        self.g = 9.81
        self.dt = 0.05       # 较大步长，减少总步数
        self.max_steps = 200
        self.z_ref = 1.0
        self.obs_dim = 2     # [z_error, vz]
        self.act_dim = 1
        self.wind_enabled = False
        self.wind_speed = 0.0
        self.rng = np.random.RandomState(seed)
        self.state = np.zeros(2)
        self.step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        z0 = self.z_ref + self.rng.uniform(-0.3, 0.3)
        self.state = np.array([max(0.1, z0), 0.0])
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        z, vz = self.state
        return np.array([self.z_ref - z, vz], dtype=np.float32)

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        z, vz = self.state

        # 推力映射
        f_hover = self.m * self.g
        f = f_hover + action * f_hover * 0.8

        # 风扰
        wind = self.rng.uniform(-self.wind_speed, self.wind_speed) if self.wind_enabled else 0.0

        # 动力学
        az = (f - self.m * self.g + wind) / self.m
        vz = vz + az * self.dt
        z = z + vz * self.dt

        # 限幅
        z = np.clip(z, 0.0, 3.0)
        if z <= 0.01:
            vz = max(0, vz)  # 地面约束

        self.state = np.array([z, vz])
        self.step_count += 1

        # 奖励（密集、平滑）
        z_err = abs(z - self.z_ref)
        reward = -z_err * 10 - abs(vz) * 0.5 - action**2 * 0.1

        # 成功奖励
        if z_err < 0.05:
            reward += 2.0

        terminated = z <= 0.01 and self.step_count > 10
        truncated = self.step_count >= self.max_steps

        info = {"pos_error": z_err, "z": z, "vz": vz}
        return self._get_obs(), reward, terminated, truncated, info


class HoverEnvWind(HoverEnv):
    def __init__(self, wind_speed=1.0, **kwargs):
        super().__init__(**kwargs)
        self.wind_enabled = True
        self.wind_speed = wind_speed
