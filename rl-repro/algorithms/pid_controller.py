"""级联 PID 控制器 — 与 hover_env 动作空间兼容

输出 4D 归一化动作 [thrust_cmd, tau_phi, tau_theta, tau_psi] ∈ [-1, 1]
与 SAC/PPO 在同一环境中公平对比
"""

import numpy as np


class PIDController:
    def __init__(self, m=0.027, g=9.81, dt=0.02):
        # 姿态外环
        self.Kp_att = 6.0
        self.Ki_att = 0.1
        self.Kd_att = 0.3

        # 角速率内环
        self.Kp_rate = 0.015
        self.Ki_rate = 0.001
        self.Kd_rate = 0.0001

        # 高度环
        self.Kp_z = 10.0
        self.Ki_z = 0.5
        self.Kd_z = 5.0

        self.m = m
        self.g = g
        self.dt = dt
        self.int_max = 0.5

        self.reset()

    def reset(self):
        self.int_phi = 0.0
        self.int_theta = 0.0
        self.int_psi = 0.0
        self.int_p = 0.0
        self.int_q = 0.0
        self.int_r = 0.0
        self.int_z = 0.0
        self.e_phi_prev = 0.0
        self.e_theta_prev = 0.0
        self.e_psi_prev = 0.0
        self.e_p_prev = 0.0
        self.e_q_prev = 0.0
        self.e_r_prev = 0.0

    def _clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def step(self, obs, ref=None):
        """根据观测输出归一化动作

        obs: [z_err, vz, phi, theta, psi, p, q, r]  (from hover_env)
        ref: [phi_ref, theta_ref, psi_ref]  默认全 0（悬停）

        returns: [thrust_cmd, tau_phi_cmd, tau_theta_cmd, tau_psi_cmd] ∈ [-1, 1]
        """
        if ref is None:
            ref = [0.0, 0.0, 0.0]

        z_err, vz = obs[0], obs[1]
        phi, theta, psi = obs[2], obs[3], obs[4]
        p, q, r = obs[5], obs[6], obs[7]

        z_ref_err = z_err  # hover_env 给的就是 z_ref - z
        phi_ref, theta_ref, psi_ref = ref

        # === 外环：姿态角 → 角速率参考 ===
        e_phi = phi_ref - phi
        e_theta = theta_ref - theta
        e_psi = psi_ref - psi

        self.int_phi = self._clamp(self.int_phi + e_phi * self.dt, -self.int_max, self.int_max)
        self.int_theta = self._clamp(self.int_theta + e_theta * self.dt, -self.int_max, self.int_max)
        self.int_psi = self._clamp(self.int_psi + e_psi * self.dt, -self.int_max, self.int_max)

        d_e_phi = (e_phi - self.e_phi_prev) / self.dt
        d_e_theta = (e_theta - self.e_theta_prev) / self.dt
        d_e_psi = (e_psi - self.e_psi_prev) / self.dt

        p_ref = self.Kp_att * e_phi + self.Ki_att * self.int_phi + self.Kd_att * d_e_phi
        q_ref = self.Kp_att * e_theta + self.Ki_att * self.int_theta + self.Kd_att * d_e_theta
        r_ref = self.Kp_att * e_psi + self.Ki_att * self.int_psi + self.Kd_att * d_e_psi

        self.e_phi_prev = e_phi
        self.e_theta_prev = e_theta
        self.e_psi_prev = e_psi

        # === 内环：角速率 → 力矩 ===
        e_p = p_ref - p
        e_q = q_ref - q
        e_r = r_ref - r

        self.int_p = self._clamp(self.int_p + e_p * self.dt, -self.int_max, self.int_max)
        self.int_q = self._clamp(self.int_q + e_q * self.dt, -self.int_max, self.int_max)
        self.int_r = self._clamp(self.int_r + e_r * self.dt, -self.int_max, self.int_max)

        d_e_p = (e_p - self.e_p_prev) / self.dt
        d_e_q = (e_q - self.e_q_prev) / self.dt
        d_e_r = (e_r - self.e_r_prev) / self.dt

        tau_phi = self.Kp_rate * e_p + self.Ki_rate * self.int_p + self.Kd_rate * d_e_p
        tau_theta = self.Kp_rate * e_q + self.Ki_rate * self.int_q + self.Kd_rate * d_e_q
        tau_psi = self.Kp_rate * e_r + self.Ki_rate * self.int_r + self.Kd_rate * d_e_r

        self.e_p_prev = e_p
        self.e_q_prev = e_q
        self.e_r_prev = e_r

        # === 高度环 ===
        e_z = z_ref_err
        self.int_z = self._clamp(self.int_z + e_z * self.dt, -self.int_max, self.int_max)
        d_e_z = -vz  # d(z_ref - z)/dt = -vz

        f_hover = self.m * self.g
        f = f_hover + self.Kp_z * e_z + self.Ki_z * self.int_z + self.Kd_z * d_e_z

        # 归一化到 [-1, 1]（与 hover_env 的动作映射一致）
        # hover_env: f = f_hover * (1 + 0.5 * action[0])
        # 所以 action[0] = (f - f_hover) / (0.5 * f_hover)
        thrust_cmd = (f - f_hover) / (0.5 * f_hover)
        thrust_cmd = self._clamp(thrust_cmd, -1.0, 1.0)

        # hover_env: tau = action[i] * 1e-4
        # 所以 action[i] = tau / 1e-4
        tau_phi_cmd = self._clamp(tau_phi / 1e-4, -1.0, 1.0)
        tau_theta_cmd = self._clamp(tau_theta / 1e-4, -1.0, 1.0)
        tau_psi_cmd = self._clamp(tau_psi / 1e-4, -1.0, 1.0)

        return np.array([thrust_cmd, tau_phi_cmd, tau_theta_cmd, tau_psi_cmd], dtype=np.float32)
