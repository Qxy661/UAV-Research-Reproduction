"""
SAC (Soft Actor-Critic) 实现
适用于连续动作空间的四旋翼控制任务

特点:
- 最大熵框架：鼓励探索
- 自动温度调节：消除 α 调参
- Off-policy：可复用历史数据
- 双 Q 网络：减少过估计
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


class QNetwork(nn.Module):
    """双 Q 网络"""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class GaussianPolicy(nn.Module):
    """高斯策略网络（重参数化）"""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        features = self.net(obs)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # 数值稳定性
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        # 重参数化技巧
        x = dist.rsample()
        action = torch.tanh(x)

        # 计算 log 概率（考虑 tanh 变换）
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class SAC:
    """SAC 算法"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=None,
        auto_alpha=True,
        target_entropy=None,
        buffer_size=100000,
        batch_size=256,
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.auto_alpha = auto_alpha

        # 创建网络
        self.q_network = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q_target = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim).to(device)

        # 目标网络初始化
        self.q_target.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 温度参数
        if auto_alpha:
            if target_entropy is None:
                self.target_entropy = -act_dim  # 默认：-dim(A)
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha if alpha is not None else 0.2

        # 经验回放
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)

    def select_action(self, obs, evaluate=False):
        """选择动作"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if evaluate:
                _, _, mean = self.policy.sample(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.policy.sample(obs_tensor)
        return action.cpu().numpy().flatten()

    def update(self):
        """更新所有网络"""
        if self.buffer.size < self.batch_size:
            return {}

        # 采样
        batch = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)

        # ========== Q 网络更新 ==========
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_obs)
            q1_target, q2_target = self.q_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * q_target

        q1, q2 = self.q_network(obs, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ========== 策略更新 ==========
        new_actions, log_probs, _ = self.policy.sample(obs)
        q1_new, q2_new = self.q_network(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ========== 温度更新 ==========
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ========== 目标网络软更新 ==========
        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
        }

    def save(self, path):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "policy": self.policy.state_dict(),
            "log_alpha": self.log_alpha if self.auto_alpha else None,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.policy.load_state_dict(checkpoint["policy"])
        if self.auto_alpha and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
