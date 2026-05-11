"""
PPO (Proximal Policy Optimization) 实现
适用于连续动作空间的四旋翼控制任务

特点:
- Clip 机制限制策略更新幅度
- GAE 优势估计
- Actor-Critic 结构
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic 网络"""

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()

        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor（策略网络）- 输出均值和对数标准差
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic（价值网络）
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        value = self.critic(features)
        return action_mean, action_std, value

    def get_action(self, obs):
        """采样动作（用于数据收集）"""
        with torch.no_grad():
            action_mean, action_std, value = self.forward(obs)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_action(self, obs, action):
        """评估动作（用于更新）"""
        action_mean, action_std, value = self.forward(obs)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PPO:
    """PPO 算法"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device="cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        # 创建网络
        self.network = ActorCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones):
        """计算 GAE 优势估计"""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            last_gae = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """更新策略和价值网络"""
        obs = torch.FloatTensor(rollout["obs"]).to(self.device)
        actions = torch.FloatTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        returns = torch.FloatTensor(rollout["returns"]).to(self.device)
        advantages = torch.FloatTensor(rollout["advantages"]).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.n_epochs):
            # Mini-batch 更新
            indices = np.random.permutation(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                # 评估动作
                new_log_probs, entropy, values = self.network.evaluate_action(
                    obs[idx], actions[idx]
                )

                # PPO-Clip 目标
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = nn.MSELoss()(values.squeeze(), returns[idx])

                # 熵奖励
                entropy_loss = -entropy.mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
