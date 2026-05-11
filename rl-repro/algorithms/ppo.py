"""
PPO (Proximal Policy Optimization) — 纯 PyTorch 实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared(obs)
        return self.actor_mean(features), torch.exp(self.actor_log_std.expand_as(self.actor_mean(features))), self.critic(features)

    def get_action(self, obs):
        with torch.no_grad():
            mean, std, value = self.forward(obs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_action(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1), value


class PPO:
    def __init__(self, obs_dim, act_dim, hidden_dim=64, lr=3e-4, gamma=0.99,
                 lam=0.95, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, n_epochs=10, batch_size=64, device="cpu"):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.network = ActorCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
        return advantages, advantages + values

    def update(self, rollout):
        obs = torch.FloatTensor(rollout["obs"]).to(self.device)
        actions = torch.FloatTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        returns = torch.FloatTensor(rollout["returns"]).to(self.device)
        advantages = torch.FloatTensor(rollout["advantages"]).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.n_epochs):
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                new_log_probs, entropy, values = self.network.evaluate_action(obs[idx], actions[idx])
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(-1), returns[idx])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
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
