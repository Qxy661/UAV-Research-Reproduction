"""
SAC (Soft Actor-Critic) — 纯 PyTorch 实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
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
            "obs": self.obs[idx], "actions": self.actions[idx],
            "rewards": self.rewards[idx], "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        features = self.net(obs)
        return self.mean(features), torch.clamp(self.log_std(features), -20, 2)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), mean


class SAC:
    def __init__(self, obs_dim, act_dim, hidden_dim=256, lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=None, auto_alpha=True, target_entropy=None,
                 buffer_size=100000, batch_size=256, device="cpu"):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.auto_alpha = auto_alpha

        self.q_network = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q_target = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim).to(device)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        if auto_alpha:
            self.target_entropy = target_entropy if target_entropy is not None else -act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha if alpha is not None else 0.2

        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)

    def select_action(self, obs, evaluate=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if evaluate:
                _, _, mean = self.policy.sample(obs_t)
                return torch.tanh(mean).cpu().numpy().flatten()
            action, _, _ = self.policy.sample(obs_t)
            return action.cpu().numpy().flatten()

    def update(self):
        if self.buffer.size < self.batch_size:
            return {}
        batch = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_obs)
            q1_t, q2_t = self.q_target(next_obs, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * (torch.min(q1_t, q2_t) - self.alpha * next_log_probs)

        q1, q2 = self.q_network(obs, actions)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_actions, log_probs, _ = self.policy.sample(obs)
        q1_new, q2_new = self.q_network(obs, new_actions)
        policy_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        for p, tp in zip(self.q_network.parameters(), self.q_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(), "alpha": self.alpha}

    def save(self, path):
        torch.save({"q_network": self.q_network.state_dict(), "policy": self.policy.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.policy.load_state_dict(ckpt["policy"])
