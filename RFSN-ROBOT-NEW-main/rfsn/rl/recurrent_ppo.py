try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("RL extras not installed. Run: pip install '.[rl]'") from e

import numpy as np

class RecurrentPPO(nn.Module):
    """
    PPO Agent with LSTM Memory.
    Input -> FC -> LSTM -> [Actor Head, Critic Head]
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(RecurrentPPO, self).__init__()
        
        # Feature Extractor
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        
        # Memory
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Actor
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        
    def forward(self, x, hidden=None):
        """Forward pass for single step or batch."""
        # x: [Batch, Seq, Dim] or [Batch, Dim]
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add seq dim
            
        x = F.tanh(self.fc1(x))
        
        # LSTM
        self.lstm.flatten_parameters()
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
            
        return out, hidden
        
    def get_action(self, x, hidden):
        """Sample action for environment step."""
        with torch.no_grad():
            x = torch.FloatTensor(x).unsqueeze(0) # [1, Dim] for single step
            features, next_hidden = self.forward(x, hidden)
            features = features[:, -1, :] # Last step
            
            mean = self.actor_mean(features)
            std = self.actor_logstd.exp()
            dist = torch.distributions.Normal(mean, std)
            
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(features)
            
        return action.squeeze(0).numpy(), log_prob.item(), value.item(), next_hidden

    def evaluate(self, x, hidden, action):
        """Evaluate actions for update."""
        # x: [Batch, Seq, Dim]
        features, _ = self.forward(x, hidden)
        
        mean = self.actor_mean(features)
        std = self.actor_logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        
        log_probs = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(features).squeeze(-1)
        
        return log_probs, values, entropy
        
    def update(self, buffer, batch_size=64, epochs=4):
        """PPO Update."""
        
        # 1. Prepare Data
        # Flatten buffer. Handling LSTM hidden states is tricky in batch updates.
        # Simplified: We treat each trajectory as a sequence or ignore LSTM state passing across batch shuffle?
        # For simplicity (Recurrent PPO Lite):
        # We will update on full trajectories or chunks.
        # Simplest: Update on full buffer as one batch (if small T=2048).
        
        states = torch.FloatTensor(np.array(buffer['states'])) # [T, Dim]
        actions = torch.FloatTensor(np.array(buffer['actions']))
        old_log_probs = torch.FloatTensor(np.array(buffer['log_probs']))
        returns = torch.FloatTensor(np.array(buffer['returns']))
        advantages = torch.FloatTensor(np.array(buffer['advantages']))
        
        # Add seq dim [T, 1, Dim] -> we treat T as Batch of Seq=1 for simple feedforward update?
        # No, LSTM needs history.
        # We should forward the whole sequence to get new values?
        # Yes. But optimize over mini-batches of time steps?
        # Recurrent PPO training is complex.
        # STRATEGY: Treat the whole rollout (T=2048) as ONE sequence with Batch=1.
        
        states = states.unsqueeze(0)   # [1, T, Dim]
        actions = actions.unsqueeze(0) # [1, T, Act]
        
        inds = np.arange(states.size(1))
        
        for _ in range(epochs):
            # Evaluate Whole Sequence
            # Reset hidden for update pass
            features, _ = self.forward(states) # features: [1, T, 128]
            
            mean = self.actor_mean(features)
            std = self.actor_logstd.exp()
            dist = torch.distributions.Normal(mean, std)
            
            log_probs = dist.log_prob(actions).sum(dim=-1).squeeze(0) # [T]
            values = self.critic(features).squeeze(0).squeeze(-1) # [T]
            entropy = dist.entropy().sum(dim=-1).squeeze(0).mean()
            
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

class MemoryBuffer:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def store(self, s, a, r, d, lp, v):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(d)
        self.log_probs.append(lp)
        self.values.append(v)
        
    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        rewards = self.rewards + [last_value]
        values = self.values + [last_value]
        
        returns = []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = rewards[i] + gamma * values[i+1] * (1 - self.dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - self.dones[i]) * gae
            returns.insert(0, gae + values[i])
            
        self.returns = returns
        self.advantages = [r - v for r, v in zip(returns, self.values)]
        
        # Normalize Adv
        adv = np.array(self.advantages)
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        return {
            'states': self.states,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'returns': self.returns,
            'advantages': self.advantages
        }
