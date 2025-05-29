import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque
import torch.nn.functional as F

# Define experience tuple
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done', 'action_mask'])


class RunningNormalizer:
    """Online state normalizer"""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        delta = batch_mean - self.mean
        self.mean += delta * len(x) / (self.count + len(x))
        self.var = (self.var * self.count + batch_var * len(x)) / (self.count + len(x))
        self.count += len(x)

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""

    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        return F.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    """Q-network with noisy layers and layer normalization"""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            NoisyLinear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.head = NoisyLinear(128, action_size)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        x = self.feature(x)
        return self.head(x)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with TD-error prioritization"""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Experience(*args)
        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []

        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        samples = [self.buffer[i] for i in indices]
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, max(errors))


class DQNAgent:
    """Fully optimized DQN agent"""

    def __init__(self, state_size, action_size, device='cuda'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Networks and optimizer
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)

        # Experience replay
        self.memory = PrioritizedReplayBuffer(100000)
        self.normalizer = RunningNormalizer(state_size)

        # Training parameters
        self.gamma = 0.99
        self.batch_size = 128
        self.update_target_every = 500
        self.tau = 0.005  # Soft update parameter
        self.steps = 0

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.noise_scale = 0.1

    def act(self, state, action_mask=None, training=True):
        state = self.normalizer.normalize(state)
        state = torch.FloatTensor(state).to(self.device)

        # Exploration policy
        if training and random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = [a for a, m in enumerate(action_mask) if m == 1]
                return random.choice(valid_actions) if valid_actions else 0
            return random.randrange(self.action_size)

        # Noisy exploration
        if training:
            self.model.reset_noise()

        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0))
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values.masked_fill(mask == 0, -float('inf'))
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0.0  # Return loss value

        # Sample batch
        batch, indices, weights = self.memory.sample(self.batch_size)
        weights = weights.to(self.device)

        # Prepare data
        states = torch.FloatTensor(
            np.array([self.normalizer.normalize(e.state) for e in batch])
        ).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(
            np.array([self.normalizer.normalize(e.next_state) for e in batch])
        ).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q = self.model(states).gather(1, actions)

        # Double DQN target calculation
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Update priorities
        errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        # Compute weighted loss
        loss = (weights * (current_q - target_q).pow(2)).mean()

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Exploration rate decay
        self.steps += 1
        if self.steps % 100 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'target_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'normalizer_mean': self.normalizer.mean,
            'normalizer_var': self.normalizer.var,
            'normalizer_count': self.normalizer.count,
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Handle key name compatibility
        model_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model_state'
        target_key = 'target_model_state_dict' if 'target_model_state_dict' in checkpoint else 'target_state'
        optim_key = 'optimizer_state_dict' if 'optimizer_state_dict' in checkpoint else 'optimizer_state'

        self.model.load_state_dict(checkpoint[model_key])
        self.target_model.load_state_dict(checkpoint[target_key])
        self.optimizer.load_state_dict(checkpoint[optim_key])

        # Load other parameters
        self.normalizer.mean = checkpoint.get('normalizer_mean', np.zeros(self.state_size))
        self.normalizer.var = checkpoint.get('normalizer_var', np.ones(self.state_size))
        self.normalizer.count = checkpoint.get('normalizer_count', 1e-4)
        self.steps = checkpoint.get('steps', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)