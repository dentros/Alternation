"""
Deep Q-Network (DQN) Agents for ALT Metrics experiments.

This module implements DQN components:
- DQNNetwork: Neural network for Q-value approximation
- ReplayBuffer: Experience replay for stable training
- DQNAgent: DQN agent with policy and target networks

Compatible with existing experiment framework for state/reward types.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple

from config import FULL_REWARD, ALPHA, GAMMA, EPSILON_INITIAL, EPSILON_MIN

#===============================================================================
# DQN Neural Network
#===============================================================================

class DQNNetwork(nn.Module):
    """Neural network for approximating Q-values."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


#===============================================================================
# Replay Buffer
#===============================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


#===============================================================================
# DQN Agent
#===============================================================================

class DQNAgent:
    """Deep Q-Network agent for multi-agent coordination."""

    def __init__(
        self,
        identifier: int,
        num_agents: int,
        num_positions: int,
        state_type: str,
        learning_rate: float = 0.001,  # Neural network learning rate (NOT ALPHA!)
        gamma: float = GAMMA,
        buffer_size: int = 10000,
        batch_size: int = 32,
        update_target_every: int = 100,
        device: str = None
    ):
        self.identifier = identifier
        self.num_agents = num_agents
        self.num_positions = num_positions
        self.state_type = state_type
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.training_steps = 0

        # Device setup (CPU/GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Action space
        self.action_space = [0, 1]  # 0=stay, 1=move
        self.action_size = 2

        # State size calculation (matches Q-learning)
        if state_type == 'Type-A':
            # Type-A: only positions (num_positions^num_agents states)
            self.state_size = num_positions ** num_agents
        else:
            # Type-B: positions + last winners (larger state space)
            self.state_size = (num_positions ** num_agents) * (2 ** num_agents)

        # Neural networks
        self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Total reward tracking
        self.total_reward = 0

    def encode_state_to_vector(self, positions: List[int], terminal_states: List[int] = None) -> np.ndarray:
        """
        Encode state as one-hot vector for neural network input.

        Args:
            positions: List of agent positions [0, num_positions-1]
            terminal_states: List of terminal indicators [0 or 1] (Type-B only)

        Returns:
            One-hot encoded state vector of size self.state_size
        """
        state_vector = np.zeros(self.state_size)

        # Calculate state index (matches Q-learning encoding)
        if self.state_type == 'Type-A':
            # Type-A: position-based index
            state_index = 0
            for i, pos in enumerate(positions):
                state_index += pos * (self.num_positions ** i)
        else:
            # Type-B: positions + terminal states
            pos_index = 0
            for i, pos in enumerate(positions):
                pos_index += pos * (self.num_positions ** i)

            term_index = 0
            if terminal_states is not None:
                for i, term in enumerate(terminal_states):
                    term_index += term * (2 ** i)

            state_index = pos_index + term_index * (self.num_positions ** self.num_agents)

        # One-hot encoding
        if 0 <= state_index < self.state_size:
            state_vector[state_index] = 1.0

        return state_vector

    def select_action(self, state_vector: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state_vector: One-hot encoded state
            epsilon: Exploration rate

        Returns:
            Action (0 or 1)
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.choice(self.action_space)
        else:
            # Exploit: greedy action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experiences yet

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions)

        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
