"""
DQN Multi-Agent Environment for ALT Metrics experiments.

This module implements the environment class for DQN experiments:
- DQNEnvironment: Multi-agent environment compatible with Q-learning interface
- Supports both state types (Type-A, Type-B)
- Supports both reward types (ILF, IQF)

Usage:
    from dqn_agents import DQNAgent
    from dqn_environment import DQNEnvironment

    env = DQNEnvironment(
        num_agents=3,
        num_positions=3,
        state_type='Type-A',
        reward_type='ILF'
    )
"""

import numpy as np
from typing import List, Tuple

from config import FULL_REWARD, ALPHA, GAMMA, EPSILON_INITIAL, EPSILON_MIN
from dqn_agents import DQNAgent

#===============================================================================
# DQN Environment (Multi-Agent)
#===============================================================================

class DQNEnvironment:
    """
    Multi-agent environment for DQN training.
    Compatible with Q-learning Environment interface.
    """

    def __init__(
        self,
        num_agents: int,
        num_positions: int = 3,
        full_reward: float = FULL_REWARD,
        state_type: str = 'Type-A',
        reward_type: str = 'ILF',
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        epsilon_initial: float = EPSILON_INITIAL,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay_target: float = 0.75
    ):
        self.num_agents = num_agents
        self.num_positions = num_positions
        self.full_reward = full_reward
        self.state_type = state_type
        self.reward_type = reward_type
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay_target = epsilon_decay_target

        # Create DQN agents
        self.agents = [
            DQNAgent(
                identifier=i,
                num_agents=num_agents,
                num_positions=num_positions,
                state_type=state_type,
                learning_rate=alpha,
                gamma=gamma
            )
            for i in range(num_agents)
        ]

        # Initialize environment state
        self.reset()

    def reset(self):
        """Reset environment for new episode."""
        self.positions = [0] * self.num_agents
        self.last_top_agents = [0] * self.num_agents
        self.current_top_agents = [0] * self.num_agents
        return self.get_state()

    def get_state(self):
        """Get current state representation."""
        if self.state_type == 'Type-A':
            return self.positions, None
        else:
            return self.positions, self.last_top_agents

    def update_for_next_episode(self):
        """Update state for next episode (Type-B only)."""
        self.last_top_agents = self.current_top_agents.copy()
        self.current_top_agents = [0] * self.num_agents

    def step(self, actions: List[int]) -> Tuple[List[int], List[float], bool]:
        """
        Execute one step in the environment.

        Args:
            actions: List of actions (one per agent)

        Returns:
            next_state: Next state representation
            rewards: Rewards for each agent
            done: Whether episode is finished
        """
        # Execute actions
        next_positions = self.positions.copy()
        someone_reached_terminal = False

        for i in range(self.num_agents):
            if actions[i] == 1:  # Move forward
                next_positions[i] += 1

                if next_positions[i] >= self.num_positions - 1:
                    next_positions[i] = self.num_positions - 1
                    someone_reached_terminal = True
                    self.current_top_agents[i] = 1

        # Update positions
        self.positions = next_positions

        # Calculate rewards if episode done
        if someone_reached_terminal:
            rewards = self.calculate_rewards()
            done = True
        else:
            rewards = [0.0] * self.num_agents
            done = False

        next_state = self.get_state()
        return next_state, rewards, done

    def calculate_rewards(self) -> List[float]:
        """Calculate rewards based on terminal states."""
        count_of_top_agents = sum(self.current_top_agents)
        rewards = [0.0] * self.num_agents

        for i in range(self.num_agents):
            if self.current_top_agents[i] == 1:
                if count_of_top_agents == self.num_agents:
                    # Everyone reached terminal - no reward
                    rewards[i] = 0.0
                elif count_of_top_agents > 1:
                    # Multiple agents reached terminal
                    if self.reward_type == 'ILF':
                        rewards[i] = self.full_reward / self.num_agents
                    else:  # IQF
                        rewards[i] = self.full_reward / (self.num_agents ** 2)
                else:
                    # Solo winner - full reward
                    rewards[i] = self.full_reward

        return rewards
