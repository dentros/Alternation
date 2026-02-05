"""
Environment module for ALT Metrics experiments.

This module contains environment classes for both random and Q-learning experiments:
- RandomPolicyEnvironment: Random action selection baseline
- Agent: Q-learning agent with Q-table
- Environment: Q-learning environment with epsilon-greedy action selection
"""

import numpy as np
import random
from typing import List, Dict

from config import FULL_REWARD, ALPHA, GAMMA, EPSILON_INITIAL, EPSILON_MIN, EPSILON_DECAY_TARGET

#===============================================================================
# Random Policy Environment
#===============================================================================

class RandomPolicyEnvironment:
    def __init__(self, num_agents, num_positions=3, full_reward=FULL_REWARD, reward_type='ILF'):
        self.num_agents = num_agents
        self.num_positions = num_positions
        self.full_reward = full_reward
        self.reward_type = reward_type  # 'ILF' or 'IQF'

        # Initialize environment state
        self.reset()

    def reset(self):
        """Reset the environment for a new episode."""
        self.positions = [0] * self.num_agents
        self.next_positions = [0] * self.num_agents
        self.actions = [0] * self.num_agents
        self.current_top_agents = [0] * self.num_agents
        self.last_top_agents = [0] * self.num_agents
        self.current_rewards = [0] * self.num_agents
        self.total_reward_per_agent = [0] * self.num_agents

    def update_for_next_episode(self):
        """Update state for next episode."""
        self.last_top_agents = self.current_top_agents.copy()
        self.current_top_agents = [0] * self.num_agents

    def step(self):
        """Run a complete episode with random actions until terminal state."""
        # Reset positions and state for this episode
        self.positions = [0] * self.num_agents
        self.next_positions = [0] * self.num_agents
        self.current_top_agents = [0] * self.num_agents
        self.current_rewards = [0] * self.num_agents

        someone_reached_terminal = False
        max_rounds = 100  # Safety limit to prevent infinite loops (matches Q-learning)
        round_count = 0

        # Continue taking actions until someone reaches terminal position
        while not someone_reached_terminal and round_count < max_rounds:
            round_count += 1

            # Generate random actions (0=stay, 1=move)
            for i in range(self.num_agents):
                self.actions[i] = random.choice([0, 1])

            # Execute actions and update positions
            for i in range(self.num_agents):
                if self.actions[i] == 1:  # Move forward
                    self.next_positions[i] += 1

                    # Check if terminal position reached
                    if self.next_positions[i] >= self.num_positions - 1:
                        # Cap at terminal position
                        self.next_positions[i] = self.num_positions - 1
                        someone_reached_terminal = True
                        self.current_top_agents[i] = 1

            # Update positions for next round
            self.positions = self.next_positions.copy()

        # Calculate rewards based on terminal states
        count_of_top_agents = sum(self.current_top_agents)

        for i in range(self.num_agents):
            if self.current_top_agents[i] == 1:
                if count_of_top_agents == self.num_agents:
                    # Everyone reached terminal - no reward
                    self.current_rewards[i] = 0
                elif count_of_top_agents > 1:
                    # Multiple but not all agents reached terminal
                    if self.reward_type == 'ILF':
                        # Inverse Linear Fractional: reward divided by n
                        self.current_rewards[i] = self.full_reward / self.num_agents
                    else:  # IQF
                        # Inverse Quadratic Fractional: reward divided by n^2
                        self.current_rewards[i] = self.full_reward / (self.num_agents * self.num_agents)
                else:  # count_of_top_agents == 1
                    # Only one agent reached terminal - full reward
                    self.current_rewards[i] = self.full_reward
            else:
                self.current_rewards[i] = 0

            # Update total rewards
            self.total_reward_per_agent[i] += self.current_rewards[i]

        # Return information about the episode
        return {
            'terminal_occurrences': count_of_top_agents,
            'top_agents': self.current_top_agents.copy(),
            'rewards': self.current_rewards.copy(),
            'done': someone_reached_terminal,
            'rounds': round_count
        }

#===============================================================================
# Q-Learning Agent
#===============================================================================

class Agent:
    def __init__(self, identifier, num_agents, num_positions, state_type):
        self.identifier = identifier
        self.num_positions = num_positions
        self.action_space = [0, 1]  # 0 for "stay", 1 for "move forward"
        self.action_size = len(self.action_space)
        self.total_reward = 0
        self.state_type = state_type

        # Calculate state size based on state representation type
        if state_type == 'Type-A':
            # Type-A: only positions
            self.state_size = self.calculate_state_size(num_agents, num_positions, terminal_states=False)
        else:
            # Type-B: positions + last winners
            self.state_size = self.calculate_state_size(num_agents, num_positions, terminal_states=True)

        # Initialize Q-table with zeros
        self.Q = np.zeros((self.state_size, self.action_size))

    def calculate_state_size(self, num_agents, num_positions, terminal_states=False):
        # Calculate maximum possible state index
        max_positions = [num_positions - 1] * num_agents

        if terminal_states:
            max_terminal_states = [1] * num_agents  # Binary: reached top or not
            return self.encode_state(max_positions, max_terminal_states, num_agents) + 1
        else:
            return self.encode_state(max_positions, None, num_agents) + 1

    def encode_state(self, positions, terminal_states, num_agents):
        # Optimized state encoding (Dec 8, 2025)
        # Type-B memory reduction: 820GB -> 484MB for 10 agents (1,693x improvement)
        if self.state_type == 'Type-A':
            # Type-A: Direct base-3 indexing (already optimal)
            a = int(''.join(str(x) for x in positions), self.num_positions)
            return a
        else:
            # Type-B: Direct combinatorial indexing
            a = int(''.join(str(x) for x in positions), self.num_positions)  # 0 to 3^n-1
            b = int(''.join(str(y) for y in terminal_states), 2)  # 0 to 2^n-1

            # OPTIMIZED: a + b * (3^n) instead of pow(10, c) * b + a
            # Result: 3^n × 2^n states (compact) instead of 10^c × 2^n + 3^n (wasteful)
            return a + b * pow(self.num_positions, num_agents)

#===============================================================================
# Q-Learning Environment
#===============================================================================

class Environment:
    def __init__(self, num_agents, num_positions=3, state_type='Type-A', reward_type='ILF', epsilon_decay_target=EPSILON_DECAY_TARGET):
        self.num_agents = num_agents
        self.num_positions = num_positions
        self.full_reward = FULL_REWARD
        self.state_type = state_type
        self.reward_type = reward_type

        # Hyperparameters
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_INITIAL
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay_target = epsilon_decay_target
        self.decay = None
        # DO NOT set self.decay here - it will be calculated in run_q_learning_experiment

        # Initialize agents
        self.agents = [Agent(i, num_agents, num_positions, state_type) for i in range(num_agents)]

        # Initialize last_top_agents before reset (needed for Type-B state encoding)
        self.last_top_agents = [0] * self.num_agents

        # Q-table tracking for learning analysis
        self.state_visits = {}  # Track visit counts per (state, agent) pair
        self.current_episode = 0  # Track current episode number

        # Terminal position tracking for analysis
        self.terminal_positions_log = []  # [(episode, agent_id, position)]
        self.exclusive_wins_log = []  # [(episode, agent_id)] - only exclusive wins

        # Initialize environment state
        self.reset()

    def reset(self):
        """Reset environment for a new episode."""
        self.positions = [0] * self.num_agents
        self.next_positions = [0] * self.num_agents
        self.actions = [0] * self.num_agents
        self.current_top_agents = [0] * self.num_agents
        # DO NOT reset last_top_agents - it should persist from previous episode for Type-B state!
        # self.last_top_agents is only set in __init__ and update_for_next_episode()
        self.current_rewards = [0] * self.num_agents
        return self.get_state()

    def get_state(self):
        """Get current state based on state representation type."""
        if self.state_type == 'Type-A':
            return self.encode_state(self.positions, None, self.num_agents)
        else:
            return self.encode_state(self.positions, self.last_top_agents, self.num_agents)

    def encode_state(self, positions, terminal_states, num_agents):
        """Encode state using OPTIMIZED method (Dec 8, 2025)."""
        if self.state_type == 'Type-A':
            # Type-A: Direct base-3 indexing (already optimal)
            a = int(''.join(str(x) for x in positions), self.num_positions)
            return a
        else:
            # Type-B: Direct combinatorial indexing
            a = int(''.join(str(x) for x in positions), self.num_positions)  # 0 to 3^n-1
            b = int(''.join(str(y) for y in terminal_states), 2)  # 0 to 2^n-1

            # OPTIMIZED: a + b * (3^n) instead of pow(10, c) * b + a
            # Result: 3^n × 2^n states (compact) instead of 10^c × 2^n + 3^n (wasteful)
            return a + b * pow(self.num_positions, num_agents)

    def update_for_next_episode(self):
        """Update environment state for the next episode."""
        self.last_top_agents = self.current_top_agents.copy()
        self.current_top_agents = [0] * self.num_agents

    def step(self):
        """Execute one step of the environment."""
        # Get current state
        state = self.get_state()

        # Get actions for all agents (epsilon-greedy)
        for i in range(self.num_agents):
            if random.uniform(0, 1) < self.epsilon:  # Using Python's random like legacy
                self.actions[i] = random.choice(self.agents[i].action_space)
            else:
                self.actions[i] = np.argmax(self.agents[i].Q[state])  # Using legacy indexing

        # Execute actions
        someone_reached_terminal = False
        count_of_top_agents = 0

        self.next_positions = self.positions.copy()

        for i in range(self.num_agents):
            if self.actions[i] == 1:  # Move forward
                self.next_positions[i] += 1

                if self.next_positions[i] == self.num_positions - 1:
                    someone_reached_terminal = True
                    count_of_top_agents += 1
                    self.current_top_agents[i] = 1
                    # Log terminal position for analysis
                    self.terminal_positions_log.append((self.current_episode, i, self.next_positions[i]))

        # Calculate rewards based on reward type
        for i in range(self.num_agents):
            if self.current_top_agents[i] == 1:
                if count_of_top_agents == self.num_agents:
                    # Everyone reached terminal - no reward
                    self.current_rewards[i] = 0
                elif count_of_top_agents > 1:
                    # Multiple but not all agents reached terminal
                    if self.reward_type == 'ILF':
                        # Inverse Linear Fractional reward
                        self.current_rewards[i] = self.full_reward / self.num_agents
                    else:  # 'IQF'
                        # Inverse Quadratic Fractional reward
                        self.current_rewards[i] = self.full_reward / (self.num_agents * self.num_agents)
                else:  # count_of_top_agents == 1
                    # Only one agent reached terminal - full reward
                    self.current_rewards[i] = self.full_reward
                    # Log exclusive win (only one winner)
                    self.exclusive_wins_log.append((self.current_episode, i))
            else:
                self.current_rewards[i] = 0

            # Update total rewards
            self.agents[i].total_reward += self.current_rewards[i]

        # Get next state
        next_state = self.encode_state(self.next_positions, self.last_top_agents, self.num_agents)

        # Update Q-values and track state visits
        for i in range(self.num_agents):
            old_value = self.agents[i].Q[state, self.actions[i]]
            next_max = np.max(self.agents[i].Q[next_state])  # Legacy indexing

            new_value = (1 - self.alpha) * old_value + self.alpha * (self.current_rewards[i] + self.gamma * next_max)
            self.agents[i].Q[state, self.actions[i]] = new_value

            # Track state visit for Q-table analysis
            state_key = (state, i)  # (state_index, agent_id)
            if state_key not in self.state_visits:
                self.state_visits[state_key] = {
                    'count': 0,
                    'last_episode': -1,
                    'last_q_value': 0,
                    'last_action': -1,
                    'was_terminal': False
                }

            self.state_visits[state_key]['count'] += 1
            self.state_visits[state_key]['last_episode'] = self.current_episode
            self.state_visits[state_key]['last_q_value'] = new_value
            self.state_visits[state_key]['last_action'] = self.actions[i]
            self.state_visits[state_key]['was_terminal'] = someone_reached_terminal

        # Update positions
        self.positions = self.next_positions.copy()

        # Return information about the step
        return {
            'terminal_occurrences': sum(self.current_top_agents),
            'top_agents': self.current_top_agents.copy(),
            'rewards': self.current_rewards.copy(),
            'done': someone_reached_terminal
        }
