"""
Q-Learning Agent for ALT Metrics experiments.

This module provides the Q-learning Agent class:
- Agent: Tabular Q-learning agent with Q-table
- State encoding for Type-A and Type-B representations
- Compatible with existing experiment framework
"""

import numpy as np

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
