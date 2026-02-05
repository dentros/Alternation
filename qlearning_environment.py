"""
Q-Learning Environment for ALT Metrics experiments.

This module provides the Q-learning Environment class:
- Environment: Multi-agent Q-learning environment with epsilon-greedy
- Supports both state types (Type-A, Type-B)
- Supports both reward types (ILF, IQF)
- Compatible with existing experiment framework

For backwards compatibility, this re-exports from environment.py.
In future versions, the Environment class will be fully migrated here.
"""

from environment import Environment

__all__ = ['Environment']
