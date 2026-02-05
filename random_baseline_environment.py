"""
Random Policy Baseline Environment for ALT Metrics experiments.

This module provides the random policy baseline environment:
- RandomPolicyEnvironment: Random action selection for null hypothesis testing
- Establishes statistical baseline for ALT metrics
- Compatible with existing experiment framework

For backwards compatibility, this re-exports from environment.py.
In future versions, the RandomPolicyEnvironment class will be fully migrated here.
"""

from environment import RandomPolicyEnvironment

__all__ = ['RandomPolicyEnvironment']
