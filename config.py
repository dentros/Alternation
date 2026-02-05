"""
Configuration and constants for ALT Metrics experiments.
"""

import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Suppress font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

#===============================================================================
# Matplotlib Settings for Publication-Ready Figures
#===============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 11
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 12
rcParams['figure.figsize'] = (7, 3.5)  # Optimized for 2-column format
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

#===============================================================================
# Color Palette
#===============================================================================

COLOR_PALETTE = {
    # ALT measures
    'FALT': '#e41a1c',    # red
    'EALT': '#377eb8',    # blue
    'qFALT': '#4daf4a',   # green
    'qEALT': '#984ea3',   # purple
    'CALT': '#ff7f00',    # orange
    'AALT': '#a65628',    # brown

    # Traditional metrics
    'Efficiency': '#e377c2',    # pink
    'Reward_Fairness': '#bcbd22', # yellow-green
    'TT_Fairness': '#17becf',   # cyan
    'Fairness': '#7f7f7f',      # gray

    # RP measures (for internal use only)
    'RP_avg': '#8c564b',  # dark brown
    'AWE_avg': '#c49c94', # light brown
    'WPE_avg': '#e377c2', # pink

    # State + Reward combinations
    'Type-A_IQF': '#9467bd',  # purple
    'Type-A_ILF': '#1f77b4',   # blue
    'Type-B_IQF': '#d62728',  # red
    'Type-B_ILF': '#2ca02c',   # green

    # Additional colors for random baselines
    'Random': '#000000',  # black
    'Perfect': '#ff0000', # bright red
}

#===============================================================================
# Directory Setup
#===============================================================================

# Detect Google Colab
try:
    from google.colab import drive
    print("Google Colab detected - mounting Google Drive for persistent storage")
    drive.mount('/content/drive')
    base_results_dir = '/content/drive/MyDrive/alt_metrics_results'
    IN_COLAB = True
except ImportError:
    print("Not running in Google Colab - using local directory")
    # Use absolute path relative to this config file (always point to project root/results)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(config_dir)  # Go up from src/ to project root
    base_results_dir = os.path.join(project_root, 'results')
    IN_COLAB = False

# Ensure base directory exists
os.makedirs(base_results_dir, exist_ok=True)

#===============================================================================
# Directory Structure Factory
#===============================================================================

def get_run_directory(experiment_type='theory', base_episodes=100, agent_type='ql', agent_counts=None, timestamp=None):
    """
    Create run directory with all metadata in folder name.

    Structure:
        results/
        ├── run_theory_base100_ql_agents[2,3,5]_20251127_143022/
        ├── run_theory_base100_dqn_agents[2,3,5]_20251127_150000/
        └── run_adaptive_base1000_ql_agents[5,8,10]_20251128_120000/

    Args:
        experiment_type: 'theory' or 'adaptive'
        base_episodes: 100, 1000, 5000, 10000, etc.
        agent_type: 'ql' (Q-learning), 'dqn' (Deep Q-Network), etc.
                   Note: 'random' baseline always runs for benchmarking (not in name)
        agent_counts: List of agent counts [2, 3, 5, 8, 10] (optional)
        timestamp: Custom timestamp string (optional, auto-generated if None)

    Returns:
        Path to run directory
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build directory name
    dir_name = f"run_{experiment_type}_base{base_episodes}_{agent_type}"

    if agent_counts:
        agents_str = ','.join(map(str, sorted(agent_counts)))
        dir_name += f"_agents[{agents_str}]"

    dir_name += f"_{timestamp}"

    run_dir = os.path.join(base_results_dir, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def create_subdirectories(base_dir):
    """
    Create standard subdirectories for experiment results.

    Args:
        base_dir: Base directory for this experiment

    Returns:
        Dictionary of subdirectory paths
    """
    subdirs = {
        'main': os.path.join(base_dir, 'main_figures'),
        'alt': os.path.join(base_dir, 'alt_metrics'),
        'random': os.path.join(base_dir, 'random_baselines'),
        'comparison': os.path.join(base_dir, 'comparison_metrics'),
        'alt_ratio': os.path.join(base_dir, 'alt_ratio_analysis'),
        'tables': os.path.join(base_dir, 'data_tables'),
        'computation_times': os.path.join(base_dir, 'computation_times'),
        'learning_phases': os.path.join(base_dir, 'learning_phases'),
        '3d_analysis': os.path.join(base_dir, '3d_analysis'),
        'qtable_analysis': os.path.join(base_dir, 'qtable_analysis'),
        'agent_performance': os.path.join(base_dir, 'agent_performance_analysis'),
        'checkpoints': os.path.join(base_dir, 'checkpoints')
    }

    # Create all subdirectories
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return subdirs


# Default directory setup (for backwards compatibility)
# These will be overridden by experiments that use get_experiment_dir()
base_dir = base_results_dir  # Legacy variable name
figure_dirs = {
    'main': os.path.join(base_dir, 'main_figures'),
    'alt': os.path.join(base_dir, 'alt_metrics'),
    'random': os.path.join(base_dir, 'random_baselines'),
    'comparison': os.path.join(base_dir, 'comparison_metrics'),
    'alt_ratio': os.path.join(base_dir, 'alt_ratio_analysis'),
    'tables': os.path.join(base_dir, 'data_tables'),
    'computation_times': os.path.join(base_dir, 'computation_times'),
    'learning_phases': os.path.join(base_dir, 'learning_phases'),
    '3d_analysis': os.path.join(base_dir, '3d_analysis'),
    'qtable_analysis': os.path.join(base_dir, 'qtable_analysis'),
    'agent_performance': os.path.join(base_dir, 'agent_performance_analysis')
}

# Default paths for backwards compatibility (ONLY used if get_run_directory not called)
# DO NOT create directories here - let run_experiments() handle it using create_subdirectories()
# NOTE: This is a temporary placeholder - will be updated by run_experiments() to point inside run directory
checkpoint_dir = os.path.join(base_results_dir, '__temp_checkpoints__')  # Will be overridden by run_experiments()

print(f"Base results directory: {os.path.abspath(base_results_dir)}")
print(f"Use get_experiment_dir(experiment_type, base_episodes, num_agents) for organized structure")

#===============================================================================
# Global Hyperparameters
#===============================================================================

# Episode configuration
BASE_EPISODES = 1000

# Agent configuration for testing
TEST_AGENT_COUNTS = [2, 3, 5, 8, 10]

# Reward structure
FULL_REWARD = 100.0  # r_high

# Q-Learning hyperparameters
ALPHA = 0.3          # Learning rate
GAMMA = 0.999        # Discount factor
EPSILON_INITIAL = 0.9
EPSILON_MIN = 0.004
EPSILON_DECAY_TARGET = 0.75  # Reach epsilon_min at 75% of episodes

# Multiprocessing configuration
USE_MULTIPROCESSING = True  # Set to False to disable parallel processing
NUM_PROCESSES = 28
