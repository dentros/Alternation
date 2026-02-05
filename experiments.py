"""
Experiments module for ALT Metrics experiments.

This module contains:
- Checkpoint management (save/load/update)
- Random policy simulation
- Q-learning experiment runners (theory-based and adaptive)
- Complete metrics calculation wrapper
- Multiprocessing experiment workers
"""

import numpy as np
import pickle
import os
import time
import glob
import random
import gc
import re
import multiprocessing as mp
from tqdm import tqdm
try:
    from IPython.display import display, Javascript
except ImportError:
    display = None  # Not in Colab/Jupyter environment
    Javascript = None
import filelock

import config
from config import (
    IN_COLAB, FULL_REWARD, BASE_EPISODES, USE_MULTIPROCESSING, NUM_PROCESSES
)
from metrics import (
    calculate_episodes_theory_based,
    compute_alt_metrics, compute_rp_metrics,
    compute_efficiency, compute_reward_fairness, compute_tt_fairness, compute_fairness,
    calculate_alt_ratio
)
from environment import RandomPolicyEnvironment, Environment

# DQN imports (optional - only loaded if DQN experiments are run)
try:
    from dqn_environment import DQNEnvironment
    from dqn_agents import DQNAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("[WARNING] DQN not available - install PyTorch to enable DQN experiments")

#===============================================================================
# Checkpoint Management
#===============================================================================

def keep_colab_alive():
    """Keep Colab session alive with more frequent pings."""
    if IN_COLAB and display is not None and Javascript is not None:
        display(Javascript('''
            function ClickConnect(){
                console.log("Keeping Colab alive...");
                document.querySelector("colab-toolbar-button#connect").click()
            }
            setInterval(ClickConnect, 30000);
        '''))
        print("KeepAlive activated: Colab will stay connected")

def save_result(result):
    """Save result with complete data for reproducibility."""
    if 'is_random_baseline' in result and result['is_random_baseline']:
        filename = f"random_{result['num_agents']}agents_{result['reward_type']}.pkl"
    else:
        filename = f"result_{result['num_agents']}agents_{result['state_type']}_{result['reward_type']}.pkl"
    filepath = os.path.join(config.checkpoint_dir, filename)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        update_master_checkpoint(result)
        return True
    except Exception as e:
        print(f"Error saving result: {e}")
        return False

def update_master_checkpoint(new_result=None):
    """Update master checkpoint file with file locking to prevent corruption."""
    lock_path = os.path.join(config.checkpoint_dir, 'checkpoint.lock')
    checkpoint_file = os.path.join(config.checkpoint_dir, 'experiment_checkpoint.pkl')

    # Clean up stale lock files (older than 1 hour)
    if os.path.exists(lock_path):
        lock_age = time.time() - os.path.getmtime(lock_path)
        if lock_age > 3600:  # 1 hour timeout
            print(f"[WARNING] Removing stale lock file (age: {lock_age/3600:.1f} hours)")
            try:
                os.remove(lock_path)
            except Exception as e:
                print(f"[ERROR] Could not remove stale lock: {e}")

    with filelock.FileLock(lock_path, timeout=60):
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    results_list = checkpoint.get('results_list', [])
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                results_list = []
        else:
            results_list = []

        if new_result is not None:
            exists = False
            for i, r in enumerate(results_list):
                # Check if result with same parameters exists
                if 'is_random_baseline' in new_result and new_result['is_random_baseline']:
                    if ('is_random_baseline' in r and r.get('is_random_baseline') and
                        r['num_agents'] == new_result['num_agents'] and
                        r['reward_type'] == new_result['reward_type']):
                        results_list[i] = new_result
                        exists = True
                        break
                else:
                    if (r['num_agents'] == new_result['num_agents'] and
                        r.get('state_type', '') == new_result.get('state_type', '') and
                        r['reward_type'] == new_result['reward_type']):
                        results_list[i] = new_result
                        exists = True
                        break

            if not exists:
                results_list.append(new_result)

        checkpoint = {
            'results_list': results_list,
            'timestamp': time.time(),
            'version': '2.0'
        }

        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            return results_list
        except Exception as e:
            print(f"Error saving master checkpoint: {e}")
            return results_list

def load_results():
    """Load results from checkpoints - robust fallback to individual files."""
    lock_path = os.path.join(config.checkpoint_dir, 'checkpoint.lock')
    master_file = os.path.join(config.checkpoint_dir, 'experiment_checkpoint.pkl')

    with filelock.FileLock(lock_path, timeout=60):
        # TRY 1: Master checkpoint (optimization - fast path)
        if os.path.exists(master_file):
            try:
                with open(master_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    if 'results_list' in checkpoint:
                        results_list = checkpoint['results_list']
                        print(f"Loaded {len(results_list)} experiments from master checkpoint")
                        return results_list
            except Exception as e:
                print(f"Warning: Could not load master checkpoint: {e}")

        # TRY 2: Individual files (ALWAYS try if master missing/corrupt)
        # Use os.listdir() instead of glob to avoid issues with brackets in directory names
        results_list = []

        if os.path.exists(config.checkpoint_dir):
            all_files = os.listdir(config.checkpoint_dir)

            # Filter for result and random checkpoint files
            checkpoint_files = [f for f in all_files
                               if (f.startswith('result_') or f.startswith('random_'))
                               and f.endswith('.pkl')]

            if checkpoint_files:
                print(f"Loading from {len(checkpoint_files)} individual checkpoint files...")
                for filename in checkpoint_files:
                    filepath = os.path.join(config.checkpoint_dir, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            result = pickle.load(f)
                            results_list.append(result)
                    except Exception as e:
                        print(f"Warning: Could not load {filename}: {e}")

        if results_list:
            print(f"✓ Loaded {len(results_list)} experiments from individual files")
            # Rebuild master checkpoint for next time (skip for now to avoid lock issues)
            # update_master_checkpoint()
            return results_list

    print("No checkpoint files found - starting fresh")
    return []

def is_experiment_completed(results_list, num_agents, state_type, reward_type, is_random=False):
    """Check if experiment already completed."""
    for r in results_list:
        if is_random:
            if ('is_random_baseline' in r and r.get('is_random_baseline') and
                r['num_agents'] == num_agents and
                r['reward_type'] == reward_type):
                return True
        else:
            if (r['num_agents'] == num_agents and
                r.get('state_type', '') == state_type and
                r['reward_type'] == reward_type and
                not r.get('is_random_baseline', False)):
                return True
    return False

def delete_all_results():
    """Delete entire results directory and recreate it."""
    import shutil

    try:
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            print(f"Deleted entire results directory: {base_dir}")

        # Recreate all directories
        os.makedirs(base_dir, exist_ok=True)
        for dir_path in figure_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        print("All results deleted and directories recreated")
    except Exception as e:
        print(f"Error deleting results: {e}")

#===============================================================================
# Checkpoint Continuation Functions
#===============================================================================

def find_compatible_runs(base_episodes, experiment_type, agent_type):
    """Find existing run directories with matching configuration.

    Args:
        base_episodes: BASE_EPISODES value (e.g., 1000)
        experiment_type: 'theory' or 'adaptive'
        agent_type: 'ql' or 'dqn'

    Returns:
        List of dicts with:
        - dir_name: directory name
        - full_path: full path to directory
        - agent_counts: list of agent counts found in checkpoints
        - checkpoint_count: number of checkpoint files
    """
    results = []
    base_dir_path = config.base_results_dir

    # Pattern: run_{experiment_type}_base{episodes}_{agent_type}_agents[...]_{timestamp}
    pattern = f"run_{experiment_type}_base{base_episodes}_{agent_type}_agents*"

    search_pattern = os.path.join(base_dir_path, pattern)
    matched_dirs = glob.glob(search_pattern)

    for dir_path in matched_dirs:
        if os.path.isdir(dir_path):
            checkpoint_dir = os.path.join(dir_path, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                # Count checkpoint files (exclude master checkpoint)
                # Escape the checkpoint_dir path for glob (handles [ ] characters in directory names)
                escaped_checkpoint_dir = glob.escape(checkpoint_dir)
                checkpoints = glob.glob(os.path.join(escaped_checkpoint_dir, '*.pkl'))
                checkpoint_count = len([f for f in checkpoints
                                       if 'experiment_checkpoint' not in f and 'checkpoint.lock' not in f])

                # Extract agent counts from checkpoint filenames
                agent_counts = set()
                for f in checkpoints:
                    match = re.search(r'(\d+)agents', os.path.basename(f))
                    if match:
                        agent_counts.add(int(match.group(1)))

                if checkpoint_count > 0:  # Only include directories with actual checkpoints
                    results.append({
                        'dir_name': os.path.basename(dir_path),
                        'full_path': dir_path,
                        'agent_counts': sorted(agent_counts),
                        'checkpoint_count': checkpoint_count,
                    })

    # Sort by most recent (timestamp in directory name)
    results.sort(key=lambda x: x['dir_name'], reverse=True)
    return results

def rename_run_directory(run_info, new_agent_counts):
    """Rename directory to include updated agent counts.

    Args:
        run_info: Dict from find_compatible_runs()
        new_agent_counts: List of agent counts to include

    Returns:
        New directory name (basename only)

    Example:
        Old: run_theory_base1000_ql_agents[2,3,5,8]_20251205_221651
        New: run_theory_base1000_ql_agents[2,3,5,8,10]_20251205_221651
    """
    from datetime import datetime

    old_path = run_info['full_path']
    old_name = run_info['dir_name']

    # Extract timestamp from old name
    match = re.search(r'_(\d{8}_\d{6})$', old_name)
    timestamp = match.group(1) if match else datetime.now().strftime('%Y%m%d_%H%M%S')

    # Extract config parts
    match = re.search(r'run_([^_]+)_base(\d+)_([^_]+)_agents', old_name)
    if not match:
        print(f"Warning: Could not parse directory name: {old_name}")
        return old_name

    experiment_type = match.group(1)
    base_episodes = match.group(2)
    agent_type = match.group(3)

    # Build new name with ALL agent counts (merge old + new)
    all_agents = sorted(set(run_info['agent_counts'] + list(new_agent_counts)))
    agents_str = ','.join(map(str, all_agents))
    new_name = f"run_{experiment_type}_base{base_episodes}_{agent_type}_agents[{agents_str}]_{timestamp}"

    # Rename directory
    new_path = os.path.join(os.path.dirname(old_path), new_name)
    try:
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name}")
        print(f"     to: {new_name}")
    except Exception as e:
        print(f"Error renaming directory: {e}")
        return old_name

    return new_name

def copy_run_directory(run_info, new_agent_counts):
    """Copy checkpoints to NEW directory with NEW timestamp.

    Args:
        run_info: Dict from find_compatible_runs()
        new_agent_counts: List of agent counts to include

    Returns:
        New directory name (basename only)

    Example:
        Old: run_theory_base1000_ql_agents[2,3,5,8]_20251205_221651
        New: run_theory_base1000_ql_agents[2,3,5,8,10]_20251207_143022
                                                         ^^^^^^^^^^^^^^ NEW timestamp
    """
    import shutil
    from datetime import datetime

    old_path = run_info['full_path']
    old_name = run_info['dir_name']
    old_checkpoint_dir = os.path.join(old_path, 'checkpoints')

    # Extract config parts
    match = re.search(r'run_([^_]+)_base(\d+)_([^_]+)_agents', old_name)
    if not match:
        print(f"Warning: Could not parse directory name: {old_name}")
        return old_name

    experiment_type = match.group(1)
    base_episodes = match.group(2)
    agent_type = match.group(3)

    # Build new name with ALL agent counts (merge old + new) and NEW timestamp
    all_agents = sorted(set(run_info['agent_counts'] + list(new_agent_counts)))
    agents_str = ','.join(map(str, all_agents))
    new_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_name = f"run_{experiment_type}_base{base_episodes}_{agent_type}_agents[{agents_str}]_{new_timestamp}"

    # Create new directory
    new_path = os.path.join(os.path.dirname(old_path), new_name)
    new_checkpoint_dir = os.path.join(new_path, 'checkpoints')

    try:
        # Create directory structure
        os.makedirs(new_checkpoint_dir, exist_ok=True)

        # Copy ALL checkpoint files from old to new
        if os.path.exists(old_checkpoint_dir):
            checkpoint_files = glob.glob(os.path.join(glob.escape(old_checkpoint_dir), '*.pkl'))
            checkpoint_files = [f for f in checkpoint_files if 'experiment_checkpoint' not in f]

            copied_count = 0
            for src_file in checkpoint_files:
                dest_file = os.path.join(new_checkpoint_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
                copied_count += 1

            print(f"Created: {new_name}")
            print(f"Copied {copied_count} checkpoint files from: {old_name}")
        else:
            print(f"Warning: No checkpoints found in {old_name}")

    except Exception as e:
        print(f"Error copying directory: {e}")
        return None

    return new_name

def run_experiments_continue(run_dir, experiment_type='theory', agent_type='ql'):
    """Continue experiments in existing run directory.

    This function reuses the existing run directory and only runs missing experiments.

    Args:
        run_dir: Path to existing run directory (absolute or relative)
        experiment_type: 'theory' or 'adaptive'
        agent_type: 'ql' or 'dqn'

    Returns:
        List of all results (existing + newly completed)
    """
    from config import TEST_AGENT_COUNTS, BASE_EPISODES, create_subdirectories

    # Ensure run_dir is absolute path
    if not os.path.isabs(run_dir):
        run_dir = os.path.abspath(run_dir)

    print(f"\n{'='*80}")
    print(f"CONTINUING EXPERIMENTS IN EXISTING DIRECTORY")
    print(f"{'='*80}")
    print(f"Run directory: {run_dir}")

    # Create subdirectories if missing (shouldn't be needed, but safe)
    subdirs = create_subdirectories(run_dir)

    # UPDATE config module variables to point to this directory
    config.checkpoint_dir = subdirs['checkpoints']
    config.figure_dirs = subdirs
    config.base_dir = run_dir

    print(f"Checkpoints: {config.checkpoint_dir}")

    # Load EXISTING results from this directory
    results_list = load_results()
    print(f"\nLoaded {len(results_list)} existing experiment results")

    # Show what's already completed
    if results_list:
        print("\nAlready completed:")
        for r in results_list:
            if r.get('is_random_baseline'):
                print(f"  - Random baseline: {r['num_agents']} agents, {r['reward_type']}")
            else:
                print(f"  - Q-Learning: {r['num_agents']} agents, {r.get('state_type', 'N/A')}, {r['reward_type']}")

    # The rest is IDENTICAL to run_experiments() - generate experiments_to_run list, run multiprocessing
    # We'll copy the remaining code from run_experiments() here

    agent_counts = TEST_AGENT_COUNTS
    state_types = ['Type-A', 'Type-B']
    reward_types = ['ILF', 'IQF']

    # Verify DQN availability if requested
    if agent_type == 'dqn' and not DQN_AVAILABLE:
        raise ImportError("DQN requested but PyTorch not available. Install with: pip install torch")

    # Print what's already complete
    print("\n" + "="*50)
    print("COMPLETED EXPERIMENTS")
    print("="*50)
    for r in results_list:
        agent_count = r['num_agents']
        state = r.get('state_type', 'Random')
        reward = r['reward_type']
        is_rand = r.get('is_random_baseline', False)
        episodes = r.get('num_episodes', 'N/A')
        if is_rand:
            print(f"  ✓ {agent_count} agents, Random Baseline, {reward} ({episodes} episodes)")
        else:
            print(f"  ✓ {agent_count} agents, {state}, {reward} ({episodes} episodes)")

    # Generate experiments to run list (skip completed ones)
    experiments_to_run = []

    for num_agents in agent_counts:
        for state_type in state_types:
            for reward_type in reward_types:
                if not is_experiment_completed(results_list, num_agents, state_type, reward_type):
                    experiments_to_run.append((
                        num_agents, state_type, reward_type, experiment_type, agent_type
                    ))

    # Random baselines
    for num_agents in agent_counts:
        for reward_type in reward_types:
            if not is_experiment_completed(results_list, num_agents, None, reward_type, is_random=True):
                experiments_to_run.append((
                    num_agents, None, reward_type, experiment_type, agent_type, True
                ))

    # Print what needs to be run
    print("\n" + "="*50)
    print("MISSING EXPERIMENTS")
    print("="*50)
    if not experiments_to_run:
        print("  (none - all experiments complete!)")
        print(f"\nNo new experiments to run! All {len(results_list)} experiments already completed.")
        return results_list

    for exp in experiments_to_run:
        if len(exp) > 5 and exp[5]:  # is_random flag
            print(f"  ✗ {exp[0]} agents, Random Baseline, {exp[2]}")
        else:
            print(f"  ✗ {exp[0]} agents, {exp[1]}, {exp[2]}")

    print(f"\nTotal missing: {len(experiments_to_run)} experiments")

    # Run experiments with multiprocessing
    if USE_MULTIPROCESSING and len(experiments_to_run) > 1:
        print(f"\nRunning {len(experiments_to_run)} experiments with {NUM_PROCESSES} parallel processes...")
        with mp.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(checkpoint_dir,)) as pool:
            pool.map(run_single_experiment_worker, experiments_to_run)
    else:
        print(f"\nRunning {len(experiments_to_run)} experiments sequentially...")
        for exp in experiments_to_run:
            run_single_experiment_worker(exp)

    # Reload results to include newly completed experiments
    results_list = load_results()
    print(f"\nAll experiments completed! Total results: {len(results_list)}")

    return results_list

#===============================================================================
# Random Policy Simulation
#===============================================================================

def simulate_random_policy(num_agents, num_episodes, reward_type='ILF'):
    """Simulate agents with purely random policies and track metrics."""
    print(f"Simulating random policy: {num_agents} agents, {num_episodes} episodes, {reward_type}...")

    env = RandomPolicyEnvironment(num_agents, reward_type=reward_type)

    # Initialize tracking variables
    top_agents_per_episode = []
    terminal_occurrences_per_episode = []
    unique_winners_per_episode = []
    round_counts = []

    # Progress tracking
    for _ in tqdm(range(num_episodes), desc="Random Episodes", ncols=100):
        # Run episode
        step_result = env.step()

        # Track results
        top_agents_per_episode.append(step_result['top_agents'])
        terminal_occurrences_per_episode.append(step_result['terminal_occurrences'])
        round_counts.append(step_result['rounds'])

        # Track unique winners
        if sum(step_result['top_agents']) == 1:
            for i in range(num_agents):
                if step_result['top_agents'][i] == 1:
                    unique_winners_per_episode.append(i)
                    break

        # Update environment for next episode
        env.update_for_next_episode()

    # Calculate metrics
    alt_metrics, beta_values, alt_computation_times = compute_alt_metrics(
        num_episodes, num_agents, terminal_occurrences_per_episode, top_agents_per_episode
    )

    # RP metrics (calculated but not displayed for Paper 1)
    rp_metrics = compute_rp_metrics(
        num_episodes, num_agents, top_agents_per_episode
    )

    # Traditional metrics
    efficiency_result = compute_efficiency(
        num_episodes, num_agents, env.total_reward_per_agent, env.full_reward
    )

    reward_fairness_result = compute_reward_fairness(
        env.total_reward_per_agent
    )

    tt_fairness_result = compute_tt_fairness(
        top_agents_per_episode, num_agents, num_episodes
    )

    fairness_result = compute_fairness(
        unique_winners_per_episode, num_agents
    )

    # Compile results
    metrics = {
        **alt_metrics,
        'RP_avg': rp_metrics['RP_avg'],
        'AWE_avg': rp_metrics['AWE_avg'],
        'WPE_avg': rp_metrics['WPE_avg'],
        'Efficiency': efficiency_result['Efficiency'],
        'Reward_Fairness': reward_fairness_result['Reward_Fairness'],
        'TT_Fairness': tt_fairness_result['TT_Fairness'],
        'Fairness': fairness_result['Fairness']
    }

    # Add ALT Ratio calculations
    result = {
        'num_agents': num_agents,
        'num_episodes': num_episodes,
        'reward_type': reward_type,
        'metrics': metrics,
        'beta_values': beta_values,
        'rp_values': {
            'AWE_per_agent': rp_metrics['AWE_per_agent'],
            'WPE_per_agent': rp_metrics['WPE_per_agent'],
            'RP_per_agent': rp_metrics['RP_per_agent']
        },
        'is_random_baseline': True,
        'avg_rounds': sum(round_counts) / len(round_counts) if round_counts else 0
    }

    # Calculate ALT Ratio
    alt_ratios, estimated_agents, percentages = calculate_alt_ratio(result)
    result['alt_ratios'] = alt_ratios
    result['estimated_agents'] = estimated_agents
    result['percentages'] = percentages

    # Save results
    success = save_result(result)

    if success:
        print(f"[OK] Completed random baseline: {num_agents} agents, {reward_type}")
    else:
        print(f"⚠ Completed but failed to save: {num_agents} agents, {reward_type}")

    return result

#===============================================================================
# Q-Learning Experiments
#===============================================================================

def run_q_learning_experiment(num_agents, num_episodes, state_type='Type-A', reward_type='ILF', decay_schedule='linear'):
    from datetime import datetime
    def debug_log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG {timestamp}] {msg}", flush=True)
    
    debug_log(f"=" * 80)
    debug_log(f"Q-LEARNING START: {num_agents} agents, {num_episodes} episodes, {state_type}, {reward_type}")
    debug_log(f"=" * 80)
    """Run Q-learning experiment with tracking of metrics."""
    print(f"Running Q-learning: {num_agents} agents, {num_episodes} episodes, {state_type}, {reward_type}...")

    env = Environment(num_agents, num_positions=3, state_type=state_type, reward_type=reward_type)

    # Calculate target episode for linear schedule
    target_episode = int(env.epsilon_decay_target * num_episodes)

    # Set decay for other code that accesses it
    env.decay = (0.9 - env.epsilon_min) / target_episode
    print(f"Will reach epsilon_min={env.epsilon_min} at episode {target_episode} (schedule: {decay_schedule})")

    # Initialize tracking variables
    top_agents_per_episode = []
    terminal_occurrences_per_episode = []
    unique_winners_per_episode = []
    round_counts = []
    epsilon_per_episode = []  # Track epsilon value per episode

    # Progress bar for tracking
    for episode in tqdm(range(num_episodes), desc="Episodes", ncols=100):
        # Update current episode for Q-table tracking
        env.current_episode = episode

        if decay_schedule == 'exponential':
            # Exponential epsilon decay
            if env.epsilon > env.epsilon_min:
                env.epsilon = max(env.epsilon_min, env.epsilon * env.decay)
        else:
            # Direct linear interpolation to epsilon_min by target_episode
            if episode < target_episode:
                progress = episode / target_episode  # 0.0 to 1.0
                env.epsilon = 0.9 - progress * (0.9 - env.epsilon_min)
            else:
                env.epsilon = env.epsilon_min

        # Reset environment
        env.reset()
        episode_done = False
        round_count = 0
        max_rounds = 100  # Safety limit to prevent infinite loops

        # Run episode
        while not episode_done and round_count < max_rounds:
            step_result = env.step()
            episode_done = step_result['done']
            round_count += 1

        # Track results
        top_agents_per_episode.append(step_result['top_agents'])
        terminal_occurrences_per_episode.append(step_result['terminal_occurrences'])
        round_counts.append(round_count)
        epsilon_per_episode.append(env.epsilon)  # Track epsilon value

        # Track unique winners
        if sum(step_result['top_agents']) == 1:
            for i in range(num_agents):
                if step_result['top_agents'][i] == 1:
                    unique_winners_per_episode.append(i)
                    break

        # Update environment for next episode
        env.update_for_next_episode()

        # Print epsilon progress at 10% intervals
        if episode > 0 and episode % (num_episodes // 10) == 0:
            print(f"Episode {episode}/{num_episodes} ({episode/num_episodes:.1%}): epsilon = {env.epsilon:.4f}")

    # Get total rewards
    total_reward_per_agent = [agent.total_reward for agent in env.agents]

    # Calculate comprehensive metrics using wrapper
    result = calculate_all_metrics(
        num_agents, num_episodes, state_type, reward_type,
        top_agents_per_episode, terminal_occurrences_per_episode,
        unique_winners_per_episode, total_reward_per_agent, epsilon_per_episode
    )

    # Add Q-table data for analysis
    # Convert state_visits format from {(state, agent): data} to {agent: {state: count}}
    state_visits_by_agent = {}
    for (state_idx, agent_id), visit_data in env.state_visits.items():
        if agent_id not in state_visits_by_agent:
            state_visits_by_agent[agent_id] = {}
        state_visits_by_agent[agent_id][state_idx] = visit_data['count']

    result['state_visits'] = state_visits_by_agent
    result['terminal_positions_log'] = env.terminal_positions_log
    result['exclusive_wins_log'] = env.exclusive_wins_log
    result['q_tables'] = {i: agent.Q for i, agent in enumerate(env.agents)}
    result['avg_rounds'] = sum(round_counts) / len(round_counts) if round_counts else 0

    # Calculate ALT Ratio
    alt_ratios, estimated_agents, percentages = calculate_alt_ratio(result)
    result['alt_ratios'] = alt_ratios
    result['estimated_agents'] = estimated_agents
    result['percentages'] = percentages

    # Save results
    success = save_result(result)

    if success:
        print(f"[OK] Completed Q-learning: {num_agents} agents, {state_type}, {reward_type}")
    else:
        print(f"⚠ Completed but failed to save: {num_agents} agents, {state_type}, {reward_type}")

    return result

def run_adaptive_q_learning_experiment(num_agents, state_type='Type-A', reward_type='ILF',
                                      alt_thresholds=None, patience=5000, max_episodes=150000):
    """Run Q-learning until ALT thresholds are met or max episodes reached."""
    print(f"Running adaptive Q-learning: {num_agents} agents, {state_type}, {reward_type}...")

    if alt_thresholds is None:
        alt_thresholds = {
            'CALT': 0.6,
            'FALT': 0.7,
            'EALT': 0.7,
            'qFALT': 0.5,
            'qEALT': 0.5,
            'AALT': 0.4
        }

    env = Environment(num_agents, num_positions=3, state_type=state_type, reward_type=reward_type)

    # Exponential decay for adaptive
    env.decay = 0.9995

    # Initialize tracking
    top_agents_per_episode = []
    terminal_occurrences_per_episode = []
    unique_winners_per_episode = []
    round_counts = []
    epsilon_per_episode = []

    converged = False
    patience_counter = 0
    episode = 0

    print(f"Max episodes: {max_episodes}, Patience: {patience}")
    print(f"Thresholds: {alt_thresholds}")

    while episode < max_episodes and not converged:
        # Update current episode
        env.current_episode = episode

        # Exponential epsilon decay
        if env.epsilon > env.epsilon_min:
            env.epsilon = max(env.epsilon_min, env.epsilon * env.decay)

        # Run episode
        env.reset()
        episode_done = False
        round_count = 0
        max_rounds = 100

        while not episode_done and round_count < max_rounds:
            step_result = env.step()
            episode_done = step_result['done']
            round_count += 1

        # Track results
        top_agents_per_episode.append(step_result['top_agents'])
        terminal_occurrences_per_episode.append(step_result['terminal_occurrences'])
        round_counts.append(round_count)
        epsilon_per_episode.append(env.epsilon)

        if sum(step_result['top_agents']) == 1:
            for i in range(num_agents):
                if step_result['top_agents'][i] == 1:
                    unique_winners_per_episode.append(i)
                    break

        env.update_for_next_episode()
        episode += 1

        # Check convergence every 1000 episodes
        if episode % 1000 == 0 and episode >= num_agents:
            alt_metrics, _, _ = compute_alt_metrics(
                episode, num_agents, terminal_occurrences_per_episode, top_agents_per_episode
            )

            thresholds_met = all(alt_metrics[k] >= v for k, v in alt_thresholds.items())

            if thresholds_met:
                patience_counter += 1
                if patience_counter >= patience:
                    converged = True
                    print(f"Converged at episode {episode}!")
            else:
                patience_counter = 0

            if episode % 10000 == 0:
                print(f"Episode {episode}: {alt_metrics}")

    num_episodes = episode
    print(f"Completed {num_episodes} episodes ({'converged' if converged else 'max reached'})")

    # Get total rewards
    total_reward_per_agent = [agent.total_reward for agent in env.agents]

    # Calculate comprehensive metrics
    result = calculate_all_metrics(
        num_agents, num_episodes, state_type, reward_type,
        top_agents_per_episode, terminal_occurrences_per_episode,
        unique_winners_per_episode, total_reward_per_agent, epsilon_per_episode
    )

    # Add Q-table data
    # Convert state_visits format from {(state, agent): data} to {agent: {state: count}}
    state_visits_by_agent = {}
    for (state_idx, agent_id), visit_data in env.state_visits.items():
        if agent_id not in state_visits_by_agent:
            state_visits_by_agent[agent_id] = {}
        state_visits_by_agent[agent_id][state_idx] = visit_data['count']

    result['state_visits'] = state_visits_by_agent
    result['terminal_positions_log'] = env.terminal_positions_log
    result['exclusive_wins_log'] = env.exclusive_wins_log
    result['q_tables'] = {i: agent.Q for i, agent in enumerate(env.agents)}
    result['avg_rounds'] = sum(round_counts) / len(round_counts) if round_counts else 0
    result['converged'] = converged

    # Calculate ALT Ratio
    alt_ratios, estimated_agents, percentages = calculate_alt_ratio(result)
    result['alt_ratios'] = alt_ratios
    result['estimated_agents'] = estimated_agents
    result['percentages'] = percentages

    # Save results
    success = save_result(result)

    if success:
        print(f"[OK] Completed adaptive Q-learning: {num_agents} agents, {state_type}, {reward_type}")
    else:
        print(f"⚠ Completed but failed to save: {num_agents} agents, {state_type}, {reward_type}")

    return result

#===============================================================================
# DQN Experiments
#===============================================================================

def run_dqn_experiment(num_agents, num_episodes, state_type='Type-A', reward_type='ILF'):
    """Run DQN experiment with tracking of metrics."""
    if not DQN_AVAILABLE:
        raise ImportError("DQN not available - install PyTorch with: pip install torch")

    print(f"Running DQN: {num_agents} agents, {num_episodes} episodes, {state_type}, {reward_type}...")

    env = DQNEnvironment(num_agents, num_positions=3, state_type=state_type, reward_type=reward_type)

    # Calculate target episode for epsilon decay
    target_episode = int(env.epsilon_decay_target * num_episodes)
    print(f"Will reach epsilon_min={env.epsilon_min} at episode {target_episode}")

    # Initialize tracking variables
    top_agents_per_episode = []
    terminal_occurrences_per_episode = []
    unique_winners_per_episode = []
    round_counts = []
    epsilon_per_episode = []

    # Progress bar for tracking
    for episode in tqdm(range(num_episodes), desc="Episodes", ncols=100):
        # Linear epsilon decay
        if episode < target_episode:
            progress = episode / target_episode
            env.epsilon = 0.9 - progress * (0.9 - env.epsilon_min)
        else:
            env.epsilon = env.epsilon_min

        # Reset environment
        state = env.reset()
        positions, terminal_states = state
        done = False
        round_count = 0
        max_rounds = 100

        # Get initial state vectors for all agents
        state_vectors = []
        for agent in env.agents:
            if state_type == 'Type-A':
                state_vec = agent.encode_state_to_vector(positions)
            else:
                state_vec = agent.encode_state_to_vector(positions, terminal_states)
            state_vectors.append(state_vec)

        # Run episode
        while not done and round_count < max_rounds:
            # Select actions for all agents
            actions = []
            for i, agent in enumerate(env.agents):
                action = agent.select_action(state_vectors[i], env.epsilon)
                actions.append(action)

            # Execute step
            next_state, rewards, done = env.step(actions)
            next_positions, next_terminal_states = next_state

            # Get next state vectors
            next_state_vectors = []
            for agent in env.agents:
                if state_type == 'Type-A':
                    next_state_vec = agent.encode_state_to_vector(next_positions)
                else:
                    next_state_vec = agent.encode_state_to_vector(next_positions, next_terminal_states)
                next_state_vectors.append(next_state_vec)

            # Store transitions and train
            for i, agent in enumerate(env.agents):
                agent.store_transition(state_vectors[i], actions[i], rewards[i],
                                     next_state_vectors[i], done)
                agent.train_step()
                agent.total_reward += rewards[i]

            # Update state
            state_vectors = next_state_vectors
            positions = next_positions
            terminal_states = next_terminal_states
            round_count += 1

        # Track results
        top_agents_per_episode.append(env.current_top_agents)
        terminal_occurrences_per_episode.append(sum(env.current_top_agents))
        round_counts.append(round_count)
        epsilon_per_episode.append(env.epsilon)

        # Track unique winners
        if sum(env.current_top_agents) == 1:
            for i in range(num_agents):
                if env.current_top_agents[i] == 1:
                    unique_winners_per_episode.append(i)
                    break

        # Update environment for next episode (Type-B only)
        env.update_for_next_episode()

        # Print progress at 10% intervals
        if episode > 0 and episode % (num_episodes // 10) == 0:
            print(f"Episode {episode}/{num_episodes} ({episode/num_episodes:.1%}): epsilon = {env.epsilon:.4f}")

    # Get total rewards
    total_reward_per_agent = [agent.total_reward for agent in env.agents]

    # Calculate comprehensive metrics using wrapper
    result = calculate_all_metrics(
        num_agents, num_episodes, state_type, reward_type,
        top_agents_per_episode, terminal_occurrences_per_episode,
        unique_winners_per_episode, total_reward_per_agent, epsilon_per_episode
    )

    # Add DQN-specific data
    result['agent_type'] = 'dqn'
    result['avg_rounds'] = sum(round_counts) / len(round_counts) if round_counts else 0

    # Note: DQN doesn't have Q-tables (uses neural networks instead)
    # We can save network parameters if needed, but for now just mark as DQN
    result['network_architecture'] = {
        'type': 'DQN',
        'hidden_units': 128,
        'layers': 3,
        'activation': 'ReLU',
        'optimizer': 'Adam',
        'learning_rate': 0.3
    }

    # Calculate ALT Ratio
    alt_ratios, estimated_agents, percentages = calculate_alt_ratio(result)
    result['alt_ratios'] = alt_ratios
    result['estimated_agents'] = estimated_agents
    result['percentages'] = percentages

    # Save results
    success = save_result(result)

    if success:
        print(f"[OK] Completed DQN: {num_agents} agents, {state_type}, {reward_type}")
    else:
        print(f"⚠ Completed but failed to save: {num_agents} agents, {state_type}, {reward_type}")

    return result

#===============================================================================
# Complete Metrics Calculation Wrapper
#===============================================================================

def calculate_all_metrics(num_agents, num_episodes, state_type, reward_type,
                         top_agents_per_episode, terminal_occurrences_per_episode,
                         unique_winners_per_episode, total_reward_per_agent, epsilon_per_episode=None):
    """Calculate all metrics for an experiment."""
    # ALT metrics
    alt_metrics, beta_values, alt_computation_times = compute_alt_metrics(
        num_episodes, num_agents, terminal_occurrences_per_episode, top_agents_per_episode
    )

    # RP metrics (calculated but not displayed)
    rp_metrics = compute_rp_metrics(
        num_episodes, num_agents, top_agents_per_episode
    )

    # Traditional metrics with 3 fairness types
    efficiency_result = compute_efficiency(
        num_episodes, num_agents, total_reward_per_agent, FULL_REWARD
    )

    reward_fairness_result = compute_reward_fairness(
        total_reward_per_agent
    )

    tt_fairness_result = compute_tt_fairness(
        top_agents_per_episode, num_agents, num_episodes
    )

    fairness_result = compute_fairness(
        unique_winners_per_episode, num_agents
    )

    # Calculate per-agent statistics for individual performance analysis
    reaches_per_agent = [0] * num_agents
    exclusive_wins_per_agent = [0] * num_agents

    for ep in range(num_episodes):
        # Count reaches to top for each agent
        for i in range(num_agents):
            if top_agents_per_episode[ep][i] == 1:
                reaches_per_agent[i] += 1

        # Count exclusive wins (solo terminal reaches)
        if sum(top_agents_per_episode[ep]) == 1:
            for i in range(num_agents):
                if top_agents_per_episode[ep][i] == 1:
                    exclusive_wins_per_agent[i] += 1
                    break

    # Last 10% metrics
    last_10_percent = max(num_agents, int(0.1 * num_episodes))
    last_alt_metrics, _, _ = compute_alt_metrics(
        last_10_percent,
        num_agents,
        terminal_occurrences_per_episode[-last_10_percent:],
        top_agents_per_episode[-last_10_percent:]
    )

    # Progression data - increased sampling for better granularity
    num_windows = min(2000, num_episodes // 2)  # Up to 2000 points for better resolution
    window_size = max(1, num_episodes // num_windows)
    episode_points = []
    progression_data = {m: [] for m in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness',
                                        'FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT',
                                        'RP_avg', 'AWE_avg', 'WPE_avg', 'Epsilon']}

    # OPTIMIZED: Initialize cumulative trackers ONCE (outside loop)
    cumulative_rewards = [0.0] * num_agents
    cumulative_unique_winners = []
    cumulative_reaches = [0] * num_agents
    max_possible_reward = 0
    last_idx = 0  # Track where we left off

    for i in range(0, num_episodes, window_size):
        end_idx = min(i + window_size, num_episodes)
        episode_points.append(end_idx)

        # OPTIMIZED: Only process NEW episodes [last_idx, end_idx) instead of [0, end_idx)
        for ep in range(last_idx, end_idx):
            for j in range(num_agents):
                if top_agents_per_episode[ep][j] == 1:
                    cumulative_reaches[j] += 1

            agents_at_terminal = sum(top_agents_per_episode[ep])

            if agents_at_terminal == 1:
                # Single exclusive winner - gets r_high
                for j, reached_terminal in enumerate(top_agents_per_episode[ep]):
                    if reached_terminal:
                        cumulative_rewards[j] += FULL_REWARD
                        cumulative_unique_winners.append(j)
            elif agents_at_terminal == num_agents:
                # Full tie - all agents at terminal - no reward
                pass  # No reward given when all agents tie
            elif agents_at_terminal > 1 and agents_at_terminal < num_agents:
                # Partial collision - some but not all agents at terminal get r_low
                if reward_type == 'ILF':
                    r_low = FULL_REWARD / num_agents  # r_low = r_high/n
                else:  # IQF
                    r_low = FULL_REWARD / (num_agents * num_agents)  # r_low = r_high/n²

                for j, reached_terminal in enumerate(top_agents_per_episode[ep]):
                    if reached_terminal:
                        cumulative_rewards[j] += r_low

            max_possible_reward += FULL_REWARD

        # Update last_idx for next window
        last_idx = end_idx

        # Efficiency - using cumulative data
        total_current_reward = sum(cumulative_rewards)
        efficiency_value = total_current_reward / max_possible_reward if max_possible_reward > 0 else 0.0
        progression_data['Efficiency'].append(efficiency_value)

        # Reward Fairness - using cumulative_rewards
        positive_rewards = [r for r in cumulative_rewards if r > 0]
        if positive_rewards:
            progression_data['Reward_Fairness'].append(min(positive_rewards) / max(positive_rewards))
        else:
            progression_data['Reward_Fairness'].append(0.0)

        # TT Fairness - using cumulative_reaches
        positive_reaches = [r for r in cumulative_reaches if r > 0]
        if positive_reaches:
            progression_data['TT_Fairness'].append(min(positive_reaches) / max(positive_reaches))
        else:
            progression_data['TT_Fairness'].append(0.0)

        # Fairness - using cumulative_unique_winners
        if cumulative_unique_winners:
            win_counts = [cumulative_unique_winners.count(j) for j in range(num_agents)]
            positive_wins = [c for c in win_counts if c > 0]
            if positive_wins:
                progression_data['Fairness'].append(min(positive_wins) / max(positive_wins))
            else:
                progression_data['Fairness'].append(0.0)
        else:
            progression_data['Fairness'].append(0.0)

        # ALT and RP metrics
        if end_idx >= num_agents:
            window_alt, _, _ = compute_alt_metrics(
                end_idx, num_agents,
                terminal_occurrences_per_episode[:end_idx],
                top_agents_per_episode[:end_idx]
            )

            for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
                progression_data[metric].append(window_alt[metric])

            window_rp = compute_rp_metrics(
                end_idx, num_agents, top_agents_per_episode[:end_idx]
            )

            progression_data['RP_avg'].append(window_rp['RP_avg'])
            progression_data['AWE_avg'].append(window_rp['AWE_avg'])
            progression_data['WPE_avg'].append(window_rp['WPE_avg'])
        else:
            for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT', 'RP_avg', 'AWE_avg', 'WPE_avg']:
                progression_data[metric].append(0.0)

        # Track epsilon value at this point
        if epsilon_per_episode and end_idx > 0 and end_idx <= len(epsilon_per_episode):
            progression_data['Epsilon'].append(epsilon_per_episode[end_idx - 1])
        else:
            progression_data['Epsilon'].append(0.0)  # N/A for random policy

    # Compile results
    metrics = {
        **alt_metrics,
        'RP_avg': rp_metrics['RP_avg'],
        'AWE_avg': rp_metrics['AWE_avg'],
        'WPE_avg': rp_metrics['WPE_avg'],
        'Efficiency': efficiency_result['Efficiency'],
        'Reward_Fairness': reward_fairness_result['Reward_Fairness'],
        'TT_Fairness': tt_fairness_result['TT_Fairness'],
        'Fairness': fairness_result['Fairness']
    }

    computation_times = {
        **alt_computation_times,
        'RP_AWE': rp_metrics['computation_times']['AWE'],
        'RP_WPE': rp_metrics['computation_times']['WPE'],
        'RP_avg': rp_metrics['computation_times']['RP_avg'],
        'RP_Total': rp_metrics['computation_times']['Total'],
        'Efficiency': efficiency_result['computation_time'],
        'Reward_Fairness': reward_fairness_result['computation_time'],
        'TT_Fairness': tt_fairness_result['computation_time'],
        'Fairness': fairness_result['computation_time']
    }

    results = {
        'num_agents': num_agents,
        'num_episodes': num_episodes,
        'state_type': state_type,
        'reward_type': reward_type,
        'metrics': metrics,
        'last_10_percent_metrics': last_alt_metrics,
        'beta_values': beta_values,
        'rp_values': {
            'AWE_per_agent': rp_metrics['AWE_per_agent'],
            'WPE_per_agent': rp_metrics['WPE_per_agent'],
            'RP_per_agent': rp_metrics['RP_per_agent']
        },
        'agent_stats': {
            'reaches_per_agent': reaches_per_agent,
            'exclusive_wins_per_agent': exclusive_wins_per_agent
        },
        'progression_data': progression_data,
        'episode_points': episode_points,
        'computation_times': computation_times
    }

    return results

#===============================================================================
# Multiprocessing Workers
#===============================================================================

def init_worker(checkpoint_dir_path):
    """Initialize worker process with correct checkpoint directory path."""
    config.checkpoint_dir = checkpoint_dir_path
    config.figure_dirs['checkpoints'] = checkpoint_dir_path

def run_single_experiment_worker(experiment_params):
    """Worker function to run a single experiment (for multiprocessing)."""
    num_agents, state_type, reward_type, experiment_type, agent_type = experiment_params

    # Reset seed for reproducibility
    random.seed(42 + num_agents)  # Different seed per agent count
    np.random.seed(42 + num_agents)

    try:
        if state_type == 'Random':
            # Random policy simulation (same for both Q-Learning and DQN)
            print(f"  [Process {os.getpid()}] Running random baseline: {num_agents} agents, {reward_type}")
            num_episodes = 10000
            result = simulate_random_policy(num_agents, num_episodes, reward_type=reward_type)
        elif experiment_type == 'theory':
            # Theory-based episodes calculation
            num_episodes = calculate_episodes_theory_based(num_agents, BASE_EPISODES)

            if agent_type == 'dqn':
                # Run DQN experiment
                print(f"  [Process {os.getpid()}] Theory-based DQN: {num_agents} agents, {state_type}, {reward_type} - {num_episodes} episodes")
                result = run_dqn_experiment(num_agents, num_episodes, state_type, reward_type)
            else:
                # Run Q-Learning experiment
                print(f"  [Process {os.getpid()}] Theory-based Q-Learning: {num_agents} agents, {state_type}, {reward_type} - {num_episodes} episodes")
                result = run_q_learning_experiment(num_agents, num_episodes, state_type, reward_type, decay_schedule='exponential')
        else:
            # Adaptive threshold-based approach (only for Q-Learning for now)
            if agent_type == 'dqn':
                print(f"[WARNING] Adaptive mode not implemented for DQN yet - falling back to theory-based")
                num_episodes = calculate_episodes_theory_based(num_agents, BASE_EPISODES)
                result = run_dqn_experiment(num_agents, num_episodes, state_type, reward_type)
            else:
                print(f"  [Process {os.getpid()}] Adaptive Q-Learning: {num_agents} agents, {state_type}, {reward_type}")
                alt_thresholds = {
                    'CALT': 0.6,
                    'FALT': 0.7,
                    'EALT': 0.7,
                    'qFALT': 0.5,
                    'qEALT': 0.5,
                    'AALT': 0.4
                }
                result = run_adaptive_q_learning_experiment(
                    num_agents, state_type, reward_type,
                    alt_thresholds=alt_thresholds,
                    patience=5000,
                    max_episodes=150000
                )

        # Clean up memory
        gc.collect()

        # Print completion message
        if state_type == 'Random':
            print(f"[OK] Completed random baseline: {num_agents} agents, {reward_type}")
        else:
            agent_name = 'DQN' if agent_type == 'dqn' else 'Q-Learning'
            print(f"[OK] Completed {agent_name}: {num_agents} agents, {state_type}, {reward_type}")

        return result

    except Exception as e:
        print(f"ERROR in worker [Process {os.getpid()}]: {num_agents} agents, {state_type}, {reward_type} ({agent_type}) - {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_experiments(experiment_type='theory', agent_type='ql'):
    """Run experiments with selected approach (with optional multiprocessing).

    Args:
        experiment_type: 'theory' for theory-based episodes, 'adaptive' for threshold-based
        agent_type: 'ql' for Q-Learning, 'dqn' for Deep Q-Network
    """
    if IN_COLAB:
        keep_colab_alive()

    # Configuration
    from config import TEST_AGENT_COUNTS, BASE_EPISODES, get_run_directory, create_subdirectories
    agent_counts = TEST_AGENT_COUNTS  # From config.py
    state_types = ['Type-A', 'Type-B']
    reward_types = ['ILF', 'IQF']

    # Verify DQN availability if requested
    if agent_type == 'dqn' and not DQN_AVAILABLE:
        raise ImportError("DQN requested but PyTorch not available. Install with: pip install torch")

    # Create run directory with metadata in name
    run_dir = get_run_directory(
        experiment_type=experiment_type,
        base_episodes=BASE_EPISODES,
        agent_type=agent_type,  # 'ql' or 'dqn'
        agent_counts=agent_counts
    )

    # Create subdirectories and UPDATE config module variables
    subdirs = create_subdirectories(run_dir)
    config.checkpoint_dir = subdirs['checkpoints']
    config.figure_dirs = subdirs
    config.base_dir = run_dir  # Also update base_dir to point to run directory

    print(f"Run directory: {run_dir}")
    print(f"Checkpoints: {config.checkpoint_dir}")

    results_list = load_results()

    # Generate experiments to run list
    experiments_to_run = []
    for num_agents in agent_counts:
        for state_type in state_types:
            for reward_type in reward_types:
                if not is_experiment_completed(results_list, num_agents, state_type, reward_type):
                    experiments_to_run.append((
                        num_agents, state_type, reward_type, experiment_type, agent_type
                    ))

    # Add random baselines (always use same baseline regardless of agent_type)
    for num_agents in agent_counts:
        for reward_type in reward_types:
            if not is_experiment_completed(results_list, num_agents, None, reward_type, is_random=True):
                experiments_to_run.append((
                    num_agents, 'Random', reward_type, experiment_type, agent_type
                ))

    if not experiments_to_run:
        print("All experiments completed!")
        return results_list

    print(f"Running {len(experiments_to_run)} experiments with {experiment_type} approach...")

    # Multiprocessing or sequential
    if USE_MULTIPROCESSING and len(experiments_to_run) > 1:
        print(f"Using multiprocessing with {NUM_PROCESSES} parallel processes")
        print("NOTE: Progress bars may overlap - this is normal")

        # Use multiprocessing Pool with initializer to set checkpoint_dir in each worker
        with mp.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(config.checkpoint_dir,)) as pool:
            new_results = pool.map(run_single_experiment_worker, experiments_to_run)

        # Add successful results to list
        for result in new_results:
            if result is not None:
                results_list.append(result)

        print(f"Completed {sum(1 for r in new_results if r is not None)}/{len(experiments_to_run)} experiments")

    else:
        # Sequential execution (original behavior)
        print("Running experiments sequentially (multiprocessing disabled)")
        for i, exp_params in enumerate(experiments_to_run):
            print(f"Progress: {i+1}/{len(experiments_to_run)} ({(i+1)/len(experiments_to_run)*100:.1f}%)")

            result = run_single_experiment_worker(exp_params)

            if result is not None:
                results_list.append(result)

            gc.collect()

    return results_list
