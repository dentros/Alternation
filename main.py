"""
Main execution module for ALT Metrics experiments.

This module provides the main entry point for running experiments
and generating visualizations.
"""

import sys
import os
import random
import numpy as np
import gc

from config import BASE_EPISODES, TEST_AGENT_COUNTS
from experiments import (
    load_results, delete_all_results, run_experiments,
    find_compatible_runs, rename_run_directory, copy_run_directory, run_experiments_continue
)

# Visualization functions migrated to visualization.py
from visualization import (
    create_all_figures,
    save_comprehensive_analysis
)

def main(auto_mode=False, experiment_type='theory', agent_type='ql'):
    """Main execution function.

    Args:
        auto_mode: If True, skip all user prompts and use defaults
        experiment_type: 'theory' or 'adaptive' (default: 'theory')
        agent_type: 'ql' for Q-Learning or 'dqn' for Deep Q-Network (default: 'ql')
    """
    print("=" * 80)
    print("ALT METRICS PAPER 1 - EXPERIMENTAL FRAMEWORK")
    print("=" * 80)
    print(f"BASE_EPISODES = {BASE_EPISODES}")
    print(f"AGENT TYPE = {agent_type.upper()}")
    print("=" * 80)

    # FIRST: Check for compatible run directories (even if no results loaded in default dir)
    if not auto_mode:
        existing_runs = find_compatible_runs(BASE_EPISODES, experiment_type, agent_type)

        if existing_runs:
            print("\n" + "=" * 80)
            print("FOUND COMPATIBLE EXISTING RUN DIRECTORIES")
            print("=" * 80)
            for i, run_info in enumerate(existing_runs, 1):
                agents_str = ','.join(map(str, run_info['agent_counts']))
                print(f"{i}. {run_info['dir_name']}")
                print(f"   Agents: [{agents_str}] - {run_info['checkpoint_count']} checkpoints")

            print("\nDo you want to:")
            print("[C] Continue from existing directory (choose one above)")
            print("[N] Create new run directory")
            choice = input("Choice (C/N): ").strip().upper()

            if choice == 'C':
                # Select which directory to use
                while True:
                    try:
                        idx = int(input(f"Enter directory number (1-{len(existing_runs)}): ")) - 1
                        if 0 <= idx < len(existing_runs):
                            break
                        print(f"Please enter a number between 1 and {len(existing_runs)}")
                    except ValueError:
                        print("Please enter a valid number")

                selected_run = existing_runs[idx]
                print(f"\nSelected: {selected_run['dir_name']}")

                # Ask how to handle the directory
                print("\nHow do you want to continue?")
                print("[R] Rename old directory (add new agents + keep old timestamp)")
                print("[C] Copy to new directory (add new agents + NEW timestamp)")
                print("[K] Keep as-is (don't rename)")
                dir_choice = input("Choice (R/C/K): ").strip().upper()

                if dir_choice == 'R':
                    new_name = rename_run_directory(selected_run, TEST_AGENT_COUNTS)
                    run_dir = os.path.join('results', new_name)
                elif dir_choice == 'C':
                    new_name = copy_run_directory(selected_run, TEST_AGENT_COUNTS)
                    if new_name:
                        run_dir = os.path.join('results', new_name)
                    else:
                        print("Error creating new directory. Using existing.")
                        run_dir = selected_run['full_path']
                else:  # K or any other choice
                    run_dir = selected_run['full_path']
                    print(f"Using existing directory as-is: {selected_run['dir_name']}")

                # Validate checkpoint directory exists
                checkpoint_dir = os.path.join(run_dir, 'checkpoints')
                if not os.path.exists(checkpoint_dir):
                    print(f"\n⚠️  WARNING: Checkpoint directory not found: {checkpoint_dir}")
                    print("Creating checkpoint directory...")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                else:
                    import glob
                    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pkl'))
                    print(f"\n✓ Found {len(checkpoints)} checkpoint files in {os.path.basename(run_dir)}/checkpoints/")

                # Use directory - continue experiments
                results_list = run_experiments_continue(run_dir, experiment_type, agent_type)

                # Clean up memory before visualization
                gc.collect()

                # Generate all figures for Paper 1
                print("\n" + "=" * 80)
                print("GENERATING FIGURES & ANALYSIS")
                print("=" * 80)
                create_all_figures(results_list)
                save_comprehensive_analysis(results_list)
                print("\n" + "=" * 80)
                print("COMPLETE - All experiments finished and figures generated!")
                print("=" * 80)
                return

    # SECOND: Check if results already exist in default checkpoint dir
    results_list = load_results()
    has_existing_results = len(results_list) > 0

    if has_existing_results:
        # Extract the approach from existing results if possible
        try:
            # Check if any result has threshold_info - this means it was an adaptive experiment
            for r in results_list:
                if 'threshold_info' in r:
                    experiment_type = 'adaptive'
                    break
        except:
            pass

        if auto_mode:
            # In auto mode, keep existing results and continue
            print(f"\nFound {len(results_list)} existing experiment results.")
            print(f"AUTO MODE: Keeping previous results. Continuing with {experiment_type} approach.")
        else:
            # Ask once about existing results
            print(f"\nFound {len(results_list)} existing experiment results.")
            print("Do you want to delete all previous results and start fresh? (y/n)")
            choice = input().lower().strip()

            if choice == 'y':
                delete_all_results()
                print("Previous results deleted. Starting fresh.")
                results_list = []

                # Ask for approach if starting fresh
                print("\nWhich approach do you want to use for determining episode count?")
                print("1. Theory-based (calculates episodes based on agent count and complexity)")
                print("2. Adaptive (runs until ALT thresholds are reached)")
                approach_choice = input("Enter choice (1 or 2): ").strip()
                experiment_type = 'theory' if approach_choice == '1' else 'adaptive'
            else:
                print(f"Keeping previous results. Continuing with {experiment_type} approach.")
    else:
        # No existing results in default dir
        if auto_mode:
            print(f"\nNo previous results found.")
            print(f"AUTO MODE: Using {experiment_type} approach.")
        else:
            print("\nNo previous results found. Which approach do you want to use for determining episode count?")
            print("1. Theory-based (calculates episodes based on agent count and complexity)")
            print("2. Adaptive (runs until ALT thresholds are reached)")
            approach_choice = input("Enter choice (1 or 2): ").strip()
            experiment_type = 'theory' if approach_choice == '1' else 'adaptive'

    # Run experiments (creates new directory)
    results_list = run_experiments(experiment_type, agent_type)

    # Clean up memory before visualization
    gc.collect()

    # Generate all figures for Paper 1
    create_all_figures(results_list)

    print("\n" + "=" * 80)
    print("RESEARCH QUESTIONS FOR PAPER 1")
    print("=" * 80)
    print("? Can ALT metrics effectively capture turn-taking patterns?")
    print("? Are traditional metrics inadequate for measuring alternation?")
    print("? Is ~65-70% alternation achieved despite high efficiency and fairness?")
    print("? Does Type-B show improvement compared to Type-A?")
    print("? Does performance decrease with increasing number of agents?")
    print("? Does the rate of CALT progress decrease over time?")
    print("? Does the last 10% of episodes show better results than the whole?")
    print("? How do Q-learning agents perform compared to random baselines?")
    print("? What is the coordination score (improvement over random) for each agent count?")

    print("\n[OK] COMPLETED! Figures and tables have been successfully saved.")
    return results_list


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Check for command-line arguments
    auto_mode = '--auto' in sys.argv or '-a' in sys.argv
    experiment_type = 'theory'  # default
    agent_type = 'ql'  # default

    if '--adaptive' in sys.argv:
        experiment_type = 'adaptive'

    if '--dqn' in sys.argv:
        agent_type = 'dqn'
        print("\n[INFO] DQN mode enabled - will use Deep Q-Networks instead of tabular Q-Learning")
        print("[INFO] Make sure PyTorch is installed: pip install torch\n")

    results = main(auto_mode=auto_mode, experiment_type=experiment_type, agent_type=agent_type)
