"""
Visualization module for ALT Metrics experiments.

Contains all visualization and analysis functions migrated from new_code.py.

Updated naming (Dec 29, 2025):
- qFALT/qEALT (quadratic) instead of EFALT/EEALT (exponential) - mathematically correct
- ILF/IQF (Inverse Linear/Quadratic Fractional) instead of EFR/EAFR - mathematically correct
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config
from config import (
    COLOR_PALETTE, BASE_EPISODES, EPSILON_DECAY_TARGET, EPSILON_MIN, EPSILON_INITIAL
)
from metrics import (
    compute_alt_metrics, compute_rp_metrics,
    compute_efficiency, compute_reward_fairness, compute_tt_fairness, compute_fairness,
    calculate_coordination_score, calculate_alt_ratio
)

def create_figure_7_altratio_analysis(results_list):
    """
    Figure 7: AltRatio Analysis - separate plots for each mode (Type-A/B × ILF/IQF).
    Creates 4 separate figures, each with 2 subplots (estimated agents, percentages).
    """
    # Define 4 modes
    modes = [
        ('Type-A', 'ILF'),
        ('Type-A', 'IQF'),
        ('Type-B', 'ILF'),
        ('Type-B', 'IQF')
    ]

    for state_type, reward_type in modes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Filter results for this mode
        results = [r for r in results_list if r.get('state_type', '') == state_type
                   and r['reward_type'] == reward_type
                   and not r.get('is_random_baseline', False)]
        results.sort(key=lambda x: x['num_agents'])

        if not results:
            plt.close()
            continue

        # Build separate agent lists for each metric (in case some metrics are missing for some agents)
        agents_per_metric = {metric: [] for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']}
        estimated_agents = {metric: [] for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']}
        percentages = {metric: [] for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']}

        for r in results:
            for metric in estimated_agents.keys():
                if 'estimated_agents' in r and metric in r['estimated_agents']:
                    agents_per_metric[metric].append(r['num_agents'])
                    estimated_agents[metric].append(r['estimated_agents'][metric])
                    percentages[metric].append(r['percentages'][metric])

        # Get unique agent counts for perfect alternation line
        all_agents = sorted(set([r['num_agents'] for r in results]))

        # Panel (a): Estimated agents for all ALT metrics
        for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
            if estimated_agents[metric]:
                ax1.plot(agents_per_metric[metric], estimated_agents[metric], 'o-',
                       label=metric, color=COLOR_PALETTE[metric], markersize=5, linewidth=1.5)

        # Add perfect alternation line - semi-transparent dashed
        if all_agents:
            ax1.plot(all_agents, all_agents, 'r--', label='Perfect Alternation',
                    linewidth=2, alpha=0.5, zorder=5)

        ax1.set_xlabel('Number of Agents', fontsize=11)
        ax1.set_ylabel('Estimated Alternating Agents', fontsize=11)
        ax1.set_title(f'(a) AltRatio Estimation - {state_type} {reward_type}', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)

        # Panel (b): Percentages for all ALT metrics (NO 65-70% range)
        x = np.arange(len(all_agents))
        width = 0.12

        metrics_to_show = ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']

        for i, metric in enumerate(metrics_to_show):
            if percentages[metric]:
                # Map metric's agent counts to x positions
                x_positions = [all_agents.index(agent) for agent in agents_per_metric[metric]]
                offset = width * (i - len(metrics_to_show)/2 + 0.5)
                ax2.bar(np.array(x_positions) + offset, percentages[metric], width,
                       label=metric, color=COLOR_PALETTE[metric])

        # NO 65-70% range lines

        ax2.set_xlabel('Number of Agents', fontsize=11)
        ax2.set_ylabel('Alternation Percentage (%)', fontsize=11)
        ax2.set_title(f'(b) Achieved Alternation - {state_type} {reward_type}', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_agents)
        ax2.legend(fontsize=9, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 105])

        plt.suptitle(f'Figure 7: ALT Ratio Analysis - {state_type} {reward_type}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.figure_dirs['main'],
                    f'figure_7_altratio_{state_type}_{reward_type}.png'), dpi=300)
        plt.show()
        plt.close()

        print(f"[OK] Saved Figure 7 for {state_type} {reward_type}")


def create_figure_6_metric_progression_rate(results_list, metric='CALT'):
    """
    Figure 6: Metric progression rate for all 4 modes (Type-A/B × ILF/IQF).
    4 subplots in 2×2 grid, normalized 0-1, no slope annotations, with random baselines.
    """
    # Create 2x2 grid for all 4 modes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Define 4 modes
    modes = [
        ('Type-A', 'ILF', 0),
        ('Type-A', 'IQF', 1),
        ('Type-B', 'ILF', 2),
        ('Type-B', 'IQF', 3)
    ]

    # Import colormap for gradient
    import matplotlib.cm as cm

    for state_type, reward_type, subplot_idx in modes:
        ax = axes[subplot_idx]

        # Get Q-learning results for this mode
        ql_results = [r for r in results_list if r.get('state_type', '') == state_type
                      and r['reward_type'] == reward_type
                      and not r.get('is_random_baseline', False)]
        ql_results.sort(key=lambda x: x['num_agents'])

        # Get random baseline for this reward type
        random_results = [r for r in results_list if r.get('is_random_baseline', False)
                          and r['reward_type'] == reward_type]

        # Generate color gradient for agent counts
        n_results = len(ql_results)
        if n_results > 0:
            colors = cm.get_cmap('viridis')(np.linspace(0.3, 0.9, n_results))

            # Plot Q-learning curves with gradient colors
            for i, result in enumerate(ql_results):
                metric_vals = result['progression_data'][metric]
                episodes = result['episode_points']

                # Normalize episodes 0-1
                norm_episodes = [ep / result['num_episodes'] for ep in episodes]

                ax.plot(norm_episodes, metric_vals,
                       label=f"{result['num_agents']} agents",
                       linestyle='-',  # Solid only
                       alpha=0.85, linewidth=2.0,
                       color=colors[i])  # Gradient color

            # Add epsilon milestone indicator (vertical line)
            if len(ql_results) > 0 and 'Epsilon' in ql_results[0]['progression_data']:
                # Get epsilon decay target from global parameters
                epsilon_decay_target = ql_results[0].get('q_learning_info', {}).get('epsilon_decay_target', EPSILON_DECAY_TARGET)

                # Draw vertical line at epsilon milestone - use metric's palette color
                ax.axvline(x=epsilon_decay_target,
                          color=COLOR_PALETTE.get(metric, 'gray'), linestyle=':',
                          linewidth=1.5, alpha=0.4,
                          label=f'ε-min@{epsilon_decay_target:.0%}')

        # Add random baseline with text annotation on line
        for random_result in random_results:
            if metric in random_result['metrics']:
                n_agents = random_result['num_agents']
                random_value = random_result['metrics'][metric]

                # Draw horizontal line
                ax.axhline(y=random_value, color='gray', linestyle=':',
                          linewidth=1.5, alpha=0.4)

                # Add small text label ON the line (right side)
                ax.text(0.98, random_value, f'random ({n_agents}a)',
                       transform=ax.get_yaxis_transform(),
                       fontsize=7, va='center', ha='right',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        ax.set_xlabel('Episode Progress (0-1)', fontsize=10)
        ax.set_ylabel(f'{metric} Value', fontsize=10)
        ax.set_title(f'{state_type} {reward_type}', fontsize=11, fontweight='bold')
        legend = ax.legend(fontsize=8, loc='best', ncol=2,
                          frameon=True, fancybox=True,
                          framealpha=0.7, edgecolor='gray',
                          facecolor='white')
        legend.get_frame().set_boxstyle('round', pad=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    plt.suptitle(f'Figure 6: {metric} Progression Rate (All Modes)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['main'], f'figure_6_{metric}_progression_rate_all_modes.png'), dpi=300)
    plt.show()
    plt.close()


def create_figure_2_type_comparison(results_list):
    """
    Figure 2: Type-A vs Type-B Progression with ALL metrics (Traditional + ALT).
    Creates comprehensive 4x2 plot showing both ILF and IQF for both state types.
    4 rows (Type-A ILF, Type-A IQF, Type-B ILF, Type-B IQF) × 2 cols (Traditional, ALT)
    """
    # Use only 2 agents for simplicity
    selected_agents = [2]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    # Find all 4 combinations
    type_a_efr = next((r for r in results_list if r['num_agents'] == selected_agents[0] and
                       r.get('state_type', '') == 'Type-A' and r['reward_type'] == 'ILF'), None)
    type_a_eafr = next((r for r in results_list if r['num_agents'] == selected_agents[0] and
                        r.get('state_type', '') == 'Type-A' and r['reward_type'] == 'IQF'), None)
    type_b_efr = next((r for r in results_list if r['num_agents'] == selected_agents[0] and
                       r.get('state_type', '') == 'Type-B' and r['reward_type'] == 'ILF'), None)
    type_b_eafr = next((r for r in results_list if r['num_agents'] == selected_agents[0] and
                        r.get('state_type', '') == 'Type-B' and r['reward_type'] == 'IQF'), None)

    results_grid = [
        (type_a_efr, 'Type-A ILF', 0),
        (type_a_eafr, 'Type-A IQF', 1),
        (type_b_efr, 'Type-B ILF', 2),
        (type_b_eafr, 'Type-B IQF', 3)
    ]

    for result, label, row_idx in results_grid:
        if result:
            x_vals = [e/result['num_episodes'] for e in result['episode_points']]

            # Left column: Traditional Metrics
            for metric in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness']:
                axes[row_idx, 0].plot(x_vals, result['progression_data'][metric],
                                     label=metric, alpha=0.8, color=COLOR_PALETTE.get(metric), linewidth=1.5)
            axes[row_idx, 0].set_title(f'{label} - Traditional ({selected_agents[0]} agents)', fontsize=10)
            axes[row_idx, 0].set_xlabel('Episode Progress', fontsize=9)
            axes[row_idx, 0].set_ylabel('Metric Value', fontsize=9)
            axes[row_idx, 0].grid(True, alpha=0.3)
            axes[row_idx, 0].legend(fontsize=7, loc='best')
            axes[row_idx, 0].set_ylim([0, 1.05])

            # Right column: ALT Metrics
            for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
                axes[row_idx, 1].plot(x_vals, result['progression_data'][metric],
                                     label=metric, alpha=0.8, color=COLOR_PALETTE.get(metric), linewidth=1.5)
            axes[row_idx, 1].set_title(f'{label} - ALT ({selected_agents[0]} agents)', fontsize=10)
            axes[row_idx, 1].set_xlabel('Episode Progress', fontsize=9)
            axes[row_idx, 1].set_ylabel('Metric Value', fontsize=9)
            axes[row_idx, 1].grid(True, alpha=0.3)
            axes[row_idx, 1].legend(fontsize=7, loc='best')
            axes[row_idx, 1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['main'], 'figure_2_type_comparison_comprehensive.png'), dpi=300)
    plt.show()
    plt.close()

def create_random_baseline_progression_plots(results_list):
    """Create progression plots for random baselines showing ALL metrics."""
    print("\n== Creating Random Baseline Progression Plots ==")

    # Filter random baseline results
    random_results = [r for r in results_list if r.get('is_random_baseline', False)
                      and r['reward_type'] == 'ILF' and 'progression_data' in r]
    random_results.sort(key=lambda x: x['num_agents'])

    for result in random_results:
        n_agents = result['num_agents']
        n_episodes = result['num_episodes']
        progression_data = result['progression_data']
        episode_points = result['episode_points']

        # Normalize episode points
        x_vals = [e / n_episodes for e in episode_points]

        # Create 2x2 plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Panel 1: Traditional Metrics
        for metric in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness']:
            ax1.plot(x_vals, progression_data[metric],
                     label=metric, alpha=0.8, color=COLOR_PALETTE.get(metric), linewidth=1.5)
        ax1.set_title(f'(a) Traditional Metrics - Random Policy ({n_agents} agents)', fontsize=10)
        ax1.set_xlabel('Episode Progress')
        ax1.set_ylabel('Metric Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # Panel 2: F-based ALT Metrics
        for metric in ['FALT', 'qFALT']:
            ax2.plot(x_vals, progression_data[metric],
                     label=metric, alpha=0.8, color=COLOR_PALETTE.get(metric), linewidth=1.5)
        ax2.set_title(f'(b) Frequency ALT Metrics - Random Policy ({n_agents} agents)', fontsize=10)
        ax2.set_xlabel('Episode Progress')
        ax2.set_ylabel('Metric Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # Panel 3: E-based ALT Metrics
        for metric in ['EALT', 'qEALT']:
            ax3.plot(x_vals, progression_data[metric],
                     label=metric, alpha=0.8, color=COLOR_PALETTE.get(metric), linewidth=1.5)
        ax3.set_title(f'(c) Episode ALT Metrics - Random Policy ({n_agents} agents)', fontsize=10)
        ax3.set_xlabel('Episode Progress')
        ax3.set_ylabel('Metric Value')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        # Panel 4: Combined ALT Metrics
        for metric in ['CALT', 'AALT']:
            ax4.plot(x_vals, progression_data[metric],
                     label=metric, alpha=0.8, color=COLOR_PALETTE.get(metric), linewidth=1.5)
        ax4.set_title(f'(d) Combined ALT Metrics - Random Policy ({n_agents} agents)', fontsize=10)
        ax4.set_xlabel('Episode Progress')
        ax4.set_ylabel('Metric Value')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(config.figure_dirs['random'],
                    f'random_baseline_progression_{n_agents}agents.png'), dpi=300)
        plt.show()
        plt.close()
        print(f"[OK] Saved random baseline progression for {n_agents} agents")

    print("[OK] Random baseline progression plots completed")

def create_random_vs_qlearning_comparison(results_list, metric='CALT'):
    """
    Create a figure comparing random policies vs Q-learning for a specific metric.
    Shows all 4 modes (Type-A/B × ILF/IQF) on the same plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define 4 modes with colors and line styles
    modes = [
        ('Type-A', 'ILF', 'o-', 0.9, '#1f77b4'),    # Blue
        ('Type-A', 'IQF', 's--', 0.9, '#ff7f0e'),  # Orange
        ('Type-B', 'ILF', 'D-.', 0.9, '#2ca02c'),   # Green
        ('Type-B', 'IQF', '^:', 0.9, '#d62728')    # Red
    ]

    # Collect all random baselines first to check if they're identical
    random_baselines_by_mode = {}
    all_values = []

    # First pass: collect random baselines
    for state_type, reward_type, _, _, _ in modes:
        random_results = [r for r in results_list if r.get('is_random_baseline', False)
                          and r['reward_type'] == reward_type]
        random_results.sort(key=lambda x: x['num_agents'])

        if random_results:
            random_agents = [r['num_agents'] for r in random_results]
            random_values = [r['metrics'][metric] for r in random_results]
            mode_key = f'{state_type}_{reward_type}'
            random_baselines_by_mode[mode_key] = (random_agents, random_values)

    # Check if all random baselines are identical
    random_values_lists = [vals for _, vals in random_baselines_by_mode.values()]
    all_random_same = len(set(tuple(vals) for vals in random_values_lists)) == 1 if random_values_lists else True

    # Second pass: plot Q-learning and random baselines
    random_plotted = False
    for state_type, reward_type, linestyle_marker, alpha_val, mode_color in modes:
        # Get Q-learning results
        qlearning_results = [r for r in results_list if r.get('state_type', '') == state_type
                            and r['reward_type'] == reward_type
                            and not r.get('is_random_baseline', False)]
        qlearning_results.sort(key=lambda x: x['num_agents'])

        if not qlearning_results:
            continue

        # Extract Q-learning data
        qlearning_agents = [r['num_agents'] for r in qlearning_results]
        qlearning_values = [r['metrics'][metric] for r in qlearning_results]

        # Collect values for auto-zoom
        all_values.extend(qlearning_values)

        # Plot Q-learning
        label = f'Q-Learning {state_type} {reward_type}'
        ax.plot(qlearning_agents, qlearning_values, linestyle_marker,
               label=label, markersize=6, linewidth=2, alpha=alpha_val,
               color=mode_color)

        # Plot random baseline
        mode_key = f'{state_type}_{reward_type}'
        if mode_key in random_baselines_by_mode:
            random_agents, random_values = random_baselines_by_mode[mode_key]
            all_values.extend(random_values)

            if all_random_same:
                # All random baselines are the same - plot only once
                if not random_plotted:
                    random_plotted = True
                    ax.plot(random_agents, random_values, 'x--',
                           label='Random baseline',
                           markersize=6, linewidth=1.5, alpha=0.4,
                           color='gray')
            else:
                # Random baselines differ (shouldn't happen!) - plot each with mode label
                ax.plot(random_agents, random_values, 'x--',
                       label=f'Random {state_type}-{reward_type}',
                       markersize=6, linewidth=1.5, alpha=0.4,
                       color=mode_color)

    # Auto-zoom based on data range
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        ax.set_ylim([min_val * 0.90, max_val * 1.05])
    else:
        ax.set_ylim([0, 1.05])  # Fallback

    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel(f'{metric} Value', fontsize=12)
    ax.set_title(f'Random vs Q-Learning Comparison: {metric} by Agent Count', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['comparison'], f'random_vs_qlearning_{metric}_all_modes.png'), dpi=300)
    plt.show()
    plt.close()


def create_alt_ratio_benchmark_comparison(results_list, metric='CALT'):
    """Compare Q-learning results with PA benchmark values - 4 bars per agent (all modes)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define 4 modes
    modes = [
        ('Type-A', 'ILF', COLOR_PALETTE.get(metric, 'blue'), 0.85),
        ('Type-A', 'IQF', COLOR_PALETTE.get(metric, 'blue'), 0.6),
        ('Type-B', 'ILF', COLOR_PALETTE.get(metric, 'blue'), 0.35),
        ('Type-B', 'IQF', COLOR_PALETTE.get(metric, 'blue'), 0.1)
    ]

    # Get all unique agent counts
    all_agent_counts = sorted(set([r['num_agents'] for r in results_list
                                   if not r.get('is_random_baseline', False)]))

    # Prepare data for each mode
    mode_data = []
    for state_type, reward_type, color, alpha_adjust in modes:
        qlearning_results = [r for r in results_list if r.get('state_type', '') == state_type
                            and r['reward_type'] == reward_type
                            and not r.get('is_random_baseline', False)]
        qlearning_results.sort(key=lambda x: x['num_agents'])

        # Get PA equivalents for this mode
        pa_equivalents = []
        for n_agents in all_agent_counts:
            result = next((r for r in qlearning_results if r['num_agents'] == n_agents), None)
            if result:
                pa_eq = result['estimated_agents'].get(metric, 0)
                pa_equivalents.append(pa_eq)
            else:
                pa_equivalents.append(0)

        mode_data.append({
            'label': f'{state_type} {reward_type}',
            'pa_equivalents': pa_equivalents,
            'color': color,
            'alpha': 0.9 - alpha_adjust
        })

    # Create grouped bar chart
    x = np.arange(len(all_agent_counts))
    bar_width = 0.2

    for i, mode_info in enumerate(mode_data):
        offset = (i - 1.5) * bar_width
        bars = ax.bar(x + offset, mode_info['pa_equivalents'],
                      width=bar_width,
                      label=mode_info['label'],
                      color=mode_info['color'],
                      alpha=mode_info['alpha'],
                      edgecolor='black',
                      linewidth=0.5)

        # Add percentage labels on bars
        for j, (pa_eq, n_agents) in enumerate(zip(mode_info['pa_equivalents'], all_agent_counts)):
            if pa_eq > 0:
                pct = (pa_eq / n_agents) * 100
                ax.annotate(f"{pct:.0f}%",
                           xy=(x[j] + offset, pa_eq),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center',
                           fontsize=7,
                           rotation=0)

    # Add perfect alternation line
    perfect_line = all_agent_counts
    ax.plot(x, perfect_line, 'r--', label='Perfect Alternation', linewidth=2.5, alpha=0.7, zorder=10)

    # Set labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(all_agent_counts)
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('PA Equivalent Agents', fontsize=12)
    ax.set_title(f'Figure 10: Perfect Alternation Equivalent Agents ({metric})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', ncol=2, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['alt_ratio'], f'pa_equivalent_{metric}_all_modes.png'), dpi=300)
    plt.show()
    plt.close()


def create_alt_ratio_comparison_table(results_list):
    """Create a table comparing ALT ratio estimates across metrics and agent counts."""
    # Get Q-learning results for Type-B ILF
    qlearning_results = [r for r in results_list if r.get('state_type', '') == 'Type-B'
                         and r['reward_type'] == 'ILF' and not r.get('is_random_baseline', False)]
    qlearning_results.sort(key=lambda x: x['num_agents'])

    # Get random results
    random_results = [r for r in results_list if r.get('is_random_baseline', False) == True and r['reward_type'] == 'ILF']
    random_results.sort(key=lambda x: x['num_agents'])

    # Create table data
    table_data = []
    header = ['Agents', 'Metric', 'Q-Learning PA Equivalent', 'Random PA Equivalent', 'Coordination Score']

    metrics = ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']

    for ql_result in qlearning_results:
        n = ql_result['num_agents']

        # Find corresponding random result
        rand_result = next((r for r in random_results if r['num_agents'] == n), None)

        if rand_result:
            for metric in metrics:
                ql_val = ql_result['estimated_agents'].get(metric, 0)
                rand_val = rand_result['estimated_agents'].get(metric, 0)

                if ql_val > 0 or rand_val > 0:
                    coord_score = calculate_coordination_score(
                        ql_result['metrics'].get(metric, 0),
                        rand_result['metrics'].get(metric, 0),
                        1.0  # Perfect value
                    )

                    table_data.append([
                        n,
                        metric,
                        f"{ql_val:.2f} ({ql_val/n:.1%})",
                        f"{rand_val:.2f} ({rand_val/n:.1%})",
                        f"{coord_score:.2f}"
                    ])

    # Create DataFrame
    df = pd.DataFrame(table_data, columns=header)

    # Save to CSV
    csv_path = os.path.join(config.figure_dirs['tables'], 'alt_ratio_comparison.csv')
    df.to_csv(csv_path, index=False)

    # Display table
    display(df)

    # Return DataFrame
    return df

def create_all_alt_ratio_figures(results_list):
    """Create all ALT ratio benchmark figures and tables."""
    metrics = ['FALT', 'EALT', 'CALT']  # Primary metrics for Paper 1

    print("\n== ALT Ratio Analysis ==")

    # Create PA equivalent comparison figures for each primary metric
    for metric in metrics:
        create_alt_ratio_benchmark_comparison(results_list, metric)

    # Create comparative table
    df = create_alt_ratio_comparison_table(results_list)

    return df

#===============================================================================
# Main Execution
#===============================================================================

def run_single_experiment_worker(experiment_params):
    """Worker function to run a single experiment (for multiprocessing)."""
    num_agents, state_type, reward_type, experiment_type = experiment_params

    # Reset seed for reproducibility
    random.seed(42 + num_agents)  # Different seed per agent count
    np.random.seed(42 + num_agents)

    try:
        if state_type == 'Random':
            # Random policy simulation
            print(f"  [Process {os.getpid()}] Running random baseline: {num_agents} agents, {reward_type}")
            num_episodes = 10000
            result = simulate_random_policy(num_agents, num_episodes, reward_type=reward_type)
        elif experiment_type == 'theory':
            # Theory-based episodes calculation
            num_episodes = calculate_episodes_theory_based(num_agents, BASE_EPISODES)
            print(f"  [Process {os.getpid()}] Theory-based: {num_agents} agents, {state_type}, {reward_type} - {num_episodes} episodes")
            result = run_q_learning_experiment(num_agents, num_episodes, state_type, reward_type, decay_schedule='exponential')
        else:
            # Adaptive threshold-based approach
            print(f"  [Process {os.getpid()}] Adaptive: {num_agents} agents, {state_type}, {reward_type}")
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
            print(f"[OK] Completed Q-learning: {num_agents} agents, {state_type}, {reward_type}")

        return result

    except Exception as e:
        print(f"ERROR in worker [Process {os.getpid()}]: {num_agents} agents, {state_type}, {reward_type} - {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_experiments(experiment_type='theory'):
    """Run experiments with selected approach (with optional multiprocessing)."""
    if IN_COLAB:
        keep_colab_alive()

    results_list = load_results()

    # For Paper 1, focus on 2-10 agents
    # PRODUCTION CONFIGURATION: Run benchmark_test.py first to determine optimal settings
    # Adjust based on benchmark results:
    #   - Full: [2, 3, 5, 8, 10, 12] if machine can handle (~92 hours)
    #   - Reduced: [2, 3, 5, 8, 10] if limited time (~67 hours)
    #   - Minimal: [2, 3, 5, 8] if limited resources (~40 hours)
    # Experiment configurations:
    # BASE=100: [2,3] ~2min (quick test with fixes) ✅ TESTING
    # BASE=1000: [2,3,5,8,10] ~40min-1hr (verification)
    # BASE=5000: [2,3,5,8,10] ~3-5hrs (mid-scale)
    # BASE=10000: [2,3,5,8,10] ~7-10hrs (production)
    agent_counts = [2, 3, 5, 8, 10]  # BASE=1000 verification run
    state_types = ['Type-A', 'Type-B']
    reward_types = ['ILF', 'IQF']

    # Generate experiments to run list
    experiments_to_run = []
    for num_agents in agent_counts:
        for state_type in state_types:
            for reward_type in reward_types:
                if not is_experiment_completed(results_list, num_agents, state_type, reward_type):
                    experiments_to_run.append((
                        num_agents, state_type, reward_type, experiment_type
                    ))

    # Add random baselines
    for num_agents in agent_counts:
        for reward_type in reward_types:
            if not is_experiment_completed(results_list, num_agents, None, reward_type, is_random=True):
                experiments_to_run.append((
                    num_agents, 'Random', reward_type, experiment_type
                ))

    if not experiments_to_run:
        print("All experiments completed!")
        return results_list

    print(f"Running {len(experiments_to_run)} experiments with {experiment_type} approach...")

    # Multiprocessing or sequential
    if USE_MULTIPROCESSING and len(experiments_to_run) > 1:
        print(f"Using multiprocessing with {NUM_PROCESSES} parallel processes")
        print("NOTE: Progress bars may overlap - this is normal")

        # Use multiprocessing Pool
        with mp.Pool(processes=NUM_PROCESSES) as pool:
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

def create_agent_performance_summary(results_list):
    """
    Create individual agent performance breakdown figure.

    Shows:
    1. How many times each individual agent reached terminal position
    2. How many exclusive wins each agent got
    3. Comprehensive table of ALL metrics (Traditional + ALT)

    Purpose: Reveal hidden inequality - even with high aggregate fairness,
    individual agents may have very unequal performance.
    """
    # Filter Q-learning results (Type-B ILF for consistency)
    qlearning_results = [r for r in results_list
                         if not r.get('is_random_baseline', False)
                         and r.get('state_type', '') == 'Type-B'
                         and r['reward_type'] == 'ILF'
                         and 'agent_stats' in r]

    qlearning_results.sort(key=lambda x: x['num_agents'])

    if not qlearning_results:
        print("No Q-learning results with agent_stats found")
        return

    # Create figure with 2 bar plots + 1 table
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[1, 0.6])

    # Top: 2 bar plots side by side
    gs_top = gs[0].subgridspec(1, 2, wspace=0.3)
    ax1 = fig.add_subplot(gs_top[0])  # Top reaches per agent
    ax2 = fig.add_subplot(gs_top[1])  # Exclusive wins per agent

    # Bottom: Comprehensive metrics table
    ax3 = fig.add_subplot(gs[1])

    # === PANEL 1: Top Reaches per Agent ===

    # Prepare data - group by agent count
    max_agents = max([r['num_agents'] for r in qlearning_results])
    agent_counts = sorted(set([r['num_agents'] for r in qlearning_results]))

    # For each agent ID, collect reaches across different agent counts
    x_positions = np.arange(max_agents)
    bar_width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_counts)))

    for idx, n_agents in enumerate(agent_counts):
        result = next((r for r in qlearning_results if r['num_agents'] == n_agents), None)
        if result:
            reaches = result['agent_stats']['reaches_per_agent']

            # Plot bars for this agent count
            positions = x_positions[:n_agents] + (idx - len(agent_counts)/2) * bar_width
            ax1.bar(positions, reaches, bar_width,
                   label=f'{n_agents} agents', color=colors[idx], alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Agent ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Reaches to Terminal', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Individual Agent Performance: Top Reaches',
                 fontsize=12, fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'A{i+1}' for i in range(max_agents)])
    # Semi-transparent rounded box legend with enhanced styling and padding
    legend1 = ax1.legend(fontsize=10, loc='best', framealpha=0.85, fancybox=True,
                        shadow=True, borderpad=1.0, labelspacing=0.8)
    legend1.get_frame().set_facecolor('#FFFFFF')  # White background
    legend1.get_frame().set_edgecolor('#2C3E50')  # Dark border color
    legend1.get_frame().set_linewidth(2.0)  # Thicker border for visibility
    ax1.grid(True, alpha=0.3, axis='y')

    # === PANEL 2: Exclusive Wins per Agent ===

    for idx, n_agents in enumerate(agent_counts):
        result = next((r for r in qlearning_results if r['num_agents'] == n_agents), None)
        if result:
            wins = result['agent_stats']['exclusive_wins_per_agent']

            # Plot bars for this agent count
            positions = x_positions[:n_agents] + (idx - len(agent_counts)/2) * bar_width
            ax2.bar(positions, wins, bar_width,
                   label=f'{n_agents} agents', color=colors[idx], alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Agent ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Exclusive Wins', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Individual Agent Performance: Exclusive Wins',
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'A{i+1}' for i in range(max_agents)])
    # Semi-transparent rounded box legend with enhanced styling and padding
    legend2 = ax2.legend(fontsize=10, loc='best', framealpha=0.85, fancybox=True,
                        shadow=True, borderpad=1.0, labelspacing=0.8)
    legend2.get_frame().set_facecolor('#FFFFFF')  # White background
    legend2.get_frame().set_edgecolor('#2C3E50')  # Dark border color
    legend2.get_frame().set_linewidth(2.0)  # Thicker border for visibility
    ax2.grid(True, alpha=0.3, axis='y')

    # === PANEL 3: Comprehensive Metrics Table ===

    # Extract ALL metrics for each agent count
    table_headers = ['Agents', 'Efficiency', 'R_Fair', 'TT_Fair', 'Fairness',
                    'FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']

    table_data = []
    cell_colors = []

    for result in qlearning_results:
        n_agents = result['num_agents']
        m = result['metrics']

        row = [
            f"{n_agents}",
            f"{m['Efficiency']:.3f}",
            f"{m['Reward_Fairness']:.3f}",
            f"{m['TT_Fairness']:.3f}",
            f"{m['Fairness']:.3f}",
            f"{m['FALT']:.3f}",
            f"{m['EALT']:.3f}",
            f"{m['qFALT']:.3f}",
            f"{m['qEALT']:.3f}",
            f"{m['CALT']:.3f}",
            f"{m['AALT']:.3f}"
        ]
        table_data.append(row)

        # Color code cells: green > 0.7, yellow > 0.5, red <= 0.5
        row_colors = ['lightgray']  # Agents column

        for metric_val in [m['Efficiency'], m['Reward_Fairness'], m['TT_Fairness'], m['Fairness'],
                          m['FALT'], m['EALT'], m['qFALT'], m['qEALT'], m['CALT'], m['AALT']]:
            if metric_val > 0.7:
                row_colors.append('lightgreen')
            elif metric_val > 0.5:
                row_colors.append('lightyellow')
            else:
                row_colors.append('lightcoral')

        cell_colors.append(row_colors)

    ax3.axis('off')
    table = ax3.table(cellText=table_data,
                     colLabels=table_headers,
                     cellLoc='center',
                     loc='center',
                     cellColours=cell_colors,
                     colWidths=[0.08] + [0.092]*10)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', fontsize=10)
            cell.set_facecolor('lightsteelblue')

    ax3.set_title('(c) Comprehensive Metrics Summary (Green > 0.7, Yellow > 0.5, Red ≤ 0.5)',
                 fontsize=11, fontweight='bold', pad=15)

    # Overall title
    fig.suptitle('Individual Agent Performance Breakdown',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = os.path.join(config.figure_dirs['comparison'], 'agent_performance_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved agent performance breakdown: {output_path}")

    plt.show()
    plt.close()

def create_all_figures(results_list):
    """Generate all key figures for Paper 1."""
    print("\n== Generating Key Figures for Paper 1 ==")

    # 1. Epsilon vs Metrics (before progression)
    print("\n=== EPSILON VS METRICS ===")
    create_epsilon_vs_metrics_analysis(results_list)

    # 2. Progression Figures (all 6 ALT metrics with epsilon milestone)
    print("\n=== PROGRESSION FIGURES ===")
    print("\nFigure 6a: FALT Progression Rate")
    create_figure_6_metric_progression_rate(results_list, 'FALT')

    print("\nFigure 6b: EALT Progression Rate")
    create_figure_6_metric_progression_rate(results_list, 'EALT')

    print("\nFigure 6c: qFALT Progression Rate")
    create_figure_6_metric_progression_rate(results_list, 'qFALT')

    print("\nFigure 6d: qEALT Progression Rate")
    create_figure_6_metric_progression_rate(results_list, 'qEALT')

    print("\nFigure 6e: CALT Progression Rate")
    create_figure_6_metric_progression_rate(results_list, 'CALT')

    print("\nFigure 6f: AALT Progression Rate")
    create_figure_6_metric_progression_rate(results_list, 'AALT')

    # 3. 3D Consolidated Plots
    print("\n=== 3D CONSOLIDATED PLOTS ===")
    create_3d_metric_progression_plots(results_list)

    # 4. Random vs Q-Learning Comparison
    print("\n=== RANDOM VS Q-LEARNING COMPARISON ===")
    print("\nFigure 8: Random vs Q-Learning Comparison (CALT)")
    create_random_vs_qlearning_comparison(results_list, 'CALT')

    print("\nFigure 9: Random vs Q-Learning Comparison (FALT)")
    create_random_vs_qlearning_comparison(results_list, 'FALT')

    print("\nFigure 9b: Random vs Q-Learning Comparison (AALT)")
    create_random_vs_qlearning_comparison(results_list, 'AALT')

    # 5. Learning Phases Analysis
    print("\n=== LEARNING PHASES ANALYSIS ===")
    create_learning_phases_analysis(results_list)

    # 5.5. Metrics Paradox Summary (NEW - shows traditional metrics fail to capture coordination)
    print("\n=== METRICS PARADOX SUMMARY ===")
    create_agent_performance_summary(results_list)

    # 6. ALT Ratio Analysis (Figure 7 style - all 6 ALT metrics for each mode)
    print("\n=== ALT RATIO ANALYSIS ===")
    print("\nFigure 7: AltRatio Analysis")
    create_figure_7_altratio_analysis(results_list)

    # 7. Computational Cost
    print("\n=== COMPUTATIONAL COST ===")
    create_rp_vs_alt_computation_comparison(results_list)

    # 8. Other Figures
    print("\n=== OTHER FIGURES ===")
    create_random_baseline_progression_plots(results_list)

    print("\nAll figures generated successfully!")

    # CRITICAL: Save comprehensive analysis data FIRST (creates CSV needed by paper1 figures)
    print("\n=== SAVING DATA TABLES ===")
    save_comprehensive_analysis(results_list)
    print("[OK] Data tables saved (comprehensive_results.csv and others)")

    # 9. Paper 1 Figures (NEW - automatic generation)
    print("\n=== PAPER 1 FIGURES ===")
    # Extract base_dir from config.figure_dirs (go up one level from subdirs)
    if hasattr(config, 'figure_dirs') and 'tables' in config.figure_dirs:
        base_dir = os.path.dirname(config.figure_dirs['tables'])
        paper1_output_dir = os.path.join(base_dir, 'paper1_figures')
        print(f"DEBUG: base_dir = {base_dir}")
        print(f"DEBUG: paper1_output_dir = {paper1_output_dir}")
        print(f"Generating Paper 1 figures in: {paper1_output_dir}")
        try:
            create_all_paper1_figures(base_dir, paper1_output_dir)
            print("[OK] Paper 1 figures generated successfully!")
        except Exception as e:
            import traceback
            print(f"[ERROR] Could not generate Paper 1 figures: {e}")
            print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
    else:
        print(f"[SKIP] Paper 1 figures - config.figure_dirs check failed")
        print(f"DEBUG: hasattr(config, 'figure_dirs') = {hasattr(config, 'figure_dirs')}")
        if hasattr(config, 'figure_dirs'):
            print(f"DEBUG: 'tables' in config.figure_dirs = {'tables' in config.figure_dirs}")
            print(f"DEBUG: config.figure_dirs keys = {list(config.figure_dirs.keys())}")

    # 10. Organize Appendix Figures (NEW - automatic organization)
    print("\n=== APPENDIX ORGANIZATION ===")
    if hasattr(config, 'figure_dirs') and 'tables' in config.figure_dirs:
        base_dir = os.path.dirname(config.figure_dirs['tables'])
        print(f"DEBUG: base_dir = {base_dir}")
        print(f"Organizing appendix figures from: {base_dir}")
        try:
            copied = organize_appendix_figures(base_dir, output_dir='appendix')
            total_copied = sum(copied.values())
            print(f"[OK] Organized {total_copied} appendix figures into 6 categories")
            for category, count in copied.items():
                if count > 0:
                    print(f"  - {category}: {count} figures")
        except Exception as e:
            import traceback
            print(f"[ERROR] Could not organize appendix figures: {e}")
            print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
    else:
        print(f"[SKIP] Appendix organization - config.figure_dirs check failed")

#===============================================================================
# Comprehensive Data Saving Functions
#===============================================================================

def save_comprehensive_data_tables(results_list):
    """Save comprehensive data tables for all metrics and experiments."""
    print("\n== Saving Comprehensive Data Tables ==")

    # Table 1: Main results summary
    main_data = []
    for result in results_list:
        row = {
            'Agents': result['num_agents'],
            'State_Type': result.get('state_type', 'Random'),
            'Reward_Type': result['reward_type'],
            'Episodes': result['num_episodes'],
            'Is_Random': result.get('is_random_baseline', False)
        }

        # Add all metrics
        for metric, value in result['metrics'].items():
            row[metric] = value

        # Add ALT ratios
        if 'alt_ratios' in result:
            for metric, ratio in result['alt_ratios'].items():
                row[f'{metric}_AltRatio'] = ratio

        # Add estimated agents
        if 'estimated_agents' in result:
            for metric, agents in result['estimated_agents'].items():
                row[f'{metric}_EstAgents'] = agents

        main_data.append(row)

    df_main = pd.DataFrame(main_data)
    df_main.to_csv(os.path.join(config.figure_dirs['tables'], 'comprehensive_results.csv'), index=False)
    print("[OK] Saved comprehensive_results.csv")

    # Table 2: Progression data for key experiments
    progression_data = []
    for result in results_list:
        if not result.get('is_random_baseline', False) and result.get('state_type') == 'Type-B' and result['reward_type'] == 'ILF':
            for i, episode_point in enumerate(result['episode_points']):
                row = {
                    'Agents': result['num_agents'],
                    'Episode_Point': episode_point,
                    'Progress': episode_point / result['num_episodes']
                }

                # Add progression metrics
                for metric in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness',
                              'FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT', 'Epsilon']:
                    if metric in result['progression_data'] and i < len(result['progression_data'][metric]):
                        row[metric] = result['progression_data'][metric][i]

                progression_data.append(row)

    if progression_data:
        df_progression = pd.DataFrame(progression_data)
        df_progression.to_csv(os.path.join(config.figure_dirs['tables'], 'progression_data.csv'), index=False)
        print("[OK] Saved progression_data.csv")

    # Table 3: Computation times
    computation_data = []
    for result in results_list:
        row = {
            'Agents': result['num_agents'],
            'State_Type': result.get('state_type', 'Random'),
            'Reward_Type': result['reward_type'],
            'Episodes': result['num_episodes']
        }

        if 'computation_times' in result:
            for metric, time_val in result['computation_times'].items():
                row[f'Time_{metric}'] = time_val

        computation_data.append(row)

    df_computation = pd.DataFrame(computation_data)
    df_computation.to_csv(os.path.join(config.figure_dirs['tables'], 'computation_times.csv'), index=False)
    print("[OK] Saved computation_times.csv")

    # Table 4: Coordination analysis
    coordination_data = []
    qlearning_results = [r for r in results_list if r.get('state_type', '') == 'Type-B' and r['reward_type'] == 'ILF' and not r.get('is_random_baseline', False)]
    random_results = [r for r in results_list if r.get('is_random_baseline', False) and r['reward_type'] == 'ILF']

    for ql_result in qlearning_results:
        n = ql_result['num_agents']
        rand_result = next((r for r in random_results if r['num_agents'] == n), None)

        if rand_result:
            for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
                ql_val = ql_result['metrics'].get(metric, 0)
                rand_val = rand_result['metrics'].get(metric, 0)
                coord_score = calculate_coordination_score(ql_val, rand_val, 1.0)

                coordination_data.append({
                    'Agents': n,
                    'Metric': metric,
                    'QL_Value': ql_val,
                    'Random_Value': rand_val,
                    'Coordination_Score': coord_score,
                    'Improvement': ql_val - rand_val,
                    'Improvement_Percent': ((ql_val - rand_val) / rand_val * 100) if rand_val > 0 else 0
                })

    df_coordination = pd.DataFrame(coordination_data)
    df_coordination.to_csv(os.path.join(config.figure_dirs['tables'], 'coordination_analysis.csv'), index=False)
    print("[OK] Saved coordination_analysis.csv")

def save_alt_metrics_analysis(results_list):
    """Save detailed ALT metrics analysis for all experiments."""
    print("\n== Saving ALT Metrics Analysis ==")

    alt_data = []
    for result in results_list:
        base_info = {
            'Agents': result['num_agents'],
            'State_Type': result.get('state_type', 'Random'),
            'Reward_Type': result['reward_type'],
            'Episodes': result['num_episodes'],
            'Is_Random': result.get('is_random_baseline', False)
        }

        # Add ALT metrics values
        for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
            if metric in result['metrics']:
                base_info[f'{metric}_Value'] = result['metrics'][metric]

        # Add ALT ratios and estimated agents
        if 'alt_ratios' in result:
            for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
                if metric in result['alt_ratios']:
                    base_info[f'{metric}_AltRatio'] = result['alt_ratios'][metric]
                    base_info[f'{metric}_EstAgents'] = result['estimated_agents'][metric]
                    base_info[f'{metric}_Percentage'] = result['percentages'][metric]

        alt_data.append(base_info)

    df_alt = pd.DataFrame(alt_data)
    df_alt.to_csv(os.path.join(config.figure_dirs['alt'], 'alt_metrics_analysis.csv'), index=False)
    print("[OK] Saved alt_metrics_analysis.csv")

    # Create ALT metrics comparison plots for all 6 metrics
    for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
        create_alt_metric_comparison_plot(results_list, metric)

def create_alt_metric_comparison_plot(results_list, metric):
    """Create comparison plots for individual ALT metrics."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Get Q-learning and random results
    qlearning_results = [r for r in results_list if r.get('state_type', '') == 'Type-B'
                         and r['reward_type'] == 'ILF' and not r.get('is_random_baseline', False)]
    random_results = [r for r in results_list if r.get('is_random_baseline', False) and r['reward_type'] == 'ILF']

    # Sort and filter
    qlearning_results.sort(key=lambda x: x['num_agents'])
    random_results.sort(key=lambda x: x['num_agents'])
    qlearning_results = [r for r in qlearning_results if r['num_agents'] >= 2]
    random_results = [r for r in random_results if r['num_agents'] >= 2]

    # Extract data
    ql_agents = [r['num_agents'] for r in qlearning_results]
    ql_values = [r['metrics'].get(metric, 0) for r in qlearning_results]

    rand_agents = [r['num_agents'] for r in random_results]
    rand_values = [r['metrics'].get(metric, 0) for r in random_results]

    # Plot
    ax.plot(ql_agents, ql_values, 's-', label='Q-Learning', color=COLOR_PALETTE.get(metric, 'blue'), markersize=5)
    ax.plot(rand_agents, rand_values, 'o--', label='Random Policy', color=COLOR_PALETTE.get('Random', 'black'), markersize=5)

    # Perfect line - make it more prominent
    min_agent = min(ql_agents + rand_agents) if ql_agents + rand_agents else 2
    max_agent = max(ql_agents + rand_agents) if ql_agents + rand_agents else 10
    agents_range = list(range(min_agent, max_agent + 1))
    perfect_values = [1.0] * len(agents_range)
    ax.plot(agents_range, perfect_values, 'r-', label='Perfect Alternation', linewidth=2.5, zorder=10)

    ax.set_xlabel('Number of Agents')
    ax.set_ylabel(f'{metric} Value')
    ax.set_title(f'{metric}: Q-Learning vs Random Policy')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['comparison'], f'{metric}_comparison.png'), dpi=300)
    plt.close()

def save_rp_metrics_analysis(results_list):
    """Save RP metrics analysis."""
    print("\n== Saving RP Metrics Analysis ==")

    rp_data = []
    for result in results_list:
        if 'rp_values' in result:
            row = {
                'Agents': result['num_agents'],
                'State_Type': result.get('state_type', 'Random'),
                'Reward_Type': result['reward_type'],
                'RP_Avg': result['metrics'].get('RP_avg', 0),
                'AWE_Avg': result['metrics'].get('AWE_avg', 0),
                'WPE_Avg': result['metrics'].get('WPE_avg', 0)
            }

            # Add per-agent RP metrics
            rp_values = result['rp_values']
            for i in range(result['num_agents']):
                if i < len(rp_values['RP_per_agent']):
                    row[f'Agent_{i}_RP'] = rp_values['RP_per_agent'][i]
                    row[f'Agent_{i}_AWE'] = rp_values['AWE_per_agent'][i]
                    row[f'Agent_{i}_WPE'] = rp_values['WPE_per_agent'][i]

            rp_data.append(row)

    if rp_data:
        df_rp = pd.DataFrame(rp_data)
        df_rp.to_csv(os.path.join(config.figure_dirs['tables'], 'rp_metrics_analysis.csv'), index=False)
        print("[OK] Saved rp_metrics_analysis.csv")

def save_computation_times_analysis(results_list):
    """Save detailed computation times analysis."""
    print("\n== Saving Computation Times Analysis ==")

    time_data = []
    for result in results_list:
        if 'computation_times' in result:
            row = {
                'Agents': result['num_agents'],
                'State_Type': result.get('state_type', 'Random'),
                'Reward_Type': result['reward_type'],
                'Episodes': result['num_episodes']
            }

            times = result['computation_times']
            for metric, time_val in times.items():
                row[metric] = time_val

            time_data.append(row)

    df_times = pd.DataFrame(time_data)
    df_times.to_csv(os.path.join(config.figure_dirs['computation_times'], 'detailed_computation_times.csv'), index=False)
    print("[OK] Saved detailed_computation_times.csv")

    # Create computation time plots
    create_computation_time_plots(results_list)

    # Create RP vs ALT computation times comparison
    create_rp_vs_alt_computation_comparison(results_list)

    # Create bar chart overview
    create_computation_time_bar_chart_overview(results_list)

def create_computation_time_plots(results_list):
    """Create plots showing computation times."""
    # ALT metrics computation times
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Filter Q-learning results
    ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                  and r.get('state_type') == 'Type-B' and r['reward_type'] == 'ILF']
    ql_results.sort(key=lambda x: x['num_agents'])

    if ql_results:
        agents = [r['num_agents'] for r in ql_results]

        # Panel 1: Individual ALT metric times
        alt_metrics = ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']
        for metric in alt_metrics:
            times = [r['computation_times'].get(metric, 0) for r in ql_results]
            ax1.plot(agents, times, 'o-', label=metric, markersize=4)

        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Computation Time (seconds)')
        ax1.set_title('(a) ALT Metrics Computation Times')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # Panel 2: Total times by category
        categories = ['ALT_Total', 'RP_Total', 'Traditional_Total']
        category_times = {cat: [] for cat in categories}

        for r in ql_results:
            alt_total = r['computation_times'].get('Total', 0)
            rp_total = r['computation_times'].get('RP_Total', 0)
            trad_total = sum([r['computation_times'].get(m, 0) for m in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness']])

            category_times['ALT_Total'].append(alt_total)
            category_times['RP_Total'].append(rp_total)
            category_times['Traditional_Total'].append(trad_total)

        x = np.arange(len(agents))
        width = 0.25
        for i, (category, times) in enumerate(category_times.items()):
            ax2.bar(x + i * width, times, width, label=category)

        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title('(b) Total Computation Times by Category')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(agents)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(config.figure_dirs['computation_times'], 'computation_times_analysis.png'), dpi=300)
        plt.close()

def create_rp_vs_alt_computation_comparison(results_list):
    """Create RP vs ALT computation times comparison as agents scale."""
    print("\n== Creating RP vs ALT Computation Times Comparison ==")

    # Filter Q-learning results
    ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                  and r.get('state_type') == 'Type-B' and r['reward_type'] == 'ILF']
    ql_results.sort(key=lambda x: x['num_agents'])

    if not ql_results:
        return

    agents = [r['num_agents'] for r in ql_results]

    # Extract detailed computation times
    rp_avg_times = []
    rp_awe_times = []
    rp_wpe_times = []
    efficiency_times = []
    reward_fairness_times = []
    tt_fairness_times = []
    fairness_times = []
    traditional_times = []

    # Full times (with dependencies) - for BOTH panels and stacked segments
    falt_times = []
    ealt_times = []
    qfalt_times = []
    qealt_times = []
    calt_times = []
    aalt_times = []
    data_collection_times = []

    for r in ql_results:
        comp_times = r.get('computation_times', {})

        # RP metrics (RP_avg = AWE + WPE)
        rp_awe = comp_times.get('RP_AWE', 0)
        rp_wpe = comp_times.get('RP_WPE', 0)
        rp_awe_times.append(rp_awe)
        rp_wpe_times.append(rp_wpe)
        rp_avg_times.append(rp_awe + rp_wpe)

        # Traditional metrics (separate)
        eff_t = comp_times.get('Efficiency', 0)
        rf_t = comp_times.get('Reward_Fairness', 0)
        tt_t = comp_times.get('TT_Fairness', 0)
        fair_t = comp_times.get('Fairness', 0)
        efficiency_times.append(eff_t)
        reward_fairness_times.append(rf_t)
        tt_fairness_times.append(tt_t)
        fairness_times.append(fair_t)
        traditional_times.append(eff_t + rf_t + tt_t + fair_t)

        # ALT metrics - Full times (with dependencies) for segments
        falt_full = comp_times.get('FALT', 0)
        ealt_full = comp_times.get('EALT', 0)
        qfalt_full = comp_times.get('qFALT', 0)
        qealt_full = comp_times.get('qEALT', 0)
        calt_full = comp_times.get('CALT', 0)
        aalt_full = comp_times.get('AALT', 0)
        data_coll = comp_times.get('data_collection', 0)

        falt_times.append(falt_full)
        ealt_times.append(ealt_full)
        qfalt_times.append(qfalt_full)
        qealt_times.append(qealt_full)
        calt_times.append(calt_full)
        aalt_times.append(aalt_full)
        data_collection_times.append(data_coll)

    # Calculate totals for plotting
    rp_times = [awe + wpe for awe, wpe in zip(rp_awe_times, rp_wpe_times)]

    # ALT_Total: Sum of all full times MINUS 5× data_collection (counted 6 times in segments, keep only 1)
    alt_times = []
    for i in range(len(falt_times)):
        segment_sum = falt_times[i] + ealt_times[i] + qfalt_times[i] + qealt_times[i] + calt_times[i] + aalt_times[i]
        # Each segment includes data_collection, so we counted it 6 times
        # Real total = segment_sum - 5 × data_collection
        alt_total_corrected = segment_sum - (5 * data_collection_times[i])
        alt_times.append(alt_total_corrected)

    # Create comparison plot with detailed breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Line plot with individual metrics breakdown

    # Helper function to replace 0.0 with small value for log scale plotting
    def plot_safe(ax, x, y, **kwargs):
        """Plot with 0.0 values replaced by 1e-6 for log scale."""
        y_safe = [max(val, 1e-6) for val in y]
        ax.plot(x, y_safe, **kwargs)

    # Traditional metrics (green - NO markers)
    plot_safe(ax1, agents, traditional_times, linestyle='-',
              label='Traditional Total', linewidth=3, color='#2ca02c')
    plot_safe(ax1, agents, efficiency_times, linestyle='--',
              label='Efficiency', linewidth=2, color='#90ee90')
    plot_safe(ax1, agents, reward_fairness_times, linestyle='-.',
              label='Reward_Fairness', linewidth=2, color='#32cd32')
    # TT_Fairness and Fairness (yellow, yellowgreen, semi-transparent, dashed)
    plot_safe(ax1, agents, tt_fairness_times, linestyle='--',
              label='TT_Fairness', linewidth=2, color='#FFD700', alpha=0.7)
    plot_safe(ax1, agents, fairness_times, linestyle='-.',
              label='Fairness', linewidth=2, color='#9ACD32', alpha=0.7)

    # RP metrics - RP_avg: SOLID BLUE
    plot_safe(ax1, agents, rp_avg_times, linestyle='-', linewidth=3, color='navy', alpha=0.9, zorder=10,
              label='RP_avg')
    plot_safe(ax1, agents, rp_awe_times, linestyle='--',
              label='RP_AWE', linewidth=2, color='#1f77b4')
    plot_safe(ax1, agents, rp_wpe_times, linestyle='-.',
              label='RP_WPE', linewidth=2, color='#6baed6')

    # ALT Total (red thick) - Calculate from full times
    alt_total_times = [f + e + qf + qe + c + a for f, e, qf, qe, c, a in
                       zip(falt_times, ealt_times, qfalt_times, qealt_times, calt_times, aalt_times)]
    plot_safe(ax1, agents, alt_total_times, linestyle='-',
              label='ALT Total', linewidth=3, color='#dc143c')

    # Individual ALT metrics (pink→red→brown gradient, dashed patterns)
    # Use FULL times (with dependencies) for left panel
    alt_metrics_names = ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']
    alt_metrics_times = [falt_times, ealt_times, qfalt_times, qealt_times, calt_times, aalt_times]
    alt_colors = ['#ff69b4', '#ff1493', '#dc143c', '#cd5c5c', '#a52a2a', '#8b4513']
    alt_line_styles = ['-', '--', '-.', ':', '--', '-.']
    for i, (metric_name, metric_times) in enumerate(zip(alt_metrics_names, alt_metrics_times)):
        plot_safe(ax1, agents, metric_times, linestyle=alt_line_styles[i],
                  label=metric_name, linewidth=1.5, color=alt_colors[i], alpha=0.9)

    ax1.set_xlabel('Number of Agents', fontsize=11)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=11)
    ax1.set_yscale('log')  # Logarithmic scale
    ax1.set_title('(a) Individual Metric Computation Times (Log Scale)', fontsize=11)
    # Legend outside plot area to avoid overlap with curves
    legend1 = ax1.legend(fontsize=8, ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5),
                        frameon=True, fancybox=True,
                        framealpha=0.95, edgecolor='gray',
                        facecolor='white')
    legend1.get_frame().set_boxstyle('round', pad=0.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: OVERLAPPED bar chart (all from base, tallest bars visible on top)
    x = np.arange(len(agents))
    width = 0.6

    # Collect all metrics with their full times for sorting
    all_metrics = []

    # Traditional
    all_metrics.append(('Traditional', traditional_times, '#2ca02c'))

    # RP metrics - Add RP_avg explicitly
    all_metrics.append(('RP_avg', rp_avg_times, 'navy'))
    all_metrics.append(('RP_AWE', rp_awe_times, '#1f77b4'))
    all_metrics.append(('RP_WPE', rp_wpe_times, '#6baed6'))

    # ALT metrics - FULL times (with dependencies)
    alt_metrics_names = ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']
    alt_metrics_times = [falt_times, ealt_times, qfalt_times, qealt_times, calt_times, aalt_times]
    alt_colors = ['#ff69b4', '#ff1493', '#dc143c', '#cd5c5c', '#a52a2a', '#8b4513']

    for name, times, color in zip(alt_metrics_names, alt_metrics_times, alt_colors):
        all_metrics.append((name, times, color))

    # Sort by DESCENDING average time (tallest bars plotted FIRST, shortest LAST on top)
    all_metrics_sorted = sorted(all_metrics, key=lambda m: np.mean(m[1]), reverse=True)

    # Plot all bars from same base (y=0) - tallest first, shortest last
    for metric_name, metric_times, color in all_metrics_sorted:
        ax2.bar(x, metric_times, width, label=metric_name, color=color,
                alpha=0.75, edgecolor='white', linewidth=0.8, zorder=100-np.mean(metric_times)*1000)

    # Add trend lines for Traditional, RP, and ALT separately (NOT total)
    # Traditional trend (green) - semi-transparent dashed
    ax2.plot(x, traditional_times, 'o--', color='#2ca02c', linewidth=1.5, markersize=5,
             alpha=0.4, zorder=1000)

    # RP trend (blue) - semi-transparent dashed
    ax2.plot(x, rp_times, 'o--', color='navy', linewidth=1.5, markersize=5,
             alpha=0.4, zorder=1000)

    # ALT trend (red) - semi-transparent dashed
    ax2.plot(x, alt_times, 'o--', color='#dc143c', linewidth=1.5, markersize=5,
             alpha=0.4, zorder=1000)

    ax2.set_xlabel('Number of Agents', fontsize=11)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=11)
    ax2.set_title('(b) Overlapped Computation Times (Tallest Visible)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents)

    # Legend positioned to avoid overlap, sorted by descending time
    legend2 = ax2.legend(fontsize=7, ncol=2, loc='upper left',
                        frameon=True, fancybox=True,
                        framealpha=0.95, edgecolor='gray',
                        facecolor='white')
    legend2.get_frame().set_boxstyle('round', pad=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['computation_times'], 'rp_vs_alt_computation_comparison.png'), dpi=300)
    plt.show()
    plt.close()
    print("[OK] Saved RP vs ALT computation comparison")

    # Save detailed data table with submetrics
    comp_data = []
    for i, n in enumerate(agents):
        # ALT total corrected (data_collection counted only once)
        total_time = traditional_times[i] + rp_times[i] + alt_times[i]

        comp_data.append({
            'Agents': n,
            # RP submetrics
            'RP_AWE': rp_awe_times[i],
            'RP_WPE': rp_wpe_times[i],
            'RP_Total': rp_times[i],
            # ALT metrics - FULL times (with dependencies, as used in segments)
            'FALT_Full': falt_times[i],
            'EALT_Full': ealt_times[i],
            'qFALT_Full': qfalt_times[i],
            'qEALT_Full': qealt_times[i],
            'CALT_Full': calt_times[i],
            'AALT_Full': aalt_times[i],
            # Data collection (shared overhead)
            'Data_Collection': data_collection_times[i],
            # ALT Total - CORRECTED (data_collection counted only 1×)
            'ALT_Total_Corrected': alt_times[i],
            # Traditional
            'Traditional_Time': traditional_times[i],
            # Totals
            'Total_Time': total_time,
            'RP_Percentage': (rp_times[i] / total_time * 100) if total_time > 0 else 0,
            'ALT_Percentage': (alt_times[i] / total_time * 100) if total_time > 0 else 0
        })

    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(os.path.join(config.figure_dirs['computation_times'], 'rp_vs_alt_comparison.csv'), index=False)
    print("[OK] Saved RP vs ALT comparison data table with detailed submetrics")


def create_computation_time_bar_chart_overview(results_list):
    """Create simple bar chart overview comparing Traditional, RP, and Best ALT (CALT) computation times."""
    print("\n== Creating Computation Time Bar Chart Overview ==")

    # Filter Q-learning results (Type-B ILF as representative)
    ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                  and r.get('state_type') == 'Type-B' and r['reward_type'] == 'ILF']
    ql_results.sort(key=lambda x: x['num_agents'])

    if not ql_results:
        print("  [WARNING] No Q-learning results found for bar chart")
        return

    agents = [r['num_agents'] for r in ql_results]

    # Extract times: Traditional (Eff + RF), RP total, CALT (representative ALT)
    traditional_times = []
    rp_times = []
    calt_times = []

    for r in ql_results:
        comp_times = r.get('computation_times', {})

        # Traditional: Efficiency + Reward_Fairness only (the two main ones used in paper)
        trad_time = comp_times.get('Efficiency', 0) + comp_times.get('Reward_Fairness', 0)
        traditional_times.append(trad_time)

        # RP: Total (AWE + WPE)
        rp_time = comp_times.get('RP_AWE', 0) + comp_times.get('RP_WPE', 0)
        rp_times.append(rp_time)

        # Best/Most-Used ALT: CALT (representative - most comprehensive)
        calt_time = comp_times.get('CALT', 0)
        calt_times.append(calt_time)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(agents))
    width = 0.25

    # Plot 3 bars per agent count
    bars1 = ax.bar(x - width, traditional_times, width,
                   label='Traditional (Eff + RF)', color='#2ca02c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, rp_times, width,
                   label='RP (AWE + WPE)', color='#1f77b4', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, calt_times, width,
                   label='Best ALT (CALT)', color='#dc143c', alpha=0.8, edgecolor='black')

    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, rotation=0)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    ax.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Computation Time Comparison: Traditional vs RP vs Best ALT (CALT)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95, fancybox=True, shadow=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')  # Log scale to see differences clearly

    # Add note about practical usage - moved higher to avoid overlap with x-axis
    note_text = "Note: In practice, use 1-2 ALT metrics (e.g., CALT), not all 6"
    ax.text(0.98, 0.15, note_text, transform=ax.transAxes,
            fontsize=10, style='italic', ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['computation_times'], 'computation_time_bar_chart_overview.png'), dpi=300)
    plt.show()
    plt.close()
    print("[OK] Saved computation time bar chart overview")


def save_all_alt_ratio_analysis(results_list):
    """Save ALT ratio analysis for all 6 metrics."""
    print("\n== Saving Complete ALT Ratio Analysis ==")

    # Create PA equivalent plots for all 6 ALT metrics
    for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
        create_complete_alt_ratio_plot(results_list, metric)

def create_complete_alt_ratio_plot(results_list, metric):
    """Create complete ALT ratio analysis for a specific metric."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Get Q-learning results
    qlearning_results = [r for r in results_list if r.get('state_type', '') == 'Type-B'
                         and r['reward_type'] == 'ILF' and not r.get('is_random_baseline', False)]
    qlearning_results.sort(key=lambda x: x['num_agents'])

    if not qlearning_results:
        return

    # Extract data
    agent_counts = [r['num_agents'] for r in qlearning_results]
    pa_equivalents = [r['estimated_agents'].get(metric, 0) for r in qlearning_results]
    percentages = [eq / n * 100 for eq, n in zip(pa_equivalents, agent_counts)]

    # Create plot
    x = np.arange(len(agent_counts))
    ax.bar(x, pa_equivalents, width=0.4, label=f'PA Equivalent ({metric})', alpha=0.7,
           color=COLOR_PALETTE.get(metric))

    # Add perfect line
    ax.plot(x, agent_counts, 'r--', label='Perfect Alternation', linewidth=2)

    # Add percentage labels
    for i, (eq, pct) in enumerate(zip(pa_equivalents, percentages)):
        ax.annotate(f"{pct:.1f}%",
                   xy=(i, eq),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('PA Equivalent Agents')
    ax.set_title(f'Perfect Alternation Equivalent Agents ({metric})')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.figure_dirs['alt_ratio'], f'pa_equivalent_{metric}.png'), dpi=300)
    plt.close()

def save_random_baselines_analysis(results_list):
    """Save random baselines analysis."""
    print("\n== Saving Random Baselines Analysis ==")

    random_data = []
    for result in results_list:
        if result.get('is_random_baseline', False):
            row = {
                'Agents': result['num_agents'],
                'Reward_Type': result['reward_type'],
                'Episodes': result['num_episodes'],
                'Avg_Rounds': result.get('avg_rounds', 0)
            }

            # Add all metrics
            for metric, value in result['metrics'].items():
                row[metric] = value

            random_data.append(row)

    if random_data:
        df_random = pd.DataFrame(random_data)
        df_random.to_csv(os.path.join(config.figure_dirs['random'], 'random_baselines_analysis.csv'), index=False)
        print("[OK] Saved random_baselines_analysis.csv")

def create_learning_phases_analysis(results_list):
    """Create Traditional vs ALT metrics comparison across learning phases (2-panel)."""
    print("\n== Creating Learning Phases Analysis ==")

    # Process all combinations of state_type and reward_type
    for state_type in ['Type-A', 'Type-B']:
        for reward_type in ['ILF', 'IQF']:
            ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                          and r.get('state_type') == state_type and r['reward_type'] == reward_type
                          and 'progression_data' in r]

            for result in ql_results:
                n_agents = result['num_agents']
                n_episodes = result['num_episodes']
                epsilon_decay_target = result.get('q_learning_info', {}).get('epsilon_decay_target', EPSILON_DECAY_TARGET)

                # Calculate episode indices for different phases
                first_10pct_idx = int(n_episodes * 0.1)
                epsilon_phase_idx = int(n_episodes * epsilon_decay_target)
                last_10pct_idx = int(n_episodes * 0.9)

                progression_data = result['progression_data']
                episode_points = result['episode_points']

                # Find all points in first 10% (for averaging)
                first_10_indices = [i for i, ep in enumerate(episode_points) if ep <= first_10pct_idx]

                # Find closest indices for other phases
                epsilon_idx = min(range(len(episode_points)),
                                  key=lambda i: abs(episode_points[i] - epsilon_phase_idx))
                last_idx = min(range(len(episode_points)),
                              key=lambda i: abs(episode_points[i] - last_10pct_idx))

                # Calculate phase values (average for first 10%, single value for others)
                phase_values = {}
                for metric in list(progression_data.keys()):
                    if len(progression_data[metric]) > 0:
                        # Before ε-decay: AVERAGE of first 10%
                        before_values = [progression_data[metric][i] for i in first_10_indices if i < len(progression_data[metric])]
                        before_avg = np.mean(before_values) if before_values else progression_data[metric][0]

                        # After ε-decay and Final: single values
                        after_value = progression_data[metric][epsilon_idx] if epsilon_idx < len(progression_data[metric]) else 0
                        final_value = progression_data[metric][last_idx] if last_idx < len(progression_data[metric]) else 0

                        phase_values[metric] = [before_avg, after_value, final_value]

                # Define phase names
                phase_names = [
                    'Before ε-decay\n(exploration)',
                    f'After ε-decay\n({epsilon_decay_target:.0%} episodes)',
                    'Final 10%\n(convergence)'
                ]

                # Create phases mapping for data table (using representative indices)
                phases = {
                    'Before ε-decay\n(exploration)': first_10_indices[len(first_10_indices)//2] if first_10_indices else 0,
                    f'After ε-decay\n({epsilon_decay_target:.0%} episodes)': epsilon_idx,
                    'Final 10%\n(convergence)': last_idx
                }

                # Create comparison plot with 1 row, 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                x = np.arange(len(phase_names))

                # Panel 1: Traditional Metrics (4 bars per phase)
                width = 0.2
                for i, metric in enumerate(['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness']):
                    values = phase_values.get(metric, [0, 0, 0])
                    ax1.bar(x + i * width - width*1.5, values, width, label=metric,
                           color=COLOR_PALETTE.get(metric))

                ax1.set_ylabel('Metric Value', fontsize=11)
                ax1.set_title(f'(a) Traditional Metrics - {state_type} {reward_type} ({n_agents} agents)', fontsize=11)
                ax1.set_xticks(x)
                ax1.set_xticklabels(phase_names, fontsize=9)
                ax1.legend(fontsize=9, loc='best')
                ax1.grid(axis='y', alpha=0.3)
                ax1.set_ylim([0, 1.05])

                # Panel 2: ALT Metrics (6 bars per phase)
                width = 0.13
                for i, metric in enumerate(['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']):
                    values = phase_values.get(metric, [0, 0, 0])
                    ax2.bar(x + i * width - width*2.5, values, width, label=metric,
                           color=COLOR_PALETTE.get(metric))

                ax2.set_ylabel('ALT Metric Value', fontsize=11)
                ax2.set_title(f'(b) ALT Metrics - {state_type} {reward_type} ({n_agents} agents)', fontsize=11)
                ax2.set_xticks(x)
                ax2.set_xticklabels(phase_names, fontsize=9)
                ax2.legend(fontsize=9, loc='best', ncol=2)
                ax2.grid(axis='y', alpha=0.3)
                ax2.set_ylim([0, 1.05])

                plt.tight_layout()
                filename = f'learning_phases_{state_type}_{reward_type}_{n_agents}agents.png'
                plt.savefig(os.path.join(config.figure_dirs['learning_phases'], filename), dpi=300)
                plt.show()
                plt.close()
                print(f"[OK] Saved learning phases for {state_type} {reward_type} {n_agents} agents")

                # Create data table
                phase_data = []
                for phase_name in phase_names:
                    idx = phases[phase_name]
                    row = {
                        'Agents': n_agents,
                        'State_Type': state_type,
                        'Reward_Type': reward_type,
                        'Phase': phase_name.replace('\n', ' '),
                        'Episode': episode_points[idx],
                        'Episode_Pct': episode_points[idx] / n_episodes * 100
                    }

                    # Add all metrics
                    for metric in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness',
                                  'FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT', 'Epsilon']:
                        if metric in progression_data and idx < len(progression_data[metric]):
                            row[metric] = progression_data[metric][idx]

                    phase_data.append(row)

                df_phases = pd.DataFrame(phase_data)
                filename = f'learning_phases_data_{state_type}_{reward_type}_{n_agents}agents.csv'
                df_phases.to_csv(os.path.join(config.figure_dirs['learning_phases'], filename), index=False)

    print("[OK] Learning phases analysis completed")

def create_epsilon_vs_metrics_analysis(results_list):
    """Create epsilon progression vs metrics (2-panel: Traditional | ALT)."""
    print("\n== Creating Epsilon vs Metrics Analysis ==")

    # Process all combinations of state_type, reward_type, and agents
    for state_type in ['Type-A', 'Type-B']:
        for reward_type in ['ILF', 'IQF']:
            ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                          and r.get('state_type') == state_type and r['reward_type'] == reward_type
                          and 'progression_data' in r and 'Epsilon' in r['progression_data']]

            for result in ql_results:
                n_agents = result['num_agents']
                n_episodes = result['num_episodes']
                progression_data = result['progression_data']
                episode_points = result['episode_points']

                # Normalize episode points to 0-1 for x-axis
                x_vals = [e / n_episodes for e in episode_points]

                # Create figure with 1 row, 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Panel 1: Epsilon + Traditional Metrics
                ax1_color = 'tab:blue'
                ax1.plot(x_vals, progression_data['Epsilon'], color=ax1_color, linewidth=2.5, label='Epsilon')
                ax1.set_xlabel('Episode Progress', fontsize=11)
                ax1.set_ylabel('Epsilon', color=ax1_color, fontsize=11)
                ax1.tick_params(axis='y', labelcolor=ax1_color)
                ax1.set_ylim([0, 1.0])  # Fixed: was [-0.05, 1.0]

                ax1_twin = ax1.twinx()
                for metric in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness']:
                    ax1_twin.plot(x_vals, progression_data[metric],
                                 label=metric, linewidth=1.5,
                                 color=COLOR_PALETTE.get(metric), alpha=0.8)
                ax1_twin.set_ylabel('Metric Value', fontsize=11)
                ax1_twin.set_ylim([0, 1.05])
                ax1.set_title(f'(a) Traditional Metrics - {state_type} {reward_type} ({n_agents} agents)', fontsize=11)
                ax1.grid(True, alpha=0.3)

                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

                # Panel 2: Epsilon + ALT Metrics
                ax2_color = 'tab:blue'
                ax2.plot(x_vals, progression_data['Epsilon'], color=ax2_color, linewidth=2.5, label='Epsilon')
                ax2.set_xlabel('Episode Progress', fontsize=11)
                ax2.set_ylabel('Epsilon', color=ax2_color, fontsize=11)
                ax2.tick_params(axis='y', labelcolor=ax2_color)
                ax2.set_ylim([0, 1.0])  # Fixed: was [-0.05, 1.0]

                ax2_twin = ax2.twinx()
                for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
                    ax2_twin.plot(x_vals, progression_data[metric],
                                 label=metric, linewidth=1.5,
                                 color=COLOR_PALETTE.get(metric), alpha=0.8)
                ax2_twin.set_ylabel('ALT Metric Value', fontsize=11)
                ax2_twin.set_ylim([0, 1.05])
                ax2.set_title(f'(b) ALT Metrics - {state_type} {reward_type} ({n_agents} agents)', fontsize=11)
                ax2.grid(True, alpha=0.3)

                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

                plt.tight_layout()
                filename = f'epsilon_vs_metrics_{state_type}_{reward_type}_{n_agents}agents.png'
                plt.savefig(os.path.join(config.figure_dirs['learning_phases'], filename), dpi=300)
                plt.show()
                plt.close()
                print(f"[OK] Saved epsilon vs metrics for {state_type} {reward_type} {n_agents} agents")

    print("[OK] Epsilon vs metrics analysis completed")

def create_3d_metric_progression_plots(results_list):
    """Create 4 separate 3D plots - one for each mode (Type-A/B × ILF/IQF).

    Each plot shows all 6 ALT metrics + Efficiency + Reward_Fairness with random baselines.
    """
    from mpl_toolkits.mplot3d import Axes3D

    print("\n== Creating 4 Separate 3D Metric Progression Plots ==")

    # Metrics to plot: 6 ALT + Efficiency + Reward_Fairness (exclude RP)
    metrics_to_plot = ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT', 'Efficiency', 'Reward_Fairness']

    # Grouped colors: Traditional (green), ALT (distinct red-toned colors)
    metric_colors = {
        'Efficiency': '#2ca02c',        # Green
        'Reward_Fairness': '#90ee90',   # Light green
        'FALT': '#ff1493',              # Deep pink
        'EALT': '#ff7f00',              # Orange
        'qFALT': '#ff0000',             # Red
        'qEALT': '#d2691e',             # Chocolate
        'CALT': '#8b4513',              # Saddle brown
        'AALT': '#9370db'               # Medium purple
    }

    # Line styles - all solid
    line_styles = {
        'FALT': '-', 'EALT': '-', 'qFALT': '-',
        'qEALT': '-', 'CALT': '-', 'AALT': '-',
        'Efficiency': '-', 'Reward_Fairness': '-'
    }

    # Define 4 modes
    modes = [
        ('Type-A', 'ILF'),
        ('Type-A', 'IQF'),
        ('Type-B', 'ILF'),
        ('Type-B', 'IQF')
    ]

    for state_type, reward_type in modes:
        # Get Q-learning results for this mode
        ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                      and r.get('state_type', '') == state_type
                      and r['reward_type'] == reward_type
                      and 'progression_data' in r]

        # Get random baseline for this reward type
        random_results = [r for r in results_list if r.get('is_random_baseline', False)
                          and r['reward_type'] == reward_type]

        if not ql_results:
            print(f"No Q-learning results found for {state_type} {reward_type}")
            continue

        # Sort by number of agents
        ql_results.sort(key=lambda x: x['num_agents'])
        random_results.sort(key=lambda x: x['num_agents'])

        # Create plot for this mode
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each metric for all agent counts
        for metric in metrics_to_plot:
            metric_color = metric_colors.get(metric, 'gray')
            metric_linestyle = line_styles.get(metric, '-')

            # Plot Q-learning lines for this metric
            for result in ql_results:
                if metric not in result['progression_data']:
                    continue

                n_agents = result['num_agents']
                n_episodes = result['num_episodes']
                progression_data = result['progression_data']
                episode_points = result['episode_points']

                # Normalize episodes to 0-1
                x_vals = np.array([e / n_episodes for e in episode_points])
                y_vals = np.array(progression_data[metric])
                z_vals = np.full_like(x_vals, n_agents)

                # Plot as line in 3D space with consistent color per metric
                # Only add label for first agent count
                label = f'{metric} ({n_agents}a)' if n_agents == ql_results[0]['num_agents'] else None
                ax.plot(x_vals, z_vals, y_vals,
                       label=label,
                       linewidth=2, alpha=0.85,
                       color=metric_color,
                       linestyle=metric_linestyle)

            # Add random baseline as small markers at x=0
            for random_result in random_results:
                n_agents = random_result['num_agents']
                random_value = random_result['metrics'].get(metric, 0)

                # Place marker at episode start (x=0)
                ax.scatter([0], [n_agents], [random_value],
                          marker='x', s=60, color=metric_color,
                          alpha=0.6, linewidths=2)

        # Labels and title
        ax.set_xlabel('Episode Progress (normalized)', fontsize=11)
        ax.set_ylabel('Number of Agents', fontsize=11)
        ax.set_zlabel('Metric Value', fontsize=11)
        ax.set_title(f'3D Consolidated Metric Progression - {state_type} {reward_type}\n(6 ALT + Efficiency + Reward Fairness)', fontsize=13)
        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.set_zlim([0, 1.05])

        # Save plot
        filename = f'3d_consolidated_metrics_{state_type}_{reward_type}.png'
        plt.savefig(os.path.join(config.figure_dirs['3d_analysis'], filename), dpi=300)
        plt.show()
        plt.close()

        print(f"[OK] Saved 3D plot for {state_type} {reward_type}: {filename}")

    print("[OK] All 4 3D plots created successfully!")


#===============================================================================
# Q-Table Learning Analysis
#===============================================================================

def save_q_table_analysis(result):
    """
    Export Q-table learning analysis for a single experiment.
    Tracks most AND least visited states to understand learning patterns.
    """
    if 'state_visits' not in result or not result['state_visits']:
        print(f"  [WARNING] No state_visits data available for this experiment")
        return

    num_agents = result['num_agents']
    state_type = result.get('state_type', 'Unknown')
    reward_type = result['reward_type']
    state_visits = result['state_visits']

    # Create all_tables directory if it doesn't exist
    tables_dir = os.path.join(config.base_dir, 'all_tables')
    os.makedirs(tables_dir, exist_ok=True)

    # Organize states by agent
    # New format: state_visits = {agent_id: {state: visit_count}}
    # We need to get Q-values from the Q-table
    q_tables = result.get('q_tables', {})

    agent_states = {}
    for agent_id, states_dict in state_visits.items():
        if agent_id not in agent_states:
            agent_states[agent_id] = []

        # Get Q-table for this agent if available
        q_table = q_tables.get(agent_id, None)

        for state_idx, visit_count in states_dict.items():
            # Get max Q-value for this state if Q-table exists
            if q_table is not None and state_idx < len(q_table):
                max_q_value = np.max(q_table[state_idx])
                best_action = np.argmax(q_table[state_idx])
            else:
                max_q_value = 0.0
                best_action = 0

            agent_states[agent_id].append({
                'state': state_idx,
                'count': visit_count,
                'last_episode': -1,  # Not tracked in new format
                'last_q_value': max_q_value,
                'last_action': best_action,
                'was_terminal': False  # Not tracked in new format
            })

    # Export data for each agent
    all_rows = []
    for agent_id in sorted(agent_states.keys()):
        states = agent_states[agent_id]

        # Sort by visit count
        states_sorted = sorted(states, key=lambda x: x['count'], reverse=True)

        # Get top 5% most visited states (minimum 100)
        top_count = max(100, int(len(states_sorted) * 0.05))
        top_states = states_sorted[:top_count]

        # Get bottom 5% least visited states (minimum 100)
        bottom_count = max(100, int(len(states_sorted) * 0.05))
        bottom_states = states_sorted[-bottom_count:]

        # Combine and mark category
        for state_data in top_states:
            all_rows.append({
                'experiment': f'{num_agents}_agents_{state_type}_{reward_type}',
                'agent_id': agent_id,
                'state': state_data['state'],
                'visit_count': state_data['count'],
                'last_episode': state_data['last_episode'],
                'last_q_value': state_data['last_q_value'],
                'last_action': state_data['last_action'],
                'was_terminal': state_data['was_terminal'],
                'category': 'most_visited'
            })

        for state_data in bottom_states:
            all_rows.append({
                'experiment': f'{num_agents}_agents_{state_type}_{reward_type}',
                'agent_id': agent_id,
                'state': state_data['state'],
                'visit_count': state_data['count'],
                'last_episode': state_data['last_episode'],
                'last_q_value': state_data['last_q_value'],
                'last_action': state_data['last_action'],
                'was_terminal': state_data['was_terminal'],
                'category': 'least_visited'
            })

    # Save to CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        filename = f'q_learning_states_{num_agents}agents_{state_type}_{reward_type}.csv'
        filepath = os.path.join(tables_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"  [OK] Saved Q-table analysis: {filename} ({len(all_rows)} state records)")
    else:
        print(f"  [WARNING] No state visit data to save")


def consolidate_all_tables(results_list):
    """
    Consolidate all CSV tables into a single 'all_tables' folder.
    Creates summary tables and copies existing analysis files.
    """
    tables_dir = os.path.join(config.base_dir, 'all_tables')
    data_tables_dir = os.path.join(config.base_dir, 'data_tables')

    os.makedirs(tables_dir, exist_ok=True)

    print("\n" + "="*80)
    print("CONSOLIDATING ALL TABLES")
    print("="*80)

    # 1. Create summary table with all experiments
    summary_rows = []
    for r in results_list:
        summary_rows.append({
            'num_agents': r['num_agents'],
            'state_type': r.get('state_type', 'Random'),
            'reward_type': r['reward_type'],
            'num_episodes': r['num_episodes'],
            'is_random': r.get('is_random_baseline', False),
            'FALT': r['metrics'].get('FALT', 0),
            'EALT': r['metrics'].get('EALT', 0),
            'CALT': r['metrics'].get('CALT', 0),
            'Efficiency': r['metrics'].get('Efficiency', 0),
            'Fairness': r['metrics'].get('Fairness', 0)
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_file = os.path.join(tables_dir, 'summary_all_experiments.csv')
    df_summary.to_csv(summary_file, index=False)
    print(f"[OK] Created summary table: summary_all_experiments.csv ({len(summary_rows)} experiments)")

    # 2. Export Q-table analysis for each Q-learning experiment
    q_learning_results = [r for r in results_list if not r.get('is_random_baseline', False)]
    print(f"\nExporting Q-table analysis for {len(q_learning_results)} Q-learning experiments...")
    for result in q_learning_results:
        save_q_table_analysis(result)

    # 3. Copy existing CSV files from data_tables/
    if os.path.exists(data_tables_dir):
        import glob
        import shutil

        csv_files = glob.glob(os.path.join(data_tables_dir, '*.csv'))
        copied_count = 0
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            dest_file = os.path.join(tables_dir, filename)
            shutil.copy2(csv_file, dest_file)
            copied_count += 1

        if copied_count > 0:
            print(f"[OK] Copied {copied_count} CSV files from data_tables/")

    print(f"\n[OK] All tables consolidated in: {tables_dir}")
    print("="*80)


def create_consolidated_terminal_positions(result, output_dir):
    """Create consolidated 3-phase terminal position graph"""
    n_agents = result.get('num_agents', 0)
    state_type = result.get('state_type', 'unknown')
    reward_type = result.get('reward_type', 'unknown')

    terminal_log = result.get('terminal_positions_log', [])
    if not terminal_log:
        print(f'  No terminal_positions_log for {n_agents} agents {state_type} {reward_type}')
        return

    num_episodes = result.get('num_episodes', len(terminal_log))

    # Define phases
    start_end = int(0.1 * num_episodes)
    eps_10pct_episode = int(result.get('epsilon_10pct_episode', 0.5 * num_episodes))
    eps_10pct_start = max(0, eps_10pct_episode - int(0.05 * num_episodes))
    eps_10pct_end = min(num_episodes, eps_10pct_episode + int(0.05 * num_episodes))
    final_start = int(0.9 * num_episodes)

    phases = {
        'Start\n(Exploration)': (0, start_end),
        'Transition\n(ε→10%)': (eps_10pct_start, eps_10pct_end),
        'Final\n(Convergence)': (final_start, num_episodes)
    }

    # Count terminal reaches per agent per phase
    phase_data = {}
    for phase_name, (start, end) in phases.items():
        agent_counts = {i: 0 for i in range(n_agents)}
        for episode, agent_id, position in terminal_log:
            if start <= episode < end:
                agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        phase_data[phase_name] = agent_counts

    # Create figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Agent colors
    colors = plt.cm.Set2(np.linspace(0, 1, n_agents))

    for ax, phase_name in zip(axes, phases.keys()):
        agent_counts = phase_data[phase_name]
        agents = list(agent_counts.keys())
        counts = list(agent_counts.values())
        total = sum(counts)

        bars = ax.bar(agents, counts, color=colors, edgecolor='black', linewidth=1.5)

        # Add count and percentage labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = (count / total * 100) if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{int(count)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Agent ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('Terminal Reaches', fontsize=11, fontweight='bold')
        ax.set_title(phase_name, fontsize=12, fontweight='bold')
        ax.set_xticks(agents)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set consistent y-axis limits across phases
        max_count = max([max(phase_data[p].values()) for p in phases.keys()])
        ax.set_ylim(0, max_count * 1.2)

    # Overall title
    fig.suptitle(f'Terminal Position Distribution Across Learning Phases\n{state_type} {reward_type} ({n_agents} agents)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = output_dir / f'terminal_positions_consolidated_{state_type}_{reward_type}_{n_agents}agents.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Saved: {output_path.name}')

def create_consolidated_exclusive_wins(result, output_dir):
    """Create consolidated 3-phase exclusive wins graph"""
    n_agents = result.get('num_agents', 0)
    state_type = result.get('state_type', 'unknown')
    reward_type = result.get('reward_type', 'unknown')

    exclusive_log = result.get('exclusive_wins_log', [])
    if not exclusive_log:
        print(f'  No exclusive_wins_log for {n_agents} agents {state_type} {reward_type}')
        return

    num_episodes = result.get('num_episodes', max([ep for ep, _ in exclusive_log]) if exclusive_log else 100)

    # Define phases
    start_end = int(0.1 * num_episodes)
    eps_10pct_episode = int(result.get('epsilon_10pct_episode', 0.5 * num_episodes))
    eps_10pct_start = max(0, eps_10pct_episode - int(0.05 * num_episodes))
    eps_10pct_end = min(num_episodes, eps_10pct_episode + int(0.05 * num_episodes))
    final_start = int(0.9 * num_episodes)

    phases = {
        'Start\n(Exploration)': (0, start_end),
        'Transition\n(ε→10%)': (eps_10pct_start, eps_10pct_end),
        'Final\n(Convergence)': (final_start, num_episodes)
    }

    # Count exclusive wins per agent per phase
    phase_data = {}
    for phase_name, (start, end) in phases.items():
        agent_counts = {i: 0 for i in range(n_agents)}
        for episode, agent_id in exclusive_log:
            if start <= episode < end:
                agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        phase_data[phase_name] = agent_counts

    # Create figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Agent colors
    colors = plt.cm.Set2(np.linspace(0, 1, n_agents))

    for ax, phase_name in zip(axes, phases.keys()):
        agent_counts = phase_data[phase_name]
        agents = list(agent_counts.keys())
        counts = list(agent_counts.values())
        total = sum(counts)

        bars = ax.bar(agents, counts, color=colors, edgecolor='black', linewidth=1.5)

        # Add count and percentage labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = (count / total * 100) if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{int(count)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Agent ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('Exclusive Wins', fontsize=11, fontweight='bold')
        ax.set_title(phase_name, fontsize=12, fontweight='bold')
        ax.set_xticks(agents)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set consistent y-axis limits across phases
        max_count = max([max(phase_data[p].values()) for p in phases.keys()])
        ax.set_ylim(0, max_count * 1.25)

    # Overall title
    fig.suptitle(f'Exclusive Wins Distribution Across Learning Phases\n{state_type} {reward_type} ({n_agents} agents)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = output_dir / f'exclusive_wins_consolidated_{state_type}_{reward_type}_{n_agents}agents.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Saved: {output_path.name}')

def create_consolidated_phase_graphs(results_list):
    qlearning_results = [r for r in results_list if not r.get('is_random_baseline', False)]
    if not qlearning_results:
        return
    from pathlib import Path; output_dir = Path(config.base_dir) / 'agent_performance_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(qlearning_results, 1):
        try:
            create_consolidated_terminal_positions(result, output_dir)
            create_consolidated_exclusive_wins(result, output_dir)
        except Exception as e:
            print(f'ERROR: {e}')

def main_consolidated():  # Renamed to avoid conflict with actual main()
    print('=' * 80)
    print('CREATING AGENT PERFORMANCE ANALYSIS GRAPHS')
    print('=' * 80)

    # Load results
    print('\nLoading results...')
    results = load_results()
    qlearning_results = [r for r in results if not r.get('is_random_baseline', False)]
    print(f'Found {len(qlearning_results)} Q-learning results')

    # Create output directory
    output_dir = Path('results/agent_performance_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate consolidated graphs for each experiment
    for i, result in enumerate(qlearning_results, 1):
        n_agents = result.get('num_agents', 'unknown')
        state_type = result.get('state_type', 'unknown')
        reward_type = result.get('reward_type', 'unknown')

        print(f'\n[{i}/{len(qlearning_results)}] Processing {n_agents} agents, {state_type}, {reward_type}')

        try:
            create_consolidated_terminal_positions(result, output_dir)
            create_consolidated_exclusive_wins(result, output_dir)
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 80)
    print('[OK] CONSOLIDATED GRAPHS COMPLETE')
    print('=' * 80)
    print(f'Graphs saved to: {output_dir}')

def create_window_based_learning_phases(results_list):
    """Create window-based learning phases analysis with 5 phases:
    1. Exploration (first 10%)
    2. Epsilon decay window (10% centered on max decay point)
    3. Post-decay (middle period)
    4. Pre-convergence (before final 10%)
    5. Convergence (final 10%)

    Only windows are shown, not cumulative values.
    """
    print("\n== Creating Window-Based Learning Phases Analysis ==")

    for state_type in ['Type-A', 'Type-B']:
        for reward_type in ['ILF', 'IQF']:
            ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                          and r.get('state_type') == state_type and r['reward_type'] == reward_type
                          and 'progression_data' in r]

            for result in ql_results:
                n_agents = result['num_agents']
                n_episodes = result['num_episodes']
                epsilon_decay_target = result.get('q_learning_info', {}).get('epsilon_decay_target', EPSILON_DECAY_TARGET)

                progression_data = result['progression_data']
                episode_points = result['episode_points']

                # Calculate window boundaries
                window_size = int(n_episodes * 0.1)  # 10% window

                # Phase 1: Exploration (0-10%)
                exploration_start = 0
                exploration_end = window_size

                # Phase 2: Epsilon decay window (centered on epsilon_decay_target)
                epsilon_center = int(n_episodes * epsilon_decay_target)
                decay_start = max(0, epsilon_center - window_size // 2)
                decay_end = min(n_episodes, epsilon_center + window_size // 2)

                # Phase 3: Convergence (last 10%)
                convergence_start = n_episodes - window_size
                convergence_end = n_episodes

                # Find episode indices for each window
                exploration_indices = [i for i, ep in enumerate(episode_points)
                                      if exploration_start <= ep < exploration_end]
                decay_indices = [i for i, ep in enumerate(episode_points)
                                if decay_start <= ep < decay_end]
                convergence_indices = [i for i, ep in enumerate(episode_points)
                                      if convergence_start <= ep <= convergence_end]

                # Calculate average values for each window
                window_values = {}
                phase_names = [
                    f'Exploration\n(0-10%)',
                    f'Epsilon Decay\n(~{epsilon_decay_target:.0%} ± 5%)',
                    f'Convergence\n(90-100%)'
                ]

                for metric in list(progression_data.keys()):
                    if len(progression_data[metric]) > 0:
                        # Exploration window average
                        exploration_vals = [progression_data[metric][i] for i in exploration_indices
                                          if i < len(progression_data[metric])]
                        exploration_avg = np.mean(exploration_vals) if exploration_vals else 0

                        # Decay window average
                        decay_vals = [progression_data[metric][i] for i in decay_indices
                                     if i < len(progression_data[metric])]
                        decay_avg = np.mean(decay_vals) if decay_vals else 0

                        # Convergence window average
                        convergence_vals = [progression_data[metric][i] for i in convergence_indices
                                          if i < len(progression_data[metric])]
                        convergence_avg = np.mean(convergence_vals) if convergence_vals else 0

                        window_values[metric] = [exploration_avg, decay_avg, convergence_avg]

                # Create 2x2 plot: TOP = Progression, BOTTOM = Window
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Learning Phases: {state_type} {reward_type} ({n_agents} agents)',
                           fontsize=13, fontweight='bold')

                # === TOP ROW: PROGRESSION LINE PLOTS ===

                # Top-left: Traditional Metrics Progression
                for metric in ['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness']:
                    if metric in progression_data:
                        axes[0, 0].plot(episode_points, progression_data[metric],
                                      label=metric, linewidth=2, color=COLOR_PALETTE.get(metric))

                axes[0, 0].set_xlabel('Episode', fontsize=10)
                axes[0, 0].set_ylabel('Metric Value', fontsize=10)
                axes[0, 0].set_title('(a) Traditional Metrics - Progression', fontsize=11, fontweight='bold')
                axes[0, 0].legend(fontsize=9, loc='best')
                axes[0, 0].grid(alpha=0.3)
                axes[0, 0].set_ylim([0, 1.05])

                # Top-right: ALT Metrics Progression
                for metric in ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']:
                    if metric in progression_data:
                        axes[0, 1].plot(episode_points, progression_data[metric],
                                      label=metric, linewidth=2, color=COLOR_PALETTE.get(metric))

                axes[0, 1].set_xlabel('Episode', fontsize=10)
                axes[0, 1].set_ylabel('ALT Metric Value', fontsize=10)
                axes[0, 1].set_title('(b) ALT Metrics - Progression', fontsize=11, fontweight='bold')
                axes[0, 1].legend(fontsize=9, loc='best', ncol=2)
                axes[0, 1].grid(alpha=0.3)
                axes[0, 1].set_ylim([0, 1.05])

                # === BOTTOM ROW: WINDOW-BASED ANALYSIS ===

                x = np.arange(len(phase_names))

                # Bottom-left: Traditional Metrics Windows
                width = 0.2
                for i, metric in enumerate(['Efficiency', 'Reward_Fairness', 'TT_Fairness', 'Fairness']):
                    values = window_values.get(metric, [0, 0, 0])
                    axes[1, 0].bar(x + i * width - width*1.5, values, width, label=metric,
                           color=COLOR_PALETTE.get(metric))

                axes[1, 0].set_ylabel('Metric Value', fontsize=10)
                axes[1, 0].set_title(f'(c) Traditional Metrics - Windows', fontsize=11, fontweight='bold')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(phase_names, fontsize=8)
                axes[1, 0].legend(fontsize=9, loc='best')
                axes[1, 0].grid(axis='y', alpha=0.3)
                axes[1, 0].set_ylim([0, 1.05])

                # Bottom-right: ALT Metrics Windows
                width = 0.13
                for i, metric in enumerate(['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']):
                    values = window_values.get(metric, [0, 0, 0])
                    axes[1, 1].bar(x + i * width - width*2.5, values, width, label=metric,
                           color=COLOR_PALETTE.get(metric))

                axes[1, 1].set_ylabel('ALT Metric Value', fontsize=10)
                axes[1, 1].set_title(f'(d) ALT Metrics - Windows', fontsize=11, fontweight='bold')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(phase_names, fontsize=8)
                axes[1, 1].legend(fontsize=9, loc='best', ncol=2)
                axes[1, 1].grid(axis='y', alpha=0.3)
                axes[1, 1].set_ylim([0, 1.05])

                plt.tight_layout()
                filename = f'learning_windows_{state_type}_{reward_type}_{n_agents}agents.png'
                plt.savefig(os.path.join(config.figure_dirs['learning_phases'], filename), dpi=300)
                plt.show()
                plt.close()
                print(f"[OK] Saved window-based learning phases for {state_type} {reward_type} {n_agents} agents")

    print("[OK] Window-based learning phases analysis completed")


def create_consolidated_learning_plots(results_list):
    """Create consolidated plots showing all agents together for each config (like old learning phases)."""
    print("\n== Creating Consolidated Learning Plots ==")

    for state_type in ['Type-A', 'Type-B']:
        for reward_type in ['ILF', 'IQF']:
            ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                          and r.get('state_type') == state_type and r['reward_type'] == reward_type
                          and 'progression_data' in r]

            if not ql_results:
                continue

            # Sort by number of agents
            ql_results = sorted(ql_results, key=lambda x: x['num_agents'])

            # Create figure with 2 panels (Traditional | ALT)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Line styles for different agent counts
            linestyles_map = {2: '-', 3: '--', 5: '-.', 8: ':', 10: (0, (3, 1, 1, 1))}  # dash-dot-dot

            for result in ql_results:
                n_agents = result['num_agents']
                progression_data = result['progression_data']
                episode_points = result['episode_points']
                n_episodes = result['num_episodes']

                # Normalize x-axis
                x_norm = [ep / n_episodes for ep in episode_points]
                linestyle = linestyles_map.get(n_agents, '-')

                # Plot Traditional metrics on left panel with DIFFERENT COLORS per metric
                for metric in ['Efficiency', 'Reward_Fairness']:
                    if metric in progression_data:
                        ax1.plot(x_norm, progression_data[metric],
                                label=f'{metric} ({n_agents}a)',
                                color=COLOR_PALETTE.get(metric),
                                linestyle=linestyle,
                                linewidth=2, alpha=0.8)

                # Plot ALT metrics on right panel with DIFFERENT COLORS per metric
                for metric in ['FALT', 'CALT', 'AALT']:
                    if metric in progression_data:
                        ax2.plot(x_norm, progression_data[metric],
                                label=f'{metric} ({n_agents}a)',
                                color=COLOR_PALETTE.get(metric),
                                linestyle=linestyle,
                                linewidth=2, alpha=0.8)

            ax1.set_xlabel('Episode Progress (normalized)', fontsize=11)
            ax1.set_ylabel('Traditional Metric Value', fontsize=11)
            ax1.set_title(f'(a) Traditional Metrics - {state_type} {reward_type}', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9, loc='best')
            ax1.grid(alpha=0.3)
            ax1.set_ylim([0, 1.05])

            ax2.set_xlabel('Episode Progress (normalized)', fontsize=11)
            ax2.set_ylabel('ALT Metric Value', fontsize=11)
            ax2.set_title(f'(b) ALT Metrics - {state_type} {reward_type}', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9, loc='best')
            ax2.grid(alpha=0.3)
            ax2.set_ylim([0, 1.05])

            plt.tight_layout()
            filename = f'consolidated_learning_{state_type}_{reward_type}.png'
            plt.savefig(os.path.join(config.figure_dirs['learning_phases'], filename), dpi=300)
            plt.show()
            plt.close()
            print(f"[OK] Saved consolidated learning plot for {state_type} {reward_type}")

    print("[OK] Consolidated learning plots completed")


def create_separate_3d_plots_by_alt_metric(results_list):
    """Create 3D grid plots: 1 figure per SETTING with 2×3 grid (6 ALT metrics).
    Each subplot shows: ALT metric + Efficiency + Reward_Fairness + Random baseline for all agents.
    """
    print("\n== Creating 3D Grid Plots (2×3 per Setting) ==")

    alt_metrics = ['FALT', 'EALT', 'qFALT', 'qEALT', 'CALT', 'AALT']

    for state_type in ['Type-A', 'Type-B']:
        for reward_type in ['ILF', 'IQF']:
            # Get Q-learning and random baseline results for this setting
            ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                          and r.get('state_type') == state_type and r['reward_type'] == reward_type
                          and 'progression_data' in r]

            random_results = [r for r in results_list if r.get('is_random_baseline', False)
                             and r['reward_type'] == reward_type]

            if not ql_results:
                continue

            # Create 2×3 grid figure
            fig = plt.figure(figsize=(18, 12))
            fig.suptitle(f'3D Metric Analysis - {state_type} {reward_type}',
                        fontsize=14, fontweight='bold', y=0.995)

            for idx, alt_metric in enumerate(alt_metrics, 1):
                ax = fig.add_subplot(2, 3, idx, projection='3d')

                # Metrics to plot in this subplot
                metrics_to_plot = [alt_metric, 'Efficiency', 'Reward_Fairness']

                # ALL ALT metrics in RED, traditional metrics use their colors
                metric_colors = {
                    alt_metric: '#dc143c',  # All ALT metrics RED
                    'Efficiency': COLOR_PALETTE.get('Efficiency', '#2ca02c'),
                    'Reward_Fairness': COLOR_PALETTE.get('Reward_Fairness', '#1f77b4')
                }

                # Plot Q-learning results
                for result in ql_results:
                    n_agents = result['num_agents']
                    progression_data = result['progression_data']
                    episode_points = result['episode_points']
                    n_episodes = result['num_episodes']

                    # Normalize episodes to 0-1
                    x_norm = [ep / n_episodes for ep in episode_points]

                    for metric in metrics_to_plot:
                        if metric not in progression_data:
                            continue

                        y_vals = [n_agents] * len(x_norm)
                        z_vals = progression_data[metric]
                        color = metric_colors.get(metric, 'gray')

                        # All solid lines (no line style differentiation)
                        ax.plot(x_norm, y_vals, z_vals,
                               color=color, linestyle='-',
                               linewidth=2, alpha=0.8)

                # Plot random baselines
                if random_results:
                    for random_result in random_results:
                        n_agents_rand = random_result['num_agents']

                        if 'progression_data' in random_result and alt_metric in random_result['progression_data']:
                            random_progression = random_result['progression_data'][alt_metric]
                            random_episodes = random_result.get('episode_points', [])
                            n_episodes_rand = random_result.get('num_episodes', len(random_progression))

                            x_norm_rand = [ep / n_episodes_rand for ep in random_episodes]
                            y_vals_rand = [n_agents_rand] * len(x_norm_rand)
                            z_vals_rand = random_progression

                            ax.plot(x_norm_rand, y_vals_rand, z_vals_rand,
                                   color='gray', linestyle='-', linewidth=2, alpha=0.5)

                ax.set_xlabel('Episodes', fontsize=9)
                ax.set_ylabel('Agents', fontsize=9)
                ax.set_zlabel('Value', fontsize=9)
                ax.set_title(f'{alt_metric}', fontsize=10, fontweight='bold')
                ax.set_zlim([0, 1.05])
                ax.view_init(elev=20, azim=45)
                ax.grid(alpha=0.3)

            # Create custom legend (metrics only, not agents)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#dc143c', linewidth=2, label='ALT Metrics'),  # All ALT RED
                Line2D([0], [0], color=COLOR_PALETTE.get('Efficiency', '#2ca02c'), linewidth=2, label='Efficiency'),
                Line2D([0], [0], color=COLOR_PALETTE.get('Reward_Fairness', '#1f77b4'), linewidth=2, label='Reward Fairness'),
                Line2D([0], [0], color='gray', linewidth=2, alpha=0.5, label='Random Baseline')
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
                      bbox_to_anchor=(0.5, -0.02))

            plt.tight_layout(rect=[0, 0.02, 1, 0.99])

            filename = f'3d_grid_{state_type}_{reward_type}.png'
            plt.savefig(os.path.join(config.figure_dirs['3d_analysis'], filename), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f"[OK] Saved 3D grid for {state_type} {reward_type}")

    print("[OK] 3D grid plots completed")


def create_qtable_visualizations(results_list):
    """Create Q-table HEATMAP showing TOP 10% most-visited states per agent.

    Layout: 2-column layout (Type-A left, Type-B right) × 2 reward types = 4 heatmaps per agent
    Each heatmap: rows=top states, columns=actions (0-4), color=Q-value, annotated with visit counts
    """
    print("\n== Creating Q-Table Heatmap Visualizations (Top 10% States) ==")

    # Get Q-learning results
    qlearning_results = [r for r in results_list if not r.get('is_random_baseline', False)]

    # Group by agent count
    agent_counts = sorted(set(r['num_agents'] for r in qlearning_results))

    for n_agents in agent_counts:
        # Create separate figure for each agent to keep it readable
        for agent_id in range(n_agents):
            # Create 2×2 grid: rows=reward types (ILF/IQF), cols=state types (Type-A/Type-B)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Q-Table Heatmap - Agent {agent_id} ({n_agents} agents total)\nTop 10% Most-Visited States',
                        fontsize=14, fontweight='bold')

            for state_idx, state_type in enumerate(['Type-A', 'Type-B']):
                for reward_idx, reward_type in enumerate(['ILF', 'IQF']):
                    ax = axes[reward_idx, state_idx]

                    # Find matching result
                    matching = [r for r in qlearning_results
                               if r['num_agents'] == n_agents
                               and r.get('state_type') == state_type
                               and r['reward_type'] == reward_type]

                    if not matching:
                        ax.text(0.5, 0.5, f'No data\n{state_type} {reward_type}',
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    result = matching[0]
                    state_visits = result.get('state_visits', {})
                    q_tables = result.get('q_tables', {})

                    if agent_id not in state_visits or agent_id not in q_tables:
                        ax.text(0.5, 0.5, 'No Q-table data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Get data for this agent
                    q_table = q_tables[agent_id]  # [num_states, num_actions]
                    visits = state_visits[agent_id]  # {state: visit_count}

                    # Sort states by visit count, take top 10%
                    sorted_states = sorted(visits.items(), key=lambda x: x[1], reverse=True)
                    top_k = max(5, min(20, len(sorted_states) // 10))  # Top 10%, but 5-20 states
                    top_states = sorted_states[:top_k]

                    # Build heatmap data: rows=states, cols=actions
                    num_actions = 2  # 0=stay, 1=move
                    heatmap_data = np.zeros((len(top_states), num_actions))
                    state_labels = []
                    visit_counts = []

                    for i, (state_key, visit_count) in enumerate(top_states):
                        if state_key < len(q_table):
                            heatmap_data[i, :] = q_table[state_key, :num_actions]
                        state_labels.append(f'S{state_key}')
                        visit_counts.append(visit_count)

                    # Create heatmap
                    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

                    # Annotate with Q-values
                    for i in range(len(top_states)):
                        for j in range(num_actions):
                            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}\n({visit_counts[i]}v)',
                                         ha="center", va="center", color="black", fontsize=7)

                    # Labels
                    ax.set_xticks(np.arange(num_actions))
                    ax.set_xticklabels(['Stay', 'Move'], fontsize=9)
                    ax.set_yticks(np.arange(len(top_states)))
                    ax.set_yticklabels(state_labels, fontsize=8)
                    ax.set_xlabel('Actions', fontsize=10)
                    ax.set_ylabel('States (ranked by visits)', fontsize=10)
                    ax.set_title(f'{state_type} {reward_type}', fontsize=11, fontweight='bold')

                    # Colorbar
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Q-Value', rotation=270, labelpad=15, fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            filename = f'qtable_heatmap_agent{agent_id}_{n_agents}agents.png'
            plt.savefig(os.path.join(config.figure_dirs['qtable_analysis'], filename), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        print(f"  [OK] Saved Q-table heatmaps for {n_agents} agents ({n_agents} files)")

    print("[OK] Q-table heatmap visualizations completed")


def create_terminal_exclusive_wins_analysis(results_list):
    """Create terminal and exclusive wins analysis with 3 windows per metric (6 subplots total: 2 rows × 3 columns)."""
    print("\n== Creating Terminal & Exclusive Wins Analysis (Window-Based) ==")

    for state_type in ['Type-A', 'Type-B']:
        for reward_type in ['ILF', 'IQF']:
            ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                          and r.get('state_type') == state_type and r['reward_type'] == reward_type]

            for result in ql_results:
                n_agents = result['num_agents']
                num_episodes = result.get('num_episodes', 0)

                # Extract data from result
                terminal_positions = result.get('terminal_positions_log', [])
                exclusive_wins = result.get('exclusive_wins_log', [])

                if not terminal_positions and not exclusive_wins:
                    print(f"  [WARNING] Skipping {state_type} {reward_type} {n_agents} agents - no data")
                    continue

                # Define 3 windows
                window1_end = int(num_episodes * 0.10)  # Exploration (0-10%)
                window2_start = int(num_episodes * 0.45)
                window2_end = int(num_episodes * 0.55)  # Epsilon decay (45-55%)
                window3_start = int(num_episodes * 0.90)  # Convergence (90-100%)

                # Create 2×3 figure (3 windows for terminal πάνω, 3 windows for exclusive κάτω)
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f'Terminal & Exclusive Wins (Window-Based) - {state_type} {reward_type} ({n_agents} agents)',
                           fontsize=14, fontweight='bold')

                window_names = ['Exploration\n(0-10%)', 'Epsilon Decay\n(45-55%)', 'Convergence\n(90-100%)']
                window_ranges = [(0, window1_end), (window2_start, window2_end), (window3_start, num_episodes)]

                colors_viridis = plt.cm.viridis(np.linspace(0.3, 0.9, n_agents))
                colors_plasma = plt.cm.plasma(np.linspace(0.3, 0.9, n_agents))

                # === TOP ROW: Terminal Arrivals per Window ===
                if terminal_positions:
                    for win_idx, (start_ep, end_ep) in enumerate(window_ranges):
                        ax = axes[0, win_idx]

                        # Count terminal arrivals per agent in this window
                        terminal_counts = {i: 0 for i in range(n_agents)}
                        for episode, agent_id, position in terminal_positions:
                            if start_ep <= episode < end_ep:
                                terminal_counts[agent_id] += 1

                        agents_list = list(range(n_agents))
                        counts = [terminal_counts[i] for i in agents_list]

                        ax.bar(agents_list, counts, color=colors_viridis, alpha=0.8, edgecolor='black')
                        ax.set_xlabel('Agent ID', fontsize=10)
                        ax.set_ylabel('Terminal Arrivals', fontsize=10)
                        ax.set_title(f'{window_names[win_idx]}', fontsize=11, fontweight='bold')
                        ax.set_xticks(agents_list)
                        ax.grid(axis='y', alpha=0.3)

                # === BOTTOM ROW: Exclusive Wins per Window ===
                if exclusive_wins:
                    for win_idx, (start_ep, end_ep) in enumerate(window_ranges):
                        ax = axes[1, win_idx]

                        # Count exclusive wins per agent in this window
                        exclusive_counts = {i: 0 for i in range(n_agents)}
                        for entry in exclusive_wins:
                            if len(entry) == 2:
                                episode, agent_id = entry
                            else:  # len(entry) == 3
                                episode, agent_id, position = entry

                            if start_ep <= episode < end_ep:
                                exclusive_counts[agent_id] += 1

                        agents_list = list(range(n_agents))
                        counts = [exclusive_counts[i] for i in agents_list]

                        ax.bar(agents_list, counts, color=colors_plasma, alpha=0.8, edgecolor='black')
                        ax.set_xlabel('Agent ID', fontsize=10)
                        ax.set_ylabel('Exclusive Wins', fontsize=10)
                        ax.set_title(f'{window_names[win_idx]}', fontsize=11, fontweight='bold')
                        ax.set_xticks(agents_list)
                        ax.grid(axis='y', alpha=0.3)

                plt.tight_layout(rect=[0, 0, 1, 0.97])
                filename = f'terminal_exclusive_wins_windows_{state_type}_{reward_type}_{n_agents}agents.png'
                plt.savefig(os.path.join(config.figure_dirs['agent_performance'], filename), dpi=300)
                plt.show()
                plt.close()
                print(f"  [OK] Saved window-based terminal/exclusive wins for {state_type} {reward_type} {n_agents} agents")

    print("[OK] Terminal & exclusive wins analysis completed")


def create_cumulative_terminal_exclusive_wins(results_list):
    """Create cumulative bar charts showing TOTAL terminal arrivals and exclusive wins across ALL episodes."""
    print("\n== Creating Cumulative Terminal & Exclusive Wins Bar Charts ==")

    for state_type in ['Type-A', 'Type-B']:
        for reward_type in ['ILF', 'IQF']:
            ql_results = [r for r in results_list if not r.get('is_random_baseline', False)
                          and r.get('state_type') == state_type and r['reward_type'] == reward_type]

            for result in ql_results:
                n_agents = result['num_agents']

                # Extract data from result
                terminal_positions = result.get('terminal_positions_log', [])
                exclusive_wins = result.get('exclusive_wins_log', [])

                if not terminal_positions and not exclusive_wins:
                    print(f"  [WARNING] Skipping {state_type} {reward_type} {n_agents} agents - no data")
                    continue

                # Create figure with 2 subplots (top: terminal arrivals, bottom: exclusive wins)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                fig.suptitle(f'Cumulative Terminal Arrivals & Exclusive Wins - {state_type} {reward_type} ({n_agents} agents)',
                           fontsize=14, fontweight='bold')

                agents_list = list(range(n_agents))
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_agents))

                # === TOP PANEL: Total Terminal Arrivals per Agent ===
                if terminal_positions:
                    terminal_counts = {i: 0 for i in range(n_agents)}
                    for episode, agent_id, position in terminal_positions:
                        terminal_counts[agent_id] += 1

                    counts = [terminal_counts[i] for i in agents_list]

                    ax1.bar(agents_list, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
                    ax1.set_xlabel('Agent ID', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Total Terminal Arrivals', fontsize=12, fontweight='bold')
                    ax1.set_title('(a) Total Terminal Arrivals Across All Episodes', fontsize=13, fontweight='bold')
                    ax1.set_xticks(agents_list)
                    ax1.set_xticklabels([f'Agent {i}' for i in agents_list])
                    ax1.grid(axis='y', alpha=0.3, linestyle='--')

                    # Add value labels on top of bars
                    for i, count in enumerate(counts):
                        ax1.text(i, count, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

                # === BOTTOM PANEL: Total Exclusive Wins per Agent ===
                if exclusive_wins:
                    exclusive_counts = {i: 0 for i in range(n_agents)}
                    for entry in exclusive_wins:
                        if len(entry) == 2:
                            episode, agent_id = entry
                        else:  # len(entry) == 3
                            episode, agent_id, position = entry
                        exclusive_counts[agent_id] += 1

                    counts = [exclusive_counts[i] for i in agents_list]

                    colors_plasma = plt.cm.plasma(np.linspace(0.3, 0.9, n_agents))
                    ax2.bar(agents_list, counts, color=colors_plasma, alpha=0.8, edgecolor='black', linewidth=1.5)
                    ax2.set_xlabel('Agent ID', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Total Exclusive Wins', fontsize=12, fontweight='bold')
                    ax2.set_title('(b) Total Exclusive Wins Across All Episodes', fontsize=13, fontweight='bold')
                    ax2.set_xticks(agents_list)
                    ax2.set_xticklabels([f'Agent {i}' for i in agents_list])
                    ax2.grid(axis='y', alpha=0.3, linestyle='--')

                    # Add value labels on top of bars
                    for i, count in enumerate(counts):
                        ax2.text(i, count, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

                plt.tight_layout(rect=[0, 0, 1, 0.97])
                filename = f'terminal_exclusive_wins_cumulative_{state_type}_{reward_type}_{n_agents}agents.png'
                plt.savefig(os.path.join(config.figure_dirs['agent_performance'], filename), dpi=300)
                plt.show()
                plt.close()
                print(f"  [OK] Saved cumulative terminal/exclusive wins for {state_type} {reward_type} {n_agents} agents")

    print("[OK] Cumulative terminal & exclusive wins analysis completed")


def save_comprehensive_analysis(results_list):
    """Save all comprehensive analysis data."""
    print("\n" + "=" * 80)
    print("SAVING COMPREHENSIVE ANALYSIS DATA")
    print("=" * 80)

    save_comprehensive_data_tables(results_list)
    consolidate_all_tables(results_list)  # NEW: Consolidate all tables including Q-table analysis
    save_alt_metrics_analysis(results_list)
    save_rp_metrics_analysis(results_list)
    save_computation_times_analysis(results_list)
    save_all_alt_ratio_analysis(results_list)
    save_random_baselines_analysis(results_list)
    create_learning_phases_analysis(results_list)
    create_epsilon_vs_metrics_analysis(results_list)
    create_3d_metric_progression_plots(results_list)
    create_consolidated_phase_graphs(results_list)  # NEW: Consolidated 3-phase learning graphs
    create_window_based_learning_phases(results_list)  # NEW: Window-based learning phases
    create_consolidated_learning_plots(results_list)  # NEW: Consolidated plots all agents
    create_separate_3d_plots_by_alt_metric(results_list)  # NEW: Separate 3D for each ALT metric
    create_qtable_visualizations(results_list)  # NEW: Q-table analysis
    create_terminal_exclusive_wins_analysis(results_list)  # NEW: Terminal & exclusive wins (window-based)
    create_cumulative_terminal_exclusive_wins(results_list)  # NEW: Terminal & exclusive wins (cumulative)

    # All visualizations now integrated into new_code.py - no external imports needed
    print("\n[OK] All comprehensive analysis data saved successfully!")

# ==============================================================================
# PAPER 1 FIGURES - PUBLICATION-READY (2-COLUMN AND 1-COLUMN FORMATS)
# ==============================================================================
# Functions for generating publication-ready figures for journal submission.
# Supports both 2-column (14×10) and 1-column (7×15) formats.
#
# Based on BASE=1000 dataset from Ubuntu visualization generation.
# All figures use consistent styling: light cyan for random baseline,
# proper metric names, no informal annotations.
# ==============================================================================

def create_paper1_figure1_2column(base_dir, output_dir):
    """
    Figure 1 (2-column): Q-Learning vs Random Baseline - CALT Comparison - REDESIGNED.

    NEW DESIGN: Grouped bars showing Random + 4 Q-Learning modes (Type-A/B × ILF/IQF)
    for all agent counts (2, 3, 5, 8, 10+).

    - Random: Filled bar (cyan)
    - Q-Learning modes: Border-only bars with different hatch patterns
    - Careful spacing to avoid text overlap

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    # Get all available agent counts
    agent_counts = sorted(comprehensive_df['Agents'].unique())

    # Define 4 Q-Learning configurations
    ql_configs = [
        ('Type-A', 'ILF', '/', 'A-ILF'),
        ('Type-A', 'IQF', '\\', 'A-IQF'),
        ('Type-B', 'ILF', '|', 'B-ILF'),
        ('Type-B', 'IQF', '-', 'B-IQF')
    ]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('Q-Learning vs Random Baseline: CALT Coordination Metrics\n(Q-Learning shows mean ± range across 4 modes)',
                 fontsize=14, fontweight='bold')

    # Bar width and spacing
    bar_width = 0.35
    x_positions = np.arange(len(agent_counts))

    # Collect data for each agent count
    random_values = []
    ql_means = []
    ql_mins = []
    ql_maxs = []
    min_labels = []
    max_labels = []

    for n in agent_counts:
        # Random baseline
        random_row = random_df[(random_df['Agents'] == n) & (random_df['Reward_Type'] == 'ILF')]
        if not random_row.empty:
            random_values.append(float(random_row['CALT'].iloc[0]))
        else:
            random_values.append(0)

        # Q-Learning: get all 4 mode values
        ql_vals = []
        ql_labels = []
        for state_type, reward_type, hatch, label in ql_configs:
            ql_row = ql_df[(ql_df['Agents'] == n) &
                           (ql_df['State_Type'] == state_type) &
                           (ql_df['Reward_Type'] == reward_type)]
            if not ql_row.empty:
                ql_vals.append(float(ql_row['CALT'].iloc[0]))
                ql_labels.append(label)

        # Calculate statistics
        if ql_vals:
            ql_means.append(np.mean(ql_vals))
            min_val = np.min(ql_vals)
            max_val = np.max(ql_vals)
            ql_mins.append(min_val)
            ql_maxs.append(max_val)

            # Find which mode achieved min/max
            min_idx = ql_vals.index(min_val)
            max_idx = ql_vals.index(max_val)
            min_labels.append(ql_labels[min_idx])
            max_labels.append(ql_labels[max_idx])
        else:
            ql_means.append(0)
            ql_mins.append(0)
            ql_maxs.append(0)
            min_labels.append('')
            max_labels.append('')

    # Convert to numpy arrays
    ql_means = np.array(ql_means)
    ql_mins = np.array(ql_mins)
    ql_maxs = np.array(ql_maxs)

    # Calculate error bars (distance from mean to min/max)
    yerr_lower = ql_means - ql_mins
    yerr_upper = ql_maxs - ql_means

    # Plot Random baseline
    ax.bar(x_positions - bar_width/2, random_values, bar_width,
           label='Random Baseline',
           color='#B3E5FC', edgecolor='#4DD0E1', linewidth=2.0, alpha=0.8)

    # Plot Q-Learning average with error bars
    ax.bar(x_positions + bar_width/2, ql_means, bar_width,
           label='Q-Learning (mean of 4 modes)',
           color='#FFE0B2', edgecolor='#FF9800', linewidth=2.0, alpha=0.8)

    # Add error bars (min-max range)
    ax.errorbar(x_positions + bar_width/2, ql_means,
                yerr=[yerr_lower, yerr_upper],
                fmt='none', ecolor='#795548', elinewidth=2, capsize=5, capthick=2,
                label='Min-Max range')

    # Add markers for min/max values with ACTUAL mode labels per agent count
    # Use colors matching the bar palette (darker shades of cyan/orange)
    ax.scatter(x_positions + bar_width/2, ql_mins, marker='v', s=100,
               color='#0277BD', edgecolor='#01579B', linewidth=1.5, zorder=10,
               label='Min')
    ax.scatter(x_positions + bar_width/2, ql_maxs, marker='^', s=100,
               color='#F57C00', edgecolor='#E65100', linewidth=1.5, zorder=10,
               label='Max')

    # Add text annotations showing actual mode for each min/max
    for i, (x_pos, min_label, max_label) in enumerate(zip(x_positions, min_labels, max_labels)):
        if min_label:
            ax.text(x_pos + bar_width/2, ql_mins[i] - 0.015, min_label,
                   ha='center', va='top', fontsize=8, color='#01579B', fontweight='bold')
        if max_label:
            ax.text(x_pos + bar_width/2, ql_maxs[i] + 0.015, max_label,
                   ha='center', va='bottom', fontsize=8, color='#E65100', fontweight='bold')

    # Axis labels
    ax.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax.set_ylabel('CALT (Coordination Metric)', fontsize=12, fontweight='bold')

    # X-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(agent_counts, fontsize=11)

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure1_qlearning_vs_random_calt.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure2_2column(base_dir, output_dir):
    """
    Figure 2 (2-column): PA-Equivalent Performance - REDESIGNED.

    NEW DESIGN: Grouped bars showing Random + 4 Q-Learning modes
    for PA-Equivalent performance (n × √CALT) for all agent counts.

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    def calculate_pa_equivalent(n, calt_value):
        """Calculate PA-equivalent: n × √CALT"""
        if calt_value <= 0:
            return 0.0
        return n * (calt_value ** 0.5)

    # Get all available agent counts
    agent_counts = sorted(comprehensive_df['Agents'].unique())

    # Define 4 Q-Learning configurations
    ql_configs = [
        ('Type-A', 'ILF', '/', 'A-ILF'),
        ('Type-A', 'IQF', '\\', 'A-IQF'),
        ('Type-B', 'ILF', '|', 'B-ILF'),
        ('Type-B', 'IQF', '-', 'B-IQF')
    ]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('PA-Equivalent Performance: Q-Learning vs Random Baseline\n(Q-Learning shows mean ± range across 4 modes)',
                 fontsize=14, fontweight='bold')

    # Bar width and spacing
    bar_width = 0.35
    x_positions = np.arange(len(agent_counts))

    # Collect data for each agent count
    random_values = []
    ql_means = []
    ql_mins = []
    ql_maxs = []
    min_labels = []
    max_labels = []

    for n in agent_counts:
        # Random baseline
        random_row = random_df[(random_df['Agents'] == n) & (random_df['Reward_Type'] == 'ILF')]
        if not random_row.empty:
            random_calt = float(random_row['CALT'].iloc[0])
            random_values.append(calculate_pa_equivalent(n, random_calt))
        else:
            random_values.append(0)

        # Q-Learning: get all 4 mode values
        ql_vals = []
        ql_labels = []
        for state_type, reward_type, hatch, label in ql_configs:
            ql_row = ql_df[(ql_df['Agents'] == n) &
                           (ql_df['State_Type'] == state_type) &
                           (ql_df['Reward_Type'] == reward_type)]
            if not ql_row.empty:
                ql_calt = float(ql_row['CALT'].iloc[0])
                ql_vals.append(calculate_pa_equivalent(n, ql_calt))
                ql_labels.append(label)

        # Calculate statistics
        if ql_vals:
            ql_means.append(np.mean(ql_vals))
            min_val = np.min(ql_vals)
            max_val = np.max(ql_vals)
            ql_mins.append(min_val)
            ql_maxs.append(max_val)

            # Find which mode achieved min/max
            min_idx = ql_vals.index(min_val)
            max_idx = ql_vals.index(max_val)
            min_labels.append(ql_labels[min_idx])
            max_labels.append(ql_labels[max_idx])
        else:
            ql_means.append(0)
            ql_mins.append(0)
            ql_maxs.append(0)
            min_labels.append('')
            max_labels.append('')

    # Convert to numpy arrays
    ql_means = np.array(ql_means)
    ql_mins = np.array(ql_mins)
    ql_maxs = np.array(ql_maxs)

    # Calculate error bars (distance from mean to min/max)
    yerr_lower = ql_means - ql_mins
    yerr_upper = ql_maxs - ql_means

    # Plot Random baseline
    ax.bar(x_positions - bar_width/2, random_values, bar_width,
           label='Random Baseline',
           color='#B3E5FC', edgecolor='#4DD0E1', linewidth=2.0, alpha=0.8)

    # Plot Q-Learning average with error bars
    ax.bar(x_positions + bar_width/2, ql_means, bar_width,
           label='Q-Learning (mean of 4 modes)',
           color='#FFE0B2', edgecolor='#FF9800', linewidth=2.0, alpha=0.8)

    # Add error bars (min-max range)
    ax.errorbar(x_positions + bar_width/2, ql_means,
                yerr=[yerr_lower, yerr_upper],
                fmt='none', ecolor='#795548', elinewidth=2, capsize=5, capthick=2,
                label='Min-Max range')

    # Add markers for min/max values with ACTUAL mode labels per agent count
    # Use colors matching the bar palette (darker shades of cyan/orange)
    ax.scatter(x_positions + bar_width/2, ql_mins, marker='v', s=100,
               color='#0277BD', edgecolor='#01579B', linewidth=1.5, zorder=10,
               label='Min')
    ax.scatter(x_positions + bar_width/2, ql_maxs, marker='^', s=100,
               color='#F57C00', edgecolor='#E65100', linewidth=1.5, zorder=10,
               label='Max')

    # Add text annotations showing actual mode for each min/max
    for i, (x_pos, min_label, max_label) in enumerate(zip(x_positions, min_labels, max_labels)):
        if min_label:
            ax.text(x_pos + bar_width/2, ql_mins[i] - 0.015, min_label,
                   ha='center', va='top', fontsize=8, color='#01579B', fontweight='bold')
        if max_label:
            ax.text(x_pos + bar_width/2, ql_maxs[i] + 0.015, max_label,
                   ha='center', va='bottom', fontsize=8, color='#E65100', fontweight='bold')

    # Axis labels
    ax.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax.set_ylabel('PA-Equivalent Agents (n × √CALT)', fontsize=12, fontweight='bold')

    # X-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(agent_counts, fontsize=11)

    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure2_pa_equivalent_with_baseline.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure3_2column(base_dir, output_dir):
    """
    Figure 3 (2-column): Traditional vs ALT Metrics Comparison.

    Creates 3 vertical panels showing Efficiency, Reward Fairness, and CALT
    for Type-B IQF configuration (best example of symmetric failure).

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle('Traditional Metrics Hide Coordination Failure\nQ-Learning vs Random Baseline (Type-B, IQF)',
                 fontsize=14, fontweight='bold')

    # Select one representative config: Type-B, IQF (shows symmetric failure best)
    state_type = 'Type-B'
    reward_type = 'IQF'

    # Auto-detect all available agent counts from data
    agent_counts = sorted(comprehensive_df['Agents'].unique())

    # Top panel: Efficiency
    ax = axes[0]
    random_vals = []
    ql_vals = []

    for n in agent_counts:
        random_row = random_df[(random_df['Agents'] == n) &
                              (random_df['Reward_Type'] == reward_type)]
        ql_row = ql_df[(ql_df['Agents'] == n) &
                       (ql_df['State_Type'] == state_type) &
                       (ql_df['Reward_Type'] == reward_type)]

        if not random_row.empty:
            random_vals.append(float(random_row['Efficiency'].iloc[0]))
        else:
            random_vals.append(0)

        if not ql_row.empty:
            ql_vals.append(float(ql_row['Efficiency'].iloc[0]))
        else:
            ql_vals.append(0)

    ax.plot(agent_counts, random_vals, 'o--', label='Random', color='#4DD0E1', linewidth=2.0, markersize=8)
    ax.plot(agent_counts, ql_vals, 's-', label='Q-Learning', color='orange', linewidth=2.5, markersize=9)

    ax.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title('Panel A: Efficiency Comparison', fontsize=12)

    # Middle panel: Reward Fairness
    ax = axes[1]
    random_vals = []
    ql_vals = []

    for n in agent_counts:
        random_row = random_df[(random_df['Agents'] == n) &
                              (random_df['Reward_Type'] == reward_type)]
        ql_row = ql_df[(ql_df['Agents'] == n) &
                       (ql_df['State_Type'] == state_type) &
                       (ql_df['Reward_Type'] == reward_type)]

        if not random_row.empty:
            random_vals.append(float(random_row['Reward_Fairness'].iloc[0]))
        else:
            random_vals.append(0)

        if not ql_row.empty:
            ql_vals.append(float(ql_row['Reward_Fairness'].iloc[0]))
        else:
            ql_vals.append(0)

    ax.plot(agent_counts, random_vals, 'o--', label='Random', color='#4DD0E1', linewidth=2.0, markersize=8)
    ax.plot(agent_counts, ql_vals, 's-', label='Q-Learning', color='green', linewidth=2.5, markersize=9)

    ax.set_ylabel('Reward Fairness', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title('Panel B: Reward Fairness Comparison', fontsize=12)

    # Bottom panel: CALT
    ax = axes[2]
    random_vals = []
    ql_vals = []

    for n in agent_counts:
        random_row = random_df[(random_df['Agents'] == n) &
                              (random_df['Reward_Type'] == reward_type)]
        ql_row = ql_df[(ql_df['Agents'] == n) &
                       (ql_df['State_Type'] == state_type) &
                       (ql_df['Reward_Type'] == reward_type)]

        if not random_row.empty:
            random_vals.append(float(random_row['CALT'].iloc[0]))
        else:
            random_vals.append(0)

        if not ql_row.empty:
            ql_vals.append(float(ql_row['CALT'].iloc[0]))
        else:
            ql_vals.append(0)

    ax.plot(agent_counts, random_vals, 'o--', label='Random', color='#4DD0E1', linewidth=2.0, markersize=8)
    ax.plot(agent_counts, ql_vals, 's-', label='Q-Learning', color='purple', linewidth=2.5, markersize=9)

    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('CALT (Coordination)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 0.6])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title('Panel C: CALT Reveals Coordination Failure', fontsize=12)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure3_traditional_vs_alt_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure4_2column(base_dir, output_dir):
    """
    Figure 4 (2-column): Symmetric Failure Example (3 Agents Case Study).

    Creates 2 horizontal panels showing Reward Fairness and CALT for 3-agent case.
    Demonstrates how high fairness masks poor coordination.

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    # Extract 3 agents, Type-B, IQF case
    n_agents = 3
    state_type = 'Type-B'
    reward_type = 'IQF'

    random_row = random_df[(random_df['Agents'] == n_agents) &
                           (random_df['Reward_Type'] == reward_type)].iloc[0]
    ql_row = ql_df[(ql_df['Agents'] == n_agents) &
                   (ql_df['State_Type'] == state_type) &
                   (ql_df['Reward_Type'] == reward_type)].iloc[0]

    # Values
    fairness_random = float(random_row['Reward_Fairness'])
    fairness_ql = float(ql_row['Reward_Fairness'])
    calt_random = float(random_row['CALT'])
    calt_ql = float(ql_row['CALT'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Symmetric Failure: High Reward Fairness Masks Poor Coordination\n3 Agents, Type-B State, IQF Reward',
                 fontsize=14, fontweight='bold')

    # Left panel: Reward Fairness comparison
    ax = axes[0]
    x = np.arange(2)
    width = 0.6
    bars = ax.bar(x, [fairness_random, fairness_ql], width,
                  color=['#E0F2F7', 'lightgreen'], edgecolor=['#4DD0E1', 'black'], linewidth=2)

    ax.set_ylabel('Reward Fairness', fontsize=12)
    ax.set_title('Panel A: Reward Fairness', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Random\nBaseline', 'Q-Learning'], fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect (1.0)')
    ax.legend(fontsize=10, loc='lower left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Right panel: CALT comparison
    ax = axes[1]
    bars = ax.bar(x, [calt_random, calt_ql], width,
                  color=['#E0F2F7', 'salmon'], edgecolor=['#4DD0E1', 'black'], linewidth=2)

    ax.set_ylabel('CALT (Coordination)', fontsize=12)
    ax.set_title('Panel B: CALT Coordination', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Random\nBaseline', 'Q-Learning'], fontsize=11)
    ax.set_ylim([0, 0.5])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add degradation annotation (adjusted position to avoid overlap)
    degradation = ((calt_ql - calt_random) / calt_random) * 100
    # Shift arrow to x=0.7 to avoid bar labels
    ax.annotate('', xy=(0.7, calt_ql), xytext=(0.7, calt_random),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
    # Place text to the LEFT of arrow, close to it (ha='right' means text ends at x position)
    ax.text(0.65, (calt_random + calt_ql)/2, f'{degradation:.1f}%',
            fontsize=11, color='red', fontweight='bold', va='center', ha='right')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure4_symmetric_failure_example.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure5_2column(base_dir, output_dir):
    """
    Figure 5 (2-column): CALT Progression During Training (Type-B, IQF) - UNIFIED NORMALIZED VIEW.

    Creates a SINGLE plot showing CALT progression for ALL agent counts (2, 3, 5, 8, 10+).
    X-axis is NORMALIZED (0-1 representing training progress), not raw episodes.
    Each agent count has a different color/line style.

    Args:
        base_dir: Path to results directory with checkpoints/
        output_dir: Path to save output figure
    """
    import pickle

    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()
    checkpoint_dir = os.path.join(base_dir, "checkpoints")

    # Select representative config: Type-B, IQF (best episodic memory case)
    state_type = 'Type-B'
    reward_type = 'IQF'

    # Get all available agent counts from checkpoints
    agent_counts_available = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f"result_") and file.endswith(f"_{state_type}_{reward_type}.pkl"):
            # Extract agent count from filename: result_Nagents_Type-B_IQF.pkl
            try:
                n_agents = int(file.split('_')[1].replace('agents', ''))
                agent_counts_available.append(n_agents)
            except:
                pass
    agent_counts_available = sorted(agent_counts_available)

    # Color palette and line styles for different agent counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('CALT Progression During Q-Learning Training (Normalized): Type-B State, IQF Reward',
                 fontsize=14, fontweight='bold')

    # Plot each agent count
    for idx, n in enumerate(agent_counts_available):
        # Load checkpoint file
        checkpoint_file = os.path.join(checkpoint_dir, f"result_{n}agents_{state_type}_{reward_type}.pkl")

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                result_data = pickle.load(f)

            # Get progression data
            progression_data = result_data.get('progression_data', {})

            if progression_data.get('CALT'):
                calt_values = np.array(progression_data['CALT'])

                # Normalize X-axis: 0 to 1 (training progress)
                normalized_x = np.linspace(0, 1, len(calt_values))

                # Plot Q-learning progression
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                ax.plot(normalized_x, calt_values, linestyle, color=color, linewidth=2.0,
                       label=f'{n} Agents (Q-Learning)', alpha=0.85)

                # Get random baseline CALT value
                random_row = random_df[(random_df['Agents'] == n) &
                                      (random_df['Reward_Type'] == reward_type)]

                if not random_row.empty:
                    random_calt_value = float(random_row['CALT'].iloc[0])

                    # Plot random baseline as horizontal dotted line (lighter shade)
                    ax.axhline(y=random_calt_value, color=color, linestyle=':', linewidth=1.5,
                              alpha=0.4, label=f'{n} Agents (Random)')

    # Axis labels
    ax.set_xlabel('Normalized Training Progress', fontsize=12, fontweight='bold')
    ax.set_ylabel('CALT (Coordination Metric)', fontsize=12, fontweight='bold')

    # Legend (outside plot area)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, framealpha=0.95, fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Set X-axis ticks (0%, 25%, 50%, 75%, 100%)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Y-axis limits
    ax.set_ylim([0, max(ax.get_ylim()[1], 0.5)])

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure5_calt_progression_training.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

# ==============================================================================
# PAPER 1 FIGURES - 1-COLUMN FORMAT (VERTICAL STACKS)
# ==============================================================================

def create_paper1_figure1_1column(base_dir, output_dir):
    """
    Figure 1 (1-column): CALT Comparison (4 vertical panels).

    Creates 4 vertical panels showing CALT values for all configurations.
    Optimized for single-column journal format (7×15).

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    fig, axes = plt.subplots(4, 1, figsize=(7, 15))
    fig.suptitle('Q-Learning vs Random: CALT Coordination Metrics',
                 fontsize=13, fontweight='bold')

    configs = [
        ('Type-A', 'ILF'),
        ('Type-A', 'IQF'),
        ('Type-B', 'ILF'),
        ('Type-B', 'IQF')
    ]

    agent_counts = [2, 3, 5, 8]

    for idx, (state_type, reward_type) in enumerate(configs):
        ax = axes[idx]
        x_positions = np.arange(len(agent_counts))
        width = 0.35

        random_calt = []
        ql_calt = []

        for n in agent_counts:
            random_row = random_df[(random_df['Agents'] == n) &
                                  (random_df['Reward_Type'] == reward_type)]
            ql_row = ql_df[(ql_df['Agents'] == n) &
                           (ql_df['State_Type'] == state_type) &
                           (ql_df['Reward_Type'] == reward_type)]

            if not random_row.empty:
                random_calt.append(float(random_row['CALT'].iloc[0]))
            else:
                random_calt.append(0)

            if not ql_row.empty:
                ql_calt.append(float(ql_row['CALT'].iloc[0]))
            else:
                ql_calt.append(0)

        bars1 = ax.bar(x_positions - width/2, random_calt, width,
                       label='Random', color='#B3E5FC', edgecolor='#4DD0E1', linewidth=1.2)
        bars2 = ax.bar(x_positions + width/2, ql_calt, width,
                       label='Q-Learning', color='salmon', edgecolor='black', linewidth=1.2)

        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('CALT', fontsize=10, fontweight='bold')
        ax.set_title(f'{state_type}, {reward_type}', fontsize=11, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(agent_counts)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(max(random_calt), max(ql_calt)) * 1.15])

    axes[-1].set_xlabel('Number of Agents', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure1_1col.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure2_1column(base_dir, output_dir):
    """
    Figure 2 (1-column): PA-Equivalent Performance (4 vertical panels).

    Creates 4 vertical panels showing PA-equivalent agents for all configurations.
    Optimized for single-column journal format (7×15).

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    def calculate_pa_equivalent(calt_value):
        if calt_value <= 0:
            return 0.0
        return calt_value ** 0.5

    fig, axes = plt.subplots(4, 1, figsize=(7, 15))
    fig.suptitle('PA-Equivalent Performance (CALT-based)',
                 fontsize=13, fontweight='bold')

    configs = [
        ('Type-A', 'ILF'),
        ('Type-A', 'IQF'),
        ('Type-B', 'ILF'),
        ('Type-B', 'IQF')
    ]

    agent_counts = [2, 3, 5, 8]

    for idx, (state_type, reward_type) in enumerate(configs):
        ax = axes[idx]
        x_positions = np.arange(len(agent_counts))
        width = 0.35

        random_equiv = []
        ql_equiv = []

        for n in agent_counts:
            random_row = random_df[(random_df['Agents'] == n) &
                                  (random_df['Reward_Type'] == reward_type)]
            ql_row = ql_df[(ql_df['Agents'] == n) &
                           (ql_df['State_Type'] == state_type) &
                           (ql_df['Reward_Type'] == reward_type)]

            if not random_row.empty:
                random_calt = float(random_row['CALT'].iloc[0])
                random_equiv.append(n * calculate_pa_equivalent(random_calt))
            else:
                random_equiv.append(0)

            if not ql_row.empty:
                ql_calt = float(ql_row['CALT'].iloc[0])
                ql_equiv.append(n * calculate_pa_equivalent(ql_calt))
            else:
                ql_equiv.append(0)

        bars1 = ax.bar(x_positions - width/2, random_equiv, width,
                       label='Random', color='#E0F2F7', edgecolor='#4DD0E1', linewidth=1.2, hatch='//')
        bars2 = ax.bar(x_positions + width/2, ql_equiv, width,
                       label='Q-Learning', color='salmon', edgecolor='black', linewidth=1.2)

        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('PA-Equivalent Agents', fontsize=10, fontweight='bold')
        ax.set_title(f'{state_type}, {reward_type}', fontsize=11, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(agent_counts)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    axes[-1].set_xlabel('Number of Agents', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure2_1col.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure3_1column(base_dir, output_dir):
    """
    Figure 3 (1-column): Traditional vs ALT Comparison (3 vertical panels).

    Creates 3 vertical panels showing Efficiency, Reward Fairness, and CALT.
    Optimized for single-column journal format (7×12).

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    state_type = 'Type-B'
    reward_type = 'IQF'
    agent_counts = [2, 3, 5, 8]

    fig, axes = plt.subplots(3, 1, figsize=(7, 12))
    fig.suptitle('Traditional Metrics Hide Coordination Failure\n(Type-B, IQF)',
                 fontsize=13, fontweight='bold')

    # Efficiency
    ax = axes[0]
    random_vals = []
    ql_vals = []

    for n in agent_counts:
        random_row = random_df[(random_df['Agents'] == n) &
                              (random_df['Reward_Type'] == reward_type)]
        ql_row = ql_df[(ql_df['Agents'] == n) &
                       (ql_df['State_Type'] == state_type) &
                       (ql_df['Reward_Type'] == reward_type)]

        if not random_row.empty:
            random_vals.append(float(random_row['Efficiency'].iloc[0]))
        else:
            random_vals.append(0)

        if not ql_row.empty:
            ql_vals.append(float(ql_row['Efficiency'].iloc[0]))
        else:
            ql_vals.append(0)

    ax.plot(agent_counts, random_vals, 'o--', label='Random', color='#4DD0E1', linewidth=2.0, markersize=8)
    ax.plot(agent_counts, ql_vals, 's-', label='Q-Learning', color='orange', linewidth=2.5, markersize=9)
    ax.set_ylabel('Efficiency', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title('Panel A: Efficiency', fontsize=11)

    # Reward Fairness
    ax = axes[1]
    random_vals = []
    ql_vals = []

    for n in agent_counts:
        random_row = random_df[(random_df['Agents'] == n) &
                              (random_df['Reward_Type'] == reward_type)]
        ql_row = ql_df[(ql_df['Agents'] == n) &
                       (ql_df['State_Type'] == state_type) &
                       (ql_df['Reward_Type'] == reward_type)]

        if not random_row.empty:
            random_vals.append(float(random_row['Reward_Fairness'].iloc[0]))
        else:
            random_vals.append(0)

        if not ql_row.empty:
            ql_vals.append(float(ql_row['Reward_Fairness'].iloc[0]))
        else:
            ql_vals.append(0)

    ax.plot(agent_counts, random_vals, 'o--', label='Random', color='#4DD0E1', linewidth=2.0, markersize=8)
    ax.plot(agent_counts, ql_vals, 's-', label='Q-Learning', color='green', linewidth=2.5, markersize=9)
    ax.set_ylabel('Reward Fairness', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title('Panel B: Reward Fairness', fontsize=11)

    # CALT
    ax = axes[2]
    random_vals = []
    ql_vals = []

    for n in agent_counts:
        random_row = random_df[(random_df['Agents'] == n) &
                              (random_df['Reward_Type'] == reward_type)]
        ql_row = ql_df[(ql_df['Agents'] == n) &
                       (ql_df['State_Type'] == state_type) &
                       (ql_df['Reward_Type'] == reward_type)]

        if not random_row.empty:
            random_vals.append(float(random_row['CALT'].iloc[0]))
        else:
            random_vals.append(0)

        if not ql_row.empty:
            ql_vals.append(float(ql_row['CALT'].iloc[0]))
        else:
            ql_vals.append(0)

    ax.plot(agent_counts, random_vals, 'o--', label='Random', color='#4DD0E1', linewidth=2.0, markersize=8)
    ax.plot(agent_counts, ql_vals, 's-', label='Q-Learning', color='purple', linewidth=2.5, markersize=9)
    ax.set_xlabel('Number of Agents', fontsize=11, fontweight='bold')
    ax.set_ylabel('CALT', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 0.6])
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title('Panel C: CALT Reveals Failure', fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure3_1col.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure4_1column(base_dir, output_dir):
    """
    Figure 4 (1-column): Symmetric Failure (2 vertical panels).

    Creates 2 vertical panels showing Reward Fairness and CALT for 3-agent case.
    Optimized for single-column journal format (7×9).

    Args:
        base_dir: Path to results directory with comprehensive_results.csv
        output_dir: Path to save output figure
    """
    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    ql_df = comprehensive_df[comprehensive_df['Is_Random'] == False].copy()
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()

    n_agents = 3
    state_type = 'Type-B'
    reward_type = 'IQF'

    random_row = random_df[(random_df['Agents'] == n_agents) &
                           (random_df['Reward_Type'] == reward_type)].iloc[0]
    ql_row = ql_df[(ql_df['Agents'] == n_agents) &
                   (ql_df['State_Type'] == state_type) &
                   (ql_df['Reward_Type'] == reward_type)].iloc[0]

    fairness_random = float(random_row['Reward_Fairness'])
    fairness_ql = float(ql_row['Reward_Fairness'])
    calt_random = float(random_row['CALT'])
    calt_ql = float(ql_row['CALT'])

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    fig.suptitle('Symmetric Failure: High Fairness Masks Poor Coordination\n(3 Agents, Type-B, IQF)',
                 fontsize=12, fontweight='bold')

    # Panel A: Reward Fairness
    ax = axes[0]
    x = np.arange(2)
    width = 0.5
    bars = ax.bar(x, [fairness_random, fairness_ql], width,
                  color=['#E0F2F7', 'lightgreen'], edgecolor=['#4DD0E1', 'black'], linewidth=2)

    ax.set_ylabel('Reward Fairness', fontsize=11, fontweight='bold')
    ax.set_title('Panel A: Reward Fairness', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Random\nBaseline', 'Q-Learning'], fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect')
    ax.legend(fontsize=9, loc='lower left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel B: CALT
    ax = axes[1]
    bars = ax.bar(x, [calt_random, calt_ql], width,
                  color=['#E0F2F7', 'salmon'], edgecolor=['#4DD0E1', 'black'], linewidth=2)

    ax.set_ylabel('CALT', fontsize=11, fontweight='bold')
    ax.set_title('Panel B: CALT Coordination', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Random\nBaseline', 'Q-Learning'], fontsize=10)
    ax.set_ylim([0, 0.5])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Degradation arrow
    degradation = ((calt_ql - calt_random) / calt_random) * 100
    ax.annotate('', xy=(0.7, calt_ql), xytext=(0.7, calt_random),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
    ax.text(0.65, (calt_random + calt_ql)/2, f'{degradation:.1f}%',
            fontsize=10, color='red', fontweight='bold', va='center', ha='right')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure4_1col.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_paper1_figure5_1column(base_dir, output_dir):
    """
    Figure 5 (1-column): CALT Progression - UNIFIED NORMALIZED VIEW.

    Creates a SINGLE plot showing CALT progression for ALL agent counts (2, 3, 5, 8, 10+).
    X-axis is NORMALIZED (0-1 representing training progress).
    Optimized for single-column journal format (7×12).

    Args:
        base_dir: Path to results directory with checkpoints/
        output_dir: Path to save output figure
    """
    import pickle

    # Load data
    comprehensive_df = pd.read_csv(os.path.join(base_dir, "data_tables", "comprehensive_results.csv"))
    random_df = comprehensive_df[comprehensive_df['Is_Random'] == True].copy()
    checkpoint_dir = os.path.join(base_dir, "checkpoints")

    state_type = 'Type-B'
    reward_type = 'IQF'

    # Get all available agent counts from checkpoints
    agent_counts_available = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f"result_") and file.endswith(f"_{state_type}_{reward_type}.pkl"):
            try:
                n_agents = int(file.split('_')[1].replace('agents', ''))
                agent_counts_available.append(n_agents)
            except:
                pass
    agent_counts_available = sorted(agent_counts_available)

    # Color palette and line styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 9))
    fig.suptitle('CALT Progression During Training (Normalized)\nType-B State, IQF Reward',
                 fontsize=12, fontweight='bold')

    # Plot each agent count
    for idx, n in enumerate(agent_counts_available):
        checkpoint_file = os.path.join(checkpoint_dir, f"result_{n}agents_{state_type}_{reward_type}.pkl")

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                result_data = pickle.load(f)

            progression_data = result_data.get('progression_data', {})

            if progression_data.get('CALT'):
                calt_values = np.array(progression_data['CALT'])
                normalized_x = np.linspace(0, 1, len(calt_values))

                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                ax.plot(normalized_x, calt_values, linestyle, color=color, linewidth=2.0,
                       label=f'{n} Agents (Q-Learning)', alpha=0.85)

                # Random baseline
                random_row = random_df[(random_df['Agents'] == n) &
                                      (random_df['Reward_Type'] == reward_type)]
                if not random_row.empty:
                    random_calt_value = float(random_row['CALT'].iloc[0])
                    ax.axhline(y=random_calt_value, color=color, linestyle=':', linewidth=1.5,
                              alpha=0.4, label=f'{n} Agents (Random)')

    ax.set_xlabel('Normalized Training Progress', fontsize=11, fontweight='bold')
    ax.set_ylabel('CALT (Coordination Metric)', fontsize=11, fontweight='bold')

    ax.legend(loc='best', frameon=True, framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_ylim([0, max(ax.get_ylim()[1], 0.5)])

    plt.tight_layout()
    output_file = os.path.join(output_dir, "figure5_1col.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

def create_consolidated_qlearning_vs_random_comparison(base_dir, output_dir):
    """
    Creates consolidated comparison showing Q-Learning vs Random for all metrics.

    Shows THE MAIN FINDING: Q-Learning performs 12-25% WORSE than random despite
    high traditional metrics (Efficiency ~0.84, Fairness 1.0).

    - 3 metrics on same graph: Efficiency, Reward_Fairness, CALT
    - All agent counts (2, 3, 5, 8) on x-axis
    - Q-Learning (solid lines) vs Random (dashed semi-transparent)
    - Different colors per metric: Orange (Efficiency), Green (Reward_Fairness), Purple (CALT)

    Args:
        base_dir: Path to results directory
        output_dir: Path to save output figure
    """
    import pandas as pd
    import numpy as np

    print("\n[...] Creating consolidated Q-Learning vs Random comparison...")

    # Load comprehensive results
    data_file = os.path.join(base_dir, "data_tables", "comprehensive_results.csv")
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)

    # Separate Q-Learning and Random results
    ql_df = df[df['Is_Random'] == False].copy()
    random_df = df[df['Is_Random'] == True].copy()

    # Average across Type-A/Type-B and ILF/IQF for each agent count
    agent_counts = sorted(ql_df['Agents'].unique())

    # Prepare data for each metric
    metrics = {
        'Efficiency': {'ql': [], 'random': [], 'color': '#FF8C42', 'label': 'Efficiency'},
        'Reward_Fairness': {'ql': [], 'random': [], 'color': '#4CAF50', 'label': 'Reward Fairness'},
        'CALT': {'ql': [], 'random': [], 'color': '#9C27B0', 'label': 'CALT'}
    }

    for n_agents in agent_counts:
        ql_subset = ql_df[ql_df['Agents'] == n_agents]
        random_subset = random_df[random_df['Agents'] == n_agents]

        for metric_name in metrics.keys():
            # Average across all 4 configurations (Type-A/B × ILF/IQF)
            ql_mean = ql_subset[metric_name].mean()
            random_mean = random_subset[metric_name].mean()

            metrics[metric_name]['ql'].append(ql_mean)
            metrics[metric_name]['random'].append(random_mean)

    # Create figure (2-column format: 10 inches wide × 6 inches tall)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each metric
    for metric_name, data in metrics.items():
        color = data['color']
        label = data['label']

        # Q-Learning (solid line, thicker)
        ax.plot(agent_counts, data['ql'],
                marker='o', markersize=8, linewidth=2.5,
                color=color, linestyle='-',
                label=f'{label} (Q-Learning)')

        # Random baseline (dashed, semi-transparent)
        ax.plot(agent_counts, data['random'],
                marker='s', markersize=6, linewidth=2,
                color=color, linestyle='--', alpha=0.5,
                label=f'{label} (Random)')

    # Formatting
    ax.set_xlabel('Number of Agents', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title('Q-Learning vs Random Baseline: All Metrics Comparison',
                 fontsize=16, fontweight='bold', pad=15)

    # Set x-axis to show only agent counts
    ax.set_xticks(agent_counts)
    ax.set_xticklabels(agent_counts)

    # Set y-axis range to 0-1
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Legend (outside plot area)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, framealpha=0.9, fontsize=10)

    # Tick labels
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, "figure_qlearning_vs_random_consolidated.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")
    plt.close()

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS FROM CONSOLIDATED COMPARISON")
    print("=" * 80)

    for metric_name, data in metrics.items():
        print(f"\n{metric_name}:")
        for i, n_agents in enumerate(agent_counts):
            ql_val = data['ql'][i]
            rand_val = data['random'][i]
            diff_pct = ((ql_val - rand_val) / rand_val) * 100
            print(f"  {n_agents} agents: Q-Learning={ql_val:.4f}, Random={rand_val:.4f}, "
                  f"Difference={diff_pct:+.1f}%")

def create_all_paper1_figures(base_dir, output_dir="CNSNS_paper/figures"):
    """
    Master function to generate all Paper 1 figures (2-column + 1-column formats).

    Generates 10 total figures:
    - 5 figures in 2-column format (14×10, 12×14, 14×6)
    - 5 figures in 1-column format (7×15, 7×12, 7×9)

    Args:
        base_dir: Path to results directory (must contain data_tables/ and checkpoints/)
        output_dir: Path to save output figures (default: CNSNS_paper/figures)

    Returns:
        Dictionary with paths to all generated figures
    """
    print("\n" + "=" * 80)
    print("GENERATING PAPER 1 FIGURES (2-COLUMN + 1-COLUMN)")
    print("=" * 80)
    print(f"Data source: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track generated files
    generated_files = {
        '2column': [],
        '1column': []
    }

    # Generate 2-column figures
    print("Creating 2-column figures (14×10, 12×14, 14×6)...")
    print("-" * 80)

    try:
        create_paper1_figure1_2column(base_dir, output_dir)
        generated_files['2column'].append("figure1_qlearning_vs_random_calt.png")
        print("  [OK] Figure 1 (2-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 1 (2-column) FAILED: {e}")

    try:
        create_paper1_figure2_2column(base_dir, output_dir)
        generated_files['2column'].append("figure2_pa_equivalent_with_baseline.png")
        print("  [OK] Figure 2 (2-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 2 (2-column) FAILED: {e}")

    try:
        create_paper1_figure3_2column(base_dir, output_dir)
        generated_files['2column'].append("figure3_traditional_vs_alt_comparison.png")
        print("  [OK] Figure 3 (2-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 3 (2-column) FAILED: {e}")

    try:
        create_paper1_figure4_2column(base_dir, output_dir)
        generated_files['2column'].append("figure4_symmetric_failure_example.png")
        print("  [OK] Figure 4 (2-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 4 (2-column) FAILED: {e}")

    try:
        create_paper1_figure5_2column(base_dir, output_dir)
        generated_files['2column'].append("figure5_calt_progression_training.png")
        print("  [OK] Figure 5 (2-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 5 (2-column) FAILED: {e}")

    print()
    print("Creating 1-column figures (7×15, 7×12, 7×9)...")
    print("-" * 80)

    try:
        create_paper1_figure1_1column(base_dir, output_dir)
        generated_files['1column'].append("figure1_1col.png")
        print("  [OK] Figure 1 (1-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 1 (1-column) FAILED: {e}")

    try:
        create_paper1_figure2_1column(base_dir, output_dir)
        generated_files['1column'].append("figure2_1col.png")
        print("  [OK] Figure 2 (1-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 2 (1-column) FAILED: {e}")

    try:
        create_paper1_figure3_1column(base_dir, output_dir)
        generated_files['1column'].append("figure3_1col.png")
        print("  [OK] Figure 3 (1-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 3 (1-column) FAILED: {e}")

    try:
        create_paper1_figure4_1column(base_dir, output_dir)
        generated_files['1column'].append("figure4_1col.png")
        print("  [OK] Figure 4 (1-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 4 (1-column) FAILED: {e}")

    try:
        create_paper1_figure5_1column(base_dir, output_dir)
        generated_files['1column'].append("figure5_1col.png")
        print("  [OK] Figure 5 (1-column)")
    except Exception as e:
        print(f"  [ERROR] Figure 5 (1-column) FAILED: {e}")

    print()
    print("=" * 80)
    print("[SUCCESS] ALL PAPER 1 FIGURES GENERATED!")
    print("=" * 80)
    print()
    print(f"Output directory: {output_dir}/")
    print()
    print("2-column figures (for 2-column journal format):")
    for fname in generated_files['2column']:
        print(f"  - {fname}")
    print()
    print("1-column figures (for 1-column journal format):")
    for fname in generated_files['1column']:
        print(f"  - {fname}")
    print()
    print("=" * 80)

    return generated_files


def organize_appendix_figures(base_dir, output_dir="appendix"):
    """
    Organize all supplementary figures into appendix/ subfolder.

    Copies relevant figures from analysis subdirectories into organized appendix structure.

    Args:
        base_dir: Path to run directory containing all analysis folders
        output_dir: Name of appendix directory (default: 'appendix')

    Returns:
        Dictionary with counts of copied figures by category
    """
    import shutil

    print("\n" + "=" * 80)
    print("ORGANIZING APPENDIX FIGURES")
    print("=" * 80)
    print(f"Source: {base_dir}")

    appendix_dir = os.path.join(base_dir, output_dir)
    os.makedirs(appendix_dir, exist_ok=True)
    print(f"Output: {appendix_dir}")
    print()

    # Track copied figures
    copied = {
        'learning_phases': 0,
        'qtable_analysis': 0,
        'agent_performance': 0,
        '3d_analysis': 0,
        'alt_ratio': 0,
        'comparison': 0
    }

    # Category 1: Learning Phases (epsilon decay and metrics evolution)
    learning_dir = os.path.join(base_dir, 'learning_phases')
    if os.path.exists(learning_dir):
        target = os.path.join(appendix_dir, '01_learning_phases')
        os.makedirs(target, exist_ok=True)
        for fname in os.listdir(learning_dir):
            if fname.endswith('.png'):
                shutil.copy2(os.path.join(learning_dir, fname), os.path.join(target, fname))
                copied['learning_phases'] += 1
        print(f"[OK] Copied {copied['learning_phases']} learning phase figures")

    # Category 2: Q-Table Analysis (heatmaps)
    qtable_dir = os.path.join(base_dir, 'qtable_analysis')
    if os.path.exists(qtable_dir):
        target = os.path.join(appendix_dir, '02_qtable_heatmaps')
        os.makedirs(target, exist_ok=True)
        for fname in os.listdir(qtable_dir):
            if fname.endswith('.png'):
                shutil.copy2(os.path.join(qtable_dir, fname), os.path.join(target, fname))
                copied['qtable_analysis'] += 1
        print(f"[OK] Copied {copied['qtable_analysis']} Q-table heatmap figures")

    # Category 3: Agent Performance Analysis
    agent_perf_dir = os.path.join(base_dir, 'agent_performance_analysis')
    if os.path.exists(agent_perf_dir):
        target = os.path.join(appendix_dir, '03_agent_performance')
        os.makedirs(target, exist_ok=True)
        for fname in os.listdir(agent_perf_dir):
            if fname.endswith('.png'):
                shutil.copy2(os.path.join(agent_perf_dir, fname), os.path.join(target, fname))
                copied['agent_performance'] += 1
        print(f"[OK] Copied {copied['agent_performance']} agent performance figures")

    # Category 4: 3D Analysis
    analysis_3d_dir = os.path.join(base_dir, '3d_analysis')
    if os.path.exists(analysis_3d_dir):
        target = os.path.join(appendix_dir, '04_3d_trajectory_analysis')
        os.makedirs(target, exist_ok=True)
        for fname in os.listdir(analysis_3d_dir):
            if fname.endswith('.png'):
                shutil.copy2(os.path.join(analysis_3d_dir, fname), os.path.join(target, fname))
                copied['3d_analysis'] += 1
        print(f"[OK] Copied {copied['3d_analysis']} 3D trajectory figures")

    # Category 5: ALT Ratio Analysis
    alt_ratio_dir = os.path.join(base_dir, 'alt_ratio_analysis')
    if os.path.exists(alt_ratio_dir):
        target = os.path.join(appendix_dir, '05_alt_ratio_analysis')
        os.makedirs(target, exist_ok=True)
        for fname in os.listdir(alt_ratio_dir):
            if fname.endswith('.png'):
                shutil.copy2(os.path.join(alt_ratio_dir, fname), os.path.join(target, fname))
                copied['alt_ratio'] += 1
        print(f"[OK] Copied {copied['alt_ratio']} ALT ratio analysis figures")

    # Category 6: Comparison Metrics
    comparison_dir = os.path.join(base_dir, 'comparison_metrics')
    if os.path.exists(comparison_dir):
        target = os.path.join(appendix_dir, '06_metric_comparisons')
        os.makedirs(target, exist_ok=True)
        for fname in os.listdir(comparison_dir):
            if fname.endswith('.png'):
                shutil.copy2(os.path.join(comparison_dir, fname), os.path.join(target, fname))
                copied['comparison'] += 1
        print(f"[OK] Copied {copied['comparison']} comparison metric figures")

    total = sum(copied.values())
    print()
    print("=" * 80)
    print(f"[SUCCESS] ORGANIZED {total} APPENDIX FIGURES!")
    print("=" * 80)
    print(f"\nAppendix directory: {appendix_dir}/")
    print()
    for category, count in copied.items():
        if count > 0:
            print(f"  {category:20s}: {count:3d} figures")
    print()
    print("=" * 80)

    return copied


def main(auto_mode=False, experiment_type='theory'):
    """Main execution function.

    Args:
        auto_mode: If True, skip all user prompts and use defaults
        experiment_type: 'theory' or 'adaptive' (default: 'theory')
    """
    print("=" * 80)
    print("ALT METRICS PAPER 1 - EXPERIMENTAL FRAMEWORK")
    print("=" * 80)

    # Check if results already exist
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
        # No existing results
        if auto_mode:
            print(f"\nNo previous results found.")
            print(f"AUTO MODE: Using {experiment_type} approach.")
        else:
            print("\nNo previous results found. Which approach do you want to use for determining episode count?")
            print("1. Theory-based (calculates episodes based on agent count and complexity)")
            print("2. Adaptive (runs until ALT thresholds are reached)")
            approach_choice = input("Enter choice (1 or 2): ").strip()
            experiment_type = 'theory' if approach_choice == '1' else 'adaptive'

    # Run experiments with selected approach
    results_list = run_experiments(experiment_type)

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
    import sys
    random.seed(42)
    np.random.seed(42)

    # Check for command-line arguments
    auto_mode = '--auto' in sys.argv or '-a' in sys.argv
    experiment_type = 'theory'  # default

    if '--adaptive' in sys.argv:
        experiment_type = 'adaptive'

    results = main(auto_mode=auto_mode, experiment_type=experiment_type)