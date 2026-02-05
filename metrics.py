"""
Metrics calculation module for ALT Metrics experiments.

This module contains all metric computation functions:
- Episode calculation (theory-based)
- ALT metrics (FALT, EALT, qFALT, qEALT, CALT, AALT)
- RP metrics (AWE, WPE, RP_avg)
- Traditional metrics (Efficiency, Reward_Fairness, TT_Fairness, Fairness)
- ALT Ratio calculation and coordination scores

IMPORTANT BUG FIXES (Dec 29, 2025):
- AALT now correctly counts exclusive win EPISODES, not agents
- Centralized batch loop for efficiency (runs once, not 6 times)
- qFALT/qEALT now simply square FALT/EALT (no recalculation)
- Renamed num_winners → num_terminal_agents (semantic clarity)
"""

import numpy as np
import time
import math

#===============================================================================
# Episode Calculation
#===============================================================================

def calculate_episodes_theory_based(num_agents, base_episodes):
    """Calculate episodes based on RL theory and coordination complexity.

    PRODUCTION: base_episodes=100000 (full experiments, ~8.7 days)
    TEST: base_episodes=1000 (verify fixes, ~10-15 minutes)
    ULTRA-QUICK: base_episodes=500 (quick testing with agents=[2,3], ~2 min)

    NOTE: base_episodes is REQUIRED parameter (no default to avoid confusion)
    """
    # For 2 agents, return the base
    if num_agents == 2:
        return base_episodes

    # State-action space complexity factor
    state_complexity = (num_agents/2)**2

    # Coordination complexity factor
    coordination_factor = math.log(math.factorial(num_agents) / math.factorial(2))

    # Combined scaling factor
    scaling_factor = state_complexity * (1 + coordination_factor)

    # Calculate the final number of episodes
    return int(base_episodes * scaling_factor)

#===============================================================================
# ALT Metrics Implementation - OPTIMIZED VERSION (Dec 29, 2025)
#===============================================================================

def compute_alt_metrics_optimized(num_episodes, num_agents, terminal_occurrences_per_episode, top_agents_per_episode):
    """
    Compute all ALT metrics in a SINGLE PASS through batches (OPTIMIZED).

    This replaces 6 separate functions with redundant batch loops.
    Now runs the batch loop ONCE and computes all metrics together.

    BUG FIXES:
    - AALT: Now correctly counts exclusive win EPISODES (g_j), not agents
    - Variable naming: num_winners → num_terminal_agents (semantic clarity)
    - qFALT/qEALT: Simply square FALT/EALT (no recalculation)

    Returns:
        alt_metrics: Dict with metric values
        beta_values: Dict with per-batch beta arrays
        computation_times: Dict with individual and total timing
    """
    overall_start = time.time()

    num_batches = num_episodes - (num_agents - 1)

    # Initialize beta arrays for all metrics
    beta_FALT = np.zeros(num_batches)
    beta_EALT = np.zeros(num_batches)
    beta_qFALT = np.zeros(num_batches)  # Renamed from EFALT
    beta_qEALT = np.zeros(num_batches)  # Renamed from EEALT
    beta_CALT = np.zeros(num_batches)
    beta_AALT = np.zeros(num_batches)

    # Individual timing for each metric (more accurate than batch_loop/6)
    time_data_collection = 0  # Shared loop overhead
    time_falt_calc = 0
    time_ealt_calc = 0
    time_qfalt_calc = 0  # Includes FALT + squaring
    time_qealt_calc = 0  # Includes EALT + squaring
    time_calt_calc = 0
    time_aalt_calc = 0

    # ========== SINGLE CENTRALIZED BATCH LOOP ==========
    for batch_id in range(num_batches):
        # ===== DATA COLLECTION (shared by all metrics) =====
        t_data_start = time.time()

        who_reached_terminal_in_batch = np.zeros(num_agents)
        number_of_winnings_in_batch_per_agent = np.zeros(num_agents)  # For AALT
        term_occ = 0
        exclusive_win_episodes = 0

        # Scan all episodes in this batch
        for eb in range(batch_id, batch_id + num_agents):
            term_occ += terminal_occurrences_per_episode[eb]

            num_terminal_in_episode = sum(top_agents_per_episode[eb])
            if num_terminal_in_episode == 1:
                exclusive_win_episodes += 1

            for i in range(num_agents):
                if top_agents_per_episode[eb][i] == 1:
                    who_reached_terminal_in_batch[i] = 1
                    number_of_winnings_in_batch_per_agent[i] += 1  # Count wins per agent

        num_terminal_agents = sum(who_reached_terminal_in_batch)
        time_data_collection += time.time() - t_data_start

        # ===== FALT: f_j / t_j =====
        t_falt_start = time.time()
        if term_occ > 0:
            beta_FALT[batch_id] = num_terminal_agents / term_occ
        else:
            beta_FALT[batch_id] = 0
        time_falt_calc += time.time() - t_falt_start

        # ===== EALT: (w_j × f_j) / n² =====
        t_ealt_start = time.time()
        beta_EALT[batch_id] = (exclusive_win_episodes * num_terminal_agents) / (num_agents ** 2)
        time_ealt_calc += time.time() - t_ealt_start

        # ===== qFALT: Square each batch's FALT value =====
        # Formula: β_j^qFALT = (β_j^FALT)²
        # Meaning: Take the FALT value we just calculated and square it
        t_qfalt_start = time.time()
        beta_qFALT[batch_id] = beta_FALT[batch_id] ** 2
        time_qfalt_calc += time.time() - t_qfalt_start

        # ===== qEALT: Square each batch's EALT value =====
        # Formula: β_j^qEALT = (β_j^EALT)²
        # Meaning: Take the EALT value we just calculated and square it
        t_qealt_start = time.time()
        beta_qEALT[batch_id] = beta_EALT[batch_id] ** 2
        time_qealt_calc += time.time() - t_qealt_start

        # ===== CALT: Weighted sum using qFALT =====
        # Formula: β_j^CALT = Σ[(n - Y_k) × β_j^qFALT] / [n(n-1)]
        t_calt_start = time.time()
        if num_agents > 1:
            calt_sum = 0
            for eb in range(batch_id, batch_id + num_agents):
                Y_k = sum(top_agents_per_episode[eb])
                calt_sum += (num_agents - Y_k) * beta_qFALT[batch_id]
            beta_CALT[batch_id] = calt_sum / (num_agents * (num_agents - 1))
        else:
            beta_CALT[batch_id] = 0
        time_calt_calc += time.time() - t_calt_start

        # ===== AALT: Agents with exactly 1 win / total terminal occurrences =====
        # Formula: β_j^AALT = g_j / t_j (g_j = count of AGENTS who won exactly 1 time)
        # Legacy code (line 139-142): countOfExclusiveWinnersInBatch / termOccPerBatch
        t_aalt_start = time.time()
        if term_occ > 0:
            count_of_exclusive_winners_in_batch = sum(1 for wins in number_of_winnings_in_batch_per_agent if wins == 1)
            beta_AALT[batch_id] = count_of_exclusive_winners_in_batch / term_occ
        else:
            beta_AALT[batch_id] = 0
        time_aalt_calc += time.time() - t_aalt_start

    time_batch_loop = time.time() - overall_start

    # ========== COMPUTE FINAL METRIC VALUES ==========
    time_averaging_start = time.time()

    falt_value = sum(beta_FALT) / num_batches
    ealt_value = sum(beta_EALT) / num_batches
    qfalt_value = sum(beta_qFALT) / num_batches
    qealt_value = sum(beta_qEALT) / num_batches
    calt_value = sum(beta_CALT) / num_batches
    aalt_value = sum(beta_AALT) / num_batches

    time_averaging = time.time() - time_averaging_start
    total_time = time.time() - overall_start

    # ========== RETURN RESULTS ==========

    alt_metrics = {
        'FALT': falt_value,
        'EALT': ealt_value,
        'qFALT': qfalt_value,   # Renamed from EFALT
        'qEALT': qealt_value,   # Renamed from EEALT
        'CALT': calt_value,
        'AALT': aalt_value
    }

    beta_values = {
        'FALT': beta_FALT,
        'EALT': beta_EALT,
        'qFALT': beta_qFALT,    # Renamed from EFALT
        'qEALT': beta_qEALT,    # Renamed from EEALT
        'CALT': beta_CALT,
        'AALT': beta_AALT
    }

    # Accurate computation times including dependencies
    computation_times = {
        # FALT: data collection + division
        'FALT': time_data_collection + time_falt_calc,

        # EALT: data collection + multiplication/division
        'EALT': time_data_collection + time_ealt_calc,

        # qFALT: data collection + FALT calculation + squaring
        'qFALT': time_data_collection + time_falt_calc + time_qfalt_calc,

        # qEALT: data collection + EALT calculation + squaring
        'qEALT': time_data_collection + time_ealt_calc + time_qealt_calc,

        # CALT: data collection + FALT + qFALT + weighted sum
        'CALT': time_data_collection + time_falt_calc + time_qfalt_calc + time_calt_calc,

        # AALT: data collection + division
        'AALT': time_data_collection + time_aalt_calc,

        # Breakdown for analysis
        'data_collection': time_data_collection,
        'averaging': time_averaging,
        'Total': total_time
    }

    return alt_metrics, beta_values, computation_times

def compute_alt_metrics(num_episodes, num_agents, terminal_occurrences_per_episode, top_agents_per_episode):
    """
    Compute all ALT metrics - REDIRECTS to optimized version.

    This function now uses the centralized batch loop implementation
    for better performance and correctness.
    """
    return compute_alt_metrics_optimized(num_episodes, num_agents, terminal_occurrences_per_episode, top_agents_per_episode)

#===============================================================================
# RP Metrics (Calculated but NOT displayed for Paper 1)
#===============================================================================

def compute_awe(num_episodes, num_agents, top_agents_per_episode):
    """Compute Average Waiting Episodes (AWE)."""
    start_time = time.time()

    winners_per_episode = []
    for ep in range(num_episodes):
        if sum(top_agents_per_episode[ep]) == 1:
            for i in range(num_agents):
                if top_agents_per_episode[ep][i] == 1:
                    winners_per_episode.append(i)
                    break
        else:
            winners_per_episode.append(-1)

    awe_values = []
    ideal_waiting = num_agents - 1

    for i in range(num_agents):
        win_episodes = [ep for ep in range(num_episodes) if winners_per_episode[ep] == i]

        if len(win_episodes) > 1:
            waiting_periods = []
            for j in range(1, len(win_episodes)):
                waiting = win_episodes[j] - win_episodes[j-1] - 1
                waiting_periods.append(waiting)

            avg_wait = sum(waiting_periods) / len(waiting_periods) if waiting_periods else 0

            if avg_wait < 2 * ideal_waiting:
                awe = 1.0 - abs(avg_wait - ideal_waiting) / ideal_waiting
            else:
                awe = 0.0
        else:
            awe = 0.0

        awe_values.append(awe)

    awe_avg = sum(awe_values) / num_agents
    computation_time = time.time() - start_time

    return awe_values, awe_avg, computation_time

def compute_wpe(num_episodes, num_agents, top_agents_per_episode):
    """Compute Waiting Periods Evaluation (WPE)."""
    start_time = time.time()

    winners_per_episode = []
    for ep in range(num_episodes):
        if sum(top_agents_per_episode[ep]) == 1:
            for i in range(num_agents):
                if top_agents_per_episode[ep][i] == 1:
                    winners_per_episode.append(i)
                    break
        else:
            winners_per_episode.append(-1)

    wpe_values = []
    ideal_periods = num_episodes / num_agents

    for i in range(num_agents):
        win_episodes = [ep for ep in range(num_episodes) if winners_per_episode[ep] == i]

        if len(win_episodes) > 0:
            actual_periods = len(win_episodes)
            if 0 not in win_episodes:
                actual_periods += 1
            if (num_episodes - 1) not in win_episodes:
                actual_periods += 1
        else:
            actual_periods = 1

        if actual_periods < 2 * ideal_periods:
            wpe = 1.0 - abs(actual_periods - ideal_periods) / ideal_periods
        else:
            wpe = 0.0

        wpe_values.append(wpe)

    wpe_avg = sum(wpe_values) / num_agents
    computation_time = time.time() - start_time

    return wpe_values, wpe_avg, computation_time

def compute_rp_metrics(num_episodes, num_agents, top_agents_per_episode, alpha=1.0, beta=1.0):
    """Compute RP metrics (AWE, WPE, RP)."""
    # Start timing for TOTAL RP_avg computation (includes AWE + WPE + overhead)
    rp_total_start = time.time()

    awe_per_agent, awe_avg, awe_time = compute_awe(num_episodes, num_agents, top_agents_per_episode)
    wpe_per_agent, wpe_avg, wpe_time = compute_wpe(num_episodes, num_agents, top_agents_per_episode)

    # Calculate weighted mean (overhead)
    rp_per_agent = []
    for i in range(num_agents):
        rp = (alpha * awe_per_agent[i] + beta * wpe_per_agent[i]) / (alpha + beta)
        rp_per_agent.append(rp)

    rp_avg = sum(rp_per_agent) / num_agents

    # Total RP_avg time = AWE + WPE + overhead
    rp_total_time = time.time() - rp_total_start
    total_time = time.time() - rp_total_start  # Same as rp_total_time

    return {
        'AWE_per_agent': awe_per_agent,
        'WPE_per_agent': wpe_per_agent,
        'RP_per_agent': rp_per_agent,
        'AWE_avg': awe_avg,
        'WPE_avg': wpe_avg,
        'RP_avg': rp_avg,
        'computation_times': {
            'AWE': awe_time,
            'WPE': wpe_time,
            'RP_avg': rp_total_time,  # Now includes AWE + WPE + overhead
            'Total': total_time
        }
    }

#===============================================================================
# Traditional Metrics
#===============================================================================

def compute_efficiency(num_episodes, num_agents, total_reward_per_agent, full_reward):
    """Compute Efficiency metric."""
    start_time = time.time()

    total_reward = sum(total_reward_per_agent)
    max_possible_reward = num_episodes * full_reward
    efficiency = total_reward / max_possible_reward if max_possible_reward > 0 else 0.0

    computation_time = time.time() - start_time

    return {'Efficiency': efficiency, 'computation_time': computation_time}

def compute_reward_fairness(total_reward_per_agent):
    """Compute Reward Fairness (min/max of rewards)."""
    start_time = time.time()

    positive_rewards = [r for r in total_reward_per_agent if r > 0]

    if not positive_rewards:
        reward_fairness = 0.0
    else:
        max_reward = max(positive_rewards)
        min_reward = min(positive_rewards)
        reward_fairness = min_reward / max_reward if max_reward > 0 else 0.0

    computation_time = time.time() - start_time

    return {'Reward_Fairness': reward_fairness, 'computation_time': computation_time}

def compute_tt_fairness(top_agents_per_episode, num_agents, num_episodes):
    """Compute Turn-Taking Fairness (min/max of reaches to terminal including collisions)."""
    start_time = time.time()

    # Count total reaches to terminal for each agent (including collisions)
    reaches_per_agent = [0] * num_agents
    for ep in range(num_episodes):
        for i in range(num_agents):
            if top_agents_per_episode[ep][i] == 1:
                reaches_per_agent[i] += 1

    positive_reaches = [r for r in reaches_per_agent if r > 0]

    if not positive_reaches:
        tt_fairness = 0.0
    else:
        max_reaches = max(positive_reaches)
        min_reaches = min(positive_reaches)
        tt_fairness = min_reaches / max_reaches if max_reaches > 0 else 0.0

    computation_time = time.time() - start_time

    return {'TT_Fairness': tt_fairness, 'computation_time': computation_time}

def compute_fairness(unique_winners_per_episode, num_agents):
    """Compute Fairness (min/max of exclusive wins without collisions)."""
    start_time = time.time()

    if not unique_winners_per_episode:
        return {'Fairness': 0.0, 'computation_time': time.time() - start_time}

    # Count exclusive wins per agent
    exclusive_wins = [unique_winners_per_episode.count(i) for i in range(num_agents)]

    agents_with_wins = [wins for wins in exclusive_wins if wins > 0]

    if not agents_with_wins:
        fairness = 0.0
    else:
        min_wins = min(agents_with_wins)
        max_wins = max(agents_with_wins)
        fairness = min_wins / max_wins if max_wins > 0 else 0.0

    computation_time = time.time() - start_time

    return {'Fairness': fairness, 'computation_time': computation_time}

#===============================================================================
# ALT Ratio Implementation
#===============================================================================

def calculate_alt_ratio(result):
    """
    Calculate AltRatio estimations using benchmarking equations for all ALT metrics.

    Updated naming (Dec 29, 2025):
    - EFALT → qFALT (quadratic FALT)
    - EEALT → qEALT (quadratic EALT)
    """
    n = result['num_agents']
    metrics = result['metrics']

    # Using exact regression equations from benchmark code
    alt_ratios = {
        'FALT': metrics['FALT'],  # Linear: FALT = AltRatio
        'EALT': metrics['EALT'],  # Linear: EALT = AltRatio
        'qFALT': np.sqrt(max(0, metrics['qFALT'] - 5.32183463e-10)),  # Power function (renamed from EFALT)
        'qEALT': np.sqrt(max(0, metrics['qEALT'] - 5.32183463e-10)),  # Power function (renamed from EEALT)
        'CALT': np.sqrt(max(0, metrics['CALT'] - 1.8790581e-10))      # Power function
    }

    # AALT has a piecewise definition
    if metrics['AALT'] > 0:
        if metrics['AALT'] >= 0.5:  # AALT = 2×AltRatio - 1 when AltRatio > 0.75
            alt_ratios['AALT'] = (metrics['AALT'] + 1) / 2
        else:  # Linear interpolation when 0 < AALT < 0.5
            alt_ratios['AALT'] = 0.5 + (0.25 * metrics['AALT'] / 0.5)
    else:
        alt_ratios['AALT'] = 0  # AALT is 0 when AltRatio ≤ 0.5

    # Calculate estimated alternating agents
    estimated_agents = {k: n * v for k, v in alt_ratios.items()}

    # Calculate percentages
    percentages = {k: v * 100 for k, v in alt_ratios.items()}

    return alt_ratios, estimated_agents, percentages

def calculate_coordination_score(observed, random, perfect=1.0):
    """
    Calculate normalized coordination score: (observed-random)/(perfect-random).

    Returns:
        > 0: Coordination above random (positive coordination)
        = 0: Same as random (no coordination)
        < 0: Worse than random (anti-coordination)
    """
    if perfect == random:  # Avoid division by zero
        return 0 if observed == random else 1

    score = (observed - random) / (perfect - random)
    # Allow negative values to show anti-coordination
    return score
