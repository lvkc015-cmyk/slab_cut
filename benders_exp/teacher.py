from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

from .benders import BendersCut
from .network_benders_scip import NetworkScipBendersSolver
from .strategies import SelectionState

TIME_SCALE_FLOOR = 5.0e-2
MASTER_TIME_SCALE_FLOOR = 5.0e-2
GAIN_SQUASH_THRESHOLD = 2.5e-1


@dataclass(frozen=True)
class RolloutScore:
    total_return: float
    cumulative_gain: float
    elapsed: float
    cumulative_master_time: float
    cumulative_subproblem_time: float
    cumulative_selection_count: int
    cumulative_penalty_scenario_ratio: float
    cumulative_unmet_demand_ratio: float
    cumulative_overflow_ratio: float
    cumulative_penalty_cost_ratio: float
    max_master_time: float
    final_pressure: float
    final_active_cut_ratio: float
    final_near_binding_cut_ratio: float


def _stable_log_ratio(value: float, scale: float) -> float:
    return math.log1p(max(0.0, value) / max(scale, 1.0e-6))


def _squash_gain_ratio(ratio: float, threshold: float = GAIN_SQUASH_THRESHOLD) -> float:
    capped_ratio = max(0.0, ratio)
    if capped_ratio <= threshold:
        return capped_ratio
    return threshold + threshold * math.log(capped_ratio / threshold)


def _baseline_scales(
    solver: NetworkScipBendersSolver,
    base_lower_bound: float,
) -> tuple[float, float, float]:
    gain_scale = max(1.0, abs(base_lower_bound))
    base_master_time = max(MASTER_TIME_SCALE_FLOOR, float(getattr(solver, "last_master_solve_time", 0.0)))
    base_subproblem_time = max(0.0, float(getattr(solver, "last_subproblem_solve_time", 0.0)))
    total_time_scale = max(TIME_SCALE_FLOOR, base_master_time + base_subproblem_time)
    return gain_scale, base_master_time, total_time_scale


def _rollout_core(
    trial: NetworkScipBendersSolver,
    *,
    base_lower_bound: float,
    initial_selection_count: int,
    horizon: int,
    gamma: float,
    cut_penalty: float,
    time_penalty_weight: float,
    master_time_penalty_weight: float,
    peak_master_time_penalty_weight: float,
    pressure_penalty_weight: float,
    near_binding_penalty_weight: float,
    continuation_selector,
) -> RolloutScore:
    start = time.perf_counter()
    gain_scale, master_time_scale, total_time_scale = _baseline_scales(trial, base_lower_bound)

    cumulative = 0.0
    cumulative_master_time = 0.0
    cumulative_subproblem_time = 0.0
    prev_lb = base_lower_bound
    total_selection = initial_selection_count
    initial_selection_penalty = max(0, initial_selection_count - 1)
    discounted_continuation_penalty = 0.0
    discounted_time_ratio = 0.0
    discounted_master_time_ratio = 0.0
    discounted_peak_master_time_ratio = 0.0
    discounted_pressure = 0.0
    discounted_near_binding = 0.0
    cumulative_penalty_scenario_ratio = 0.0
    cumulative_unmet_demand_ratio = 0.0
    cumulative_overflow_ratio = 0.0
    cumulative_penalty_cost_ratio = 0.0
    max_master_time = 0.0
    best_upper_bound = math.inf
    final_pressure = 0.0
    final_active_cut_ratio = 0.0
    final_near_binding_cut_ratio = 0.0

    for step in range(horizon):
        discount = gamma**step
        master = trial.solve_master()
        feasible, expected_recourse, candidates, penalty_stats, sub_time = trial.solve_subproblems(master)

        step_master_time = float(master.solve_time)
        step_subproblem_time = float(sub_time)
        step_elapsed = step_master_time + step_subproblem_time

        cumulative_master_time += step_master_time
        cumulative_subproblem_time += step_subproblem_time
        max_master_time = max(max_master_time, step_master_time)

        discounted_time_ratio += discount * _stable_log_ratio(step_elapsed, total_time_scale)
        step_master_metric = _stable_log_ratio(step_master_time, master_time_scale)
        discounted_master_time_ratio += discount * step_master_metric
        discounted_peak_master_time_ratio = max(
            discounted_peak_master_time_ratio,
            discount * step_master_metric,
        )

        cumulative_penalty_scenario_ratio += discount * penalty_stats.penalty_scenario_ratio
        cumulative_unmet_demand_ratio += discount * penalty_stats.unmet_demand_ratio
        cumulative_overflow_ratio += discount * penalty_stats.overflow_ratio
        cumulative_penalty_cost_ratio += discount * penalty_stats.penalty_cost_ratio

        lower_bound = master.objective
        gain_ratio = max(0.0, lower_bound - prev_lb) / gain_scale
        cumulative += discount * _squash_gain_ratio(gain_ratio)
        prev_lb = lower_bound

        current_pressure = len(trial.optimality_cuts) / max(1, trial.instance.n_edges * trial.instance.n_scenarios)
        final_pressure = current_pressure
        discounted_pressure += discount * current_pressure

        if trial.last_master_polyhedral_state is not None:
            final_active_cut_ratio = float(trial.last_master_polyhedral_state.active_cut_ratio)
            final_near_binding_cut_ratio = float(trial.last_master_polyhedral_state.near_binding_cut_ratio)
            discounted_near_binding += discount * final_near_binding_cut_ratio

        if not feasible or not candidates:
            continue

        fixed = sum(cost * float(value) for cost, value in zip(trial.instance.fixed_costs, master.x))
        best_upper_bound = min(best_upper_bound, fixed + expected_recourse)
        state = SelectionState(
            iteration=step + 1,
            lower_bound=lower_bound,
            upper_bound=best_upper_bound,
            cut_count=len(trial.optimality_cuts),
            n_facilities=trial.instance.n_edges,
            n_scenarios=trial.instance.n_scenarios,
            stagnation=0,
            fixed_costs=trial.instance.fixed_costs,
            active_cuts_by_scenario=trial.active_cuts_by_scenario(),
            instance=trial.instance,
            x=list(master.x),
            max_iters=trial.max_iters,
            master_solution=master,
            polyhedral_state=trial.last_master_polyhedral_state,
            disconnected_scenario_ratio=penalty_stats.disconnected_scenario_ratio,
            penalty_scenario_ratio=penalty_stats.penalty_scenario_ratio,
            unmet_demand_ratio=penalty_stats.unmet_demand_ratio,
            overflow_ratio=penalty_stats.overflow_ratio,
            penalty_cost_ratio=penalty_stats.penalty_cost_ratio,
        )
        continuation = continuation_selector(candidates, state)
        discounted_continuation_penalty += discount * len(continuation)
        total_selection += len(continuation)
        for cand in continuation:
            trial.optimality_cuts.append(
                BendersCut(
                    scenario=cand.scenario,
                    const=cand.const,
                    coeffs=cand.coeffs,
                    cut_type=cand.cut_type,
                )
            )

    elapsed = time.perf_counter() - start
    score = (
        cumulative
        - cut_penalty * (initial_selection_penalty + discounted_continuation_penalty)
        - time_penalty_weight * discounted_time_ratio
        - master_time_penalty_weight * discounted_master_time_ratio
        - peak_master_time_penalty_weight * discounted_peak_master_time_ratio
        - pressure_penalty_weight * discounted_pressure
        - near_binding_penalty_weight * discounted_near_binding
    )
    return RolloutScore(
        total_return=score,
        cumulative_gain=cumulative,
        elapsed=elapsed,
        cumulative_master_time=cumulative_master_time,
        cumulative_subproblem_time=cumulative_subproblem_time,
        cumulative_selection_count=total_selection,
        cumulative_penalty_scenario_ratio=cumulative_penalty_scenario_ratio,
        cumulative_unmet_demand_ratio=cumulative_unmet_demand_ratio,
        cumulative_overflow_ratio=cumulative_overflow_ratio,
        cumulative_penalty_cost_ratio=cumulative_penalty_cost_ratio,
        max_master_time=max_master_time,
        final_pressure=final_pressure,
        final_active_cut_ratio=final_active_cut_ratio,
        final_near_binding_cut_ratio=final_near_binding_cut_ratio,
    )

# 克隆一个 BendersSolver 实例，并在克隆的实例中添加一些新的割。
# 该函数首先创建一个浅复制的 BendersSolver 实例，然后将原实例中的割列表复制到新实例中，并重置与主问题相关的属性。
# 最后，将传入的新割添加到新实例的割列表中，并返回这个新的 BendersSolver 实例。
def clone_with_added_cuts(solver: NetworkScipBendersSolver, selected: list) -> NetworkScipBendersSolver:
    trial = copy.copy(solver)
    trial.optimality_cuts = list(solver.optimality_cuts)
    trial.feasibility_cuts = list(solver.feasibility_cuts)
    trial.master_model = None
    trial.master_x_vars = []
    trial.master_theta_vars = []
    trial.master_feas_cuts_loaded = 0
    trial.master_cuts_loaded = 0
    trial.master_lp_model = None
    trial.master_lp_x_vars = []
    trial.master_lp_theta_vars = []
    trial.master_lp_feas_cuts_loaded = 0
    trial.master_lp_cuts_loaded = 0
    for cand in selected:
        trial.optimality_cuts.append(
            BendersCut(
                scenario=cand.scenario,
                const=cand.const,
                coeffs=cand.coeffs,
                cut_type=cand.cut_type,
            )
        )
    return trial


def rollout_return(
    solver: NetworkScipBendersSolver,
    initial_selection: list,
    base_lower_bound: float,
    *,
    horizon: int,
    gamma: float,
    cut_penalty: float,
    time_penalty_weight: float,
    master_time_penalty_weight: float,
    peak_master_time_penalty_weight: float,
    pressure_penalty_weight: float,
    near_binding_penalty_weight: float,
    continuation_selector,
    allow_empty: bool = False,
) -> RolloutScore:
    if not initial_selection:
        if allow_empty:
            trial = clone_with_added_cuts(solver, [])
            return _rollout_core(
                trial,
                base_lower_bound=base_lower_bound,
                initial_selection_count=0,
                horizon=horizon,
                gamma=gamma,
                cut_penalty=cut_penalty,
                time_penalty_weight=time_penalty_weight,
                master_time_penalty_weight=master_time_penalty_weight,
                peak_master_time_penalty_weight=peak_master_time_penalty_weight,
                pressure_penalty_weight=pressure_penalty_weight,
                near_binding_penalty_weight=near_binding_penalty_weight,
                continuation_selector=continuation_selector,
            )
        return RolloutScore(
            total_return=-1.0e9,
            cumulative_gain=0.0,
            elapsed=0.0,
            cumulative_master_time=0.0,
            cumulative_subproblem_time=0.0,
            cumulative_selection_count=0,
            cumulative_penalty_scenario_ratio=0.0,
            cumulative_unmet_demand_ratio=0.0,
            cumulative_overflow_ratio=0.0,
            cumulative_penalty_cost_ratio=0.0,
            max_master_time=0.0,
            final_pressure=0.0,
            final_active_cut_ratio=0.0,
            final_near_binding_cut_ratio=0.0,
        )

    trial = clone_with_added_cuts(solver, initial_selection)
    return _rollout_core(
        trial,
        base_lower_bound=base_lower_bound,
        initial_selection_count=len(initial_selection),
        horizon=horizon,
        gamma=gamma,
        cut_penalty=cut_penalty,
        time_penalty_weight=time_penalty_weight,
        master_time_penalty_weight=master_time_penalty_weight,
        peak_master_time_penalty_weight=peak_master_time_penalty_weight,
        pressure_penalty_weight=pressure_penalty_weight,
        near_binding_penalty_weight=near_binding_penalty_weight,
        continuation_selector=continuation_selector,
    )


def evaluate_prefix_rollouts(
    solver: NetworkScipBendersSolver,
    ranked_candidates: list,
    base_lower_bound: float,
    *,
    horizon: int,
    gamma: float,
    cut_penalty: float,
    time_penalty_weight: float,
    master_time_penalty_weight: float,
    peak_master_time_penalty_weight: float,
    pressure_penalty_weight: float,
    near_binding_penalty_weight: float,
    continuation_selector,
) -> tuple[int, list[tuple[int, RolloutScore]]]:
    prefix_rollouts: list[tuple[int, RolloutScore]] = []
    best_prefix_size = 0
    best_score = -math.inf
    for prefix_size in range(0, len(ranked_candidates) + 1):
        rollout = rollout_return(
            solver,
            ranked_candidates[:prefix_size],
            base_lower_bound,
            horizon=horizon,
            gamma=gamma,
            cut_penalty=cut_penalty,
            time_penalty_weight=time_penalty_weight,
            master_time_penalty_weight=master_time_penalty_weight,
            peak_master_time_penalty_weight=peak_master_time_penalty_weight,
            pressure_penalty_weight=pressure_penalty_weight,
            near_binding_penalty_weight=near_binding_penalty_weight,
            continuation_selector=continuation_selector,
            allow_empty=True,
        )
        prefix_rollouts.append((prefix_size, rollout))
        if rollout.total_return > best_score:
            best_score = rollout.total_return
            best_prefix_size = prefix_size
    return best_prefix_size, prefix_rollouts
