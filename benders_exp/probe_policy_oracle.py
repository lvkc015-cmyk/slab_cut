from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from joblib import load

from .benders import BendersCut
from .network import add_instance_generator_args, generate_network_design_instance, instance_generator_kwargs_from_args
from .network_benders_scip import NetworkScipBendersSolver, route1_all_open_feasibility_check
from .policy_actions import default_continuation_selector, learned_candidate_selection, probe_score_fields
from .strategies import SelectionState, SlabLikeStrategy, dot
from .teacher import evaluate_prefix_rollouts, rollout_return


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe candidate-level policy oracle and learned model decisions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-nodes", type=int, default=9)
    parser.add_argument("--n-edges", type=int, default=22)
    parser.add_argument("--n-scenarios", type=int, default=8)
    parser.add_argument("--n-commodities", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--rollout-horizon", type=int, default=4)
    parser.add_argument("--rollout-gamma", type=float, default=0.92)
    parser.add_argument("--cut-penalty", type=float, default=0.002)
    parser.add_argument("--time-penalty-weight", type=float, default=0.005)
    parser.add_argument("--master-time-penalty-weight", type=float, default=0.0)
    parser.add_argument("--peak-master-time-penalty-weight", type=float, default=0.0)
    parser.add_argument("--pressure-penalty-weight", type=float, default=0.0)
    parser.add_argument("--near-binding-penalty-weight", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scip-time-limit", type=float, default=0.0)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs") / "policy_probe")
    add_instance_generator_args(parser)
    args = parser.parse_args()

    payload = load(args.model_path) if args.model_path is not None else None
    args.out.mkdir(parents=True, exist_ok=True)
    instance_kwargs = instance_generator_kwargs_from_args(args)
    instance = generate_network_design_instance(
        n_nodes=args.n_nodes,
        n_edges=args.n_edges,
        n_scenarios=args.n_scenarios,
        n_commodities=args.n_commodities,
        seed=args.seed,
        **instance_kwargs,
    )
    precheck_ok, precheck_reason = route1_all_open_feasibility_check(
        instance,
        time_limit=args.scip_time_limit if args.scip_time_limit > 0 else None,
        threads=args.threads,
    )
    if not precheck_ok:
        raise RuntimeError(
            "route-1 probe requires an instance with a feasible all-open design; "
            + precheck_reason
        )
    solver = NetworkScipBendersSolver(
        instance=instance,
        strategy=SlabLikeStrategy(),
        max_iters=args.max_iters,
        tol=args.tol,
        log_dir=None,
        scip_time_limit=args.scip_time_limit if args.scip_time_limit > 0 else None,
        threads=args.threads,
    )

    upper_bound = math.inf
    last_lb = -math.inf
    stagnation = 0
    rows: list[dict[str, object]] = []

    for it in range(1, args.max_iters + 1):
        master = solver.solve_master()
        lower_bound = master.objective
        stagnation = stagnation + 1 if lower_bound <= last_lb + 1.0e-7 else 0
        last_lb = max(last_lb, lower_bound)

        feasible, expected_recourse, candidates, penalty_stats, _ = solver.solve_subproblems(master)
        if not feasible:
            continue

        fixed = dot(instance.fixed_costs, [float(v) for v in master.x])
        upper_bound = min(upper_bound, fixed + expected_recourse)
        state = SelectionState(
            iteration=it,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            cut_count=len(solver.optimality_cuts),
            n_facilities=instance.n_edges,
            n_scenarios=instance.n_scenarios,
            stagnation=stagnation,
            fixed_costs=instance.fixed_costs,
            active_cuts_by_scenario=solver.active_cuts_by_scenario(),
            instance=instance,
            x=list(master.x),
            max_iters=args.max_iters,
            master_solution=master,
            polyhedral_state=solver.last_master_polyhedral_state,
            disconnected_scenario_ratio=penalty_stats.disconnected_scenario_ratio,
            penalty_scenario_ratio=penalty_stats.penalty_scenario_ratio,
            unmet_demand_ratio=penalty_stats.unmet_demand_ratio,
            overflow_ratio=penalty_stats.overflow_ratio,
            penalty_cost_ratio=penalty_stats.penalty_cost_ratio,
        )

        viable = [cand for cand in candidates if cand.violation > 1.0e-7]
        baseline_rollout = rollout_return(
            solver,
            [],
            lower_bound,
            horizon=args.rollout_horizon,
            gamma=args.rollout_gamma,
            cut_penalty=args.cut_penalty,
            time_penalty_weight=args.time_penalty_weight,
            master_time_penalty_weight=args.master_time_penalty_weight,
            peak_master_time_penalty_weight=args.peak_master_time_penalty_weight,
            pressure_penalty_weight=args.pressure_penalty_weight,
            near_binding_penalty_weight=args.near_binding_penalty_weight,
            continuation_selector=default_continuation_selector,
            allow_empty=True,
        )
        oracle_scores: list[tuple[int, float]] = []
        oracle_rollouts: list = []
        for idx, cand in enumerate(viable):
            result = rollout_return(
                solver,
                [cand],
                lower_bound,
                horizon=args.rollout_horizon,
                gamma=args.rollout_gamma,
                cut_penalty=args.cut_penalty,
                time_penalty_weight=args.time_penalty_weight,
                master_time_penalty_weight=args.master_time_penalty_weight,
                peak_master_time_penalty_weight=args.peak_master_time_penalty_weight,
                pressure_penalty_weight=args.pressure_penalty_weight,
                near_binding_penalty_weight=args.near_binding_penalty_weight,
                continuation_selector=default_continuation_selector,
            )
            oracle_scores.append((idx, result.total_return - baseline_rollout.total_return))
            oracle_rollouts.append((idx, result))
        oracle_scores.sort(key=lambda item: item[1], reverse=True)
        ranked_candidates = [viable[idx] for idx, _score in oracle_scores]
        best_prefix_size, prefix_rollouts = evaluate_prefix_rollouts(
            solver,
            ranked_candidates,
            lower_bound,
            horizon=args.rollout_horizon,
            gamma=args.rollout_gamma,
            cut_penalty=args.cut_penalty,
            time_penalty_weight=args.time_penalty_weight,
            master_time_penalty_weight=args.master_time_penalty_weight,
            peak_master_time_penalty_weight=args.peak_master_time_penalty_weight,
            pressure_penalty_weight=args.pressure_penalty_weight,
            near_binding_penalty_weight=args.near_binding_penalty_weight,
            continuation_selector=default_continuation_selector,
        )
        oracle_selected = ranked_candidates[:best_prefix_size]
        best_rollout = None
        if oracle_rollouts:
            best_rollout = max(oracle_rollouts, key=lambda item: item[1].total_return)[1]

        model_selected = []
        ranked_scores = []
        diagnostics = None
        if payload is not None:
            model_selected, ranked_scores, diagnostics = learned_candidate_selection(payload, candidates, state)

        row = {
            "iteration": it,
            "rel_gap": state.rel_gap,
            "pressure": state.pressure,
            "candidate_count": len(viable),
            "oracle_selected_count": len(oracle_selected),
            "oracle_best_prefix_size": best_prefix_size,
            "model_selected_count": len(model_selected),
        }
        if best_rollout is not None:
            row.update(
                {
                    "oracle_best_total_return": best_rollout.total_return,
                    "oracle_best_gain": best_rollout.cumulative_gain,
                    "oracle_best_elapsed": best_rollout.elapsed,
                    "oracle_best_master_time": best_rollout.cumulative_master_time,
                    "oracle_best_subproblem_time": best_rollout.cumulative_subproblem_time,
                    "oracle_best_total_selected": best_rollout.cumulative_selection_count,
                    "oracle_best_penalty_scenario_ratio": best_rollout.cumulative_penalty_scenario_ratio,
                    "oracle_best_unmet_demand_ratio": best_rollout.cumulative_unmet_demand_ratio,
                    "oracle_best_overflow_ratio": best_rollout.cumulative_overflow_ratio,
                    "oracle_best_penalty_cost_ratio": best_rollout.cumulative_penalty_cost_ratio,
                    "oracle_best_max_master_time": best_rollout.max_master_time,
                    "oracle_best_final_pressure": best_rollout.final_pressure,
                    "oracle_best_final_active_cut_ratio": best_rollout.final_active_cut_ratio,
                    "oracle_best_final_near_binding_cut_ratio": best_rollout.final_near_binding_cut_ratio,
                }
            )
        if diagnostics is not None:
            row.update(
                {
                    "model_hard_cap": diagnostics.hard_cap,
                    "model_predicted_quota": diagnostics.predicted_quota,
                    "model_quota_score": diagnostics.quota_score,
                    "model_above_threshold_count": diagnostics.above_threshold_count,
                    "model_unique_scored_count": diagnostics.unique_scored_count,
                    "model_diversity_reject_count": diagnostics.diversity_reject_count,
                    "model_loop_iterations": diagnostics.loop_iterations,
                    "model_stop_reason": diagnostics.stop_reason,
                    "model_threshold": diagnostics.threshold,
                    "model_diversity_threshold": diagnostics.diversity_threshold,
                    "model_max_budget_ratio": diagnostics.max_budget_ratio,
                }
            )
        for rank, (_idx, score) in enumerate(oracle_scores[:64]):
            row[f"cand_score_rank_{rank}"] = score
            row[f"cand_selected_rank_{rank}"] = 1 if rank < len(oracle_selected) else 0
        rows.append(row)

        for cand in solver.strategy.select(candidates, state):
            solver.optimality_cuts.append(
                BendersCut(
                    scenario=cand.scenario,
                    const=cand.const,
                    coeffs=cand.coeffs,
                    cut_type=cand.cut_type,
                )
            )

    csv_path = args.out / "probe_iterations.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "iteration",
                "rel_gap",
                "pressure",
                "candidate_count",
                "oracle_selected_count",
                "oracle_best_prefix_size",
                "model_selected_count",
                "oracle_best_total_return",
                "oracle_best_gain",
                "oracle_best_elapsed",
                "oracle_best_master_time",
                "oracle_best_subproblem_time",
                "oracle_best_total_selected",
                "oracle_best_penalty_scenario_ratio",
                "oracle_best_unmet_demand_ratio",
                "oracle_best_overflow_ratio",
                "oracle_best_penalty_cost_ratio",
                "oracle_best_max_master_time",
                "oracle_best_final_pressure",
                "oracle_best_final_active_cut_ratio",
                "oracle_best_final_near_binding_cut_ratio",
                "model_hard_cap",
                "model_predicted_quota",
                "model_quota_score",
                "model_above_threshold_count",
                "model_unique_scored_count",
                "model_diversity_reject_count",
                "model_loop_iterations",
                "model_stop_reason",
                "model_threshold",
                "model_diversity_threshold",
                "model_max_budget_ratio",
                *probe_score_fields(),
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"probe_csv={csv_path}")


if __name__ == "__main__":
    main()
