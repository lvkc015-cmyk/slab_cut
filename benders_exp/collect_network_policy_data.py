from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .benders import BendersCut
from .features import POLICY_FEATURE_NAMES, QUOTA_FEATURE_NAMES, candidate_feature_dict, state_feature_dict
from .network import (
    add_instance_generator_args,
    generate_network_design_instance,
    instance_generator_kwargs_from_args,
)
from .network_benders_scip import NetworkScipBendersSolver, route1_all_open_feasibility_check
from .policy_actions import default_continuation_selector
from .policy_stopping import STOPPING_FEATURE_NAMES, build_stopping_feature_dict
from .strategies import SelectionState, SlabLikeStrategy, dot
from .teacher import evaluate_prefix_rollouts, rollout_return


def collect_for_seed(
    seed: int,
    writer: csv.DictWriter,
    *,
    n_nodes: int,
    n_edges: int,
    n_scenarios: int,
    n_commodities: int,
    max_iters: int,
    tol: float,
    rollout_horizon: int,
    rollout_gamma: float,
    cut_penalty: float,
    time_penalty_weight: float,
    master_time_penalty_weight: float,
    peak_master_time_penalty_weight: float,
    pressure_penalty_weight: float,
    near_binding_penalty_weight: float,
    scip_time_limit: float | None,
    threads: int,
    instance_kwargs: dict[str, float | int | str],
) -> dict[str, int]: 

    
    instance = generate_network_design_instance(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_scenarios=n_scenarios,
        n_commodities=n_commodities,
        seed=seed,
        **instance_kwargs,
    )

    rollout_policy = SlabLikeStrategy()
    
    solver = NetworkScipBendersSolver(
        instance=instance,
        strategy=rollout_policy,
        max_iters=max_iters,
        tol=tol,
        log_dir=None,
        scip_time_limit=scip_time_limit,
        threads=threads,
    )

    upper_bound = math.inf
    last_lb = -math.inf
    stagnation = 0
    rows_written = 0
    infeasible_iters = 0
    feasible_iters = 0
    feasible_no_candidates = 0
    feasible_no_viable = 0
    skipped_infeasible_instance = 0
    skip_reason = ""

    precheck_ok, precheck_reason = route1_all_open_feasibility_check(
        instance,
        time_limit=scip_time_limit,
        threads=threads,
    )
    if not precheck_ok:
        skipped_infeasible_instance = 1
        skip_reason = precheck_reason
        return {
            "rows_written": rows_written,
            "infeasible_iters": infeasible_iters,
            "feasible_iters": feasible_iters,
            "feasible_no_candidates": feasible_no_candidates,
            "feasible_no_viable": feasible_no_viable,
            "skipped_infeasible_instance": skipped_infeasible_instance,
            "skip_reason": skip_reason,
        }

    for it in range(1, max_iters + 1):
        
        master = solver.solve_master()
        lower_bound = master.objective
        
        stagnation = stagnation + 1 if lower_bound <= last_lb + 1.0e-7 else 0
       
        last_lb = max(last_lb, lower_bound)

       
        feasible, expected_recourse, candidates, penalty_stats, _ = solver.solve_subproblems(master)
        if not feasible:
            infeasible_iters += 1
            continue
        feasible_iters += 1

       
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
            max_iters=max_iters, 
            master_solution=master, 
            polyhedral_state=solver.last_master_polyhedral_state, 
            disconnected_scenario_ratio=penalty_stats.disconnected_scenario_ratio,
            penalty_scenario_ratio=penalty_stats.penalty_scenario_ratio,
            unmet_demand_ratio=penalty_stats.unmet_demand_ratio,
            overflow_ratio=penalty_stats.overflow_ratio,
            penalty_cost_ratio=penalty_stats.penalty_cost_ratio,
        )

        
        viable = [cand for cand in candidates if cand.violation > 1.0e-7]
        if not candidates:
            feasible_no_candidates += 1
            continue
        if not viable:
            feasible_no_viable += 1
            continue
        
        query_id = f"seed{seed}_n{n_nodes}_e{n_edges}_s{n_scenarios}_k{n_commodities}_it{it}"
        baseline_rollout = rollout_return(
            solver,
            [],
            lower_bound,
            horizon=rollout_horizon,
            gamma=rollout_gamma,
            cut_penalty=cut_penalty,
            time_penalty_weight=time_penalty_weight,
            master_time_penalty_weight=master_time_penalty_weight,
            peak_master_time_penalty_weight=peak_master_time_penalty_weight,
            pressure_penalty_weight=pressure_penalty_weight,
            near_binding_penalty_weight=near_binding_penalty_weight,
            continuation_selector=default_continuation_selector,
            allow_empty=True,
        )
        
        rollout_rows: list[tuple[object, object]] = []
        
        for cand in viable:
            result = rollout_return(
                solver, 
                [cand], 
                lower_bound, 
                horizon=rollout_horizon, 
                gamma=rollout_gamma, 
                cut_penalty=cut_penalty, 
                time_penalty_weight=time_penalty_weight,
                master_time_penalty_weight=master_time_penalty_weight, 
                peak_master_time_penalty_weight=peak_master_time_penalty_weight, 
                pressure_penalty_weight=pressure_penalty_weight, 
                near_binding_penalty_weight=near_binding_penalty_weight, 
                continuation_selector=default_continuation_selector,
            )
            rollout_rows.append((cand, result))

        ranked_rollouts = sorted(
            rollout_rows,
            key=lambda item: item[1].total_return - baseline_rollout.total_return,
            reverse=True,
        )
        best_prefix_size, prefix_rollouts = evaluate_prefix_rollouts(
            solver,
            [cand for cand, _result in ranked_rollouts],
            lower_bound,
            horizon=rollout_horizon,
            gamma=rollout_gamma,
            cut_penalty=cut_penalty,
            time_penalty_weight=time_penalty_weight,
            master_time_penalty_weight=master_time_penalty_weight,
            peak_master_time_penalty_weight=peak_master_time_penalty_weight,
            pressure_penalty_weight=pressure_penalty_weight,
            near_binding_penalty_weight=near_binding_penalty_weight,
            continuation_selector=default_continuation_selector,
        )
        prefix_score_map = {prefix_size: rollout.total_return for prefix_size, rollout in prefix_rollouts}
        selected_count = best_prefix_size
        
        rank_by_id = {id(cand): rank for rank, (cand, _result) in enumerate(ranked_rollouts, start=1)}

        
        quota_cache = state_feature_dict(candidates, state, instance, master.x, max_iters=max_iters)
        
        quota_features = [quota_cache[name] for name in QUOTA_FEATURE_NAMES]
        
        state_feature_row = {f"quota_{name}": value for name, value in zip(QUOTA_FEATURE_NAMES, quota_features)}
        # 
        cache = quota_cache
       
        ranked_feature_rows: list[tuple[object, object, dict[str, float]]] = []
        
        for cand, result in ranked_rollouts:
            feat_dict = candidate_feature_dict(
                cand, 
                candidates, 
                state, 
                instance, 
                master.x, 
                [],
                max_iters=max_iters,
                cached_state_features=cache, # 之前计算并缓存的状态特征，可能用于加速特征提取过程，避免重复计算一些与状态相关的特征
            )
            ranked_feature_rows.append((cand, result, feat_dict))

        
        ranking_scores = [float(result.total_return - baseline_rollout.total_return) for _cand, result, _feat in ranked_feature_rows]
        processed_scores: list[float] = []
        kept = 0
        stop_rank = selected_count + 1
        
        for rank_idx, (cand, result, feat_dict) in enumerate(ranked_feature_rows, start=1):
            
            current_score = float(result.total_return - baseline_rollout.total_return)
            
            next_score = ranking_scores[rank_idx] if rank_idx < len(ranking_scores) else 0.0
            
            stop_features = build_stopping_feature_dict(
                state_features=state_feature_row, 
                candidate_features=feat_dict, 
                rank_index=rank_idx, 
                total_candidates=len(viable), 
                selected_count=kept, 
                processed_scores=processed_scores, 
                current_score=current_score,
                next_score=next_score,
                top_score=ranking_scores[0] if ranking_scores else 0.0, 
            )
            
            row = {
                "seed": seed,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_scenarios": n_scenarios,
                "n_commodities": n_commodities,
                "query_id": query_id,
                "oracle_rank": rank_by_id[id(cand)],
                "oracle_selected": 1 if rank_idx <= selected_count else 0,
                "oracle_selected_count": selected_count,
                "candidate_count": len(viable),
                "target_return": current_score,
                "prefix_best_score": prefix_score_map.get(best_prefix_size, 0.0),
                "prefix_current_score": prefix_score_map.get(rank_idx, 0.0),
                "target_gain": result.cumulative_gain,
                "rollout_elapsed": result.elapsed,
                "rollout_master_time": result.cumulative_master_time,
                "rollout_subproblem_time": result.cumulative_subproblem_time,
                "rollout_total_selected": result.cumulative_selection_count,
                "rollout_penalty_scenario_ratio": result.cumulative_penalty_scenario_ratio,
                "rollout_unmet_demand_ratio": result.cumulative_unmet_demand_ratio,
                "rollout_overflow_ratio": result.cumulative_overflow_ratio,
                "rollout_penalty_cost_ratio": result.cumulative_penalty_cost_ratio,
                "rollout_max_master_time": result.max_master_time,
                "rollout_final_pressure": result.final_pressure,
                "rollout_final_active_cut_ratio": result.final_active_cut_ratio,
                "rollout_final_near_binding_cut_ratio": result.final_near_binding_cut_ratio,
                "use_for_stopping": 1 if rank_idx <= stop_rank else 0,
                "stop_target": 1 if rank_idx == stop_rank else 0,
            }
           
            for name, value in zip(QUOTA_FEATURE_NAMES, quota_features):
                row[f"quota_{name}"] = value
           
            for name in POLICY_FEATURE_NAMES:
                row[name] = feat_dict[name]
            
            for name in STOPPING_FEATURE_NAMES:
                if name not in row:
                    row[name] = stop_features[name]
            writer.writerow(row)
            rows_written += 1
            processed_scores.append(current_score)
            if rank_idx < stop_rank:
                kept += 1
        
        rollout_selected = rollout_policy.select(candidates, state)
        # 
        for cand in rollout_selected:
            solver.optimality_cuts.append(
                BendersCut(
                    scenario=cand.scenario, 
                    const=cand.const,
                    coeffs=cand.coeffs,
                    cut_type=cand.cut_type,
                )
            )

        # 
        gap = max(0.0, upper_bound - lower_bound)
        if gap <= tol or gap / (abs(upper_bound) + 1.0) <= tol:
            break

    return {
        "rows_written": rows_written,
        "infeasible_iters": infeasible_iters,
        "feasible_iters": feasible_iters,
        "feasible_no_candidates": feasible_no_candidates,
        "feasible_no_viable": feasible_no_viable,
        "skipped_infeasible_instance": skipped_infeasible_instance,
        "skip_reason": skip_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect candidate-level policy data for network cut control.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--sizes", nargs="+", default=["8,16,6,4", "9,22,8,5", "10,26,10,6"])
    parser.add_argument("--max-iters", type=int, default=120)
    # tol 参数定义了算法的收敛容差，通常用于判断当前的解是否已经足够接近最优解
    parser.add_argument("--tol", type=float, default=1.0e-5)

    # 这些参数是否合理？
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
    parser.add_argument("--out", type=Path, default=Path("data") / "network_policy_training.csv")
    add_instance_generator_args(parser)
    return parser


def parse_sizes(tokens: list[str]) -> list[tuple[int, int, int, int]]:
    if all("," not in token for token in tokens):
        if len(tokens) % 4 != 0:
            raise ValueError("size arguments must be comma quadruples or groups of four integers")
        values = [int(token) for token in tokens]
        return [
            (values[i], values[i + 1], values[i + 2], values[i + 3])
            for i in range(0, len(values), 4)
        ]
    parsed = []
    for token in tokens:
        parts = token.split(",")
        if len(parts) != 4:
            raise ValueError(f"invalid size quadruple: {token}")
        parsed.append(tuple(int(part) for part in parts))
    return parsed


def main() -> None:
    args = build_parser().parse_args()
    sizes = parse_sizes(args.sizes)
    instance_kwargs = instance_generator_kwargs_from_args(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 定义 CSV 文件的列名，这些列名包括了实验的各种参数、状态特征、候选割特征、rollout 结果以及停止决策的特征等信息。
    fieldnames = [
        "seed",
        "n_nodes", #网络设计问题中节点的数量，这个参数定义了问题的规模和复杂度，通常节点越多，问题越难求解
        "n_edges", 
        "n_scenarios",
        "n_commodities", # 网络设计问题中商品的数量，这个参数定义了问题的规模和复杂度，通常商品越多，问题越难求解
        "query_id", # 一个唯一的标识符，用于标识每一个实验查询，通常包含了随机种子、问题规模参数和迭代次数等信息，便于后续分析和追踪
        # 监督标签
        "oracle_rank", # 当前候选割在根据 rollout 结果进行排序后的排名，排名越靠前表示这个割在 rollout 中表现越好，这个特征有助于模型学习如何根据候选割的特征来预测它在未来迭代中的表现
        # 监督标签       
        "oracle_selected", # 一个二元标签，表示当前候选割是否被认为是“有用”的割，通常根据它在 rollout 中的总收益是否大于0来判断，这个特征有助于模型学习如何根据候选割的特征来预测它是否应该被选择
        "oracle_selected_count",
        "candidate_count",
        # 监督标签
        "target_return",
        "prefix_best_score",
        "prefix_current_score",
        "target_gain",
        "rollout_elapsed",
        "rollout_master_time",
        "rollout_subproblem_time",
        "rollout_total_selected",
        "rollout_penalty_scenario_ratio",
        "rollout_unmet_demand_ratio",
        "rollout_overflow_ratio",
        "rollout_penalty_cost_ratio",
        "rollout_max_master_time",
        "rollout_final_pressure",
        "rollout_final_active_cut_ratio",
        "rollout_final_near_binding_cut_ratio",
        # 监督标签
        "use_for_stopping",
        "stop_target",
        *STOPPING_FEATURE_NAMES,
    ]
    # 定义 SCIP 的时间限制，如果设置为 0 或负数，则不设置时间限制
    scip_time_limit = args.scip_time_limit if args.scip_time_limit > 0 else None

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        total_rows_written = 0
        diagnostics: list[tuple[int, int, int, int, int, dict[str, int]]] = []
        # 对于每一组问题规模参数（节点数、边数、场景数、商品数）和每一个随机种子，调用 collect_for_seed 函数来收集数据，并把收集到的数据写入 CSV 文件中。这个过程会遍历所有指定的规模参数组合和随机种子，系统地生成大量的训练数据，用于后续的机器学习模型训练。
        for n_nodes, n_edges, n_scenarios, n_commodities in sizes:
            for seed in args.seeds:
                stats = collect_for_seed(
                    seed,
                    writer,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    n_scenarios=n_scenarios,
                    n_commodities=n_commodities,
                    max_iters=args.max_iters,
                    tol=args.tol,
                    rollout_horizon=args.rollout_horizon,
                    rollout_gamma=args.rollout_gamma,
                    cut_penalty=args.cut_penalty,
                    time_penalty_weight=args.time_penalty_weight,
                    master_time_penalty_weight=args.master_time_penalty_weight,
                    peak_master_time_penalty_weight=args.peak_master_time_penalty_weight,
                    pressure_penalty_weight=args.pressure_penalty_weight,
                    near_binding_penalty_weight=args.near_binding_penalty_weight,
                    scip_time_limit=scip_time_limit,
                    threads=args.threads,
                    instance_kwargs=instance_kwargs,
                )
                total_rows_written += stats["rows_written"]
                diagnostics.append((seed, n_nodes, n_edges, n_scenarios, n_commodities, stats))
                print(
                    "collected "
                    f"seed={seed} size={n_nodes},{n_edges},{n_scenarios},{n_commodities} "
                    f"rows={stats['rows_written']} infeasible_iters={stats['infeasible_iters']} "
                    f"feasible_iters={stats['feasible_iters']} "
                    f"feasible_no_candidates={stats['feasible_no_candidates']} "
                    f"feasible_no_viable={stats['feasible_no_viable']} "
                    f"skipped_infeasible_instance={stats['skipped_infeasible_instance']}"
                )
                if stats["skipped_infeasible_instance"]:
                    print(f"  skip_reason={stats['skip_reason']}")
    if total_rows_written == 0:
        details = "; ".join(
            (
                f"seed={seed} size={n_nodes},{n_edges},{n_scenarios},{n_commodities}: "
                f"infeasible_iters={stats['infeasible_iters']}, "
                f"feasible_iters={stats['feasible_iters']}, "
                f"feasible_no_candidates={stats['feasible_no_candidates']}, "
                f"feasible_no_viable={stats['feasible_no_viable']}, "
                f"skipped_infeasible_instance={stats['skipped_infeasible_instance']}, "
                f"skip_reason={stats['skip_reason']!r}"
            )
            for seed, n_nodes, n_edges, n_scenarios, n_commodities, stats in diagnostics
        )
        raise RuntimeError(
            "no training rows were collected under feasible-only policy; " + details
        )
    print(f"dataset={args.out}")


if __name__ == "__main__":
    main()
