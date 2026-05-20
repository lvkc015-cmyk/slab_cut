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
from .network_benders_scip import NetworkScipBendersSolver
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
) -> None: 

    # 生成特定的随机网络设计实例
    instance = generate_network_design_instance(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_scenarios=n_scenarios,
        n_commodities=n_commodities,
        seed=seed,
        **instance_kwargs,
    )
    # 策略
    rollout_policy = SlabLikeStrategy()
    # 求解器
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

    for it in range(1, max_iters + 1):
        # 求解主问题 (Master Problem)，获得当前的下界 (lower_bound) 和网络建设方案 (master.x)
        master = solver.solve_master()
        lower_bound = master.objective
        # 如果当前的下界没有带来实质性的进步（即当前下界 $\le$ 历史最高下界 + 极小容差），则认为进入了停滞状态，停滞计数器加1；否则重置停滞计数器
        stagnation = stagnation + 1 if lower_bound <= last_lb + 1.0e-7 else 0
        # 更新历史最佳记录
        last_lb = max(last_lb, lower_bound)

        # 求解子问题 (Subproblems)，返回可行性状态、期望补偿成本、最优性候选割 (candidates) 和可行性割 (feasibility)
        feasible, expected_recourse, candidates, penalty_stats, _ = solver.solve_subproblems(master)
        if not feasible:
            continue

        # 计算当前主问题给出的网络设计方案的固定成本
        fixed = dot(instance.fixed_costs, [float(v) for v in master.x])
        # 计算当前的上界：主问题的固定成本 + 期望补偿成本；如果这个值比历史上记录的上界更好（更小），就更新上界
        upper_bound = min(upper_bound, fixed + expected_recourse)
        #状态
        state = SelectionState(
            iteration=it, #当前迭代次数
            lower_bound=lower_bound, #当前主问题的下界
            upper_bound=upper_bound, #当前的上界
            cut_count=len(solver.optimality_cuts),#当前主问题中已经添加的最优性割的数量
            n_facilities=instance.n_edges, #网络设计问题中潜在设施的总数（即边的数量）
            n_scenarios=instance.n_scenarios, #随机场景的数量
            stagnation=stagnation, #主问题下界停滞的迭代次数
            fixed_costs=instance.fixed_costs, #网络设计问题中每条边的固定建设成本
            active_cuts_by_scenario=solver.active_cuts_by_scenario(), #当前主问题中针对每个随机场景活跃的割的数量
            instance=instance, #网络设计实例对象，包含了问题的所有数据和参数
            x=list(master.x), #当前主问题给出的网络设计方案（每条边的建设决策，通常是0或1）
            max_iters=max_iters, #算法允许的最大迭代次数
            master_solution=master, #当前主问题的求解结果对象，包含了更多关于求解过程和结果的信息
            polyhedral_state=solver.last_master_polyhedral_state, #当前主问题的多面体状态，包含了当前主问题的割集合和其他相关信息
            disconnected_scenario_ratio=penalty_stats.disconnected_scenario_ratio,
            penalty_scenario_ratio=penalty_stats.penalty_scenario_ratio,
            unmet_demand_ratio=penalty_stats.unmet_demand_ratio,
            overflow_ratio=penalty_stats.overflow_ratio,
            penalty_cost_ratio=penalty_stats.penalty_cost_ratio,
        )

        #从当前的最优性候选割中筛选出那些违反程度（violation）大于一个非常小的阈值（1.0e-7）的割，把真正能起作用的候选割筛选出来。
        viable = [cand for cand in candidates if cand.violation > 1.0e-7]
        # 为当前的实验设置一个唯一的查询ID，格式包含了随机种子、问题规模参数（节点数、边数、场景数、商品数）和当前迭代次数等信息，便于后续分析和追踪
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
        # 对每一个真正有潜力的候选割，使用当前的策略进行一个 rollout（模拟未来的决策过程），计算这个候选割在未来几步迭代中可能带来的收益（return）和其他相关指标。把每个候选割和它对应的 rollout 结果存储在一个列表中，准备后续的排序和特征提取。
        rollout_rows: list[tuple[object, object]] = []
        # 对于每一个真正有潜力的候选割，使用当前的策略进行一个 rollout（模拟未来的决策过程），计算这个候选割在未来几步迭代中可能带来的收益（return）和其他相关指标。把每个候选割和它对应的 rollout 结果存储在一个列表中，准备后续的排序和特征提取。
        for cand in viable:
            result = rollout_return(
                solver, #当前的求解器对象，包含了问题实例、当前的主问题状态、割池等信息
                [cand], #一个包含当前候选割的列表，表示我们要评估这个特定的割
                lower_bound, #当前主问题的下界，作为评估候选割潜在收益的基准
                horizon=rollout_horizon, #rollout的时间范围，即模拟未来多少步迭代
                gamma=rollout_gamma, #rollout中未来奖励的折扣因子，通常在0和1之间，值越小表示越重视短期收益
                cut_penalty=cut_penalty, #在rollout过程中，如果添加了新的割，可能会对求解器的性能产生影响，这个参数用来惩罚过多添加割的行为，鼓励策略选择更有效的割
                time_penalty_weight=time_penalty_weight, #在rollout过程中，求解器的运行时间也是一个重要的考虑因素，这个参数用来惩罚过长的求解时间，鼓励策略选择那些能够更快收敛的割
                master_time_penalty_weight=master_time_penalty_weight, #在rollout过程中，主问题的求解时间可能比子问题更关键，这个参数专门用来惩罚主问题求解时间过长的情况，鼓励策略选择那些能够快速改善主问题下界的割
                peak_master_time_penalty_weight=peak_master_time_penalty_weight, #在rollout过程中，主问题求解时间的峰值可能对整体性能有较大影响，这个参数用来惩罚主问题求解时间的峰值过高，鼓励策略选择那些能够稳定控制主问题求解时间的割
                pressure_penalty_weight=pressure_penalty_weight, #在rollout过程中，主问题的压力（pressure）是一个衡量当前解的质量和求解难度的重要指标，这个参数用来惩罚压力过高的情况，鼓励策略选择那些能够有效降低压力的割
                near_binding_penalty_weight=near_binding_penalty_weight, #在rollout过程中，接近绑定的割（near-binding cuts）可能对求解器的性能产生较大影响，这个参数用来惩罚过多接近绑定的割，鼓励策略选择那些能够更好地平衡割的紧度和求解效率的割
                continuation_selector=default_continuation_selector,#在rollout过程中，当评估一个候选割时，可能需要在未来的迭代中继续选择其他割来添加到求解器中，这个参数指定了一个函数，用于根据当前的状态和候选割来选择未来迭代中要继续评估的割
            )
            rollout_rows.append((cand, result))

        # 根据 rollout 的结果对候选割进行排序，按照它们的总收益（total_return）从高到低排序。然后统计有多少候选割的总收益是正的，这些通常被认为是“有用”的割。接着为每个候选割分配一个排名，排名越靠前表示这个割在 rollout 中表现越好。
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
        # 为每个候选割分配一个排名，排名越靠前表示这个割在 rollout 中表现越好。这里使用了一个字典来存储候选割对象的 id 和它们对应的排名，方便后续在构建训练数据时快速查找每个候选割的排名信息。
        rank_by_id = {id(cand): rank for rank, (cand, _result) in enumerate(ranked_rollouts, start=1)}

        # 提取当前状态的特征，这些特征可能包括主问题的下界、上界、割的数量、网络设计方案的特征等信息。然后对于每个候选割，提取它的特征，这些特征可能包括割的违反程度、涉及的场景、对主问题下界的潜在改善等信息。最后构建一个包含所有这些特征和标签（如是否被选中）的训练数据行，并写入 CSV 文件中，供后续的机器学习模型训练使用。
        quota_cache = state_feature_dict(candidates, state, instance, master.x, max_iters=max_iters)
        # 从状态特征缓存中提取出与配额相关的特征值，这些特征值对应于预定义的 QUOTA_FEATURE_NAMES 列表中的特征名称。然后构建一个字典 state_feature_row，其中键是以 "quota_" 为前缀的特征名称，值是对应的特征值。这个字典将用于后续构建训练数据行时，提供当前状态下的配额相关特征信息。
        quota_features = [quota_cache[name] for name in QUOTA_FEATURE_NAMES]
        # 从状态特征缓存中提取出与配额相关的特征值，这些特征值对应于预定义的 QUOTA_FEATURE_NAMES 列表中的特征名称。然后构建一个字典 state_feature_row，其中键是以 "quota_" 为前缀的特征名称，值是对应的特征值。这个字典将用于后续构建训练数据行时，提供当前状态下的配额相关特征信息。
        state_feature_row = {f"quota_{name}": value for name, value in zip(QUOTA_FEATURE_NAMES, quota_features)}
        # 
        cache = quota_cache
        # 对于每个候选割，提取它的特征，这些特征可能包括割的违反程度、涉及的场景、对主问题下界的潜在改善等信息。最后构建一个包含所有这些特征和标签（如是否被选中）的训练数据行，并写入 CSV 文件中，供后续的机器学习模型训练使用。
        ranked_feature_rows: list[tuple[object, object, dict[str, float]]] = []
        # 对于每个候选割，提取它的特征，这些特征可能包括割的违反程度、涉及的场景、对主问题下界的潜在改善等信息。最后构建一个包含所有这些特征和标签（如是否被选中）的训练数据行，并写入 CSV 文件中，供后续的机器学习模型训练使用。
        for cand, result in ranked_rollouts:
            feat_dict = candidate_feature_dict(
                cand, #当前的候选割对象，包含了割的具体信息，如涉及的场景、违反程度、潜在改善等
                candidates, #当前迭代中所有的候选割列表，可能用于计算相对特征或其他基于候选集合的特征
                state, # 当前的选择状态对象，包含了当前迭代的各种状态信息，如下界、上界、割的数量等，可能用于计算与当前状态相关的特征
                instance, # 当前的网络设计实例对象，包含了问题的所有数据和参数，可能用于计算与实例相关的特征
                master.x, # 当前主问题给出的网络设计方案，可能用于计算与当前解相关的特征
                [],
                max_iters=max_iters,
                cached_state_features=cache, # 之前计算并缓存的状态特征，可能用于加速特征提取过程，避免重复计算一些与状态相关的特征
            )
            ranked_feature_rows.append((cand, result, feat_dict))

        # 根据 rollout 的结果对候选割进行排序，按照它们的总收益（total_return）从高到低排序。然后统计有多少候选割的总收益是正的，这些通常被认为是“有用”的割。接着为每个候选割分配一个排名，排名越靠前表示这个割在 rollout 中表现越好。
        ranking_scores = [float(result.total_return - baseline_rollout.total_return) for _cand, result, _feat in ranked_feature_rows]
        processed_scores: list[float] = []
        kept = 0
        stop_rank = selected_count + 1
        # 对于每个候选割，提取它的特征，这些特征可能包括割的违反程度、涉及的场景、对主问题下界的潜在改善等信息。最后构建一个包含所有这些特征和标签（如是否被选中）的训练数据行，并写入 CSV 文件中，供后续的机器学习模型训练使用。
        for rank_idx, (cand, result, feat_dict) in enumerate(ranked_feature_rows, start=1):
            # 当前候选割的总收益作为当前得分，排名列表中的下一个得分作为下一个得分（如果存在的话），然后构建一个包含当前状态特征、候选割特征、排名信息、得分信息等的字典 stop_features，这些特征将用于训练一个停止决策模型，帮助模型学习在什么情况下应该停止选择更多的割。
            current_score = float(result.total_return - baseline_rollout.total_return)
            # 排名列表中的下一个得分作为下一个得分（如果存在的话），然后构建一个包含当前状态特征、候选割特征、排名信息、得分信息等的字典 stop_features，这些特征将用于训练一个停止决策模型，帮助模型学习在什么情况下应该停止选择更多的割。
            next_score = ranking_scores[rank_idx] if rank_idx < len(ranking_scores) else 0.0
            # 构建停止决策模型的特征字典
            stop_features = build_stopping_feature_dict(
                state_features=state_feature_row, # 当前状态的特征字典，包含了与当前迭代状态相关的各种特征信息，如下界、上界、割的数量等，这些特征有助于模型理解当前的求解状态
                candidate_features=feat_dict, # 当前候选割的特征字典，包含了与当前候选割相关的各种特征信息，如违反程度、涉及的场景、潜在改善等，这些特征有助于模型理解这个割的具体情况
                rank_index=rank_idx, # 当前候选割的排名索引，表示这个割在所有候选割中的相对位置，排名越靠前表示这个割在 rollout 中表现越好，这个特征有助于模型理解这个割的相对优劣
                total_candidates=len(viable), # 当前迭代中真正有潜力的候选割的总数量，这个特征有助于模型理解当前的选择空间大小
                selected_count=kept, # 到目前为止已经被认为是“有用”的割的数量，这个特征有助于模型理解当前已经选择了多少个割，以及当前这个割在已经选择的割中的位置
                processed_scores=processed_scores, # 到目前为止已经处理过的候选割的得分列表，这个特征有助于模型理解当前这个割的得分在已经处理过的割中的相对位置，以及得分的分布情况
                current_score=current_score,
                next_score=next_score,
                top_score=ranking_scores[0] if ranking_scores else 0.0, # 当前迭代中候选割的最高得分，这个特征有助于模型理解当前这个割的得分与最高得分之间的差距，以及当前这个割在所有候选割中的表现水平
            )
            # 构建一个包含当前状态特征、候选割特征、排名信息、得分信息等的字典 stop_features，这些特征将用于训练一个停止决策模型，帮助模型学习在什么情况下应该停止选择更多的割。
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
            # 构建一个包含当前状态特征、候选割特征、排名信息、得分信息等的字典 stop_features，这些特征将用于训练一个停止决策模型，帮助模型学习在什么情况下应该停止选择更多的割。
            for name, value in zip(QUOTA_FEATURE_NAMES, quota_features):
                row[f"quota_{name}"] = value
            # 构建一个包含当前状态特征、候选割特征、排名信息、得分信息等的字典 stop_features，这些特征将用于训练一个停止决策模型，帮助模型学习在什么情况下应该停止选择更多的割。
            for name in POLICY_FEATURE_NAMES:
                row[name] = feat_dict[name]
            # 构建一个包含当前状态特征、候选割特征、排名信息、得分信息等的字典 stop_features，这些特征将用于训练一个停止决策模型，帮助模型学习在什么情况下应该停止选择更多的割。
            for name in STOPPING_FEATURE_NAMES:
                if name not in row:
                    row[name] = stop_features[name]
            writer.writerow(row)
            processed_scores.append(current_score)
            if rank_idx < stop_rank:
                kept += 1
        # 根据 rollout 的结果和当前的策略选择一些候选割来添加到求解器的主问题中，这些被选择的割通常是那些在 rollout 中表现较好的割，或者根据当前策略评估认为有潜力改善求解器性能的割。然后把这些被选择的割添加到求解器的最优性割池中，以便在后续的迭代中使用这些割来改进主问题的下界。
        rollout_selected = rollout_policy.select(candidates, state)
        # 
        for cand in rollout_selected:
            solver.optimality_cuts.append(
                BendersCut(
                    scenario=cand.scenario, #当前候选割涉及的随机场景索引，表示这个割是针对哪个随机场景生成的，这个信息对于求解器在后续迭代中正确地应用这个割非常重要
                    const=cand.const,
                    coeffs=cand.coeffs,
                    cut_type=cand.cut_type,
                )
            )

        # 
        gap = max(0.0, upper_bound - lower_bound)
        if gap <= tol or gap / (abs(upper_bound) + 1.0) <= tol:
            break


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
        # 对于每一组问题规模参数（节点数、边数、场景数、商品数）和每一个随机种子，调用 collect_for_seed 函数来收集数据，并把收集到的数据写入 CSV 文件中。这个过程会遍历所有指定的规模参数组合和随机种子，系统地生成大量的训练数据，用于后续的机器学习模型训练。
        for n_nodes, n_edges, n_scenarios, n_commodities in sizes:
            for seed in args.seeds:
                collect_for_seed(
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
                print(f"collected seed={seed} size={n_nodes},{n_edges},{n_scenarios},{n_commodities}")
    print(f"dataset={args.out}")


if __name__ == "__main__":
    main()
