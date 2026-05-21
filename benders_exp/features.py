from __future__ import annotations

import math
from collections import deque
from statistics import mean
from typing import Any

from .network import NetworkDesignInstance

# 定义了一个包含所有特征名称的列表，这些特征名称对应于在网络设计问题中用于描述当前状态和候选割的各种特征。
#这些特征包括了迭代比例、相对间隙、压力、割密度、停滞比例、主问题求解时间代理、节点数量、边密度、场景数量、商品数量、候选割密度、平均违规程度、最大违规程度、平均效能、最大效能、平均支持比例、最大支持比例、平均目标平行度、平均新颖性、平均场景分布、候选割场景覆盖率、候选割余弦相似度均值、候选割余弦相似度最大值、候选割支持重叠均值、候选割支持重叠最大值、候选割Gram矩阵谱比率、候选割冗余比率、候选割与活跃割的余弦相似度均值、候选割与活跃割的余弦相似度最大值、候选割与活跃割的新颖性均值，以及图结构相关的特征如组件比例、最大组件比例、叶子节点比例、开放边比例等。
FEATURE_NAMES = [
    "iteration_ratio", # 当前迭代次数与最大迭代次数的比例，这个特征有助于模型了解当前求解过程处于哪个阶段，通常在迭代初期模型可能需要更多的探索，而在迭代后期可能需要更多的利用
    "rel_gap", # 当前的相对间隙，定义为 (upper_bound - lower_bound) / max(1.0, abs(lower_bound))，这个特征反映了当前解的质量与最优解之间的差距，间隙越小表示当前解越接近最优解，这有助于模型判断是否需要继续探索更好的割
    "has_upper",
    "pressure",
    "cut_density",
    "stagnation_ratio",
    "master_time_proxy",
    "edge_density",
    "candidate_density",
    "avg_violation",
    "max_violation",
    "avg_efficacy",
    "max_efficacy",
    "avg_support_ratio",
    "max_support_ratio",
    "avg_obj_parallelism",
    "avg_novelty",
    "avg_scenario_spread",
    "candidate_scenario_coverage",
    "candidate_cosine_mean",
    "candidate_cosine_max",
    "candidate_overlap_mean",
    "candidate_overlap_max",
    "candidate_gram_spectral_ratio",
    "candidate_redundancy_ratio",
    "pool_to_active_cosine_mean",
    "pool_to_active_cosine_max",
    "pool_to_active_novelty_mean",
    "component_ratio",
    "largest_component_ratio",
    "leaf_ratio",
    "open_edge_ratio",
    "open_cost_ratio",
    "open_capacity_ratio",
    "open_length_ratio",
    "mean_degree_open",
    "degree_std_open",
    "bridge_ratio_open",
    "commodity_reachability_ratio",
    "commodity_path_stretch",
    "commodity_direct_length_ratio",
    "x_open_ratio",
    "x_fractional_ratio_lp",
    "x_integrality_gap_proxy",
    "x_entropy_open_pattern",
    "x_cost_weighted_open_ratio",
    "x_capacity_weighted_open_ratio",
    "avg_abs_reduced_cost_x_lp",
    "max_abs_reduced_cost_x_lp",
    "reduced_cost_dispersion_x_lp",
    "near_zero_reduced_cost_ratio_lp",
    "theta_mean",
    "theta_std",
    "theta_cv",
    "theta_active_ratio",
    "theta_scenario_skew",
    "fixed_cost_share",
    "recourse_share",
    "active_cut_ratio",
    "near_binding_cut_ratio",
    "avg_cut_slack",
    "min_cut_slack",
    "cut_slack_cv",
    "scenario_cut_imbalance",
    "avg_active_cut_scenario_ratio",
    "max_active_cut_scenario_ratio",
    "disconnected_scenario_ratio",
    "penalty_scenario_ratio",
    "unmet_demand_ratio",
    "overflow_ratio",
    "penalty_cost_ratio",
    "avg_mean_abs_beta",
    "avg_beta_gini",
    "avg_beta_entropy",
    "avg_beta_support_ratio",
    "scenario_beta_dispersion",
    "recourse_obj_mean",
    "recourse_obj_std",
    "recourse_obj_cv",
]

# 定义了一个包含候选割特征名称的列表，这些特征专门用于描述当前迭代中生成的候选割的各种属性，如违规程度、效能、支持比例、目标平行度、新颖性、场景分布等。这些特征有助于模型评估每个候选割的潜在价值，并预测它们在未来迭代中的表现。
CANDIDATE_FEATURE_NAMES = [
    "cand_violation",
    "cand_efficacy",
    "cand_support_ratio",
    "cand_abs_beta_mean",
    "cand_abs_beta_max",
    "cand_beta_gini",
    "cand_beta_entropy",
    "cand_obj_parallelism",
    "cand_novelty_to_active",
    "cand_novelty_to_selected",
    "cand_cosine_to_selected_max",
    "cand_overlap_to_selected_max",
    "cand_open_support_ratio",
    "cand_closed_support_ratio",
    "cand_lp_frac_support_mean",
    "cand_reduced_cost_support_mean",
    "cand_scenario_load_ratio",
    "cand_same_scenario_active_ratio",
    "cand_theta_gap_ratio",
    "cand_value_at_x_ratio",
]

POLICY_FEATURE_NAMES = FEATURE_NAMES + CANDIDATE_FEATURE_NAMES
QUOTA_FEATURE_NAMES = FEATURE_NAMES

# 定义了一个函数 finite，用于检查给定的浮点数值是否是有限的（即不是无穷大或NaN）。
#如果值不是有限的，函数会抛出一个 ValueError 异常。
def finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite feature value: {value}")
    return value

# 定义了一个函数 safe_mean，用于计算一个浮点数列表的平均值。
#函数首先计算列表中所有值的总和，然后除以列表的长度（如果列表为空，则除以1以避免除零错误）。
def safe_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))

# 定义了一个函数 safe_std，用于计算一个浮点数列表的标准差。
def safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = safe_mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(abs(v) for v in values)

# 定义了一个函数 entropy_from_weights，用于计算一个浮点数列表的熵值。
def entropy_from_weights(weights: list[float]) -> float:
    total = sum(abs(w) for w in weights)
    if total <= 1.0e-12:
        return 0.0
    probs = [abs(w) / total for w in weights if abs(w) > 1.0e-12]
    return -sum(p * math.log(p + 1.0e-12) for p in probs) / max(1.0, math.log(max(2, len(weights))))

# 定义了一个函数 gini_from_weights，用于计算一个浮点数列表的基尼系数。
def gini_from_weights(weights: list[float]) -> float:
    values = sorted(abs(w) for w in weights)
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total <= 1.0e-12:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2.0 * weighted) / (n * total) - (n + 1) / n

# 定义了几个函数用于计算向量之间的点积、范数、余弦相似度、支持重叠等，这些函数在计算候选割特征和状态特征时非常有用。
def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cut_lhs_norm(coeffs: list[float], theta_coeff: float = 1.0) -> float:
    return math.sqrt(theta_coeff * theta_coeff + sum(x * x for x in coeffs))


def cosine_abs(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom <= 1.0e-12:
        return 0.0
    return abs(dot(a, b) / denom)


def support_overlap(a: list[float], b: list[float]) -> float:
    sa = {idx for idx, value in enumerate(a) if abs(value) > 1.0e-9}
    sb = {idx for idx, value in enumerate(b) if abs(value) > 1.0e-9}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

# 定义了一个函数 scenario_loads，用于计算网络设计问题中每个场景的总需求量。
def scenario_loads(instance: NetworkDesignInstance) -> list[float]:
    return [sum(comm.demand for comm in scenario.commodities) for scenario in instance.scenarios]


def recourse_reference_scale(
    instance: NetworkDesignInstance,
    state: Any,
    candidates: list[Any],
) -> float:
    theta_values = list(getattr(getattr(state, "master_solution", None), "theta", []))
    candidate_values = [float(getattr(c, "value_at_x", 0.0)) for c in candidates]
    demand_values = scenario_loads(instance)
    fixed_costs = list(instance.fixed_costs)
    return max(
        1.0,
        max_abs(theta_values),
        max_abs(candidate_values),
        safe_mean(demand_values),
        safe_mean(fixed_costs),
    )


def coefficient_reference_scale(instance: NetworkDesignInstance, recourse_scale: float) -> float:
    return max(1.0, recourse_scale / max(1, instance.n_edges))

# 定义了一个函数 build_open_graph，用于根据当前的解向量 x 构建一个表示开放边的图的邻接表表示。
def build_open_graph(instance: NetworkDesignInstance, x: list[int]) -> tuple[list[list[tuple[int, int]]], list[int]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(instance.n_nodes)]
    open_edges: list[int] = []
    for edge_idx, ((u, v), value) in enumerate(zip(instance.edges, x)):
        if int(value) != 1:
            continue
        adjacency[u].append((v, edge_idx))
        adjacency[v].append((u, edge_idx))
        open_edges.append(edge_idx)
    return adjacency, open_edges

# 定义了一个函数 connected_components，用于计算一个图的连通分量。输入是一个邻接表表示的图，输出是一个包含所有连通分量的列表，每个连通分量是一个节点索引的列表。
def connected_components(adjacency: list[list[tuple[int, int]]]) -> list[list[int]]:
    seen = [False] * len(adjacency)
    components: list[list[int]] = []
    for root in range(len(adjacency)):
        if seen[root]:
            continue
        comp: list[int] = []
        queue = deque([root])
        seen[root] = True
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nxt, _edge in adjacency[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        components.append(comp)
    return components

# 定义了一个函数 shortest_path_length，用于计算在当前开放边构成的图中，从源节点到汇节点的最短路径长度。
#输入是图的邻接表表示、边的长度列表、源节点索引和汇节点索引。输出是最短路径长度，如果不可达则返回 None。
def shortest_path_length(
    adjacency: list[list[tuple[int, int]]],
    lengths: list[float],
    source: int,
    sink: int,
) -> float | None:
    if source == sink:
        return 0.0
    queue = deque([source])
    parent = {source: (-1, -1)}
    while queue:
        node = queue.popleft()
        for nxt, edge_idx in adjacency[node]:
            if nxt in parent:
                continue
            parent[nxt] = (node, edge_idx)
            if nxt == sink:
                total = 0.0
                cur = sink
                while parent[cur][0] != -1:
                    prev, used = parent[cur]
                    total += lengths[used]
                    cur = prev
                return total
            queue.append(nxt)
    return None


# 定义了一个函数 bridge_ratio，用于计算当前开放边构成的图中桥（即割边）的比例。
# 输入是图的邻接表表示和开放边的数量，输出是桥的数量除以开放边数量的比例。
def bridge_ratio(adjacency: list[list[tuple[int, int]]], edge_count: int) -> float:
    if edge_count <= 0:
        return 0.0
    timer = 0
    tin = [-1] * len(adjacency)
    low = [-1] * len(adjacency)
    bridges = 0

    def dfs(node: int, parent_edge: int) -> None:
        nonlocal timer, bridges
        tin[node] = timer
        low[node] = timer
        timer += 1
        for nxt, edge_idx in adjacency[node]:
            if edge_idx == parent_edge:
                continue
            if tin[nxt] != -1:
                low[node] = min(low[node], tin[nxt])
                continue
            dfs(nxt, edge_idx)
            low[node] = min(low[node], low[nxt])
            if low[nxt] > tin[node]:
                bridges += 1

    for node in range(len(adjacency)):
        if tin[node] == -1:
            dfs(node, -1)
    return bridges / edge_count

# 定义了一个函数 graph_state_features，用于计算当前解向量 x 对应的图状态的各种特征。
# 这些特征包括组件比例、最大组件比例、叶子节点比例、开放边比例、开放边成本比例、开放边容量比例、开放边长度比例、平均度数、度数标准差、桥比例、商品可达性比例、商品路径伸缩比、商品直接长度比等。这些特征有助于模型了解当前解对应的图结构和其对问题约束的满足程度。
def graph_state_features(instance: NetworkDesignInstance, x: list[int]) -> dict[str, float]:
    adjacency, open_edges = build_open_graph(instance, x)
    components = connected_components(adjacency)
    component_sizes = [len(comp) for comp in components] or [0]
    degrees = [len(neighbors) for neighbors in adjacency]
    open_edge_ratio = len(open_edges) / max(1, instance.n_edges)
    total_fixed = sum(instance.fixed_costs)
    total_capacity = sum(instance.capacities)
    total_length = sum(instance.edge_lengths)

    open_cost = sum(instance.fixed_costs[e] for e in open_edges)
    open_capacity = sum(instance.capacities[e] for e in open_edges)
    open_length = sum(instance.edge_lengths[e] for e in open_edges)

    reachable = 0.0
    stretch_values: list[float] = []
    direct_ratio_values: list[float] = []
    for scenario in instance.scenarios:
        for commodity in scenario.commodities:
            path_len = shortest_path_length(adjacency, instance.edge_lengths, commodity.source, commodity.sink)
            direct = math.dist(instance.node_xy[commodity.source], instance.node_xy[commodity.sink])
            if path_len is not None:
                reachable += 1.0
                if direct > 1.0e-9:
                    stretch_values.append(path_len / direct)
                    direct_ratio_values.append(direct / max(path_len, 1.0e-9))
            else:
                stretch_values.append(5.0)
                direct_ratio_values.append(0.0)

    total_commodities = max(1, instance.n_scenarios * instance.n_commodities)
    mean_degree = sum(degrees) / max(1, instance.n_nodes)
    degree_var = sum((deg - mean_degree) ** 2 for deg in degrees) / max(1, instance.n_nodes)

    return {
        "component_ratio": len(components) / max(1, instance.n_nodes),
        "largest_component_ratio": max(component_sizes) / max(1, instance.n_nodes),
        "leaf_ratio": sum(1 for deg in degrees if deg == 1) / max(1, instance.n_nodes),
        "open_edge_ratio": open_edge_ratio,
        "open_cost_ratio": open_cost / max(1.0, total_fixed),
        "open_capacity_ratio": open_capacity / max(1.0, total_capacity),
        "open_length_ratio": open_length / max(1.0, total_length),
        "mean_degree_open": mean_degree / max(1.0, instance.n_nodes - 1),
        "degree_std_open": math.sqrt(degree_var) / max(1.0, instance.n_nodes - 1),
        "bridge_ratio_open": bridge_ratio(adjacency, len(open_edges)),
        "commodity_reachability_ratio": reachable / total_commodities,
        "commodity_path_stretch": safe_mean(stretch_values),
        "commodity_direct_length_ratio": safe_mean(direct_ratio_values),
    }

# 定义了一个函数 candidate_stats，用于计算当前迭代中生成的候选割的各种统计特征。
def candidate_stats(
    candidates: list[Any],
    state: Any,
    instance: NetworkDesignInstance,
) -> dict[str, float]:
    viable = [c for c in candidates if c.violation > 1.0e-7]
    if not viable:
        return {
            "candidate_density": 0.0,
            "avg_violation": 0.0,
            "max_violation": 0.0,
            "avg_efficacy": 0.0,
            "max_efficacy": 0.0,
            "avg_support_ratio": 0.0,
            "max_support_ratio": 0.0,
            "avg_obj_parallelism": 0.0,
            "avg_novelty": 0.0,
            "avg_scenario_spread": 0.0,
            "candidate_scenario_coverage": 0.0,
            "candidate_cosine_mean": 0.0,
            "candidate_cosine_max": 0.0,
            "candidate_overlap_mean": 0.0,
            "candidate_overlap_max": 0.0,
            "candidate_gram_spectral_ratio": 0.0,
            "candidate_redundancy_ratio": 0.0,
            "pool_to_active_cosine_mean": 0.0,
            "pool_to_active_cosine_max": 0.0,
            "pool_to_active_novelty_mean": 0.0,
            "avg_mean_abs_beta": 0.0,
            "avg_beta_gini": 0.0,
            "avg_beta_entropy": 0.0,
            "avg_beta_support_ratio": 0.0,
            "scenario_beta_dispersion": 0.0,
            "recourse_obj_mean": 0.0,
            "recourse_obj_std": 0.0,
            "recourse_obj_cv": 0.0,
        }

    recourse_scale = recourse_reference_scale(instance, state, viable)
    coeff_scale = coefficient_reference_scale(instance, recourse_scale)

    violations = [c.violation / recourse_scale for c in viable]
    efficacies = [c.violation / (cut_lhs_norm(c.coeffs) + 1.0e-9) for c in viable]
    supports = [
        sum(1 for x in c.coeffs if abs(x) > 1.0e-9) / max(1, len(c.coeffs))
        for c in viable
    ]
    objpars = [cosine_abs(c.coeffs, state.fixed_costs) for c in viable]
    novelty: list[float] = []
    scenario_intensity = scenario_loads(instance)
    scenario_spread: list[float] = []
    beta_means: list[float] = []
    beta_ginis: list[float] = []
    beta_entropies: list[float] = []
    beta_supports: list[float] = []
    recourse_objs = [c.value_at_x / recourse_scale for c in viable]
    active_vectors = [vec for vectors in state.active_cuts_by_scenario.values() for vec in vectors]

    for c in viable:
        active = state.active_cuts_by_scenario.get(c.scenario, [])
        if not active:
            novelty.append(1.0)
        else:
            novelty.append(1.0 - max(cosine_abs(c.coeffs, a) for a in active))
        load = scenario_intensity[c.scenario] if c.scenario < len(scenario_intensity) else 0.0
        scenario_spread.append(load / max(1.0, safe_mean(scenario_intensity)))
        beta_means.append(safe_mean([abs(v) for v in c.coeffs]) / coeff_scale)
        beta_ginis.append(gini_from_weights(c.coeffs))
        beta_entropies.append(entropy_from_weights(c.coeffs))
        beta_supports.append(sum(1 for v in c.coeffs if abs(v) > 1.0e-9) / max(1, len(c.coeffs)))

    pair_cosines: list[float] = []
    pair_overlaps: list[float] = []
    for i in range(len(viable)):
        for j in range(i + 1, len(viable)):
            pair_cosines.append(cosine_abs(viable[i].coeffs, viable[j].coeffs))
            pair_overlaps.append(support_overlap(viable[i].coeffs, viable[j].coeffs))

    max_row_sum = 0.0
    if viable:
        for lhs in viable:
            row_sum = sum(cosine_abs(lhs.coeffs, rhs.coeffs) for rhs in viable)
            max_row_sum = max(max_row_sum, row_sum)
    gram_spectral_ratio = max_row_sum / max(1.0, len(viable))

    active_cosines: list[float] = []
    for c in viable:
        for active in active_vectors:
            active_cosines.append(cosine_abs(c.coeffs, active))

    scenario_counts = {}
    for c in viable:
        scenario_counts[c.scenario] = scenario_counts.get(c.scenario, 0) + 1
    scenario_cover = len(scenario_counts) / max(1, instance.n_scenarios)

    return {
        "candidate_density": len(viable) / max(1, instance.n_scenarios),
        "avg_violation": mean(violations),
        "max_violation": max(violations),
        "avg_efficacy": mean(efficacies),
        "max_efficacy": max(efficacies),
        "avg_support_ratio": mean(supports),
        "max_support_ratio": max(supports),
        "avg_obj_parallelism": mean(objpars),
        "avg_novelty": mean(novelty),
        "avg_scenario_spread": mean(scenario_spread),
        "candidate_scenario_coverage": scenario_cover,
        "candidate_cosine_mean": safe_mean(pair_cosines),
        "candidate_cosine_max": max(pair_cosines) if pair_cosines else 0.0,
        "candidate_overlap_mean": safe_mean(pair_overlaps),
        "candidate_overlap_max": max(pair_overlaps) if pair_overlaps else 0.0,
        "candidate_gram_spectral_ratio": gram_spectral_ratio,
        "candidate_redundancy_ratio": sum(1 for value in pair_cosines if value >= 0.90) / max(1, len(pair_cosines)),
        "pool_to_active_cosine_mean": safe_mean(active_cosines),
        "pool_to_active_cosine_max": max(active_cosines) if active_cosines else 0.0,
        "pool_to_active_novelty_mean": mean(novelty),
        "avg_mean_abs_beta": mean(beta_means),
        "avg_beta_gini": mean(beta_ginis),
        "avg_beta_entropy": mean(beta_entropies),
        "avg_beta_support_ratio": mean(beta_supports),
        "scenario_beta_dispersion": safe_std(beta_means) / max(1.0e-9, safe_mean(beta_means)),
        "recourse_obj_mean": safe_mean(recourse_objs),
        "recourse_obj_std": safe_std(recourse_objs),
        "recourse_obj_cv": safe_std(recourse_objs) / max(1.0e-9, safe_mean([abs(v) for v in recourse_objs])),
    }

# 定义了一个函数 master_polyhedral_features，用于计算主问题解和多面体状态相关的特征。
def master_polyhedral_features(
    instance: NetworkDesignInstance,
    state: Any,
) -> dict[str, float]:
    master = state.master_solution
    poly = state.polyhedral_state
    x = state.x or []
    if master is None or poly is None or not x:
        return {
            "x_open_ratio": 0.0,
            "x_fractional_ratio_lp": 0.0,
            "x_integrality_gap_proxy": 0.0,
            "x_entropy_open_pattern": 0.0,
            "x_cost_weighted_open_ratio": 0.0,
            "x_capacity_weighted_open_ratio": 0.0,
            "avg_abs_reduced_cost_x_lp": 0.0,
            "max_abs_reduced_cost_x_lp": 0.0,
            "reduced_cost_dispersion_x_lp": 0.0,
            "near_zero_reduced_cost_ratio_lp": 0.0,
            "theta_mean": 0.0,
            "theta_std": 0.0,
            "theta_cv": 0.0,
            "theta_active_ratio": 0.0,
            "theta_scenario_skew": 0.0,
            "fixed_cost_share": 0.0,
            "recourse_share": 0.0,
            "active_cut_ratio": 0.0,
            "near_binding_cut_ratio": 0.0,
            "avg_cut_slack": 0.0,
            "min_cut_slack": 0.0,
            "cut_slack_cv": 0.0,
            "scenario_cut_imbalance": 0.0,
            "avg_active_cut_scenario_ratio": 0.0,
            "max_active_cut_scenario_ratio": 0.0,
        }

    x_lp = list(master.x_lp)
    x_red = [abs(value) for value in master.x_reduced_costs_lp]
    theta = list(master.theta)
    theta_scale = max(1.0, max_abs(theta))
    fixed_cost_scale = max(1.0, safe_mean(list(instance.fixed_costs)))
    fixed_cost = sum(cost * float(value) for cost, value in zip(instance.fixed_costs, x))
    recourse = sum(theta) / max(1, instance.n_scenarios)
    total_obj = max(1.0, fixed_cost + recourse)
    open_ratio = sum(x) / max(1, instance.n_edges)
    frac_ratio = sum(1 for value in x_lp if 1.0e-6 < value < 1.0 - 1.0e-6) / max(1, len(x_lp))
    integrality_gap = safe_mean([abs(round(value) - value) for value in x_lp])
    entropy = entropy_from_weights([float(value) for value in x])
    near_zero_rc = sum(1 for value in x_red if value <= 1.0e-6) / max(1, len(x_red))

    return {
        "x_open_ratio": open_ratio,
        "x_fractional_ratio_lp": frac_ratio,
        "x_integrality_gap_proxy": integrality_gap,
        "x_entropy_open_pattern": entropy,
        "x_cost_weighted_open_ratio": fixed_cost / max(1.0, sum(instance.fixed_costs)),
        "x_capacity_weighted_open_ratio": sum(cap * float(val) for cap, val in zip(instance.capacities, x)) / max(1.0, sum(instance.capacities)),
        "avg_abs_reduced_cost_x_lp": safe_mean(x_red) / fixed_cost_scale,
        "max_abs_reduced_cost_x_lp": (max(x_red) / fixed_cost_scale) if x_red else 0.0,
        "reduced_cost_dispersion_x_lp": safe_std(x_red) / fixed_cost_scale,
        "near_zero_reduced_cost_ratio_lp": near_zero_rc,
        "theta_mean": safe_mean(theta) / theta_scale,
        "theta_std": safe_std(theta) / theta_scale,
        "theta_cv": safe_std(theta) / max(1.0e-9, safe_mean([abs(v) for v in theta])),
        "theta_active_ratio": sum(1 for value in theta if value > 1.0e-8) / max(1, len(theta)),
        "theta_scenario_skew": (max(theta) / max(1.0e-9, safe_mean(theta))) if theta else 0.0,
        "fixed_cost_share": fixed_cost / total_obj,
        "recourse_share": recourse / total_obj,
        "active_cut_ratio": float(poly.active_cut_ratio),
        "near_binding_cut_ratio": float(poly.near_binding_cut_ratio),
        "avg_cut_slack": float(poly.avg_cut_slack) / theta_scale,
        "min_cut_slack": float(poly.min_cut_slack) / theta_scale,
        "cut_slack_cv": float(poly.cut_slack_cv),
        "scenario_cut_imbalance": float(poly.scenario_cut_imbalance),
        "avg_active_cut_scenario_ratio": safe_mean(list(poly.active_cut_scenario_ratios)),
        "max_active_cut_scenario_ratio": max(list(poly.active_cut_scenario_ratios)) if poly.active_cut_scenario_ratios else 0.0,
    }

# 定义了一个函数 state_feature_dict，用于计算当前状态的所有特征，并将它们组织成一个字典返回。
def state_feature_dict(
    candidates: list[Any],
    state: Any,
    instance: NetworkDesignInstance,
    x: list[int],
    *,
    max_iters: int = 100,
) -> dict[str, float]:
    graph = graph_state_features(instance, x)
    cand = candidate_stats(candidates, state, instance)
    master = master_polyhedral_features(instance, state)
    return {
        "iteration_ratio": float(state.iteration) / max(1, max_iters),
        "rel_gap": finite(min(state.rel_gap, 10.0)),
        "has_upper": 1.0 if state.upper_bound < 1.0e90 else 0.0,
        "pressure": finite(state.pressure),
        "cut_density": float(state.cut_count) / max(1, instance.n_edges * instance.n_scenarios),
        "stagnation_ratio": float(state.stagnation) / max(1, max_iters),
        "master_time_proxy": finite(math.log1p(float(getattr(state.master_solution, "solve_time", 0.0)))),
        "node_count": float(instance.n_nodes),
        "edge_density": float(instance.n_edges) / max(1, instance.n_nodes * (instance.n_nodes - 1) / 2),
        "scenario_count": float(instance.n_scenarios),
        "commodity_count": float(instance.n_commodities),
        "disconnected_scenario_ratio": float(state.disconnected_scenario_ratio),
        "penalty_scenario_ratio": float(state.penalty_scenario_ratio),
        "unmet_demand_ratio": float(state.unmet_demand_ratio),
        "overflow_ratio": float(state.overflow_ratio),
        "penalty_cost_ratio": float(state.penalty_cost_ratio),
        **cand,
        **graph,
        **master,
    }


def feature_vector_from_dict(features: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for name in FEATURE_NAMES:
        raw = features[name]
        values.append(finite(float(raw)))
    return values


def feature_vector(
    candidates: list[Any],
    state: Any,
    instance: NetworkDesignInstance,
    x: list[int],
    *,
    max_iters: int = 100,
) -> list[float]:
    return feature_vector_from_dict(
        state_feature_dict(candidates, state, instance, x, max_iters=max_iters)
    )

# 定义了一个函数 candidate_feature_dict，用于计算一个特定候选割的特征，并将它们组织成一个字典返回。
def candidate_feature_dict(
    candidate: Any,
    candidates: list[Any],
    state: Any,
    instance: NetworkDesignInstance,
    x: list[int],
    selected: list[Any],
    *,
    max_iters: int = 100,
    cached_state_features: dict[str, float] | None = None,
) -> dict[str, float]:
    state_features = cached_state_features or state_feature_dict(
        candidates,
        state,
        instance,
        x,
        max_iters=max_iters,
    )
    recourse_scale = recourse_reference_scale(instance, state, candidates)
    coeff_scale = coefficient_reference_scale(instance, recourse_scale)

    support = [idx for idx, value in enumerate(candidate.coeffs) if abs(value) > 1.0e-9]
    support_ratio = len(support) / max(1, len(candidate.coeffs))
    open_support = sum(1 for idx in support if int(x[idx]) == 1)
    closed_support = len(support) - open_support
    abs_beta = [abs(candidate.coeffs[idx]) for idx in support]
    x_lp = list(getattr(state.master_solution, "x_lp", []))
    x_red = [abs(value) for value in getattr(state.master_solution, "x_reduced_costs_lp", [])]
    theta_value = max(1.0, abs(candidate.theta_value), recourse_scale)
    scenario_load = scenario_loads(instance)
    load_ratio = 0.0
    if 0 <= candidate.scenario < len(scenario_load):
        load_ratio = scenario_load[candidate.scenario] / max(1.0, safe_mean(scenario_load))

    active_same_scenario = state.active_cuts_by_scenario.get(candidate.scenario, [])
    novelty_to_active = 1.0
    if active_same_scenario:
        novelty_to_active = 1.0 - max(cosine_abs(candidate.coeffs, vec) for vec in active_same_scenario)

    novelty_to_selected = 1.0
    cosine_to_selected_max = 0.0
    overlap_to_selected_max = 0.0
    if selected:
        cosine_to_selected_max = max(cosine_abs(candidate.coeffs, other.coeffs) for other in selected)
        overlap_to_selected_max = max(support_overlap(candidate.coeffs, other.coeffs) for other in selected)
        novelty_to_selected = 1.0 - cosine_to_selected_max

    lp_frac_values = [x_lp[idx] for idx in support if idx < len(x_lp)]
    red_values = [x_red[idx] for idx in support if idx < len(x_red)]
    active_ratio = 0.0
    if state.polyhedral_state is not None and candidate.scenario < len(state.polyhedral_state.active_cut_scenario_ratios):
        active_ratio = float(state.polyhedral_state.active_cut_scenario_ratios[candidate.scenario])

    candidate_features = {
        "cand_violation": candidate.violation / recourse_scale,
        "cand_efficacy": candidate.violation / (cut_lhs_norm(candidate.coeffs) + 1.0e-9),
        "cand_support_ratio": support_ratio,
        "cand_abs_beta_mean": safe_mean(abs_beta) / coeff_scale,
        "cand_abs_beta_max": (max(abs_beta) / coeff_scale) if abs_beta else 0.0,
        "cand_beta_gini": gini_from_weights(candidate.coeffs),
        "cand_beta_entropy": entropy_from_weights(candidate.coeffs),
        "cand_obj_parallelism": cosine_abs(candidate.coeffs, state.fixed_costs),
        "cand_novelty_to_active": novelty_to_active,
        "cand_novelty_to_selected": novelty_to_selected,
        "cand_cosine_to_selected_max": cosine_to_selected_max,
        "cand_overlap_to_selected_max": overlap_to_selected_max,
        "cand_open_support_ratio": open_support / max(1, len(support)),
        "cand_closed_support_ratio": closed_support / max(1, len(support)),
        "cand_lp_frac_support_mean": safe_mean(lp_frac_values),
        "cand_reduced_cost_support_mean": safe_mean(red_values),
        "cand_scenario_load_ratio": load_ratio,
        "cand_same_scenario_active_ratio": active_ratio,
        "cand_theta_gap_ratio": candidate.violation / theta_value,
        "cand_value_at_x_ratio": candidate.value_at_x / recourse_scale,
    }
    return {**state_features, **candidate_features}

# 定义了一个函数 candidate_feature_vector_from_dict，用于将候选割特征字典转换为一个浮点数列表，按照 CANDIDATE_FEATURE_NAMES 中定义的顺序。
def candidate_feature_vector_from_dict(features: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for name in POLICY_FEATURE_NAMES:
        raw = features[name]
        values.append(finite(float(raw)))
    return values

# 定义了一个函数 candidate_feature_vector，用于计算一个特定候选割的特征，并将它们转换为一个浮点数列表，按照 POLICY_FEATURE_NAMES 中定义的顺序。
def candidate_feature_vector(
    candidate: Any,
    candidates: list[Any],
    state: Any,
    instance: NetworkDesignInstance,
    x: list[int],
    selected: list[Any],
    *,
    max_iters: int = 100,
    cached_state_features: dict[str, float] | None = None,
) -> list[float]:
    return candidate_feature_vector_from_dict(
        candidate_feature_dict(
            candidate,
            candidates,
            state,
            instance,
            x,
            selected,
            max_iters=max_iters,
            cached_state_features=cached_state_features,
        )
    )
