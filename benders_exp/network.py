from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


TOPOLOGY_MODES = ("uniform", "distance_decay", "clustered")
SCENARIO_MODES = ("uniform", "mixed_extreme")


@dataclass(frozen=True)
class Commodity:
    source: int
    sink: int
    demand: float


@dataclass(frozen=True)
class NetworkScenario:
    probability: float
    commodities: list[Commodity]
    flow_costs: list[float]
    label: str = "regular"


@dataclass(frozen=True)
class NetworkDesignInstance:
    n_nodes: int
    edges: list[tuple[int, int]]
    edge_lengths: list[float]
    fixed_costs: list[float]
    capacities: list[float]
    node_xy: list[tuple[float, float]]
    scenarios: list[NetworkScenario]
    seed: int
    node_clusters: list[int]
    cluster_centers: list[tuple[float, float]]
    topology_mode: str
    scenario_mode: str

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def n_commodities(self) -> int:
        if not self.scenarios:
            return 0
        return len(self.scenarios[0].commodities)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_probabilities(weights: list[float]) -> list[float]:
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [max(0.0, weight) / total for weight in weights]


def weighted_choice(rng: random.Random, items: list[object], weights: list[float]) -> object:
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1]


def weighted_sample_without_replacement(
    rng: random.Random,
    items: list[tuple[int, int]],
    weights: list[float],
    count: int,
) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    chosen: list[tuple[int, int]] = []
    pool_items = list(items)
    pool_weights = list(weights)
    while pool_items and len(chosen) < count:
        edge = weighted_choice(rng, pool_items, pool_weights)
        index = pool_items.index(edge)
        chosen.append(edge)
        pool_items.pop(index)
        pool_weights.pop(index)
    return chosen


def add_instance_generator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topology-mode", choices=TOPOLOGY_MODES, default="clustered")
    parser.add_argument("--distance-decay", type=float, default=0.22)
    parser.add_argument("--shortcut-rate", type=float, default=0.12)
    parser.add_argument("--cluster-count", type=int, default=3)
    parser.add_argument("--cluster-strength", type=float, default=3.0)
    parser.add_argument("--scenario-mode", choices=SCENARIO_MODES, default="mixed_extreme")
    parser.add_argument("--extreme-scenario-rate", type=float, default=0.15)
    parser.add_argument("--hub-scenario-rate", type=float, default=0.20)
    parser.add_argument("--intercluster-scenario-rate", type=float, default=0.20)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--extreme-demand-scale", type=float, default=2.5)
    parser.add_argument("--hub-count", type=int, default=2)
    parser.add_argument("--cost-spike-rate", type=float, default=0.18)
    parser.add_argument("--cost-spike-scale", type=float, default=2.0)
    parser.add_argument("--capacity-factor", type=float, default=6.0)


def instance_generator_kwargs_from_args(args: argparse.Namespace) -> dict[str, float | int | str]:
    return {
        "topology_mode": args.topology_mode,
        "distance_decay": args.distance_decay,
        "shortcut_rate": args.shortcut_rate,
        "cluster_count": args.cluster_count,
        "cluster_strength": args.cluster_strength,
        "scenario_mode": args.scenario_mode,
        "extreme_scenario_rate": args.extreme_scenario_rate,
        "hub_scenario_rate": args.hub_scenario_rate,
        "intercluster_scenario_rate": args.intercluster_scenario_rate,
        "demand_scale": args.demand_scale,
        "extreme_demand_scale": args.extreme_demand_scale,
        "hub_count": args.hub_count,
        "cost_spike_rate": args.cost_spike_rate,
        "cost_spike_scale": args.cost_spike_scale,
        "capacity_factor": args.capacity_factor,
    }


def instance_generator_cli_args(args: argparse.Namespace) -> list[str]:
    return [
        "--topology-mode",
        str(args.topology_mode),
        "--distance-decay",
        str(args.distance_decay),
        "--shortcut-rate",
        str(args.shortcut_rate),
        "--cluster-count",
        str(args.cluster_count),
        "--cluster-strength",
        str(args.cluster_strength),
        "--scenario-mode",
        str(args.scenario_mode),
        "--extreme-scenario-rate",
        str(args.extreme_scenario_rate),
        "--hub-scenario-rate",
        str(args.hub_scenario_rate),
        "--intercluster-scenario-rate",
        str(args.intercluster_scenario_rate),
        "--demand-scale",
        str(args.demand_scale),
        "--extreme-demand-scale",
        str(args.extreme_demand_scale),
        "--hub-count",
        str(args.hub_count),
        "--cost-spike-rate",
        str(args.cost_spike_rate),
        "--cost-spike-scale",
        str(args.cost_spike_scale),
        "--capacity-factor",
        str(args.capacity_factor),
    ]


def sample_node_positions(
    n_nodes: int,
    rng: random.Random,
    topology_mode: str,
    cluster_count: int,
) -> tuple[list[tuple[float, float]], list[int], list[tuple[float, float]]]:
    if topology_mode != "clustered":
        positions = [(rng.random(), rng.random()) for _ in range(n_nodes)]
        return positions, [0] * n_nodes, [(0.5, 0.5)]

    n_clusters = max(2, min(cluster_count, n_nodes))
    grid_side = math.ceil(math.sqrt(n_clusters))
    grid_points = [
        ((i + 0.5) / grid_side, (j + 0.5) / grid_side)
        for i in range(grid_side)
        for j in range(grid_side)
    ]
    rng.shuffle(grid_points)
    centers = [
        (
            clip01(cx + rng.uniform(-0.08, 0.08)),
            clip01(cy + rng.uniform(-0.08, 0.08)),
        )
        for cx, cy in grid_points[:n_clusters]
    ]

    positions: list[tuple[float, float]] = []
    clusters: list[int] = []
    for _ in range(n_nodes):
        cluster = rng.randrange(n_clusters)
        cx, cy = centers[cluster]
        x = clip01(rng.gauss(cx, 0.12))
        y = clip01(rng.gauss(cy, 0.12))
        positions.append((x, y))
        clusters.append(cluster)
    return positions, clusters, centers


def pair_weight(
    edge: tuple[int, int],
    *,
    positions: list[tuple[float, float]],
    clusters: list[int],
    topology_mode: str,
    distance_decay: float,
    shortcut_rate: float,
    cluster_strength: float,
) -> float:
    u, v = edge
    distance = euclidean(positions[u], positions[v])
    local_scale = max(0.05, distance_decay)
    if topology_mode == "uniform":
        return 1.0
    base = math.exp(-distance / local_scale)
    if distance > 0.55:
        base += shortcut_rate
    if topology_mode == "distance_decay":
        return max(base, 1.0e-6)
    same_cluster = clusters[u] == clusters[v]
    cluster_multiplier = cluster_strength if same_cluster else 1.0 / max(1.0, cluster_strength)
    return max(base * cluster_multiplier, 1.0e-6)


def build_edge_set(
    n_nodes: int,
    n_edges: int,
    positions: list[tuple[float, float]],
    clusters: list[int],
    rng: random.Random,
    *,
    topology_mode: str,
    distance_decay: float,
    shortcut_rate: float,
    cluster_strength: float,
) -> set[tuple[int, int]]:
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    all_weights = [
        pair_weight(
            edge,
            positions=positions,
            clusters=clusters,
            topology_mode=topology_mode,
            distance_decay=distance_decay,
            shortcut_rate=shortcut_rate,
            cluster_strength=cluster_strength,
        )
        for edge in all_pairs
    ]

    edge_set: set[tuple[int, int]] = set()
    connected = {rng.randrange(n_nodes)}
    disconnected = set(range(n_nodes)) - connected
    while disconnected:
        candidate_edges = []
        candidate_weights = []
        for edge, weight in zip(all_pairs, all_weights):
            u, v = edge
            if (u in connected) ^ (v in connected):
                candidate_edges.append(edge)
                candidate_weights.append(weight)
        if not candidate_edges:
            raise RuntimeError("failed to connect generated network")
        chosen = weighted_choice(rng, candidate_edges, candidate_weights)
        edge_set.add(chosen)
        u, v = chosen
        connected.add(u)
        connected.add(v)
        disconnected.discard(u)
        disconnected.discard(v)

    remaining = [edge for edge in all_pairs if edge not in edge_set]
    remaining_weights = [
        pair_weight(
            edge,
            positions=positions,
            clusters=clusters,
            topology_mode=topology_mode,
            distance_decay=distance_decay,
            shortcut_rate=shortcut_rate,
            cluster_strength=cluster_strength,
        )
        for edge in remaining
    ]
    extras = weighted_sample_without_replacement(
        rng,
        remaining,
        remaining_weights,
        max(0, n_edges - len(edge_set)),
    )
    edge_set.update(extras)
    return edge_set


def choose_hubs(
    n_nodes: int,
    positions: list[tuple[float, float]],
    clusters: list[int],
    cluster_centers: list[tuple[float, float]],
    hub_count: int,
) -> list[int]:
    hub_count = max(1, min(hub_count, n_nodes))
    if len(cluster_centers) <= 1:
        center = cluster_centers[0] if cluster_centers else (0.5, 0.5)
        ranked = sorted(range(n_nodes), key=lambda node: euclidean(positions[node], center))
        return ranked[:hub_count]

    cluster_sizes = [clusters.count(idx) for idx in range(len(cluster_centers))]
    ranked_clusters = sorted(range(len(cluster_centers)), key=lambda idx: (-cluster_sizes[idx], idx))

    hubs: list[int] = []
    used: set[int] = set()
    for cluster_idx in ranked_clusters:
        cluster_nodes = [node for node in range(n_nodes) if clusters[node] == cluster_idx and node not in used]
        if not cluster_nodes:
            continue
        center = cluster_centers[cluster_idx]
        best = min(cluster_nodes, key=lambda node: euclidean(positions[node], center))
        hubs.append(best)
        used.add(best)
        if len(hubs) >= hub_count:
            return hubs

    fallback_center = (0.5, 0.5)
    fallback = sorted(
        [node for node in range(n_nodes) if node not in used],
        key=lambda node: euclidean(positions[node], fallback_center),
    )
    hubs.extend(fallback[: hub_count - len(hubs)])
    return hubs[:hub_count]


def sample_scenario_labels(
    n_scenarios: int,
    rng: random.Random,
    *,
    scenario_mode: str,
    extreme_scenario_rate: float,
    hub_scenario_rate: float,
    intercluster_scenario_rate: float,
) -> list[str]:
    if scenario_mode == "uniform":
        return ["regular"] * n_scenarios
    regular_rate = max(0.0, 1.0 - extreme_scenario_rate - hub_scenario_rate - intercluster_scenario_rate)
    labels = ["regular", "hub_surge", "inter_cluster", "disrupted"]
    probs = normalize_probabilities(
        [regular_rate, hub_scenario_rate, intercluster_scenario_rate, extreme_scenario_rate]
    )
    return [weighted_choice(rng, labels, probs) for _ in range(n_scenarios)]


def sample_regular_pair(rng: random.Random, n_nodes: int) -> tuple[int, int]:
    s = rng.randrange(n_nodes)
    t = rng.randrange(n_nodes)
    while t == s:
        t = rng.randrange(n_nodes)
    return s, t


def sample_hub_pair(rng: random.Random, n_nodes: int, hubs: list[int]) -> tuple[int, int]:
    if hubs and rng.random() < 0.7:
        s = rng.choice(hubs)
        t = rng.randrange(n_nodes)
        while t == s:
            t = rng.randrange(n_nodes)
        return s, t
    return sample_regular_pair(rng, n_nodes)


def sample_intercluster_pair(
    rng: random.Random,
    n_nodes: int,
    clusters: list[int],
) -> tuple[int, int]:
    s = rng.randrange(n_nodes)
    candidates = [node for node in range(n_nodes) if clusters[node] != clusters[s]]
    if not candidates:
        return sample_regular_pair(rng, n_nodes)
    t = rng.choice(candidates)
    return s, t


def build_scenario_commodities(
    rng: random.Random,
    *,
    label: str,
    n_nodes: int,
    n_commodities: int,
    hubs: list[int],
    clusters: list[int],
    demand_scale: float,
    extreme_demand_scale: float,
) -> list[Commodity]:
    commodities: list[Commodity] = []
    chosen_pairs: set[tuple[int, int]] = set()
    while len(commodities) < n_commodities:
        if label == "hub_surge":
            pair = sample_hub_pair(rng, n_nodes, hubs)
        elif label == "inter_cluster":
            pair = sample_intercluster_pair(rng, n_nodes, clusters)
        else:
            pair = sample_regular_pair(rng, n_nodes)
        if pair in chosen_pairs:
            continue
        chosen_pairs.add(pair)
        if label == "disrupted":
            demand = rng.uniform(4.0, 10.0) * demand_scale * extreme_demand_scale
        elif label == "hub_surge":
            demand = rng.uniform(3.0, 8.0) * demand_scale * 1.6
        elif label == "inter_cluster":
            demand = rng.uniform(2.5, 7.0) * demand_scale * 1.3
        else:
            demand = rng.uniform(2.0, 6.0) * demand_scale
        commodities.append(Commodity(source=pair[0], sink=pair[1], demand=demand))
    return commodities


def build_flow_costs(
    rng: random.Random,
    *,
    lengths: list[float],
    label: str,
    cost_spike_rate: float,
    cost_spike_scale: float,
) -> list[float]:
    flow_costs = [1.0 + 35.0 * length * rng.uniform(0.85, 1.2) for length in lengths]
    if label != "disrupted":
        return flow_costs
    spike_count = max(1, round(len(flow_costs) * cost_spike_rate))
    spike_indices = rng.sample(range(len(flow_costs)), min(spike_count, len(flow_costs)))
    for idx in spike_indices:
        flow_costs[idx] *= cost_spike_scale
    return flow_costs


def generate_network_design_instance(
    n_nodes: int,
    n_edges: int,
    n_scenarios: int,
    n_commodities: int,
    seed: int,
    *,
    topology_mode: str = "clustered",
    distance_decay: float = 0.22,
    shortcut_rate: float = 0.12,
    cluster_count: int = 3,
    cluster_strength: float = 3.0,
    scenario_mode: str = "mixed_extreme",
    extreme_scenario_rate: float = 0.15,
    hub_scenario_rate: float = 0.20,
    intercluster_scenario_rate: float = 0.20,
    demand_scale: float = 1.0,
    extreme_demand_scale: float = 2.5,
    hub_count: int = 2,
    cost_spike_rate: float = 0.18,
    cost_spike_scale: float = 2.0,
    capacity_factor: float = 6.0,
) -> NetworkDesignInstance:
    """Generate a stochastic fixed-charge network design instance."""

    if n_edges < n_nodes - 1:
        raise ValueError("n_edges must be at least n_nodes - 1")
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        raise ValueError(f"n_edges must be at most {max_edges} for a simple graph")
    if topology_mode not in TOPOLOGY_MODES:
        raise ValueError(f"unsupported topology_mode: {topology_mode}")
    if scenario_mode not in SCENARIO_MODES:
        raise ValueError(f"unsupported scenario_mode: {scenario_mode}")

    rng = random.Random(seed)
    positions, clusters, cluster_centers = sample_node_positions(n_nodes, rng, topology_mode, cluster_count)
    edge_set = build_edge_set(
        n_nodes,
        n_edges,
        positions,
        clusters,
        rng,
        topology_mode=topology_mode,
        distance_decay=distance_decay,
        shortcut_rate=shortcut_rate,
        cluster_strength=cluster_strength,
    )
    edges = sorted(edge_set)
    lengths = [euclidean(positions[u], positions[v]) for u, v in edges]
    fixed_costs = [20.0 + 120.0 * length * rng.uniform(0.8, 1.25) for length in lengths]

    scenario_labels = sample_scenario_labels(
        n_scenarios,
        rng,
        scenario_mode=scenario_mode,
        extreme_scenario_rate=extreme_scenario_rate,
        hub_scenario_rate=hub_scenario_rate,
        intercluster_scenario_rate=intercluster_scenario_rate,
    )
    demand_by_label = {
        "regular": 4.0 * demand_scale,
        "hub_surge": 5.5 * demand_scale * 1.6,
        "inter_cluster": 4.75 * demand_scale * 1.3,
        "disrupted": 7.0 * demand_scale * extreme_demand_scale,
    }
    expected_demand_per_commodity = sum(demand_by_label[label] for label in scenario_labels) / max(1, n_scenarios)
    scenario_demand_scale = expected_demand_per_commodity * n_commodities
    base_capacity = capacity_factor * scenario_demand_scale / max(1, n_edges)
    capacities = [max(1.0, base_capacity * rng.uniform(0.75, 1.5)) for _ in edges]

    hubs = choose_hubs(n_nodes, positions, clusters, cluster_centers, hub_count)
    scenarios: list[NetworkScenario] = []
    scenario_probability = 1.0 / max(1, n_scenarios)
    for label in scenario_labels:
        commodities = build_scenario_commodities(
            rng,
            label=label,
            n_nodes=n_nodes,
            n_commodities=n_commodities,
            hubs=hubs,
            clusters=clusters,
            demand_scale=demand_scale,
            extreme_demand_scale=extreme_demand_scale,
        )
        flow_costs = build_flow_costs(
            rng,
            lengths=lengths,
            label=label,
            cost_spike_rate=cost_spike_rate,
            cost_spike_scale=cost_spike_scale,
        )
        scenarios.append(
            NetworkScenario(
                probability=scenario_probability,
                commodities=commodities,
                flow_costs=flow_costs,
                label=label,
            )
        )

    return NetworkDesignInstance(
        n_nodes=n_nodes,
        edges=edges,
        edge_lengths=lengths,
        fixed_costs=fixed_costs,
        capacities=capacities,
        node_xy=positions,
        scenarios=scenarios,
        seed=seed,
        node_clusters=clusters,
        cluster_centers=cluster_centers,
        topology_mode=topology_mode,
        scenario_mode=scenario_mode,
    )
