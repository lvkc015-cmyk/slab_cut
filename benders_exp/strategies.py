from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .policy_actions import learned_candidate_selection


EPS = 1.0e-9


@dataclass(frozen=True)
class CutCandidate:
    scenario: int
    const: float
    coeffs: list[float]
    cut_type: str
    value_at_x: float
    theta_value: float

    @property
    def violation(self) -> float:
        return max(0.0, self.value_at_x - self.theta_value)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cut_lhs_norm(coeffs: list[float], theta_coeff: float = 1.0) -> float:
    return math.sqrt(theta_coeff * theta_coeff + sum(x * x for x in coeffs))


def cosine_abs(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom <= EPS:
        return 0.0
    return abs(dot(a, b) / denom)


def robust_scale(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= EPS:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


@dataclass
class SelectionState:
    iteration: int
    lower_bound: float
    upper_bound: float
    cut_count: int
    n_facilities: int
    n_scenarios: int
    stagnation: int
    fixed_costs: list[float]
    active_cuts_by_scenario: dict[int, list[list[float]]]
    instance: Any | None = None
    x: list[int] | None = None
    max_iters: int = 100
    master_solution: Any | None = None
    polyhedral_state: Any | None = None
    disconnected_scenario_ratio: float = 0.0
    penalty_scenario_ratio: float = 0.0
    unmet_demand_ratio: float = 0.0
    overflow_ratio: float = 0.0
    penalty_cost_ratio: float = 0.0

    @property
    def rel_gap(self) -> float:
        if self.upper_bound >= 1.0e90:
            return 1.0e90
        return max(0.0, (self.upper_bound - self.lower_bound) / (abs(self.upper_bound) + 1.0))

    @property
    def pressure(self) -> float:
        denom = max(1, self.n_facilities * self.n_scenarios)
        return self.cut_count / denom


class CutSelectionStrategy:
    name = "base"

    def select(self, candidates: list[CutCandidate], state: SelectionState) -> list[CutCandidate]:
        raise NotImplementedError


class MultiCutStrategy(CutSelectionStrategy):
    name = "multi"

    def select(self, candidates: list[CutCandidate], state: SelectionState) -> list[CutCandidate]:
        return [c for c in candidates if c.violation > 1.0e-7]


class TopKViolationStrategy(CutSelectionStrategy):
    name = "topk"

    def __init__(self, k: int) -> None:
        self.k = k

    def select(self, candidates: list[CutCandidate], state: SelectionState) -> list[CutCandidate]:
        viable = [c for c in candidates if c.violation > 1.0e-7]
        viable.sort(key=lambda c: c.violation, reverse=True)
        return viable[: self.k]


class SlabLikeStrategy(CutSelectionStrategy):
    name = "slab"

    def __init__(self, k_min: int = 1, k_max: int = 6, rho: float = 0.35) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.rho = rho

    def budget(self, state: SelectionState, n_candidates: int) -> int:
        if n_candidates <= 0:
            return 0
        raw = self.k_max
        if state.stagnation >= 3:
            raw += 2
        adjusted = max(self.k_min, int(math.ceil(raw * math.exp(-self.rho * state.pressure))))
        return max(self.k_min, min(n_candidates, adjusted))

    def select(self, candidates: list[CutCandidate], state: SelectionState) -> list[CutCandidate]:
        viable = [c for c in candidates if c.violation > 1.0e-7]
        if not viable:
            return []

        k = self.budget(state, len(viable))
        violations = [c.violation for c in viable]
        efficacies = [c.violation / (cut_lhs_norm(c.coeffs) + EPS) for c in viable]
        sparsities = [
            1.0 - sum(1 for x in c.coeffs if abs(x) > 1.0e-9) / max(1, len(c.coeffs))
            for c in viable
        ]
        objpars = [cosine_abs(c.coeffs, state.fixed_costs) for c in viable]

        novelty = []
        for c in viable:
            active = state.active_cuts_by_scenario.get(c.scenario, [])
            if not active:
                novelty.append(1.0)
            else:
                novelty.append(1.0 - max(cosine_abs(c.coeffs, a) for a in active))

        nv = robust_scale(violations)
        ne = robust_scale(efficacies)
        ns = robust_scale(sparsities)
        no = robust_scale(objpars)
        nd = robust_scale(novelty)

        if state.rel_gap > 0.15 and state.pressure < 1.5:
            weights = (0.35, 0.35, 0.15, 0.05, 0.10)
        elif state.pressure >= 1.5:
            weights = (0.18, 0.22, 0.30, 0.20, 0.10)
        else:
            weights = (0.20, 0.25, 0.30, 0.15, 0.10)

        scored = []
        for idx, c in enumerate(viable):
            score = (
                weights[0] * nv[idx]
                + weights[1] * ne[idx]
                + weights[2] * nd[idx]
                + weights[3] * ns[idx]
                + weights[4] * no[idx]
            )
            scored.append((score, c))
        scored.sort(key=lambda item: item[0], reverse=True)

        selected: list[CutCandidate] = []
        threshold = 0.92 if state.pressure < 1.5 else 0.82
        for _, cand in scored:
            if len(selected) >= k:
                break
            if all(cosine_abs(cand.coeffs, chosen.coeffs) <= threshold for chosen in selected):
                selected.append(cand)

        if not selected:
            selected.append(max(viable, key=lambda c: c.violation))
        return selected


class PolicyModelStrategy(CutSelectionStrategy):
    name = "policy"

    def __init__(self, model_path: str | Path, fallback: str = "slab", k: int = 3) -> None:
        from joblib import load

        payload: Any = load(model_path)
        if not isinstance(payload, dict) or ("ranking_model" not in payload and "candidate_model" not in payload):
            raise ValueError(f"invalid policy payload: {model_path}")
        self.payload = payload

    def select(self, candidates: list[CutCandidate], state: SelectionState) -> list[CutCandidate]:
        instance = getattr(state, "instance", None)
        x = getattr(state, "x", None)
        if instance is None or x is None:
            raise ValueError("policy selection requires state.instance and state.x")
        selected, _ranked, _diag = learned_candidate_selection(self.payload, candidates, state)
        return selected


def make_strategy(name: str, k: int = 3) -> CutSelectionStrategy:
    key = name.lower()
    if key == "multi":
        return MultiCutStrategy()
    if key == "topk":
        return TopKViolationStrategy(k)
    if key == "slab":
        return SlabLikeStrategy(k_max=max(k, 1))
    if key.startswith("policy:"):
        return PolicyModelStrategy(name.split(":", 1)[1], k=k)
    raise ValueError(f"unknown strategy: {name}")
