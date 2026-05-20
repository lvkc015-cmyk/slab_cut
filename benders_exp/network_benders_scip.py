from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from pyscipopt import Model, SCIP_PARAMSETTING, quicksum

from .benders import BendersCut, BendersResult, FeasibilityCut, MasterSolution
from .network import NetworkDesignInstance, NetworkScenario
from .scip_utils import configure_scip
from .strategies import CutCandidate, CutSelectionStrategy, SelectionState, dot


EPS = 1.0e-8


@dataclass(frozen=True)
class MasterPolyhedralState:
    x_lp: list[float]
    theta_lp: list[float]
    x_reduced_costs: list[float]
    theta_reduced_costs: list[float]
    lp_objective: float
    active_cut_ratio: float
    near_binding_cut_ratio: float
    avg_cut_slack: float
    min_cut_slack: float
    cut_slack_cv: float
    scenario_cut_imbalance: float
    active_cut_scenario_ratios: list[float]


def disconnected_component_cut(
    instance: NetworkDesignInstance,
    x: list[int],
    scenario: NetworkScenario,
) -> FeasibilityCut | None:
    adjacency = [[] for _ in range(instance.n_nodes)]
    for e, (u, v) in enumerate(instance.edges):
        if x[e] == 1:
            adjacency[u].append(v)
            adjacency[v].append(u)

    for commodity in scenario.commodities:
        seen = {commodity.source}
        stack = [commodity.source]
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if commodity.sink not in seen:
            coeffs = []
            for u, v in instance.edges:
                crosses = (u in seen and v not in seen) or (v in seen and u not in seen)
                coeffs.append(1.0 if crosses else 0.0)
            if sum(coeffs) > 0:
                return FeasibilityCut(coeffs=coeffs, rhs=1.0)
    return None


def all_commodities_connected(
    instance: NetworkDesignInstance,
    x: list[int],
    scenario: NetworkScenario,
) -> bool:
    return disconnected_component_cut(instance, x, scenario) is None


def route1_all_open_feasibility_check(
    instance: NetworkDesignInstance,
    *,
    time_limit: float | None = None,
    threads: int = 1,
) -> tuple[bool, str]:
    all_open = [1] * instance.n_edges
    for s, scenario in enumerate(instance.scenarios):
        if not all_commodities_connected(instance, all_open, scenario):
            return False, f"scenario={s} disconnected under all-open design"
        phase1 = solve_network_phase1_scip(
            instance,
            scenario,
            all_open,
            time_limit=time_limit,
            threads=threads,
        )
        if not phase1.feasible:
            return False, f"scenario={s} phase1 objective={phase1.objective:.12g} under all-open design"
    return True, ""


@dataclass(frozen=True)
class NetworkSubproblemResult:
    feasible: bool
    objective: float
    const: float
    coeffs: list[float]
    solve_time: float
    unmet_demand: float = 0.0
    overflow: float = 0.0


@dataclass(frozen=True)
class PenaltyStats:
    disconnected_scenario_ratio: float
    penalty_scenario_ratio: float
    unmet_demand_ratio: float
    overflow_ratio: float
    penalty_cost_ratio: float
    expected_unmet_demand: float
    expected_overflow: float
    expected_penalty_cost: float


def _build_network_recourse_model(
    instance: NetworkDesignInstance,
    scenario: NetworkScenario,
    x: list[int],
    *,
    name: str,
    time_limit: float | None = None,
    threads: int = 1,
) -> tuple[Model, dict[tuple[int, int], object], dict[tuple[int, int], object], list, list, list]:
    model = Model(name)
    configure_scip(model, time_limit=time_limit, threads=threads)
    # Official PySCIPOpt guidance for reliable dual extraction:
    # disable presolving, heuristics, and propagation so the LP solver
    # works on the unmodified problem.
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setParam("presolving/maxrounds", 0)

    flow_forward: dict[tuple[int, int], object] = {}
    flow_backward: dict[tuple[int, int], object] = {}
    unmet = []
    overflow = []

    for e, _edge in enumerate(instance.edges):
        overflow.append(model.addVar(lb=0.0, vtype="C", name=f"overflow_{e}"))
        for k, _commodity in enumerate(scenario.commodities):
            flow_forward[(e, k)] = model.addVar(lb=0.0, vtype="C", name=f"f_{e}_{k}_uv")
            flow_backward[(e, k)] = model.addVar(lb=0.0, vtype="C", name=f"f_{e}_{k}_vu")

    for k, _commodity in enumerate(scenario.commodities):
        unmet.append(model.addVar(lb=0.0, vtype="C", name=f"unmet_{k}"))

    for k, commodity in enumerate(scenario.commodities):
        for node in range(instance.n_nodes):
            expr = quicksum(flow_forward[(e, k)] for e, (u, _v) in enumerate(instance.edges) if u == node)
            expr += quicksum(flow_backward[(e, k)] for e, (_u, v) in enumerate(instance.edges) if v == node)
            expr -= quicksum(flow_backward[(e, k)] for e, (u, _v) in enumerate(instance.edges) if u == node)
            expr -= quicksum(flow_forward[(e, k)] for e, (_u, v) in enumerate(instance.edges) if v == node)
            if node == commodity.source:
                model.addCons(expr + unmet[k] == commodity.demand, name=f"bal_src_{k}_{node}")
            elif node == commodity.sink:
                model.addCons(expr - unmet[k] == -commodity.demand, name=f"bal_sink_{k}_{node}")
            else:
                model.addCons(expr == 0.0, name=f"bal_mid_{k}_{node}")

    capacity_cons = []
    for e, (_u, _v) in enumerate(instance.edges):
        usage = quicksum(flow_forward[(e, k)] + flow_backward[(e, k)] for k, _commodity in enumerate(scenario.commodities))
        rhs = instance.capacities[e] * float(x[e])
        cons = model.addCons(usage - overflow[e] <= rhs, name=f"cap_{e}")
        capacity_cons.append(cons)

    return model, flow_forward, flow_backward, unmet, overflow, capacity_cons


def solve_network_phase1_scip(
    instance: NetworkDesignInstance,
    scenario: NetworkScenario,
    x: list[int],
    *,
    time_limit: float | None = None,
    threads: int = 1,
) -> NetworkSubproblemResult:
    start = time.perf_counter()
    model, _ff, _fb, unmet, overflow, capacity_cons = _build_network_recourse_model(
        instance,
        scenario,
        x,
        name="network_recourse_phase1",
        time_limit=time_limit,
        threads=threads,
    )
    model.setObjective(quicksum(unmet) + quicksum(overflow), sense="minimize")
    model.optimize()
    status = str(model.getStatus()).lower()
    if status != "optimal":
        return NetworkSubproblemResult(False, math.inf, 0.0, [], time.perf_counter() - start)

    cap_duals = [float(model.getDualsolLinear(cons)) for cons in capacity_cons]
    coeffs = [instance.capacities[e] * cap_duals[e] for e in range(instance.n_edges)]
    obj = float(model.getObjVal())
    const = obj - dot(coeffs, [float(v) for v in x])
    unmet_total = sum(float(model.getVal(var)) for var in unmet)
    overflow_total = sum(float(model.getVal(var)) for var in overflow)
    return NetworkSubproblemResult(
        feasible=obj <= 1.0e-7,
        objective=obj,
        const=const,
        coeffs=coeffs,
        solve_time=time.perf_counter() - start,
        unmet_demand=unmet_total,
        overflow=overflow_total,
    )


def solve_network_phase2_scip(
    instance: NetworkDesignInstance,
    scenario: NetworkScenario,
    x: list[int],
    *,
    time_limit: float | None = None,
    threads: int = 1,
) -> NetworkSubproblemResult:
    start = time.perf_counter()
    model, flow_forward, flow_backward, unmet, overflow, capacity_cons = _build_network_recourse_model(
        instance,
        scenario,
        x,
        name="network_recourse_phase2",
        time_limit=time_limit,
        threads=threads,
    )
    for k, var in enumerate(unmet):
        model.addCons(var == 0.0, name=f"fix_unmet_{k}")
    for e, var in enumerate(overflow):
        model.addCons(var == 0.0, name=f"fix_overflow_{e}")
    objective = quicksum(
        scenario.flow_costs[e] * (flow_forward[(e, k)] + flow_backward[(e, k)])
        for e in range(instance.n_edges)
        for k, _commodity in enumerate(scenario.commodities)
    )
    model.setObjective(objective, sense="minimize")
    model.optimize()
    status = str(model.getStatus()).lower()
    if status != "optimal":
        return NetworkSubproblemResult(False, math.inf, 0.0, [], time.perf_counter() - start)

    by_name = {cons.name: cons for cons in model.getConss()}
    resolved_capacity_cons = []
    for e, cons in enumerate(capacity_cons):
        cons_name = f"cap_{e}"
        if cons_name in by_name:
            resolved_capacity_cons.append(by_name[cons_name])
        else:
            raise RuntimeError(f"phase2 missing transformed capacity constraint {cons_name}")

    cap_duals = [float(model.getDualsolLinear(cons)) for cons in resolved_capacity_cons]
    coeffs = [instance.capacities[e] * cap_duals[e] for e in range(instance.n_edges)]
    obj = float(model.getObjVal())
    const = obj - dot(coeffs, [float(v) for v in x])
    unmet_total = sum(float(model.getVal(var)) for var in unmet)
    overflow_total = sum(float(model.getVal(var)) for var in overflow)
    return NetworkSubproblemResult(
        feasible=True,
        objective=obj,
        const=const,
        coeffs=coeffs,
        solve_time=time.perf_counter() - start,
        unmet_demand=unmet_total,
        overflow=overflow_total,
    )


def solve_network_dual_scip(
    instance: NetworkDesignInstance,
    scenario: NetworkScenario,
    x: list[int],
    *,
    time_limit: float | None = None,
    threads: int = 1,
) -> NetworkSubproblemResult:
    """Backward-compatible wrapper for the phase-2 optimality LP."""
    return solve_network_phase2_scip(
        instance,
        scenario,
        x,
        time_limit=time_limit,
        threads=threads,
    )


@dataclass
class NetworkScipBendersSolver:
    instance: NetworkDesignInstance
    strategy: CutSelectionStrategy
    max_iters: int = 100
    tol: float = 1.0e-5
    log_dir: Path | None = None
    scip_time_limit: float | None = None
    threads: int = 1
    optimality_cuts: list[BendersCut] = field(default_factory=list)
    last_master_polyhedral_state: MasterPolyhedralState | None = None
    master_model: Model | None = field(default=None, init=False, repr=False)
    master_x_vars: list = field(default_factory=list, init=False, repr=False)
    master_theta_vars: list = field(default_factory=list, init=False, repr=False)
    master_feas_cuts_loaded: int = field(default=0, init=False, repr=False)
    master_cuts_loaded: int = field(default=0, init=False, repr=False)
    feasibility_cuts: list[FeasibilityCut] = field(default_factory=list)

    def _create_master_model(
        self,
        *,
        relax_x: bool,
    ) -> tuple[Model, list, list]:
        model = Model("network_benders_master_lp" if relax_x else "network_benders_master")
        configure_scip(model, time_limit=self.scip_time_limit, threads=self.threads)

        n = self.instance.n_edges
        s_count = self.instance.n_scenarios
        x_vtype = "C" if relax_x else "B"
        x = [model.addVar(lb=0.0, ub=1.0, vtype=x_vtype, name=f"x_{e}") for e in range(n)]
        theta = [model.addVar(lb=0.0, vtype="C", name=f"theta_{s}") for s in range(s_count)]

        model.addCons(quicksum(x) >= self.instance.n_nodes - 1, name="min_edges")

        obj = quicksum(self.instance.fixed_costs[e] * x[e] for e in range(n))
        obj += quicksum(self.instance.scenarios[s].probability * theta[s] for s in range(s_count))
        model.setObjective(obj, sense="minimize")
        return model, x, theta

    def _ensure_master_models(self) -> None:
        if self.master_model is None:
            self.master_model, self.master_x_vars, self.master_theta_vars = self._create_master_model(
                relax_x=False
            )

    def _sync_cuts_to_model(
        self,
        model: Model,
        x_vars: list,
        theta_vars: list,
        *,
        feas_loaded_attr: str,
        loaded_attr: str,
    ) -> None:
        feas_loaded = int(getattr(self, feas_loaded_attr))
        loaded = int(getattr(self, loaded_attr))
        need_feas = feas_loaded < len(self.feasibility_cuts)
        need_opt = loaded < len(self.optimality_cuts)
        if not need_feas and not need_opt:
            return

        model.freeTransform()
        for cut_idx in range(feas_loaded, len(self.feasibility_cuts)):
            cut = self.feasibility_cuts[cut_idx]
            model.addCons(
                quicksum(cut.coeffs[e] * x_vars[e] for e in range(self.instance.n_edges)) >= cut.rhs,
                name=f"feas_cut_{cut_idx}",
            )
        for cut_idx in range(loaded, len(self.optimality_cuts)):
            cut = self.optimality_cuts[cut_idx]
            model.addCons(
                theta_vars[cut.scenario]
                >= cut.const + quicksum(cut.coeffs[e] * x_vars[e] for e in range(self.instance.n_edges)),
                name=f"opt_cut_s{cut.scenario}_{cut_idx}",
            )
        setattr(self, feas_loaded_attr, len(self.feasibility_cuts))
        setattr(self, loaded_attr, len(self.optimality_cuts))

    def _master_polyhedral_state(
        self,
        master_x: list[int],
        master_theta: list[float],
    ) -> MasterPolyhedralState:
        # LP reduced costs from a repeatedly transformed/reoptimized persistent
        # SCIP LP model were observed to become numerically corrupted
        # intermittently (for example 1e104-scale values while all other
        # quantities stayed moderate). For polyhedral diagnostics we therefore
        # solve a fresh LP master each iteration and read reduced costs from
        # that clean solve.
        lp_model, x_vars, theta_vars = self._create_master_model(relax_x=True)
        for cut_idx, cut in enumerate(self.feasibility_cuts):
            lp_model.addCons(
                quicksum(cut.coeffs[e] * x_vars[e] for e in range(self.instance.n_edges)) >= cut.rhs,
                name=f"feas_cut_{cut_idx}",
            )
        for cut_idx, cut in enumerate(self.optimality_cuts):
            lp_model.addCons(
                theta_vars[cut.scenario]
                >= cut.const + quicksum(cut.coeffs[e] * x_vars[e] for e in range(self.instance.n_edges)),
                name=f"opt_cut_s{cut.scenario}_{cut_idx}",
            )
        lp_model.optimize()
        status = str(lp_model.getStatus()).lower()
        if status != "optimal":
            raise RuntimeError(f"SCIP network LP master did not solve to optimality, status={status}")

        x_tvars = [lp_model.getTransformedVar(var) for var in x_vars]
        theta_tvars = [lp_model.getTransformedVar(var) for var in theta_vars]

        x_lp = [float(lp_model.getVal(var)) for var in x_tvars]
        theta_lp = [max(0.0, float(lp_model.getVal(var))) for var in theta_tvars]
        x_red = [float(lp_model.getVarRedcost(var)) for var in x_tvars]
        theta_red = [float(lp_model.getVarRedcost(var)) for var in theta_tvars]
        lp_objective = float(lp_model.getObjVal())

        if not self.optimality_cuts:
            return MasterPolyhedralState(
                x_lp=x_lp,
                theta_lp=theta_lp,
                x_reduced_costs=x_red,
                theta_reduced_costs=theta_red,
                lp_objective=lp_objective,
                active_cut_ratio=0.0,
                near_binding_cut_ratio=0.0,
                avg_cut_slack=0.0,
                min_cut_slack=0.0,
                cut_slack_cv=0.0,
                scenario_cut_imbalance=0.0,
                active_cut_scenario_ratios=[0.0 for _ in range(self.instance.n_scenarios)],
            )

        slacks: list[float] = []
        normalized_slacks: list[float] = []
        active_counts = [0 for _ in range(self.instance.n_scenarios)]
        near_binding = 0
        for cut in self.optimality_cuts:
            lhs = master_theta[cut.scenario]
            rhs = cut.const + dot(cut.coeffs, [float(v) for v in master_x])
            slack = lhs - rhs
            slacks.append(slack)
            slack_scale = max(1.0, abs(lhs), abs(rhs))
            normalized_slacks.append(slack / slack_scale)
            if abs(slack) <= 1.0e-6:
                active_counts[cut.scenario] += 1
            if slack <= 1.0e-4:
                near_binding += 1

        active_total = sum(active_counts)
        active_cut_ratio = active_total / max(1, len(self.optimality_cuts))
        near_binding_cut_ratio = near_binding / max(1, len(self.optimality_cuts))
        avg_cut_slack = sum(normalized_slacks) / max(1, len(normalized_slacks))
        min_cut_slack = min(normalized_slacks)
        variance = sum((slack - avg_cut_slack) ** 2 for slack in normalized_slacks) / max(1, len(normalized_slacks))
        slack_std = math.sqrt(variance)
        cut_slack_cv = slack_std / max(abs(avg_cut_slack), 1.0e-9)
        active_ratios = [count / max(1, len(self.optimality_cuts)) for count in active_counts]
        imbalance_center = sum(active_counts) / max(1, len(active_counts))
        imbalance = math.sqrt(
            sum((count - imbalance_center) ** 2 for count in active_counts) / max(1, len(active_counts))
        ) / max(1.0, imbalance_center)

        return MasterPolyhedralState(
            x_lp=x_lp,
            theta_lp=theta_lp,
            x_reduced_costs=x_red,
            theta_reduced_costs=theta_red,
            lp_objective=lp_objective,
            active_cut_ratio=active_cut_ratio,
            near_binding_cut_ratio=near_binding_cut_ratio,
            avg_cut_slack=avg_cut_slack,
            min_cut_slack=min_cut_slack,
            cut_slack_cv=cut_slack_cv,
            scenario_cut_imbalance=imbalance,
            active_cut_scenario_ratios=active_ratios,
        )

    def solve_master(self) -> MasterSolution:
        self._ensure_master_models()
        if self.master_model is None:
            raise RuntimeError("persistent master was not initialized")
        start = time.perf_counter()
        model = self.master_model
        x = self.master_x_vars
        theta = self.master_theta_vars
        self._sync_cuts_to_model(
            model,
            x,
            theta,
            feas_loaded_attr="master_feas_cuts_loaded",
            loaded_attr="master_cuts_loaded",
        )
        model.optimize()

        status = str(model.getStatus()).lower()
        if status != "optimal":
            raise RuntimeError(f"SCIP network master did not solve to optimality, status={status}")

        master_x = [1 if model.getVal(var) >= 0.5 else 0 for var in x]
        master_theta = [max(0.0, model.getVal(var)) for var in theta]
        self.last_master_polyhedral_state = self._master_polyhedral_state(
            master_x,
            master_theta,
        )

        return MasterSolution(
            x=master_x,
            theta=master_theta,
            objective=model.getObjVal(),
            solve_time=time.perf_counter() - start,
            x_lp=list(self.last_master_polyhedral_state.x_lp),
            theta_lp=list(self.last_master_polyhedral_state.theta_lp),
            x_reduced_costs_lp=list(self.last_master_polyhedral_state.x_reduced_costs),
            theta_reduced_costs_lp=list(self.last_master_polyhedral_state.theta_reduced_costs),
            lp_objective=self.last_master_polyhedral_state.lp_objective,
        )

    def solve_subproblems(
        self, master: MasterSolution
    ) -> tuple[bool, float, list[CutCandidate], PenaltyStats, float]:
        candidates: list[CutCandidate] = []
        expected_recourse = 0.0
        subproblem_time = 0.0
        disconnected_scenarios = 0
        penalized_scenarios = 0
        expected_unmet_demand = 0.0
        expected_overflow = 0.0
        expected_penalty_cost = 0.0
        expected_total_demand = 0.0
        seen_feasibility_keys = {
            (round(cut.rhs, 12), tuple(round(value, 12) for value in cut.coeffs))
            for cut in self.feasibility_cuts
        }

        for s, scenario in enumerate(self.instance.scenarios):
            disconnection_cut = disconnected_component_cut(self.instance, master.x, scenario)
            if disconnection_cut is not None:
                disconnected_scenarios += 1
                key = (
                    round(disconnection_cut.rhs, 12),
                    tuple(round(value, 12) for value in disconnection_cut.coeffs),
                )
                if key not in seen_feasibility_keys:
                    self.feasibility_cuts.append(disconnection_cut)
                    seen_feasibility_keys.add(key)
                scenario_total_demand = sum(comm.demand for comm in scenario.commodities)
                expected_total_demand += scenario.probability * scenario_total_demand
                expected_unmet_demand += scenario.probability * scenario_total_demand
                expected_penalty_cost += scenario.probability * scenario_total_demand
                continue
            phase1 = solve_network_phase1_scip(
                self.instance,
                scenario,
                master.x,
                time_limit=self.scip_time_limit,
                threads=self.threads,
            )
            subproblem_time += phase1.solve_time
            scenario_total_demand = sum(comm.demand for comm in scenario.commodities)
            expected_total_demand += scenario.probability * scenario_total_demand
            expected_unmet_demand += scenario.probability * phase1.unmet_demand
            expected_overflow += scenario.probability * phase1.overflow
            expected_penalty_cost += scenario.probability * phase1.objective
            if phase1.objective > EPS:
                penalized_scenarios += 1
                capacity_cut = FeasibilityCut(coeffs=[-value for value in phase1.coeffs], rhs=phase1.const)
                key = (
                    round(capacity_cut.rhs, 12),
                    tuple(round(value, 12) for value in capacity_cut.coeffs),
                )
                if key not in seen_feasibility_keys:
                    self.feasibility_cuts.append(capacity_cut)
                    seen_feasibility_keys.add(key)
                continue
            result = solve_network_dual_scip(
                self.instance,
                scenario,
                master.x,
                time_limit=self.scip_time_limit,
                threads=self.threads,
            )
            subproblem_time += result.solve_time
            if not result.feasible:
                raise RuntimeError(f"phase-2 recourse LP failed for scenario {s}")
            expected_recourse += scenario.probability * result.objective
            value = result.const + dot(result.coeffs, [float(v) for v in master.x])
            candidates.append(
                CutCandidate(
                    scenario=s,
                    const=result.const,
                    coeffs=result.coeffs,
                    cut_type="network_dual_optimality",
                    value_at_x=value,
                    theta_value=master.theta[s],
                )
            )

        demand_scale = max(1.0, expected_total_demand)
        recourse_scale = max(1.0, abs(expected_recourse), expected_penalty_cost)
        penalty_stats = PenaltyStats(
            disconnected_scenario_ratio=disconnected_scenarios / max(1, self.instance.n_scenarios),
            penalty_scenario_ratio=penalized_scenarios / max(1, self.instance.n_scenarios),
            unmet_demand_ratio=expected_unmet_demand / demand_scale,
            overflow_ratio=expected_overflow / demand_scale,
            penalty_cost_ratio=expected_penalty_cost / recourse_scale,
            expected_unmet_demand=expected_unmet_demand,
            expected_overflow=expected_overflow,
            expected_penalty_cost=expected_penalty_cost,
        )
        feasible = disconnected_scenarios == 0 and penalized_scenarios == 0
        return feasible, expected_recourse, candidates, penalty_stats, subproblem_time

    def active_cuts_by_scenario(self) -> dict[int, list[list[float]]]:
        active: dict[int, list[list[float]]] = {}
        for cut in self.optimality_cuts:
            active.setdefault(cut.scenario, []).append(cut.coeffs)
        return active

    def run(self) -> BendersResult:
        start = time.perf_counter()
        log_rows: list[dict[str, object]] = []
        upper_bound = math.inf
        best_obj = math.inf
        best_x: list[int] | None = None
        last_lb = -math.inf
        stagnation = 0
        status = "iteration_limit"

        for it in range(1, self.max_iters + 1):
            master = self.solve_master()
            lower_bound = master.objective
            stagnation = stagnation + 1 if lower_bound <= last_lb + 1.0e-7 else 0
            last_lb = max(last_lb, lower_bound)

            feasible, expected_recourse, candidates, penalty_stats, subproblem_time = self.solve_subproblems(master)

            selected: list[CutCandidate] = []
            selection_time = 0.0
            if feasible:
                fixed = dot(self.instance.fixed_costs, [float(v) for v in master.x])
                incumbent = fixed + expected_recourse
                if incumbent < upper_bound:
                    upper_bound = incumbent
                    best_obj = incumbent
                    best_x = list(master.x)

                state = SelectionState(
                    iteration=it,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    cut_count=len(self.optimality_cuts),
                    n_facilities=self.instance.n_edges,
                    n_scenarios=self.instance.n_scenarios,
                    stagnation=stagnation,
                    fixed_costs=self.instance.fixed_costs,
                    active_cuts_by_scenario=self.active_cuts_by_scenario(),
                    instance=self.instance,
                    x=list(master.x),
                    max_iters=self.max_iters,
                    master_solution=master,
                    polyhedral_state=self.last_master_polyhedral_state,
                    disconnected_scenario_ratio=penalty_stats.disconnected_scenario_ratio,
                    penalty_scenario_ratio=penalty_stats.penalty_scenario_ratio,
                    unmet_demand_ratio=penalty_stats.unmet_demand_ratio,
                    overflow_ratio=penalty_stats.overflow_ratio,
                    penalty_cost_ratio=penalty_stats.penalty_cost_ratio,
                )
                sel_start = time.perf_counter()
                selected = self.strategy.select(candidates, state)
                selection_time = time.perf_counter() - sel_start
                for cand in selected:
                    self.optimality_cuts.append(
                        BendersCut(cand.scenario, cand.const, cand.coeffs, cand.cut_type)
                    )

                gap = max(0.0, upper_bound - lower_bound)
                rel_gap = gap / (abs(upper_bound) + 1.0)
                if gap <= self.tol or rel_gap <= self.tol:
                    status = "optimal"
                    log_rows.append(
                        self._row(
                            it,
                            master,
                            lower_bound,
                            upper_bound,
                            len(selected),
                            rel_gap,
                            subproblem_time,
                            selection_time,
                            penalty_stats,
                        )
                    )
                    break
            else:
                rel_gap = math.inf

            log_rows.append(
                self._row(
                    it,
                    master,
                    lower_bound,
                    upper_bound,
                    len(selected),
                    rel_gap,
                    subproblem_time,
                    selection_time,
                    penalty_stats,
                )
            )

        runtime = time.perf_counter() - start
        log_path = ""
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(self.log_dir / f"{self.strategy.name}_network_scip_iterations.csv")
            self._write_log(Path(log_path), log_rows)
            master_time = sum(float(row.get("master_solve_time", 0.0)) for row in log_rows)
            subproblem_time = sum(float(row.get("subproblem_solve_time", 0.0)) for row in log_rows)
            selection_time = sum(float(row.get("selection_time", 0.0)) for row in log_rows)
            (self.log_dir / f"{self.strategy.name}_network_scip_summary.json").write_text(
                json.dumps(
                    {
                        "backend": "network_scip",
                        "strategy": self.strategy.name,
                        "status": status,
                        "objective": best_obj,
                        "lower_bound": log_rows[-1]["lower_bound"] if log_rows else math.nan,
                        "upper_bound": upper_bound,
                        "iterations": len(log_rows),
                        "cuts_added": len(self.optimality_cuts),
                        "best_x": best_x,
                        "runtime": runtime,
                        "master_time": master_time,
                        "subproblem_time": subproblem_time,
                        "selection_time": selection_time,
                        "avg_disconnected_scenario_ratio": sum(float(row.get("disconnected_scenario_ratio", 0.0)) for row in log_rows) / max(1, len(log_rows)),
                        "avg_penalty_scenario_ratio": sum(float(row.get("penalty_scenario_ratio", 0.0)) for row in log_rows) / max(1, len(log_rows)),
                        "avg_unmet_demand_ratio": sum(float(row.get("unmet_demand_ratio", 0.0)) for row in log_rows) / max(1, len(log_rows)),
                        "avg_overflow_ratio": sum(float(row.get("overflow_ratio", 0.0)) for row in log_rows) / max(1, len(log_rows)),
                        "avg_penalty_cost_ratio": sum(float(row.get("penalty_cost_ratio", 0.0)) for row in log_rows) / max(1, len(log_rows)),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        return BendersResult(
            strategy=self.strategy.name,
            status=status,
            objective=best_obj,
            lower_bound=float(log_rows[-1]["lower_bound"]) if log_rows else math.nan,
            upper_bound=upper_bound,
            iterations=len(log_rows),
            cuts_added=len(self.optimality_cuts),
            runtime=runtime,
            log_path=log_path,
        )

    def _row(
        self,
        iteration: int,
        master: MasterSolution,
        lower_bound: float,
        upper_bound: float,
        opt_cuts_added: int,
        rel_gap: float,
        subproblem_time: float,
        selection_time: float,
        penalty_stats: PenaltyStats,
    ) -> dict[str, object]:
        return {
            "iteration": iteration,
            "master_objective": master.objective,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "relative_gap": rel_gap,
            "x": "".join(str(v) for v in master.x),
            "theta_sum": sum(master.theta),
            "master_solve_time": master.solve_time,
            "subproblem_solve_time": subproblem_time,
            "selection_time": selection_time,
            "feasibility_cuts_total": len(self.feasibility_cuts),
            "optimality_cuts_total": len(self.optimality_cuts),
            "optimality_cuts_added": opt_cuts_added,
            "disconnected_scenario_ratio": penalty_stats.disconnected_scenario_ratio,
            "penalty_scenario_ratio": penalty_stats.penalty_scenario_ratio,
            "unmet_demand_ratio": penalty_stats.unmet_demand_ratio,
            "overflow_ratio": penalty_stats.overflow_ratio,
            "penalty_cost_ratio": penalty_stats.penalty_cost_ratio,
            "expected_unmet_demand": penalty_stats.expected_unmet_demand,
            "expected_overflow": penalty_stats.expected_overflow,
            "expected_penalty_cost": penalty_stats.expected_penalty_cost,
        }

    def _write_log(self, path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
