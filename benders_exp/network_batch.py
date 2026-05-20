from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .network import add_instance_generator_args, generate_network_design_instance, instance_generator_kwargs_from_args
from .network_benders_scip import NetworkScipBendersSolver
from .strategies import make_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stochastic network design Benders experiments.")
    parser.add_argument("--strategies", nargs="+", default=["multi", "topk", "slab", "policy"])
    parser.add_argument("--model-path", type=Path, default=Path("runs") / "network_policy_protocol" / "models" / "network_policy_model.joblib")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--n-edges", type=int, default=16)
    parser.add_argument("--n-scenarios", type=int, default=8)
    parser.add_argument("--n-commodities", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("runs") / "network_batch")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scip-time-limit", type=float, default=0.0)
    add_instance_generator_args(parser)
    return parser

# 跑批自动化测试台
def main() -> None:
    args = build_parser().parse_args()
    instance_kwargs = instance_generator_kwargs_from_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()

    for seed in args.seeds:
        instance = generate_network_design_instance(
            n_nodes=args.n_nodes,
            n_edges=args.n_edges,
            n_scenarios=args.n_scenarios,
            n_commodities=args.n_commodities,
            seed=seed,
            **instance_kwargs,
        )
        for strategy_name in args.strategies:
            run_dir = args.out / f"seed_{seed}" / strategy_name
            resolved = strategy_name
            if strategy_name == "policy":
                resolved = f"policy:{args.model_path}"
            strategy = make_strategy(resolved, k=args.k)
            solver = NetworkScipBendersSolver(
                instance=instance,
                strategy=strategy,
                max_iters=args.max_iters,
                tol=args.tol,
                log_dir=run_dir,
                scip_time_limit=args.scip_time_limit if args.scip_time_limit > 0 else None,
                threads=args.threads,
            )
            result = solver.run()
            summary_file = run_dir / f"{strategy.name}_network_scip_summary.json"
            summary = {}
            if summary_file.exists():
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            row = {
                "seed": seed,
                "backend": "network_scip",
                "n_nodes": args.n_nodes,
                "n_edges": args.n_edges,
                "n_scenarios": args.n_scenarios,
                "n_commodities": args.n_commodities,
                "strategy": strategy_name,
                "status": result.status,
                "objective": result.objective,
                "true_objective": "",
                "abs_error": "",
                "lower_bound": result.lower_bound,
                "upper_bound": result.upper_bound,
                "iterations": result.iterations,
                "cuts_added": result.cuts_added,
                "runtime": result.runtime,
                "master_time": summary.get("master_time", ""),
                "subproblem_time": summary.get("subproblem_time", ""),
                "selection_time": summary.get("selection_time", ""),
                "log_path": result.log_path,
            }
            rows.append(row)
            print(
                f"seed={seed} strategy={strategy_name} status={result.status} "
                f"obj={result.objective:.4f} iters={result.iterations} cuts={result.cuts_added}"
            )

    summary_path = args.out / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary={summary_path}")
    print(f"batch_runtime={time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
