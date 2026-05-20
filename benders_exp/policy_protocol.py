from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from .collect_network_policy_data import parse_sizes
from .network import add_instance_generator_args, instance_generator_cli_args


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def merge_summaries(paths: list[Path], out: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    if fieldnames is None:
        return
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the structured network policy protocol.")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[20, 21, 22, 23, 24])
    parser.add_argument("--train-sizes", nargs="+", default=["8,16,6,4", "9,22,8,5", "10,26,10,6"])
    parser.add_argument("--test-sizes", nargs="+", default=["9,22,8,5", "10,26,10,6"])
    parser.add_argument("--max-iters", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--rollout-horizon", type=int, default=3)
    parser.add_argument("--rollout-gamma", type=float, default=0.90)
    parser.add_argument("--cut-penalty", type=float, default=0.002)
    parser.add_argument("--time-penalty-weight", type=float, default=0.005)
    parser.add_argument("--master-time-penalty-weight", type=float, default=0.0)
    parser.add_argument("--peak-master-time-penalty-weight", type=float, default=0.0)
    parser.add_argument("--pressure-penalty-weight", type=float, default=0.0)
    parser.add_argument("--near-binding-penalty-weight", type=float, default=0.0)
    parser.add_argument("--diversity-threshold", type=float, default=0.90)
    parser.add_argument("--max-budget-ratio", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scip-time-limit", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("runs") / "network_policy_protocol")
    add_instance_generator_args(parser)
    args = parser.parse_args()
    train_sizes = parse_sizes(args.train_sizes)
    test_sizes = parse_sizes(args.test_sizes)

    args.out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    training_dir = args.out / "training_data"
    models_dir = args.out / "models"
    probes_dir = args.out / "probes"
    summary_dir = args.out / "summary"
    batch_root = args.out / "batch_logs"
    for path in (training_dir, models_dir, probes_dir, summary_dir, batch_root):
        path.mkdir(parents=True, exist_ok=True)

    data_path = training_dir / "network_policy_training.csv"
    model_path = models_dir / "network_policy_model.joblib"
    merged_summary = summary_dir / "summary_all.csv"

    collect_cmd = [
        py,
        "-m",
        "benders_exp.collect_network_policy_data",
        "--seeds",
        *[str(seed) for seed in args.train_seeds],
        "--sizes",
        *[f"{n_nodes},{n_edges},{n_scenarios},{n_commodities}" for n_nodes, n_edges, n_scenarios, n_commodities in train_sizes],
        "--max-iters",
        str(args.max_iters),
        "--tol",
        str(args.tol),
        "--rollout-horizon",
        str(args.rollout_horizon),
        "--rollout-gamma",
        str(args.rollout_gamma),
        "--cut-penalty",
        str(args.cut_penalty),
        "--time-penalty-weight",
        str(args.time_penalty_weight),
        "--master-time-penalty-weight",
        str(args.master_time_penalty_weight),
        "--peak-master-time-penalty-weight",
        str(args.peak_master_time_penalty_weight),
        "--pressure-penalty-weight",
        str(args.pressure_penalty_weight),
        "--near-binding-penalty-weight",
        str(args.near_binding_penalty_weight),
        "--threads",
        str(args.threads),
        "--out",
        str(data_path),
        *instance_generator_cli_args(args),
    ]
    if args.scip_time_limit > 0:
        collect_cmd.extend(["--scip-time-limit", str(args.scip_time_limit)])
    run(collect_cmd)

    train_cmd = [
        py,
        "-m",
        "benders_exp.train_policy_model",
        "--dataset",
        str(data_path),
        "--model-out",
        str(model_path),
        "--diversity-threshold",
        str(args.diversity_threshold),
        "--max-budget-ratio",
        str(args.max_budget_ratio),
    ]
    run(train_cmd)

    probe_seed = args.train_seeds[0]
    probe_n_nodes, probe_n_edges, probe_n_scenarios, probe_n_commodities = train_sizes[0]

    probe_cmd = [
        py,
        "-m",
        "benders_exp.probe_policy_oracle",
        "--seed",
        str(probe_seed),
        "--n-nodes",
        str(probe_n_nodes),
        "--n-edges",
        str(probe_n_edges),
        "--n-scenarios",
        str(probe_n_scenarios),
        "--n-commodities",
        str(probe_n_commodities),
        "--max-iters",
        str(min(args.max_iters, 40)),
        "--rollout-horizon",
        str(args.rollout_horizon),
        "--rollout-gamma",
        str(args.rollout_gamma),
        "--cut-penalty",
        str(args.cut_penalty),
        "--time-penalty-weight",
        str(args.time_penalty_weight),
        "--master-time-penalty-weight",
        str(args.master_time_penalty_weight),
        "--peak-master-time-penalty-weight",
        str(args.peak_master_time_penalty_weight),
        "--pressure-penalty-weight",
        str(args.pressure_penalty_weight),
        "--near-binding-penalty-weight",
        str(args.near_binding_penalty_weight),
        "--threads",
        str(args.threads),
        "--model-path",
        str(model_path),
        "--out",
        str(probes_dir / f"probe_train_seed{probe_seed}"),
        *instance_generator_cli_args(args),
    ]
    if args.scip_time_limit > 0:
        probe_cmd.extend(["--scip-time-limit", str(args.scip_time_limit)])
    run(probe_cmd)

    summary_paths: list[Path] = []
    for n_nodes, n_edges, n_scenarios, n_commodities in test_sizes:
        batch_dir = batch_root / f"n{n_nodes}_e{n_edges}_s{n_scenarios}_k{n_commodities}"
        batch_cmd = [
            py,
            "-m",
            "benders_exp.network_batch",
            "--seeds",
            *[str(seed) for seed in args.test_seeds],
            "--strategies",
            "multi",
            "topk",
            "slab",
            "policy",
            "--model-path",
            str(model_path),
            "--n-nodes",
            str(n_nodes),
            "--n-edges",
            str(n_edges),
            "--n-scenarios",
            str(n_scenarios),
            "--n-commodities",
            str(n_commodities),
            "--max-iters",
            str(args.max_iters),
            "--tol",
            str(args.tol),
            "--threads",
            str(args.threads),
            "--out",
            str(batch_dir),
            *instance_generator_cli_args(args),
        ]
        if args.scip_time_limit > 0:
            batch_cmd.extend(["--scip-time-limit", str(args.scip_time_limit)])
        run(batch_cmd)
        summary_paths.append(batch_dir / "summary.csv")

    merge_summaries(summary_paths, merged_summary)
    run([py, "-m", "benders_exp.summarize", str(merged_summary)])
    run([py, "-m", "benders_exp.check_summary", str(merged_summary)])

    print(f"dataset={data_path}")
    print(f"model={model_path}")
    print(f"summary={merged_summary}")


if __name__ == "__main__":
    main()
