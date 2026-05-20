from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def to_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Benders batch summary.csv file.")
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    with args.summary.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.get("backend", ""),
                row.get("n_nodes", row.get("n_customers", "")),
                row.get("n_edges", row.get("n_facilities", "")),
                row.get("n_scenarios", ""),
                row["strategy"],
            )
        ].append(row)

    print("backend,n_nodes,n_edges,n_scenarios,strategy,runs,optimal,avg_iterations,avg_cuts,avg_runtime,avg_master_time,avg_subproblem_time,avg_selection_time,max_abs_error")
    for backend, n_nodes, n_edges, n_scenarios, strategy in sorted(groups):
        items = groups[(backend, n_nodes, n_edges, n_scenarios, strategy)]
        n = len(items)
        optimal = sum(1 for row in items if row["status"] == "optimal")
        avg_iterations = sum(to_float(row["iterations"]) for row in items) / n
        avg_cuts = sum(to_float(row["cuts_added"]) for row in items) / n
        avg_runtime = sum(to_float(row["runtime"]) for row in items) / n
        avg_master_time = sum(to_float(row.get("master_time", "")) for row in items) / n
        avg_subproblem_time = sum(to_float(row.get("subproblem_time", "")) for row in items) / n
        avg_selection_time = sum(to_float(row.get("selection_time", "")) for row in items) / n
        max_abs_error = max(to_float(row["abs_error"]) for row in items)
        print(
            f"{backend},{n_nodes},{n_edges},{n_scenarios},{strategy},{n},{optimal},{avg_iterations:.3f},"
            f"{avg_cuts:.3f},{avg_runtime:.6f},{avg_master_time:.6f},{avg_subproblem_time:.6f},{avg_selection_time:.6f},{max_abs_error:.6g}"
        )


if __name__ == "__main__":
    main()
