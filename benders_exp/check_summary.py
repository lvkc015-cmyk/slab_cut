from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def as_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)

# 定义了一个函数 main，用于检查一个批量实验结果的 summary.csv 文件的一致性。
#它会统计一些基本信息，如行数、分组数、非最优解数量、下界大于上界的情况数量、绝对误差的最大值、可比较的最优解分组数量以及目标值的最大差异等。
# 如果发现任何不一致的情况（如非最优解、下界大于上界、目标值差异过大或绝对误差过大），则会以非零状态退出。
def main() -> None:
    parser = argparse.ArgumentParser(description="Check consistency of a batch summary.csv.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    args = parser.parse_args()

    with args.summary.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    nonoptimal = [r for r in rows if r["status"] != "optimal"]
    lb_gt_ub = [
        r
        for r in rows
        if as_float(r["lower_bound"]) > as_float(r["upper_bound"]) + args.tol
    ]
    abs_errors = [as_float(r.get("abs_error", "")) for r in rows if r.get("abs_error", "") != ""]

    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.get("seed", ""),
                row.get("n_nodes", row.get("n_customers", "")),
                row.get("n_edges", row.get("n_facilities", "")),
                row.get("n_scenarios", ""),
                row.get("n_commodities", ""),
            )
        ].append(row)

    max_spread = 0.0
    worst_key = None
    comparable_groups = 0
    for key, items in groups.items():
        if any(row.get("status", "") != "optimal" for row in items):
            continue
        objectives = [as_float(row["objective"]) for row in items]
        spread = max(objectives) - min(objectives)
        comparable_groups += 1
        if spread > max_spread:
            max_spread = spread
            worst_key = key

    print(f"rows={len(rows)}")
    print(f"groups={len(groups)}")
    print(f"nonoptimal={len(nonoptimal)}")
    print(f"lower_bound_gt_upper_bound={len(lb_gt_ub)}")
    if abs_errors:
        print(f"max_abs_error={max(abs_errors):.6g}")
    print(f"comparable_optimal_groups={comparable_groups}")
    print(f"max_objective_spread={max_spread:.6g}")
    if worst_key is not None:
        print(f"worst_group={worst_key}")

    failed_nonoptimal = bool(nonoptimal)
    failed_abs_error = bool(abs_errors) and max(abs_errors) > args.tol
    if failed_nonoptimal or lb_gt_ub or max_spread > args.tol or failed_abs_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
