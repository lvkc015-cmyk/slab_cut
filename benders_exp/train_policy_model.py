from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from .features import POLICY_FEATURE_NAMES
from .policy_stopping import STOPPING_FEATURE_NAMES


def read_dataset(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def signed_log1p(value: float) -> float:
    if value >= 0.0:
        return math.log1p(value)
    return -math.log1p(-value)


def standardized_by_query(rows: list[dict[str, str]], value_name: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        query_id = row.get("query_id", "")
        grouped.setdefault(query_id, []).append(clean_float(row.get(value_name, "0.0")))

    stats: dict[str, tuple[float, float]] = {}
    for query_id, values in grouped.items():
        mean_value = sum(values) / max(1, len(values))
        variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
        std_value = math.sqrt(variance)
        stats[query_id] = (mean_value, std_value)

    standardized: list[float] = []
    for row in rows:
        query_id = row.get("query_id", "")
        mean_value, std_value = stats.get(query_id, (0.0, 0.0))
        value = clean_float(row.get(value_name, "0.0"))
        if std_value <= 1.0e-12:
            standardized.append(0.0)
        else:
            standardized.append((value - mean_value) / std_value)
    return standardized


def clean_float(value: str, limit: float = 1.0e6) -> float:
    raw = float(value)
    if not math.isfinite(raw):
        raise ValueError(f"non-finite feature value: {value}")
    if abs(raw) > limit:
        raise ValueError(f"feature magnitude exceeds limit {limit}: {raw}")
    return raw


def nonconstant_feature_names(rows: list[dict[str, str]], names: list[str]) -> list[str]:
    kept: list[str] = []
    for name in names:
        values = [clean_float(row[name]) for row in rows]
        if max(values) != min(values):
            kept.append(name)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate-level ranking and stopping models for network cut control.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models") / "network_policy_model.joblib")
    parser.add_argument("--diversity-threshold", type=float, default=0.90)
    parser.add_argument("--max-budget-ratio", type=float, default=1.00)
    args = parser.parse_args()

    rows = read_dataset(args.dataset)
    if not rows:
        raise ValueError(f"empty dataset: {args.dataset}")

    ranking_feature_names = nonconstant_feature_names(rows, POLICY_FEATURE_NAMES)
    if not ranking_feature_names:
        raise ValueError("all ranking features are constant on the dataset")

    rank_x: list[list[float]] = []
    rank_primary_y = [1 if int(float(row.get("oracle_selected", "0.0"))) > 0 else 0 for row in rows]
    rank_aux_y = standardized_by_query(rows, "target_return")
    for row_idx, row in enumerate(rows):
        try:
            rank_x.append([clean_float(row[name]) for name in ranking_feature_names])
        except ValueError as exc:
            bad_columns = []
            for name in ranking_feature_names:
                try:
                    clean_float(row[name])
                except ValueError:
                    bad_columns.append((name, row[name]))
            details = ", ".join(f"{name}={value}" for name, value in bad_columns[:8])
            raise ValueError(f"invalid dataset row {row_idx}: {details}") from exc

    ranking_model: object | None
    ranking_constant: int | None = None
    if len(set(rank_primary_y)) < 2:
        ranking_model = None
        ranking_constant = sorted(set(rank_primary_y))[0]
    else:
        ranking_model = GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            random_state=0,
        )
        ranking_model.fit(rank_x, rank_primary_y)

    ranking_aux_model: object | None
    ranking_aux_constant: float | None = None
    if not rank_aux_y or max(rank_aux_y) == min(rank_aux_y):
        ranking_aux_model = None
        ranking_aux_constant = rank_aux_y[0] if rank_aux_y else 0.0
    else:
        ranking_aux_model = GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            random_state=1,
        )
        ranking_aux_model.fit(rank_x, rank_aux_y)

    stopping_rows = [
        (row_idx, row)
        for row_idx, row in enumerate(rows)
        if int(float(row.get("use_for_stopping", "1"))) > 0
    ]
    if not stopping_rows:
        raise ValueError("no stopping rows with use_for_stopping=1 in dataset")
    stopping_feature_names = nonconstant_feature_names([row for _row_idx, row in stopping_rows], STOPPING_FEATURE_NAMES)
    if not stopping_feature_names:
        raise ValueError("all stopping features are constant on the stopping subset")
    stopping_x = []
    stopping_y = []
    for row_idx, row in stopping_rows:
        try:
            stopping_x.append([clean_float(row[name]) for name in stopping_feature_names])
            stopping_y.append(1 if int(float(row.get("stop_target", 0.0))) > 0 else 0)
        except ValueError as exc:
            bad_columns = []
            for name in stopping_feature_names:
                try:
                    clean_float(row[name])
                except ValueError:
                    bad_columns.append((name, row[name]))
            details = ", ".join(f"{name}={value}" for name, value in bad_columns[:8])
            raise ValueError(f"invalid stopping row {row_idx}: {details}") from exc

    stopping_model: object | None
    stopping_constant: int | None = None
    if len(set(stopping_y)) < 2:
        stopping_model = None
        stopping_constant = sorted(set(stopping_y))[0]
    else:
        stopping_model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            random_state=101,
        )
        stopping_model.fit(stopping_x, stopping_y)
    payload = {
        "selection_mode": "ranking_stopping",
        "ranking_feature_names": ranking_feature_names,
        "ranking_model": ranking_model,
        "ranking_constant": ranking_constant,
        "ranking_aux_model": ranking_aux_model,
        "ranking_aux_constant": ranking_aux_constant,
        "stopping_feature_names": stopping_feature_names,
        "stopping_model": stopping_model,
        "stopping_constant": stopping_constant,
        "candidate_model": ranking_model,
        "diversity_threshold": float(args.diversity_threshold),
        "max_budget_ratio": float(args.max_budget_ratio),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, args.model_out)
    print(f"model={args.model_out}")


if __name__ == "__main__":
    main()
