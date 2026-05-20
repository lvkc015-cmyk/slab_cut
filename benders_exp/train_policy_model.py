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


def clean_float(value: str, limit: float = 1.0e6) -> float:
    raw = float(value)
    if not math.isfinite(raw):
        raise ValueError(f"non-finite feature value: {value}")
    if abs(raw) > limit:
        raise ValueError(f"feature magnitude exceeds limit {limit}: {raw}")
    return raw


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

    rank_x: list[list[float]] = []
    rank_y: list[float] = []
    for row_idx, row in enumerate(rows):
        try:
            rank_x.append([clean_float(row[name]) for name in POLICY_FEATURE_NAMES])
            rank_y.append(clean_float(row.get("target_return", "0.0")))
        except ValueError as exc:
            bad_columns = []
            for name in POLICY_FEATURE_NAMES:
                try:
                    clean_float(row[name])
                except ValueError:
                    bad_columns.append((name, row[name]))
            details = ", ".join(f"{name}={value}" for name, value in bad_columns[:8])
            raise ValueError(f"invalid dataset row {row_idx}: {details}") from exc

    stopping_rows = [
        (row_idx, row)
        for row_idx, row in enumerate(rows)
        if int(float(row.get("use_for_stopping", "1"))) > 0
    ]
    stopping_x = []
    stopping_y = []
    for row_idx, row in stopping_rows:
        try:
            stopping_x.append([clean_float(row[name]) for name in STOPPING_FEATURE_NAMES])
            stopping_y.append(1 if int(float(row.get("stop_target", 0.0))) > 0 else 0)
        except ValueError as exc:
            bad_columns = []
            for name in STOPPING_FEATURE_NAMES:
                try:
                    clean_float(row[name])
                except ValueError:
                    bad_columns.append((name, row[name]))
            details = ", ".join(f"{name}={value}" for name, value in bad_columns[:8])
            raise ValueError(f"invalid stopping row {row_idx}: {details}") from exc

    ranking_model = GradientBoostingRegressor(
        n_estimators=160,
        learning_rate=0.05,
        max_depth=3,
        random_state=0,
    )
    ranking_model.fit(rank_x, rank_y)

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
        "ranking_feature_names": POLICY_FEATURE_NAMES,
        "ranking_model": ranking_model,
        "stopping_feature_names": STOPPING_FEATURE_NAMES,
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
