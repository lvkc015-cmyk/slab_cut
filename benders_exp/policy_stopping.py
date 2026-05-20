from __future__ import annotations

from .features import POLICY_FEATURE_NAMES, QUOTA_FEATURE_NAMES


STOPPING_META_FEATURE_NAMES = [
    "stop_rank_ratio",
    "stop_prefix_ratio",
    "stop_selected_ratio",
    "stop_score",
    "stop_next_score",
    "stop_score_gap_next",
    "stop_score_gap_top",
    "stop_prev_mean_score",
    "stop_prev_min_score",
    "stop_prev_max_score",
    "stop_candidate_count",
]

STOPPING_FEATURE_NAMES = [
    *[f"quota_{name}" for name in QUOTA_FEATURE_NAMES],
    *POLICY_FEATURE_NAMES,
    *STOPPING_META_FEATURE_NAMES,
]


def build_stopping_feature_dict(
    *,
    state_features: dict[str, float],
    candidate_features: dict[str, float],
    rank_index: int,
    total_candidates: int,
    selected_count: int,
    processed_scores: list[float],
    current_score: float,
    next_score: float,
    top_score: float,
) -> dict[str, float]:
    total = max(1, total_candidates)
    prefix_count = max(0, rank_index - 1)
    if processed_scores:
        prev_mean = sum(processed_scores) / len(processed_scores)
        prev_min = min(processed_scores)
        prev_max = max(processed_scores)
    else:
        prev_mean = 0.0
        prev_min = 0.0
        prev_max = 0.0
    features = dict(state_features)
    features.update(candidate_features)
    features.update(
        {
            "stop_rank_ratio": rank_index / total,
            "stop_prefix_ratio": prefix_count / total,
            "stop_selected_ratio": selected_count / total,
            "stop_score": current_score,
            "stop_next_score": next_score,
            "stop_score_gap_next": current_score - next_score,
            "stop_score_gap_top": top_score - current_score,
            "stop_prev_mean_score": prev_mean,
            "stop_prev_min_score": prev_min,
            "stop_prev_max_score": prev_max,
            "stop_candidate_count": float(total_candidates),
        }
    )
    return features
