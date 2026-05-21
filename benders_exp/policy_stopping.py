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
]

STOPPING_FEATURE_NAMES = [
    *[f"quota_{name}" for name in QUOTA_FEATURE_NAMES],
    *POLICY_FEATURE_NAMES,
    *STOPPING_META_FEATURE_NAMES,
]


def _local_score_scale(
    current_score: float,
    next_score: float,
    top_score: float,
    processed_scores: list[float],
) -> float:
    scale = max(abs(current_score), abs(next_score), abs(top_score), 1.0e-6)
    for score in processed_scores:
        scale = max(scale, abs(score))
    return scale


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
    score_scale = _local_score_scale(current_score, next_score, top_score, processed_scores)
    safe_scale = max(score_scale, 1.0e-6)
    features = dict(state_features)
    features.update(candidate_features)
    features.update(
        {
            "stop_rank_ratio": rank_index / total,
            "stop_prefix_ratio": prefix_count / total,
            "stop_selected_ratio": selected_count / total,
            "stop_score": current_score / safe_scale,
            "stop_next_score": next_score / safe_scale,
            "stop_score_gap_next": (current_score - next_score) / safe_scale,
            "stop_score_gap_top": (top_score - current_score) / safe_scale,
            "stop_prev_mean_score": prev_mean / safe_scale,
            "stop_prev_min_score": prev_min / safe_scale,
            "stop_prev_max_score": prev_max / safe_scale,
        }
    )
    return features
