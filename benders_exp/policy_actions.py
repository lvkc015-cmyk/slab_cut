from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .features import POLICY_FEATURE_NAMES, candidate_feature_vector, state_feature_dict
from .policy_stopping import STOPPING_FEATURE_NAMES, build_stopping_feature_dict


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_abs(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom <= 1.0e-12:
        return 0.0
    return abs(dot(a, b) / denom)


@dataclass(frozen=True)
class PolicyDecision:
    selected_count: int
    mean_score: float
    min_score: float


@dataclass(frozen=True)
class PolicySelectionDiagnostics:
    viable_count: int
    threshold: float
    diversity_threshold: float
    max_budget_ratio: float
    hard_cap: int
    unique_scored_count: int
    above_threshold_count: int
    diversity_reject_count: int
    loop_iterations: int
    stop_reason: str
    predicted_quota: int
    quota_score: float


def policy_score_fields(max_rank: int = 64) -> list[str]:
    return [f"cand_score_rank_{idx}" for idx in range(max_rank)]


def probe_score_fields(max_rank: int = 64) -> list[str]:
    fields: list[str] = []
    for idx in range(max_rank):
        fields.append(f"cand_score_rank_{idx}")
        fields.append(f"cand_selected_rank_{idx}")
    return fields


def viable_candidates(candidates: list[Any]) -> list[Any]:
    return [cand for cand in candidates if cand.violation > 1.0e-7]

# 生成状态特征的字典，供后续模型使用
def state_cache(candidates: list[Any], state: Any) -> dict[str, float]:
    instance = state.instance
    x = state.x
    if instance is None or x is None:
        raise ValueError("state_cache requires state.instance and state.x")
    return state_feature_dict(candidates, state, instance, x, max_iters=state.max_iters)

# 
def model_score_one(model: Any, features: list[float]) -> float:
    if hasattr(model, "predict_proba"):
        # 如果模型具有 predict_proba 方法，则调用该方法获取预测概率，并根据概率值返回相应的分数。
        probs = model.predict_proba([features])
        # 如果模型输出的概率列表中每个元素的长度大于 1，则返回第二个元素（通常是正类的概率）；如果长度等于 1，则返回第一个元素（通常是唯一类别的概率）。如果模型没有 predict_proba 方法，则直接调用 predict 方法返回预测值。
        if len(probs) and len(probs[0]) > 1:
            return float(probs[0][1])
        if len(probs) and len(probs[0]) == 1:
            return float(probs[0][0])
    return float(model.predict([features])[0])


def candidate_priority(
    candidate: Any,
    candidates: list[Any],
    state: Any,
    selected: list[Any],
) -> tuple[float, float]:
    instance = state.instance
    x = state.x
    if instance is None or x is None:
        raise ValueError("candidate_priority requires state.instance and state.x")
    feats = candidate_feature_vector(
        candidate,
        candidates,
        state,
        instance,
        x,
        selected,
        max_iters=state.max_iters,
        cached_state_features=state_cache(candidates, state),
    )
    return (candidate.violation, sum(feats))


def default_continuation_selector(candidates: list[Any], state: Any) -> list[Any]:
    viable = viable_candidates(candidates)
    ranked = sorted(
        viable,
        key=lambda cand: candidate_priority(cand, candidates, state, []),
        reverse=True,
    )
    keep = min(len(ranked), len(ranked) // 5)
    selected: list[Any] = []
    for cand in ranked:
        if len(selected) >= keep:
            break
        if all(cosine_abs(cand.coeffs, other.coeffs) <= 0.88 for other in selected):
            selected.append(cand)
    return selected

# 选割
def learned_candidate_selection(
    payload: dict[str, Any], # 包含了模型、特征名称、阈值等信息
    candidates: list[Any], # 当前迭代中生成的候选割列表
    state: Any, # 当前迭代的状态，包括实例信息、当前解向量、最大迭代次数等
) -> tuple[list[Any], list[tuple[Any, float]], PolicySelectionDiagnostics]:
    # 首先通过 viable_candidates 函数过滤出 violation 大于 1.0e-7 的割作为可行候选割。
    viable = viable_candidates(candidates)
    instance = state.instance
    x = state.x
    if instance is None or x is None:
        raise ValueError("learned_candidate_selection requires state.instance and state.x")
    if not viable:
        return [], [], PolicySelectionDiagnostics(
            viable_count=0,
            threshold=0.0,
            diversity_threshold=float(payload["diversity_threshold"]),
            max_budget_ratio=float(payload["max_budget_ratio"]),
            hard_cap=0,
            unique_scored_count=0,
            above_threshold_count=0,
            diversity_reject_count=0,
            loop_iterations=0,
            stop_reason="no_viable",
            predicted_quota=0,
            quota_score=0.0,
        )

    # 生成一个状态特征的缓存字典，供后续模型使用
    cache = state_cache(candidates, state)
    # 从 payload 中读取多样性阈值和最大预算比例，
    diversity = float(payload["diversity_threshold"])
    max_ratio = float(payload["max_budget_ratio"])
    # 计算一个硬性上限 hard_cap，限制最终选择的割数量不超过 viable 数量的 max_ratio 部分
    hard_cap = max(0, min(len(viable), int(round(max_ratio * len(viable)))))

    ranked_features: list[tuple[Any, float, dict[str, float]]] = []
    for cand in viable:
        # 计算该割的特征向量，
        features = candidate_feature_vector(
            cand,
            candidates,
            state,
            instance,
            x,
            [],
            max_iters=state.max_iters,
            cached_state_features=cache,
        )
        # 使用 payload 中的模型对该割进行评分，得到一个分数 score
        score = model_score_one(payload["candidate_model"], features)
        # 将割、分数和特征映射组成一个元组，添加到 ranked_features 列表中。特征映射是一个字典，将特征名称与对应的特征值进行关联，供后续分析使用。
        feature_map = {
            name: value
            for name, value in zip(payload["ranking_feature_names"], features)
        }
        ranked_features.append((cand, score, feature_map))
    ranked_features.sort(key=lambda item: item[1], reverse=True)
    ranked_scores = [(cand, score) for cand, score, _feat in ranked_features]

    stopping_model = payload["stopping_model"]
    stopping_constant = payload["stopping_constant"]
    state_feature_row = {f"quota_{name}": value for name, value in cache.items()}
    selected_prefix: list[Any] = []
    processed_scores: list[float] = []
    diversity_reject_count = 0
    loop_iterations = 0
    stop_reason = "exhausted_candidates"
    stop_probability = 0.0
    top_score = ranked_features[0][1] if ranked_features else 0.0

    for rank_idx, (cand, score, feature_map) in enumerate(ranked_features, start=1):
        if len(selected_prefix) >= hard_cap:
            stop_reason = "hard_cap"
            break
        next_score = ranked_features[rank_idx][1] if rank_idx < len(ranked_features) else 0.0
        stop_features = build_stopping_feature_dict(
            state_features=state_feature_row,
            candidate_features=feature_map,
            rank_index=rank_idx,
            total_candidates=len(ranked_features),
            selected_count=len(selected_prefix),
            processed_scores=processed_scores,
            current_score=score,
            next_score=next_score,
            top_score=top_score,
        )
        stop_vec = [float(stop_features[name]) for name in payload["stopping_feature_names"]]
        if stopping_constant is not None:
            stop_probability = float(stopping_constant)
        elif stopping_model is not None and hasattr(stopping_model, "predict_proba"):
            probs = stopping_model.predict_proba([stop_vec])[0]
            stop_probability = float(probs[1]) if len(probs) > 1 else float(probs[0])
        elif stopping_model is not None:
            stop_probability = float(stopping_model.predict([stop_vec])[0])
        else:
            stop_probability = 0.0
        loop_iterations += 1
        if stop_probability >= 0.5:
            stop_reason = "stopping_rule"
            break
        if all(cosine_abs(cand.coeffs, other.coeffs) <= diversity for other in selected_prefix):
            selected_prefix.append(cand)
        else:
            diversity_reject_count += 1
        processed_scores.append(score)
    else:
        stop_reason = "exhausted_candidates"

    diagnostics = PolicySelectionDiagnostics(
        viable_count=len(viable),
        threshold=0.0,
        diversity_threshold=diversity,
        max_budget_ratio=max_ratio,
        hard_cap=hard_cap,
        unique_scored_count=len(ranked_scores),
        above_threshold_count=sum(1 for _cand, score in ranked_scores if score > 0.0),
        diversity_reject_count=diversity_reject_count,
        loop_iterations=loop_iterations,
        stop_reason=stop_reason,
        predicted_quota=len(selected_prefix),
        quota_score=stop_probability,
    )
    return selected_prefix, ranked_scores, diagnostics


def decision_summary(selected: list[Any], ranked_scores: list[tuple[Any, float]]) -> PolicyDecision:
    if not selected:
        return PolicyDecision(selected_count=0, mean_score=0.0, min_score=0.0)
    score_map = {id(cand): score for cand, score in ranked_scores}
    values = [score_map[id(cand)] for cand in selected]
    return PolicyDecision(
        selected_count=len(selected),
        mean_score=sum(values) / max(1, len(values)),
        min_score=min(values),
    )

