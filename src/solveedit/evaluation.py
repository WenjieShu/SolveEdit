"""Deterministic SolveEdit contract weighting, verdict normalization and scoring."""

from __future__ import annotations

from typing import Any


VERDICTS = {"pass", "partial", "fail", "abstain"}


def contract_questions(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Return reward and penalty questions in their canonical source order."""

    if contract.get("schema_version") == "solveedit.atomic_contract.v3":
        return _atomic_v3_questions(contract)

    questions: list[dict[str, Any]] = []
    for field, role, weight_key in (
        ("reward_questions", "reward", "points"),
        ("penalty_questions", "penalty", "max_deduct"),
    ):
        for item in contract.get(field, []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            questions.append(
                {
                    **item,
                    "role": role,
                    "weight": float(item.get(weight_key, 0.0) or 0.0),
                }
            )
    return questions


def _atomic_v3_questions(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten v3 criterion groups using the frozen hierarchical weighting.

    Within each role, applicable properties receive equal mass. Criterion
    groups divide their property's mass equally, and atoms within a group
    share that group's mass. This prevents a mechanical split from increasing
    a requirement's influence.
    """

    groups = [item for item in contract.get("criterion_groups", []) if isinstance(item, dict)]
    group_counts: dict[tuple[str, str], int] = {}
    properties_by_role: dict[str, set[str]] = {}
    for group in groups:
        role = str(group.get("role") or "").strip().lower()
        prop = str(group.get("property") or "").strip().lower()
        atoms = [item for item in group.get("atoms", []) if isinstance(item, dict) and item.get("id")]
        if role not in {"required", "protected"} or not prop or not atoms:
            continue
        group_counts[(role, prop)] = group_counts.get((role, prop), 0) + 1
        properties_by_role.setdefault(role, set()).add(prop)

    questions: list[dict[str, Any]] = []
    for group in groups:
        role = str(group.get("role") or "").strip().lower()
        prop = str(group.get("property") or "").strip().lower()
        atoms = [item for item in group.get("atoms", []) if isinstance(item, dict) and item.get("id")]
        property_count = len(properties_by_role.get(role, set()))
        group_count = group_counts.get((role, prop), 0)
        if role not in {"required", "protected"} or not atoms or not property_count or not group_count:
            continue
        atom_weight = 1.0 / property_count / group_count / len(atoms)
        for atom in atoms:
            questions.append(
                {
                    **atom,
                    "role": "reward" if role == "required" else "penalty",
                    "contract_role": role,
                    "dimension": prop,
                    "property": prop,
                    "weight": atom_weight,
                    "question": atom.get("question_en") or atom.get("question_zh"),
                    "checker": atom.get("checker_family"),
                    "criterion_group_id": group.get("criterion_group_id"),
                    "legacy_parent_id": group.get("legacy_parent_id"),
                }
            )
    return questions


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def normalize_quality_gate(raw: Any) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else {}
    status = str(value.get("status") or "abstain").strip().lower()
    if status not in {"pass", "fail", "abstain"}:
        status = "abstain"
    return {
        "status": status,
        "passed": status == "pass",
        "confidence": _clamp01(value.get("confidence")),
        "reason": str(value.get("reason") or "").strip(),
        "visual_evidence": str(value.get("visual_evidence") or "").strip(),
    }


def normalize_atomic_results(
    raw_results: Any,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate API judgments and derive deterministic trigger scores.

    API verdicts always describe whether the question's *positive criteria*
    pass.  Reward questions therefore map pass/partial/fail to 1/.5/0.
    Penalty questions are positive preservation/constraint questions in the
    current contracts, so their deduction trigger is inverted to 0/.5/1.
    Abstentions never earn reward and never trigger an unproven penalty.
    """

    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict) and item.get("id"):
                by_id[str(item["id"])] = item
    elif isinstance(raw_results, dict):
        for key, item in raw_results.items():
            if isinstance(item, dict):
                by_id[str(key)] = {"id": str(key), **item}

    normalized = []
    for question in contract_questions(contract):
        raw = by_id.get(str(question["id"]), {})
        verdict = str(raw.get("verdict") or raw.get("status") or "abstain").strip().lower()
        if verdict not in VERDICTS:
            verdict = "abstain"
        if question["role"] == "reward":
            trigger_score = {"pass": 1.0, "partial": 0.5, "fail": 0.0, "abstain": 0.0}[verdict]
        else:
            trigger_score = {"pass": 0.0, "partial": 0.5, "fail": 1.0, "abstain": 0.0}[verdict]
        normalized.append(
            {
                "id": question["id"],
                "role": question["role"],
                "dimension": question.get("dimension"),
                "weight": question["weight"],
                "question": question.get("question"),
                "verdict": verdict,
                "trigger_score": trigger_score,
                "confidence": _clamp01(raw.get("confidence")),
                "reason": str(raw.get("reason") or "").strip(),
                "visual_evidence": str(raw.get("visual_evidence") or raw.get("evidence") or "").strip(),
            }
        )
    return normalized


def score_case(
    contract: dict[str, Any],
    quality_gate: dict[str, Any],
    atomic_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen reward-minus-penalty policy and report coverage."""

    rewards = [item for item in atomic_results if item["role"] == "reward"]
    penalties = [item for item in atomic_results if item["role"] == "penalty"]
    reward_budget = sum(max(0.0, float(item["weight"])) for item in rewards)
    penalty_budget = sum(max(0.0, float(item["weight"])) for item in penalties)
    reward_score = (
        sum(item["trigger_score"] * max(0.0, float(item["weight"])) for item in rewards) / reward_budget
        if reward_budget
        else 0.0
    )
    penalty_deduct = sum(
        item["trigger_score"] * max(0.0, float(item["weight"])) for item in penalties
    )
    question_budget = reward_budget + penalty_budget
    covered_budget = sum(
        max(0.0, float(item["weight"]))
        for item in atomic_results
        if item["verdict"] != "abstain"
    )
    coverage = covered_budget / question_budget if question_budget else 0.0
    penalty_lambda = 0.5 if contract.get("schema_version") == "solveedit.atomic_contract.v3" else 1.0
    base_score = max(0.0, min(1.0, reward_score - penalty_lambda * penalty_deduct))
    if quality_gate.get("status") != "pass":
        final_score = 0.0
    else:
        final_score = base_score
    return {
        "policy": str(contract.get("scoring_policy") or "task_success_minus_penalty"),
        "penalty_lambda": penalty_lambda,
        "reward_score": round(reward_score, 6),
        "penalty_deduct": round(penalty_deduct, 6),
        "base_score": round(base_score, 6),
        "final_score": round(final_score, 6),
        "reward_budget": round(reward_budget, 6),
        "penalty_budget": round(penalty_budget, 6),
        "evidence_coverage": round(coverage, 6),
        "abstain_count": sum(item["verdict"] == "abstain" for item in atomic_results),
        "question_count": len(atomic_results),
        "leaderboard_eligible": quality_gate.get("status") == "pass" and coverage >= 0.9,
    }


def normalize_case_response(raw: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    quality_gate = normalize_quality_gate(raw.get("quality_gate"))
    atomic_results = normalize_atomic_results(raw.get("atomic_results"), contract)
    return {
        "quality_gate": quality_gate,
        "atomic_results": atomic_results,
        "scoring": score_case(contract, quality_gate, atomic_results),
        "failure_summary": raw.get("failure_summary", []),
        "overall_reason": str(raw.get("overall_reason") or "").strip(),
    }
