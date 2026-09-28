"""Validation for planner records without evaluation-only fields."""

from __future__ import annotations

from typing import Any


FORBIDDEN_PLANNER_KEYS = {
    "contract",
    "contract_path",
    "reward_questions",
    "penalty_questions",
    "manual_facts",
    "intended_final_state",
    "gt_image",
    "reference_image",
    "dependency_regime",
    "dependency_level",
    "visual_dependency_level",
}

REQUIRED_SOURCES = {"binding", "state", "rule", "preservation"}
EVIDENCE_SOURCES = {"binding", "observed_state", "in_scene_rule", "preservation"}


def find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    """Return forbidden key paths from an arbitrary JSON-like object."""

    issues: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_PLANNER_KEYS:
                issues.append(child_path)
            issues.extend(find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return issues


def _list(value: Any, field: str, issues: list[str]) -> list[Any]:
    if not isinstance(value, list):
        issues.append(f"{field}_must_be_list")
        return []
    return value


def _bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and 0 <= value[0] < value[2] <= 1
        and 0 <= value[1] < value[3] <= 1
    )


def validate_planner_input(record: dict[str, Any]) -> list[str]:
    issues = [f"forbidden_key:{path}" for path in find_forbidden_keys(record)]
    for field in ("uid", "case_id", "input_image", "instruction"):
        if not str(record.get(field) or "").strip():
            issues.append(f"missing:{field}")
    return issues


def validate_inspection(value: dict[str, Any], expected_uid: str) -> list[str]:
    issues = [f"forbidden_key:{path}" for path in find_forbidden_keys(value)]
    if value.get("uid") != expected_uid:
        issues.append("uid_mismatch")
    variables = _list(value.get("unresolved_variables"), "unresolved_variables", issues)
    queries = _list(value.get("visual_queries"), "visual_queries", issues)
    _list(value.get("preservation_hypotheses"), "preservation_hypotheses", issues)
    for index, item in enumerate(variables):
        if not isinstance(item, dict):
            issues.append(f"variable_{index}_not_object")
            continue
        if item.get("required_source") not in REQUIRED_SOURCES:
            issues.append(f"variable_{index}_bad_source")
        for field in ("name", "question", "why_unresolved"):
            if not str(item.get(field) or "").strip():
                issues.append(f"variable_{index}_missing:{field}")
    for index, item in enumerate(queries):
        if not isinstance(item, dict):
            issues.append(f"query_{index}_not_object")
            continue
        if not _bbox(item.get("bbox")):
            issues.append(f"query_{index}_bad_bbox")
        if not str(item.get("query") or "").strip():
            issues.append(f"query_{index}_missing_query")
    return issues


def validate_plan(value: dict[str, Any], expected_uid: str) -> list[str]:
    issues = [f"forbidden_key:{path}" for path in find_forbidden_keys(value)]
    if value.get("uid") != expected_uid:
        issues.append("uid_mismatch")
    variables = _list(value.get("unresolved_variables"), "unresolved_variables", issues)
    evidence = _list(value.get("evidence_ledger"), "evidence_ledger", issues)
    candidates = _list(value.get("candidate_transitions"), "candidate_transitions", issues)
    _list(value.get("self_contract"), "self_contract", issues)
    _list(value.get("uncertainties"), "uncertainties", issues)
    for index, item in enumerate(variables):
        if not isinstance(item, dict) or item.get("status") not in {"resolved", "uncertain"}:
            issues.append(f"variable_{index}_bad_status")
    claim_ids: set[str] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            issues.append(f"evidence_{index}_not_object")
            continue
        claim_id = str(item.get("claim_id") or "")
        if not claim_id or claim_id in claim_ids:
            issues.append(f"evidence_{index}_bad_claim_id")
        claim_ids.add(claim_id)
        if item.get("source") not in EVIDENCE_SOURCES:
            issues.append(f"evidence_{index}_bad_source")
        if item.get("bbox") is not None and not _bbox(item.get("bbox")):
            issues.append(f"evidence_{index}_bad_bbox")
    candidate_ids = {
        str(item.get("transition_id"))
        for item in candidates
        if isinstance(item, dict) and item.get("transition_id")
    }
    selected = str(value.get("selected_transition_id") or "")
    if not candidate_ids:
        issues.append("candidate_transitions_empty")
    if selected not in candidate_ids:
        issues.append("selected_transition_missing")
    challenge = value.get("contrastive_challenge")
    if not isinstance(challenge, dict):
        issues.append("contrastive_challenge_missing")
    else:
        if challenge.get("outcome") not in {"supported", "uncertain", "not_applicable"}:
            issues.append("contrastive_challenge_bad_outcome")
        cited = challenge.get("discriminating_claim_ids", [])
        if not isinstance(cited, list) or any(str(item) not in claim_ids for item in cited):
            issues.append("contrastive_challenge_bad_claim_ids")
    instruction = str(value.get("compiled_instruction") or "").strip()
    if not instruction:
        issues.append("compiled_instruction_missing")
    if len(instruction.split()) > 180:
        issues.append("compiled_instruction_over_180_words")
    return issues
