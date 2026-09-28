"""Two-pass evidence-grounded visual planner for SolveEdit-Plan."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .planner_schema import validate_inspection, validate_plan, validate_planner_input


class PlannerError(RuntimeError):
    """A persistent planner or schema failure."""


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("planner response must be one JSON object")
    return value


def request_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    images: list[Path],
    timeout: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": data_url(path)}} for path in images)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1200]
        raise PlannerError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise PlannerError(f"network error: {exc}") from exc
    message = raw["choices"][0]["message"]
    value = parse_json_content(message.get("content"))
    meta = {key: raw.get(key) for key in ("id", "model", "created", "usage") if key in raw}
    return value, meta


def inspection_prompt(record: dict[str, Any]) -> str:
    schema = {
        "schema_version": "solveedit.plan_inspection.v1",
        "uid": record["uid"],
        "unresolved_variables": [
            {
                "name": "target | action | destination | current_state | rule | preservation",
                "question": "what must be resolved from pixels?",
                "required_source": "binding | state | rule | preservation",
                "why_unresolved": "why the request alone does not determine it",
            }
        ],
        "visual_queries": [
            {
                "query_id": "q1",
                "variable": "variable name",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "query": "concrete question for this crop",
            }
        ],
        "preservation_hypotheses": ["content likely outside the edit scope"],
    }
    return (
        "You are the Inspect stage of SolveEdit-Plan. Determine which executable edit variables "
        "the user request leaves unresolved and what visible evidence must answer them. You have no "
        "benchmark annotation, reference image, rubric, or expected answer. Use only the shown source "
        "image and request. Do not solve the edit yet. Do not infer hidden author intent. Bboxes use "
        "normalized [x1,y1,x2,y2] coordinates and should request informative crops. Return exactly one "
        "JSON object, with at most 6 unresolved variables and 6 visual queries. If language already fixes "
        "the semantic transformation, still identify grounding and preservation evidence.\n\n"
        f"Case ID: {record['case_id']}\nUID: {record['uid']}\n"
        f"User request: {record['instruction']}\n\nRequired schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def resolution_prompt(record: dict[str, Any], inspection: dict[str, Any], crop_labels: list[str]) -> str:
    schema = {
        "schema_version": "solveedit.plan.v1",
        "uid": record["uid"],
        "unresolved_variables": [
            {"name": "string", "status": "resolved | uncertain", "resolution": "string", "claim_ids": ["e1"]}
        ],
        "evidence_ledger": [
            {
                "claim_id": "e1",
                "claim": "visible fact, not a hidden intention",
                "source": "binding | observed_state | in_scene_rule | preservation",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "evidence": "specific visible support",
                "confidence": 0.0,
            }
        ],
        "candidate_transitions": [
            {
                "transition_id": "z1",
                "targets": ["role-based visible target"],
                "actions": ["concrete action"],
                "destination": ["role-based destination or final relation"],
                "goal_state": ["observable postcondition"],
                "preserve": ["non-target content"],
                "supporting_claim_ids": ["e1"],
                "validity": "supported | plausible | rejected",
            }
        ],
        "selected_transition_id": "z1",
        "contrastive_challenge": {
            "alternative_transition_id": "z2 or empty",
            "question": "which visible fact supports z1 and rejects the closest alternative?",
            "discriminating_claim_ids": ["e1"],
            "outcome": "supported | uncertain | not_applicable",
        },
        "self_contract": ["atomic observable obligation inferred from the plan"],
        "compiled_instruction": "concise executable instruction ending in Output only the final image.",
        "uncertainties": [],
    }
    return (
        "You are the Resolve stage of SolveEdit-Plan. Solve the underspecified edit using only the "
        "source image, user request, Inspect record, and supplied crops. Construct role-based candidate "
        "semantic transitions. Every resolved variable and selected action must cite visible claims. "
        "Challenge the selected transition with the closest plausible alternative and identify the visual "
        "evidence that discriminates them. If evidence cannot discriminate, record uncertainty rather than "
        "inventing a fact. Multiple valid outcomes are allowed; select one executable supported transition. "
        "Preserve content outside its scope. The compiled instruction must be <=180 words, must not mention "
        "the benchmark or analysis process, and must end with 'Output only the final image.' Return exactly "
        "one JSON object.\n\n"
        f"Case ID: {record['case_id']}\nUID: {record['uid']}\n"
        f"User request: {record['instruction']}\n"
        f"Inspect record: {json.dumps(inspection, ensure_ascii=False)}\n"
        f"Image order: source image, then {crop_labels or ['no additional crops']}.\n\n"
        f"Required schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def make_crops(image_path: Path, inspection: dict[str, Any], output_dir: Path) -> tuple[list[Path], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels: list[str] = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        for index, query in enumerate(inspection.get("visual_queries", [])[:6], 1):
            bbox = query.get("bbox")
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and 0 <= bbox[0] < bbox[2] <= 1
                and 0 <= bbox[1] < bbox[3] <= 1
            ):
                continue
            left, top, right, bottom = (
                round(bbox[0] * image.width),
                round(bbox[1] * image.height),
                round(bbox[2] * image.width),
                round(bbox[3] * image.height),
            )
            crop = image.crop((left, top, right, bottom))
            path = output_dir / f"crop_{index:02d}.jpg"
            crop.save(path, quality=92)
            paths.append(path)
            labels.append(f"crop {index}: {query.get('query_id', '')} - {query.get('query', '')}")
    return paths, labels


def run_case(
    *,
    record: dict[str, Any],
    case_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    input_issues = validate_planner_input(record)
    if input_issues:
        raise PlannerError(f"invalid planner input: {input_issues}")
    image_path = Path(record["input_image"]).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    inspection: dict[str, Any] | None = None
    inspect_meta: dict[str, Any] = {}
    inspect_error = ""
    for attempt in range(1, retries + 1):
        try:
            candidate, candidate_meta = request_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=inspection_prompt(record),
                images=[image_path],
                timeout=timeout,
                max_tokens=5000,
            )
            issues = validate_inspection(candidate, str(record["uid"]))
            if issues:
                raise PlannerError(f"inspection schema: {issues}")
            inspection, inspect_meta = candidate, candidate_meta
            break
        except Exception as exc:  # noqa: BLE001 - persist provider/schema failures.
            inspect_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    if inspection is None:
        raise PlannerError(inspect_error or "inspection failed")
    (case_dir / "inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    crops, crop_labels = make_crops(image_path, inspection, case_dir / "crops")
    plan: dict[str, Any] | None = None
    resolve_meta: dict[str, Any] = {}
    resolve_error = ""
    for attempt in range(1, retries + 1):
        try:
            candidate, candidate_meta = request_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=resolution_prompt(record, inspection, crop_labels),
                images=[image_path, *crops],
                timeout=timeout,
                max_tokens=9000,
            )
            issues = validate_plan(candidate, str(record["uid"]))
            if issues:
                raise PlannerError(f"plan schema: {issues}")
            plan, resolve_meta = candidate, candidate_meta
            break
        except Exception as exc:  # noqa: BLE001 - persist provider/schema failures.
            resolve_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    if plan is None:
        raise PlannerError(resolve_error or "resolution failed")
    plan["planner_model"] = model
    plan["planner_variant"] = "solveedit_plan"
    plan["planner_call_budget"] = 2
    plan["planner_vision_call_budget"] = 2
    plan["original_instruction"] = record["instruction"]
    plan["inspection_path"] = str((case_dir / "inspection.json").resolve())
    (case_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (case_dir / "response_meta.json").write_text(
        json.dumps({"inspect": inspect_meta, "resolve": resolve_meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan
