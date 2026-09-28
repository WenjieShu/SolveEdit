#!/usr/bin/env python3
"""Evaluate edited images against user-supplied SolveEdit atomic contracts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from solveedit.evaluation import contract_questions, normalize_case_response


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except ImportError:
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def question_payload(contract: dict[str, Any]) -> list[dict[str, Any]]:
    payload = []
    for item in contract_questions(contract):
        payload.append(
            {
                "id": item["id"],
                "role": item["role"],
                "weight": item["weight"],
                "dimension": item.get("dimension"),
                "question": item.get("question"),
                "pass_criteria": item.get("pass_criteria"),
                "partial_criteria": item.get("partial_criteria"),
                "fail_criteria": item.get("fail_criteria"),
                "checker_family": item.get("checker"),
                "evidence_required": item.get("evidence_required", []),
            }
        )
    return payload


def build_prompt(contract: dict[str, Any]) -> str:
    manual_facts = contract.get("manual_facts", {})
    context = {
        "case_id": contract.get("case_id"),
        "case_family": contract.get("case_family"),
        "task_family": contract.get("task_family"),
        "instruction": contract.get("instruction"),
        "intended_final_state": contract.get("intended_final_state") or manual_facts.get("intended_final_state"),
        "important_entities": manual_facts.get("important_entities", []),
        "allowed_changes": manual_facts.get("allowed_changes", []),
        "forbidden_changes": manual_facts.get("forbidden_changes", []),
        "reasoning_profile": contract.get("reasoning_profile", {}),
    }
    schema = {
        "quality_gate": {
            "status": "pass | fail | abstain",
            "confidence": 0.0,
            "reason": "short reason",
            "visual_evidence": "visible evidence",
        },
        "atomic_results": [
            {
                "id": "exact question id",
                "verdict": "pass | partial | fail | abstain",
                "confidence": 0.0,
                "reason": "apply the supplied criteria",
                "visual_evidence": "specific visible objects, positions, text, or state",
            }
        ],
        "failure_summary": [{"question_id": "id", "type": "short taxonomy", "severity": "minor | major | critical"}],
        "overall_reason": "brief case-level summary",
    }
    return (
        "You are the evaluator for a SolveEdit image-edit case.\n"
        "IMAGE 1 is the original INPUT. IMAGE 2 is the MODEL OUTPUT to score. "
        "No target reference is supplied; use the input, instruction, and contract.\n\n"
        "First apply the quality gate. Fail it only when the output is missing/blank, severely corrupted, unrelated to the input, or no longer a plausible edit of the input. A merely incorrect edit must pass the quality gate and fail its atomic questions.\n"
        "Then answer every atomic question exactly once. Verdicts describe whether the positive pass criteria are satisfied: pass, partial, fail, or abstain. Use abstain only when the visible evidence is genuinely insufficient or the contract contains an unresolved fact.\n"
        "Be strict about counts, labels, routes, exact object-zone relations, missing required entities, and unintended edits. Inspect the input and output comparatively. Do not give a global 'looks good' judgment.\n"
        "Use the supplied contract criteria directly. Checker names are evidence hints, not external measurements. No detector measurements are supplied in this run, so do not invent boxes, IoUs, OCR confidence, or mask percentages. State only evidence directly visible in the images.\n"
        "For protected/penalty-role questions, still judge the written positive criteria: pass means preserved/clean, fail means the violation occurred. The deterministic scorer will invert protected verdicts.\n"
        "Return JSON only. Keep all question IDs unchanged.\n\n"
        f"Case context:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        f"Atomic questions and executable decision guidance:\n{json.dumps(question_payload(contract), ensure_ascii=False, indent=2)}\n\n"
        f"Required response schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    return str(response.get("output_text") or "").strip()


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("API response must be a JSON object")
    return value


class FatalAPIError(RuntimeError):
    pass


def call_api(
    *,
    key: str,
    base_url: str,
    model: str,
    prompt: str,
    images: list[tuple[str, Path]],
    timeout: int,
    max_tokens: int,
    max_image_side: int,
    jpeg_quality: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, path in images:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path, max_image_side, jpeg_quality)}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return json.loads(response.read().decode("utf-8")), headers
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        message = f"HTTP {error.code}: {detail[:2000]}"
        if error.code in {401, 403, 404}:
            raise FatalAPIError(message) from error
        raise RuntimeError(message) from error
    except URLError as error:
        raise RuntimeError(f"network error: {error.reason}") from error


def evaluate_record(record: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    uid = str(record["uid"])
    case_dir = output_dir / "cases" / safe_name(uid)
    result_path = case_dir / "eval_result.json"
    output_image = record.get("output_image")
    if not output_image or not Path(output_image).is_file():
        return {"uid": uid, "case_id": record["case_id"], "status": "generation_failed", "result_path": None, "score": 0.0}
    if result_path.is_file() and args.resume:
        result = load(result_path)
        return {"uid": uid, "case_id": record["case_id"], "status": "skipped_existing", "result_path": str(result_path), "score": result["scoring"]["final_score"]}

    contract = load(Path(record["contract_path"]))
    if contract.get("schema_version") != "solveedit.atomic_contract.v3":
        raise ValueError(f"{uid}: expected solveedit.atomic_contract.v3 contract")
    questions = contract_questions(contract)
    question_ids = [str(item["id"]) for item in questions]
    if len(question_ids) != len(set(question_ids)) or {item["role"] for item in questions} != {"reward", "penalty"}:
        raise ValueError(f"{uid}: contract needs unique atom IDs and both required and protected criteria")
    prompt = build_prompt(contract)
    images = [("IMAGE 1 — ORIGINAL INPUT", Path(record["input_image"])), ("IMAGE 2 — MODEL OUTPUT TO SCORE", Path(record["output_image"]))]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "request_prompt.txt").write_text(prompt, encoding="utf-8")
    request_meta = {
        "uid": uid,
        "case_id": record["case_id"],
        "evaluated_model": record.get("model"),
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "evaluation_guidance": "contract_only",
        "images": [{"label": label, "path": str(path), "sha256": sha256(path)} for label, path in images],
        "question_ids": question_ids,
    }
    write_json(case_dir / "request_meta.json", request_meta)
    if args.dry_run:
        return {"uid": uid, "case_id": record["case_id"], "status": "dry_run", "result_path": None, "score": None}

    last_error: Exception | None = None
    started = time.time()
    for attempt in range(args.retries + 1):
        try:
            response, headers = call_api(
                key=args.api_key,
                base_url=args.base_url,
                model=args.judge_model,
                prompt=prompt,
                images=images,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                max_image_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
            )
            text = extract_text(response)
            raw = extract_json(text)
            normalized = normalize_case_response(raw, contract)
            result = {
                "schema_version": "solveedit.eval_result.v1",
                "uid": uid,
                "case_id": record["case_id"],
                "evaluated_model": record.get("model"),
                "judge": {
                    "model": args.judge_model,
                    "base_url": args.base_url,
                    "response_id": response.get("id"),
                    "returned_model": response.get("model"),
                    "request_id": headers.get("x-request-id"),
                    "usage": response.get("usage"),
                    "latency_seconds": round(time.time() - started, 3),
                },
                **normalized,
            }
            write_json(case_dir / "raw_response.json", response)
            (case_dir / "raw_text.txt").write_text(text, encoding="utf-8")
            write_json(result_path, result)
            return {"uid": uid, "case_id": record["case_id"], "status": "completed", "result_path": str(result_path), "score": result["scoring"]["final_score"]}
        except FatalAPIError:
            raise
        except (RuntimeError, ValueError, json.JSONDecodeError, TimeoutError) as error:
            last_error = error
            if attempt < args.retries:
                time.sleep(args.retry_delay * (2**attempt))
    (case_dir / "error.txt").write_text(str(last_error), encoding="utf-8")
    return {"uid": uid, "case_id": record["case_id"], "status": "failed", "error": str(last_error), "result_path": None, "score": None}


def write_summary(output_dir: Path, items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    statuses = Counter(str(item["status"]) for item in items)
    results = [load(Path(item["result_path"])) for item in items if item.get("result_path")]
    scores = [float(item["score"]) for item in items if item.get("score") is not None]
    summary = {
        "schema_version": "solveedit.eval_run.v1",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "records": len(items),
        "scored_records": len(scores),
        "statuses": dict(statuses),
        "mean_score": round(sum(scores) / len(scores), 6) if scores else None,
        "leaderboard_eligible": sum(bool(result["scoring"]["leaderboard_eligible"]) for result in results),
        "items": items,
    }
    write_json(output_dir / "run_manifest.json", summary)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "case_id", "status", "final_score", "result_path"])
        for item in items:
            writer.writerow([item.get("uid"), item.get("case_id"), item.get("status"), item.get("score"), item.get("result_path")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--base-url", default=os.getenv("SOLVEEDIT_API_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-image-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    args.api_key = os.getenv("SOLVEEDIT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not args.api_key and not args.dry_run:
        parser.error("Set SOLVEEDIT_API_KEY or OPENAI_API_KEY")
    return args


def main() -> int:
    args = parse_args()
    manifest = load(args.eval_manifest.resolve())
    records = list(manifest.get("records", []))
    if not records:
        raise SystemExit("Evaluation manifest contains no records")
    for record in records:
        for field in ("contract_path", "input_image", "output_image"):
            value = record.get(field)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    record[field] = str((args.eval_manifest.resolve().parent / path).resolve())
    selected = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    try:
        for number, record in enumerate(selected, 1):
            item = evaluate_record(record, args, output_dir)
            items.append(item)
            write_summary(output_dir, items, args)
            print(json.dumps({"progress": f"{number}/{len(selected)}", **item}, ensure_ascii=False), flush=True)
    except FatalAPIError as error:
        write_summary(output_dir, items, args)
        raise SystemExit(f"Fatal API error; stopping shard without retrying remaining cases: {error}")
    write_summary(output_dir, items, args)
    return 1 if any(item["status"] == "failed" for item in items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
