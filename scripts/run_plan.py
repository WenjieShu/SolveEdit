#!/usr/bin/env python3
"""Run SolveEdit-Plan over a manifest of images and editing requests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solveedit.planner import run_case, safe_name


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.getenv("SOLVEEDIT_API_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    api_key = os.getenv("SOLVEEDIT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("Set SOLVEEDIT_API_KEY or OPENAI_API_KEY")
    manifest_path = args.manifest.resolve()
    rows = read_jsonl(manifest_path)
    for row in rows:
        image = Path(str(row.get("input_image", "")))
        if not image.is_absolute():
            row["input_image"] = str((manifest_path.parent / image).resolve())
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f".shard_{args.shard_index:02d}" if args.num_shards > 1 else ""
    status_path = output_dir / f"status{shard_suffix}.jsonl"
    summary_path = output_dir / f"run_summary{shard_suffix}.json"
    records = []
    for index, row in enumerate(rows, 1):
        case_dir = output_dir / "cases" / safe_name(str(row["uid"]))
        plan_path = case_dir / "plan.json"
        status = {"uid": row["uid"], "case_id": row["case_id"], "status": "failed", "plan_path": str(plan_path)}
        started = time.time()
        if args.resume and plan_path.is_file():
            status["status"] = "skipped_existing"
        else:
            try:
                run_case(
                    record=row,
                    case_dir=case_dir,
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                status["status"] = "complete"
            except Exception as exc:  # noqa: BLE001 - record exact per-case failure.
                status["error"] = f"{type(exc).__name__}: {exc}"
        status["elapsed_seconds"] = round(time.time() - started, 3)
        records.append(status)
        status_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8"
        )
        print(json.dumps({"progress": f"{index}/{len(rows)}", **status}, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": "solveedit.plan_run.v1",
        "model": args.model,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "requested": len(rows),
        "complete": sum(item["status"] in {"complete", "skipped_existing"} for item in records),
        "failed": sum(item["status"] == "failed" for item in records),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
