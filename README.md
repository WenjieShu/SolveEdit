# SolveEdit code

This repository contains the two-stage **SolveEdit-Plan** visual planner and the deterministic scoring core with a VLM evaluation runner. The benchmark images and atomic contracts are not included in this release; they will be distributed separately. The evaluation runner currently uses VLM verdicts only. It does not run the specialized checkers described in the paper, so this code alone cannot reproduce the reported leaderboard.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Both runners use an OpenAI-compatible `chat/completions` endpoint with image inputs. Set `SOLVEEDIT_API_KEY` (or `OPENAI_API_KEY`) in your environment. The default API base URL is `https://api.openai.com/v1`; use `--base-url` for another compatible provider. Select a vision-capable model explicitly.

## SolveEdit-Plan

The planner first identifies unresolved edit variables and useful image crops (Inspect), then chooses an evidence-supported transition and compiles an editing instruction (Resolve). It produces a plan; calling an image or video editor is outside this runner.

Prepare a JSONL manifest with one record per line. `input_image` can be relative to the manifest:

```json
{"uid":"example-1","case_id":"example-1","input_image":"images/example.png","instruction":"Move the item to the correct place."}
```

```bash
python scripts/run_plan.py --manifest my_cases.jsonl --output-dir outputs/plan --model YOUR_VISION_MODEL
```

Each case writes `inspection.json`, image crops, `plan.json`, and API usage metadata. Planner input validation rejects evaluation-only fields, including contracts and reference images.

## Evaluation

The evaluator reads user-supplied `solveedit.atomic_contract.v3` contracts and input/output image pairs. An evaluation manifest is a JSON object with a `records` array. Paths can be relative to the manifest:

```json
{"records":[{"uid":"example-1","case_id":"example-1","contract_path":"contracts/example-1.json","input_image":"images/example.png","output_image":"outputs/example.png","model":"my-editor"}]}
```

```bash
python scripts/run_eval.py --eval-manifest my_eval.json --output-dir outputs/eval --judge-model YOUR_VISION_MODEL
```

Use `--dry-run` to inspect prompts and file handling without an API call. The evaluator asks for a quality gate and pass/partial/fail/abstain judgments for each atom. It applies hierarchical weights within required and protected groups and computes `max(0, R - 0.5 D)` after the quality gate. Missing output images receive zero and remain in the run mean. The result JSON retains atomic verdicts, reasons, and visible evidence for inspection.

## Scope

No benchmark contracts, case images, generated outputs, credentials, or private production scripts are included. The published evaluator omits preassigned SAM/YOLO checker calls and disagreement adjudication; its scores should not be presented as the paper's full evaluation protocol. The planner output is an instruction for an external generator, not a generated image.

Run the local checks with `python -m unittest discover -s tests`.
