import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from solveedit.evaluation import contract_questions, normalize_case_response
from solveedit.planner_schema import validate_planner_input


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "schema_version": "solveedit.atomic_contract.v3",
            "criterion_groups": [
                {"role": "required", "property": "semantic_accuracy", "atoms": [{"id": "r1"}, {"id": "r2"}]},
                {"role": "required", "property": "relational_accuracy", "atoms": [{"id": "r3"}]},
                {"role": "protected", "property": "semantic_accuracy", "atoms": [{"id": "p1"}]},
            ],
        }

    def test_hierarchical_weights_and_abstentions(self):
        weights = {item["id"]: item["weight"] for item in contract_questions(self.contract)}
        self.assertEqual(weights, {"r1": 0.25, "r2": 0.25, "r3": 0.5, "p1": 1.0})
        result = normalize_case_response(
            {"quality_gate": {"status": "pass"}, "atomic_results": [
                {"id": "r1", "verdict": "pass"},
                {"id": "r2", "verdict": "abstain"},
                {"id": "r3", "verdict": "partial"},
                {"id": "p1", "verdict": "fail"},
            ]},
            self.contract,
        )
        self.assertEqual(result["scoring"]["reward_score"], 0.5)
        self.assertEqual(result["scoring"]["penalty_deduct"], 1.0)
        self.assertEqual(result["scoring"]["final_score"], 0.0)

    def test_quality_gate_zeroes_score(self):
        result = normalize_case_response(
            {"quality_gate": {"status": "fail"}, "atomic_results": [{"id": "r1", "verdict": "pass"}]},
            self.contract,
        )
        self.assertEqual(result["scoring"]["final_score"], 0.0)


class PlannerTests(unittest.TestCase):
    def test_evaluation_fields_are_rejected(self):
        record = {"uid": "a", "case_id": "a", "input_image": "a.png", "instruction": "Fix it", "contract_path": "private.json"}
        self.assertIn("forbidden_key:$.contract_path", validate_planner_input(record))


class RunnerTests(unittest.TestCase):
    def test_prompt_keeps_evidence_requirements(self):
        from importlib.util import module_from_spec, spec_from_file_location

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
        spec = spec_from_file_location("solveedit_run_eval", path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        contract = {
            "schema_version": "solveedit.atomic_contract.v3",
            "criterion_groups": [
                {"role": "required", "property": "semantic_accuracy", "atoms": [
                    {"id": "r1", "evidence_requirements": ["Read the visible label"]},
                ]},
            ],
        }
        self.assertIn("Read the visible label", module.build_prompt(contract))

    def test_eval_dry_run_and_missing_generation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            Image.new("RGB", (8, 8), "white").save(work / "input.png")
            Image.new("RGB", (8, 8), "black").save(work / "output.png")
            contract = {
                "schema_version": "solveedit.atomic_contract.v3",
                "criterion_groups": [
                    {"role": "required", "property": "semantic_accuracy", "atoms": [{"id": "r1"}]},
                    {"role": "protected", "property": "visual_quality", "atoms": [{"id": "p1"}]},
                ],
            }
            (work / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            manifest = {"records": [
                {"uid": "one", "case_id": "one", "contract_path": "contract.json", "input_image": "input.png", "output_image": "output.png"},
                {"uid": "two", "case_id": "two", "contract_path": "contract.json", "input_image": "input.png", "output_image": "missing.png"},
            ]}
            (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(root / "src"))
            subprocess.run(
                [sys.executable, str(root / "scripts" / "run_eval.py"), "--eval-manifest", str(work / "manifest.json"),
                 "--output-dir", str(work / "results"), "--judge-model", "test-model", "--dry-run"],
                env=env, check=True, capture_output=True, text=True,
            )
            summary = json.loads((work / "results" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["statuses"], {"dry_run": 1, "generation_failed": 1})
            self.assertEqual(summary["mean_score"], 0.0)
            self.assertEqual(summary["scored_records"], 1)


if __name__ == "__main__":
    unittest.main()
