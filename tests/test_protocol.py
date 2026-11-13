from __future__ import annotations

from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dispam.agent import (
    OpenAICompatibleVisibilityAgent, ReplayVisibilityAgent,
    VisibilityAgent, select_visible_slice,
)
from dispam.datasets import (
    load_splits, load_split_case_ids, make_three_way_split,
    split_protocol, validate_checkpoint_protocol,
)
from scripts.create_splits import create_splits
from scripts import build_pam_cache, evaluate

ROOT = Path(__file__).resolve().parents[1]


class RecordingAgent(VisibilityAgent):
    def __init__(self, accepted=()):
        self.accepted = set(accepted)
        self.calls = []

    def is_visible(self, case_id, target, slice_index, image_2d, prompt):
        self.calls.append(slice_index)
        return slice_index in self.accepted


class SplitProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "split.json"

    def write(self, payload):
        self.path.write_text(json.dumps(payload))
        return self.path

    def test_shipped_partitions_are_disjoint_fixed_and_reproducible(self):
        counts = {"kits19": (179, 0, 20), "brats18": (228, 0, 57), "lits": (104, 0, 27)}
        for dataset, expected in counts.items():
            with self.subTest(dataset=dataset):
                path = ROOT / "configs" / "splits" / f"{dataset}.json"
                splits = load_splits(path)
                self.assertEqual(tuple(map(len, splits.values())), expected)
                recreated = create_splits(json.loads(path.read_text()), 0.0, 0)
                self.assertEqual(recreated["train"], splits["train"])
                self.assertEqual(recreated["val"], splits["val"])
                self.assertEqual(recreated["test"], splits["test"])

    def test_missing_validation_never_aliases_test(self):
        self.write({"train": ["a"], "test": ["b"]})
        self.assertEqual(load_split_case_ids(self.path, "train"), ["a"])
        self.assertEqual(load_split_case_ids(self.path, "val"), [])
        self.assertEqual(load_split_case_ids(self.path, "test"), ["b"])
        with self.assertRaisesRegex(ValueError, "test"):
            load_splits(self.write({"train": ["a"], "val": ["b"]}))

    def test_overlap_and_normalized_duplicate_cases_are_rejected(self):
        for payload in [
            {"train": ["a"], "val": ["b"], "test": ["b"]},
            {"train": ["a", "folder/a"], "val": ["b"], "test": ["c"]},
            {"train": ["a"], "train_case_ids": ["x"], "val": ["b"], "test": ["c"]},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                load_splits(self.write(payload))

    def test_holdout_preserves_outer_test_and_full_development_pool(self):
        original = {"train_case_ids": [f"case_{i}" for i in range(20)], "test_case_ids": ["outer_test"]}
        result = create_splits(original, 0.1, 0)
        self.assertEqual(result["test"], ["outer_test"])
        self.assertEqual(set(result["train"]) | set(result["val"]), set(original["train_case_ids"]))
        self.assertEqual(len(result["val"]), 2)
        self.assertEqual(result, create_splits(original, 0.1, 0))
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_three_way_split(["a", "b"], ["a"])

    def test_missing_or_different_checkpoint_partition_is_rejected(self):
        self.write({"train": ["a"], "val": ["b"], "test": ["c"]})
        protocol = split_protocol("kits19", self.path)
        checkpoint = {"metadata": {"target": "kidney", "data_protocol": protocol}}
        validate_checkpoint_protocol(checkpoint, protocol, "kidney")
        with self.assertRaisesRegex(ValueError, "no train/val/test provenance"):
            validate_checkpoint_protocol({"metadata": {}}, protocol)
        other = dict(protocol, val=["c"], test=["b"])
        with self.assertRaisesRegex(ValueError, "split differs"):
            validate_checkpoint_protocol(checkpoint, other)
        with self.assertRaisesRegex(ValueError, "target"):
            validate_checkpoint_protocol(checkpoint, protocol, "tumor")


class AgentProtocolTests(unittest.TestCase):
    def test_each_candidate_is_queried_once_and_selection_stops_at_first_yes(self):
        agent = RecordingAgent({3, 1})
        result = select_visible_slice(agent, "case", "kidney", [5, 3, 1], np.zeros((6, 4, 4)), "Visible?")
        self.assertEqual(result, 3)
        self.assertEqual(agent.calls, [5, 3])

    def test_all_no_returns_no_selection_without_extra_review(self):
        agent = RecordingAgent()
        result = select_visible_slice(agent, "case", "kidney", [2, 0, 1], np.zeros((3, 4, 4)), "Visible?")
        self.assertIsNone(result)
        self.assertEqual(agent.calls, [2, 0, 1])

    def test_replay_requires_one_boolean_not_legacy_votes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            path.write_text(json.dumps({"case/kidney/0": [True, False, True]}))
            with self.assertRaisesRegex(ValueError, "one boolean"):
                ReplayVisibilityAgent(path)
            path.write_text(json.dumps({"case/kidney/0": False, "case/kidney/1": True}))
            agent = ReplayVisibilityAgent(path)
            self.assertEqual(select_visible_slice(agent, "case", "kidney", [0, 1], np.zeros((2, 4, 4)), "Visible?"), 1)

    def test_api_sends_one_deterministic_request(self):
        agent = OpenAICompatibleVisibilityAgent("gpt-5.5", "https://example.invalid/chat/completions")
        response = BytesIO(json.dumps({"choices": [{"message": {"content": "Yes"}}]}).encode())
        with patch.dict(os.environ, {"VLM_API_KEY": "unit-test-key"}), patch("dispam.agent.request.urlopen", return_value=response) as call:
            self.assertTrue(agent.is_visible("case", "kidney", 0, np.zeros((4, 4)), "Visible?"))
        self.assertEqual(call.call_count, 1)
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual((payload["temperature"], payload["top_p"]), (0, 1))
        self.assertEqual(payload["model"], "gpt-5.5")

    def test_cli_has_no_repetitions_or_clinician_override(self):
        base = ["build_pam_cache.py", "--dataset", "kits19", "--dataset-root", "data", "--split-file", "split.json", "--targets", "kidney", "--cache-root", "cache", "--pam-checkpoint", "pam.pth", "--prompt-file", "prompts.json"]
        with patch("sys.argv", base):
            args = build_pam_cache.parse_args()
        self.assertEqual(args.splits, ["train"])
        self.assertEqual(args.agent_model, "gpt-5.5")
        for extra in (["--agent-repetitions", "3"], ["--clinician-fallback", "manual.json"], ["--splits", "test"]):
            with patch("sys.argv", base + extra), patch("sys.stderr", new_callable=StringIO), self.assertRaises(SystemExit):
                build_pam_cache.parse_args()

    def test_evaluation_defaults_to_test(self):
        argv = ["evaluate.py", "--dataset", "kits19", "--dataset-root", "data", "--split-file", "split.json", "--bank", "kidney:vista3d=model.pt", "--output-dir", "outputs"]
        with patch("sys.argv", argv):
            self.assertEqual(evaluate.parse_args().split, "test")


if __name__ == "__main__":
    unittest.main()
