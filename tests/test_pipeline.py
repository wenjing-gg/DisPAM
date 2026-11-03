"""CPU integration of HCDD -> fixed regions -> CDFT -> held-out testing.

Small convolutional students replace the large model architectures; data loading,
training loops, losses, checkpoint I/O and split validation execute normally.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from io import StringIO
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from dispam.cache import cache_files, save_cache
from dispam.datasets import CaseRecord, split_protocol
from dispam.models.registry import MODEL_NAMES
from scripts import train_dual_distillation as hcdd
from scripts import build_cdft_regions as regions
from scripts import train_cdft as cdft
from scripts import evaluate, build_pam_cache
from tests.test_protocol import RecordingAgent


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv3d(1, 4, 1)
        self.output = nn.Conv3d(4, 1, 1)
        nn.init.constant_(self.output.bias, 1.0)

    def forward(self, image):
        return self.output(torch.relu(self.decoder(image)))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cases = {}
        for role in ("train", "val", "test"):
            image = self.root / f"{role}_image.npy"
            label = self.root / f"{role}_label.npy"
            np.save(image, np.arange(64, dtype=np.float32).reshape(4, 4, 4))
            np.save(label, np.ones((4, 4, 4), dtype=np.uint8))
            self.cases[role] = [CaseRecord(role, (image,), label)]
        self.split = self.root / "split.json"
        self.split.write_text(json.dumps({role: [role] for role in self.cases}))
        self.protocol = split_protocol("kits19", self.split)
        self.cache = self.root / "cache"
        ones = np.ones((4, 4, 4), dtype=np.uint8)
        save_cache(cache_files(self.cache, "train", "kidney", "train"), ones, ones, ones * 0.9, {"agent_queries_per_candidate": 1})
        self.base = ["--dataset", "kits19", "--dataset-root", str(self.root), "--split-file", str(self.split), "--target", "kidney", "--device", "cpu", "--patch-size", "4", "4", "4"]
        self.train_queries = []

    def training_cases(self, dataset, root, split_file, role):
        self.assertIn(role, ("train", "val"))
        self.train_queries.append(role)
        return self.cases[role]

    def run_main(self, module, argv, case_loader, extra_patches=()):
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", [module.__name__] + argv))
            stack.enter_context(patch.object(module, "cases_for_split", side_effect=case_loader))
            stack.enter_context(patch("dispam.models.registry.build_model", side_effect=lambda *a, **kw: TinyStudent()))
            for item in extra_patches:
                stack.enter_context(item)
            stack.enter_context(redirect_stdout(StringIO()))
            module.main()

    def test_training_selection_and_final_test_use_separate_cases(self):
        checkpoint_args = []
        for model in MODEL_NAMES:
            output = self.root / "dual" / model
            def validation(model, cases, *args):
                self.assertEqual([c.case_id for c in cases], ["val"])
                return next(scores), []
            scores = iter([0.9, 0.4])
            self.run_main(hcdd, self.base + ["--model", model, "--cache-root", str(self.cache), "--output-dir", str(output), "--epochs", "2", "--validation-interval", "1", "--num-workers", "0"], self.training_cases, [patch.object(hcdd, "build_model", side_effect=lambda *a: TinyStudent()), patch.object(hcdd, "evaluate_model", side_effect=validation)])
            best = torch.load(output / "best.pt", map_location="cpu")
            last = torch.load(output / "last.pt", map_location="cpu")
            self.assertEqual((best["epoch"], last["epoch"]), (1, 2))
            self.assertEqual(best["metadata"]["data_protocol"], self.protocol)
            checkpoint_args += ["--checkpoint", f"{model}={output / 'best.pt'}"]
        region_root = self.root / "regions"
        self.run_main(regions, self.base + checkpoint_args + ["--output-root", str(region_root), "--min-component-voxels", "1"], self.training_cases)
        output = self.root / "cdft"
        scores = iter([0.8, 0.3])
        def validation(models, cases, *args):
            self.assertEqual([c.case_id for c in cases], ["val"])
            return {"mean_dice": next(scores)}, []
        cdft_args = self.base + checkpoint_args + ["--cache-root", str(self.cache), "--region-root", str(region_root), "--output-dir", str(output), "--epochs", "2", "--num-workers", "0"]
        metadata_path = region_root / "train" / "kidney" / "train.json"
        region_metadata = metadata_path.read_text()
        metadata = json.loads(region_metadata)
        metadata.pop("data_protocol")
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "rebuild them"):
            self.run_main(cdft, cdft_args, self.training_cases)
        metadata_path.write_text(region_metadata)
        self.run_main(cdft, cdft_args, self.training_cases, [patch.object(cdft, "evaluate_ensemble", side_effect=validation)])
        self.assertNotIn("test", self.train_queries)
        banks = []
        for model in MODEL_NAMES:
            best = torch.load(output / "best" / f"{model}.pt", map_location="cpu")
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(best["metadata"]["data_protocol"], self.protocol)
            banks += ["--bank", f"kidney:{model}={output / 'best' / (model + '.pt')}"]
        def held_out_cases(dataset, root, split_file, role):
            self.assertEqual(role, "test")
            return self.cases[role]
        # evaluate.py has no --target argument: targets come from the bank specs.
        evaluation_args = self.base[:self.base.index("--target")] + ["--device", "cpu", "--patch-size", "4", "4", "4"] + banks + ["--output-dir", str(self.root / "evaluation")]
        self.run_main(evaluate, evaluation_args, held_out_cases)
        summary = json.loads((self.root / "evaluation" / "summary.json").read_text())
        self.assertEqual((summary["split"], summary["purpose"]), ("test", "final_test"))
        rows = (self.root / "evaluation" / "kidney_cases.csv").read_text()
        self.assertEqual(
            {row["case_id"] for row in csv.DictReader(StringIO(rows))}, {"test"}
        )
        # An old weight file cannot become an independent-test checkpoint merely
        # by being passed a new split file.
        old = torch.load(output / "best" / "vista3d.pt", map_location="cpu")
        old["metadata"].pop("data_protocol")
        torch.save(old, output / "best" / "vista3d.pt")
        with self.assertRaisesRegex(ValueError, "no train/val/test provenance"):
            self.run_main(evaluate, evaluation_args, held_out_cases)

    def test_all_rejected_present_target_does_not_generate_a_cache(self):
        prompts = self.root / "prompts.json"
        prompts.write_text(json.dumps({"kits19.kidney": "Visible?"}))
        agent = RecordingAgent()
        argv = ["--dataset", "kits19", "--dataset-root", str(self.root), "--split-file", str(self.split), "--targets", "kidney", "--cache-root", str(self.root / "empty_cache"), "--pam-checkpoint", "pam.pth", "--prompt-file", str(prompts), "--device", "cpu"]
        with self.assertRaisesRegex(RuntimeError, "no candidate received an affirmative"):
            self.run_main(build_pam_cache, argv, self.training_cases, [patch.object(build_pam_cache, "build_agent", return_value=agent), patch.object(build_pam_cache, "FrozenPAM")])
        self.assertEqual(len(agent.calls), 4)
        self.assertEqual(len(set(agent.calls)), 4)
        self.assertFalse(list((self.root / "empty_cache").rglob("*.npz")))


if __name__ == "__main__":
    unittest.main()
