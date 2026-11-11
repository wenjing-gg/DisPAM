from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dispam.components import (
    build_consensus_divergence,
    filter_small_components,
    gaussian_importance_map,
    sliding_window_probability,
    sliding_window_starts,
)
from dispam.cache import (
    CDFTPatchDataset,
    cache_files,
    region_files,
    save_cache,
    save_regions,
)
from dispam.aggregation import aggregate_case_runs, summarize_case_averages
from dispam.datasets import CaseRecord, load_split_case_ids
from dispam.losses import (
    SharedDescriptorEncoder,
    FeatureAdapter,
    consensus_divergence_contrastive_loss,
    consensus_foreground_loss,
)
from dispam.models.registry import forward_with_decoder_features
from dispam.pam import boxes_from_binary_mask, build_multi_perturbation_targets
from dispam.slice_selector import propagation_aware_slice_estimation


class ConstantLogitModel(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.logit = float(math.log(probability / (1.0 - probability)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (inputs.shape[0], 1, *inputs.shape[-3:]),
            self.logit,
            dtype=inputs.dtype,
            device=inputs.device,
        )


class FeatureToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Conv3d(1, 4, 3, padding=1)
        self.output = nn.Conv3d(4, 1, 1)

    def forward(self, image_hwd: torch.Tensor) -> torch.Tensor:
        standard = image_hwd.permute(0, 1, 4, 2, 3)
        logits = self.output(torch.relu(self.decoder(standard)))
        return logits.permute(0, 1, 3, 4, 2)


class SequencePAM:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = list(outputs)

    def propagate(
        self,
        volume_zyx: np.ndarray,
        prompt_slice: int,
        prompt_mask: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        del volume_zyx, prompt_slice, prompt_mask, threshold
        return self.outputs.pop(0)


class DisPAMCoreTests(unittest.TestCase):
    def test_five_run_metrics_are_averaged_per_case_before_summary(self) -> None:
        runs = []
        for offset in range(5):
            runs.append(
                {
                    "case_a": {"dice": 0.6 + 0.01 * offset},
                    "case_b": {"dice": 0.8 + 0.01 * offset},
                }
            )
        rows = aggregate_case_runs(runs, ("dice",))
        self.assertAlmostEqual(float(rows[0]["dice"]), 0.62)
        self.assertAlmostEqual(float(rows[1]["dice"]), 0.82)
        summary = summarize_case_averages(rows, ("dice",), run_count=5)
        self.assertAlmostEqual(float(summary["mean_dice"]), 0.72)
        self.assertAlmostEqual(float(summary["std_dice"]), 0.10)
        self.assertEqual(int(summary["run_count"]), 5)

    def test_sliding_window_starts_reaches_last_boundary(self) -> None:
        self.assertEqual(sliding_window_starts(10, 4, 0.5), [0, 2, 4, 6])
        self.assertEqual(sliding_window_starts(11, 4, 0.5), [0, 2, 4, 6, 7])
        self.assertEqual(sliding_window_starts(3, 4, 0.5), [0])

    def test_gaussian_importance_map_is_positive_and_center_weighted(self) -> None:
        weight = gaussian_importance_map((5, 5, 5))
        self.assertEqual(weight.shape, (5, 5, 5))
        self.assertGreater(float(weight.min()), 0.0)
        self.assertAlmostEqual(float(weight.max()), 1.0, places=6)
        self.assertGreater(float(weight[2, 2, 2]), float(weight[0, 0, 0]))

    def test_sliding_window_probability_handles_padding_and_overlap(self) -> None:
        model = ConstantLogitModel(0.8)
        image = np.zeros((1, 3, 6, 7), dtype=np.float32)
        probability, info = sliding_window_probability(
            model,
            image,
            (4, 4, 4),
            0.5,
            2,
            torch.device("cpu"),
            False,
        )
        self.assertEqual(probability.shape, (3, 6, 7))
        self.assertTrue(np.allclose(probability, 0.8, atol=1e-6))
        self.assertGreater(int(info["window_count"]), 1)

    def test_small_components_are_removed(self) -> None:
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[0, 0, 0] = True
        mask[2:4, 2:4, 2:4] = True
        filtered, rows = filter_small_components(mask, min_component_voxels=8)
        self.assertFalse(bool(filtered[0, 0, 0]))
        self.assertEqual(int(filtered.sum()), 8)
        self.assertEqual([row["voxels"] for row in rows], [8])

    def test_regions_follow_vote_and_population_std_rules(self) -> None:
        probabilities = [
            torch.full((1, 1, 4, 4, 4), 0.1, dtype=torch.float32)
            for _ in range(4)
        ]
        for model_index in range(3):
            probabilities[model_index][0, 0, 0, 0, 0] = 0.60
        probabilities[3][0, 0, 0, 0, 0] = 0.49
        high_dispersion = (0.99, 0.51, 0.51, 0.49)
        for model_index, value in enumerate(high_dispersion):
            probabilities[model_index][0, 0, 1, 1, 1] = value
        for model_index in range(2):
            probabilities[model_index][0, 0, 2, 2, 2] = 0.9
        regions = build_consensus_divergence(
            probabilities,
            probability_threshold=0.5,
            consensus_min_votes=3,
            probability_std_threshold=0.12,
            min_component_voxels=1,
        )
        self.assertEqual(float(regions.consensus[0, 0, 0, 0, 0]), 1.0)
        self.assertEqual(float(regions.divergence[0, 0, 0, 0, 0]), 0.0)
        self.assertEqual(float(regions.consensus[0, 0, 1, 1, 1]), 0.0)
        self.assertEqual(float(regions.divergence[0, 0, 1, 1, 1]), 1.0)
        self.assertEqual(float(regions.divergence[0, 0, 2, 2, 2]), 1.0)
        self.assertEqual(float(regions.divergence[0, 0, 3, 3, 3]), 0.0)
        expected_std = float(np.std(np.asarray(high_dispersion), ddof=0))
        self.assertAlmostEqual(
            float(regions.probability_std[0, 0, 1, 1, 1]),
            expected_std,
            places=6,
        )
        self.assertEqual(
            int(torch.sum(regions.consensus * regions.divergence).item()),
            0,
        )

    def test_core_requires_both_confidence_filters_and_binary_agreement(self) -> None:
        normal = np.full((2, 2, 2), 0.8, dtype=np.float32)
        dilation = np.asarray(
            [[[0.9, 0.9], [0.6, 0.1]], [[0.9, 0.9], [0.9, 0.9]]],
            dtype=np.float32,
        )
        erosion = np.asarray(
            [[[0.8, 0.6], [0.9, 0.2]], [[0.8, 0.8], [0.8, 0.8]]],
            dtype=np.float32,
        )
        output = build_multi_perturbation_targets(
            SequencePAM([normal, dilation, erosion]),
            np.zeros((2, 2, 2), dtype=np.float32),
            0,
            np.ones((2, 2), dtype=bool),
            0.5,
            0.7,
        )
        self.assertEqual(int(output["hard_target"].sum()), 7)
        self.assertEqual(int(output["hard_target"][0, 1, 1]), 0)
        self.assertEqual(int(output["core_region"][0, 0, 0]), 1)
        self.assertEqual(int(output["core_region"][0, 0, 1]), 0)
        self.assertEqual(int(output["core_region"][0, 1, 0]), 0)
        self.assertAlmostEqual(float(output["core_probability"][0, 0, 0]), 0.85)

    def test_hard_target_uses_confident_agreement_and_retains_normal_elsewhere(self) -> None:
        normal = np.array([[[0.1, 0.9, 0.1, 0.9, 0.1]]], dtype=np.float32)
        dilation = np.array([[[0.9, 0.1, 0.9, 0.6, 0.9]]], dtype=np.float32)
        erosion = np.array([[[0.8, 0.2, 0.1, 0.9, 0.8]]], dtype=np.float32)
        output = build_multi_perturbation_targets(
            SequencePAM([normal, dilation, erosion]),
            np.zeros((1, 1, 5), dtype=np.float32), 0,
            np.ones((1, 5), dtype=bool), 0.5, 0.7,
        )
        np.testing.assert_array_equal(output["hard_target"], [[[1, 0, 0, 1, 1]]])
        np.testing.assert_array_equal(output["core_region"], [[[1, 1, 0, 0, 1]]])

    def test_box_extraction_encloses_all_components(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:3, 2:5] = True
        mask[5:7, 6:8] = True
        self.assertEqual(
            boxes_from_binary_mask(mask),
            [(2, 1, 8, 7)],
        )

    def test_box_extraction_empty_mask(self) -> None:
        self.assertEqual(boxes_from_binary_mask(np.zeros((8, 8), dtype=bool)), [])

    def test_consensus_loss_does_not_create_background_targets(self) -> None:
        logits = torch.zeros((1, 1, 2, 2, 2), requires_grad=True)
        region = torch.zeros_like(logits)
        region[0, 0, 0, 0, 0] = 1
        loss = consensus_foreground_loss(logits, region)
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        self.assertLess(float(logits.grad[0, 0, 0, 0, 0]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 1, 1, 1]), 0.0)

    def test_decoder_features_feed_shared_descriptor_encoder(self) -> None:
        model = FeatureToyModel()
        image = torch.randn(1, 1, 8, 8, 4)
        logits, features = forward_with_decoder_features(model, image)
        adapter = FeatureAdapter(features.shape[1], hidden_dim=6)
        encoder = SharedDescriptorEncoder(hidden_dim=6, embedding_dim=5)
        descriptors = encoder(adapter(features))
        self.assertEqual(tuple(logits.shape), (1, 1, 8, 8, 4))
        self.assertEqual(tuple(descriptors.shape), (1, 5, 8, 8, 4))
        norms = torch.linalg.vector_norm(descriptors, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_contrastive_objective_is_finite_and_differentiable(self) -> None:
        fields = [
            torch.nn.functional.normalize(
                torch.randn(1, 4, 3, 3, 3, requires_grad=True),
                dim=1,
            )
            for _ in range(4)
        ]
        consensus = torch.zeros((1, 3, 3, 3))
        consensus[:, 0, 0, 0] = 1
        divergence = torch.zeros((1, 3, 3, 3))
        divergence[:, 2, 2, 2] = 1
        loss = consensus_divergence_contrastive_loss(
            fields,
            [[consensus]],
            [[divergence]],
            temperature=0.15,
        )
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_pase_returns_a_stable_complete_order(self) -> None:
        image = np.zeros((1, 7, 12, 12), dtype=np.float32)
        image[0, 2:5, 3:9, 3:9] = 1.0
        first = propagation_aware_slice_estimation(image)
        second = propagation_aware_slice_estimation(image)
        self.assertEqual(first.order, second.order)
        self.assertEqual(sorted(first.order), list(range(7)))
        self.assertEqual(int(np.argmax(first.score)), first.order[0])

    def test_split_loader_accepts_explicit_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(
                json.dumps({"train_case_ids": ["case_1", "data/case_2/"], "val_case_ids": ["case_val"], "test_case_ids": ["case_test"]}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_split_case_ids(path, "train"),
                ["case_1", "case_2"],
            )

    def test_cdft_dataset_crops_fixed_full_volume_component_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.npy"
            label_path = root / "label.npy"
            np.save(image_path, np.zeros((6, 6, 6), dtype=np.float32))
            np.save(label_path, np.zeros((6, 6, 6), dtype=np.float32))
            case = CaseRecord("case_a", (image_path,), label_path)
            hard = np.zeros((6, 6, 6), dtype=np.uint8)
            hard[2:4, 2:4, 2:4] = 1
            core = hard.copy()
            save_cache(
                cache_files(root / "cache", "train", "kidney", case.case_id),
                hard,
                core,
                hard.astype(np.float32),
                {"case_id": case.case_id},
            )
            consensus = np.zeros((6, 6, 6), dtype=np.int32)
            consensus[2:4, 2:4, 2:4] = 7
            divergence = np.zeros((6, 6, 6), dtype=np.int32)
            divergence[4:6, 4:6, 4:6] = 11
            save_regions(
                region_files(root / "regions", "train", "kidney", case.case_id),
                consensus,
                divergence,
                {"case_id": case.case_id},
            )
            dataset = CDFTPatchDataset(
                [case],
                root / "cache",
                root / "regions",
                "train",
                "kidney",
                (4, 4, 4),
                1,
                1.0,
                1.0,
                0,
            )
            item = dataset[0]
            self.assertIn(7, torch.unique(item["consensus_labels"]).tolist())
            self.assertEqual(tuple(item["image"].shape), (1, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
