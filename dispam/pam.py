"""Frozen PAM propagation and optional Box2Mask inference.

Paper mapping:
  Multi-Perturbation PAM Propagation. Normal, one-pixel Dilation, and one-pixel
  Erosion prompts propagate recursively from the selected slice to both volume
  ends. Only Dilation and Erosion define the confidence-filtered Core target.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as ndi
except Exception as exc:  # pragma: no cover
    raise RuntimeError("SciPy is required for prompt perturbation.") from exc

from .component_selection import label_components_6

EPS = 1e-6


def _import_network_module():
    """Import PAM with its bundled network primitives without changing PrimusV2."""
    prefix = "dynamic_network_architectures"
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == prefix or name.startswith(prefix + ".")
    }
    for name in list(saved):
        del sys.modules[name]
    bundled = importlib.import_module(
        "dispam.pam_model.pam_dynamic_network_architectures"
    )
    sys.modules[prefix] = bundled
    try:
        module = importlib.import_module("dispam.pam_model.network")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(saved)
    return module


def _checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}.")
    return {str(key).replace("module.", ""): value for key, value in payload.items()}


def _resize_2d(array: np.ndarray, shape_hw: Sequence[int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).view(1, 1, *array.shape)
    align_corners = False if mode == "bilinear" else None
    output = F.interpolate(
        tensor,
        size=tuple(int(value) for value in shape_hw),
        mode=mode,
        align_corners=align_corners,
    )
    return output[0, 0].numpy()


def _merge_logits(prediction: Any) -> torch.Tensor:
    if isinstance(prediction, torch.Tensor):
        return prediction
    if not isinstance(prediction, (list, tuple)) or not prediction:
        raise TypeError(f"Unsupported PAM output type: {type(prediction)!r}")
    output_size = prediction[0].shape[-2:]
    values = [
        value
        if value.shape[-2:] == output_size
        else F.interpolate(value, output_size, mode="bilinear", align_corners=False)
        for value in prediction
    ]
    return torch.stack(values).mean(dim=0)


def _foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


class FrozenPAM:
    def __init__(
        self,
        propagation_checkpoint: Path,
        device: torch.device,
        target_size: int = 224,
        crop_ratio: float = 1.5,
        box2mask_checkpoint: Path | None = None,
    ) -> None:
        module = _import_network_module()
        self.device = device
        self.target_size = int(target_size)
        self.crop_ratio = float(crop_ratio)
        self.propagation_model = module.PAM(
            conv_dim=2,
            input_channels=1,
            n_stages=6,
            max_channels=512,
            num_classes=2,
            deep_supervision=True,
            n_attn_stage=4,
            from_scratch_ratio=0.0,
        )
        self.propagation_model.load_state_dict(
            _checkpoint_state(propagation_checkpoint),
            strict=True,
        )
        self.propagation_model.to(device).eval().requires_grad_(False)
        self.box2mask_model = None
        if box2mask_checkpoint is not None:
            args = type(
                "BoxArgs",
                (),
                {
                    "conv_dim": 2,
                    "input_channels": 1,
                    "n_stages": 6,
                    "max_channels": 512,
                    "num_classes": 2,
                    "deep_supervision": True,
                },
            )()
            self.box2mask_model = module.Box2SegNet(args)
            self.box2mask_model.load_state_dict(
                _checkpoint_state(box2mask_checkpoint),
                strict=True,
            )
            self.box2mask_model.to(device).eval().requires_grad_(False)

    @staticmethod
    def _normalization(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        foreground = image[mask > 0]
        values = foreground if foreground.size else image.reshape(-1)
        low, high = np.percentile(values, [0.5, 99.5])
        return float(low), float(max(high, low + 1.0))

    @staticmethod
    def _crop_bounds(mask: np.ndarray, ratio: float, minimum: int = 16) -> Tuple[int, int, int, int]:
        height, width = mask.shape
        coordinates = np.argwhere(mask > 0)
        if coordinates.size == 0:
            return 0, 0, width, height
        low = coordinates.min(axis=0)
        high = coordinates.max(axis=0) + 1
        center_yx = (low + high - 1) / 2.0
        size = max(minimum, int(np.ceil(max(high - low) * float(ratio))))
        y0 = int(round(center_yx[0] - size / 2))
        x0 = int(round(center_yx[1] - size / 2))
        y0 = min(max(y0, 0), max(0, height - size))
        x0 = min(max(x0, 0), max(0, width - size))
        return x0, y0, min(width, x0 + size), min(height, y0 + size)

    def _image_tensor(
        self,
        image: np.ndarray,
        low: float,
        high: float,
        bounds: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        x0, y0, x1, y1 = bounds
        clipped = np.clip(image, low, high)
        scaled = (clipped - low) / max(high - low, EPS)
        crop = scaled[y0:y1, x0:x1]
        resized = _resize_2d(crop, (self.target_size, self.target_size), "bilinear")
        return torch.from_numpy(((resized - 0.5) / 0.5).astype(np.float32)).unsqueeze(0)

    def _propagate_step(
        self,
        guide_image: np.ndarray,
        guide_mask: np.ndarray,
        query_image: np.ndarray,
    ) -> np.ndarray:
        if not np.any(guide_mask):
            return np.zeros_like(guide_mask, dtype=np.float32)
        low, high = self._normalization(guide_image, guide_mask)
        bounds = self._crop_bounds(guide_mask, self.crop_ratio)
        x0, y0, x1, y1 = bounds
        guide_tensor = self._image_tensor(guide_image, low, high, bounds)
        query_tensor = self._image_tensor(query_image, low, high, bounds)
        prompt_crop = guide_mask[y0:y1, x0:x1].astype(np.float32)
        prompt = _resize_2d(
            prompt_crop,
            (self.target_size, self.target_size),
            "nearest",
        )
        batch = {
            "guiding_x": guide_tensor.unsqueeze(0).to(self.device),
            "guiding_prp": torch.from_numpy(prompt).unsqueeze(0).to(self.device),
            "adjacent_x": query_tensor.unsqueeze(0).unsqueeze(0).to(self.device),
        }
        with torch.inference_mode():
            output = self.propagation_model(batch)
            probability = _foreground_probability(
                _merge_logits(output["predictions"])
            )[0].float().cpu().numpy()
        restored = _resize_2d(probability, (y1 - y0, x1 - x0), "bilinear")
        full = np.zeros_like(guide_mask, dtype=np.float32)
        full[y0:y1, x0:x1] = restored
        return np.clip(full, 0.0, 1.0)

    def propagate(
        self,
        volume_zyx: np.ndarray,
        prompt_slice: int,
        prompt_mask: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        volume = np.asarray(volume_zyx, dtype=np.float32)
        prompt = np.asarray(prompt_mask, dtype=bool)
        depth = volume.shape[0]
        selected = int(prompt_slice)
        probabilities = np.zeros(volume.shape, dtype=np.float32)
        probabilities[selected] = prompt.astype(np.float32)
        running_mask = prompt
        for index in range(selected + 1, depth):
            probabilities[index] = self._propagate_step(
                volume[index - 1],
                running_mask,
                volume[index],
            )
            running_mask = probabilities[index] >= float(threshold)
        running_mask = prompt
        for index in range(selected - 1, -1, -1):
            probabilities[index] = self._propagate_step(
                volume[index + 1],
                running_mask,
                volume[index],
            )
            running_mask = probabilities[index] >= float(threshold)
        return probabilities

    def mask_from_boxes(
        self,
        image_2d: np.ndarray,
        boxes_xyxy: Sequence[Sequence[int]],
        threshold: float,
    ) -> np.ndarray:
        if self.box2mask_model is None:
            raise RuntimeError("Box2Mask checkpoint was not loaded.")
        output = np.zeros(image_2d.shape, dtype=bool)
        low, high = np.percentile(image_2d, [0.5, 99.5])
        for raw_box in boxes_xyxy:
            x0, y0, x1, y1 = (int(value) for value in raw_box)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image_2d.shape[1], x1), min(image_2d.shape[0], y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = np.clip(image_2d[y0:y1, x0:x1], low, high)
            crop = (crop - low) / max(float(high - low), EPS)
            resized = _resize_2d(crop, (self.target_size, self.target_size), "bilinear")
            tensor = torch.from_numpy(((resized - 0.5) / 0.5).astype(np.float32))
            batch = {"query_xs": tensor.view(1, 1, 1, self.target_size, self.target_size).to(self.device)}
            with torch.inference_mode():
                logits = _merge_logits(self.box2mask_model(batch)["predictions"])
                probability = _foreground_probability(logits)[0].float().cpu().numpy()
            restored = _resize_2d(probability, (y1 - y0, x1 - x0), "bilinear")
            output[y0:y1, x0:x1] |= restored >= float(threshold)
        return output


def boxes_from_binary_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return one bounding box enclosing all target pixels, or none if empty.

    Coordinates are (x_min, y_min, x_max, y_max), with exclusive maxima.
    """
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if coordinates.size == 0:
        return []
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0) + 1
    return [(int(low[1]), int(low[0]), int(high[1]), int(high[0]))]


def perturb_prompt(mask: np.ndarray, mode: str) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    structure = ndi.generate_binary_structure(rank=2, connectivity=1)
    if mode == "normal":
        return binary
    if mode == "dilation":
        return ndi.binary_dilation(binary, structure=structure, iterations=1)
    if mode == "erosion":
        eroded = ndi.binary_erosion(binary, structure=structure, iterations=1)
        return eroded if np.any(eroded) else binary
    raise ValueError(f"Unsupported prompt perturbation {mode!r}.")


def build_multi_perturbation_targets(
    pam: FrozenPAM,
    volume_zyx: np.ndarray,
    selected_slice: int,
    prompt_mask: np.ndarray,
    binarization_threshold: float,
    confidence_threshold: float,
) -> Dict[str, np.ndarray]:
    # Multi-Perturbation PAM: normal, dilation, and erosion implement
    # M^{star,0}, M^{star,1}, and M^{star,2}, respectively.
    probabilities = {
        mode: pam.propagate(
            volume_zyx,
            selected_slice,
            perturb_prompt(prompt_mask, mode),
            binarization_threshold,
        )
        for mode in ("normal", "dilation", "erosion")
    }
    hard_target = probabilities["normal"] >= float(binarization_threshold)
    dilation = probabilities["dilation"]
    erosion = probabilities["erosion"]
    dilation_confident = np.maximum(dilation, 1.0 - dilation) >= float(
        confidence_threshold
    )
    erosion_confident = np.maximum(erosion, 1.0 - erosion) >= float(
        confidence_threshold
    )
    dilation_binary = dilation >= float(binarization_threshold)
    erosion_binary = erosion >= float(binarization_threshold)
    # The Hard target comes from the Normal branch; the Core region and
    # probability come from confidence-filtered Dilation/Erosion agreement.
    core_region = (
        dilation_confident
        & erosion_confident
        & (dilation_binary == erosion_binary)
    )
    core_probability = ((dilation + erosion) / 2.0).astype(np.float32)
    return {
        "hard_target": hard_target.astype(np.uint8),
        "core_region": core_region.astype(np.uint8),
        "core_probability": core_probability,
    }
