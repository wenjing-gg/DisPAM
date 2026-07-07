#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import ctypes
import gzip
import importlib
import json
import math
import os
import random
import re
import sys
import time
import warnings
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import numpy as np

try:
    from aggrigator import AggregationMethods, UncertaintyMap

    HAS_AGGRIGATOR = True
except Exception:
    AggregationMethods = None
    UncertaintyMap = None
    HAS_AGGRIGATOR = False

try:
    import cv2

    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    from scipy import ndimage as ndi

    HAS_SCIPY = True
except Exception:
    ndi = None
    HAS_SCIPY = False

try:
    import nibabel as nib

    HAS_NIB = True
except Exception:
    nib = None
    HAS_NIB = False

try:
    from PIL import Image

    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False

try:
    def _clear_torch_modules() -> None:
        for name in list(sys.modules.keys()):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)

    def _load_local_itt_stub() -> bool:
        repo_root = Path(__file__).resolve().parents[1]
        stub_dir = repo_root / "tools" / "torch_itt_stub"
        stub_src = stub_dir / "libittnotify_stub.c"
        stub_so = stub_dir / "libittnotify.so"
        if not stub_src.exists():
            return False
        if not stub_so.exists() or stub_src.stat().st_mtime > stub_so.stat().st_mtime:
            stub_dir.mkdir(parents=True, exist_ok=True)
            cc_bin = os.environ.get("CC_BIN", "gcc")
            try:
                subprocess.run(
                    [cc_bin, "-shared", "-fPIC", "-O2", str(stub_src), "-o", str(stub_so)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                return False
        try:
            ctypes.CDLL(str(stub_so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except Exception:
            return False
        return True

    def _import_torch_modules():
        try:
            torch_mod = importlib.import_module("torch")
            nn_mod = importlib.import_module("torch.nn")
            f_mod = importlib.import_module("torch.nn.functional")
            return torch_mod, nn_mod, f_mod
        except ImportError as exc:
            if "iJIT_" not in str(exc):
                raise
            _clear_torch_modules()
            if not _load_local_itt_stub():
                raise
            torch_mod = importlib.import_module("torch")
            nn_mod = importlib.import_module("torch.nn")
            f_mod = importlib.import_module("torch.nn.functional")
            return torch_mod, nn_mod, f_mod

    torch, nn, F = _import_torch_modules()

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

EPS = 1e-8
HAS_CV2_RESIZE = bool(HAS_CV2 and hasattr(cv2, "resize"))
HAS_CV2_IMENCODE = bool(HAS_CV2 and hasattr(cv2, "imencode"))
PROMPT_ANNOTATION_TYPES = ("box", "slice")
BOX_PROMPT_VARIANTS = (
    "tight_components",
    "expand10_components",
    "expand20_components",
    "square10_components",
    "square20_components",
)
BOX_PROMPT_VARIANT_BASELINE = "tight_components"
BOX2MASK_RUNTIME_MODES = (
    "legacy",
    "aligned",
)
BOX2MASK_RUNTIME_MODE_DEFAULT = "aligned"
DEFAULT_EVAL_PROB_THRESH = 0.3
DEFAULT_BOX_DYNAMIC_CROP_RATIO = 1.3
DEFAULT_SLICE_DYNAMIC_CROP_RATIO = 1.5


def _resize_2d_array(
    image: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    mode: str,
) -> np.ndarray:
    arr = np.asarray(image)
    target_dtype = arr.dtype.type
    if arr.ndim != 2:
        raise ValueError(f"_resize_2d_array expects a 2D array, got shape={arr.shape}")
    out_w = int(out_w)
    out_h = int(out_h)
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid resize target: {(out_h, out_w)}")
    if HAS_CV2_RESIZE:
        interpolation = cv2.INTER_LINEAR if str(mode) == "linear" else cv2.INTER_NEAREST
        return cv2.resize(arr, (out_w, out_h), interpolation=interpolation)
    if HAS_TORCH:
        tensor = _to_torch_tensor(np.asarray(arr, dtype=np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kwargs: Dict[str, Any] = {
            "size": (out_h, out_w),
            "mode": "bilinear" if str(mode) == "linear" else "nearest",
        }
        if str(mode) != "nearest":
            kwargs["align_corners"] = False
        resized = F.interpolate(tensor, **kwargs).squeeze(0).squeeze(0).cpu().numpy()
        if str(mode) == "nearest" and np.issubdtype(arr.dtype, np.integer):
            return np.rint(resized).astype(target_dtype)
        if np.issubdtype(arr.dtype, np.floating):
            return np.asarray(resized, dtype=target_dtype)
        return resized.astype(np.float32, copy=False)
    if HAS_SCIPY:
        order = 1 if str(mode) == "linear" else 0
        zoom_factors = (float(out_h) / float(arr.shape[0]), float(out_w) / float(arr.shape[1]))
        resized = ndi.zoom(np.asarray(arr, dtype=np.float32), zoom=zoom_factors, order=order)
        resized = np.asarray(resized)
        if resized.shape != (out_h, out_w):
            resized = resized[:out_h, :out_w]
            if resized.shape != (out_h, out_w):
                padded = np.zeros((out_h, out_w), dtype=resized.dtype)
                padded[: resized.shape[0], : resized.shape[1]] = resized
                resized = padded
        if str(mode) == "nearest" and np.issubdtype(arr.dtype, np.integer):
            return np.rint(resized).astype(target_dtype)
        if np.issubdtype(arr.dtype, np.floating):
            return np.asarray(resized, dtype=target_dtype)
        return resized.astype(np.float32, copy=False)
    raise RuntimeError("No available backend for 2D resize: need cv2.resize, torch, or scipy.ndimage.")


def _to_torch_tensor(array: Any, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(array)
    return torch.tensor(arr.tolist(), dtype=dtype)


def _encode_png_base64(gray_u8: np.ndarray) -> str:
    gray = np.asarray(gray_u8, dtype=np.uint8)
    if gray.ndim == 3 and gray.shape[-1] == 3 and HAS_PIL:
        bio = BytesIO()
        Image.fromarray(gray, mode="RGB").save(bio, format="PNG")
        return base64.b64encode(bio.getvalue()).decode("ascii")
    if HAS_CV2_IMENCODE:
        ok, buf = cv2.imencode(".png", gray)
        if not bool(ok):
            raise RuntimeError("Failed to encode CT slice PNG for Agent.")
        return base64.b64encode(buf.tobytes()).decode("ascii")
    if HAS_PIL:
        bio = BytesIO()
        mode = "RGB" if gray.ndim == 3 and gray.shape[-1] == 3 else "L"
        Image.fromarray(gray, mode=mode).save(bio, format="PNG")
        return base64.b64encode(bio.getvalue()).decode("ascii")
    raise RuntimeError("No available PNG encoder backend: need cv2.imencode or Pillow.")
FINAL_PROMPT_PERTURBATION_SCHEDULE = ("original",)
SELECTION_SOURCE = "classwise_pam_center40_agent_selection"
SELECTION_SCORE_PROFILE = "classwise_pam_center40_agent_v1"
SELECTION_SCORE_WEIGHTS = {
    "confidence": 0.2,
    "structure": 0.3,
    "continuity": 0.5,
}
CENTER_SLICE_FRACTION = 0.40
REQUIRED_LABEL_MIN_AREA_RATIO = 0.001
AGENT_CONFIG_PATH = Path(os.environ.get("DISPAM_AGENT_CONFIG", ""))
AGENT_VERSION_PATH = Path(os.environ.get("CODEX_VERSION_JSON", ""))


def _parse_toml_string_value(raw_value: str) -> str:
    value = str(raw_value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_agent_settings_from_config(path: Path = AGENT_CONFIG_PATH) -> Dict[str, str]:
    if not path.is_file():
        return {}
    provider_name = ""
    top_settings: Dict[str, str] = {}
    provider_settings: Dict[str, Dict[str, str]] = {}
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        parsed_value = _parse_toml_string_value(value)
        if not section:
            if parsed_value and key in {
                "model_provider",
                "model",
                "model_reasoning_effort",
                "reasoning_effort",
            }:
                top_settings[key] = parsed_value
            if key == "model_provider":
                provider_name = parsed_value
        elif section.startswith("model_providers."):
            provider_key = section.split(".", 1)[1].strip().strip('"').strip("'")
            if parsed_value and key in {"base_url", "wire_api"}:
                provider_settings.setdefault(provider_key, {})[key] = parsed_value.rstrip("/")
    settings = dict(top_settings)
    if provider_name and provider_settings.get(provider_name):
        settings.update(provider_settings[provider_name])
        return settings
    if len(provider_settings) == 1:
        settings.update(next(iter(provider_settings.values())))
    return settings


def _extract_api_key_from_mapping(payload: Dict[str, Any]) -> str:
    for key in ("OPENAI_API_KEY", "openai_api_key", "api_key", "DISPAM_AGENT_API_KEY", "dispam_agent_api_key"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            return value
    for value in payload.values():
        if isinstance(value, dict):
            nested = _extract_api_key_from_mapping(value)
            if nested:
                return nested
    return ""


def _iter_embedded_json_objects(text: str) -> List[Dict[str, Any]]:
    rows = text.splitlines()
    idx = 0
    while idx < len(rows):
        if rows[idx].strip().startswith("{"):
            block = [rows[idx]]
            depth = rows[idx].count("{") - rows[idx].count("}")
            idx += 1
            while idx < len(rows) and depth > 0:
                block.append(rows[idx])
                depth += rows[idx].count("{") - rows[idx].count("}")
                idx += 1
            try:
                payload = json.loads("\n".join(block))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload
            continue
        idx += 1


def load_agent_api_key_from_config(path: Path = AGENT_CONFIG_PATH) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    for payload in _iter_embedded_json_objects(text):
        value = _extract_api_key_from_mapping(payload)
        if value:
            return value
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key not in {"OPENAI_API_KEY", "openai_api_key", "api_key", "DISPAM_AGENT_API_KEY", "dispam_agent_api_key"}:
            continue
        parsed_value = _parse_toml_string_value(value)
        if parsed_value:
            return parsed_value
    return ""


_AGENT_CONFIG_SETTINGS = _load_agent_settings_from_config()
AGENT_MODEL_PROVIDER = os.environ.get("DISPAM_AGENT_PROVIDER", _AGENT_CONFIG_SETTINGS.get("model_provider") or "openai")
AGENT_MODEL = os.environ.get("DISPAM_AGENT_MODEL", _AGENT_CONFIG_SETTINGS.get("model") or "gpt-4o")
AGENT_REASONING_EFFORT = (
    _AGENT_CONFIG_SETTINGS.get("model_reasoning_effort")
    or _AGENT_CONFIG_SETTINGS.get("reasoning_effort")
    or "xhigh"
)
AGENT_BASE_URL = os.environ.get("DISPAM_AGENT_BASE_URL", _AGENT_CONFIG_SETTINGS.get("base_url") or "https://api.openai.com/v1")
AGENT_WIRE_API = _AGENT_CONFIG_SETTINGS.get("wire_api") or "chat_completions"
AGENT_FALLBACK_BASE_URLS = tuple(v.strip() for v in os.environ.get("DISPAM_AGENT_FALLBACK_BASE_URLS", "").split(",") if v.strip())
AGENT_USE_SYSTEM_PROXY = False
AGENT_MAX_RETRIES = 3
AGENT_RETRY_BACKOFF_SECONDS = 1.0
AGENT_REQUEST_TIMEOUT_SECONDS = 120.0
AGENT_REFERENCE_EXAMPLES_ENABLED = True
AGENT_REFERENCE_EXAMPLES_PER_BUCKET = 2
AGENT_REFERENCE_CANDIDATE_DEPTH_PER_BUCKET = 16
AGENT_REFERENCE_MIN_SLICE_GAP = 8
AGENT_PROMPT = (
    "You are reviewing a single axial KiTS19 CT slice for a strict class-completeness check.\n\n"
    "Task: Decide whether BOTH target classes are present on this single slice:\n"
    "- class 1\n"
    "- class 2\n\n"
    "Several labeled reference slices may be shown before the target slice. "
    "Each reference includes the correct answer and is provided only as calibration.\n\n"
    "Important calibration rule: a large or very clear class-1-only slice is still NO if no "
    "class 2 is visibly present on that same slice. A small but real class 2 on the same slice "
    "still counts as YES.\n\n"
    "Question: On this slice alone, are BOTH class 1 and class 2 visibly present?\n\n"
    "Answer strictly with one word only: yes or no.\n\n"
    "Answer yes only if:\n"
    "- class 1 is visibly present on this slice, and\n"
    "- class 2 is visibly present on this same slice, and\n"
    "- both classes can be identified from this slice itself without using adjacent slices.\n\n"
    "Answer no if:\n"
    "- the slice shows class 1 but no visible class 2, or\n"
    "- the slice shows a possible class 2 region but no clear class 1, or\n"
    "- either class is absent, ambiguous, extremely tiny, heavily cut off, or not confidently identifiable, or\n"
    "- you are uncertain whether both classes are truly present on this slice.\n\n"
    "Be conservative: if unsure, answer no. Do not infer from neighboring slices.\n\n"
    "Do not explain your answer."
)


def agent_user_agent() -> str:
    env_version = os.environ.get("CODEX_CLI_VERSION", "").strip()
    if env_version:
        return f"codex-cli/{env_version}"
    try:
        payload = json.loads(AGENT_VERSION_PATH.read_text(encoding="utf-8"))
        version = str(payload.get("latest_version", "") or "").strip()
        if version:
            return f"codex-cli/{version}"
    except Exception:
        pass
    return "codex-cli/0.136.0"
ZERO_CAM_FEATURES = {
    "cam_mass_inside_target": 0.0,
    "cam_peak_value": 0.0,
    "cam_entropy": 1.0,
    "cam_center_distance_to_target_center": 1.0,
    "cam_border_fraction": 1.0,
    "cam_topk_mean": 0.0,
    "cam_compactness": 0.0,
    "cam_bbox_area_ratio": 0.0,
    "cam_foreground_alignment_iou": 0.0,
    "cam_area_ratio": 0.0,
    "cam_prompt_mass_inside_target": 0.0,
    "cam_prompt_peak_value": 0.0,
    "cam_prompt_entropy": 1.0,
    "cam_prompt_center_distance_to_target_center": 1.0,
    "cam_prompt_border_fraction": 1.0,
    "cam_prompt_topk_mean": 0.0,
    "cam_prompt_compactness": 0.0,
    "cam_prompt_bbox_area_ratio": 0.0,
    "cam_prompt_foreground_alignment_iou": 0.0,
    "cam_prompt_area_ratio": 0.0,
    "cam_focus_available": 0.0,
    "cam_focus_mass_ratio": 0.0,
    "cam_focus_area_ratio": 0.0,
    "cam_focus_compactness": 0.0,
    "cam_available": 0.0,
}
SliceEvidence = Dict[str, Any]
PromptMeta = Dict[str, Any]


def iter_with_optional_tqdm(items: List[Any], desc: str, unit: str):
    try:
        from tqdm import tqdm

        return tqdm(
            items,
            total=len(items),
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            position=1,
            leave=False,
        )
    except Exception:
        return items


def make_optional_tqdm(total: int, desc: str, unit: str):
    try:
        from tqdm import tqdm

        return tqdm(
            total=int(total),
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            position=1,
            leave=False,
        )
    except Exception:
        return None


def normalize_prompt_annotation_type(prompt_annotation_type: Optional[str]) -> str:
    mode = str(prompt_annotation_type or "box").strip().lower()
    if mode not in PROMPT_ANNOTATION_TYPES:
        raise ValueError(
            f"Unsupported prompt_annotation_type='{prompt_annotation_type}'. "
            f"Expected one of {PROMPT_ANNOTATION_TYPES}."
        )
    return mode

class PAMInferencer(ABC):
    @abstractmethod
    def propagate_prob(
        self, volume: np.ndarray, prompt_slice_id: int, prompt_mask_2d: np.ndarray
    ) -> np.ndarray:
        """Return prob volume [D,H,W] float32 in [0,1]."""

    def prompt_mask_from_box(
        self,
        volume: np.ndarray,
        prompt_slice_id: int,
        box_xyxy: Tuple[int, int, int, int],
        prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    ) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement box-to-mask prompting.")

    def prompt_masks_from_boxes(
        self,
        volume: np.ndarray,
        prompt_slice_id: int,
        boxes_xyxy: List[Tuple[int, int, int, int]],
        prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    ) -> List[np.ndarray]:
        return [
            self.prompt_mask_from_box(
                volume=volume,
                prompt_slice_id=prompt_slice_id,
                box_xyxy=tuple(int(v) for v in box_xyxy),
                prob_thresh=prob_thresh,
            )
            for box_xyxy in boxes_xyxy
        ]

    def propagate_logits(
        self, volume: np.ndarray, prompt_slice_id: int, prompt_mask_2d: np.ndarray
    ) -> np.ndarray:
        """Return logits volume [D,2,H,W] float32."""
        prob = np.clip(
            self.propagate_prob(volume, prompt_slice_id, prompt_mask_2d).astype(np.float32),
            1e-5,
            1.0 - 1e-5,
        )
        fg_logit = np.log(prob) - np.log(1.0 - prob)
        fg_logit = np.clip(fg_logit, -12.0, 12.0).astype(np.float32)
        bg_logit = (-fg_logit).astype(np.float32)
        return np.stack([bg_logit, fg_logit], axis=1).astype(np.float32)

    def propagate_logits_batch(
        self,
        volume: np.ndarray,
        prompt_slice_ids: List[int],
        prompt_masks_2d: List[np.ndarray],
    ) -> List[np.ndarray]:
        return [
            self.propagate_logits(volume, int(slice_id), np.asarray(prompt_mask, dtype=bool))
            for slice_id, prompt_mask in zip(prompt_slice_ids, prompt_masks_2d)
        ]

    def propagate_mask(
        self,
        volume: np.ndarray,
        prompt_slice_id: int,
        prompt_mask_2d: np.ndarray,
        thresh: float = 0.5,
    ) -> np.ndarray:
        """Return hard mask [D,H,W] bool."""
        prob = self.propagate_prob(volume, prompt_slice_id, prompt_mask_2d)
        return prob >= thresh

    def propagate_prob_from_box(
        self,
        volume: np.ndarray,
        prompt_slice_id: int,
        box_xyxy: Tuple[int, int, int, int],
        box_prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    ) -> np.ndarray:
        prompt_mask_2d = self.prompt_mask_from_box(
            volume=volume,
            prompt_slice_id=prompt_slice_id,
            box_xyxy=box_xyxy,
            prob_thresh=box_prob_thresh,
        )
        return self.propagate_prob(volume, prompt_slice_id, prompt_mask_2d)

    def propagate_mask_from_box(
        self,
        volume: np.ndarray,
        prompt_slice_id: int,
        box_xyxy: Tuple[int, int, int, int],
        box_prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
        thresh: float = 0.5,
    ) -> np.ndarray:
        prob = self.propagate_prob_from_box(
            volume=volume,
            prompt_slice_id=prompt_slice_id,
            box_xyxy=box_xyxy,
            box_prob_thresh=box_prob_thresh,
        )
        return prob >= thresh


class RealPAMWrapper(PAMInferencer):
    """
    Real PAM wrapper:
      - Uses PAM encoder features for centrality embeddings.
      - Provides one-pass probability propagation from one 2D prompt.
    """

    def __init__(
        self,
        pam_root: str,
        checkpoint_path: Optional[str] = None,
        box2mask_checkpoint_path: Optional[str] = None,
        box2mask_runtime_mode: str = BOX2MASK_RUNTIME_MODE_DEFAULT,
        load_box2mask: bool = True,
        device: str = "cpu",
        modality: str = "CT",
        target_size: int = 224,
        dynamic_crop_ratio: float = 1.5,
        prop_batch_size: int = 32,
        emb_batch_size: int = 32,
    ):
        self.pam_root = Path(pam_root)
        if checkpoint_path is None:
            checkpoint_path = str(self.pam_root / "tutorials" / "checkpoints" / "propmask.pth")
        self.load_box2mask = bool(load_box2mask)
        if self.load_box2mask and box2mask_checkpoint_path is None:
            box2mask_checkpoint_path = str(
                self.pam_root / "tutorials" / "checkpoints" / "box2mask.pth"
            )
        self.checkpoint_path = Path(checkpoint_path)
        self.box2mask_checkpoint_path = (
            None if box2mask_checkpoint_path is None else Path(box2mask_checkpoint_path)
        )
        self.box2mask_runtime_mode = normalize_box2mask_runtime_mode(box2mask_runtime_mode)
        self.device = device
        self.modality = modality
        self.target_size = int(target_size)
        self.dynamic_crop_ratio = float(dynamic_crop_ratio)
        self.prop_batch_size = int(prop_batch_size)
        self.emb_batch_size = int(emb_batch_size)

        self.model = None
        self.box2mask_model = None
        self._load_model()

    def _load_model(self) -> None:
        if not HAS_TORCH:
            raise RuntimeError("torch is required for RealPAMWrapper.")
        if not self.pam_root.exists():
            raise FileNotFoundError(f"PAM root not found: {self.pam_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PAM checkpoint not found: {self.checkpoint_path}")
        if self.load_box2mask and self.box2mask_checkpoint_path is None:
            raise FileNotFoundError("PAM box2mask checkpoint path is required in box prompt mode.")
        if self.load_box2mask and not self.box2mask_checkpoint_path.exists():
            raise FileNotFoundError(f"PAM box2mask checkpoint not found: {self.box2mask_checkpoint_path}")

        checkpoint_utils = importlib.import_module("PAM.checkpoint_utils")
        checkpoint_utils.prepare_pam_imports(self.pam_root)
        pam_mod = importlib.import_module("model.PAM")
        PAMCls = pam_mod.PAM
        Box2SegCls = pam_mod.Box2SegNet

        model = PAMCls(
            conv_dim=2,
            input_channels=1,
            n_stages=6,
            max_channels=512,
            num_classes=2,
            deep_supervision=True,
            n_attn_stage=4,
            from_scratch_ratio=0.0,
        )
        box2seg_args = argparse.Namespace(
            conv_dim=2,
            input_channels=1,
            n_stages=6,
            max_channels=512,
            num_classes=2,
            deep_supervision=True,
        )
        prop_load_info = checkpoint_utils.load_checkpoint_into_model(
            model=model,
            checkpoint_path=self.checkpoint_path,
            map_location=self.device,
            tag="propmask",
        )
        print(
            f"[Info] RealPAMWrapper loaded propmask checkpoint: "
            f"{prop_load_info['loaded']}/{prop_load_info['model_param_count']} params "
            f"({prop_load_info['load_message']})."
        )

        model = model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        if self.load_box2mask:
            box2mask_model = Box2SegCls(box2seg_args)
            box_load_info = checkpoint_utils.load_checkpoint_into_model(
                model=box2mask_model,
                checkpoint_path=self.box2mask_checkpoint_path,
                map_location=self.device,
                tag="box2mask",
            )
            print(
                f"[Info] RealPAMWrapper loaded box2mask checkpoint: "
                f"{box_load_info['loaded']}/{box_load_info['model_param_count']} params "
                f"({box_load_info['load_message']})."
            )
            box2mask_model = box2mask_model.to(self.device)
            box2mask_model.eval()
            for p in box2mask_model.parameters():
                p.requires_grad_(False)
            self.box2mask_model = box2mask_model

    @staticmethod
    def _window_transform(img: np.ndarray, center: float, width: float) -> np.ndarray:
        low = center - width / 2.0
        high = center + width / 2.0
        return np.clip(img, low, high)

    @staticmethod
    def _internal_transform(img: np.ndarray, minv: float, maxv: float) -> np.ndarray:
        return np.clip(img, minv, maxv)

    @staticmethod
    def _normalize_uint8(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        imin = float(np.min(image))
        imax = float(np.max(image))
        if imax - imin < 1e-6:
            return np.zeros_like(image, dtype=np.uint8)
        out = (image - imin) / (imax - imin + EPS)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _preprocess_to_model_tensor(img_u8: np.ndarray, target_size: int) -> torch.Tensor:
        img_resized = _resize_2d_array(img_u8, target_size, target_size, mode="linear")
        img_f = img_resized.astype(np.float32) / 255.0
        img_f = (img_f - 0.5) / 0.5  # same as A.Normalize(mean=[0.5], std=[0.5]) for uint8 input
        return _to_torch_tensor(img_f, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

    @staticmethod
    def _normalize_crop_unit_float(
        image: np.ndarray,
        low_pct: float = 5.0,
        high_pct: float = 95.0,
    ) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(arr, low_pct))
        hi = float(np.percentile(arr, high_pct))
        if hi - lo < 1e-6:
            lo = float(np.min(arr))
            hi = float(np.max(arr))
        if hi - lo < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, lo, hi)
        return np.clip((arr - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _preprocess_float_to_model_tensor(img_unit: np.ndarray, target_size: int) -> torch.Tensor:
        img_resized = _resize_2d_array(
            np.asarray(img_unit, dtype=np.float32),
            target_size,
            target_size,
            mode="linear",
        ).astype(np.float32)
        img_f = np.clip(img_resized, 0.0, 1.0)
        img_f = (img_f - 0.5) / 0.5
        return _to_torch_tensor(img_f, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

    @staticmethod
    def _crop_bbox_from_mask(mask: np.ndarray, ratio: float = 1.5, min_size: int = 16) -> Tuple[int, int, int, int]:
        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return 0, 0, w, h
        y1 = int(np.min(ys))
        y2 = int(np.max(ys))
        x1 = int(np.min(xs))
        x2 = int(np.max(xs))
        cy = int((y1 + y2) / 2.0)
        cx = int((x1 + x2) / 2.0)
        box_h = y2 - y1 + 1
        box_w = x2 - x1 + 1
        crop_size = int(max(min_size, ratio * max(box_h, box_w)))
        sx = max(0, cx - crop_size // 2)
        sy = max(0, cy - crop_size // 2)
        ex = min(w, sx + crop_size)
        ey = min(h, sy + crop_size)
        sx = max(0, ex - crop_size)
        sy = max(0, ey - crop_size)
        return sx, sy, ex, ey

    @staticmethod
    def _clip_box_xyxy(
        box_xyxy: Tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        if len(box_xyxy) != 4:
            raise ValueError(f"box_xyxy must have 4 elements, got {box_xyxy}")
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        x1 = int(np.clip(x1, 0, max(0, width - 1)))
        y1 = int(np.clip(y1, 0, max(0, height - 1)))
        x2 = int(np.clip(x2, x1 + 1, width))
        y2 = int(np.clip(y2, y1 + 1, height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid clipped box_xyxy: {(x1, y1, x2, y2)}")
        return x1, y1, x2, y2

    @staticmethod
    def _crop_bbox_from_box(
        box_xyxy: Tuple[int, int, int, int],
        width: int,
        height: int,
        ratio: float = 1.5,
        min_size: int = 16,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = RealPAMWrapper._clip_box_xyxy(box_xyxy, width=width, height=height)
        cx = 0.5 * float(x1 + x2)
        cy = 0.5 * float(y1 + y2)
        box_w = int(max(1, x2 - x1))
        box_h = int(max(1, y2 - y1))
        crop_size = int(max(min_size, math.ceil(float(ratio) * max(box_h, box_w))))
        sx = int(math.floor(cx - crop_size / 2.0))
        sy = int(math.floor(cy - crop_size / 2.0))
        ex = sx + crop_size
        ey = sy + crop_size
        if sx < 0:
            ex -= sx
            sx = 0
        if sy < 0:
            ey -= sy
            sy = 0
        if ex > width:
            sx -= ex - width
            ex = width
        if ey > height:
            sy -= ey - height
            ey = height
        sx = int(max(0, sx))
        sy = int(max(0, sy))
        ex = int(min(width, ex))
        ey = int(min(height, ey))
        if ex <= sx or ey <= sy:
            raise RuntimeError(f"Invalid crop from box {box_xyxy}: {(sx, sy, ex, ey)}")
        return sx, sy, ex, ey

    @staticmethod
    def _build_box_mask(
        height: int,
        width: int,
        box_xyxy: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = RealPAMWrapper._clip_box_xyxy(box_xyxy, width=width, height=height)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    @staticmethod
    def _merge_ds_logits(predictions: object) -> torch.Tensor:
        if isinstance(predictions, torch.Tensor):
            return predictions
        if isinstance(predictions, (list, tuple)):
            if len(predictions) == 0:
                raise RuntimeError("Empty predictions list from PAM model.")
            pred_size = predictions[0].shape[-2:]
            merged: List[torch.Tensor] = []
            for p in predictions:
                if p.shape[-2:] != pred_size:
                    p = F.interpolate(p, size=pred_size, mode="bilinear", align_corners=True)
                merged.append(p)
            return torch.mean(torch.stack(merged, dim=0), dim=0)
        raise TypeError(f"Unsupported predictions type: {type(predictions)}")

    def _compute_modality_normalization(
        self, guiding_img: np.ndarray, guiding_mask: np.ndarray
    ) -> Tuple[float, float]:
        fg = guiding_img[guiding_mask > 0]
        if fg.size == 0:
            fg = guiding_img.reshape(-1)
        if self.modality.upper() == "CT":
            lo = float(np.percentile(fg, 0.5))
            hi = float(np.percentile(fg, 99.5))
            center = (lo + hi) / 2.0
            width = max(1.0, hi - lo)
            return center, width
        minv = float(np.percentile(fg, 0.5))
        maxv = float(np.percentile(fg, 99.5))
        if maxv - minv < 1e-6:
            maxv = minv + 1.0
        return minv, maxv

    def _apply_modality_transform(self, img: np.ndarray, norm_a: float, norm_b: float) -> np.ndarray:
        if self.modality.upper() == "CT":
            x = self._window_transform(img, norm_a, norm_b)
        else:
            x = self._internal_transform(img, norm_a, norm_b)
        return self._normalize_uint8(x)

    def propagate_logits(
        self, volume: np.ndarray, prompt_slice_id: int, prompt_mask_2d: np.ndarray
    ) -> np.ndarray:
        """
        One-pass PAM propagation from one prompt slice to all other slices.
        Args:
            volume: [D,H,W] float32
            prompt_slice_id: int
            prompt_mask_2d: [H,W] bool
        Returns:
            logits: [D,2,H,W] float32
        """
        if self.model is None:
            raise RuntimeError("Real PAM model not loaded.")

        vol = np.asarray(volume, dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"volume must be [D,H,W], got {vol.shape}")
        d, h, w = vol.shape
        z = int(np.clip(prompt_slice_id, 0, d - 1))

        mask = np.asarray(prompt_mask_2d > 0, dtype=np.uint8)
        if mask.shape != (h, w):
            raise ValueError(f"prompt mask shape mismatch: expected {(h,w)}, got {mask.shape}")
        if not mask.any():
            raise ValueError("prompt_mask_2d must contain foreground pixels.")

        guiding_img = vol[z]
        norm_a, norm_b = self._compute_modality_normalization(guiding_img, mask)
        sx, sy, ex, ey = self._crop_bbox_from_mask(mask, ratio=self.dynamic_crop_ratio, min_size=16)
        crop_h = ey - sy
        crop_w = ex - sx
        if crop_h <= 0 or crop_w <= 0:
            raise RuntimeError("Invalid crop size from prompt mask.")

        guiding_u8 = self._apply_modality_transform(guiding_img, norm_a, norm_b)
        guiding_crop = guiding_u8[sy:ey, sx:ex]
        guiding_t = self._preprocess_to_model_tensor(guiding_crop, self.target_size)

        mask_crop = mask[sy:ey, sx:ex].astype(np.float32)
        mask_rs = _resize_2d_array(mask_crop, self.target_size, self.target_size, mode="nearest")
        guiding_prp_t = _to_torch_tensor((mask_rs > 0.5).astype(np.float32), dtype=torch.float32)

        logits_vol = np.zeros((d, 2, h, w), dtype=np.float32)
        logits_vol[:, 0] = 6.0
        logits_vol[:, 1] = -6.0
        logits_vol[z, 0] = np.where(mask > 0, -6.0, 6.0).astype(np.float32)
        logits_vol[z, 1] = np.where(mask > 0, 6.0, -6.0).astype(np.float32)

        adjacent_ids = [i for i in range(d) if i != z]
        if len(adjacent_ids) == 0:
            return logits_vol.astype(np.float32)

        with torch.no_grad():
            for st in range(0, len(adjacent_ids), self.prop_batch_size):
                ids = adjacent_ids[st : st + self.prop_batch_size]
                adj_tensors = []
                for sid in ids:
                    img_u8 = self._apply_modality_transform(vol[sid], norm_a, norm_b)
                    crop = img_u8[sy:ey, sx:ex]
                    adj_tensors.append(self._preprocess_to_model_tensor(crop, self.target_size))
                adjacent_x = torch.stack(adj_tensors, dim=0)  # [N,1,H,W]

                batch = {
                    "guiding_x": guiding_t.unsqueeze(0).to(self.device),  # [1,1,H,W]
                    "guiding_prp": guiding_prp_t.unsqueeze(0).to(self.device),  # [1,H,W]
                    "adjacent_x": adjacent_x.unsqueeze(0).to(self.device),  # [1,N,1,H,W]
                }
                pred_dict = self.model(batch)
                logits = self._merge_ds_logits(pred_dict["predictions"])  # [N,2,h,w]
                logits_np = logits.detach().cpu().numpy().astype(np.float32)

                for sid, one_logit in zip(ids, logits_np):
                    resized = np.zeros((2, crop_h, crop_w), dtype=np.float32)
                    for cls in range(2):
                        resized[cls] = _resize_2d_array(
                            one_logit[cls],
                            crop_w,
                            crop_h,
                            mode="linear",
                        ).astype(np.float32)
                    logits_vol[sid, :, sy:ey, sx:ex] = resized

        return logits_vol.astype(np.float32)

    def propagate_logits_batch(
        self,
        volume: np.ndarray,
        prompt_slice_ids: List[int],
        prompt_masks_2d: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Batched PAM propagation for multiple prompt slices on the same volume.

        The model already supports a prompt batch dimension B:
        guiding_x=[B,1,H,W], guiding_prp=[B,H,W], adjacent_x=[B,Nq,1,H,W].
        """
        if self.model is None:
            raise RuntimeError("Real PAM model not loaded.")
        if len(prompt_slice_ids) != len(prompt_masks_2d):
            raise ValueError(
                f"prompt_slice_ids/prompt_masks length mismatch: "
                f"{len(prompt_slice_ids)} vs {len(prompt_masks_2d)}"
            )
        if len(prompt_slice_ids) <= 0:
            return []

        vol = np.asarray(volume, dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"volume must be [D,H,W], got {vol.shape}")
        d, h, w = vol.shape

        prompt_packs: List[Dict[str, Any]] = []
        logits_vols: List[np.ndarray] = []
        for prompt_slice_id, prompt_mask_2d in zip(prompt_slice_ids, prompt_masks_2d):
            z = int(np.clip(int(prompt_slice_id), 0, d - 1))
            mask = np.asarray(prompt_mask_2d > 0, dtype=np.uint8)
            if mask.shape != (h, w):
                raise ValueError(f"prompt mask shape mismatch: expected {(h,w)}, got {mask.shape}")
            if not mask.any():
                raise ValueError("prompt_mask_2d must contain foreground pixels.")

            guiding_img = vol[z]
            norm_a, norm_b = self._compute_modality_normalization(guiding_img, mask)
            sx, sy, ex, ey = self._crop_bbox_from_mask(
                mask,
                ratio=self.dynamic_crop_ratio,
                min_size=16,
            )
            crop_h = int(ey - sy)
            crop_w = int(ex - sx)
            if crop_h <= 0 or crop_w <= 0:
                raise RuntimeError("Invalid crop size from prompt mask.")

            guiding_u8 = self._apply_modality_transform(guiding_img, norm_a, norm_b)
            guiding_crop = guiding_u8[sy:ey, sx:ex]
            guiding_t = self._preprocess_to_model_tensor(guiding_crop, self.target_size)

            mask_crop = mask[sy:ey, sx:ex].astype(np.float32)
            mask_rs = _resize_2d_array(
                mask_crop,
                self.target_size,
                self.target_size,
                mode="nearest",
            )
            guiding_prp_t = _to_torch_tensor((mask_rs > 0.5).astype(np.float32), dtype=torch.float32)

            logits_vol = np.zeros((d, 2, h, w), dtype=np.float32)
            logits_vol[:, 0] = 6.0
            logits_vol[:, 1] = -6.0
            logits_vol[z, 0] = np.where(mask > 0, -6.0, 6.0).astype(np.float32)
            logits_vol[z, 1] = np.where(mask > 0, 6.0, -6.0).astype(np.float32)
            logits_vols.append(logits_vol)
            prompt_packs.append(
                {
                    "z": z,
                    "mask": mask,
                    "norm_a": norm_a,
                    "norm_b": norm_b,
                    "sx": int(sx),
                    "sy": int(sy),
                    "ex": int(ex),
                    "ey": int(ey),
                    "crop_h": crop_h,
                    "crop_w": crop_w,
                    "guiding_t": guiding_t,
                    "guiding_prp_t": guiding_prp_t,
                }
            )

        all_slice_ids = list(range(d))
        with torch.no_grad():
            for st in range(0, len(all_slice_ids), self.prop_batch_size):
                ids = all_slice_ids[st : st + self.prop_batch_size]
                adjacent_batches: List[torch.Tensor] = []
                for pack in prompt_packs:
                    adj_tensors = []
                    sx = int(pack["sx"])
                    sy = int(pack["sy"])
                    ex = int(pack["ex"])
                    ey = int(pack["ey"])
                    norm_a = float(pack["norm_a"])
                    norm_b = float(pack["norm_b"])
                    for sid in ids:
                        img_u8 = self._apply_modality_transform(vol[int(sid)], norm_a, norm_b)
                        crop = img_u8[sy:ey, sx:ex]
                        adj_tensors.append(
                            self._preprocess_to_model_tensor(crop, self.target_size)
                        )
                    adjacent_batches.append(torch.stack(adj_tensors, dim=0))

                batch = {
                    "guiding_x": torch.stack(
                        [pack["guiding_t"] for pack in prompt_packs],
                        dim=0,
                    ).to(self.device),
                    "guiding_prp": torch.stack(
                        [pack["guiding_prp_t"] for pack in prompt_packs],
                        dim=0,
                    ).to(self.device),
                    "adjacent_x": torch.stack(adjacent_batches, dim=0).to(self.device),
                }
                pred_dict = self.model(batch)
                logits = self._merge_ds_logits(pred_dict["predictions"])
                batch_size = len(prompt_packs)
                if logits.ndim != 4 or logits.shape[0] != batch_size * len(ids):
                    raise RuntimeError(
                        f"Unexpected batched PAM logits shape: {tuple(logits.shape)}, "
                        f"expected first dim={batch_size * len(ids)}"
                    )
                logits = logits.view(batch_size, len(ids), 2, *logits.shape[-2:])
                logits_np = logits.detach().cpu().numpy().astype(np.float32)

                for batch_idx, pack in enumerate(prompt_packs):
                    sx = int(pack["sx"])
                    sy = int(pack["sy"])
                    ex = int(pack["ex"])
                    ey = int(pack["ey"])
                    crop_h = int(pack["crop_h"])
                    crop_w = int(pack["crop_w"])
                    prompt_z = int(pack["z"])
                    for local_idx, sid in enumerate(ids):
                        sid = int(sid)
                        if sid == prompt_z:
                            continue
                        one_logit = logits_np[batch_idx, local_idx]
                        resized = np.zeros((2, crop_h, crop_w), dtype=np.float32)
                        for cls in range(2):
                            resized[cls] = _resize_2d_array(
                                one_logit[cls],
                                crop_w,
                                crop_h,
                                mode="linear",
                            ).astype(np.float32)
                        logits_vols[batch_idx][sid, :, sy:ey, sx:ex] = resized

        return [np.asarray(logits_vol, dtype=np.float32) for logits_vol in logits_vols]

    def propagate_prob(
        self, volume: np.ndarray, prompt_slice_id: int, prompt_mask_2d: np.ndarray
    ) -> np.ndarray:
        logits = self.propagate_logits(volume, prompt_slice_id, prompt_mask_2d)
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        prob = exp_logits[:, 1] / (np.sum(exp_logits, axis=1) + EPS)
        return np.clip(prob.astype(np.float32), 0.0, 1.0)

    def prompt_masks_from_boxes(
        self,
        volume: np.ndarray,
        prompt_slice_id: int,
        boxes_xyxy: List[Tuple[int, int, int, int]],
        prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    ) -> List[np.ndarray]:
        if self.box2mask_model is None:
            raise RuntimeError("Real PAM box2mask model not loaded.")

        vol = np.asarray(volume, dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"volume must be [D,H,W], got {vol.shape}")
        d, h, w = vol.shape
        z = int(np.clip(prompt_slice_id, 0, d - 1))
        guiding_img = vol[z]
        if len(boxes_xyxy) <= 0:
            raise ValueError("prompt_masks_from_boxes requires at least one box.")
        query_xs: List[torch.Tensor] = []
        crop_infos: List[Dict[str, Any]] = []
        for box_xyxy in boxes_xyxy:
            box_tuple = tuple(int(v) for v in box_xyxy)
            if self.box2mask_runtime_mode == "aligned":
                sx, sy, ex, ey = self._clip_box_xyxy(box_tuple, width=w, height=h)
                guiding_crop = np.asarray(guiding_img[sy:ey, sx:ex], dtype=np.float32)
                guiding_crop = self._normalize_crop_unit_float(guiding_crop, low_pct=5.0, high_pct=95.0)
                query_xs.append(self._preprocess_float_to_model_tensor(guiding_crop, self.target_size))
            else:
                box_mask = self._build_box_mask(height=h, width=w, box_xyxy=box_tuple)
                if not np.any(box_mask):
                    raise ValueError(f"box_xyxy produced an empty box mask: {box_tuple}")
                norm_a, norm_b = self._compute_modality_normalization(guiding_img, box_mask)
                guiding_u8 = self._apply_modality_transform(guiding_img, norm_a, norm_b)
                sx, sy, ex, ey = self._crop_bbox_from_box(
                    box_xyxy=box_tuple,
                    width=w,
                    height=h,
                    ratio=self.dynamic_crop_ratio,
                    min_size=16,
                )
                guiding_crop = guiding_u8[sy:ey, sx:ex]
                query_xs.append(self._preprocess_to_model_tensor(guiding_crop, self.target_size))
            crop_h = int(ey - sy)
            crop_w = int(ex - sx)
            if crop_h <= 0 or crop_w <= 0:
                raise RuntimeError(f"Invalid box2mask crop size from box {box_tuple}.")
            crop_infos.append(
                {
                    "sx": int(sx),
                    "sy": int(sy),
                    "ex": int(ex),
                    "ey": int(ey),
                    "box_tuple": box_tuple,
                    "crop_h": int(crop_h),
                    "crop_w": int(crop_w),
                }
            )
        batch = {
            "query_xs": torch.stack(query_xs, dim=0).unsqueeze(0).to(self.device),  # [1,N,1,H,W]
        }
        with torch.no_grad():
            pred_dict = self.box2mask_model(batch)
            logits = self._merge_ds_logits(pred_dict["predictions"])
            if logits.ndim != 4 or logits.shape[0] != len(crop_infos) or logits.shape[1] != 2:
                raise RuntimeError(
                    f"Unexpected box2mask logits shape: {tuple(logits.shape)}"
                )
            logits = logits.detach().cpu().numpy().astype(np.float32)

        prompt_masks: List[np.ndarray] = []
        for one_logits, info in zip(logits, crop_infos):
            sx = int(info["sx"])
            sy = int(info["sy"])
            ex = int(info["ex"])
            ey = int(info["ey"])
            crop_h = int(info["crop_h"])
            crop_w = int(info["crop_w"])
            box_tuple = tuple(int(v) for v in info["box_tuple"])
            prompt_mask = np.zeros((h, w), dtype=np.uint8)
            if self.box2mask_runtime_mode == "aligned":
                pred_mask = np.argmax(one_logits, axis=0).astype(np.float32)
                pred_mask = _resize_2d_array(
                    pred_mask,
                    crop_w,
                    crop_h,
                    mode="nearest",
                ).astype(np.float32)
                prompt_mask[sy:ey, sx:ex] = (pred_mask > 0.5).astype(np.uint8)
            else:
                logits_max = np.max(one_logits, axis=0, keepdims=True)
                exp_logits = np.exp(one_logits - logits_max)
                fg_prob = exp_logits[1] / (np.sum(exp_logits, axis=0) + EPS)
                fg_prob = _resize_2d_array(
                    fg_prob.astype(np.float32),
                    crop_w,
                    crop_h,
                    mode="linear",
                ).astype(np.float32)
                prompt_mask[sy:ey, sx:ex] = (fg_prob >= float(prob_thresh)).astype(np.uint8)
            if int(np.sum(prompt_mask)) <= 0:
                raise RuntimeError(
                    f"box2mask predicted empty prompt mask for slice={z}, box={box_tuple}."
                )
            prompt_masks.append(prompt_mask.astype(np.uint8))
        return prompt_masks

    def prompt_mask_from_box(
        self,
        volume: np.ndarray,
        prompt_slice_id: int,
        box_xyxy: Tuple[int, int, int, int],
        prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    ) -> np.ndarray:
        return self.prompt_masks_from_boxes(
            volume=volume,
            prompt_slice_id=prompt_slice_id,
            boxes_xyxy=[tuple(int(v) for v in box_xyxy)],
            prob_thresh=prob_thresh,
        )[0]

    def extract_slice_embeddings(self, volume: np.ndarray, device: str = "cpu") -> Optional[np.ndarray]:
        """
        Extract per-slice embedding from real PAM encoder.
        Args:
            volume: [D,H,W]
        Returns:
            emb: [D,512] or None.
        """
        if self.model is None:
            return None
        if not HAS_TORCH:
            return None

        vol = np.asarray(volume, dtype=np.float32)
        d = vol.shape[0]
        p1, p99 = np.percentile(vol, [1, 99])
        x = np.clip((vol - p1) / (p99 - p1 + EPS), 0.0, 1.0).astype(np.float32)

        embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, d, self.emb_batch_size):
                batch = _to_torch_tensor(x[i : i + self.emb_batch_size], dtype=torch.float32).to(self.device).unsqueeze(1)  # [B,1,H,W]
                batch = F.interpolate(batch, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
                batch = (batch - 0.5) / 0.5
                skips = self.model.proposing_stage.encoder(batch)
                feat = skips[-1].mean(dim=(2, 3))  # [B,512]
                embs.append(feat.detach().cpu().numpy().astype(np.float32))
        if len(embs) == 0:
            return None
        return np.concatenate(embs, axis=0)


def build_selector_config(
    *,
    device: str,
    pam_root: str,
    pam_ckpt: Optional[str] = None,
    pam_box2mask_ckpt: Optional[str] = None,
    pam_box2mask_runtime_mode: str = BOX2MASK_RUNTIME_MODE_DEFAULT,
    prompt_annotation_type: str = "box",
    box_prompt_variant: str = BOX_PROMPT_VARIANT_BASELINE,
    seed: int = 0,
    eval_prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    final_core_support_thresh: float = 0.70,
    final_candidate_support_thresh: float = 0.25,
    final_keep_lcc: int = 1,
) -> SimpleNamespace:
    normalized_prompt_type = normalize_prompt_annotation_type(prompt_annotation_type)
    default_dynamic_crop_ratio = (
        DEFAULT_BOX_DYNAMIC_CROP_RATIO
        if normalized_prompt_type == "box"
        else DEFAULT_SLICE_DYNAMIC_CROP_RATIO
    )
    config = {
        "device": str(device),
        "pam_backend": "real",
        "pam_root": str(pam_root),
        "pam_ckpt": pam_ckpt,
        "pam_box2mask_ckpt": pam_box2mask_ckpt,
        "pam_box2mask_runtime_mode": normalize_box2mask_runtime_mode(pam_box2mask_runtime_mode),
        "prompt_annotation_type": normalized_prompt_type,
        "box_prompt_variant": normalize_box_prompt_variant(box_prompt_variant),
        "pam_modality": "CT",
        "pam_target_size": 224,
        "pam_dynamic_crop_ratio": float(default_dynamic_crop_ratio),
        "pam_prop_batch_size": 64,
        "pam_prompt_batch_size": 8,
        "pam_auto_batch_retry": 1,
        "pam_emb_batch_size": 64,
        "agent_model_provider": AGENT_MODEL_PROVIDER,
        "agent_model": AGENT_MODEL,
        "agent_reasoning_effort": AGENT_REASONING_EFFORT,
        "agent_base_url": AGENT_BASE_URL,
        "agent_fallback_base_urls": list(AGENT_FALLBACK_BASE_URLS),
        "agent_use_system_proxy": int(AGENT_USE_SYSTEM_PROXY),
        "agent_prompt": AGENT_PROMPT,
        "agent_reference_examples_enabled": int(AGENT_REFERENCE_EXAMPLES_ENABLED),
        "agent_reference_examples_per_bucket": int(AGENT_REFERENCE_EXAMPLES_PER_BUCKET),
        "agent_reference_candidate_depth_per_bucket": int(
            AGENT_REFERENCE_CANDIDATE_DEPTH_PER_BUCKET
        ),
        "agent_reference_min_slice_gap": int(AGENT_REFERENCE_MIN_SLICE_GAP),
        "agent_disable_response_storage": 1,
        "agent_max_retries": int(AGENT_MAX_RETRIES),
        "agent_retry_backoff_seconds": float(AGENT_RETRY_BACKOFF_SECONDS),
        "agent_request_timeout_seconds": float(AGENT_REQUEST_TIMEOUT_SECONDS),
        "eval_prob_thresh": float(eval_prob_thresh),
        "final_core_support_thresh": float(final_core_support_thresh),
        "final_candidate_support_thresh": float(final_candidate_support_thresh),
        "final_keep_lcc": int(final_keep_lcc),
        "seed": int(seed),
        "topn": 10,
    }
    return SimpleNamespace(**config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def infer_nifti_depth_axis(shape: Tuple[int, int, int]) -> int:
    dims = tuple(int(v) for v in shape)
    equal_pairs = [
        (pair[0], pair[1], pair[2])
        for pair in ((0, 1, 2), (0, 2, 1), (1, 2, 0))
        if dims[pair[0]] == dims[pair[1]]
    ]
    if equal_pairs:
        inplane_a, inplane_b, depth_axis = max(
            equal_pairs,
            key=lambda item: (dims[item[0]] * dims[item[1]], -dims[item[2]]),
        )
        del inplane_a, inplane_b
        return int(depth_axis)
    return 0


def load_volume(path: Path) -> np.ndarray:
    """
    Returns:
        volume: [D,H,W] float32
    """
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    name = path.name.lower()
    if name.endswith(".npy"):
        arr = np.load(str(path))
    elif name.endswith(".nrrd") or name.endswith(".nhdr"):
        arr = load_nrrd_volume(path)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        if not HAS_NIB:
            raise RuntimeError("nibabel is not installed; cannot read .nii/.nii.gz.")
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"NIfTI must be 3D, got shape={arr.shape}")
        depth_axis = infer_nifti_depth_axis(tuple(int(v) for v in arr.shape))
        arr = np.moveaxis(arr, depth_axis, 0)
    else:
        raise ValueError(f"Unsupported volume format: {path.suffixes}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Volume must be 3D [D,H,W], got shape={arr.shape}")
    return arr

def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    if not HAS_SCIPY:
        return mask
    labeled, n = ndi.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


def keep_top_components(mask: np.ndarray, keep_k: int = 1) -> np.ndarray:
    """Keep top-k connected components by area."""
    m = np.asarray(mask, dtype=bool)
    if keep_k <= 1:
        return largest_connected_component(m)
    if (not m.any()) or (not HAS_SCIPY):
        return m
    labeled, n = ndi.label(m)
    if n <= 1:
        return m
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    top_ids = np.argsort(counts)[::-1][: int(keep_k)]
    top_ids = top_ids[top_ids > 0]
    if top_ids.size == 0:
        return np.zeros_like(m, dtype=bool)
    return np.isin(labeled, top_ids)


def _mask_center_score(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0.0
    h, w = m.shape
    pts = np.argwhere(m)
    cy = float(np.mean(pts[:, 0]))
    cx = float(np.mean(pts[:, 1]))
    ny = (cy - (h - 1) / 2.0) / (h / 2.0 + EPS)
    nx = (cx - (w - 1) / 2.0) / (w / 2.0 + EPS)
    dist = float(math.sqrt(nx * nx + ny * ny))
    return float(math.exp(-(dist * dist) / (2.0 * 0.75 * 0.75)))


def _mask_border_fraction(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 1.0
    border = np.zeros_like(m, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return float(np.sum(m & border) / (np.sum(m) + EPS))


def perturb_binary_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    strength: str = "weak",
) -> np.ndarray:
    """
    Args:
        mask: [H,W] bool
    Returns:
        perturbed_mask: [H,W] bool
    """
    del rng
    original = mask.astype(bool).copy()
    m = original.copy()
    if not m.any():
        raise ValueError("Cannot perturb an empty binary mask.")

    mode = str(strength).strip().lower()
    if HAS_SCIPY:
        if mode == "weak":
            m = ndi.binary_dilation(original, iterations=1)
        elif mode == "strong":
            m = ndi.binary_erosion(original, iterations=1)
        else:
            raise ValueError(f"Unsupported prompt perturbation strength: {strength}")
    else:
        if mode == "weak":
            padded = np.pad(original, 1, mode="constant", constant_values=False)
            m = np.zeros_like(original, dtype=bool)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    m |= padded[1 + dy : 1 + dy + original.shape[0], 1 + dx : 1 + dx + original.shape[1]]
        elif mode == "strong":
            padded = np.pad(original, 1, mode="constant", constant_values=False)
            m = np.ones_like(original, dtype=bool)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    m &= padded[1 + dy : 1 + dy + original.shape[0], 1 + dx : 1 + dx + original.shape[1]]
        else:
            raise ValueError(f"Unsupported prompt perturbation strength: {strength}")

    m = largest_connected_component(np.asarray(m, dtype=bool)) if np.any(m) else m
    if np.any(m):
        return np.asarray(m, dtype=bool)

    if HAS_SCIPY:
        dist = ndi.distance_transform_edt(original)
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    else:
        pts = np.argwhere(original)
        center = np.mean(pts, axis=0)
        y, x = pts[int(np.argmin(np.sum((pts - center) ** 2, axis=1)))]
    seed = np.zeros_like(original, dtype=bool)
    seed[int(y), int(x)] = True
    return seed


def build_prompt_perturbation_schedule(
    perturbation_schedule: Optional[Sequence[str]] = None,
) -> List[str]:
    raw_values = (
        list(FINAL_PROMPT_PERTURBATION_SCHEDULE)
        if perturbation_schedule is None
        else [str(value) for value in perturbation_schedule]
    )
    schedule = [str(value).strip().lower() for value in raw_values if str(value).strip()]
    if not schedule:
        schedule = ["original"]
    allowed = {"original", "weak", "strong"}
    invalid = [value for value in schedule if value not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid final prompt perturbation values: {invalid}. Expected values from {sorted(allowed)}."
        )
    if "original" not in schedule:
        schedule.insert(0, "original")
    return schedule


def _keep_components_touching_seed(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool) & m
    if (not m.any()) or (not seed.any()):
        return np.zeros_like(m, dtype=bool)
    if not HAS_SCIPY:
        return m
    labeled, n = ndi.label(m)
    if n <= 1:
        return m
    keep_ids = np.unique(labeled[seed])
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size <= 0:
        return np.zeros_like(m, dtype=bool)
    return np.isin(labeled, keep_ids)


def run_final_prompt_ensemble(
    volume: np.ndarray,
    pam: PAMInferencer,
    prompt_slice_id: int,
    prompt_mask_2d: np.ndarray,
    prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    core_support_thresh: float = 0.70,
    candidate_support_thresh: float = 0.25,
    keep_lcc: bool = True,
    seed: int = 0,
    perturbation_schedule: Optional[Sequence[str]] = None,
    anchor_to_prompt: bool = False,
) -> Dict[str, np.ndarray | int | float]:
    prompt_mask = np.asarray(prompt_mask_2d, dtype=bool)
    if not prompt_mask.any():
        raise ValueError("Final prompt ensemble requires a non-empty prompt mask.")

    perturbation_schedule = build_prompt_perturbation_schedule(perturbation_schedule)
    n_runs = len(perturbation_schedule)
    rng = np.random.default_rng(seed)
    sum_prob: Optional[np.ndarray] = None
    hit_count: Optional[np.ndarray] = None

    for run_idx, perturbation_strength in enumerate(perturbation_schedule):
        if perturbation_strength == "original":
            run_prompt = prompt_mask
        else:
            run_prompt = perturb_binary_mask(prompt_mask, rng, strength=perturbation_strength)
        prob = np.asarray(
            pam.propagate_prob(volume, int(prompt_slice_id), run_prompt.astype(np.uint8)),
            dtype=np.float32,
        )
        if sum_prob is None:
            sum_prob = np.zeros_like(prob, dtype=np.float32)
            hit_count = np.zeros_like(prob, dtype=np.uint16)
        sum_prob += prob
        hit_count += (prob >= float(prob_thresh)).astype(np.uint16)

    if sum_prob is None or hit_count is None:
        raise RuntimeError("Final prompt ensemble produced no predictions.")

    mean_prob = (sum_prob / float(n_runs)).astype(np.float32)
    support_ratio = (hit_count.astype(np.float32) / float(n_runs)).astype(np.float32)

    core_mask = support_ratio >= float(core_support_thresh)
    if not core_mask.any():
        core_mask = support_ratio >= 0.50
    candidate_mask = (
        (mean_prob >= float(prob_thresh))
        & (support_ratio >= float(candidate_support_thresh))
    )
    candidate_mask |= core_mask

    prompt_seed_mask = np.zeros_like(candidate_mask, dtype=bool)
    prompt_seed_mask[int(prompt_slice_id)] = prompt_mask
    seed_mask = (
        prompt_seed_mask
        if bool(anchor_to_prompt)
        else np.asarray(np.logical_or(core_mask, prompt_seed_mask), dtype=bool)
    )

    if seed_mask.any() and candidate_mask.any():
        final_mask = _keep_components_touching_seed(candidate_mask, seed_mask)
        if not final_mask.any():
            final_mask = prompt_seed_mask.copy() if bool(anchor_to_prompt) else candidate_mask.copy()
    elif candidate_mask.any():
        final_mask = candidate_mask.copy()
    else:
        final_mask = mean_prob >= float(prob_thresh)
        if bool(anchor_to_prompt):
            final_mask = _keep_components_touching_seed(final_mask, prompt_seed_mask)
            if not final_mask.any():
                final_mask = prompt_seed_mask.copy()
        core_mask = final_mask.copy()

    if bool(keep_lcc) and not bool(anchor_to_prompt) and final_mask.any():
        final_mask = largest_connected_component(final_mask)
        core_mask &= final_mask

    if bool(anchor_to_prompt) and final_mask.any():
        core_mask &= final_mask

    if not core_mask.any() and final_mask.any():
        core_mask = np.asarray(final_mask, dtype=bool)

    final_mask = np.asarray(final_mask, dtype=bool)
    core_mask = np.asarray(core_mask, dtype=bool)
    support_ratio = np.clip(support_ratio, 0.0, 1.0).astype(np.float32)
    mean_prob = np.clip(mean_prob, 0.0, 1.0).astype(np.float32)

    final_support = support_ratio[final_mask]
    core_support = support_ratio[core_mask]
    return {
        "final_mask": final_mask,
        "core_mask": core_mask,
        "support_ratio": support_ratio,
        "mean_prob": mean_prob,
        "ensemble_size": int(n_runs),
        "weak_perturbation_count": int(sum(x == "weak" for x in perturbation_schedule)),
        "strong_perturbation_count": int(sum(x == "strong" for x in perturbation_schedule)),
        "final_voxels": int(np.sum(final_mask)),
        "core_voxels": int(np.sum(core_mask)),
        "final_support_mean": float(np.mean(final_support)) if final_support.size > 0 else 0.0,
        "core_support_mean": float(np.mean(core_support)) if core_support.size > 0 else 0.0,
        "support_peak": float(np.max(support_ratio)) if support_ratio.size > 0 else 0.0,
    }


def normalize_minmax(
    x: np.ndarray,
    abs_range_floor: float = 1e-12,
    rel_range_floor: float = 0.0,
    low_var_fill: float = 1.0,
) -> np.ndarray:
    """
    Args:
        x: [N]
        abs_range_floor: if (max-min) below this, treat as low-variance.
        rel_range_floor: if (max-min)/max(|mean|,eps) below this, treat as low-variance.
        low_var_fill: constant used when low-variance.
    Returns:
        y: [N] in [0,1] (or constant low_var_fill when low-variance).
    """
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    rng = x_max - x_min
    if rng < float(abs_range_floor):
        return np.full_like(x, float(low_var_fill), dtype=np.float32)
    if rel_range_floor > 0.0:
        scale = max(abs(float(np.mean(x))), EPS)
        if (rng / scale) < float(rel_range_floor):
            return np.full_like(x, float(low_var_fill), dtype=np.float32)
    return ((x - x_min) / (rng + EPS)).astype(np.float32)


def top_fraction_mean(values: np.ndarray, frac: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 0.0:
        return float(np.mean(arr))
    k = max(1, int(math.ceil(arr.size * frac)))
    if k >= arr.size:
        return float(np.mean(arr))
    topk = np.partition(arr, arr.size - k)[-k:]
    return float(np.mean(topk))


def masked_quantile(
    values: np.ndarray,
    valid_mask: np.ndarray,
    q: float,
    default: float,
) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(arr)
    if int(np.sum(valid)) <= 0:
        return float(default)
    return float(np.quantile(arr[valid], float(np.clip(q, 0.0, 1.0))))


def adaptive_top_fraction(
    value_count: int,
    target_keep: int,
    min_frac: float,
    max_frac: float,
) -> float:
    if int(value_count) <= 0:
        return float(max_frac)
    frac = float(target_keep) / float(max(1, int(value_count)))
    return float(np.clip(frac, float(min_frac), float(max_frac)))


def compute_adaptive_cam_gate(
    candidate_mask: np.ndarray,
    cam_score: np.ndarray,
    cam_prompt_topk_mean: np.ndarray,
    cam_prompt_mass_inside_target: np.ndarray,
    cam_focus_available: np.ndarray,
    cam_focus_nonempty: np.ndarray,
) -> Dict[str, Any]:
    candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    focus_available = np.asarray(cam_focus_available, dtype=np.float32).reshape(-1)
    focus_nonempty = np.asarray(cam_focus_nonempty, dtype=bool).reshape(-1)
    usable = (
        candidate
        & (focus_available > 0.5)
        & focus_nonempty
        & np.isfinite(np.asarray(cam_score, dtype=np.float32).reshape(-1))
        & np.isfinite(np.asarray(cam_prompt_topk_mean, dtype=np.float32).reshape(-1))
        & np.isfinite(np.asarray(cam_prompt_mass_inside_target, dtype=np.float32).reshape(-1))
    )
    reliability = np.zeros_like(np.asarray(cam_score, dtype=np.float32).reshape(-1), dtype=np.float32)
    strong = np.zeros_like(usable, dtype=bool)
    threshold = 0.0
    if int(np.sum(usable)) > 0:
        score_norm = normalize_masked_signal(cam_score, usable, low_var_fill=1.0)
        topk_norm = normalize_masked_signal(cam_prompt_topk_mean, usable, low_var_fill=1.0)
        mass_norm = normalize_masked_signal(
            cam_prompt_mass_inside_target, usable, low_var_fill=1.0
        )
        reliability = np.clip(
            0.40 * score_norm + 0.30 * topk_norm + 0.30 * mass_norm,
            0.0,
            1.0,
        ).astype(np.float32)
        threshold = float(
            max(
                masked_quantile(reliability, usable, q=0.60, default=0.5),
                0.5 * float(np.max(reliability[usable])),
            )
        )
        strong = usable & (reliability >= threshold)
    return {
        "usable_mask": usable.astype(bool),
        "strong_mask": strong.astype(bool),
        "reliability": reliability.astype(np.float32),
        "threshold": float(threshold),
    }


def _split_nrrd_header_payload(blob: bytes) -> Tuple[str, bytes]:
    pos = blob.find(b"\r\n\r\n")
    sep_len = 4
    if pos < 0:
        pos = blob.find(b"\n\n")
        sep_len = 2
    if pos < 0:
        raise ValueError("Invalid NRRD: cannot find header terminator.")
    header = blob[:pos].decode("ascii", errors="strict")
    payload = blob[pos + sep_len :]
    return header, payload


def _parse_nrrd_header(header: str) -> Dict[str, str]:
    lines = header.splitlines()
    if len(lines) == 0 or (not lines[0].startswith("NRRD")):
        raise ValueError("Invalid NRRD header magic.")
    meta: Dict[str, str] = {}
    for ln in lines[1:]:
        s = ln.strip()
        if (not s) or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        meta[k.strip().lower()] = v.strip()
    return meta


def _nrrd_dtype(type_name: str, endian: str) -> np.dtype:
    t = type_name.strip().lower()
    mapping = {
        "signed char": np.int8,
        "int8": np.int8,
        "int8_t": np.int8,
        "uchar": np.uint8,
        "unsigned char": np.uint8,
        "uint8": np.uint8,
        "uint8_t": np.uint8,
        "short": np.int16,
        "short int": np.int16,
        "signed short": np.int16,
        "signed short int": np.int16,
        "int16": np.int16,
        "int16_t": np.int16,
        "ushort": np.uint16,
        "unsigned short": np.uint16,
        "unsigned short int": np.uint16,
        "uint16": np.uint16,
        "uint16_t": np.uint16,
        "int": np.int32,
        "signed int": np.int32,
        "int32": np.int32,
        "int32_t": np.int32,
        "uint": np.uint32,
        "unsigned int": np.uint32,
        "uint32": np.uint32,
        "uint32_t": np.uint32,
        "float": np.float32,
        "double": np.float64,
    }
    if t not in mapping:
        raise ValueError(f"Unsupported NRRD type: {type_name}")
    dt = np.dtype(mapping[t])
    if dt.itemsize > 1:
        e = endian.strip().lower()
        if e == "little":
            dt = dt.newbyteorder("<")
        elif e == "big":
            dt = dt.newbyteorder(">")
    return dt


def _infer_depth_axis(meta: Dict[str, str], sizes: List[int]) -> int:
    sd = meta.get("space directions", "")
    if sd:
        tokens = re.findall(r"\([^\)]*\)|none", sd, flags=re.IGNORECASE)
        if len(tokens) == len(sizes):
            spacings = []
            for tok in tokens:
                if tok.lower() == "none":
                    spacings.append(-1.0)
                else:
                    vals = [float(x.strip()) for x in tok.strip()[1:-1].split(",")]
                    spacings.append(float(np.sqrt(np.sum(np.square(vals)))))
            if max(spacings) > 0:
                return int(np.argmax(np.asarray(spacings)))
    return int(np.argmin(np.asarray(sizes)))


def load_nrrd_volume(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    header, payload = _split_nrrd_header_payload(blob)
    meta = _parse_nrrd_header(header)
    if int(meta.get("dimension", "0")) != 3:
        raise ValueError(f"Only 3D NRRD is supported, got dimension={meta.get('dimension')}")
    sizes = [int(x) for x in meta["sizes"].split()]
    if len(sizes) != 3:
        raise ValueError(f"Invalid NRRD sizes: {meta.get('sizes')}")
    encoding = meta.get("encoding", "raw").strip().lower()
    if encoding in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    elif encoding == "raw":
        pass
    else:
        raise ValueError(f"Unsupported NRRD encoding: {encoding} (only raw/gzip supported)")

    dt = _nrrd_dtype(meta["type"], meta.get("endian", "little"))
    total = int(np.prod(np.asarray(sizes, dtype=np.int64)))
    arr = np.frombuffer(payload, dtype=dt, count=total)
    if arr.size != total:
        raise ValueError(f"NRRD payload size mismatch: expected {total}, got {arr.size} ({path})")
    arr = arr.reshape(tuple(sizes), order="F")
    depth_axis = _infer_depth_axis(meta, sizes)
    arr = np.moveaxis(arr, depth_axis, 0)
    return np.asarray(arr, dtype=np.float32)



def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def choose_device(device: str) -> str:
    def safe_cuda_available() -> bool:
        if not HAS_TORCH:
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return bool(torch.cuda.is_available())
        except Exception:
            return False

    dev = str(device).strip().lower()
    if dev == "cpu":
        return "cpu"

    if dev == "cuda":
        if safe_cuda_available():
            return "cuda"
        raise RuntimeError("CUDA requested but unavailable.")

    if dev.startswith("cuda:"):
        if not safe_cuda_available():
            raise RuntimeError(f"{device} requested but CUDA is unavailable.")
        try:
            idx = int(dev.split(":", 1)[1])
        except Exception:
            raise ValueError(f"Invalid CUDA device '{device}'.")
        try:
            count = int(torch.cuda.device_count())
        except Exception:
            count = 0
        if idx < 0 or idx >= count:
            raise ValueError(f"Requested {device}, but visible CUDA devices={count}.")
        return f"cuda:{idx}"

    raise ValueError(f"Unknown device '{device}'.")


def logits_to_foreground_prob(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1] != 2:
        raise ValueError(f"logits must be [D,2,H,W], got {arr.shape}")
    arr_max = np.max(arr, axis=1, keepdims=True)
    exp_arr = np.exp(arr - arr_max)
    denom = np.sum(exp_arr, axis=1) + EPS
    return (exp_arr[:, 1] / denom).astype(np.float32)


def compute_logits_uncertainty_and_confidence(
    logits: np.ndarray,
    prompt_slice_id: int,
) -> Dict[str, Any]:
    prob = logits_to_foreground_prob(logits)
    entropy_map = np.clip(binary_entropy(prob) / math.log(2.0), 0.0, 1.0).astype(np.float32)
    return {
        "prob": prob.astype(np.float32),
        "entropy_map": entropy_map.astype(np.float32),
    }


def compute_aggrigator_confidence_score(
    uncertainty_map: np.ndarray,
    pred_mask: np.ndarray,
) -> float:
    mask = np.asarray(pred_mask > 0, dtype=np.uint8)
    uncertainty = np.asarray(uncertainty_map, dtype=np.float32)
    if uncertainty.shape != mask.shape:
        raise ValueError(
            f"uncertainty_map and pred_mask shape mismatch: {uncertainty.shape} vs {mask.shape}"
        )
    if int(np.sum(mask > 0)) <= 0:
        return 0.0
    if HAS_AGGRIGATOR and AggregationMethods is not None and UncertaintyMap is not None:
        unc = UncertaintyMap(uncertainty, mask=mask, name="pam_entropy")
        foreground_uncertainty = float(AggregationMethods.class_mean_weighted_by_occurrence(unc))
    else:
        foreground_uncertainty = float(np.mean(uncertainty[mask > 0]))
    return float(np.clip(1.0 - foreground_uncertainty, 0.0, 1.0))


def compute_gradcam_filter_score(cam_features: Dict[str, float]) -> float:
    focus = float(cam_features.get("cam_prompt_mass_inside_target", cam_features.get("cam_mass_inside_target", 0.0)))
    entropy = 1.0 - float(cam_features.get("cam_prompt_entropy", cam_features.get("cam_entropy", 1.0)))
    topk = float(cam_features.get("cam_prompt_topk_mean", cam_features.get("cam_topk_mean", 0.0)))
    compactness = float(
        cam_features.get("cam_prompt_compactness", cam_features.get("cam_compactness", 0.0))
    )
    score = (
        0.50 * topk
        + 0.20 * focus
        + 0.20 * entropy
        + 0.10 * compactness
    )
    score *= float(cam_features.get("cam_available", 0.0))
    return float(np.clip(score, 0.0, 1.0))


def compute_prompt_candidate_features(mask: np.ndarray) -> Dict[str, float]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {
            "prompt_candidate_available": 0.0,
            "prompt_area_ratio": 0.0,
            "prompt_area_score": 0.0,
            "prompt_center_score": 0.0,
            "prompt_border_fraction": 1.0,
        }
    area_ratio = float(np.mean(m))
    small_enough = float(np.clip(area_ratio / 0.005, 0.0, 1.0))
    not_too_large = float(np.clip((0.60 - area_ratio) / 0.30, 0.0, 1.0))
    area_score = float(math.sqrt(max(0.0, small_enough * not_too_large)))
    return {
        "prompt_candidate_available": 1.0,
        "prompt_area_ratio": float(area_ratio),
        "prompt_area_score": float(np.clip(area_score, 0.0, 1.0)),
        "prompt_center_score": float(_mask_center_score(m)),
        "prompt_border_fraction": float(_mask_border_fraction(m)),
    }


def extract_required_label_ids(label_volume: np.ndarray) -> List[int]:
    arr = np.asarray(label_volume, dtype=np.int32)
    if arr.ndim != 3:
        raise ValueError(f"label_volume must be [D,H,W], got {arr.shape}")
    return sorted(int(v) for v in np.unique(arr).tolist() if int(v) > 0)


def compute_required_label_area_diagnostics(
    label_slice_2d: np.ndarray,
    required_label_ids: List[int],
) -> Dict[str, Any]:
    arr = np.asarray(label_slice_2d, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"label_slice_2d must be [H,W], got {arr.shape}")

    required_ids = [int(v) for v in required_label_ids]
    slice_positive_label_ids = sorted(int(v) for v in np.unique(arr).tolist() if int(v) > 0)
    present = set(slice_positive_label_ids)
    missing_required_label_ids = [label_id for label_id in required_ids if label_id not in present]
    slice_area = float(max(1, int(arr.shape[0]) * int(arr.shape[1])))

    area_ratios: Dict[str, float] = {}
    area_scores: Dict[str, float] = {}
    ratio_values: List[float] = []
    score_values: List[float] = []
    for label_id in required_ids:
        area_ratio = float(np.sum(arr == int(label_id)) / slice_area)
        area_score = float(np.clip(area_ratio / float(REQUIRED_LABEL_MIN_AREA_RATIO), 0.0, 1.0))
        area_ratios[str(label_id)] = area_ratio
        area_scores[str(label_id)] = area_score
        ratio_values.append(area_ratio)
        score_values.append(area_score)

    return {
        "required_label_ids": list(required_ids),
        "slice_positive_label_ids": list(slice_positive_label_ids),
        "all_required_label_ids_present": bool(len(missing_required_label_ids) <= 0),
        "missing_required_label_ids": list(missing_required_label_ids),
        "missing_required_label_count": int(len(missing_required_label_ids)),
        "required_label_area_ratios": area_ratios,
        "required_label_area_scores": area_scores,
        "min_required_label_area_ratio": float(min(ratio_values)) if ratio_values else 0.0,
        "min_required_label_area_score": float(min(score_values)) if score_values else 0.0,
    }


def smooth_1d_signal(signal: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if arr.size <= 1:
        return arr.astype(np.float32)
    sig = float(max(0.0, sigma))
    if HAS_SCIPY and sig > 0.0:
        return ndi.gaussian_filter1d(arr, sigma=sig, mode="nearest").astype(np.float32)
    if sig <= 0.0:
        return arr.astype(np.float32)
    radius = max(1, int(math.ceil(sig * 2.0)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2.0 * sig * sig + EPS)).astype(np.float32)
    kernel = kernel / (np.sum(kernel) + EPS)
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def compute_profile_continuity_score(area_profile: np.ndarray) -> float:
    p = np.asarray(area_profile, dtype=np.float32).reshape(-1)
    if p.size <= 1:
        return 0.0
    nz = p > 0
    if int(np.sum(nz)) <= 1:
        return 0.0
    delta = np.abs(np.diff(p))
    scale = np.maximum(np.maximum(p[:-1], p[1:]), 1e-6)
    rel = delta / scale
    return float(np.clip(1.0 - np.mean(np.clip(rel, 0.0, 1.0)), 0.0, 1.0))


def compute_prompt_anchored_core_score(
    prob: np.ndarray,
    prompt_slice_id: int,
    prob_thresh: float = 0.5,
) -> float:
    p = np.asarray(prob, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"prob must be [D,H,W], got {p.shape}")
    d = int(p.shape[0])
    if d <= 0:
        return 0.0

    sid = int(np.clip(prompt_slice_id, 0, d - 1))
    hard = p >= float(prob_thresh)
    raw_area = np.mean(hard, axis=(1, 2)).astype(np.float32)
    foreground = raw_area > 0.0
    if not bool(foreground[sid]):
        return 0.0

    left = sid
    while left > 0 and bool(foreground[left - 1]):
        left -= 1
    right = sid
    while right < d - 1 and bool(foreground[right + 1]):
        right += 1

    segment_area = smooth_1d_signal(raw_area[left : right + 1], sigma=1.0).astype(np.float32)
    if float(np.max(segment_area)) <= EPS:
        segment_area = raw_area[left : right + 1].astype(np.float32)
    segment_length = int(right - left + 1)
    local_idx = int(sid - left)

    z = np.arange(left, right + 1, dtype=np.float32)
    mass = float(np.sum(segment_area))
    if mass <= EPS:
        center_z = float(sid)
    else:
        center_z = float(np.sum(z * segment_area) / (mass + EPS))
    radius = max(1.0, float(segment_length - 1) / 2.0)
    center_score = float(np.clip(1.0 - abs(float(sid) - center_z) / radius, 0.0, 1.0))

    local_peak_score = float(
        np.clip(float(segment_area[local_idx]) / max(float(np.max(segment_area)), EPS), 0.0, 1.0)
    )
    expected_min_length = float(max(3, int(math.ceil(0.05 * float(d)))))
    support_score = float(np.clip(float(segment_length) / expected_min_length, 0.0, 1.0))

    return float(
        np.clip(
            0.50 * center_score
            + 0.30 * local_peak_score
            + 0.20 * support_score,
            0.0,
            1.0,
        )
    )


def compute_propagation_reasoning_features(prob: np.ndarray, prompt_slice_id: int) -> Dict[str, float]:
    p = np.asarray(prob, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"prob must be [D,H,W], got {p.shape}")
    hard = p >= 0.5
    area_profile = np.mean(hard, axis=(1, 2)).astype(np.float32)
    continuity = compute_profile_continuity_score(area_profile)
    return {
        "prop_nonzero_slice_ratio": float(np.mean(area_profile > 0)),
        "prop_continuity_score": float(continuity),
    }


def compute_prompt_observation_features(
    prob: np.ndarray,
    uncertainty_volume: np.ndarray,
    prompt_slice_id: int,
    prompt_mask_2d: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    prob_arr = np.asarray(prob, dtype=np.float32)
    if prob_arr.ndim != 3:
        raise ValueError(f"prob must be [D,H,W], got {prob_arr.shape}")
    uncertainty_arr = np.asarray(uncertainty_volume, dtype=np.float32)
    if uncertainty_arr.shape != prob_arr.shape:
        raise ValueError(
            "uncertainty_volume shape mismatch: "
            f"expected {prob_arr.shape}, got {uncertainty_arr.shape}"
        )

    d = int(prob_arr.shape[0])
    sid = int(np.clip(prompt_slice_id, 0, d - 1))
    z_scale = float(max(1, d - 1))
    low_uncertainty = np.clip(1.0 - uncertainty_arr, 0.0, 1.0).astype(np.float32)
    reliable = np.clip((prob_arr - 0.5) / 0.5, 0.0, 1.0).astype(np.float32) * low_uncertainty
    reliable_profile = np.mean(reliable, axis=(1, 2)).astype(np.float32)
    smooth_profile = smooth_1d_signal(reliable_profile, sigma=1.0).astype(np.float32)
    if float(np.max(smooth_profile)) <= EPS:
        smooth_profile = reliable_profile.astype(np.float32)

    peak_idx = int(np.argmax(smooth_profile)) if smooth_profile.size > 0 else int(sid)
    peak_value = float(np.max(smooth_profile)) if smooth_profile.size > 0 else 0.0
    mass = float(np.sum(smooth_profile))
    if mass <= EPS:
        center_z = float(peak_idx)
    else:
        z = np.arange(d, dtype=np.float32)
        center_z = float(np.sum(z * smooth_profile) / (mass + EPS))

    prompt_mask = None
    if prompt_mask_2d is not None:
        prompt_mask = np.asarray(prompt_mask_2d, dtype=bool)
        if prompt_mask.shape != prob_arr.shape[1:]:
            raise ValueError(
                "prompt_mask_2d shape mismatch: "
                f"expected {prob_arr.shape[1:]}, got {prompt_mask.shape}"
            )
    prompt_prob = prob_arr[sid]
    prompt_reliable = reliable[sid]
    if prompt_mask is not None and int(np.sum(prompt_mask)) > 0:
        prompt_mask_prob_mean = float(np.mean(prompt_prob[prompt_mask]))
        prompt_mask_reliability = float(np.mean(prompt_reliable[prompt_mask]))
    else:
        prompt_mask_prob_mean = float(np.mean(prompt_prob))
        prompt_mask_reliability = float(np.mean(prompt_reliable))

    pred_mask = prob_arr >= 0.5
    if int(np.sum(pred_mask)) > 0:
        masked_volume_uncertainty = float(np.mean(uncertainty_arr[pred_mask]))
    else:
        masked_volume_uncertainty = float(np.mean(uncertainty_arr))

    return {
        "prompt_mask_prob_mean": float(np.clip(prompt_mask_prob_mean, 0.0, 1.0)),
        "prompt_mask_reliability": float(np.clip(prompt_mask_reliability, 0.0, 1.0)),
        "prompt_foreground_ratio": float(np.mean(pred_mask[sid].astype(np.float32))),
        "prompt_peak_alignment": float(
            np.clip(reliable_profile[sid] / max(peak_value, EPS), 0.0, 1.0)
        ),
        "prompt_center_alignment": float(
            np.clip(1.0 - abs(float(sid) - center_z) / z_scale, 0.0, 1.0)
        ),
        "support_continuity": float(compute_profile_continuity_score(smooth_profile)),
        "reliable_volume_ratio": float(np.mean(reliable)),
        "masked_volume_uncertainty": float(np.clip(masked_volume_uncertainty, 0.0, 1.0)),
        "support_peak_slice_id": int(peak_idx),
        "support_center_z": float(center_z),
        "support_peak_value": float(np.clip(peak_value, 0.0, 1.0)),
    }


def normalize_masked_signal(
    values: np.ndarray,
    valid_mask: np.ndarray,
    low_var_fill: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    valid_ids = np.where(valid)[0]
    if valid_ids.size <= 0:
        return out
    out[valid_ids] = normalize_minmax(arr[valid_ids], low_var_fill=float(low_var_fill))
    return out.astype(np.float32)

def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)
    inter = int(np.logical_and(pred_b, tgt_b).sum())
    return float((2.0 * inter + eps) / (int(pred_b.sum()) + int(tgt_b.sum()) + eps))


def center_slice_bounds(depth: int, fraction: float = CENTER_SLICE_FRACTION) -> Tuple[int, int]:
    d = int(depth)
    frac = float(np.clip(float(fraction), 0.0, 1.0))
    span = int(max(1, round(d * frac)))
    start = int(max(0, (d - span) // 2))
    end = int(min(d, start + span))
    return start, end


def classwise_macro_mean(values: Dict[str, float], required_label_ids: List[int]) -> float:
    vals = [float(values.get(str(int(label_id)), 0.0)) for label_id in required_label_ids]
    if len(vals) <= 0:
        return 0.0
    return float(np.clip(np.mean(np.asarray(vals, dtype=np.float32)), 0.0, 1.0))


def encode_ct_slice_png_base64(slice_2d: np.ndarray) -> str:
    gray = ct_slice_to_uint8(slice_2d)
    return _encode_png_base64(gray)


def ct_slice_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    img = np.asarray(slice_2d, dtype=np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo = float(np.min(img))
        hi = float(np.max(img))
    if hi <= lo:
        gray = np.zeros_like(img, dtype=np.uint8)
    else:
        gray = np.clip((img - lo) / (hi - lo + EPS), 0.0, 1.0)
        gray = np.asarray(gray * 255.0, dtype=np.uint8)
    return gray


def encode_ct_slice_with_candidate_png_base64(
    slice_2d: np.ndarray,
    candidate_mask_2d: np.ndarray,
) -> str:
    gray = ct_slice_to_uint8(slice_2d)
    mask = np.asarray(candidate_mask_2d, dtype=bool)
    if mask.shape != gray.shape or not np.any(mask):
        return _encode_png_base64(gray)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    rgb[mask, 0] = 0.35 * rgb[mask, 0] + 0.65 * 255.0
    rgb[mask, 1] = 0.35 * rgb[mask, 1]
    rgb[mask, 2] = 0.35 * rgb[mask, 2]
    if HAS_SCIPY:
        boundary = np.logical_xor(mask, ndi.binary_erosion(mask))
    else:
        boundary = mask
    rgb[boundary, 0] = 255.0
    rgb[boundary, 1] = 255.0
    rgb[boundary, 2] = 0.0
    return _encode_png_base64(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


def estimate_ct_body_area(slice_2d: np.ndarray) -> int:
    img = np.asarray(slice_2d, dtype=np.float32)
    if img.size <= 0:
        return 0
    min_value = float(np.nanmin(img))
    max_value = float(np.nanmax(img))
    if min_value <= -700.0 and max_value >= -100.0:
        thresh = -500.0
    else:
        thresh = float(np.percentile(img, 20))
    return int(np.sum(img > thresh))


def build_agent_reference_description(bucket: str, row: SliceEvidence) -> str:
    positive_ids = {int(v) for v in row.get("slice_positive_label_ids", [])}
    missing_ids = {int(v) for v in row.get("missing_required_label_ids", [])}
    required_area_ratios = {
        str(key): float(value)
        for key, value in dict(row.get("required_label_area_ratios", {})).items()
    }
    label1_area_ratio = float(required_area_ratios.get("1", 0.0))
    label2_area_ratio = float(required_area_ratios.get("2", 0.0))
    if bucket == "background":
        return (
            "Reference NO example: pure background for the target classes. "
            "Neither class 1 nor class 2 is visibly present on this slice."
        )
    if bucket == "incomplete":
        if 1 in positive_ids and 2 in missing_ids:
            if label1_area_ratio >= 0.03:
                return (
                    "Reference NO example: large class-1-only slice. Class 1 occupies a large "
                    "area, but class 2 is completely absent on this slice."
                )
            return (
                "Reference NO example: class 1 is visible, but class 2 is not visibly present on this slice."
            )
        if 2 in positive_ids and 1 in missing_ids:
            return (
                "Reference NO example: class 2 is visible, but class 1 is not visibly present on this slice."
            )
        return (
            "Reference NO example: some foreground is visible, but not all required target "
            "classes are present on this slice."
        )
    if 0.0 < label2_area_ratio <= 0.0015:
        return (
            "Reference YES example: class 1 is visible and a small but real class 2 is also "
            "present on the same slice."
        )
    return (
        "Reference YES example: both class 1 and class 2 are visibly present on the same slice."
    )


def build_agent_reference_example_bank(
    volume: np.ndarray,
    rows: List[SliceEvidence],
    center_start: int,
    center_end: int,
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, Any]]]:
    candidate_depth = int(
        max(
            1,
            getattr(
                args,
                "agent_reference_candidate_depth_per_bucket",
                AGENT_REFERENCE_CANDIDATE_DEPTH_PER_BUCKET,
            ),
        )
    )
    mid_slice = 0.5 * float(int(center_start) + int(center_end) - 1)
    bucketed: Dict[str, List[Dict[str, Any]]] = {
        "background": [],
        "incomplete": [],
        "complete": [],
    }
    for row in rows:
        sid = int(row.get("slice_id", -1))
        if sid < int(center_start) or sid >= int(center_end):
            continue
        positive_ids = [int(v) for v in row.get("slice_positive_label_ids", [])]
        if len(positive_ids) <= 0:
            bucket = "background"
        elif bool(row.get("all_required_label_ids_present", False)):
            bucket = "complete"
        else:
            bucket = "incomplete"
        prompt_area_ratio = float(row.get("prompt_area_ratio", 0.0))
        min_required_ratio = float(row.get("min_required_label_area_ratio", 0.0))
        center_distance = abs(float(sid) - mid_slice)
        body_area = int(estimate_ct_body_area(np.asarray(volume[sid], dtype=np.float32)))
        required_area_ratios = {
            str(key): float(value)
            for key, value in dict(row.get("required_label_area_ratios", {})).items()
        }
        label1_area_ratio = float(required_area_ratios.get("1", 0.0))
        label2_area_ratio = float(required_area_ratios.get("2", 0.0))
        dominant_present_required_ratio = float(
            max(
                [
                    float(required_area_ratios.get(str(label_id), 0.0))
                    for label_id in positive_ids
                    if int(label_id) > 0
                ]
                or [0.0]
            )
        )
        is_label1_only = bool(label1_area_ratio > 0.0 and label2_area_ratio <= 0.0)
        item = {
            "bucket": bucket,
            "slice_id": sid,
            "expected_answer": "yes" if bucket == "complete" else "no",
            "description": build_agent_reference_description(bucket, row),
            "png_b64": encode_ct_slice_png_base64(np.asarray(volume[sid], dtype=np.float32)),
            "prompt_area_ratio": prompt_area_ratio,
            "min_required_label_area_ratio": min_required_ratio,
            "required_label_area_ratios": required_area_ratios,
            "label1_area_ratio": label1_area_ratio,
            "label2_area_ratio": label2_area_ratio,
            "dominant_present_required_ratio": dominant_present_required_ratio,
            "is_label1_only": bool(is_label1_only),
            "body_area": body_area,
            "center_distance": center_distance,
        }
        bucketed[bucket].append(item)

    raw_background = list(bucketed["background"])
    raw_incomplete = list(bucketed["incomplete"])
    raw_complete = list(bucketed["complete"])

    all_complete_slice_ids = [int(item["slice_id"]) for item in raw_complete]
    all_incomplete_slice_ids = [int(item["slice_id"]) for item in raw_incomplete]
    for item in raw_complete:
        item["distance_to_incomplete"] = float(
            min(
                [abs(int(item["slice_id"]) - int(other_sid)) for other_sid in all_incomplete_slice_ids]
                or [1e9]
            )
        )
    for item in raw_incomplete:
        item["distance_to_complete"] = float(
            min(
                [abs(int(item["slice_id"]) - int(other_sid)) for other_sid in all_complete_slice_ids]
                or [1e9]
            )
        )

    def merge_candidate_lists(
        candidate_lists: List[List[Dict[str, Any]]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[int] = set()
        max_rank = max((len(candidates) for candidates in candidate_lists), default=0)
        for rank in range(int(max_rank)):
            for candidates in candidate_lists:
                if rank >= len(candidates):
                    continue
                item = candidates[rank]
                sid = int(item["slice_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                merged.append(item)
                if len(merged) >= int(limit):
                    return merged
        return merged[: int(limit)]

    bucketed["background"] = sorted(
        raw_background,
        key=lambda item: (
            int(item["body_area"]),
            -float(item["center_distance"]),
            -int(item["slice_id"]),
        ),
        reverse=True,
    )[:candidate_depth]
    incomplete_boundary = sorted(
        raw_incomplete,
        key=lambda item: (
            int(bool(item["is_label1_only"])),
            -float(item.get("distance_to_complete", 1e9)),
            float(item["label1_area_ratio"]),
            float(item["prompt_area_ratio"]),
        ),
        reverse=True,
    )
    incomplete_strong = sorted(
        raw_incomplete,
        key=lambda item: (
            int(bool(item["is_label1_only"])),
            float(item["label1_area_ratio"]),
            float(item["prompt_area_ratio"]),
            int(item["body_area"]),
        ),
        reverse=True,
    )
    complete_typical = sorted(
        raw_complete,
        key=lambda item: (
            float(item["label2_area_ratio"]),
            float(item["min_required_label_area_ratio"]),
            float(item["prompt_area_ratio"]),
            int(item["body_area"]),
        ),
        reverse=True,
    )
    complete_edge = sorted(
        [item for item in raw_complete if float(item.get("label2_area_ratio", 0.0)) > 0.0],
        key=lambda item: (
            float(item["label2_area_ratio"]),
            float(item.get("distance_to_incomplete", 1e9)),
            -float(item["label1_area_ratio"]),
        ),
    )
    bucketed["incomplete"] = merge_candidate_lists(
        [incomplete_boundary, incomplete_strong],
        limit=candidate_depth,
    )
    bucketed["complete"] = merge_candidate_lists(
        [complete_typical, complete_edge],
        limit=candidate_depth,
    )
    return bucketed


def _select_reference_examples_with_spacing(
    items: List[Dict[str, Any]],
    per_bucket: int,
    excluded: set[int],
    min_slice_gap: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in items:
        sid = int(item.get("slice_id", -1))
        if sid in excluded:
            continue
        if any(abs(sid - int(prev.get("slice_id", -1))) < int(min_slice_gap) for prev in selected):
            continue
        selected.append(item)
        if len(selected) >= int(per_bucket):
            return selected
    if len(selected) >= int(per_bucket):
        return selected
    for item in items:
        sid = int(item.get("slice_id", -1))
        if sid in excluded:
            continue
        if any(int(prev.get("slice_id", -1)) == sid for prev in selected):
            continue
        selected.append(item)
        if len(selected) >= int(per_bucket):
            break
    return selected


def _pick_single_reference(
    candidates: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    excluded: set[int],
    min_slice_gap: int,
) -> Optional[Dict[str, Any]]:
    for item in candidates:
        sid = int(item.get("slice_id", -1))
        if sid in excluded:
            continue
        if any(abs(sid - int(prev.get("slice_id", -1))) < int(min_slice_gap) for prev in selected):
            continue
        return item
    for item in candidates:
        sid = int(item.get("slice_id", -1))
        if sid in excluded:
            continue
        if any(int(prev.get("slice_id", -1)) == sid for prev in selected):
            continue
        return item
    return None


def _select_complete_reference_examples(
    items: List[Dict[str, Any]],
    per_bucket: int,
    excluded: set[int],
    min_slice_gap: int,
    prefer_edge_examples: bool = True,
) -> List[Dict[str, Any]]:
    if per_bucket <= 0 or len(items) <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    typical_candidates = sorted(
        items,
        key=lambda item: (
            float(item.get("label2_area_ratio", 0.0)),
            float(item.get("min_required_label_area_ratio", 0.0)),
            -float(item.get("center_distance", 0.0)),
        ),
        reverse=True,
    )
    edge_candidates = sorted(
        [item for item in items if float(item.get("label2_area_ratio", 0.0)) > 0.0],
        key=lambda item: (
            float(item.get("label2_area_ratio", 0.0)),
            float(item.get("distance_to_incomplete", 1e9)),
            -float(item.get("label1_area_ratio", 0.0)),
        ),
    )
    candidate_groups: List[List[Dict[str, Any]]] = [typical_candidates]
    if prefer_edge_examples:
        candidate_groups.append(edge_candidates)
    for candidates in candidate_groups:
        if len(selected) >= int(per_bucket):
            break
        picked = _pick_single_reference(
            candidates=candidates,
            selected=selected,
            excluded=excluded,
            min_slice_gap=min_slice_gap,
        )
        if picked is not None:
            selected.append(picked)
    if len(selected) < int(per_bucket):
        for picked in _select_reference_examples_with_spacing(
            items=typical_candidates,
            per_bucket=per_bucket,
            excluded=excluded,
            min_slice_gap=min_slice_gap,
        ):
            sid = int(picked.get("slice_id", -1))
            if any(int(prev.get("slice_id", -1)) == sid for prev in selected):
                continue
            selected.append(picked)
            if len(selected) >= int(per_bucket):
                break
    return selected[:per_bucket]


def _select_incomplete_reference_examples(
    items: List[Dict[str, Any]],
    per_bucket: int,
    excluded: set[int],
    min_slice_gap: int,
    target_slice_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if per_bucket <= 0 or len(items) <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    boundary_candidates = sorted(
        items,
        key=lambda item: (
            int(bool(item.get("is_label1_only", False))),
            -float(item.get("distance_to_complete", 1e9)),
            float(item.get("label1_area_ratio", 0.0)),
        ),
        reverse=True,
    )
    if target_slice_id is None:
        secondary_candidates = sorted(
            items,
            key=lambda item: (
                int(bool(item.get("is_label1_only", False))),
                float(item.get("label1_area_ratio", 0.0)),
                float(item.get("prompt_area_ratio", 0.0)),
            ),
            reverse=True,
        )
    else:
        secondary_candidates = sorted(
            items,
            key=lambda item: (
                int(bool(item.get("is_label1_only", False))),
                -abs(int(item.get("slice_id", -1)) - int(target_slice_id)),
                float(item.get("label1_area_ratio", 0.0)),
                float(item.get("prompt_area_ratio", 0.0)),
            ),
            reverse=True,
        )
    for candidates in (boundary_candidates, secondary_candidates):
        if len(selected) >= int(per_bucket):
            break
        picked = _pick_single_reference(
            candidates=candidates,
            selected=selected,
            excluded=excluded,
            min_slice_gap=min_slice_gap,
        )
        if picked is not None:
            selected.append(picked)
    if len(selected) < int(per_bucket):
        for picked in _select_reference_examples_with_spacing(
            items=secondary_candidates,
            per_bucket=per_bucket,
            excluded=excluded,
            min_slice_gap=min_slice_gap,
        ):
            sid = int(picked.get("slice_id", -1))
            if any(int(prev.get("slice_id", -1)) == sid for prev in selected):
                continue
            selected.append(picked)
            if len(selected) >= int(per_bucket):
                break
    return selected[:per_bucket]


def select_agent_reference_examples(
    reference_example_bank: Optional[Dict[str, List[Dict[str, Any]]]],
    exclude_slice_ids: Optional[List[int]],
    args: argparse.Namespace,
    target_bucket: Optional[str] = None,
    target_slice_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    enabled = parse_bool_like(
        getattr(args, "agent_reference_examples_enabled", AGENT_REFERENCE_EXAMPLES_ENABLED),
        default=AGENT_REFERENCE_EXAMPLES_ENABLED,
    )
    if not enabled or not reference_example_bank:
        return []
    per_bucket = int(
        max(
            0,
            getattr(
                args,
                "agent_reference_examples_per_bucket",
                AGENT_REFERENCE_EXAMPLES_PER_BUCKET,
            ),
        )
    )
    if per_bucket <= 0:
        return []
    min_slice_gap = int(
        max(
            0,
            getattr(args, "agent_reference_min_slice_gap", AGENT_REFERENCE_MIN_SLICE_GAP),
        )
    )
    excluded = {int(v) for v in (exclude_slice_ids or [])}
    selected: List[Dict[str, Any]] = []
    for bucket in ("background", "incomplete", "complete"):
        if bucket == "complete":
            chosen_items = _select_complete_reference_examples(
                items=list(reference_example_bank.get(bucket, [])),
                per_bucket=per_bucket,
                excluded=excluded,
                min_slice_gap=min_slice_gap,
                prefer_edge_examples=bool(str(target_bucket or "") == "complete"),
            )
        elif bucket == "incomplete":
            chosen_items = _select_incomplete_reference_examples(
                items=list(reference_example_bank.get(bucket, [])),
                per_bucket=per_bucket,
                excluded=excluded,
                min_slice_gap=min_slice_gap,
                target_slice_id=target_slice_id,
            )
        else:
            chosen_items = _select_reference_examples_with_spacing(
                items=list(reference_example_bank.get(bucket, [])),
                per_bucket=per_bucket,
                excluded=excluded,
                min_slice_gap=min_slice_gap,
            )
        for item in chosen_items:
            selected.append(
                {
                    "bucket": str(item["bucket"]),
                    "slice_id": int(item["slice_id"]),
                    "expected_answer": str(item["expected_answer"]),
                    "description": str(item["description"]),
                    "png_b64": str(item["png_b64"]),
                }
            )
    return selected


def parse_agent_yes_no(text: str) -> str:
    compact = str(text or "").strip().lower()
    compact = re.sub(r"[^a-z]", "", compact)
    if compact == "yes":
        return "yes"
    if compact == "no":
        return "no"
    return "no"


def extract_chat_completion_delta_text(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    summary: Dict[str, Any] = {}
    choices = payload.get("choices") or []
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                summary["finish_reason"] = finish_reason
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
    usage = payload.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage
    return "".join(parts), summary


def extract_responses_output_text(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    summary: Dict[str, Any] = {}
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        parts.append(output_text)
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                if str(chunk.get("type", "")) in {"output_text", "text"}:
                    text = chunk.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    usage = payload.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage
    status = payload.get("status")
    if status is not None:
        summary["status"] = status
    return "".join(parts).strip(), summary


def read_chat_completion_sse_text(resp: Any) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    summary: Dict[str, Any] = {"chunk_count": 0, "json_error_count": 0}
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            summary["json_error_count"] = int(summary.get("json_error_count", 0)) + 1
            summary["last_bad_data"] = data[:200]
            continue
        summary["chunk_count"] = int(summary.get("chunk_count", 0)) + 1
        delta_text, chunk_summary = extract_chat_completion_delta_text(payload)
        if delta_text:
            parts.append(delta_text)
            parsed = parse_agent_yes_no("".join(parts))
            if parsed in {"yes", "no"}:
                summary["early_answer"] = parsed
                summary["early_stop"] = True
                break
        summary.update(chunk_summary)
    return "".join(parts).strip(), summary


def summarize_agent_stream(summary: Dict[str, Any]) -> str:
    keep = {
        "chunk_count": summary.get("chunk_count"),
        "json_error_count": summary.get("json_error_count"),
        "finish_reason": summary.get("finish_reason"),
        "usage": summary.get("usage"),
        "last_bad_data": summary.get("last_bad_data"),
    }
    return json.dumps(keep, ensure_ascii=False)[:500]


def should_retry_agent_http_error(status_code: int) -> bool:
    code = int(status_code)
    if code in {401, 403}:
        return False
    return code == 429 or 500 <= code <= 599


def classify_agent_url_error(exc: URLError) -> str:
    reason = getattr(exc, "reason", None)
    errno = getattr(reason, "errno", None)
    if errno == -3:
        return "dns_temporary_failure"
    if errno == -2:
        return "dns_name_not_known"
    if errno == 111:
        return "connection_refused"
    if isinstance(reason, TimeoutError):
        return "timeout"
    if isinstance(reason, OSError) and "timed out" in str(reason).lower():
        return "timeout"
    return "url_error"


def normalize_agent_base_urls(primary_base_url: str, fallback_base_urls: Any) -> List[str]:
    urls: List[str] = []
    for raw_url in [primary_base_url]:
        url = str(raw_url or "").strip().rstrip("/")
        if url and url not in urls:
            urls.append(url)
    if not urls:
        raise ValueError("agent_base_url_missing: Agent base URL is required.")
    if isinstance(fallback_base_urls, str):
        raw_fallbacks: Any = [
            item.strip() for item in fallback_base_urls.split(",") if item.strip()
        ]
    else:
        raw_fallbacks = fallback_base_urls
    if raw_fallbacks is None:
        raw_fallbacks = []
    for raw_url in raw_fallbacks:
        url = str(raw_url or "").strip().rstrip("/")
        if url and url not in urls:
            urls.append(url)
    return urls


def parse_bool_like(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def agent_proxy_mode(args: argparse.Namespace) -> str:
    use_proxy = parse_bool_like(
        getattr(args, "agent_use_system_proxy", os.environ.get("AGENT_USE_SYSTEM_PROXY")),
        default=AGENT_USE_SYSTEM_PROXY,
    )
    return "system_proxy" if use_proxy else "direct_no_proxy"


def agent_urlopen(req: urllib_request.Request, timeout: float, proxy_mode: str) -> Any:
    if proxy_mode == "system_proxy":
        return urllib_request.urlopen(req, timeout=timeout)
    opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


def ask_agent_slice_complete(
    slice_2d: np.ndarray,
    args: argparse.Namespace,
    reference_examples: Optional[List[Dict[str, Any]]] = None,
    candidate_mask_2d: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("agent_api_key_missing: OPENAI_API_KEY is required for Agent filtering.")
    base_url = str(getattr(args, "agent_base_url", AGENT_BASE_URL)).rstrip("/")
    base_urls = normalize_agent_base_urls(
        primary_base_url=base_url,
        fallback_base_urls=getattr(args, "agent_fallback_base_urls", AGENT_FALLBACK_BASE_URLS),
    )
    model = str(getattr(args, "agent_model", AGENT_MODEL))
    effort = str(getattr(args, "agent_reasoning_effort", AGENT_REASONING_EFFORT))
    wire_api = str(getattr(args, "agent_wire_api", AGENT_WIRE_API)).strip().lower().replace("-", "_")
    prompt = str(getattr(args, "agent_prompt", AGENT_PROMPT))
    max_retries = int(max(1, getattr(args, "agent_max_retries", AGENT_MAX_RETRIES)))
    backoff_seconds = float(
        max(0.0, getattr(args, "agent_retry_backoff_seconds", AGENT_RETRY_BACKOFF_SECONDS))
    )
    request_timeout_seconds = float(
        max(1.0, getattr(args, "agent_request_timeout_seconds", AGENT_REQUEST_TIMEOUT_SECONDS))
    )
    proxy_mode = agent_proxy_mode(args)
    has_candidate_mask = False
    if candidate_mask_2d is not None:
        candidate_arr = np.asarray(candidate_mask_2d, dtype=bool)
        has_candidate_mask = bool(candidate_arr.shape == np.asarray(slice_2d).shape and np.any(candidate_arr))
    else:
        candidate_arr = np.zeros(np.asarray(slice_2d).shape, dtype=bool)
    png_b64 = (
        encode_ct_slice_with_candidate_png_base64(slice_2d, candidate_arr)
        if has_candidate_mask
        else encode_ct_slice_png_base64(slice_2d)
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, example in enumerate(reference_examples or [], start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"Reference {idx}: correct_answer={str(example['expected_answer']).lower()}. "
                    f"{str(example['description'])}"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{str(example['png_b64'])}"},
            }
        )
    if has_candidate_mask:
        target_text = (
            "Target slice to classify: answer only for the next image. "
            "The red/yellow overlay marks the image-only candidate region proposed for selection; "
            "use this marked region as the candidate, not any manual annotation. "
            "Return exactly one word: yes or no."
        )
    else:
        target_text = (
            "Target slice to classify: answer only for the next image. "
            "Return exactly one word: yes or no."
        )
    content.append({"type": "text", "text": target_text})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
        }
    )
    if wire_api in {"chat", "chat_completion", "chat_completions"}:
        endpoint_suffix = "chat/completions"
        body = {
            "model": model,
            "stream": True,
            "reasoning": {"effort": effort},
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
    elif wire_api == "responses":
        endpoint_suffix = "responses"
        response_content: List[Dict[str, Any]] = []
        for item in content:
            item_type = str(item.get("type", ""))
            if item_type == "text":
                response_content.append({"type": "input_text", "text": str(item.get("text", ""))})
            elif item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and str(image_url.get("url", "")):
                    response_content.append(
                        {"type": "input_image", "image_url": str(image_url.get("url"))}
                    )
        body = {
            "model": model,
            "stream": False,
            "reasoning": {"effort": effort},
            "input": [
                {
                    "role": "user",
                    "content": response_content,
                }
            ],
        }
    else:
        raise RuntimeError(f"Unsupported agent_wire_api={wire_api!r}")
    data = json.dumps(body).encode("utf-8")
    last_error = ""
    error_history: List[str] = []
    for attempt in range(1, max_retries + 1):
        for endpoint_index, endpoint_base_url in enumerate(base_urls, start=1):
            endpoint = f"{endpoint_base_url}/{endpoint_suffix}"
            endpoint_tag = (
                f"endpoint={endpoint} endpoint_index={endpoint_index}/{len(base_urls)} "
                f"proxy_mode={proxy_mode} wire_api={wire_api}"
            )
            req = urllib_request.Request(
                endpoint,
                data=data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": agent_user_agent(),
                },
                method="POST",
            )
            try:
                with agent_urlopen(req, timeout=request_timeout_seconds, proxy_mode=proxy_mode) as resp:
                    if wire_api == "responses":
                        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                        raw_text, stream_summary = extract_responses_output_text(payload)
                    else:
                        raw_text, stream_summary = read_chat_completion_sse_text(resp)
                if not raw_text.strip():
                    last_error = (
                        "agent_stream_empty_response "
                        f"{endpoint_tag} attempt={attempt}/{max_retries}: "
                        f"{summarize_agent_stream(stream_summary)}"
                    )
                    error_history.append(last_error)
                    continue
                answer = parse_agent_yes_no(raw_text)
                return {
                    "answer": answer,
                    "raw_text": raw_text[:200],
                    "model": model,
                    "provider": str(getattr(args, "agent_model_provider", AGENT_MODEL_PROVIDER)),
                    "attempts": int(attempt),
                    "reference_examples_used": [
                        {
                            "bucket": str(example["bucket"]),
                            "slice_id": int(example["slice_id"]),
                            "expected_answer": str(example["expected_answer"]),
                        }
                        for example in (reference_examples or [])
                    ],
                }
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = (
                    f"agent_http_error_{exc.code} {endpoint_tag} "
                    f"attempt={attempt}/{max_retries}: {detail}"
                )
                error_history.append(last_error)
                if not should_retry_agent_http_error(int(exc.code)):
                    raise RuntimeError(last_error) from exc
            except TimeoutError as exc:
                last_error = (
                    f"agent_timeout {endpoint_tag} "
                    f"attempt={attempt}/{max_retries}: {exc}"
                )
                error_history.append(last_error)
            except URLError as exc:
                kind = classify_agent_url_error(exc)
                last_error = (
                    f"agent_{kind} {endpoint_tag} "
                    f"attempt={attempt}/{max_retries}: {exc}"
                )
                error_history.append(last_error)
        if attempt < max_retries:
            time.sleep(backoff_seconds * float(2 ** (attempt - 1)))
    recent_errors = " | ".join(error_history[-min(len(error_history), 6) :])
    raise RuntimeError(
        f"agent_request_failed after {max_retries} attempts across {len(base_urls)} endpoints: "
        f"{recent_errors or last_error}"
    )

def connected_components_2d(mask: np.ndarray) -> List[np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return []
    if HAS_SCIPY:
        labeled, n = ndi.label(m)
    elif HAS_CV2:
        num_labels, labeled = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        n = int(num_labels) - 1
    else:
        raise RuntimeError("connected_components_2d requires scipy or cv2.")
    if int(n) <= 1:
        return [m.astype(bool)]
    counts = np.bincount(np.asarray(labeled).ravel())
    counts[0] = 0
    out: List[np.ndarray] = []
    for comp_id in np.argsort(counts)[::-1]:
        if int(comp_id) <= 0 or int(counts[int(comp_id)]) <= 0:
            continue
        out.append(np.asarray(labeled == int(comp_id), dtype=bool))
    return out


def mask_to_box_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if ys.size <= 0 or xs.size <= 0:
        raise ValueError("mask_to_box_xyxy requires a non-empty mask.")
    x1 = int(np.min(xs))
    y1 = int(np.min(ys))
    x2 = int(np.max(xs)) + 1
    y2 = int(np.max(ys)) + 1
    return (x1, y1, x2, y2)


def normalize_box_prompt_variant(variant: str) -> str:
    text = str(variant or "").strip().lower()
    if text in {"", "baseline", "default"}:
        return BOX_PROMPT_VARIANT_BASELINE
    if text not in BOX_PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported box_prompt_variant='{variant}'. Expected one of {BOX_PROMPT_VARIANTS}."
        )
    return text


def normalize_box2mask_runtime_mode(mode: Optional[str]) -> str:
    text = str(mode or "").strip().lower()
    if text in {"", "baseline", "default"}:
        return BOX2MASK_RUNTIME_MODE_DEFAULT
    if text not in BOX2MASK_RUNTIME_MODES:
        raise ValueError(
            f"Unsupported box2mask_runtime_mode='{mode}'. Expected one of {BOX2MASK_RUNTIME_MODES}."
        )
    return text


def _scale_box_xyxy(
    box_xyxy: Tuple[int, int, int, int],
    width: int,
    height: int,
    scale: float = 1.0,
    square: bool = False,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    box_w = max(1, int(x2 - x1))
    box_h = max(1, int(y2 - y1))
    base_w = float(max(box_w, box_h)) if square else float(box_w)
    base_h = float(max(box_w, box_h)) if square else float(box_h)
    scaled_w = max(1.0, float(base_w) * float(scale))
    scaled_h = max(1.0, float(base_h) * float(scale))
    cx = 0.5 * float(x1 + x2)
    cy = 0.5 * float(y1 + y2)
    new_x1 = int(math.floor(cx - 0.5 * scaled_w))
    new_x2 = int(math.ceil(cx + 0.5 * scaled_w))
    new_y1 = int(math.floor(cy - 0.5 * scaled_h))
    new_y2 = int(math.ceil(cy + 0.5 * scaled_h))
    new_x1 = int(np.clip(new_x1, 0, max(0, int(width) - 1)))
    new_y1 = int(np.clip(new_y1, 0, max(0, int(height) - 1)))
    new_x2 = int(np.clip(new_x2, new_x1 + 1, int(width)))
    new_y2 = int(np.clip(new_y2, new_y1 + 1, int(height)))
    return (new_x1, new_y1, new_x2, new_y2)


def transform_box_xyxy_for_prompt_variant(
    box_xyxy: Tuple[int, int, int, int],
    width: int,
    height: int,
    box_prompt_variant: str = BOX_PROMPT_VARIANT_BASELINE,
) -> Tuple[int, int, int, int]:
    variant = normalize_box_prompt_variant(box_prompt_variant)
    if variant == "tight_components":
        return tuple(int(v) for v in box_xyxy)
    if variant == "expand10_components":
        return _scale_box_xyxy(box_xyxy, width=width, height=height, scale=1.10, square=False)
    if variant == "expand20_components":
        return _scale_box_xyxy(box_xyxy, width=width, height=height, scale=1.20, square=False)
    if variant == "square10_components":
        return _scale_box_xyxy(box_xyxy, width=width, height=height, scale=1.10, square=True)
    if variant == "square20_components":
        return _scale_box_xyxy(box_xyxy, width=width, height=height, scale=1.20, square=True)
    raise AssertionError(f"Unhandled box prompt variant: {variant}")


def _resolve_positive_label_ids(
    label_slice_2d: np.ndarray,
    label_ids: Optional[List[int]] = None,
) -> List[int]:
    arr = np.asarray(label_slice_2d)
    if arr.ndim != 2:
        raise ValueError(f"label_slice_2d must be [H,W], got {arr.shape}")
    if label_ids is None:
        label_ids = [int(v) for v in np.unique(arr).tolist() if int(v) > 0]
    return [int(v) for v in label_ids if int(np.sum(arr == int(v))) > 0]


def _merge_box_xyxy_list(
    boxes: List[List[int] | Tuple[int, int, int, int]],
) -> Optional[List[int]]:
    if len(boxes) <= 0:
        return None
    xs1 = [int(b[0]) for b in boxes]
    ys1 = [int(b[1]) for b in boxes]
    xs2 = [int(b[2]) for b in boxes]
    ys2 = [int(b[3]) for b in boxes]
    return [int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))]


def _build_prompt_meta(
    *,
    prompt_source: str,
    prompt_mask_area: int,
    component_area: int,
    union_box_xyxy: Optional[List[int]] = None,
    box_count: int = 0,
    uses_box2mask: bool = False,
    boxes: Optional[List[Dict[str, Any]]] = None,
) -> PromptMeta:
    union_box_area = 0
    if union_box_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in union_box_xyxy]
        union_box_area = int(max(0, x2 - x1) * max(0, y2 - y1))
    return {
        "prompt_source": str(prompt_source),
        "prompt_mask_area": int(prompt_mask_area),
        "sum_component_area": int(component_area),
        "union_box_xyxy": union_box_xyxy,
        "union_box_area": int(union_box_area),
        "box_count": int(box_count),
        "uses_box2mask": bool(uses_box2mask),
        "boxes": [] if boxes is None else list(boxes),
    }


def prompt_box2mask_area(prompt_meta: PromptMeta) -> Optional[int]:
    if not bool(prompt_meta.get("uses_box2mask", False)):
        return None
    return int(prompt_meta["prompt_mask_area"])


def prompt_box_xyxy(prompt_meta: PromptMeta) -> Optional[List[int]]:
    box = prompt_meta.get("union_box_xyxy")
    if box is None:
        return None
    return [int(v) for v in box]


def prompt_meta_to_selection_fields(prompt_meta: PromptMeta) -> Dict[str, Any]:
    return {
        "selected_prompt_source": str(prompt_meta.get("prompt_source", "unknown")),
        "selected_prompt_mask_area": int(prompt_meta["prompt_mask_area"]),
        "selected_prompt_component_area": int(prompt_meta["sum_component_area"]),
        "selected_prompt_box_xyxy": prompt_box_xyxy(prompt_meta),
        "selected_prompt_box_area": int(prompt_meta["union_box_area"]),
        "selected_prompt_box_count": int(prompt_meta["box_count"]),
        "selected_box2mask_prompt_area": prompt_box2mask_area(prompt_meta),
    }


def _collect_label_box_targets(
    label_slice_2d: np.ndarray,
    label_ids: Optional[List[int]] = None,
    box_prompt_variant: str = BOX_PROMPT_VARIANT_BASELINE,
) -> List[Dict[str, Any]]:
    arr = np.asarray(label_slice_2d)
    if arr.ndim != 2:
        raise ValueError(f"label_slice_2d must be [H,W], got {arr.shape}")
    variant = normalize_box_prompt_variant(box_prompt_variant)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    out: List[Dict[str, Any]] = []
    for label_id in _resolve_positive_label_ids(arr, label_ids=label_ids):
        cls_mask = np.asarray(arr == int(label_id), dtype=np.uint8)
        for component_index, component_mask in enumerate(connected_components_2d(cls_mask)):
            component_mask_u8 = np.asarray(component_mask, dtype=np.uint8)
            raw_box_xyxy = mask_to_box_xyxy(component_mask_u8)
            box_xyxy = transform_box_xyxy_for_prompt_variant(
                raw_box_xyxy,
                width=width,
                height=height,
                box_prompt_variant=variant,
            )
            x1, y1, x2, y2 = [int(v) for v in box_xyxy]
            rx1, ry1, rx2, ry2 = [int(v) for v in raw_box_xyxy]
            out.append(
                {
                    "label_id": int(label_id),
                    "component_index": int(component_index),
                    "component_area": int(np.sum(component_mask_u8)),
                    "component_mask": component_mask_u8,
                    "raw_box_xyxy": [rx1, ry1, rx2, ry2],
                    "raw_box_area": int(max(0, rx2 - rx1) * max(0, ry2 - ry1)),
                    "box_xyxy": [x1, y1, x2, y2],
                    "box_area": int(max(0, x2 - x1) * max(0, y2 - y1)),
                    "box_prompt_variant": variant,
                }
            )
    return out


def extract_label_boxes_on_slice(
    label_slice_2d: np.ndarray,
    label_ids: Optional[List[int]] = None,
    box_prompt_variant: str = BOX_PROMPT_VARIANT_BASELINE,
) -> List[Dict[str, Any]]:
    return _collect_label_box_targets(
        label_slice_2d=label_slice_2d,
        label_ids=label_ids,
        box_prompt_variant=box_prompt_variant,
    )


def build_prompt_mask_from_label_boxes(
    volume: np.ndarray,
    prompt_slice_id: int,
    label_slice_2d: np.ndarray,
    pam: PAMInferencer,
    prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    label_ids: Optional[List[int]] = None,
    box_prompt_variant: str = BOX_PROMPT_VARIANT_BASELINE,
) -> Tuple[np.ndarray, PromptMeta]:
    variant = normalize_box_prompt_variant(box_prompt_variant)
    box_targets = _collect_label_box_targets(
        label_slice_2d,
        label_ids=label_ids,
        box_prompt_variant=variant,
    )
    if len(box_targets) <= 0:
        raise RuntimeError("No positive labels found for GT box prompting on this slice.")

    h, w = np.asarray(label_slice_2d).shape
    merged_prompt_mask = np.zeros((h, w), dtype=np.uint8)
    prompt_masks = pam.prompt_masks_from_boxes(
        volume=volume,
        prompt_slice_id=int(prompt_slice_id),
        boxes_xyxy=[tuple(int(v) for v in meta["box_xyxy"]) for meta in box_targets],
        prob_thresh=float(prob_thresh),
    )
    if len(prompt_masks) != len(box_targets):
        raise RuntimeError(
            f"prompt_masks_from_boxes returned {len(prompt_masks)} masks for {len(box_targets)} boxes."
        )
    box_details: List[Dict[str, Any]] = []
    component_area = 0
    for meta, one_prompt_mask in zip(box_targets, prompt_masks):
        one_prompt_mask = np.asarray(one_prompt_mask > 0, dtype=np.uint8)
        merged_prompt_mask = np.maximum(merged_prompt_mask, one_prompt_mask)
        component_area += int(meta["component_area"])
        box_details.append(
            {
                "label_id": int(meta["label_id"]),
                "component_index": int(meta.get("component_index", 0)),
                "raw_box_xyxy": [int(v) for v in meta.get("raw_box_xyxy", meta["box_xyxy"])],
                "raw_box_area": int(meta.get("raw_box_area", meta["box_area"])),
                "box_xyxy": [int(v) for v in meta["box_xyxy"]],
                "box_area": int(meta["box_area"]),
                "box_prompt_variant": variant,
                "box2mask_prompt_area": int(np.sum(one_prompt_mask > 0)),
            }
        )

    if int(np.sum(merged_prompt_mask)) <= 0:
        raise RuntimeError("Merged prompt mask from per-label GT boxes is empty.")

    return merged_prompt_mask.astype(np.uint8), _build_prompt_meta(
        prompt_source="box2mask",
        prompt_mask_area=int(np.sum(merged_prompt_mask > 0)),
        component_area=int(component_area),
        union_box_xyxy=_merge_box_xyxy_list([x["box_xyxy"] for x in box_details]),
        box_count=int(len(box_details)),
        uses_box2mask=True,
        boxes=box_details,
    )


def build_prompt_mask_from_slice(
    label_slice_2d: np.ndarray,
    label_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, PromptMeta]:
    positive_ids = _resolve_positive_label_ids(label_slice_2d, label_ids=label_ids)
    if len(positive_ids) <= 0:
        raise RuntimeError("No positive labels found for full-mask prompting on this slice.")
    prompt_mask = np.isin(np.asarray(label_slice_2d, dtype=np.int32), positive_ids).astype(np.uint8)
    prompt_area = int(np.sum(prompt_mask > 0))
    if prompt_area <= 0:
        raise RuntimeError("Full-mask prompt is empty.")
    return prompt_mask.astype(np.uint8), _build_prompt_meta(
        prompt_source="slice",
        prompt_mask_area=int(prompt_area),
        component_area=int(prompt_area),
        union_box_xyxy=None,
        box_count=0,
        uses_box2mask=False,
        boxes=[],
    )


def build_prompt_mask_from_annotation(
    volume: np.ndarray,
    prompt_slice_id: int,
    label_slice_2d: np.ndarray,
    pam: PAMInferencer,
    prompt_annotation_type: str = "box",
    prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    label_ids: Optional[List[int]] = None,
    box_prompt_variant: str = BOX_PROMPT_VARIANT_BASELINE,
) -> Tuple[np.ndarray, PromptMeta]:
    mode = normalize_prompt_annotation_type(prompt_annotation_type)
    if mode == "slice":
        return build_prompt_mask_from_slice(label_slice_2d=label_slice_2d, label_ids=label_ids)
    return build_prompt_mask_from_label_boxes(
        volume=volume,
        prompt_slice_id=prompt_slice_id,
        label_slice_2d=label_slice_2d,
        pam=pam,
        prob_thresh=prob_thresh,
        label_ids=label_ids,
        box_prompt_variant=box_prompt_variant,
    )


def _clip01_mean(*values: float) -> float:
    if len(values) <= 0:
        return 0.0
    return float(np.clip(np.mean(np.asarray(values, dtype=np.float32)), 0.0, 1.0))


def _new_slice_evidence(slice_id: int) -> SliceEvidence:
    return {
        "slice_id": int(slice_id),
        "selection_prompt_source": "none",
        "prompt_candidate_available": False,
        "cam_pass": False,
        "logits_ok": False,
        "prompt_mask": None,
        "prompt_area": 0,
        "prompt_area_ratio": 0.0,
        "prompt_area_score": 0.0,
        "prompt_center_score": 0.0,
        "prompt_border_fraction": 1.0,
        "cam_score": 0.0,
        "cam_reliability": 0.0,
        "cam_focus_mass_ratio": 0.0,
        "cam_focus_compactness": 0.0,
        "confidence_score": 0.0,
        "structure_score": 0.0,
        "continuity_score": 0.0,
        "class_prompt_masks": {},
        "class_prompt_areas": {},
        "class_prompt_component_areas": {},
        "class_prompt_sources": {},
        "class_box2mask_prompt_areas": {},
        "class_prompt_box_counts": {},
        "class_prompt_box_areas": {},
        "class_prompt_build_errors": {},
        "class_confidence_scores": {},
        "class_structure_scores": {},
        "class_continuity_scores": {},
        "class_dice": {},
        "class_pred_voxels": {},
        "class_gt_voxels": {},
        "selected_class_pred_masks": {},
        "agent_asked": False,
        "agent_answer": "",
        "agent_raw_text": "",
        "agent_filter_pass": False,
        "required_label_ids": [],
        "slice_positive_label_ids": [],
        "all_required_label_ids_present": False,
        "missing_required_label_ids": [],
        "missing_required_label_count": 0,
        "required_label_area_ratios": {},
        "required_label_area_scores": {},
        "min_required_label_area_ratio": 0.0,
        "min_required_label_area_score": 0.0,
        "score": 0.0,
        "retained_for_scoring": False,
        "filter_reason": "not_evaluated",
    }


def _drop_internal_masks(rows: List[SliceEvidence]) -> None:
    for row in rows:
        row.pop("prompt_mask", None)
        row.pop("class_prompt_masks", None)
        row.pop("selected_class_pred_masks", None)


def _prepare_label_prompt_stage(
    volume: np.ndarray,
    label: np.ndarray,
    pam: PAMInferencer,
    args: argparse.Namespace,
) -> tuple[List[SliceEvidence], Dict[str, Any]]:
    if volume.shape != label.shape:
        raise ValueError(f"volume/label shape mismatch: {volume.shape} vs {label.shape}")
    d, h, w = volume.shape
    rows = [_new_slice_evidence(sid) for sid in range(d)]
    required_label_ids = extract_required_label_ids(label)
    prompt_annotation_type = normalize_prompt_annotation_type(
        getattr(args, "prompt_annotation_type", "slice")
    )
    gt = np.asarray(label > 0, dtype=bool)
    positive_slice_ids = np.where(gt.reshape(d, -1).sum(axis=1) > 0)[0].astype(int)
    center_start, center_end = center_slice_bounds(d, CENTER_SLICE_FRACTION)
    center_ids = set(range(center_start, center_end))

    for sid in range(d):
        row = rows[int(sid)]
        row.update(
            compute_required_label_area_diagnostics(
                label_slice_2d=np.asarray(label[int(sid)], dtype=np.int32),
                required_label_ids=required_label_ids,
            )
        )

    prompt_candidate_count = 0
    all_label_slice_count = 0
    missing_required_label_count = 0
    center_positive_count = 0
    failed_prompt_count = 0
    for sid in range(d):
        row = rows[int(sid)]
        if bool(row.get("all_required_label_ids_present", False)):
            all_label_slice_count += 1
        else:
            missing_required_label_count += int(sid in positive_slice_ids)

        if sid not in center_ids:
            row["filter_reason"] = "outside_center40"
            continue

        if sid not in positive_slice_ids:
            row["filter_reason"] = "center_background"
            continue

        center_positive_count += 1

        label_slice = np.asarray(label[int(sid)], dtype=np.int32)
        class_masks: Dict[str, np.ndarray] = {}
        class_prompt_areas: Dict[str, int] = {}
        class_prompt_component_areas: Dict[str, int] = {}
        class_prompt_sources: Dict[str, str] = {}
        class_box2mask_prompt_areas: Dict[str, int] = {}
        class_prompt_box_counts: Dict[str, int] = {}
        class_prompt_box_areas: Dict[str, int] = {}
        class_prompt_build_errors: Dict[str, str] = {}
        prompt_build_failed = False
        for label_id in required_label_ids:
            key = str(int(label_id))
            class_label_slice = np.asarray(label_slice == int(label_id), dtype=np.uint8)
            if int(np.sum(class_label_slice)) <= 0:
                class_masks[key] = np.zeros((h, w), dtype=bool)
                class_prompt_areas[key] = 0
                class_prompt_component_areas[key] = 0
                class_prompt_sources[key] = prompt_annotation_type
                class_box2mask_prompt_areas[key] = 0
                class_prompt_box_counts[key] = 0
                class_prompt_box_areas[key] = 0
                continue
            try:
                prompt_mask, prompt_meta = build_prompt_mask_from_annotation(
                    volume=volume,
                    prompt_slice_id=int(sid),
                    label_slice_2d=class_label_slice,
                    pam=pam,
                    prompt_annotation_type=prompt_annotation_type,
                    prob_thresh=float(
                        getattr(args, "eval_prob_thresh", DEFAULT_EVAL_PROB_THRESH)
                    ),
                    box_prompt_variant=str(
                        getattr(args, "box_prompt_variant", BOX_PROMPT_VARIANT_BASELINE)
                    ),
                )
            except Exception as exc:
                prompt_build_failed = True
                class_prompt_build_errors[key] = " ".join(str(exc).split())
                class_masks[key] = np.zeros((h, w), dtype=bool)
                class_prompt_areas[key] = 0
                class_prompt_component_areas[key] = int(np.sum(class_label_slice > 0))
                class_prompt_sources[key] = prompt_annotation_type
                class_box2mask_prompt_areas[key] = 0
                class_prompt_box_counts[key] = 0
                class_prompt_box_areas[key] = 0
                continue
            class_masks[key] = np.asarray(prompt_mask > 0, dtype=bool)
            class_prompt_areas[key] = int(np.sum(class_masks[key]))
            class_prompt_component_areas[key] = int(prompt_meta.get("sum_component_area", 0))
            class_prompt_sources[key] = str(prompt_meta.get("prompt_source", prompt_annotation_type))
            class_box2mask_prompt_areas[key] = int(
                prompt_box2mask_area(prompt_meta) or 0
            )
            class_prompt_box_counts[key] = int(prompt_meta.get("box_count", 0))
            class_prompt_box_areas[key] = int(prompt_meta.get("union_box_area", 0))
        row["class_prompt_masks"] = class_masks
        row["class_prompt_areas"] = class_prompt_areas
        row["class_prompt_component_areas"] = class_prompt_component_areas
        row["class_prompt_sources"] = class_prompt_sources
        row["class_box2mask_prompt_areas"] = class_box2mask_prompt_areas
        row["class_prompt_box_counts"] = class_prompt_box_counts
        row["class_prompt_box_areas"] = class_prompt_box_areas
        row["class_prompt_build_errors"] = class_prompt_build_errors
        row["prompt_candidate_available"] = True
        row["selection_prompt_source"] = prompt_annotation_type
        row["prompt_area"] = int(sum(class_prompt_areas.values()))
        row["prompt_area_ratio"] = float(row["prompt_area"] / float(max(1, h * w)))
        if prompt_build_failed:
            failed_prompt_count += 1
        row["filter_reason"] = "pending_pam_logits"
        prompt_candidate_count += 1

    return rows, {
        "prompt_annotation_type": str(prompt_annotation_type),
        "required_label_ids": list(required_label_ids),
        "positive_slice_count": int(len(positive_slice_ids)),
        "center_slice_start": int(center_start),
        "center_slice_end": int(center_end),
        "center_slice_count": int(center_end - center_start),
        "center_positive_slice_count": int(center_positive_count),
        "all_label_slice_count": int(all_label_slice_count),
        "missing_required_label_count": int(missing_required_label_count),
        "prompt_candidate_count": int(prompt_candidate_count),
        "failed_prompt_count": int(failed_prompt_count),
        "required_label_min_area_ratio": float(REQUIRED_LABEL_MIN_AREA_RATIO),
        "image_height": int(h),
        "image_width": int(w),
    }


def _score_propagation_stage(
    volume: np.ndarray,
    pam: PAMInferencer,
    rows: List[SliceEvidence],
    label: Optional[np.ndarray] = None,
    prob_thresh: float = DEFAULT_EVAL_PROB_THRESH,
    case_id: Optional[str] = None,
) -> int:
    if label is None:
        raise ValueError("label is required for classwise PAM scoring.")
    label_arr = np.asarray(label, dtype=np.int32)
    required_label_ids = extract_required_label_ids(label_arr)
    valid_count = 0
    selected_rows = [
        row
        for row in rows
        if bool(row.get("prompt_candidate_available", False))
        and bool(row.get("agent_filter_pass", False))
    ]
    progress_desc = f"pam {case_id}".strip() if case_id else "pam"
    progress_bar = make_optional_tqdm(
        total=len(selected_rows),
        desc=progress_desc,
        unit="slice",
    )

    for candidate_idx, row in enumerate(selected_rows, start=1):
        sid = int(row["slice_id"])
        class_masks = dict(row.get("class_prompt_masks", {}))
        missing_prompt_ids = [
            int(label_id)
            for label_id in required_label_ids
            if int(np.sum(np.asarray(class_masks.get(str(int(label_id)), []), dtype=bool))) <= 0
        ]
        if missing_prompt_ids:
            row["logits_error"] = f"agent_pass_but_empty_class_prompt_ids={missing_prompt_ids}"
            row["logits_ok"] = False
            row["filter_reason"] = "class_prompt_or_pam_failed"
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        prompt_masks = [
            np.asarray(class_masks[str(int(label_id))], dtype=bool)
            for label_id in required_label_ids
        ]
        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"slice={sid} classes={required_label_ids} full_z={volume.shape[0]}"
            )
        else:
            label_text = str(case_id) if case_id else "case"
            print(
                f"[PAM][{label_text}] slice {candidate_idx}/{len(selected_rows)} "
                f"slice={sid} classes={required_label_ids}"
            )
        try:
            logits_list = pam.propagate_logits_batch(
                volume=volume,
                prompt_slice_ids=[sid for _ in required_label_ids],
                prompt_masks_2d=prompt_masks,
            )
        except Exception as exc:
            row["logits_error"] = str(exc)
            row["logits_ok"] = False
            row["filter_reason"] = "class_prompt_or_pam_failed"
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        if len(logits_list) != len(required_label_ids):
            raise RuntimeError(
                f"propagate_logits_batch returned {len(logits_list)} logits volumes for "
                f"{len(required_label_ids)} class prompts."
            )

        class_confidence_scores: Dict[str, float] = {}
        class_structure_scores: Dict[str, float] = {}
        class_continuity_scores: Dict[str, float] = {}
        class_dice: Dict[str, float] = {}
        class_pred_voxels: Dict[str, int] = {}
        class_gt_voxels: Dict[str, int] = {}

        for label_id, prompt_mask, logits in zip(required_label_ids, prompt_masks, logits_list):
            key = str(int(label_id))
            logits = np.asarray(logits, dtype=np.float32)
            base_metrics = compute_logits_uncertainty_and_confidence(
                logits=logits,
                prompt_slice_id=sid,
            )
            prob = np.asarray(base_metrics["prob"], dtype=np.float32)
            entropy_map = np.asarray(base_metrics["entropy_map"], dtype=np.float32)
            observation_metrics = compute_prompt_observation_features(
                prob=prob,
                uncertainty_volume=entropy_map,
                prompt_slice_id=sid,
                prompt_mask_2d=np.asarray(prompt_mask, dtype=bool),
            )
            prop_metrics = compute_propagation_reasoning_features(prob, sid)
            pred_mask = np.asarray(prob >= float(prob_thresh), dtype=bool)
            gt_class = np.asarray(label_arr == int(label_id), dtype=bool)

            class_confidence_scores[key] = compute_aggrigator_confidence_score(
                uncertainty_map=entropy_map,
                pred_mask=np.asarray(pred_mask, dtype=np.uint8),
            )
            class_structure_scores[key] = compute_prompt_anchored_core_score(
                prob=prob,
                prompt_slice_id=sid,
                prob_thresh=float(prob_thresh),
            )
            class_continuity_scores[key] = _clip01_mean(
                float(prop_metrics["prop_continuity_score"]),
                float(prop_metrics["prop_nonzero_slice_ratio"]),
                float(observation_metrics["support_continuity"]),
            )
            class_dice[key] = float(
                dice_coefficient(
                    np.asarray(pred_mask, dtype=np.uint8),
                    np.asarray(gt_class, dtype=np.uint8),
                )
            )
            class_pred_voxels[key] = int(np.sum(pred_mask))
            class_gt_voxels[key] = int(np.sum(gt_class))

        row["class_confidence_scores"] = class_confidence_scores
        row["class_structure_scores"] = class_structure_scores
        row["class_continuity_scores"] = class_continuity_scores
        row["class_dice"] = class_dice
        row["class_pred_voxels"] = class_pred_voxels
        row["class_gt_voxels"] = class_gt_voxels
        row["confidence_score"] = classwise_macro_mean(class_confidence_scores, required_label_ids)
        row["structure_score"] = classwise_macro_mean(class_structure_scores, required_label_ids)
        row["continuity_score"] = classwise_macro_mean(class_continuity_scores, required_label_ids)
        row["selected_macro_dice"] = classwise_macro_mean(class_dice, required_label_ids)
        row["logits_ok"] = True
        row["retained_for_scoring"] = True
        row["filter_reason"] = ""
        valid_count += 1
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    _drop_internal_masks(rows)
    return int(valid_count)


def _selection_filter_reason(row: SliceEvidence) -> str:
    if not bool(row.get("prompt_candidate_available", False)):
        return str(row.get("filter_reason", "not_prompt_candidate"))
    if not bool(row.get("logits_ok", False)):
        return str(row.get("filter_reason", "class_prompt_or_pam_failed"))
    if bool(row.get("agent_asked", False)) and not bool(row.get("agent_filter_pass", False)):
        return "agent_rejected"
    return ""


def _build_slice_row(row: SliceEvidence) -> Dict[str, Any]:
    return {
        "slice_id": int(row["slice_id"]),
        "Score": float(row["score"]),
        "ConfidenceScore": float(row["confidence_score"]),
        "StructureScore": float(row["structure_score"]),
        "ContinuityScore": float(row["continuity_score"]),
        "ContinuityGatePass": bool(row.get("continuity_gate_pass", False)),
        "ContinuityGateThreshold": float(row.get("continuity_gate_threshold", 0.0)),
        "SelectionEligible": bool(row.get("selection_eligible", False)),
        "PromptCandidateAvailable": bool(row.get("prompt_candidate_available", False)),
        "LogitsComputed": bool(row["logits_ok"]),
        "RetainedForScoring": bool(row["retained_for_scoring"]),
        "FilterReason": str(row["filter_reason"]),
        "ClassScores": {
            "confidence": {
                str(key): float(value)
                for key, value in dict(row.get("class_confidence_scores", {})).items()
            },
            "structure": {
                str(key): float(value)
                for key, value in dict(row.get("class_structure_scores", {})).items()
            },
            "continuity": {
                str(key): float(value)
                for key, value in dict(row.get("class_continuity_scores", {})).items()
            },
        },
        "ClassDice": {
            str(key): float(value)
            for key, value in dict(row.get("class_dice", {})).items()
        },
        "ClassPromptAreas": {
            str(key): int(value)
            for key, value in dict(row.get("class_prompt_areas", {})).items()
        },
        "ClassPredVoxels": {
            str(key): int(value)
            for key, value in dict(row.get("class_pred_voxels", {})).items()
        },
        "ClassGtVoxels": {
            str(key): int(value)
            for key, value in dict(row.get("class_gt_voxels", {})).items()
        },
        "AgentAsked": bool(row.get("agent_asked", False)),
        "AgentAnswer": str(row.get("agent_answer", "")),
        "AgentFilterPass": bool(row.get("agent_filter_pass", False)),
        "required_label_ids": [int(v) for v in row.get("required_label_ids", [])],
        "slice_positive_label_ids": [int(v) for v in row.get("slice_positive_label_ids", [])],
        "all_required_label_ids_present": bool(row.get("all_required_label_ids_present", False)),
        "missing_required_label_count": int(row.get("missing_required_label_count", 0)),
        "missing_required_label_ids": [
            int(v) for v in row.get("missing_required_label_ids", [])
        ],
        "required_label_area_ratios": {
            str(key): float(value)
            for key, value in dict(row.get("required_label_area_ratios", {})).items()
        },
        "required_label_area_scores": {
            str(key): float(value)
            for key, value in dict(row.get("required_label_area_scores", {})).items()
        },
        "min_required_label_area_ratio": float(row.get("min_required_label_area_ratio", 0.0)),
        "min_required_label_area_score": float(row.get("min_required_label_area_score", 0.0)),
    }


def _build_topn_row(rank: int, row: SliceEvidence) -> Dict[str, Any]:
    return {
        "rank": int(rank),
        "slice_id": int(row["slice_id"]),
        "Score": float(row["score"]),
        "ConfidenceScore": float(row["confidence_score"]),
        "StructureScore": float(row["structure_score"]),
        "ContinuityScore": float(row["continuity_score"]),
        "ClassDice": {
            str(key): float(value)
            for key, value in dict(row.get("class_dice", {})).items()
        },
        "AgentAsked": bool(row.get("agent_asked", False)),
        "AgentAnswer": str(row.get("agent_answer", "")),
        "ContinuityGatePass": bool(row.get("continuity_gate_pass", False)),
        "ContinuityGateThreshold": float(row.get("continuity_gate_threshold", 0.0)),
        "SelectionEligible": bool(row.get("selection_eligible", False)),
    }


def _load_label_from_args(args: argparse.Namespace) -> np.ndarray:
    label_path = getattr(args, "label", None)
    if label_path is None or not str(label_path).strip():
        raise ValueError("label array or args.label path is required for GT prompt candidate selection.")
    return load_volume(Path(str(label_path)))


def _format_pam_logits_failure_diagnostics(
    rows: List[SliceEvidence],
    prompt_candidate_count: int,
    case_id: Optional[str] = None,
    max_examples: int = 3,
) -> str:
    error_counts: Dict[str, int] = {}
    examples: List[str] = []
    pending_count = 0
    for row in rows:
        if not bool(row.get("prompt_candidate_available", False)):
            continue
        if bool(row.get("logits_ok", False)):
            continue
        sid = int(row.get("slice_id", -1))
        error_text = str(row.get("logits_error") or row.get("filter_reason") or "unknown")
        error_counts[error_text] = int(error_counts.get(error_text, 0)) + 1
        if row.get("filter_reason") == "pending_pam_logits" and not row.get("logits_error"):
            pending_count += 1
        if len(examples) < int(max_examples):
            compact_error = " ".join(error_text.split())
            if len(compact_error) > 180:
                compact_error = compact_error[:177] + "..."
            examples.append(f"slice={sid}: {compact_error}")

    top_errors = sorted(error_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:3]
    top_error_text = "; ".join(f"{count}x {error}" for error, count in top_errors) or "none"
    case_text = "" if case_id is None else f"case={case_id}, "
    example_text = "; ".join(examples) or "none"
    return (
        f"{case_text}prompt_candidates={int(prompt_candidate_count)}, "
        f"failed_logits={int(sum(error_counts.values()))}, "
        f"not_attempted_or_pending={int(pending_count)}, "
        f"top_errors=[{top_error_text}], examples=[{example_text}]"
    )


def run_agent_selection(
    volume: np.ndarray,
    rows: List[SliceEvidence],
    args: argparse.Namespace,
    case_id: Optional[str] = None,
    reference_example_bank: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    valid_rows = [row for row in rows if bool(row.get("prompt_candidate_available", False))]
    if len(valid_rows) <= 0:
        raise RuntimeError("No prompt candidate slice is available for Agent filtering.")
    ordered = sorted(
        valid_rows,
        key=lambda row: (
            int(bool(row.get("all_required_label_ids_present", False))),
            -int(row.get("missing_required_label_count", 0)),
            float(row.get("min_required_label_area_score", 0.0)),
            float(row.get("prompt_area_ratio", 0.0)),
            -abs(float(row.get("slice_id", 0.0)) - (float(volume.shape[0] - 1) / 2.0)),
        ),
        reverse=True,
    )
    agent_responses: List[Dict[str, Any]] = []
    checked_slice_ids: List[int] = []
    passed_slice_ids: List[int] = []
    for row in ordered:
        sid = int(row["slice_id"])
        checked_slice_ids.append(sid)
        target_bucket = (
            "background"
            if len([int(v) for v in row.get("slice_positive_label_ids", [])]) <= 0
            else (
                "complete"
                if bool(row.get("all_required_label_ids_present", False))
                else "incomplete"
            )
        )
        reference_examples = select_agent_reference_examples(
            reference_example_bank=reference_example_bank,
            exclude_slice_ids=[sid],
            args=args,
            target_bucket=target_bucket,
            target_slice_id=sid,
        )
        row["agent_asked"] = True
        try:
            response = ask_agent_slice_complete(
                slice_2d=np.asarray(volume[sid], dtype=np.float32),
                args=args,
                reference_examples=reference_examples,
            )
        except Exception as exc:
            row["agent_answer"] = "error"
            row["agent_raw_text"] = " ".join(str(exc).split())[:200]
            agent_responses.append(
                {
                    "slice_id": sid,
                    "answer": "error",
                    "raw_text": row["agent_raw_text"],
                    "score": float(row.get("score", 0.0)),
                    "reference_examples_used": [
                        {
                            "bucket": str(example["bucket"]),
                            "slice_id": int(example["slice_id"]),
                            "expected_answer": str(example["expected_answer"]),
                        }
                        for example in reference_examples
                    ],
                }
            )
            raise
        answer = str(response.get("answer", "no")).strip().lower()
        raw_text = str(response.get("raw_text", ""))[:200]
        row["agent_answer"] = answer
        row["agent_raw_text"] = raw_text
        row["agent_filter_pass"] = bool(answer == "yes")
        agent_responses.append(
            {
                "slice_id": sid,
                "answer": answer,
                "raw_text": raw_text,
                "score": float(row.get("score", 0.0)),
                "attempts": int(response.get("attempts", 1)),
                "reference_examples_used": list(response.get("reference_examples_used", [])),
            }
        )
        label_text = str(case_id) if case_id else "case"
        print(
            f"[Agent][{label_text}] slice={sid} score={float(row.get('score', 0.0)):.4f} "
            f"answer={answer}"
        )
        if answer == "yes":
            passed_slice_ids.append(sid)
            row["filter_reason"] = "pending_pam_logits"
            break
        else:
            row["filter_reason"] = "agent_rejected"
    if len(passed_slice_ids) <= 0:
        raise RuntimeError(
            "Agent rejected all prompt candidate slices. "
            f"checked_slice_ids={checked_slice_ids}"
        )
    return {
        "agent_checked_slice_ids": checked_slice_ids,
        "agent_pass_slice_ids": passed_slice_ids,
        "agent_pass_count": int(len(passed_slice_ids)),
        "agent_responses": agent_responses,
    }


def prepare_best_slice_candidates(
    volume: np.ndarray,
    pam: PAMInferencer,
    args: argparse.Namespace,
    out_dir: Path,
    case_id: Optional[str] = None,
    label: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    del out_dir
    if label is None:
        label = _load_label_from_args(args)
    if volume.shape != label.shape:
        raise ValueError(f"volume/label shape mismatch: {volume.shape} vs {label.shape}")

    d, h, w = volume.shape
    print(f"[Info] Volume shape: D={d}, H={h}, W={w}")
    rows, prompt_summary = _prepare_label_prompt_stage(
        volume=volume,
        label=np.asarray(label),
        pam=pam,
        args=args,
    )
    print(
        "[Info] Center-slice class prompt candidates: "
        f"{int(prompt_summary['prompt_candidate_count'])}/{d} "
        f"(positive_slices={int(prompt_summary['positive_slice_count'])}, "
        f"center_slices={int(prompt_summary['center_slice_count'])}, "
        f"center_positive_slices={int(prompt_summary['center_positive_slice_count'])}, "
        f"failed_prompts={int(prompt_summary['failed_prompt_count'])})"
    )
    if int(prompt_summary["prompt_candidate_count"]) <= 0:
        raise RuntimeError(
            "No center-30% slice is available for classwise prompt candidate scoring. "
            f"required_label_ids={prompt_summary['required_label_ids']}"
        )

    agent_pack = run_agent_selection(
        volume=volume,
        rows=rows,
        args=args,
        case_id=case_id,
        reference_example_bank=build_agent_reference_example_bank(
            volume=volume,
            rows=rows,
            center_start=int(prompt_summary["center_slice_start"]),
            center_end=int(prompt_summary["center_slice_end"]),
            args=args,
        ),
    )
    print(
        f"[Info] Agent-passed candidates: {int(agent_pack['agent_pass_count'])}/{int(prompt_summary['prompt_candidate_count'])}"
    )

    valid_candidate_count = _score_propagation_stage(
        volume=volume,
        pam=pam,
        rows=rows,
        label=np.asarray(label),
        prob_thresh=float(getattr(args, "eval_prob_thresh", DEFAULT_EVAL_PROB_THRESH)),
        case_id=case_id,
    )
    print(
        f"[Info] PAM logits candidates: {int(valid_candidate_count)}/{d} "
        f"(center40, prompt_candidates={int(prompt_summary['prompt_candidate_count'])})"
    )
    if valid_candidate_count <= 0:
        diagnostics = _format_pam_logits_failure_diagnostics(
            rows=rows,
            prompt_candidate_count=int(prompt_summary["prompt_candidate_count"]),
            case_id=case_id,
        )
        raise RuntimeError(
            "No GT prompt candidate completed PAM logits propagation. "
            f"{diagnostics}"
        )

    return {
        "case_id": case_id,
        "volume_shape": [int(d), int(h), int(w)],
        "rows": rows,
        "prompt_annotation_type": str(prompt_summary["prompt_annotation_type"]),
        "required_label_ids": list(prompt_summary["required_label_ids"]),
        "positive_slice_count": int(prompt_summary["positive_slice_count"]),
        "center_slice_start": int(prompt_summary["center_slice_start"]),
        "center_slice_end": int(prompt_summary["center_slice_end"]),
        "center_slice_count": int(prompt_summary["center_slice_count"]),
        "center_positive_slice_count": int(prompt_summary["center_positive_slice_count"]),
        "all_label_slice_count": int(prompt_summary["all_label_slice_count"]),
        "missing_required_label_count": int(prompt_summary["missing_required_label_count"]),
        "prompt_candidate_count": int(prompt_summary["prompt_candidate_count"]),
        "failed_prompt_count": int(prompt_summary["failed_prompt_count"]),
        "required_label_min_area_ratio": float(prompt_summary["required_label_min_area_ratio"]),
        "agent_checked_slice_ids": list(agent_pack["agent_checked_slice_ids"]),
        "agent_pass_slice_ids": list(agent_pack["agent_pass_slice_ids"]),
        "agent_pass_count": int(agent_pack["agent_pass_count"]),
        "agent_responses": list(agent_pack["agent_responses"]),
    }


def compute_selection_score(row: SliceEvidence) -> float:
    return float(
        np.clip(
            float(SELECTION_SCORE_WEIGHTS["confidence"]) * float(row["confidence_score"])
            + float(SELECTION_SCORE_WEIGHTS["structure"]) * float(row["structure_score"])
            + float(SELECTION_SCORE_WEIGHTS["continuity"]) * float(row["continuity_score"]),
            0.0,
            1.0,
        )
    )


def finalize_best_slice_selection(
    prepared: Dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    reference_feature_dicts: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    del out_dir, reference_feature_dicts
    rows = [dict(row) for row in prepared.get("rows", [])]
    if len(rows) <= 0:
        raise RuntimeError("No candidate is available for final slice selection.")

    selection_score = np.zeros(len(rows), dtype=np.float32)
    valid = np.zeros(len(rows), dtype=bool)
    for idx, row in enumerate(rows):
        valid[idx] = bool(row.get("logits_ok", False))
    if int(np.sum(valid)) <= 0:
        raise RuntimeError("No valid candidate is available for final slice selection.")

    for idx, row in enumerate(rows):
        if not valid[idx]:
            continue
        selection_score[idx] = compute_selection_score(row)
        row["selection_eligible"] = True
        row["continuity_gate_pass"] = True
        row["continuity_gate_threshold"] = 0.0

    valid_ids = np.where(valid)[0].astype(int)
    topn = int(max(1, getattr(args, "topn", 10)))
    order = sorted(
        valid_ids.tolist(),
        key=lambda idx: (
            float(selection_score[idx]),
            float(rows[idx]["continuity_score"]),
            float(rows[idx]["structure_score"]),
            float(rows[idx]["confidence_score"]),
        ),
        reverse=True,
    )
    selected_slice_id = int(prepared.get("agent_selected_slice_id", rows[int(order[0])]["slice_id"]))
    selected_matches = [row for row in rows if int(row.get("slice_id", -1)) == selected_slice_id]
    if len(selected_matches) <= 0:
        raise RuntimeError(f"Agent selected slice {selected_slice_id}, but it is missing from rows.")
    best_row = selected_matches[0]

    for idx, row in enumerate(rows):
        row["score"] = float(selection_score[idx])
        row["retained_for_scoring"] = bool(valid[idx])
        row["filter_reason"] = _selection_filter_reason(row)

    slice_rows = sorted(
        [_build_slice_row(row) for row in rows],
        key=lambda item: (
            int(bool(item["RetainedForScoring"])),
            int(bool(item["SelectionEligible"])),
            float(item["Score"]),
            float(item["ContinuityScore"]),
            float(item["StructureScore"]),
        ),
        reverse=True,
    )
    topn_list = [_build_topn_row(rank, rows[idx]) for rank, idx in enumerate(order[:topn], start=1)]

    prompt_candidate_count = int(
        sum(int(bool(row.get("prompt_candidate_available", False))) for row in rows)
    )
    reasoning_summary = {
        "score_source": SELECTION_SOURCE,
        "score_profile": SELECTION_SCORE_PROFILE,
        "best_slice_id": int(best_row["slice_id"]),
        "agent_selected_slice_id": int(selected_slice_id),
        "selected_slice_id_after_agent_filter": int(selected_slice_id),
        "agent_checked_slice_ids": list(prepared.get("agent_checked_slice_ids", [])),
        "agent_pass_slice_ids": list(prepared.get("agent_pass_slice_ids", [])),
        "agent_pass_count": int(prepared.get("agent_pass_count", 0)),
        "agent_responses": list(prepared.get("agent_responses", [])),
        "required_label_ids": list(prepared.get("required_label_ids", [])),
        "positive_slice_count": int(prepared.get("positive_slice_count", prompt_candidate_count)),
        "center_slice_start": int(prepared.get("center_slice_start", 0)),
        "center_slice_end": int(prepared.get("center_slice_end", 0)),
        "center_slice_count": int(prepared.get("center_slice_count", 0)),
        "center_positive_slice_count": int(prepared.get("center_positive_slice_count", 0)),
        "prompt_candidate_count": int(prompt_candidate_count),
        "valid_candidate_count": int(np.sum(valid)),
        "selection_eligible_count": int(np.sum(valid)),
        "selected_label_dice": {
            str(key): float(value)
            for key, value in dict(best_row.get("class_dice", {})).items()
        },
        "selected_macro_dice": float(best_row.get("selected_macro_dice", 0.0)),
        "best_confidence_score": float(best_row["confidence_score"]),
        "best_structure_score": float(best_row["structure_score"]),
        "best_continuity_score": float(best_row["continuity_score"]),
    }

    return {
        "best_slice_id": int(best_row["slice_id"]),
        "topn": topn_list,
        "candidate_ids": [
            int(row["slice_id"])
            for row in rows
            if bool(row.get("prompt_candidate_available", False))
        ],
        "valid_candidate_ids": [int(rows[idx]["slice_id"]) for idx in valid_ids.tolist()],
        "selection_eligible_ids": [int(rows[idx]["slice_id"]) for idx in valid_ids.tolist()],
        "selection_source": SELECTION_SOURCE,
        "reasoning_summary": reasoning_summary,
        "slice_rows": slice_rows,
        "selected_label_dice": dict(best_row.get("class_dice", {})),
        "selected_label_pred_voxels": dict(best_row.get("class_pred_voxels", {})),
        "selected_label_gt_voxels": dict(best_row.get("class_gt_voxels", {})),
        "selected_macro_dice": float(best_row.get("selected_macro_dice", 0.0)),
        "agent_checked_slice_ids": list(prepared.get("agent_checked_slice_ids", [])),
        "agent_responses": list(prepared.get("agent_responses", [])),
        "config": {
            "topn": int(topn),
            "prompt_annotation_type": str(prepared.get("prompt_annotation_type", getattr(args, "prompt_annotation_type", "slice"))),
            "required_label_ids": list(prepared.get("required_label_ids", [])),
            "seed": int(args.seed),
            "pam_backend": str(args.pam_backend),
            "score_profile": SELECTION_SCORE_PROFILE,
            "score_weights": dict(SELECTION_SCORE_WEIGHTS),
            "center_slice_fraction": float(CENTER_SLICE_FRACTION),
            "agent_model_provider": str(getattr(args, "agent_model_provider", AGENT_MODEL_PROVIDER)),
            "agent_model": str(getattr(args, "agent_model", AGENT_MODEL)),
            "agent_reasoning_effort": str(
                getattr(args, "agent_reasoning_effort", AGENT_REASONING_EFFORT)
            ),
            "box_prompt_variant": str(
                getattr(args, "box_prompt_variant", BOX_PROMPT_VARIANT_BASELINE)
            ),
            "agent_reference_examples_enabled": bool(
                parse_bool_like(
                    getattr(
                        args,
                        "agent_reference_examples_enabled",
                        AGENT_REFERENCE_EXAMPLES_ENABLED,
                    ),
                    default=AGENT_REFERENCE_EXAMPLES_ENABLED,
                )
            ),
            "agent_reference_examples_per_bucket": int(
                getattr(
                    args,
                    "agent_reference_examples_per_bucket",
                    AGENT_REFERENCE_EXAMPLES_PER_BUCKET,
                )
            ),
            "agent_reference_candidate_depth_per_bucket": int(
                getattr(
                    args,
                    "agent_reference_candidate_depth_per_bucket",
                    AGENT_REFERENCE_CANDIDATE_DEPTH_PER_BUCKET,
                )
            ),
            "agent_reference_min_slice_gap": int(
                getattr(
                    args,
                    "agent_reference_min_slice_gap",
                    AGENT_REFERENCE_MIN_SLICE_GAP,
                )
            ),
        },
    }

def select_best_slice(
    volume: np.ndarray,
    pam: PAMInferencer,
    args: argparse.Namespace,
    out_dir: Path,
    reference_feature_dicts: Optional[List[Dict[str, float]]] = None,
    case_id: Optional[str] = None,
    label: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    del reference_feature_dicts
    prepared = prepare_best_slice_candidates(
        volume=volume,
        pam=pam,
        args=args,
        out_dir=out_dir,
        case_id=case_id,
        label=label,
    )
    return finalize_best_slice_selection(
        prepared=prepared,
        args=args,
        out_dir=out_dir,
    )

def build_pam_backend(args: argparse.Namespace) -> PAMInferencer:
    if args.pam_backend != "real":
        raise ValueError("Global policy: only --pam_backend real is allowed.")
    prompt_annotation_type = normalize_prompt_annotation_type(
        getattr(args, "prompt_annotation_type", "slice")
    )
    return RealPAMWrapper(
        pam_root=args.pam_root,
        checkpoint_path=args.pam_ckpt,
        box2mask_checkpoint_path=getattr(args, "pam_box2mask_ckpt", None),
        box2mask_runtime_mode=getattr(args, "pam_box2mask_runtime_mode", BOX2MASK_RUNTIME_MODE_DEFAULT),
        load_box2mask=bool(prompt_annotation_type == "box"),
        device=choose_device(args.device),
        modality=args.pam_modality,
        target_size=args.pam_target_size,
        dynamic_crop_ratio=args.pam_dynamic_crop_ratio,
        prop_batch_size=args.pam_prop_batch_size,
        emb_batch_size=args.pam_emb_batch_size,
    )
