"""Target-visibility agents used after PASE ranking."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Dict, Sequence
from urllib import request

import numpy as np
from PIL import Image

from .slice_selector import majority_visible


class VisibilityAgent(ABC):
    @abstractmethod
    def is_visible(
        self,
        case_id: str,
        target: str,
        slice_index: int,
        image_2d: np.ndarray,
        prompt: str,
        repetition: int,
    ) -> bool:
        raise NotImplementedError


class ReplayVisibilityAgent(VisibilityAgent):
    """Replay released deterministic decisions for exact reproduction."""

    def __init__(self, decisions_file: Path) -> None:
        payload = json.loads(Path(decisions_file).read_text(encoding="utf-8"))
        self.decisions: Dict[str, object] = dict(payload)

    def is_visible(
        self,
        case_id: str,
        target: str,
        slice_index: int,
        image_2d: np.ndarray,
        prompt: str,
        repetition: int,
    ) -> bool:
        del image_2d, prompt
        key = f"{case_id}/{target}/{int(slice_index)}"
        values = self.decisions.get(key)
        if not isinstance(values, list) or repetition >= len(values):
            raise KeyError(f"Missing replay decision {key}, repetition={repetition}.")
        return bool(values[repetition])


def _png_data_url(image_2d: np.ndarray) -> str:
    image = np.asarray(image_2d, dtype=np.float32)
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 1.0)
    scaled = np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    buffer = BytesIO()
    Image.fromarray(np.asarray(scaled * 255.0, dtype=np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatibleVisibilityAgent(VisibilityAgent):
    """Call an OpenAI-compatible multimodal endpoint with deterministic decoding."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key_env: str = "VLM_API_KEY",
        timeout_seconds: int = 120,
    ) -> None:
        self.model = str(model)
        self.endpoint = str(endpoint)
        self.api_key_env = str(api_key_env)
        self.timeout_seconds = int(timeout_seconds)

    def is_visible(
        self,
        case_id: str,
        target: str,
        slice_index: int,
        image_2d: np.ndarray,
        prompt: str,
        repetition: int,
    ) -> bool:
        del case_id, target, slice_index, repetition
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set.")
        payload = {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _png_data_url(image_2d)},
                        },
                    ],
                }
            ],
        }
        http_request = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = str(result["choices"][0]["message"]["content"]).strip().lower()
        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False
        raise RuntimeError(f"Visibility response must start with Yes or No, got {text!r}.")


def select_visible_slice(
    agent: VisibilityAgent,
    case_id: str,
    target: str,
    candidate_order: Sequence[int],
    volume_zyx: np.ndarray,
    prompt: str,
    repetitions: int = 3,
) -> int | None:
    for slice_index in candidate_order:
        decisions = [
            agent.is_visible(
                case_id,
                target,
                int(slice_index),
                volume_zyx[int(slice_index)],
                prompt,
                repetition,
            )
            for repetition in range(int(repetitions))
        ]
        if majority_visible(decisions):
            return int(slice_index)
    return None
