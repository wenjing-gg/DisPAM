"""Target-visibility agents used after PASE ranking.

Candidates are queried once in descending PASE order. The first affirmative
visibility decision selects the guiding slice.
"""

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


class VisibilityAgent(ABC):
    @abstractmethod
    def is_visible(
        self,
        case_id: str,
        target: str,
        slice_index: int,
        image_2d: np.ndarray,
        prompt: str,
    ) -> bool:
        raise NotImplementedError


class ReplayVisibilityAgent(VisibilityAgent):
    """Replay released deterministic decisions for exact reproduction."""

    def __init__(self, decisions_file: Path) -> None:
        payload = json.loads(Path(decisions_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or any(
            type(value) is not bool for value in payload.values()
        ):
            raise ValueError(
                "Replay decisions must contain one boolean per case/target/slice; "
                "regenerate multi-vote records."
            )
        self.decisions: Dict[str, bool] = dict(payload)

    def is_visible(
        self,
        case_id: str,
        target: str,
        slice_index: int,
        image_2d: np.ndarray,
        prompt: str,
    ) -> bool:
        del image_2d, prompt
        key = f"{case_id}/{target}/{int(slice_index)}"
        if key not in self.decisions:
            raise KeyError(f"Missing replay decision {key}.")
        return self.decisions[key]


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
    ) -> bool:
        del case_id, target, slice_index
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
) -> int | None:
    for slice_index in candidate_order:
        if agent.is_visible(
            case_id,
            target,
            int(slice_index),
            volume_zyx[int(slice_index)],
            prompt,
        ):
            return int(slice_index)
    return None
