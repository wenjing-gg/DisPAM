"""Deterministic execution helpers.

Paper mapping:
  Experiments / Implementation Details. Every independent run uses one released
  seed and initializes Python, NumPy, PyTorch, CUDA, and DataLoader workers.
"""

from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(base_seed: int) -> Callable[[int], None]:
    def _seed(worker_id: int) -> None:
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

    return _seed


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
