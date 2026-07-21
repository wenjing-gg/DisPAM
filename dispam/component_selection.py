"""Six-connected 3D component labeling."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


def label_components_6(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got {binary.shape}.")
    if ndi is not None:
        structure = ndi.generate_binary_structure(rank=3, connectivity=1)
        labels, count = ndi.label(binary, structure=structure)
        return np.asarray(labels, dtype=np.int32), int(count)

    labels = np.zeros(binary.shape, dtype=np.int32)
    neighbors = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )
    component_id = 0
    for seed in np.argwhere(binary):
        z0, y0, x0 = (int(value) for value in seed)
        if labels[z0, y0, x0] != 0:
            continue
        component_id += 1
        labels[z0, y0, x0] = component_id
        stack = [(z0, y0, x0)]
        while stack:
            z, y, x = stack.pop()
            for dz, dy, dx in neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx
                if not (
                    0 <= nz < binary.shape[0]
                    and 0 <= ny < binary.shape[1]
                    and 0 <= nx < binary.shape[2]
                ):
                    continue
                if binary[nz, ny, nx] and labels[nz, ny, nx] == 0:
                    labels[nz, ny, nx] = component_id
                    stack.append((nz, ny, nx))
    return labels, component_id
