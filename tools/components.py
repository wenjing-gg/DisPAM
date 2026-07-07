from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def label_components_6(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    labels = np.zeros(binary.shape, dtype=np.int32)
    component_id = 0
    neighbors = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )
    shape = binary.shape
    for seed in np.argwhere(binary):
        z0, y0, x0 = (int(seed[0]), int(seed[1]), int(seed[2]))
        if labels[z0, y0, x0] != 0:
            continue
        component_id += 1
        stack = [(z0, y0, x0)]
        labels[z0, y0, x0] = component_id
        while stack:
            z, y, x = stack.pop()
            for dz, dy, dx in neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx
                if nz < 0 or ny < 0 or nx < 0:
                    continue
                if nz >= shape[0] or ny >= shape[1] or nx >= shape[2]:
                    continue
                if not binary[nz, ny, nx] or labels[nz, ny, nx] != 0:
                    continue
                labels[nz, ny, nx] = component_id
                stack.append((nz, ny, nx))
    return labels, int(component_id)


def select_component(
    *,
    teacher_mask: np.ndarray,
    prob: np.ndarray,
    seed_thresh: float,
    seed_power: float,
    min_component_voxels: int,
    challenge_density_ratio: float = 1.20,
    challenge_min_selected_fraction: float = 0.02,
    challenge_min_voxels: int = 512,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labeled, count = label_components_6(np.asarray(teacher_mask, dtype=bool))
    rows: List[Dict[str, Any]] = []
    eps = 1e-6
    for component_id in range(1, int(count) + 1):
        mask = labeled == component_id
        voxels = int(mask.sum())
        if voxels < int(min_component_voxels):
            continue
        values = np.asarray(prob[mask], dtype=np.float32)
        expected_mass = float(values.sum())
        seed_fraction = float(np.mean(values >= float(seed_thresh))) if values.size else 0.0
        coverage = float(values.mean()) if values.size else 0.0
        score = float(np.log(expected_mass + eps) + float(seed_power) * np.log(seed_fraction + eps))
        rows.append(
            {
                "component_id": int(component_id),
                "voxels": voxels,
                "score": score,
                "student_score": score,
                "density_evidence": coverage,
                "expected_mass": expected_mass,
                "seed_fraction": seed_fraction,
                "coverage": coverage,
            }
        )
    if not rows:
        return np.zeros_like(teacher_mask, dtype=bool), {
            "component_count": int(count),
            "selected_component": None,
            "selected_components": [],
            "component_rows": [],
        }

    selected_row = max(rows, key=lambda row: float(row["student_score"]))
    selected_density = float(selected_row.get("density_evidence", selected_row.get("coverage", 0.0)))
    selected_voxels = int(selected_row["voxels"])
    min_voxels = max(
        int(min_component_voxels),
        int(challenge_min_voxels),
        int(np.ceil(float(challenge_min_selected_fraction) * float(selected_voxels))),
    )
    challengers = [
        row
        for row in rows
        if int(row["component_id"]) != int(selected_row["component_id"])
        and int(row["voxels"]) >= int(min_voxels)
        and float(row.get("density_evidence", row.get("coverage", 0.0)))
        >= float(challenge_density_ratio) * max(float(selected_density), eps)
    ]
    if challengers:
        selected_row = max(
            challengers,
            key=lambda row: (
                float(row.get("density_evidence", row.get("coverage", 0.0))),
                float(row["student_score"]),
            ),
        )
    selected = labeled == int(selected_row["component_id"])
    return selected, {
        "component_count": int(count),
        "selected_component": selected_row,
        "selected_components": [selected_row],
        "component_rows": sorted(rows, key=lambda row: float(row["score"]), reverse=True),
    }
