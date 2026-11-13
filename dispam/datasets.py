"""Public dataset discovery and explicit split handling.

Paper mapping:
  Experiments / Datasets. KiTS19 and LiTS provide CT volumes; BraTS18 provides
  FLAIR MRI volumes. Labels are used for evaluation and prompt simulation,
  never as dense optimization targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_paths: Tuple[Path, ...]
    label_path: Path


DATASET_ALIASES = {
    "kits": "kits19",
    "kits19": "kits19",
    "brats": "brats18",
    "brats18": "brats18",
    "lits": "lits",
    "lits17": "lits",
}

TARGETS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "kits19": {"kidney": (1,), "tumor": (2,)},
    "brats18": {"wt": (1, 2, 4), "tc": (1, 4), "et": (4,)},
    "lits": {"liver": (1,), "tumor": (2,)},
}


def canonical_dataset_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset {name!r}; expected {sorted(DATASET_ALIASES)}.")
    return DATASET_ALIASES[key]


def target_label_ids(dataset: str, target: str) -> Tuple[int, ...]:
    dataset_name = canonical_dataset_name(dataset)
    target_name = str(target).strip().lower()
    if target_name not in TARGETS[dataset_name]:
        raise ValueError(
            f"Unsupported target {target!r} for {dataset_name}; "
            f"expected {sorted(TARGETS[dataset_name])}."
        )
    return TARGETS[dataset_name][target_name]


def _normalize_case_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/").rstrip("/")
    return Path(text).name if text else ""


def load_splits(path: Path) -> Dict[str, List[str]]:
    """Read disjoint training/test lists and an optional validation list."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Split file must contain a JSON object: {path}")
    splits: Dict[str, List[str]] = {}
    for name in ("train", "val", "test"):
        aliases = [payload[key] for key in (name, f"{name}_case_ids") if key in payload]
        if name == "val" and not aliases:
            splits[name] = []
            continue
        if not aliases or any(not isinstance(ids, list) or (not ids and name != "val") for ids in aliases):
            raise ValueError(f"An explicit, nonempty {name!r} split is required.")
        normalized = [[_normalize_case_id(value) for value in ids] for ids in aliases]
        if any(ids != normalized[0] for ids in normalized[1:]):
            raise ValueError(f"Conflicting case lists for {name!r}.")
        ids = normalized[0]
        if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate case identifiers in {name!r}.")
        for other, other_ids in splits.items():
            overlap = set(ids).intersection(other_ids)
            if overlap:
                raise ValueError(f"{other}/{name} case overlap: {sorted(overlap)[:10]}")
        splits[name] = ids
    return splits


def load_split_case_ids(path: Path, split: str) -> List[str]:
    split_name = str(split).strip().lower()
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}.")
    return load_splits(path)[split_name]


def make_three_way_split(
    development_ids: Sequence[str],
    test_ids: Sequence[str],
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> Dict[str, object]:
    """Hold validation cases out of a development pool without moving test cases."""
    development = sorted(_normalize_case_id(value) for value in development_ids)
    test = [_normalize_case_id(value) for value in test_ids]
    if not 0 <= validation_fraction < 1 or len(development) < 2 or not test:
        raise ValueError("Need a development pool, a test set, and 0 <= val fraction < 1.")
    if any(not value for value in development + test):
        raise ValueError("Case identifiers must not be empty.")
    if len(set(development)) != len(development) or len(set(test)) != len(test):
        raise ValueError("Duplicate case identifiers in the input pools.")
    if set(development).intersection(test):
        raise ValueError("Development and test pools overlap.")
    count = (min(len(development) - 1, max(1, round(len(development) * validation_fraction)))
             if validation_fraction else 0)
    val = set(random.Random(seed).sample(development, count))
    return {
        "schema_version": 2,
        "validation_seed": seed,
        "validation_fraction": validation_fraction,
        "train": [case_id for case_id in development if case_id not in val],
        "val": sorted(val),
        "test": test,
    }


def split_protocol(dataset: str, path: Path) -> Dict[str, object]:
    return {
        "dataset": canonical_dataset_name(dataset),
        **{name: sorted(ids) for name, ids in load_splits(path).items()},
    }


def validate_checkpoint_protocol(
    checkpoint: Dict[str, object],
    protocol: Dict[str, object],
    target: str | None = None,
) -> None:
    metadata = checkpoint.get("metadata", {})
    recorded = metadata.get("data_protocol") if isinstance(metadata, dict) else None
    if recorded is None:
        raise ValueError(
            "Checkpoint has no train/val/test provenance. Supply a checkpoint "
            "with recorded case partitions for refinement or evaluation."
        )
    if recorded != protocol:
        raise ValueError("Checkpoint training/validation/test split differs from this run.")
    if target is not None and metadata.get("target") != target:
        raise ValueError("Checkpoint target differs from the requested target.")


def _discover_kits19(root: Path) -> Iterable[CaseRecord]:
    excluded = {"case_00001", "case_00046", "case_00141", "case_00142"}
    for image_path in sorted(root.glob("case_*/imaging.nii*")):
        case_id = image_path.parent.name
        label_path = image_path.parent / (
            "segmentation.nii.gz" if (image_path.parent / "segmentation.nii.gz").exists()
            else "segmentation.nii"
        )
        if case_id not in excluded and label_path.exists():
            yield CaseRecord(case_id, (image_path,), label_path)


def _discover_brats18(root: Path) -> Iterable[CaseRecord]:
    for flair_path in sorted(root.glob("*/*/*_flair.nii*")):
        case_dir = flair_path.parent
        case_id = case_dir.name
        labels = sorted(case_dir.glob(f"{case_id}_seg.nii*"))
        if labels:
            yield CaseRecord(case_id, (flair_path,), labels[0])


def _discover_lits(root: Path) -> Iterable[CaseRecord]:
    pattern = re.compile(r"^volume-(?P<case_id>.+)\.nii(?:\.gz)?$")
    for image_path in sorted((root / "ct").glob("volume-*.nii*")):
        match = pattern.match(image_path.name)
        if match is None:
            continue
        case_id = match.group("case_id")
        candidates = [
            root / "seg" / f"segmentation-{case_id}.nii",
            root / "seg" / f"segmentation-{case_id}.nii.gz",
        ]
        label_path = next((path for path in candidates if path.exists()), None)
        if label_path is not None:
            yield CaseRecord(case_id, (image_path,), label_path)


def discover_cases(dataset: str, dataset_root: Path) -> Dict[str, CaseRecord]:
    dataset_name = canonical_dataset_name(dataset)
    root = Path(dataset_root)
    if dataset_name == "kits19":
        records = _discover_kits19(root)
    elif dataset_name == "brats18":
        records = _discover_brats18(root)
    else:
        records = _discover_lits(root)
    output = {record.case_id: record for record in records}
    if not output:
        raise RuntimeError(f"No complete {dataset_name} cases found under {root}.")
    return output


def cases_for_split(
    dataset: str,
    dataset_root: Path,
    split_file: Path,
    split: str,
) -> List[CaseRecord]:
    case_ids = load_split_case_ids(split_file, split)
    lookup = discover_cases(dataset, dataset_root)
    missing = [case_id for case_id in case_ids if case_id not in lookup]
    if missing:
        raise RuntimeError(
            f"{len(missing)} split-selected cases are incomplete or unknown: {missing[:10]}"
        )
    return [lookup[case_id] for case_id in case_ids]


def serialize_case(record: CaseRecord) -> Dict[str, object]:
    return {
        "case_id": record.case_id,
        "images": [str(path) for path in record.image_paths],
        "label": str(record.label_path),
    }


def validate_shape_count(shapes: Sequence[Sequence[int]], case_id: str) -> None:
    normalized = [tuple(int(value) for value in shape) for shape in shapes]
    if not normalized or any(shape != normalized[0] for shape in normalized[1:]):
        raise RuntimeError(f"{case_id}: inconsistent image/label shapes: {normalized}")
