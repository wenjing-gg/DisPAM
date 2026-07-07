#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    default_root: Path
    default_split_config: Optional[Path]
    case_glob: str
    image_filename: str
    label_filename: str
    pam_modality: str
    excluded_case_ids: frozenset[str]
    default_images_dir: Optional[Path] = None
    default_labels_dir: Optional[Path] = None
    case_id_regex: Optional[str] = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_root: Path
    default_split_config: Optional[Path]
    images_dir: Path
    labels_dir: Path
    case_glob: str
    image_filename: str
    label_filename: str
    pam_modality: str
    excluded_case_ids: frozenset[str]
    case_id_regex: Optional[str] = None


DEFAULT_DATASET_NAME = "kits19"
DEFAULT_DATA_ROOT = Path(os.environ.get("DISPAM_DATA_ROOT", "data"))
DATASET_NAME_ALIASES: Dict[str, str] = {
    "kits": "kits19",
    "kits19": "kits19",
    "brats": "brats18_flair",
    "brats18": "brats18_flair",
    "brats18_flair": "brats18_flair",
    "lits": "lits",
    "lits17": "lits",
    "lits2017": "lits",
}


DATASET_DEFINITIONS: Dict[str, DatasetDefinition] = {
    "kits19": DatasetDefinition(
        name="kits19",
        default_root=DEFAULT_DATA_ROOT / "kits19",
        default_split_config=DEFAULT_DATA_ROOT / "kits19" / "split.json",
        case_glob="case_*/imaging.nii",
        image_filename="imaging.nii",
        label_filename="segmentation.nii",
        pam_modality="CT",
        excluded_case_ids=frozenset(
            {
                "case_00001",  # Local imaging.nii is truncated/corrupted.
                "case_00046",  # Local imaging.nii is truncated/corrupted.
                "case_00141",  # Local imaging.nii is truncated/corrupted.
                "case_00142",  # Local imaging.nii is truncated/corrupted.
            }
        ),
    ),
    "brats18_flair": DatasetDefinition(
        name="brats18_flair",
        default_root=DEFAULT_DATA_ROOT / "brats18_flair",
        default_split_config=DEFAULT_DATA_ROOT / "brats18_flair" / "split.json",
        case_glob="*/*/*_flair.nii",
        image_filename="{case_id}_flair.nii",
        label_filename="{case_id}_seg.nii",
        pam_modality="MR",
        excluded_case_ids=frozenset(),
    ),
    "lits": DatasetDefinition(
        name="lits",
        default_root=DEFAULT_DATA_ROOT / "lits",
        default_split_config=DEFAULT_DATA_ROOT / "lits" / "split.json",
        default_images_dir=Path("ct"),
        default_labels_dir=Path("seg"),
        case_glob="volume-*.nii",
        image_filename="volume-{case_id}.nii",
        label_filename="segmentation-{case_id}.nii",
        pam_modality="CT",
        excluded_case_ids=frozenset(),
        case_id_regex=r"^volume-(?P<case_id>.+)\.nii$",
    ),
}


def get_supported_dataset_names() -> List[str]:
    return sorted(DATASET_DEFINITIONS.keys())


def normalize_dataset_name(dataset: str = DEFAULT_DATASET_NAME) -> str:
    dataset_key = str(dataset).strip().lower()
    if dataset_key not in DATASET_NAME_ALIASES:
        raise ValueError(
            f"Unsupported dataset='{dataset}'. Expected one of {sorted(DATASET_NAME_ALIASES)}."
        )
    canonical_name = DATASET_NAME_ALIASES[dataset_key]
    if canonical_name not in DATASET_DEFINITIONS:
        raise ValueError(
            f"Dataset alias resolved to unsupported canonical dataset='{canonical_name}'."
        )
    return canonical_name


def get_dataset_definition(dataset: str = DEFAULT_DATASET_NAME) -> DatasetDefinition:
    return DATASET_DEFINITIONS[normalize_dataset_name(dataset)]


def resolve_split_config_path(
    dataset: str = DEFAULT_DATASET_NAME,
    run_config: Optional[str] = None,
) -> Optional[Path]:
    if run_config is not None:
        return Path(run_config)
    definition = get_dataset_definition(dataset)
    return None if definition.default_split_config is None else Path(definition.default_split_config)


def _normalize_case_id_entry(value: str) -> str:
    text = str(value).strip().replace("\\", "/").rstrip("/")
    if not text:
        return ""
    return Path(text).name


def load_split_case_ids(
    split_config_path: Path,
    split: str,
) -> List[str]:
    path = Path(split_config_path)
    split_key = str(split).strip().lower()
    if path.suffix.lower() == ".txt":
        case_ids = [_normalize_case_id_entry(line) for line in path.read_text().splitlines()]
        return [case_id for case_id in case_ids if case_id]

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported split config payload type in {path}: {type(payload).__name__}")

    candidate_keys: List[str] = []
    if split_key == "train":
        candidate_keys.extend(["train_case_ids", "train"])
    elif split_key == "val":
        candidate_keys.extend(["val_case_ids", "val", "test_case_ids", "test"])
    else:
        candidate_keys.extend([f"{split_key}_case_ids", split_key])

    raw_case_ids: List[str] = []
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list) and value:
            raw_case_ids = [str(item) for item in value]
            break
    if not raw_case_ids:
        raise RuntimeError(
            f"No case ids found for split '{split}' in {path}. Tried keys={candidate_keys}."
        )
    case_ids = [_normalize_case_id_entry(item) for item in raw_case_ids]
    case_ids = [case_id for case_id in case_ids if case_id]
    if not case_ids:
        raise RuntimeError(f"Split '{split}' in {path} resolved to no valid case ids.")
    return case_ids


def resolve_dataset_config(
    dataset: str = DEFAULT_DATASET_NAME,
    dataset_root: Optional[str] = None,
    images_dir: Optional[str] = None,
    labels_dir: Optional[str] = None,
) -> DatasetConfig:
    definition = get_dataset_definition(dataset)

    resolved_root = Path(dataset_root) if dataset_root is not None else Path(definition.default_root)
    default_images_dir = (
        None
        if definition.default_images_dir is None
        else definition.default_images_dir
        if definition.default_images_dir.is_absolute()
        else resolved_root / definition.default_images_dir
    )
    default_labels_dir = (
        None
        if definition.default_labels_dir is None
        else definition.default_labels_dir
        if definition.default_labels_dir.is_absolute()
        else resolved_root / definition.default_labels_dir
    )
    resolved_images_dir = (
        Path(images_dir)
        if images_dir is not None
        else Path(default_images_dir)
        if default_images_dir is not None
        else resolved_root
    )
    resolved_labels_dir = (
        Path(labels_dir)
        if labels_dir is not None
        else Path(default_labels_dir)
        if default_labels_dir is not None
        else resolved_root
    )

    return DatasetConfig(
        name=definition.name,
        dataset_root=resolved_root,
        default_split_config=definition.default_split_config,
        images_dir=resolved_images_dir,
        labels_dir=resolved_labels_dir,
        case_glob=definition.case_glob,
        image_filename=definition.image_filename,
        label_filename=definition.label_filename,
        pam_modality=definition.pam_modality,
        excluded_case_ids=definition.excluded_case_ids,
        case_id_regex=definition.case_id_regex,
    )


def collect_cases(dataset_config: DatasetConfig) -> Tuple[List[Tuple[str, Path, Path]], List[str]]:
    root = Path(dataset_config.dataset_root)
    image_root = Path(dataset_config.images_dir)
    label_root = Path(dataset_config.labels_dir)
    image_files = sorted(image_root.glob(str(dataset_config.case_glob)))

    cases: List[Tuple[str, Path, Path]] = []
    missing: List[str] = []
    case_id_pattern = (
        re.compile(str(dataset_config.case_id_regex)) if dataset_config.case_id_regex else None
    )
    for image_path in image_files:
        if case_id_pattern is not None:
            match = case_id_pattern.match(image_path.name)
            if match is None:
                continue
            case_id = str(match.group("case_id"))
        else:
            case_id = image_path.parent.name
        image_filename = str(dataset_config.image_filename).format(case_id=case_id)
        label_filename = str(dataset_config.label_filename).format(case_id=case_id)
        if image_path.name != image_filename:
            continue
        if case_id_pattern is not None:
            label_path = label_root / label_filename
        else:
            try:
                rel_parent = image_path.parent.relative_to(image_root)
                label_path = label_root / rel_parent / label_filename
            except ValueError:
                label_path = image_path.parent / label_filename
        if case_id in dataset_config.excluded_case_ids:
            continue
        if not label_path.exists():
            try:
                missing.append(str(image_path.relative_to(root)))
            except ValueError:
                missing.append(str(image_path))
            continue
        cases.append((case_id, image_path, label_path))
    return cases, missing
