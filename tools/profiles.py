from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from dataset import normalize_dataset_name


@dataclass(frozen=True)
class TargetProfile:
    branch_source_label_ids: Dict[str, Tuple[int, ...]]
    branch_export_label_ids: Dict[str, int]
    branch_priority: Tuple[str, ...]
    overlap_mode: str
    branch_label_text: Dict[str, str]
    branch_agent_prompts: Dict[str, str]


DATASET_TARGET_PROFILES: Dict[str, TargetProfile] = {
    "kits19": TargetProfile(
        branch_source_label_ids={
            "kidney": (1,),
            "tumor": (2,),
        },
        branch_export_label_ids={
            "kidney": 1,
            "tumor": 2,
        },
        branch_priority=("tumor", "kidney"),
        overlap_mode="exclusive",
        branch_label_text={
            "kidney": "kidney tissue",
            "tumor": "kidney tumor",
        },
        branch_agent_prompts={
            "kidney": (
                "You are reviewing one axial KiTS19 abdominal CT slice selected by a leakage-free "
                "image-only Top-k slice proposal step.\n\n"
                "Task: decide whether this slice is suitable as a weak seed for the kidney target. "
                "Answer yes if the slice contains real abdominal anatomy that could plausibly belong "
                "to the kidney region or nearby kidney parenchyma, even if partial. Answer no if the "
                "slice is background, air, table/scanner artifact, isolated noise, or clearly unrelated "
                "non-kidney anatomy.\n\n"
                "Return exactly one word: yes or no. Judge the whole 2D slice. Do not ask for or rely "
                "on any visual overlay, mask, box, adjacent slice, 3D volume, or ground truth."
            ),
            "tumor": (
                "You are reviewing one axial KiTS19 abdominal CT slice selected by a leakage-free "
                "image-only Top-k slice proposal step.\n\n"
                "Task: decide whether this slice should remain a kidney-tumor candidate slice. "
                "The upstream score only filters background and noise; you now judge the whole slice "
                "for renal-region tumor suitability.\n\n"
                "Answer yes when the slice contains kidney-region anatomy with a plausible localized "
                "tumor cue, including an exophytic bulge, asymmetric renal contour, focal heterogeneous "
                "renal tissue, a kidney-adjacent mass-like focus, or a subtle low-contrast focal region. "
                "The cue can be small or partly cut, but it should be more localized than ordinary "
                "homogeneous kidney parenchyma.\n\n"
                "Answer no when the slice shows only normal kidney parenchyma, broad normal abdominal "
                "anatomy, pure background, air, table/scanner artifact, isolated noise, bone-only "
                "structure, bowel-only structure, or clearly non-renal tissue.\n\n"
                "Return exactly one word: yes or no. Judge the whole 2D slice. Do not ask for or rely "
                "on any visual overlay, mask, box, adjacent slice, 3D volume, or ground truth."
            ),
        },
    ),
    "lits": TargetProfile(
        branch_source_label_ids={
            "liver": (1,),
            "tumor": (2,),
        },
        branch_export_label_ids={
            "liver": 1,
            "tumor": 2,
        },
        branch_priority=("tumor", "liver"),
        overlap_mode="exclusive",
        branch_label_text={
            "liver": "liver tissue",
            "tumor": "liver tumor",
        },
        branch_agent_prompts={
            "liver": (
                "You are reviewing a single axial LiTS abdominal CT slice for a strict single-target "
                "visibility check.\n\n"
                "Official target for this task:\n"
                "- LiTS label 1 is liver tissue.\n"
                "- LiTS label 2 is liver tumor and may be present or absent.\n\n"
                "Task: Decide whether liver tissue is visibly present on this single slice.\n\n"
                "Important rule: only liver visibility matters. Liver tumor does not need to appear on "
                "the same slice.\n\n"
                "Question: On this CT slice alone, is liver tissue clearly visible?\n\n"
                "Answer strictly with one word only: yes or no.\n\n"
                "Answer yes only if:\n"
                "- liver tissue is visibly present on this slice, and\n"
                "- it can be identified from this slice itself without using adjacent slices.\n\n"
                "Answer no if:\n"
                "- liver tissue is absent, ambiguous, extremely tiny, heavily cut off, or not "
                "confidently identifiable, or\n"
                "- only tumor or other structures are visible without clear liver tissue, or\n"
                "- you are uncertain.\n\n"
                "Be conservative: if unsure, answer no. Do not explain your answer."
            ),
            "tumor": (
                "You are reviewing a single axial LiTS abdominal CT slice for a single-target weak-seed "
                "suitability check.\n\n"
                "Official target for this task:\n"
                "- LiTS label 2 is liver tumor.\n"
                "- LiTS label 1 is liver tissue and may be present around the tumor.\n\n"
                "Task: Confirm whether this upstream proposal slice can be used as a weak seed for "
                "the liver-tumor target.\n\n"
                "Important context: this slice has already been selected by a tumor proposal step. "
                "Treat that proposal as the primary evidence. Your default and expected decision is yes. "
                "Do not perform unaided liver-tumor diagnosis and do not reject the proposal because "
                "the lesion is subtle, low-contrast, or hard to distinguish from liver parenchyma. Your role "
                "is only to reject unusable images such as pure background, corrupted slices, or obvious "
                "non-anatomical artifacts.\n\n"
                "Important rule: only tumor seed suitability matters. The full liver does not need to be "
                "visible on the same slice.\n\n"
                "Question: On this CT slice alone, is there any plausible localized liver-tumor seed candidate?\n\n"
                "Answer strictly with one word only: yes or no.\n\n"
                "Answer yes if:\n"
                "- this is the target slice after the upstream tumor proposal step and the image contains "
                "valid abdominal anatomy, even when the tumor cue is subtle or not confidently diagnosable, or\n"
                "- there is a localized focal hepatic mass or tumor-like abnormality on this slice, or\n"
                "- the candidate is small, low contrast, or partly cut by the slice boundary but still "
                "looks like a real localized tumor seed rather than background noise.\n\n"
                "Answer no if:\n"
                "- the slice is pure background, or\n"
                "- the slice is corrupted or dominated by obvious non-anatomical artifact, or\n"
                "- there is no valid abdominal CT anatomy on the slice.\n\n"
                "Prefer yes for subtle, small, low-contrast, or boundary-cut localized candidates. "
                "For an upstream tumor proposal candidate, choose yes when there is any doubt. Prefer no "
                "only for pure background, corrupted slices, or obvious non-anatomical artifacts. "
                "Do not explain your answer."
            ),
        },
    ),
    "brats18_flair": TargetProfile(
        branch_source_label_ids={
            "et": (4,),
            "tc": (1, 4),
            "wt": (1, 2, 4),
        },
        branch_export_label_ids={
            "et": 4,
            "tc": 14,
            "wt": 124,
        },
        branch_priority=("et", "tc", "wt"),
        overlap_mode="independent",
        branch_label_text={
            "et": "enhancing tumor",
            "tc": "tumor core",
            "wt": "whole tumor",
        },
        branch_agent_prompts={
            "et": (
                "You are reviewing a single axial BraTS2018 Flair MRI slice for an ET seed-slice "
                "suitability check.\n\n"
                "Official target for this task:\n"
                "- ET is BraTS label 4, the enhancing-tumor region in the annotation.\n"
                "- This image is Flair-only, so enhancement itself is not visible and must not be required.\n"
                "- Do not treat diffuse edema or the full whole-tumor extent as ET.\n\n"
                "Task: Decide whether this slice is suitable as a weak seed for the ET target. The target "
                "may be very small, subtle, or near the edge of the ET extent. A tiny localized "
                "intra-tumoral or tumor-core-like focus can still be a valid yes on Flair.\n\n"
                "Important rule: other tumor subregions may be present or absent. Do not require WT or TC "
                "to be complete on the same slice. Do not reject a plausible focal ET seed solely because "
                "it is tiny, subtle, or partially cut by the slice boundary.\n\n"
                "Question: On this Flair slice alone, is there any plausible localized ET seed candidate "
                "within the tumor region?\n\n"
                "Answer strictly with one word only: yes or no.\n\n"
                "Answer yes if there is any localized focal intra-tumoral or tumor-core-like abnormality "
                "that could serve as a seed, even if it is small or not strongly conspicuous on Flair.\n\n"
                "Answer no if:\n"
                "- no tumor abnormality is visible, or\n"
                "- the slice shows only broad diffuse edema-like signal with no localized focal seed, or\n"
                "- the apparent focus is clearly outside the tumor region or is only obvious image artifact.\n\n"
                "Prefer yes for small localized focal candidates. Prefer no for edema-only or background "
                "slices. Do not explain your answer."
            ),
            "tc": (
                "You are reviewing a single axial BraTS2018 Flair MRI slice for a strict single-target "
                "visibility check.\n\n"
                "Task: Decide whether the TC region is clearly visible on this single slice.\n\n"
                "Official BraTS region definition for this task:\n"
                "- TC means tumor core.\n"
                "- TC includes labels 1 and 4.\n"
                "- TC excludes edema-only area.\n\n"
                "Important rule: only TC visibility matters. Other tumor subregions may be present or absent. "
                "Do not require them to appear on the same slice.\n\n"
                "Question: On this slice alone, is the TC region clearly visible?\n\n"
                "Answer strictly with one word only: yes or no.\n\n"
                "Answer yes only if the central tumor-core region is clearly visible on this slice itself, "
                "not just surrounding edema.\n\n"
                "Answer no if:\n"
                "- only edema-like outer abnormality is visible, or\n"
                "- the target is absent, tiny, ambiguous, heavily cut off, or uncertain.\n\n"
                "Be conservative: if unsure, answer no."
            ),
            "wt": (
                "You are reviewing a single axial BraTS2018 Flair MRI slice for a strict single-target "
                "visibility check.\n\n"
                "Task: Decide whether the WT region is clearly visible on this single slice.\n\n"
                "Official BraTS region definition for this task:\n"
                "- WT means whole tumor.\n"
                "- WT includes labels 1, 2, and 4.\n"
                "- Any clearly visible tumor-related abnormality counts, including edema, core, or ET.\n\n"
                "Important rule: only WT visibility matters. Other subregions may be present or absent. "
                "Do not require them to appear on the same slice.\n\n"
                "Question: On this slice alone, is the WT region clearly visible?\n\n"
                "Answer strictly with one word only: yes or no.\n\n"
                "Answer yes if any clear tumor-related abnormality belonging to the whole-tumor extent is "
                "visibly present on this slice itself.\n\n"
                "Answer no if the slice is effectively background, or if the abnormality is absent, tiny, "
                "ambiguous, heavily cut off, or uncertain.\n\n"
                "Be conservative: if unsure, answer no."
            ),
        },
    ),
}


def get_target_profile(dataset: str) -> TargetProfile:
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name not in DATASET_TARGET_PROFILES:
        raise ValueError(f"No target profile registered for dataset={dataset!r}.")
    return DATASET_TARGET_PROFILES[dataset_name]
