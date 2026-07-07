# DisPAM

DisPAM is a sparse-prompt distillation workflow for 3D medical image segmentation. It uses PAM as a prompt-driven teacher, builds slice-anchored teacher caches, fine-tunes lightweight student models, and evaluates multi-student probability voting on KiTS19, LiTS, and BraTS18-FLAIR style datasets.

The repository is organized around four stages:

1. Select leakage-free sparse prompt slices.
2. Build PAM teacher caches from selected slice anchors.
3. Train prompt-free student models from the teacher caches.
4. Evaluate one or more student checkpoints with sliding-window inference and voting.

## Repository Layout

```text
.
├── dataset.py                 # Dataset registry and case discovery helpers
├── run_dispam.sh              # Unified CLI wrapper for the main workflow
├── tools/
│   ├── slices.py              # Sparse prompt slice selection and optional PAM evaluation
│   ├── cache.py               # PAM teacher-cache construction
│   ├── train.py               # Student training with component embedding contrast
│   ├── eval.py                # Multi-student sliding-window evaluation
│   ├── pam.py                 # PAM inference utilities used by DisPAM
│   └── profiles.py            # Dataset targets and label profiles
├── StudentBank/               # Student network registry and model definitions
└── PAM/                       # PAM implementation and upstream documentation
```

## Supported Datasets

The dataset registry currently supports:

| Name | Aliases | Expected task targets |
| --- | --- | --- |
| `kits19` | `kits`, `kits19` | `kidney`, `tumor` |
| `lits` | `lits`, `lits17`, `lits2017` | `liver`, `tumor` |
| `brats18_flair` | `brats`, `brats18`, `brats18_flair` | `et`, `tc`, `wt` |

Default dataset roots are resolved under `data/` unless `--dataset-root` or the `DISPAM_DATA_ROOT` environment variable is provided.

## Installation

Create a Python environment with PyTorch and the medical-imaging dependencies used by PAM and the student models:

```bash
conda create -n dispam python=3.10 -y
conda activate dispam

# Install the PyTorch build that matches your CUDA driver.
# Example only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install numpy scipy scikit-image scikit-learn SimpleITK nibabel tqdm einops monai timm dynamic-network-architectures matplotlib opencv-python
```

Some student backbones include optional CUDA extensions. Install any additional project-specific dependencies required by the selected student model in the same environment.

## PAM Checkpoints

PAM checkpoints are not tracked in git because they are large binary artifacts. Download them from the links documented in [PAM/README.md](PAM/README.md), then place the files at:

```text
PAM/tutorials/checkpoint/box2mask.pth
PAM/tutorials/checkpoint/propmask.pth
```

If your checkpoints live elsewhere, pass their paths explicitly:

```bash
--box2mask-ckpt /path/to/box2mask.pth
--propmask-ckpt /path/to/propmask.pth
```

## Data Layout

### KiTS19

```text
data/kits19/
├── split.json
├── case_00000/
│   ├── imaging.nii
│   └── segmentation.nii
└── ...
```

### LiTS

```text
data/lits/
├── split.json
├── ct/
│   └── volume-0.nii
└── seg/
    └── segmentation-0.nii
```

### BraTS18-FLAIR

```text
data/brats18_flair/
├── split.json
├── HGG/
│   └── <case_id>/
│       ├── <case_id>_flair.nii
│       └── <case_id>_seg.nii
└── LGG/
    └── <case_id>/
        ├── <case_id>_flair.nii
        └── <case_id>_seg.nii
```

`split.json` should contain split lists such as:

```json
{
  "train_case_ids": ["case_00000"],
  "val_case_ids": ["case_00001"]
}
```

Text files with one case id per line are also accepted by the split loader.

## Quick Start

Use `run_dispam.sh` as the main entry point:

```bash
chmod +x run_dispam.sh
./run_dispam.sh --help
```

### 1. Select Prompt Slices

```bash
./run_dispam.sh select-slice \
  --dataset kits19 \
  --dataset-root data/kits19 \
  --run-config data/kits19/split.json \
  --split val \
  --target-names kidney tumor \
  --max-cases 20 \
  --eval-pam 1 \
  --pam-device cuda:0 \
  --out outputs/select/kits19_val.json
```

To run slice selection without an API-backed agent filter:

```bash
./run_dispam.sh select-slice \
  --dataset kits19 \
  --dataset-root data/kits19 \
  --run-config data/kits19/split.json \
  --use-agent-selection 0 \
  --eval-pam 0 \
  --out outputs/select/kits19_val.json
```

### 2. Build Teacher Cache

```bash
./run_dispam.sh build-cache \
  --dataset kits19 \
  --dataset-root data/kits19 \
  --run-config data/kits19/split.json \
  --source-slice-cache-root outputs/select \
  --teacher-cache-root outputs/teacher_cache/kits19 \
  --splits train val \
  --device cuda:0 \
  --overwrite
```

### 3. Train Student

```bash
./run_dispam.sh train-student \
  --dataset kits19 \
  --dataset-root data/kits19 \
  --run-config data/kits19/split.json \
  --teacher-cache-root outputs/teacher_cache/kits19 \
  --target-name tumor \
  --student-model segmambav2 \
  --init-checkpoint /path/to/student_init.pth \
  --out-root outputs/students/kits19_tumor \
  --device cuda:0 \
  --epochs 20 \
  --amp
```

### 4. Evaluate Voting

```bash
./run_dispam.sh eval \
  --dataset kits19 \
  --dataset-root data/kits19 \
  --run-config data/kits19/split.json \
  --target tumor \
  --student segmambav2=outputs/students/kits19_tumor/best.pth \
  --device cuda:0 \
  --patch-size 96 96 96 \
  --out-dir outputs/eval/kits19_tumor \
  --save-case-artifacts
```

Repeat `--student model=checkpoint.pth` to evaluate an ensemble with probability voting.

## Outputs

Typical outputs are written under `outputs/`:

| Directory | Contents |
| --- | --- |
| `outputs/select/` | Slice-selection metadata and optional PAM evaluation results |
| `outputs/teacher_cache/` | Teacher probabilities, masks, prompts, and per-case metadata |
| `outputs/students/` | Student checkpoints and training logs |
| `outputs/eval/` | Per-target metrics, summary CSV/JSON files, and optional case artifacts |

These directories are intentionally ignored by git.

## Reproducibility Notes

- Set the random seed with `--seed` for slice selection, cache generation, and training.
- Keep the same `split.json`, PAM checkpoints, student initialization checkpoint, and CUDA/PyTorch environment when comparing runs.
- The repository ignores datasets, checkpoints, teacher caches, and evaluation artifacts by default. Store large assets in external storage or a release artifact system instead of committing them to git.

## Citation

This repository builds on PAM. If you use the PAM teacher, cite the original PAM work:

```bibtex
@article{chen2025pam,
  title={PAM: A propagation-based model for segmenting any 3d objects across multi-modal medical images},
  author={Chen, Zifan and Nan, Xinyu and Li, Jiazheng and Zhao, Jie and Li, Haifeng and Lin, Ziling and Li, Haoshen and Chen, Heyun and Liu, Yiting and Tang, Lei and Li, Zhang and Bin, Dong},
  journal={npj Digital Medicine},
  year={2025},
  volume={8},
  number={753},
  doi={10.1038/s41746-025-02087-y},
  url={https://doi.org/10.1038/s41746-025-02087-y}
}
```

## License

No license file is included in this repository yet. Add a license before public redistribution if needed.

