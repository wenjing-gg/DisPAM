<div align="center">

# DisPAM

### From One Slice to the Whole Volume
**Single-Slice Propagation Facilitates Weakly Supervised 3D Medical Image Segmentation**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

[Setup & Usage](docs/USAGE.md) · [Configuration](configs/hyperparameters.json) · [Tests](tests/)

<img src="assets/framework.png" alt="DisPAM framework" width="100%">

</div>

DisPAM learns prompt-free 3D segmentation from **one annotated axial slice per
present target**. It connects anatomical slice selection, volumetric teacher
supervision, and heterogeneous student refinement.

| Stage | Role |
| :--- | :--- |
| **Anatomy-Agent Selection** | PASE ranks candidates; a multimodal agent assesses each candidate once. |
| **Sparse-Prompt Dual Distillation** | Frozen PAM propagation supplies complementary Hard and Core supervision. |
| **Consensus-Divergence Fine-Tune** | Cross-model spatial relationships refine a four-student bank. |

## Quick start

```bash
git clone https://github.com/wenjing-gg/DisPAM.git
cd DisPAM
conda env create -f environment.yml
conda activate dispam
```

Follow the [end-to-end guide](docs/USAGE.md) to prepare data and PAM weights,
build training caches, train the students, and evaluate the final bank.

## Data and evaluation

The revised protocol uses fixed, **disjoint train / validation / test** case
collections. Training updates use `train`; checkpoint selection uses `val`;
final performance is computed on `test`.

| Dataset | Train | Validation | Test | Targets |
| :--- | ---: | ---: | ---: | :--- |
| KiTS19 | 161 | 18 | 20 | Kidney, tumor |
| BraTS18 | 205 | 23 | 57 | WT, TC, ET |
| LiTS | 94 | 10 | 27 | Liver, tumor |

Use the same [split files](configs/splits/) for every method and random seed.
This protocol requires fresh checkpoints; evaluation checks their recorded
training and selection splits. See [protocol details](docs/USAGE.md#data-protocol)
for migration from the earlier two-way release.

## Single-pass agent

Candidates follow descending PASE order. Each candidate receives **one binary
visibility decision** with `temperature=0` and `top_p=1`; the first affirmative
candidate is selected. API and replay examples are in the
[usage guide](docs/USAGE.md#guiding-slice-selection).

## Tests

```bash
python -m unittest discover -s tests -v
```

The CPU tests cover data separation, checkpoint provenance, single-pass agent
behavior, and a small training-to-test integration run.

## Acknowledgements and license

DisPAM builds on PAM, MONAI/VISTA3D, Primus, SegMamba, U-RWKV, timm, and
dynamic-network-architectures. See [third-party notices](THIRD_PARTY_NOTICES.md)
and the [Apache-2.0 license](LICENSE).
