# Third-Party Notices

This repository integrates or adapts the following research software. Original
copyright statements in source files take precedence.

## VISTA3D

`dispam/models/vista3d.py` is derived from MONAI VISTA3D code and retains the
MONAI Consortium Apache-2.0 notice in the source file.

## Primus and PrimusV2

`dispam/models/primusv2.py` and the bundled
`dynamic_network_architectures/` modules derive from MIC-DKFZ software and
retain Apache-2.0 notices where provided.

## U-RWKV

`dispam/models/urwkv.py` contains the binary segmentation architecture used in
the experiments. The upstream U-RWKV repository declares the MIT License.

## SegMambaV2

`dispam/models/segmambav2.py` contains the architecture definition used in the
experiments and is distributed under the confirmed upstream research-software
terms retained by this release.

## PAM

`dispam/pam_model/` contains the PAM network definition needed to load the
reported frozen propagation checkpoint. PAM checkpoints are not redistributed.

## timm and dynamic-network-architectures

Bundled compatibility modules preserve upstream notices. These packages are
also version-pinned in the environment specification.

This notice does not replace upstream licenses or dataset terms.
