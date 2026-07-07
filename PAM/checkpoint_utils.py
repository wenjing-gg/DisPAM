#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _clear_torch_modules() -> None:
    for name in list(sys.modules.keys()):
        if name == "torch" or name.startswith("torch."):
            sys.modules.pop(name, None)


def _load_local_itt_stub() -> bool:
    repo_root = Path(__file__).resolve().parents[1]
    stub_dir = repo_root / "tools" / "torch_itt_stub"
    stub_src = stub_dir / "libittnotify_stub.c"
    stub_so = stub_dir / "libittnotify.so"
    if not stub_src.exists():
        return False
    if not stub_so.exists() or stub_src.stat().st_mtime > stub_so.stat().st_mtime:
        stub_dir.mkdir(parents=True, exist_ok=True)
        cc_bin = os.environ.get("CC_BIN", "gcc")
        try:
            subprocess.run(
                [cc_bin, "-shared", "-fPIC", "-O2", str(stub_src), "-o", str(stub_so)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False
    try:
        ctypes.CDLL(str(stub_so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except Exception:
        return False
    return True


def _import_torch_modules():
    try:
        torch_mod = importlib.import_module("torch")
        nn_mod = importlib.import_module("torch.nn")
        return torch_mod, nn_mod
    except ImportError as exc:
        if "iJIT_" not in str(exc):
            raise
        _clear_torch_modules()
        if not _load_local_itt_stub():
            raise
        torch_mod = importlib.import_module("torch")
        nn_mod = importlib.import_module("torch.nn")
        return torch_mod, nn_mod


torch, nn = _import_torch_modules()


def prepare_pam_imports(pam_root: str | Path) -> None:
    pam_root_path = Path(pam_root).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    for path in (repo_root, pam_root_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    dna = importlib.import_module("dynamic_network_architectures_bak")
    sys.modules["dynamic_network_architectures"] = dna
    for submodule in ("architectures", "building_blocks", "initialization"):
        module = importlib.import_module(f"dynamic_network_architectures_bak.{submodule}")
        sys.modules[f"dynamic_network_architectures.{submodule}"] = module


def import_pam_model_module(pam_root: str | Path):
    prepare_pam_imports(pam_root)
    return importlib.import_module("model.PAM")


def build_base_pam_model(pam_root: str | Path) -> nn.Module:
    pam_mod = import_pam_model_module(pam_root)
    return pam_mod.PAM(
        conv_dim=2,
        input_channels=1,
        n_stages=6,
        max_channels=512,
        num_classes=2,
        deep_supervision=True,
        n_attn_stage=4,
        from_scratch_ratio=0.0,
    )


def build_base_box2seg_model(pam_root: str | Path) -> nn.Module:
    pam_mod = import_pam_model_module(pam_root)
    args = argparse.Namespace(
        conv_dim=2,
        input_channels=1,
        n_stages=6,
        max_channels=512,
        num_classes=2,
        deep_supervision=True,
    )
    return pam_mod.Box2SegNet(args)


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        state_dict = torch.load(str(checkpoint), map_location=map_location, weights_only=True)
    except TypeError:
        state_dict = torch.load(str(checkpoint), map_location=map_location)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)}")
    return dict(state_dict)


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    tag: str = "model",
) -> Dict[str, Any]:
    del tag
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location=map_location)
    model_state = model.state_dict()
    loaded = 0
    for key, value in state_dict.items():
        normalized_key = str(key).replace("module.", "")
        if normalized_key in model_state:
            model_state[normalized_key] = value
            loaded += 1
    load_message = model.load_state_dict(model_state)
    return {
        "loaded": int(loaded),
        "model_param_count": int(len(model_state)),
        "load_message": str(load_message),
    }
