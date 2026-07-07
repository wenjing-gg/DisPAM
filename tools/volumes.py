from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import nibabel as nib

    HAS_NIB = True
except Exception:
    nib = None
    HAS_NIB = False


def _split_nrrd_header_payload(blob: bytes) -> Tuple[str, bytes]:
    pos = blob.find(b"\r\n\r\n")
    sep_len = 4
    if pos < 0:
        pos = blob.find(b"\n\n")
        sep_len = 2
    if pos < 0:
        raise ValueError("Invalid NRRD: cannot find header terminator.")
    header = blob[:pos].decode("ascii", errors="strict")
    payload = blob[pos + sep_len :]
    return header, payload


def _parse_nrrd_header(header: str) -> Dict[str, str]:
    lines = header.splitlines()
    if len(lines) == 0 or not lines[0].startswith("NRRD"):
        raise ValueError("Invalid NRRD header magic.")
    meta: Dict[str, str] = {}
    for ln in lines[1:]:
        s = ln.strip()
        if (not s) or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        meta[k.strip().lower()] = v.strip()
    return meta


def _nrrd_dtype(type_name: str, endian: str) -> np.dtype:
    t = type_name.strip().lower()
    mapping = {
        "signed char": np.int8,
        "int8": np.int8,
        "int8_t": np.int8,
        "uchar": np.uint8,
        "unsigned char": np.uint8,
        "uint8": np.uint8,
        "uint8_t": np.uint8,
        "short": np.int16,
        "short int": np.int16,
        "signed short": np.int16,
        "signed short int": np.int16,
        "int16": np.int16,
        "int16_t": np.int16,
        "ushort": np.uint16,
        "unsigned short": np.uint16,
        "unsigned short int": np.uint16,
        "uint16": np.uint16,
        "uint16_t": np.uint16,
        "int": np.int32,
        "signed int": np.int32,
        "int32": np.int32,
        "int32_t": np.int32,
        "uint": np.uint32,
        "unsigned int": np.uint32,
        "uint32": np.uint32,
        "uint32_t": np.uint32,
        "float": np.float32,
        "double": np.float64,
    }
    if t not in mapping:
        raise ValueError(f"Unsupported NRRD type: {type_name}")
    dt = np.dtype(mapping[t])
    if dt.itemsize > 1:
        e = endian.strip().lower()
        if e == "little":
            dt = dt.newbyteorder("<")
        elif e == "big":
            dt = dt.newbyteorder(">")
    return dt


def _infer_depth_axis(meta: Dict[str, str], sizes: List[int]) -> int:
    sd = meta.get("space directions", "")
    if sd:
        tokens = re.findall(r"\([^\)]*\)|none", sd, flags=re.IGNORECASE)
        if len(tokens) == len(sizes):
            spacings = []
            for tok in tokens:
                if tok.lower() == "none":
                    spacings.append(-1.0)
                else:
                    vals = [float(x.strip()) for x in tok.strip()[1:-1].split(",")]
                    spacings.append(float(np.sqrt(np.sum(np.square(vals)))))
            if max(spacings) > 0:
                return int(np.argmax(np.asarray(spacings)))
    return int(np.argmin(np.asarray(sizes)))


def load_nrrd_volume(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    header, payload = _split_nrrd_header_payload(blob)
    meta = _parse_nrrd_header(header)
    if int(meta.get("dimension", "0")) != 3:
        raise ValueError(f"Only 3D NRRD is supported, got dimension={meta.get('dimension')}")
    sizes = [int(x) for x in meta["sizes"].split()]
    if len(sizes) != 3:
        raise ValueError(f"Invalid NRRD sizes: {meta.get('sizes')}")
    encoding = meta.get("encoding", "raw").strip().lower()
    if encoding in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    elif encoding != "raw":
        raise ValueError(f"Unsupported NRRD encoding: {encoding} (only raw/gzip supported)")

    dt = _nrrd_dtype(meta["type"], meta.get("endian", "little"))
    total = int(np.prod(np.asarray(sizes, dtype=np.int64)))
    arr = np.frombuffer(payload, dtype=dt, count=total)
    if arr.size != total:
        raise ValueError(
            f"NRRD payload size mismatch: expected {total} elements, got {arr.size} ({path})"
        )
    arr = arr.reshape(tuple(sizes), order="F")
    depth_axis = _infer_depth_axis(meta, sizes)
    arr = np.moveaxis(arr, depth_axis, 0)
    return np.asarray(arr, dtype=np.float32)


def infer_nifti_depth_axis(shape: Tuple[int, int, int]) -> int:
    dims = tuple(int(v) for v in shape)
    equal_pairs = [
        (pair[0], pair[1], pair[2])
        for pair in ((0, 1, 2), (0, 2, 1), (1, 2, 0))
        if dims[pair[0]] == dims[pair[1]]
    ]
    if equal_pairs:
        inplane_a, inplane_b, depth_axis = max(
            equal_pairs,
            key=lambda item: (dims[item[0]] * dims[item[1]], -dims[item[2]]),
        )
        del inplane_a, inplane_b
        return int(depth_axis)
    return 0


def load_volume_any(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    name = path.name.lower()
    if name.endswith(".npy"):
        arr = np.load(str(path))
    elif name.endswith(".nrrd") or name.endswith(".nhdr"):
        arr = load_nrrd_volume(path)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        if not HAS_NIB:
            raise RuntimeError("nibabel is not installed; cannot read .nii/.nii.gz.")
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"NIfTI must be 3D, got shape={arr.shape}")
        depth_axis = infer_nifti_depth_axis(tuple(int(v) for v in arr.shape))
        arr = np.moveaxis(arr, depth_axis, 0)
    else:
        from tools.pam import load_volume

        arr = load_volume(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Volume must be 3D [D,H,W], got shape={arr.shape}")
    return arr
