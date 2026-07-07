#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"

if [[ -z "${ITT_LIB:-}" ]]; then
  PY_PREFIX="$(${PY} -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
  CANDIDATE_ITT_LIB="${PY_PREFIX}/lib/libittnotify.so"
  if [[ -n "${PY_PREFIX}" && -f "${CANDIDATE_ITT_LIB}" ]]; then
    ITT_LIB="${CANDIDATE_ITT_LIB}"
  fi
fi
if [[ -n "${ITT_LIB:-}" && -f "${ITT_LIB}" ]]; then
  export LD_PRELOAD="${ITT_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
if [[ -n "${TOOTH_SITE:-}" ]]; then
  export PYTHONPATH="${TOOTH_SITE}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

usage() {
  cat <<'USAGE'
Usage:
  ./run_dispam.sh select-slice [args...]
  ./run_dispam.sh build-cache [args...]
  ./run_dispam.sh train-student [args...]
  ./run_dispam.sh eval [args...]

Modes:
  select-slice  Run leakage-free sparse-prompt slice selection and optional PAM evaluation.
  build-cache   Build a box-prompt Teacher Cache from existing slice-anchor metadata.
  train-student Fine-tune one student with foreground-aware random crops.
  eval          Evaluate multi-student voting with sliding-window inference.

Common paths can be passed directly to the underlying Python tools, for example:
  --dataset kits19 --dataset-root <dataset-root> --run-config <split.json>
USAGE
}

mode="${1:-}"
if [[ -z "${mode}" || "${mode}" == "-h" || "${mode}" == "--help" ]]; then
  usage
  exit 0
fi
shift

case "${mode}" in
  select-slice)
    exec "${PY}" "${ROOT}/tools/slices.py" "$@"
    ;;
  build-cache)
    exec "${PY}" "${ROOT}/tools/cache.py" "$@"
    ;;
  train-student)
    exec "${PY}" "${ROOT}/tools/train.py" "$@"
    ;;
  eval)
    exec "${PY}" "${ROOT}/tools/eval.py" "$@"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    usage >&2
    exit 2
    ;;
esac
