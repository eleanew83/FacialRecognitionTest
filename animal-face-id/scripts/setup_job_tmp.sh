#!/bin/bash
# Source after activating Python. Multiprocessing sockets need a short TMPDIR;
# experiment directories can exceed AF_UNIX's path-length limit.
MACAQUE_JOB_TMPDIR=$(mktemp -d "/tmp/mq.${SLURM_JOB_ID:-local}.XXXXXX")
export TMPDIR="$MACAQUE_JOB_TMPDIR"
trap 'rm -rf -- "$MACAQUE_JOB_TMPDIR"' EXIT

# Exercise the exact descriptor-sharing path used by PyTorch DataLoader workers.
# Fail before training if the local IPC setup is unusable.
python - <<'PY'
import os
from multiprocessing.reduction import DupFd

original = os.open(os.devnull, os.O_RDONLY)
try:
    duplicate = DupFd(original).detach()
    os.close(duplicate)
finally:
    os.close(original)
print(f"Multiprocessing IPC check passed; TMPDIR={os.environ['TMPDIR']}", flush=True)
PY
