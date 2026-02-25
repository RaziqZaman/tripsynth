#!/usr/bin/env python3
"""Simple CUDA/GPU diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import sys


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return True, out
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        return False, str(exc)


def check_nvidia_smi() -> None:
    print("== nvidia-smi check ==")
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found on PATH.")
        return

    ok, out = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if ok:
        print("GPU detected by NVIDIA driver:")
        for line in out.splitlines():
            print(f"  - {line}")
    else:
        print(f"nvidia-smi failed: {out}")


def check_torch_cuda() -> None:
    print("\n== PyTorch CUDA check ==")
    try:
        import torch
    except Exception as exc:
        print(f"PyTorch not available: {exc}")
        return

    print(f"torch version: {torch.__version__}")
    print(f"torch built with CUDA: {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")

    if available:
        count = torch.cuda.device_count()
        print(f"CUDA device count: {count}")
        for idx in range(count):
            print(f"  - [{idx}] {torch.cuda.get_device_name(idx)}")
    else:
        print("CUDA is not available to PyTorch.")


def main() -> int:
    print("Howdy! Checking CUDA access...\n")
    check_nvidia_smi()
    check_torch_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
