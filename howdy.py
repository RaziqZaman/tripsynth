#!/usr/bin/env python3
"""Simple CUDA/GPU diagnostics."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return True, out
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        return False, str(exc)


class Tee:
    """Write stdout to the terminal and a log file at the same time."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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
    device = "cuda" if available else "cpu"
    print(f"running tensor multiplication on: {device}")

    if available:
        count = torch.cuda.device_count()
        print(f"CUDA device count: {count}")
        for idx in range(count):
            print(f"  - [{idx}] {torch.cuda.get_device_name(idx)}")
    else:
        print("CUDA is not available to PyTorch.")

    try:
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        result = left @ right
        print("tensor matmul result:")
        print(result)
    except Exception as exc:
        print(f"tensor multiplication failed: {exc}")


def main() -> int:
    with open("howdy.txt", "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            print("Howdy! Checking CUDA access...\n")
            check_nvidia_smi()
            check_torch_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
