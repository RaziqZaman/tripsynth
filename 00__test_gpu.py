#!/usr/bin/env python3
"""Quick CUDA/PyTorch smoke test."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this environment.")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return 1

    device = torch.device("cuda")
    print(f"device count: {torch.cuda.device_count()}")
    print(f"device 0: {torch.cuda.get_device_name(0)}")

    a = torch.randn((2048, 2048), device=device)
    b = torch.randn((2048, 2048), device=device)
    c = a @ b
    torch.cuda.synchronize()

    print(f"matmul ok: shape={tuple(c.shape)} mean={c.mean().item():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
