#!/usr/bin/env python3
"""Convert an .npz of trained adapters back to a PyTorch state_dict (.pt).

This is the way OUT of the qlora pipeline: finetuned LoRA matrices
(`qlora_wgpu.PagedAdam` products, or anything saved with np.savez) become
a `torch.save`-compatible file for merge/eval in the PyTorch/PEFT world.

Usage:
    python tools/npz2pt.py adapters.npz adapters.pt [--dtype float32]

    # then in torch:
    #   sd = torch.load("adapters.pt", weights_only=True)
"""

import argparse

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="input .npz")
    ap.add_argument("dst", help="output .pt (torch.save of a dict)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    npz = np.load(args.src)
    dtype = getattr(torch, args.dtype)
    state = {name: torch.from_numpy(np.ascontiguousarray(npz[name])).to(dtype) for name in npz.files}
    torch.save(state, args.dst)
    print(f"wrote {args.dst}: {len(state)} tensors ({args.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
