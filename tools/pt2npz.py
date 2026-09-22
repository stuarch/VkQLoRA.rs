#!/usr/bin/env python3
"""Convert a PyTorch checkpoint (.pt/.pth/.bin) to uncompressed .npz.

The .npz side is the interchange format read natively by `qlora-io`
(no pickle, no torch required downstream).

Usage:
    python tools/pt2npz.py model.pt model.npz [--prefix ATTN.] [--dtype float32]

Notes:
  - Only floating and integer tensors are converted (all cast to float32
    by default); complex / quantized / sparse tensors abort with an error.
  - SECURITY: torch.load executes pickle code. Only convert files from
    sources you trust (same rule as torch itself). Prefer
    `weights_only=True` when the checkpoint allows it.
  - Run inside `guix shell -f guix.scm` (provides python-pytorch + numpy).
"""

import argparse
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help=".pt/.pth/.bin checkpoint")
    ap.add_argument("dst", help="output .npz (uncompressed, np.savez)")
    ap.add_argument("--prefix", default="", help="only convert keys starting with this")
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="target dtype (float16 keeps half precision; qlora-io converts to f32)",
    )
    ap.add_argument(
        "--allow-weights-only-failure",
        action="store_true",
        help="fall back to full torch.load if weights_only=True fails (less safe)",
    )
    args = ap.parse_args()

    try:
        obj = torch.load(args.src, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
        if not args.allow_weights_only_failure:
            print(
                f"weights_only load failed ({exc}); refusing unsafe full load. "
                "Re-run with --allow-weights-only-failure only for files you trust.",
                file=sys.stderr,
            )
            return 2
        print("WARNING: falling back to full pickle load (trusted source only)", file=sys.stderr)
        obj = torch.load(args.src, map_location="cpu", weights_only=False)

    # Unwrap common checkpoint envelopes: {"state_dict": ...}, {"model": ...}.
    if isinstance(obj, dict) and not any(isinstance(v, torch.Tensor) for v in obj.values()):
        for key in ("state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        print(f"unsupported checkpoint root: {type(obj)} (need a state_dict)", file=sys.stderr)
        return 2

    out = {}
    skipped = []
    for name, tensor in obj.items():
        if args.prefix and not name.startswith(args.prefix):
            continue
        if not isinstance(tensor, torch.Tensor):
            skipped.append((name, f"not a tensor ({type(tensor)})"))
            continue
        if tensor.is_complex() or tensor.is_quantized or tensor.is_sparse:
            skipped.append((name, f"unsupported tensor kind {tensor.dtype}"))
            continue
        if tensor.is_floating_point() or tensor.dtype in (
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, torch.bool,
        ):
            arr = tensor.detach().to("cpu")
            arr = arr.to(getattr(torch, args.dtype)).numpy()
            # Fortran-order arrays would round-trip fine, but keep C-order
            # for maximal compatibility.
            out[name] = np.ascontiguousarray(arr)
        else:
            skipped.append((name, f"unsupported dtype {tensor.dtype}"))

    if not out:
        print("no convertible tensors found", file=sys.stderr)
        return 2
    # np.savez (NOT savez_compressed): qlora-io reads stored entries only.
    np.savez(args.dst, **out)
    print(f"wrote {args.dst}: {len(out)} arrays")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
