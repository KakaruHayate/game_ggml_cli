#!/usr/bin/env python3
"""Inspect a GAME GGUF file produced by convert_pt_to_gguf.py."""

from __future__ import annotations

import argparse
import pathlib
import sys

try:
    import gguf  # type: ignore
except ImportError:
    sys.stderr.write("error: the 'gguf' Python package is required\n")
    raise


def _format_shape(dims) -> str:
    return "[" + ", ".join(str(int(d)) for d in dims) + "]"


def inspect(path: pathlib.Path, *, tensor_limit: int) -> None:
    reader = gguf.GGUFReader(str(path))

    print(f"== {path.name} ==")
    print(f"  tensors: {len(reader.tensors)}")
    print(f"  kv pairs: {len(reader.fields)}")
    print()

    # Metadata
    print("-- Metadata --")
    for name, field in sorted(reader.fields.items()):
        # `field.contents()` returns scalars or strings unwrapped.
        try:
            val = field.contents()
        except Exception:
            val = f"<{len(field.parts)} parts>"
        if isinstance(val, str) and len(val) > 120:
            val = val[:117] + "..."
        print(f"  {name} = {val!r}")
    print()

    # Tensors
    print(f"-- Tensors (first {tensor_limit} of {len(reader.tensors)}) --")
    total_params = 0
    for t in reader.tensors:
        total_params += int(t.n_elements)
    for t in reader.tensors[:tensor_limit]:
        print(f"  {t.name:<60s} {t.tensor_type.name:<6s} {_format_shape(t.shape)}")
    if len(reader.tensors) > tensor_limit:
        print(f"  ... and {len(reader.tensors) - tensor_limit} more")
    print()
    print(f"  total params: {total_params:,}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", type=pathlib.Path)
    p.add_argument("-n", "--tensor-limit", type=int, default=20,
                   help="How many tensors to print before truncating (default: 20)")
    args = p.parse_args(argv)
    inspect(args.path, tensor_limit=args.tensor_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
