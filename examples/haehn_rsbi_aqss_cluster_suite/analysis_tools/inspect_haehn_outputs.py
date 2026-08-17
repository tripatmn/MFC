#!/usr/bin/env python3
"""Inventory Haehn/RSBI post-process outputs without assuming one format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TIMESTEP_RE = re.compile(r"(?:t[_-]?step[_-]?|step[_-]?|time[_-]?)(\d+)|\b(\d{4,})\b", re.IGNORECASE)


def try_import_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except Exception:
        return None


def timestep_from_name(path: Path) -> int | None:
    for match in TIMESTEP_RE.finditer(path.name):
        for group in match.groups():
            if group is not None:
                try:
                    return int(group)
                except ValueError:
                    pass
    return None


def hdf5_inventory(path: Path, h5py_module) -> dict:
    info = {"readable_hdf5": False, "datasets": [], "groups": []}
    if h5py_module is None:
        return info

    try:
        with h5py_module.File(path, "r") as handle:
            info["readable_hdf5"] = True

            def visit(name, obj):
                kind = obj.__class__.__name__
                if kind == "Dataset":
                    info["datasets"].append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
                else:
                    info["groups"].append(name)

            handle.visititems(visit)
    except Exception as exc:
        info["hdf5_error"] = str(exc)
    return info


def candidate_roots(run_dir: Path) -> list[Path]:
    roots = []
    for name in ("D", "p_all", "post_process", "post_process_output"):
        path = run_dir / name
        if path.exists():
            roots.append(path)
    if not roots:
        roots = [run_dir]
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default="analysis_out", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else run_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    h5py_module = try_import_h5py()
    roots = candidate_roots(run_dir)
    files = []

    for root in roots:
        candidates = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in sorted(candidates):
            rel = path.relative_to(run_dir) if path.is_relative_to(run_dir) else path
            record = {
                "path": str(rel),
                "size_bytes": path.stat().st_size,
                "timestep_guess": timestep_from_name(path),
                "suffix": path.suffix,
            }
            if path.suffix.lower() in {".h5", ".hdf5", ".hdf"} or h5py_module is not None:
                record.update(hdf5_inventory(path, h5py_module))
            files.append(record)

    timesteps = sorted({item["timestep_guess"] for item in files if item["timestep_guess"] is not None})
    inventory = {
        "run_dir": str(run_dir),
        "roots": [str(root.relative_to(run_dir) if root.is_relative_to(run_dir) else root) for root in roots],
        "h5py_available": h5py_module is not None,
        "file_count": len(files),
        "timesteps": timesteps,
        "files": files,
    }

    text_lines = [
        "Haehn/RSBI Output Inventory",
        f"run_dir: {run_dir}",
        f"h5py_available: {h5py_module is not None}",
        f"roots: {', '.join(inventory['roots'])}",
        f"file_count: {len(files)}",
        f"timesteps: {timesteps if timesteps else 'not detected'}",
        "",
        "Files:",
    ]
    for item in files:
        text_lines.append(f"- {item['path']} ({item['size_bytes']} bytes, timestep={item['timestep_guess']})")
        if item.get("readable_hdf5"):
            datasets = item.get("datasets", [])
            text_lines.append(f"  HDF5 datasets: {len(datasets)}")
            for dataset in datasets[:50]:
                text_lines.append(f"    - {dataset['name']} shape={dataset['shape']} dtype={dataset['dtype']}")
            if len(datasets) > 50:
                text_lines.append(f"    - ... {len(datasets) - 50} more datasets")
        elif item.get("hdf5_error"):
            text_lines.append(f"  HDF5: not readable ({item['hdf5_error']})")

    (output_dir / "output_inventory.txt").write_text("\n".join(text_lines) + "\n")
    (output_dir / "output_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    print(f"Wrote inventory to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
