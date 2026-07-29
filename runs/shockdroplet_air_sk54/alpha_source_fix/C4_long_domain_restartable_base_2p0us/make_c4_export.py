#!/usr/bin/env python3
"""Compact exporter for C4 Mach-3 SK54 shock-droplet saves.

The exporter keeps the old raw ASCII ``D/`` layout support, and also supports
parallel-I/O MFC restart saves named ``restart_data/lustre_<save>.dat``.  For
parallel-I/O saves it deliberately does *not* parse the Lustre restart binary.
Instead, one save at a time is read by MFC ``post_process`` into its documented
``format = 2`` binary database, then converted to the temporary ASCII ``D/``
field layout used by the existing lightweight analyzers.  Temporary files are
removed after each save unless ``--keep-temp`` is requested.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-c4-export")

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
RUN_ROOT = REPO / "runs/shockdroplet_air_sk54/alpha_source_fix"
DEFAULT_CASE = SCRIPT.parent
DEFAULT_OUT = SCRIPT.parent / "c4_export"
C3_HELPER = RUN_ROOT / "C3_vs_chemistry_ON_OFF_0_to_2us/make_c3_on_off_comparison.py"
T_SAVE = 5.0e-8
NAME_LEN = 50
GAS_MASS_FLOOR = 1.0e-8
FRAME_TIMES_US = [0.50, 0.85, 1.00, 1.50, 1.90, 1.95, 2.00, 3.00, 4.00, 5.00]
FRAME_VARIABLES = [
    "pressure",
    "valid_gas_temperature",
    "liquid_alpha",
    "NC12H26",
    "O2",
    "OH",
    "HO2",
    "H2O2",
    "H2O",
    "CO",
    "CO2",
]
SPECIES_OF_INTEREST = ["NC12H26", "O2", "OH", "HO2", "H2O2", "H2O", "CO", "CO2"]
BASE_FIELDS = [
    "liquid_alpha_rho",
    "vapor_alpha_rho",
    "air_alpha_rho",
    "liquid_alpha",
    "vapor_alpha",
    "air_alpha",
    "pressure",
]
LUSTRE_RE = re.compile(r"^lustre_(\d+)\.dat$")
RAW_RE = re.compile(r"^(?P<kind>[^.]+)\.(?P<index>\d+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")
REL_DENOM_EPS = 1.0e-300


def load_c3_helper():
    if not C3_HELPER.is_file():
        raise RuntimeError(f"required helper not found: {C3_HELPER}")
    spec = importlib.util.spec_from_file_location("c3_export_helper", C3_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import helper: {C3_HELPER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_c3_helper()
raw = base.raw
gas_metrics = base.gas_metrics


def mfc_raw_dirs(run_dir: Path) -> list[Path]:
    """Preserve the original analyzers' old-layout D/ and p_all/ support."""
    return [path for path in (Path(run_dir) / "D", Path(run_dir) / "p_all") if path.is_dir()]


raw.raw_dirs = mfc_raw_dirs


@dataclass(frozen=True)
class SaveItem:
    case: str
    run_dir: Path
    raw_step: int
    global_save: int
    origin: str


@dataclass(frozen=True)
class Layout:
    kind: str
    saves: list[int]


def ff(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_case_dict(case_py: Path) -> dict:
    ns = runpy.run_path(str(case_py), run_name="__mfc_case_loader__")
    if "case" not in ns or not isinstance(ns["case"], dict):
        raise RuntimeError(f"{case_py} did not define a case dictionary")
    return dict(ns["case"])


def species_names() -> list[str]:
    return base.species_names()


def ensure_raw_mappings(all_species: list[str]) -> None:
    base.ensure_species_fields(all_species)
    raw.FIELDS.setdefault("pressure", ("prim", 6))
    raw.FIELDS.setdefault("liquid_alpha_rho", ("cons", 1))
    raw.FIELDS.setdefault("vapor_alpha_rho", ("cons", 2))
    raw.FIELDS.setdefault("air_alpha_rho", ("cons", 3))
    raw.FIELDS.setdefault("liquid_alpha", ("cons", 7))
    raw.FIELDS.setdefault("vapor_alpha", ("cons", 8))
    raw.FIELDS.setdefault("air_alpha", ("cons", 9))


def old_layout_saves(run_dir: Path) -> list[int]:
    steps: set[int] = set()
    for root in (run_dir / "D", run_dir / "p_all"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.*.*.*.dat"):
            match = RAW_RE.match(path.name)
            if match:
                steps.add(int(match.group("step")))
    return sorted(steps)


def lustre_saves(run_dir: Path) -> list[int]:
    restart_dir = run_dir / "restart_data"
    if not restart_dir.is_dir():
        return []
    saves: list[int] = []
    for path in restart_dir.glob("lustre_*.dat"):
        match = LUSTRE_RE.match(path.name)
        if match:
            saves.append(int(match.group(1)))
    return sorted(set(saves))


def discover_layout(run_dir: Path) -> Layout:
    old = old_layout_saves(run_dir)
    if old:
        return Layout("old", old)
    parallel = lustre_saves(run_dir)
    if parallel:
        return Layout("parallel", parallel)
    raise RuntimeError(
        f"no saved states discovered in {run_dir}; expected old D/p_all files "
        "or restart_data/lustre_<number>.dat files"
    )


def select_saves(discovered: list[int], requested: list[int] | None) -> list[int]:
    if not requested:
        return discovered
    have = set(discovered)
    missing = [save for save in requested if save not in have]
    if missing:
        raise RuntimeError(f"requested saves are unavailable: {missing}; discovered {discovered[:8]}...{discovered[-8:] if discovered else []}")
    return list(requested)


def read_fortran_record(handle) -> bytes | None:
    marker = handle.read(4)
    if not marker:
        return None
    if len(marker) != 4:
        raise RuntimeError("truncated Fortran record marker")
    nbytes = int(np.frombuffer(marker, dtype="<i4")[0])
    if nbytes < 0 or nbytes > 10_000_000_000:
        raise RuntimeError(f"implausible Fortran record length {nbytes}")
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise RuntimeError("truncated Fortran record payload")
    tail = handle.read(4)
    if len(tail) != 4:
        raise RuntimeError("truncated Fortran record trailer")
    tail_nbytes = int(np.frombuffer(tail, dtype="<i4")[0])
    if tail_nbytes != nbytes:
        raise RuntimeError(f"Fortran record marker mismatch {nbytes} != {tail_nbytes}")
    return data


def read_post_binary(path: Path) -> dict:
    with path.open("rb") as f:
        header = read_fortran_record(f)
        if header is None:
            raise RuntimeError(f"empty binary database file: {path}")
        ints = np.frombuffer(header, dtype="<i4", count=4)
        if ints.size != 4:
            raise RuntimeError(f"bad binary database header in {path}")
        m, n, p, dbvars = [int(v) for v in ints]
        grid_record = read_fortran_record(f)
        if grid_record is None:
            raise RuntimeError(f"missing grid record in {path}")
        offset = 0
        dtype = "<f8"
        x_count = m + 2
        y_count = n + 2 if n > 0 else 0
        z_count = p + 2 if p > 0 else 0
        expected8 = 8 * (x_count + y_count + z_count)
        expected4 = 4 * (x_count + y_count + z_count)
        if len(grid_record) == expected4:
            dtype = "<f4"
        elif len(grid_record) != expected8:
            raise RuntimeError(
                f"unexpected grid record size {len(grid_record)} in {path}; expected {expected8} or {expected4}"
            )
        x_cb = np.frombuffer(grid_record, dtype=dtype, count=x_count, offset=offset).astype(float)
        offset += np.dtype(dtype).itemsize * x_count
        if n > 0:
            y_cb = np.frombuffer(grid_record, dtype=dtype, count=y_count, offset=offset).astype(float)
            offset += np.dtype(dtype).itemsize * y_count
        else:
            y_cb = np.array([0.0, 1.0], dtype=float)
        if p > 0:
            # C4 is 2-D, but keep the reader honest for general MFC binary files.
            _z_cb = np.frombuffer(grid_record, dtype=dtype, count=z_count, offset=offset).astype(float)
        shape = (m + 1, n + 1, p + 1)
        nvals = int(np.prod(shape))
        variables: dict[str, np.ndarray] = {}
        for _ in range(dbvars):
            record = read_fortran_record(f)
            if record is None:
                raise RuntimeError(f"missing variable record in {path}")
            name = record[:NAME_LEN].decode("ascii", errors="ignore").strip()
            payload = record[NAME_LEN:]
            if len(payload) == nvals * 8:
                vdtype = "<f8"
            elif len(payload) == nvals * 4:
                vdtype = "<f4"
            else:
                raise RuntimeError(
                    f"unexpected payload size for variable {name!r} in {path}: {len(payload)} bytes for {nvals} values"
                )
            arr = np.frombuffer(payload, dtype=vdtype, count=nvals).astype(float)
            arr = arr.reshape(shape, order="F")
            if p == 0:
                arr = arr[:, :, 0]
            variables[name] = np.array(arr, copy=True)
        return {"m": m, "n": n, "p": p, "x_cb": x_cb, "y_cb": y_cb, "variables": variables, "path": path}


def centers(boundaries: np.ndarray) -> np.ndarray:
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def write_raw_field(path: Path, x: np.ndarray, y: np.ndarray, values_xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.empty((x.size * y.size, 3), dtype=float)
    pos = 0
    for iy, yy in enumerate(y):
        rows[pos:pos + x.size, 0] = x
        rows[pos:pos + x.size, 1] = yy
        rows[pos:pos + x.size, 2] = values_xy[:, iy]
        pos += x.size
    np.savetxt(path, rows, fmt="%.16e")


def binary_to_raw_d(binary_dir: Path, d_dir: Path, save: int, all_species: list[str]) -> dict:
    rank_files = sorted(binary_dir.glob("p*/" + f"{save}.dat"), key=lambda p: int(p.parent.name[1:]) if p.parent.name[1:].isdigit() else 10**9)
    if not rank_files:
        raise RuntimeError(f"post_process did not produce binary rank files for save {save} under {binary_dir}")
    summary = {"rank_files": len(rank_files), "written_files": 0, "missing_binary_variables": []}
    for rank_file in rank_files:
        rank = int(rank_file.parent.name[1:])
        db = read_post_binary(rank_file)
        x = centers(db["x_cb"])
        y = centers(db["y_cb"])
        var = db["variables"]
        gas_mass = None
        if "alpha_rho2" in var and "alpha_rho3" in var:
            gas_mass = var["alpha_rho2"] + var["alpha_rho3"]
        mapping: list[tuple[str, int, np.ndarray | None]] = [
            ("cons", 1, var.get("alpha_rho1")),
            ("cons", 2, var.get("alpha_rho2")),
            ("cons", 3, var.get("alpha_rho3")),
            ("cons", 7, var.get("alpha1")),
            ("cons", 8, var.get("alpha2")),
            ("cons", 9, var.get("alpha3")),
            ("prim", 6, var.get("pres")),
        ]
        for kind, index, values in mapping:
            if values is None:
                summary["missing_binary_variables"].append(f"{rank_file}:{kind}.{index}")
                continue
            write_raw_field(d_dir / f"{kind}.{index}.{rank}.{save}.dat", x, y, values)
            summary["written_files"] += 1
        if gas_mass is None:
            summary["missing_binary_variables"].append(f"{rank_file}:alpha_rho2/alpha_rho3 for rhoY reconstruction")
            continue
        for species_index, name in enumerate(all_species, start=1):
            y_name = f"Y_{name}"
            y_values = var.get(y_name)
            if y_values is None:
                summary["missing_binary_variables"].append(f"{rank_file}:{y_name}")
                continue
            rhoY = gas_mass * y_values
            write_raw_field(d_dir / f"cons.{12 + species_index}.{rank}.{save}.dat", x, y, rhoY)
            summary["written_files"] += 1
    if summary["missing_binary_variables"]:
        preview = ", ".join(summary["missing_binary_variables"][:12])
        raise RuntimeError(f"missing variables while converting post_process binary to D fields: {preview}")
    return summary


def make_temp_case(raw_case: Path, save: int, temp_dir: Path) -> Path:
    case = load_case_dict(raw_case / "case.py")
    case.update({
        "parallel_io": "T",
        "format": 2,
        "prim_vars_wrt": "T",
        "cons_vars_wrt": "T",
        "pres_wrt": "T",
        "chem_wrt_T": "T",
        "n_start": save,
        "t_stop": save * T_SAVE,
        "t_step_start": save,
        "t_step_stop": save,
        "t_step_save": 1,
    })
    for i in range(1, 55):
        case[f"chem_wrt_Y({i})"] = "T"
    case_py = temp_dir / "case.py"
    case_py.write_text("#!/usr/bin/env python3\nimport json\ncase = " + repr(case) + "\nif __name__ == '__main__':\n    print(json.dumps(case))\n")
    restart_src = raw_case / "restart_data"
    if not restart_src.is_dir():
        raise RuntimeError(f"parallel-I/O layout requires restart_data in {raw_case}")
    os.symlink(restart_src, temp_dir / "restart_data", target_is_directory=True)
    return case_py


@contextlib.contextmanager
def prepared_save_dir(
    layout: Layout,
    raw_case: Path,
    save: int,
    export_out: Path,
    post_ranks: int,
    post_jobs: int,
    keep_temp: bool,
):
    if layout.kind == "old":
        yield raw_case, {"layout": "old", "post_process": "not used"}
        return
    tmp_parent = export_out / ".tmp_postprocess"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    temp_obj = tempfile.TemporaryDirectory(prefix=f"save_{save:06d}_", dir=tmp_parent)
    temp_dir = Path(temp_obj.name)
    try:
        case_py = make_temp_case(raw_case, save, temp_dir)
        cmd = [
            "bash",
            str(REPO / "mfc.sh"),
            "run",
            str(case_py),
            "-t",
            "post_process",
            "-j",
            str(post_jobs),
            "-n",
            str(post_ranks),
            "--no-build",
        ]
        proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False)
        log_path = temp_dir / "post_process.log"
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"post_process failed for save {save}; see {log_path}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        all_species = species_names()
        summary = binary_to_raw_d(temp_dir / "binary", temp_dir / "D", save, all_species)
        summary.update({"layout": "parallel", "post_process_log": str(log_path)})
        yield temp_dir, summary
    finally:
        if keep_temp:
            print(f"kept temporary post_process directory: {temp_dir}")
        else:
            temp_obj.cleanup()


def nearest_save(target_us: float, saves: list[int]) -> int | None:
    if not saves:
        return None
    return min(saves, key=lambda save: abs(save * T_SAVE * 1e6 - target_us))


def read_available_fields(item: SaveItem, names: list[str]) -> dict[str, dict]:
    return base.read_available_fields(item, names)


def empty_field() -> dict:
    return base.empty_field()


def finite_values(field: dict) -> np.ndarray:
    return base.finite_values(field)


def grid(field: dict):
    return base.grid(field)


def edges(values: np.ndarray) -> np.ndarray:
    return base.edges(values)


def valid_gas_temperature(fields: dict[str, dict]) -> dict:
    return base.valid_gas_temperature(fields)


def load_frame_field(item: SaveItem, variable: str) -> tuple[dict, str, str, bool]:
    if variable == "pressure":
        field = raw.read_field(item.run_dir, "pressure", item.raw_step)
        if field.get("available"):
            field = dict(field)
            field["values"] = {key: value / 1.0e6 for key, value in field["values"].items()}
        return field, "Pressure", "MPa", False
    if variable == "valid_gas_temperature":
        names = ["pressure", "liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho"]
        fields = read_available_fields(item, names)
        if not all(fields[name].get("available") for name in names):
            return empty_field(), "Valid-gas temperature", "K", False
        return valid_gas_temperature(fields), "Valid-gas temperature", "K", False
    if variable == "liquid_alpha":
        return raw.read_field(item.run_dir, "liquid_alpha", item.raw_step), "Liquid alpha", "dimensionless", False
    field = raw.read_field(item.run_dir, f"rhoY_{variable}", item.raw_step)
    return field, f"{variable} partial density", r"kg m$^{-3}$", True


def full_domain_arrays(field: dict, alpha_field: dict, valid_by_presence: bool = False):
    return base.full_domain_arrays(field, alpha_field, valid_by_presence=valid_by_presence)


def full_domain_field(field: dict, alpha_field: dict) -> dict:
    return base.full_domain_field(field, alpha_field)


def mask_description(variable: str) -> str:
    if variable == "valid_gas_temperature":
        return "actual valid-gas mask: gas_alpha > 0.5, gas_mass > 1e-8, alpha_liq < 0.5"
    return "finite field values on the full liquid-alpha domain grid"


def save_field_npz(path: Path, field: dict, item: SaveItem, variable: str, alpha_field: dict) -> bool:
    arrays = full_domain_arrays(field, alpha_field, valid_by_presence=(variable == "valid_gas_temperature"))
    if arrays is None:
        return False
    x, y, values, valid_mask, alpha_values = arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        values=values,
        time=np.array(item.global_save * T_SAVE),
        save=np.array(item.global_save),
        raw_step=np.array(item.raw_step),
        variable=np.array(variable),
        valid_mask=valid_mask,
        mask_description=np.array(mask_description(variable)),
        alpha_liq=alpha_values,
    )
    return True


def choose_scale(fields: list[dict], variable: str) -> dict:
    return base.choose_scale(fields, variable)


def draw_single_export_field(field: dict, alpha: dict, variable: str, item: SaveItem, path: Path) -> bool:
    g = grid(field)
    if g is None:
        return False
    x, y, values = g
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    scale = choose_scale([field], variable)
    fig, ax = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    if scale["log"]:
        mesh = ax.pcolormesh(
            edges(x * 1.0e6),
            edges(y * 1.0e6),
            np.where(values > 0.0, values, np.nan),
            shading="auto",
            cmap="magma",
            norm=colors.LogNorm(vmin=scale["vmin"], vmax=scale["vmax"]),
        )
    else:
        mesh = ax.pcolormesh(
            edges(x * 1.0e6),
            edges(y * 1.0e6),
            values,
            shading="auto",
            cmap="viridis",
            vmin=scale["vmin"],
            vmax=scale["vmax"],
        )
    ag = grid(alpha)
    if ag is not None:
        axx, ayy, aa = ag
        if axx.size >= 2 and ayy.size >= 2 and np.nanmin(aa) <= 0.5 <= np.nanmax(aa):
            ax.contour(axx * 1.0e6, ayy * 1.0e6, aa, levels=[0.5], colors="white", linewidths=1.1)
            ax.contour(axx * 1.0e6, ayy * 1.0e6, aa, levels=[0.5], colors="black", linewidths=0.35)
    fig.colorbar(mesh, ax=ax, pad=0.02)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"C4 {variable}, save {item.global_save}, t={item.global_save * T_SAVE * 1e6:.2f} µs")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return True


def add_liquid_loss(rows: list[dict]) -> None:
    base_mass = next((ff(r.get("liquid_mass")) for r in rows if math.isfinite(ff(r.get("liquid_mass")))), math.nan)
    for row in rows:
        liquid = ff(row.get("liquid_mass"))
        if math.isfinite(base_mass) and abs(base_mass) > REL_DENOM_EPS and math.isfinite(liquid):
            row["liquid_mass_loss"] = base_mass - liquid
            row["liquid_mass_loss_fraction"] = (base_mass - liquid) / base_mass
        else:
            row["liquid_mass_loss"] = math.nan
            row["liquid_mass_loss_fraction"] = math.nan


def fuel_inventory_rows(rows: list[dict]) -> list[dict]:
    initial = next(
        (
            ff(row.get("integrated_NC12H26")) + ff(row.get("liquid_mass"))
            for row in rows
            if math.isfinite(ff(row.get("integrated_NC12H26"))) and math.isfinite(ff(row.get("liquid_mass")))
        ),
        math.nan,
    )
    out: list[dict] = []
    for row in rows:
        gas = ff(row.get("integrated_NC12H26"))
        liquid = ff(row.get("liquid_mass"))
        combined = gas + liquid if math.isfinite(gas) and math.isfinite(liquid) else math.nan
        out.append({
            "save": row.get("global_save", row.get("save")),
            "raw_step": row.get("raw_step"),
            "time_s": row.get("time_s"),
            "time_us": row.get("time_us"),
            "gas_parent_fuel_mass": gas,
            "liquid_dodecane_mass": liquid,
            "combined_parent_dodecane_mass": combined,
            "gas_fraction_of_initial_combined": gas / initial if math.isfinite(initial) and abs(initial) > REL_DENOM_EPS and math.isfinite(gas) else math.nan,
            "liquid_fraction_of_initial_combined": liquid / initial if math.isfinite(initial) and abs(initial) > REL_DENOM_EPS and math.isfinite(liquid) else math.nan,
            "combined_fraction_of_initial": combined / initial if math.isfinite(initial) and abs(initial) > REL_DENOM_EPS and math.isfinite(combined) else math.nan,
        })
    return out


def export_case(args: argparse.Namespace) -> None:
    raw_case = args.raw_case.resolve()
    out = args.export_out.resolve()
    all_species = species_names()
    ensure_raw_mappings(all_species)
    layout = discover_layout(raw_case)
    saves = select_saves(layout.saves, args.saves)
    if not saves:
        raise RuntimeError(f"zero saves selected from {raw_case}")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    time_rows: list[dict] = []
    manifest: list[dict] = []
    post_rows: list[dict] = []
    for save in saves:
        with prepared_save_dir(layout, raw_case, save, out, args.post_ranks, args.post_jobs, args.keep_temp) as (run_dir, prep):
            item = SaveItem("C4_EXPORT", run_dir, save, save, prep.get("layout", layout.kind))
            row = base.compute_metrics_incremental(item, all_species, "C4_EXPORT")
            row["source_layout"] = layout.kind
            row["source_case_dir"] = str(raw_case)
            rows.append(row)
            time_rows.append({
                "case": "C4_EXPORT",
                "save": save,
                "raw_step": save,
                "time_s": save * T_SAVE,
                "time_us": save * T_SAVE * 1e6,
                "source_layout": layout.kind,
                "source_case_dir": str(raw_case),
            })
            post_rows.append({"save": save, **prep})
            alpha = raw.read_field(run_dir, "liquid_alpha", save) if base.field_available(run_dir, "liquid_alpha", save) else empty_field()
            for variable in FRAME_VARIABLES:
                target_saves = [nearest_save(t, saves) for t in FRAME_TIMES_US]
                if save not in set(v for v in target_saves if v is not None):
                    continue
                target_us = min(FRAME_TIMES_US, key=lambda t: abs(save * T_SAVE * 1e6 - t))
                field, _label, _unit, _is_species = load_frame_field(item, variable)
                entry = {
                    "target_time_us": target_us,
                    "actual_time_us": save * T_SAVE * 1e6,
                    "time_mismatch_us": save * T_SAVE * 1e6 - target_us,
                    "save": save,
                    "raw_step": save,
                    "variable": variable,
                }
                if not field.get("available") or finite_values(field).size == 0:
                    entry.update({"status": "skipped", "reason": "field unavailable/no finite values"})
                    manifest.append(entry)
                    continue
                if not alpha.get("available"):
                    entry.update({"status": "skipped", "reason": "liquid alpha unavailable for full-domain export"})
                    manifest.append(entry)
                    continue
                stem = f"{variable}_save{save:03d}_t{save * T_SAVE * 1e6:.2f}us"
                png = out / "field_frames" / variable / f"{stem}.png"
                npz = out / "field_data" / variable / f"{stem}.npz"
                full_field = full_domain_field(field, alpha)
                png_ok = draw_single_export_field(full_field, alpha, variable, item, png)
                npz_ok = save_field_npz(npz, field, item, variable, alpha)
                entry.update({
                    "status": "written" if png_ok and npz_ok else "partial",
                    "reason": "" if png_ok and npz_ok else "PNG or NPZ write failed",
                    "png": str(png.relative_to(out)) if png_ok else "",
                    "npz": str(npz.relative_to(out)) if npz_ok else "",
                })
                manifest.append(entry)
    add_liquid_loss(rows)
    write_csv(out / "c4_timeseries.csv", rows)
    write_csv(out / "c4_time_map.csv", time_rows)
    write_csv(out / "c4_species_bounds_by_save.csv", [
        {key: row.get(key, "") for key in [
            "global_save",
            "raw_step",
            "time_s",
            "time_us",
            "species_negative_rhoY_count",
            "species_negative_Y_count",
            "species_Y_above_one_count",
            "species_sumY_min",
            "species_sumY_max",
            "species_sumY_max_abs_error",
            "species_sumY_violation_count",
            "species_min_rhoY",
            "species_min_rhoY_name",
            "species_min_Y",
            "species_max_Y",
            "species_max_Y_name",
            "missing_species_fields",
            "species_diagnostics_available",
            "species_diagnostics_missing_reason",
        ]}
        for row in rows
    ])
    write_csv(out / "c4_mass_consistency_by_save.csv", [
        {key: row.get(key, "") for key in [
            "global_save",
            "raw_step",
            "time_s",
            "time_us",
            "integrated_valid_gas_mass",
            "integrated_valid_gas_sum_rhoY",
            "mass_consistency_integrated_diff",
            "mass_consistency_relative_error",
        ]}
        for row in rows
    ])
    write_csv(out / "c4_fuel_inventory.csv", fuel_inventory_rows(rows))
    write_csv(out / "c4_frame_manifest.csv", manifest)
    write_csv(out / "c4_postprocess_manifest.csv", post_rows)
    base.make_export_trend_plots(rows, out / "trend_plots")
    summary = [
        "C4 compact export",
        "=================",
        "",
        f"raw_case: {raw_case}",
        f"output_dir: {out}",
        f"source_layout: {layout.kind}",
        f"saves: {saves}",
        f"timeseries_rows: {len(rows)}",
        f"frame_manifest_rows: {len(manifest)}",
        "",
        "Parallel-I/O Lustre files are read only by MFC post_process. The exporter converts temporary format=2 post_process binary output to the existing D-style analyzer fields one save at a time.",
        "NPZ fields: x, y, values, time, save, raw_step, variable, valid_mask, mask_description, alpha_liq.",
        "Raw restart_data, D, and p_all data are not copied into the compact export.",
    ]
    (out / "c4_export_summary.txt").write_text("\n".join(summary) + "\n")


def dry_run(args: argparse.Namespace) -> None:
    all_species = species_names()
    ensure_raw_mappings(all_species)
    raw_case = args.raw_case.resolve()
    print(f"repo: {REPO}")
    print(f"raw_case: {raw_case}")
    print(f"export_out: {args.export_out.resolve()}")
    print("field mappings:")
    for name in ["liquid_alpha_rho", "vapor_alpha_rho", "air_alpha_rho", "liquid_alpha", "vapor_alpha", "air_alpha", "pressure", "rhoY_NC12H26", "rhoY_O2"]:
        print(f"  {name}: {raw.FIELDS.get(name)}")
    try:
        layout = discover_layout(raw_case)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return
    saves = select_saves(layout.saves, args.saves)
    print(f"layout: {layout.kind}")
    print(f"discovered_saves_count: {len(layout.saves)}")
    print(f"discovered_save_range: {layout.saves[0]}..{layout.saves[-1]}")
    print(f"selected_saves: {saves}")
    if layout.kind == "parallel":
        print("parallel adapter: MFC post_process format=2 per save, temporary binary-to-D conversion via MFC f_binary_reader.m format")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-case", action="store_true", help="Export compact C4 artifacts from old D/p_all or parallel-I/O restart_data/lustre saves")
    parser.add_argument("--export-c3", action="store_true", help="Backward-compatible alias for --export-case")
    parser.add_argument("--raw-case", type=Path, default=DEFAULT_CASE, help="MFC case directory containing case.py and either D/p_all or restart_data/lustre_*.dat")
    parser.add_argument("--export-out", type=Path, default=DEFAULT_OUT, help="Compact export output directory")
    parser.add_argument("--saves", type=int, nargs="*", help="Optional explicit save indices to process, e.g. --saves 0 1")
    parser.add_argument("--post-ranks", type=int, default=4, help="MPI ranks for temporary MFC post_process conversion of parallel-I/O saves")
    parser.add_argument("--post-jobs", type=int, default=8, help="Build/run job count passed to mfc.sh for post_process")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary post_process conversion directories for debugging")
    parser.add_argument("--dry-run", action="store_true", help="Discover saves and print mappings without exporting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.export_case or args.export_c3:
        export_case(args)
        return
    raise SystemExit("No action requested. Use --export-case, --export-c3, or --dry-run.")


if __name__ == "__main__":
    main()
