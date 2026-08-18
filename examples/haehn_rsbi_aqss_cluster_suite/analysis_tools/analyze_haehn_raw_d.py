#!/usr/bin/env python3
"""Analyze Haehn/RSBI raw MFC D/*.dat field outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ATM_PA = 101325.0
RU = 8.31446261815324
LOG_FLOOR = 1.0e-30
FILENAME_RE = re.compile(r"^(?P<var>[A-Za-z_]+\.\d+)\.(?P<part>\d+)\.(?P<step>\d+)\.dat$")

SPECIES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2", "XE"]
MOLECULAR_WEIGHTS = {
    "H2": 2.01588e-3,
    "H": 1.00794e-3,
    "O": 15.9994e-3,
    "O2": 31.9988e-3,
    "OH": 17.00734e-3,
    "H2O": 18.01528e-3,
    "HO2": 33.00674e-3,
    "H2O2": 34.01468e-3,
    "AR": 39.948e-3,
    "N2": 28.0134e-3,
    "XE": 131.293e-3,
}


@dataclass
class Field:
    x: np.ndarray
    y: np.ndarray
    value: np.ndarray
    files: list[str]


def parse_var_index(var_name: str) -> int:
    return int(var_name.split(".", 1)[1])


def var_name(prefix_index: int) -> str:
    return f"prim.{prefix_index}"


def parse_steps(text: str | None, available: list[int] | None = None) -> list[int]:
    if text is None or text.lower() == "all":
        return list(available or [])
    steps = []
    for item in text.split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    return steps


def parse_limits(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated limits, got {text!r}")
    return parts[0], parts[1]


def discover_files(run_dir: Path) -> dict[int, dict[str, list[Path]]]:
    d_dir = run_dir / "D"
    search_dir = d_dir if d_dir.is_dir() else run_dir
    discovered: dict[int, dict[str, list[Path]]] = {}
    for path in sorted(search_dir.glob("*.dat")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        step = int(match.group("step"))
        var = match.group("var")
        discovered.setdefault(step, {}).setdefault(var, []).append(path)
    return discovered


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 3:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected at least x, y, value")
    return data[:, 0], data[:, 1], data[:, -1]


def load_field(files: list[Path]) -> Field:
    xs, ys, values = [], [], []
    for path in sorted(files):
        x, y, value = load_table(path)
        xs.append(x)
        ys.append(y)
        values.append(value)
    return Field(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        value=np.concatenate(values),
        files=[str(path) for path in sorted(files)],
    )


def align_to(reference: Field, field: Field | None) -> np.ndarray:
    if field is None:
        return np.full(reference.value.shape, np.nan)
    if (
        len(field.value) == len(reference.value)
        and np.allclose(field.x, reference.x, rtol=0.0, atol=1.0e-13)
        and np.allclose(field.y, reference.y, rtol=0.0, atol=1.0e-13)
    ):
        return field.value
    lookup = {(f"{x:.14e}", f"{y:.14e}"): value for x, y, value in zip(field.x, field.y, field.value)}
    return np.array([lookup.get((f"{x:.14e}", f"{y:.14e}"), np.nan) for x, y in zip(reference.x, reference.y)])


def pressure_to_atm_and_pa(p_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    finite = p_raw[np.isfinite(p_raw)]
    if finite.size and np.nanmax(np.abs(finite)) <= 200.0:
        return p_raw, p_raw * ATM_PA, "atm-like prim pressure converted to Pa for T"
    return p_raw / ATM_PA, p_raw, "Pa prim pressure"


def compute_temperature(p_pa: np.ndarray, rho: np.ndarray, species_values: dict[str, np.ndarray]) -> np.ndarray:
    inv_mw_sum = np.zeros_like(p_pa, dtype=float)
    for name in SPECIES:
        yk = species_values.get(name)
        if yk is not None:
            inv_mw_sum += yk / MOLECULAR_WEIGHTS[name]
    rmix = RU * inv_mw_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        temp = p_pa / (rho * rmix)
    temp[~np.isfinite(temp)] = np.nan
    return temp


def safe_stat(func, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(func(finite))


def field_map_for_step(discovered: dict[int, dict[str, list[Path]]], step: int) -> dict[str, Field]:
    return {var: load_field(files) for var, files in sorted(discovered.get(step, {}).items())}


def crop_mask(x: np.ndarray, y: np.ndarray, xlim, ylim) -> np.ndarray:
    mask = np.ones_like(x, dtype=bool)
    if xlim:
        mask &= (x >= xlim[0]) & (x <= xlim[1])
    if ylim:
        mask &= (y >= ylim[0]) & (y <= ylim[1])
    return mask


def grid_from_points(x: np.ndarray, y: np.ndarray, value: np.ndarray):
    ux = np.unique(x)
    uy = np.unique(y)
    if ux.size * uy.size > max(1, x.size) * 2:
        return None
    x_index = {val: idx for idx, val in enumerate(ux)}
    y_index = {val: idx for idx, val in enumerate(uy)}
    grid = np.full((uy.size, ux.size), np.nan)
    for xi, yi, vi in zip(x, y, value):
        grid[y_index[yi], x_index[xi]] = vi
    return ux, uy, grid


def plot_field(output_path: Path, x: np.ndarray, y: np.ndarray, value: np.ndarray, title: str, label: str, robust: bool, xlim, ylim) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    mask = crop_mask(x, y, xlim, ylim) & np.isfinite(value)
    if not np.any(mask):
        return
    xp, yp, vp = x[mask], y[mask], value[mask]
    actual_min = safe_stat(np.nanmin, vp)
    actual_max = safe_stat(np.nanmax, vp)
    if robust:
        vmin, vmax = np.nanpercentile(vp, [1.0, 99.0])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = actual_min, actual_max
    else:
        vmin, vmax = actual_min, actual_max

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    gridded = grid_from_points(xp, yp, vp)
    if gridded is not None:
        ux, uy, grid = gridded
        image = ax.imshow(
            grid,
            origin="lower",
            extent=[float(ux.min()), float(ux.max()), float(uy.min()), float(uy.max())],
            aspect="equal",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
    else:
        image = ax.scatter(xp, yp, c=vp, s=1, marker="s", linewidths=0, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal", adjustable="box")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    scale_note = "robust 1-99%" if robust else "full scale"
    ax.set_title(f"{title} ({scale_note}; actual min={actual_min:.4e}, max={actual_max:.4e})")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(label)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    colorbar.formatter = formatter
    colorbar.update_ticks()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_step(step: int, fields: dict[str, Field], args) -> tuple[dict[str, object], dict[str, np.ndarray], Field | None]:
    rho_field = fields.get(args.rho_var)
    pressure_field = fields.get(args.pressure_var)
    if rho_field is None or pressure_field is None:
        return {"step": step, "missing": "rho or pressure"}, {}, None

    rho = rho_field.value
    p_raw = align_to(rho_field, pressure_field)
    p_atm, p_pa, pressure_source = pressure_to_atm_and_pa(p_raw)

    start = parse_var_index(args.species_start_var)
    species_values = {}
    for offset, name in enumerate(SPECIES):
        species_values[name] = align_to(rho_field, fields.get(var_name(start + offset)))

    sum_y = np.zeros_like(rho, dtype=float)
    for values in species_values.values():
        sum_y += values
    temp = compute_temperature(p_pa, rho, species_values)

    row = {
        "step": step,
        "pressure_source": pressure_source,
        "rho_min": safe_stat(np.nanmin, rho),
        "rho_max": safe_stat(np.nanmax, rho),
        "p_min_atm": safe_stat(np.nanmin, p_atm),
        "p_max_atm": safe_stat(np.nanmax, p_atm),
        "Tmax_K": safe_stat(np.nanmax, temp),
        "T_999_K": float(np.nanpercentile(temp[np.isfinite(temp)], 99.9)) if np.any(np.isfinite(temp)) else math.nan,
        "sumY_min": safe_stat(np.nanmin, sum_y),
        "sumY_max": safe_stat(np.nanmax, sum_y),
        "max_abs_sumY_err": safe_stat(np.nanmax, np.abs(sum_y - 1.0)),
    }
    for name in ("H2", "O2", "OH", "H2O", "HO2"):
        values = species_values[name]
        row[f"max_Y{name}"] = safe_stat(np.nanmax, values)
        row[f"sum_Y{name}"] = safe_stat(np.nansum, values)
        row[f"raw_sum_Y{name}"] = row[f"sum_Y{name}"]

    plot_data = {"T_K": temp, "P_atm": p_atm}
    plot_data.update({f"Y{name}": values for name, values in species_values.items()})
    return row, plot_data, rho_field


def write_summary(output_dir: Path, rows: list[dict[str, object]], inventory: dict) -> None:
    lines = [
        "# Haehn Raw D Analysis Summary",
        "",
        f"- Run directory: `{inventory['run_dir']}`",
        f"- Steps analyzed: {', '.join(str(row['step']) for row in rows) if rows else 'none'}",
        f"- Files inventoried: {inventory['file_count']}",
        "",
    ]
    if rows:
        max_temp = max(float(row.get("Tmax_K", math.nan)) for row in rows)
        max_heat_note = "not computed from D fields"
        worst_sumy = max(float(row.get("max_abs_sumY_err", math.nan)) for row in rows)
        worst_min_sumy = min(float(row.get("sumY_min", math.nan)) for row in rows)
        max_oh = max(float(row.get("max_YOH", math.nan)) for row in rows)
        lines.extend(
            [
                f"- Max `Tmax_K`: {max_temp:.6e}",
                f"- Worst `max_abs_sumY_err`: {worst_sumy:.6e}",
                f"- Worst `sumY_min`: {worst_min_sumy:.6e}",
                f"- Max `YOH`: {max_oh:.6e}",
                f"- Heat budget: {max_heat_note}",
                "",
            ]
        )
    lines.extend(["## Notes", "", "- `prim.4` is treated as pressure by default.", "- `prim.5` is not used as pressure.", "- Pressure is reported in atm. Atm-like pressure is converted to Pa only for temperature reconstruction.", ""])
    (output_dir / "summary.md").write_text("\n".join(lines))


def make_plots(output_dir: Path, step: int, reference: Field, plot_data: dict[str, np.ndarray], xlim, ylim) -> None:
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        print(f"Skipping plots for step {step}: matplotlib unavailable ({exc})")
        return

    plot_specs = [
        ("T_K", "T_K", "T [K]", False),
        ("T_K", "T_K_robust", "T [K]", True),
        ("P_atm", "P_atm", "Pressure [atm]", False),
        ("P_atm", "P_atm_robust", "Pressure [atm]", True),
        ("YH2", "YH2", "YH2", False),
        ("YO2", "YO2", "YO2", False),
        ("YOH", "YOH", "YOH", False),
        ("YH2O", "YH2O", "YH2O", False),
        ("YHO2", "YHO2", "YHO2", False),
    ]
    for key, label, color_label, robust in plot_specs:
        if key in plot_data:
            plot_field(output_dir / f"step_{step:06d}_{label}.png", reference.x, reference.y, plot_data[key], f"step {step} {label}", color_label, robust, xlim, ylim)

    for key in ("YOH", "YH2O", "YHO2"):
        if key in plot_data:
            log_values = np.log10(np.maximum(plot_data[key], LOG_FLOOR))
            plot_field(output_dir / f"step_{step:06d}_log10_{key}.png", reference.x, reference.y, log_values, f"step {step} log10({key})", f"log10({key})", False, xlim, ylim)


def run_analysis(args) -> int:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (run_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_files(run_dir)
    available_steps = sorted(discovered)
    steps = parse_steps(args.steps, available_steps)
    plot_steps = parse_steps(args.plot_steps, available_steps) if args.plot_steps else []
    xlim = parse_limits(args.xlim)
    ylim = parse_limits(args.ylim)

    rows = []
    inventory = {
        "run_dir": str(run_dir),
        "available_steps": available_steps,
        "requested_steps": steps,
        "plot_steps": plot_steps,
        "file_count": sum(len(files) for by_var in discovered.values() for files in by_var.values()),
        "variables_by_step": {str(step): sorted(by_var) for step, by_var in discovered.items()},
        "files_by_step_var": {
            str(step): {var: [str(path) for path in files] for var, files in by_var.items()} for step, by_var in discovered.items()
        },
    }

    for step in steps:
        fields = field_map_for_step(discovered, step)
        row, plot_data, reference = analyze_step(step, fields, args)
        rows.append(row)
        if step in plot_steps and reference is not None:
            make_plots(output_dir, step, reference, plot_data, xlim, ylim)

    csv_path = output_dir / "haehn_raw_D_summary_by_step.csv"
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "step",
        "pressure_source",
        "rho_min",
        "rho_max",
        "p_min_atm",
        "p_max_atm",
        "Tmax_K",
        "T_999_K",
        "max_YH2",
        "max_YO2",
        "max_YOH",
        "max_YH2O",
        "max_YHO2",
        "sum_YH2",
        "sum_YO2",
        "sum_YOH",
        "sum_YH2O",
        "sum_YHO2",
        "raw_sum_YH2",
        "raw_sum_YO2",
        "raw_sum_YOH",
        "raw_sum_YH2O",
        "raw_sum_YHO2",
        "sumY_min",
        "sumY_max",
        "max_abs_sumY_err",
    ]
    fieldnames = [name for name in preferred if name in fieldnames] + [name for name in fieldnames if name not in preferred]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    write_summary(output_dir, rows, inventory)
    print(f"Wrote raw D analysis to {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("raw_d_analysis"))
    parser.add_argument("--steps", default="all", help="Comma list of timesteps or all.")
    parser.add_argument("--plot-steps", default=None, help="Comma list of timesteps to plot.")
    parser.add_argument("--xlim", default=None, help="xmin,xmax crop.")
    parser.add_argument("--ylim", default=None, help="ymin,ymax crop.")
    parser.add_argument("--pressure-var", default="prim.4")
    parser.add_argument("--rho-var", default="prim.1")
    parser.add_argument("--species-start-var", default="prim.6")
    args = parser.parse_args()

    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
