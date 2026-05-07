#!/usr/bin/env python3
import argparse
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARS = {
    "liquid": ("cons", 1),
    "vapor": ("cons", 2),
    "pressure": ("prim", 6),
    "alpha_rho1": ("cons", 1),
    "alpha_rho2": ("cons", 2),
    "c12h26": ("cons", 13),
    "o2": ("cons", 14),
    "n2": ("cons", 15),
    "co2": ("cons", 16),
    "h2o": ("cons", 17),
}

SPECIES = ("c12h26", "o2", "n2", "co2", "h2o")


def data_dir(root):
    root = pathlib.Path(root)
    if (root/"D").is_dir():
        return root/"D"
    return root


def steps(root, kind, var):
    found = []
    for path in data_dir(root).glob(f"{kind}.{var}.00.*.dat"):
        found.append(int(path.name.rsplit(".", 2)[1]))
    return sorted(found)


def read_raw(root, kind, var, step):
    path = data_dir(root)/f"{kind}.{var}.00.{step:06d}.dat"
    return np.loadtxt(path)


def read_values(root, kind, var, step):
    return read_raw(root, kind, var, step)[:, -1]


def grid(root, kind, var, step):
    raw = read_raw(root, kind, var, step)
    if raw.shape[1] < 3:
        x = np.arange(raw.shape[0])
        y = np.array([0])
        return x, y, raw[:, -1].reshape((raw.shape[0], 1))
    x = np.unique(raw[:, 0])
    y = np.unique(raw[:, 1])
    values = raw[:, -1].reshape((len(x), len(y)), order="F")
    return x, y, values


def area_weight(root):
    first = steps(root, "cons", 1)[0]
    raw = read_raw(root, "cons", 1, first)
    if raw.shape[1] < 3:
        x = np.unique(raw[:, 0])
        return (x.max() - x.min())/max(len(x) - 1, 1)
    x = np.unique(raw[:, 0])
    y = np.unique(raw[:, 1])
    dx = (x.max() - x.min())/max(len(x) - 1, 1)
    dy = (y.max() - y.min())/max(len(y) - 1, 1)
    return dx*dy


def integrate(root, kind, var, step):
    return float(np.sum(read_values(root, kind, var, step))*area_weight(root))


def finite_sweep(root):
    total = 0
    bad = []
    for path in sorted(data_dir(root).glob("*.dat")):
        total += 1
        arr = np.loadtxt(path)
        if not np.isfinite(arr).all():
            bad.append(path.name)
    return total, bad


def find_log(root):
    root = pathlib.Path(root)
    candidates = [
        root/"run_stdout.log",
        root/"stdout.log",
        root/"run.log",
        root/"mfc.out",
    ]
    candidates.extend(sorted(root.glob("*.log")))
    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_mdot(root):
    log = find_log(root)
    if log is None:
        return []
    text = log.read_text(errors="replace")
    patterns = [
        re.compile(
            r"m_dot_evap\s+@\s+t_step\s+=\s+(\d+)\s+min\s+=\s+([-+0-9.Ee]+)\s+max\s+=\s+([-+0-9.Ee]+)\s+mean\s+=\s+([-+0-9.Ee]+)"
        ),
        re.compile(
            r"m_dot_evap stats step\s+(\d+)\s+min\s+([-+0-9.Ee]+)\s+max\s+([-+0-9.Ee]+)\s+mean\s+([-+0-9.Ee]+)"
        ),
    ]
    out = []
    for pattern in patterns:
        out.extend((int(s), float(mn), float(mx), float(mean)) for s, mn, mx, mean in pattern.findall(text))
    return sorted(set(out))


def series(root, kind, var):
    return [(step, integrate(root, kind, var, step)) for step in steps(root, kind, var)]


def pressure_series(root):
    out = []
    for step in steps(root, "prim", 6):
        values = read_values(root, "prim", 6, step)
        out.append((step, float(np.min(values)), float(np.max(values))))
    return out


def analyze_one(root, label):
    root = pathlib.Path(root)
    total, bad = finite_sweep(root)
    cons_steps = steps(root, "cons", 1)
    initial = cons_steps[0]
    final = cons_steps[-1]
    masses = {
        name: {
            "series": series(root, kind, var),
            "initial": integrate(root, kind, var, initial),
            "final": integrate(root, kind, var, final),
            "change": integrate(root, kind, var, final) - integrate(root, kind, var, initial),
        }
        for name, (kind, var) in VARS.items()
        if kind == "cons"
    }
    return {
        "label": label,
        "root": str(root),
        "conservative_steps": cons_steps,
        "primitive_pressure_steps": steps(root, "prim", 6),
        "finite": not bad,
        "file_count": total,
        "bad_files": bad,
        "m_dot_evap": parse_mdot(root),
        "masses": masses,
        "pressure": pressure_series(root),
    }


def final_at(summary, name):
    return summary["masses"][name]["final"]


def pass_fail(off, on):
    off_ok = (
        off["finite"]
        and off["masses"]["liquid"]["change"] < 0.0
        and off["masses"]["vapor"]["change"] > 0.0
        and off["masses"]["c12h26"]["change"] > 0.0
    )
    on_ok = (
        on["finite"]
        and on["masses"]["vapor"]["change"] > 0.0
        and final_at(on, "c12h26") < final_at(off, "c12h26")
        and final_at(on, "o2") < final_at(off, "o2")
        and final_at(on, "co2") > final_at(off, "co2")
        and final_at(on, "h2o") > final_at(off, "h2o")
    )
    return {"off": off_ok, "on": on_ok, "overall": off_ok and on_ok}


def plot_grid(root, name, step, output, title):
    kind, var = VARS[name]
    x, y, values = grid(root, kind, var, step)
    plt.figure(figsize=(8, 4.8))
    plt.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
    )
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_diff(off_root, on_root, name, step, output):
    kind, var = VARS[name]
    x, y, off = grid(off_root, kind, var, step)
    _, _, on = grid(on_root, kind, var, step)
    plt.figure(figsize=(8, 4.8))
    plt.imshow(
        (on - off).T,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
    )
    plt.colorbar()
    plt.title(f"ON - OFF final {name}")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_mass_series(off, on, outdir):
    plt.figure(figsize=(9, 5.5))
    for summary, style in ((off, "-"), (on, "--")):
        label = "OFF" if summary is off else "ON"
        for name in ("liquid", "vapor", "c12h26", "o2", "co2", "h2o"):
            data = summary["masses"][name]["series"]
            plt.plot([s for s, _ in data], [v for _, v in data], style, marker="o", label=f"{label} {name}")
    plt.xlabel("step")
    plt.ylabel("area-integrated conservative field")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir/"mass_time_series.png", dpi=180)
    plt.close()


def write_summary(outdir, off, on, comparisons, checks):
    path = outdir/"analyzer_summary.txt"
    with path.open("w") as f:
        f.write("Reduced 2D dodecane global-one-step validation\n")
        f.write(f"PASS off={checks['off']} on={checks['on']} overall={checks['overall']}\n\n")
        for summary in (off, on):
            f.write(f"[{summary['label']}]\n")
            f.write(f"root={summary['root']}\n")
            f.write(f"finite={summary['finite']} files={summary['file_count']} bad={summary['bad_files']}\n")
            f.write(f"conservative_steps={summary['conservative_steps']}\n")
            f.write(f"primitive_pressure_steps={summary['primitive_pressure_steps']}\n")
            for step, mn, mx, mean in summary["m_dot_evap"]:
                f.write(f"m_dot_evap step={step} min={mn:.16e} max={mx:.16e} mean={mean:.16e}\n")
            for name in ("liquid", "vapor", "c12h26", "o2", "n2", "co2", "h2o"):
                stats = summary["masses"][name]
                f.write(
                    f"{name} initial={stats['initial']:.16e} final={stats['final']:.16e} "
                    f"change={stats['change']:.16e}\n"
                )
            for step, mn, mx in summary["pressure"]:
                f.write(f"pressure step={step} min={mn:.16e} max={mx:.16e}\n")
            f.write("\n")
        f.write("[ON-OFF final]\n")
        for name, value in comparisons.items():
            f.write(f"{name}={value:.16e}\n")
        f.write("\n[Source accumulation]\n")
        f.write(f"c12h26_source_accumulation_OFF={off['masses']['c12h26']['change']:.16e}\n")
        f.write(f"c12h26_source_accumulation_ON={on['masses']['c12h26']['change']:.16e}\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Analyze refined reduced-2D dodecane global-one-step validation.")
    parser.add_argument("--off", required=True, help="Reactions-off run directory containing D/")
    parser.add_argument("--on", required=True, help="Reactions-on run directory containing D/")
    parser.add_argument("--out", default="dodecane_global_reduced_analysis", help="Analysis/plot output directory")
    args = parser.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    off = analyze_one(args.off, "reactions_off")
    on = analyze_one(args.on, "reactions_on")
    common_steps = sorted(set(off["conservative_steps"]) & set(on["conservative_steps"]))
    final_step = common_steps[-1]

    comparisons = {
        "c12h26_ON_minus_OFF": final_at(on, "c12h26") - final_at(off, "c12h26"),
        "o2_ON_minus_OFF": final_at(on, "o2") - final_at(off, "o2"),
        "co2_ON_minus_OFF": final_at(on, "co2") - final_at(off, "co2"),
        "h2o_ON_minus_OFF": final_at(on, "h2o") - final_at(off, "h2o"),
    }
    checks = pass_fail(off, on)

    plot_mass_series(off, on, outdir)
    for name in ("c12h26", "o2", "co2", "h2o"):
        plot_diff(args.off, args.on, name, final_step, outdir/f"final_ON_minus_OFF_{name}.png")
    for name in ("pressure", "alpha_rho1", "alpha_rho2", "c12h26", "co2", "h2o"):
        kind, var = VARS[name]
        step_list = steps(args.on, kind, var)
        plot_grid(args.on, name, step_list[-1], outdir/f"final_ON_{name}.png", f"ON final {name}")

    summary_path = write_summary(outdir, off, on, comparisons, checks)
    (outdir/"analyzer_summary.json").write_text(json.dumps({
        "off": off,
        "on": on,
        "comparisons": comparisons,
        "checks": checks,
    }, indent=2))
    print(f"summary={summary_path}")
    print(f"PASS off={checks['off']} on={checks['on']} overall={checks['overall']}")


if __name__ == "__main__":
    main()
