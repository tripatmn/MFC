# Haehn/RSBI AQSS Analysis Tools

These tools analyze completed cluster run folders. They do not run MFC, do not submit jobs, and do not require Cantera.

## Log Analysis

Parse stdout/stderr logs and AQSS diagnostic lines:

```bash
python3 examples/haehn_rsbi_aqss_cluster_suite/analysis_tools/analyze_haehn_rsbi_log.py \
  --run-dir /path/to/run/folder \
  --out-glob "*.out"
```

Outputs are written to `/path/to/run/folder/analysis_out` by default:

- `aqss_diag_by_rank.csv`
- `aqss_diag_reduced.csv`
- `aqss_pre_repairs.csv`
- `aqss_aborts.csv`
- `summary.md`

Use `--output-dir /path/to/analysis_out` to place outputs elsewhere.

## Output Inventory

Inventory post-process outputs under `D/`, `p_all/`, or equivalent folders. If `h5py` is installed, readable HDF5 files include dataset names, shapes, and dtypes.

```bash
python3 examples/haehn_rsbi_aqss_cluster_suite/analysis_tools/inspect_haehn_outputs.py \
  --run-dir /path/to/run/folder
```

Outputs:

- `output_inventory.txt`
- `output_inventory.json`

## Suggested Post-Job Workflow

From `/home/tripatmn/mfc_work/MFC_aqss_proto_preserved` after a cluster job finishes:

```bash
python3 examples/haehn_rsbi_aqss_cluster_suite/analysis_tools/analyze_haehn_rsbi_log.py \
  --run-dir /path/to/run/folder \
  --out-glob "*.out"

python3 examples/haehn_rsbi_aqss_cluster_suite/analysis_tools/inspect_haehn_outputs.py \
  --run-dir /path/to/run/folder
```

Read `analysis_out/summary.md` first. A run with only `AQSS_PRE_REPAIR` and zero heat activity is still an interface-stability diagnostic. A run with `AQSS_POST_ABORT` is an AQSS-created invalidity and should be treated separately.
