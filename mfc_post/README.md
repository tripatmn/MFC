# mfc_post input and processing foundation

Run from the repository root:

```console
./mfc-post inspect /path/to/case
./mfc-post inspect /path/to/case --format json
./mfc-post inspect /path/to/copied-run --mechanism mechanism.yaml --phase gas
./mfc-post process /path/to/case --execution serial
mpiexec -n 8 ./mfc-post process /path/to/case --execution mpi
./mfc-post process /path/to/case --index-start 0 --index-stop 100
./mfc-post render /path/to/case \
  --selected-times-us 5.0 \
  --fields temperature,OH,NC12H26,O2,phi,alpha_liq \
  --no-mp4 --out-dir /path/to/render_clean
./mfc-post render /path/to/case \
  --selected-times-us 5.0 --fields temperature \
  --temperature-mask nonliquid \
  --no-mp4 --out-dir /path/to/render_nonliquid
./mfc-post render /path/to/case \
  --time-range-us 0,10 --stride 2 --fields temperature \
  --overlay temperature,phi \
  --no-mp4 --out-dir /path/to/render_clean_overlay
./mfc-post analyze /path/to/case --selected-times-us 0,5,10 \
  --out-dir /path/to/analysis
mpiexec -n 8 ./mfc-post analyze /path/to/case --time-range-us 0,100 \
  --stride 2 --execution mpi --out-dir /path/to/mpi-analysis
./mfc-post plot /path/to/analysis --plot-set all
./mfc-post plot /path/to/analysis/scalar_timeseries.csv \
  --fields max_valid_gas_temperature_K,integrated_rhoY_OH \
  --out-dir /path/to/distinct-trends
```

`analyze`, `plot`, and `render` reject a nonempty output directory by default.
Pass `--overwrite` to replace that command's known outputs without deleting
unrelated files from the directory.

Inspection inventories each MFC output family independently. It never combines
or silently deduplicates source families. Canonical raw conservative `Field`
objects remain unchanged by processing.

The JSON document uses schema identifier `mfc-post.inspect/v1` and contains:

```text
schema_version
case
metadata_sources[]
run_metadata
equation_layout
sources[]
timeline{source_family: timeline}
conflicts[]
recommendation
warnings[]
```

The recommendation is computed from field completeness, precision, spatial
resolution, temporal coverage, cadence, and preservation of raw state. It is
not a fixed source-family priority.

`--index-start` is inclusive and `--index-stop` is exclusive. Processing writes
`results.json`, `metrics.csv`, `timeline.json`, `quality.json`, and
`provenance.json`; rank 0 is the only writer. MPI execution is optional and
requires `mpi4py`. Automatic execution stays serial unless it detects an MPI
communicator with more than one rank.

The processing foundation computes liquid mass, dense-liquid measure and
equivalent diameter, raw species inventories, closure residuals, and invalid
cell counts. For Model 3 it also reconstructs partition-local physical state
using the solver's conservative limiter, mixture EOS, chemistry-gas density,
bounded species mass fractions, molecular weight, and chemistry temperature.
It reports physical ranges and centralized mask counts; every clipping,
normalization, and derivation is recorded in provenance. If a copied production
run no longer contains a resolvable mechanism path, `--mechanism` and `--phase`
must identify the exact mechanism rather than assuming a species layout.

Clean rendering chooses nearest requested saves or an inclusive physical-time
range followed by stride. It writes one subdirectory per field plus
`manifest.json` and `provenance.json`; it never computes scalar history, trends,
or MP4 output. Static PNGs contain only a concise title, micrometer axes, and a
colorbar. All frames for a field share batch-wide color limits. Each frame has
a thin `alpha_liq = 0.5` contour; masked gas-field pixels reveal a pale
reconstructed-liquid-fraction underlay instead of an unexplained gray blob.
Temperature is
chemistry-clipped and chemistry-valid/gas-dominated masked; species images use
gas-phase `Y_k`, and `phi` uses the configured mechanism molecular weights.
This strict temperature policy remains the default through
`--temperature-mask strict_gas`. For droplet deformation and near-wake views,
`--temperature-mask nonliquid` plots finite reconstructed temperature wherever
`alpha_liq <= 0.5`; species masks are unchanged. Temperature directories and
filenames include `strict_gas` or `nonliquid`, and per-frame plotted/masked cell
counts and the exact policy are recorded in the manifest and provenance.
`--overlay temperature,phi` adds clean equivalence-ratio contour lines with
default level 1.0. Pass `--overlay-levels 0.5,1.0,2.0` only when multiple
equivalence-ratio contours are wanted.

Scalar analysis writes `scalar_timeseries.csv`, `quality.json`, and
`provenance.json`. Raw conservative species inventories are integrated using
the actual cell measures; temperature, mass-fraction, hot-region, combustible,
and overlap diagnostics use the recorded valid-gas policy. Presentation
columns include gas-mass-weighted species fractions, liquid area and `A/A0`,
and separate liquid, vapor, and total NC12H26 inventories. `A0` is the first
chronological state in the selected analysis, so a full-history run should
include the initial save. MPI runs assign
whole states to workers when temporal concurrency is favorable and otherwise
reduce spatial partitions. Rank 0 orders and writes the sole history.

Trend plotting reads only `scalar_timeseries.csv`; it does not rediscover the
case or open `p_all`. The default/all set writes stable clean temperature,
product, radical, hot-region, near-stoichiometric, liquid-area-ratio, and
dodecane-inventory PNG names. Product and radical presentation curves are
gas-mass-weighted mass fractions; conservative integrated `rhoY` fields remain
available through `--fields`.
`--plot-set` also accepts `species`, `thermal`, or `mixing`; `--fields` writes
individual requested CSV series. The manifest records the CSV hash and exact
series used by each PNG.
