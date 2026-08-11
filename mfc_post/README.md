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
  --fields 'temperature,Y[NC12H26],Y[OH],Y[CO2],Y[H2O]' \
  --no-zoom --no-mp4 --skip-scalars --skip-trends \
  --out-dir /path/to/distinct-render-output
```

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

Selected-state rendering chooses the nearest available physical save and reads
only those states. It writes full-domain PNGs plus `manifest.json`,
`provenance.json`, and `frames.csv`; it never computes scalar history, trends,
or MP4 output. Temperature uses the chemistry-clipped field with the
chemistry-valid and gas-dominated masks, while the manifest separately reports
the raw reconstructed temperature range and clipping count. Species images use
gas-phase `Y_k`, never conservative `rhoY_k`. A nonempty output directory is
rejected so separate serial and MPI validations cannot overwrite one another.
