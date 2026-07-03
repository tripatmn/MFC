# Nautilus HPC Submission & Validation Guide

This document lists the steps to submit and analyze the combined stable reacting shock-droplet simulation on the **Nautilus Cluster**.

---

## 1. Directory Structure & Sync

* **Target Folder on Nautilus:**
  `/p/work1/tripatmn/Cloud/mfc_combustion/MFC_sk54_combined_bcYminus3_limiter`
  *(Do NOT modify `/p/work1/tripatmn/Cloud/mfc_combustion/MFC_shockdroplet_sk54_t2e6_guarded`)*

* **Branch:** `ag/chem-positivity-diagnostic`
* **Commit:** `a90aaf95` (Add guarded chemistry positivity limiter)

Make sure your repository on the cluster is synced to this commit:
```bash
cd /p/work1/tripatmn/Cloud/mfc_combustion/MFC_sk54_combined_bcYminus3_limiter
git fetch origin
git checkout ag/chem-positivity-diagnostic
git pull --ff-only
```

---

## 2. Load Modules & Build

Configure the environment modules and rebuild the target binaries for GPU+MPI on Nautilus:

```bash
# Load compiler modules
. ./mfc.sh load -c nautilus -t gpu

# Build the GPU + MPI target binaries from scratch to ensure correct compilation
./mfc.sh build -j 8 --gpu --mpi --scratch
```

---

## 3. Submit the First Run (t = 190 ns)

Submit only the 190 ns simulation to Slurm:

```bash
# Using the launcher script
./examples/2D_dodecane_global_reduced/submit_t190ns.sh

# Or run the CLI command directly:
./mfc.sh run examples/2D_dodecane_global_reduced/case_hpc_combined_stable_t190ns.py \
    -t pre_process simulation \
    -e batch -c nautilus -N 1 -n 2 -j 8 \
    --gpu acc --mpi --clean
```

### Monitoring the Job:
```bash
squeue -u tripatmn
```

---

## 4. Run Analysis & Report Results

Once the 190 ns job completes, run the shock-droplet analyzer on the output folder (normally created under `runs/case-hpc-combined-stable-t190ns_<timestamp>/` or similar):

```bash
python3 examples/2D_dodecane_global_reduced/analyze_shockdroplet_air_sk54.py \
    --run-dir runs/case-hpc-combined-stable-t190ns_<timestamp> \
    --out-dir runs/analysis_t190ns
```

Compile and report the 13 required metrics:
1. **Exit code** (verify `0` status in `.out` / `.err` files)
2. **Reached t_stop = 1.9e-7 s or failed**
3. **Final time & step**
4. **dt history** (extract time step trends near 1.5e-7 to end)
5. **min rhoY_NC12H26**
6. **min tracked species densities**
7. **max reconstructed temperature**
8. **max CO2, H2O, OH, HO2, H2O2**
9. **Total NaN/nonfinite count** (should be 0)
10. **Verify boundary dt collapse is gone**
11. **Verify chemistry positivity issues are gone**
12. **MPI rank count & GPU count**
13. **git status --short**

---

## 5. Next Staged Submissions

If the 190 ns validation is successful, proceed to submit the next staged run:

1. **t = 300 ns:**
   ```bash
   ./mfc.sh run examples/2D_dodecane_global_reduced/case_hpc_combined_stable_t300ns.py \
       -t pre_process simulation -e batch -c nautilus -N 1 -n 2 -j 8 --gpu acc --mpi --clean
   ```
2. **t = 500 ns:**
   ```bash
   ./mfc.sh run examples/2D_dodecane_global_reduced/case_hpc_combined_stable_t500ns.py \
       -t pre_process simulation -e batch -c nautilus -N 1 -n 2 -j 8 --gpu acc --mpi --clean
   ```
3. **t = 1.0 us:**
   ```bash
   ./mfc.sh run examples/2D_dodecane_global_reduced/case_hpc_combined_stable_t1us.py \
       -t pre_process simulation -e batch -c nautilus -N 1 -n 2 -j 8 --gpu acc --mpi --clean
   ```
