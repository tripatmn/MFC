# Run Phase 2 validation test and generate visualization plots.
# Run from MFC root: .\examples\2D_burning_droplet\run_test_and_visualize.ps1

$ErrorActionPreference = "Stop"
$MfcRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$MfcRoot\mfc.sh")) {
    Write-Host "Run this script from MFC root, or set MFC root. Current script: $PSScriptRoot"
    exit 1
}

Set-Location $MfcRoot

Write-Host "=============================================="
Write-Host "Phase 2: Run test + visualize"
Write-Host "=============================================="
Write-Host ""

# 1) Run simulation (creates output in examples/2D_burning_droplet)
Write-Host "Step 1: Running Phase 2 validation (100 steps, reactions + diffusion)..."
& ./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4 -- --steps 100
if ($LASTEXITCODE -ne 0) {
    Write-Host "Simulation failed (exit $LASTEXITCODE)."
    exit $LASTEXITCODE
}
Write-Host ""

# 2) Find Python and run visualization (try PATH, then Anaconda/Miniconda)
$pythonExe = $null
foreach ($name in @("python3", "python")) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) {
        try {
            $v = & $c.Source --version 2>&1
            if ($LASTEXITCODE -eq 0 -and $v -notmatch "Microsoft Store") {
                $pythonExe = $c.Source
                break
            }
        } catch {}
    }
}
if (-not $pythonExe) {
    $condaPaths = @(
        "$env:USERPROFILE\anaconda3\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:ProgramData\anaconda3\python.exe",
        "$env:ProgramData\miniconda3\python.exe"
    )
    foreach ($p in $condaPaths) {
        if (Test-Path $p) {
            $pythonExe = $p
            break
        }
    }
}
if (-not $pythonExe) {
    Write-Host "Step 2: Python not found (PATH or Anaconda). To generate plots, run from MFC root:"
    Write-Host "  & 'C:\Users\YourName\anaconda3\python.exe' examples/2D_burning_droplet/visualize_phase2_simple.py"
    Write-Host "  (use your actual Anaconda path; common: $env:USERPROFILE\anaconda3\python.exe)"
    Write-Host ""
    Write-Host "Simulation finished successfully. No images generated."
    exit 0
}

Write-Host "Step 2: Generating plots (using $pythonExe)..."
& $pythonExe examples/2D_burning_droplet/visualize_phase2_simple.py
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done. Check: examples/2D_burning_droplet/figures/phase2/"
} else {
    Write-Host "Visualization script exited with $LASTEXITCODE. You can run it manually:"
    Write-Host "  $pythonExe examples/2D_burning_droplet/visualize_phase2_simple.py"
}
Write-Host ""
Write-Host "=============================================="
