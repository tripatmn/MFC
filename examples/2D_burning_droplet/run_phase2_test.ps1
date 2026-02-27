# PowerShell script for Phase 2 testing on Windows

Write-Host "=========================================="
Write-Host "Phase 2 Multiphase Chemistry Coupling Test"
Write-Host "=========================================="
Write-Host ""

# Build first
Write-Host "Step 1: Building MFC..."
& ./mfc.sh build -t pre_process simulation -j 4

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Running Phase 2 validation test..."
Write-Host ""

# Run test with diffusion only (safer for initial test)
Write-Host "Test 1: Diffusion only (no reactions) - Quick validation"
& ./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py `
    -t pre_process simulation -j 4 `
    -- --no-reactions --steps 50

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Test 1 PASSED: Diffusion test completed"
} else {
    Write-Host ""
    Write-Host "❌ Test 1 FAILED: Check output above"
    exit 1
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Phase 2 test completed successfully!"
Write-Host "=========================================="
Write-Host ""
Write-Host "Check output files in: silo_hdf5/"
Write-Host ""
