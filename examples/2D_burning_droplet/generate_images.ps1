# PowerShell script to run test and generate images

Write-Host "=========================================="
Write-Host "Phase 2 Test + Image Generation"
Write-Host "=========================================="
Write-Host ""

# Step 1: Run the test
Write-Host "Step 1: Running Phase 2 validation test..."
& ./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py `
    -t pre_process simulation -j 4 `
    -- --no-reactions --steps 50

if ($LASTEXITCODE -ne 0) {
    Write-Host "Test failed!"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Generating visualization images..."
Write-Host ""

# Step 2: Generate images
# Try python3 first, then python
$python_cmd = $null
if (Get-Command python3 -ErrorAction SilentlyContinue) {
    $python_cmd = "python3"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python_cmd = "python"
} else {
    Write-Host "Python not found. Please install Python to generate images."
    Write-Host "You can run the visualization script manually later:"
    Write-Host "  python examples/2D_burning_droplet/visualize_phase2_simple.py"
    exit 0
}

& $python_cmd examples/2D_burning_droplet/visualize_phase2_simple.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Images generated successfully!"
    Write-Host ""
    Write-Host "Check images in: examples/2D_burning_droplet/figures/phase2/"
} else {
    Write-Host ""
    Write-Host "⚠️  Visualization script had issues, but test passed."
    Write-Host "You can try running manually:"
    Write-Host "  python examples/2D_burning_droplet/visualize_phase2_simple.py"
}

Write-Host ""
Write-Host "=========================================="
