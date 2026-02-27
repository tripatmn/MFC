@echo off
REM Run Phase 2 test and visualization. Run from MFC root:
REM   examples\2D_burning_droplet\run_test_and_visualize.cmd

cd /d "%~dp0\..\.."
if not exist "mfc.sh" (
    echo Run this from MFC root. Current dir: %CD%
    exit /b 1
)

echo ==============================================
echo Phase 2: Run test + visualize
echo ==============================================
echo.

echo Step 1: Running Phase 2 validation (100 steps)...
call mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4 -- --steps 100
if errorlevel 1 (
    echo Simulation failed.
    exit /b 1
)
echo.

echo Step 2: Generating plots...
where python >nul 2>&1
if errorlevel 1 (
    where python3 >nul 2>&1
    if errorlevel 1 (
        echo Python not found. To generate plots, run:
        echo   python examples/2D_burning_droplet/visualize_phase2_simple.py
        exit /b 0
    )
    python3 examples/2D_burning_droplet/visualize_phase2_simple.py
) else (
    python examples/2D_burning_droplet/visualize_phase2_simple.py
)
echo.
echo Done. Check: examples/2D_burning_droplet/figures/phase2/
echo ==============================================
