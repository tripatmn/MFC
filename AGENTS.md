# AGENTS.md — MFC Coupled Vaporization + Combustion (Shock–Droplet)

## 0) Hard Rules
- Work only in this local repo.
- Do not push to GitHub unless I explicitly ask.
- Keep diffs minimal; avoid unrelated refactors.
- If unsure: propose a design first (no edits).
- After ANY code change: build + run smoke tests (Section 6) and report PASS/FAIL.

## 1) Project Goal (What we are building)
Implement coupled physics in MFC for:
Shock wave in oxidizing gas (e.g., O2/N2/...) impacts a liquid dodecane droplet:
1) shock–droplet interaction
2) droplet vaporization (fuel vapor generation)
3) fuel vapor mixes with oxidizer
4) gas-phase combustion

## 2) Current Code Reality (Do not “wish” it different)
- Phase change/vaporization logic exists but is multi-fluid oriented (liquid/vapor fluids).
- Chemistry exists and uses a global species block (single-mixture style).
- We are bridging this mismatch with minimal architectural changes.

## 3) Architectural Constraints (Non-negotiable unless I approve)
- Liquid remains a separate fluid.
- Species are global (do NOT redesign to per-fluid species arrays).
- Chemistry must operate on a designated gas fluid density:
  - `chem_gas_fluid_id` selects which fluid provides rho_g for Yk = rhoYk / rho_g.
- No silent species renormalization hacks. Clamp negatives only unless explicitly instructed.
- Conservation matters:
  - Vaporization mass transfer must become a source to fuel vapor species and an energy-consistent coupling.
  - Avoid “magic” fixes that break mass/energy.

## 4) Development Strategy (How to work)
- Small commits:
  1) plumbing-only
  2) validate
  3) then add physics
- Prefer adding new parameters / hooks behind flags (default off) so behavior is unchanged unless enabled.
- Keep a short “smoke” run path that exits on its own (no Ctrl+C dependence).

## 5) Coupling Roadmap (Implementation ladder)
Milestone A — Infrastructure:
- Maintain short-run smoke cases for droplet and reactive examples.
- Ensure chemistry uses rho_g from `chem_gas_fluid_id` (already implemented).

Milestone B — Species-source plumbing:
- Add a minimal, well-gated species source hook in RHS (reactions off initially).
- Validate species injection changes outputs as expected.

Milestone C — Vaporization → fuel species:
- Replace dummy species source with mass transfer derived from phase change.
- Verify fuel vapor species increases near droplet region with reactions OFF.

Milestone D — Turn on combustion:
- Enable reactions; verify O2 consumption + temperature rise + product formation.

Milestone E — New case:
- Create a 2D burning droplet case based on 2D shock–droplet scaffold.

## 6) Commands (Must run after edits)

Build:
bash ./mfc.sh build -j 8

Droplet smoke test (must exit after ~21 steps):
bash ./mfc.sh run examples/2D_shockdroplet_muscl_smoke/case.py -t pre_process simulation -j 4

Reactive smoke test (must exit after ~21 steps):
bash ./mfc.sh run examples/1D_reactive_shocktube_smoke/case.py -t pre_process simulation -j 4

## 7) Reporting Format (When you finish a change)
- List files changed
- Describe change in 3–6 bullets
- Build result (PASS/FAIL + key error if fail)
- Smoke tests result (PASS/FAIL + key error if fail)
- Any open questions/risks 