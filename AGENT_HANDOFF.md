# Agent Handoff: Silicon Photonics Simulation Framework

## Project purpose

We are building an open-source silicon photonics device-physics and simulation framework step by step.

The goal is not only to generate a simulation framework, but to create a learning environment for integrated photonic device simulation practice:

- layout-driven design with gdsfactory
- shared specs between layout and simulation
- analytic estimates before numerical simulation
- eigenmode / MPB / Meep workflows
- convergence testing
- field-profile and polarization diagnostics
- wavelength sweeps and group-index extraction
- compact-model extraction and device-level metrics
- disciplined Git-based engineering workflow

Current broad focus:

> Build a practical, layout-aware silicon photonics simulation framework and use it to move from waveguide mode physics to ring-resonator compact modeling.

---

## User learning context

The user already understands system-level silicon photonic links, link budgets, insertion loss, bandwidth, energy/bit, modulation formats, and packaging-level issues.

The user is learning lower-level integrated device physics and hands-on simulation skill.

Teaching style requested:

1. Start from physical intuition.
2. Explain only the required theory.
3. Connect physics to design knobs and performance metrics.
4. Show how to simulate.
5. Point out misleading results and common mistakes.
6. Give small concrete exercises.
7. Give professional checkpoints and short quizzes.

Important habit:

> Quiz the user after completing each major step.

The user prefers going slowly and understanding the code before implementing blindly.

---

## Environment

Repo:

```text
/Users/blow/siphotonics-framework
```

Conda environment:

```bash
conda activate siphotonics-clean
```

Expected Python:

```bash
which python
# /Users/blow/miniconda3/envs/siphotonics-clean/bin/python
```

Confirmed working tools:

- Python 3.11
- Meep 1.33.0
- gdsfactory 9.41.0
- SAX 0.17.0
- pytest
- VS Code
- KLayout separately

Health check:

```bash
cd /Users/blow/siphotonics-framework
conda activate siphotonics-clean
python -c "import meep as mp; print('meep', mp.__version__)"
python -c "import gdsfactory as gf; print('gdsfactory', gf.__version__)"
python -c "import sax; print('sax', sax.__version__)"
pytest
```

VS Code note:

If imports like `meep`, `numpy`, `scipy`, or `matplotlib` are grayed out or marked missing, VS Code/Pylance is probably using the wrong interpreter. Select:

```text
/Users/blow/miniconda3/envs/siphotonics-clean/bin/python
```

using:

```text
Cmd + Shift + P -> Python: Select Interpreter
```

Then reload VS Code.

---

## Current repo structure

```text
siphotonics-framework/
  AGENT_HANDOFF.md
  README.md
  pyproject.toml
  .gitignore

  src/
    __init__.py

    pdk/
      __init__.py
      materials.py
      layers.py
      cross_sections.py
      specs.py

    devices/
      __init__.py
      straight.py

    simulation/
      __init__.py
      waveguide_mode.py
      waveguide_mode_numeric.py

    compact_models/
      __init__.py
      ring.py

  tests/
    test_pdk.py
    test_waveguide_mode.py
    test_ring.py

  data/
    sweeps/
    fields/
    sparameters/

  results/
    figures/

  notebooks/
```

Generated files such as `.gds`, `.csv`, `.png`, `.npz`, `.h5`, etc. are generally ignored by Git unless deliberately released.

---

## Files and what they do

### `src/pdk/materials.py`

Defines simple SOI material constants:

```python
SI_N_1550 = 3.476
SIO2_N_1550 = 1.444
AIR_N_1550 = 1.0
WAVELENGTH_UM = 1.55
THICKNESS_SI_UM = 0.22
```

These are simplified constant-index values at 1550 nm.

Important caveat:

> Current wavelength/group-index sweeps keep these material indices fixed unless explicitly changed. Therefore the current group-index estimate captures waveguide dispersion only, not material dispersion.

---

### `src/pdk/layers.py`

Defines GDS layers:

```python
WG = (1, 0)
SLAB = (2, 0)
PORT = (1, 10)
TEXT = (10, 0)
```

---

### `src/pdk/cross_sections.py`

Defines the gdsfactory strip waveguide cross section.

Important distinction:

- this defines layout width and GDS layer
- it does not define optical thickness, refractive index, wavelength, or mode

---

### `src/pdk/specs.py`

Defines shared design intent:

```python
StripWaveguideSpec(width_um=0.5, thickness_um=0.22, wavelength_um=1.55)
```

Purpose:

> Prevent layout and simulation from silently using different waveguide dimensions.

---

### `src/devices/straight.py`

Defines a reusable gdsfactory straight waveguide PCell.

Run:

```bash
python -m src.devices.straight
```

Generates:

```text
straight_waveguide.gds
```

The GDS is generated output and should normally remain untracked.

Concepts already covered with the user:

- gdsfactory components have local coordinates
- built-in components generate ports automatically
- ports are used to connect references in larger layouts
- GDS stores top-view mask geometry, not the full optical simulation stack
- layout object and simulation object are related but distinct

---

### `src/simulation/waveguide_mode.py`

Implements analytic Effective Index Method, EIM.

Physical model:

1. Solve vertical symmetric TE0 slab:
   - 220 nm Si in oxide
   - gives vertical slab effective index

2. Use that vertical effective index as the lateral core index:
   - 500 nm lateral slab in oxide
   - gives approximate rectangular waveguide effective index

Representative values:

```text
EIM vertical slab n_eff ~= 2.8478
EIM rectangular waveguide n_eff ~= 2.6292
```

The file also sweeps width and saves:

```text
data/sweeps/waveguide_width_sweep_eim.csv
results/figures/waveguide_width_sweep_eim.png
```

Key teaching point:

> EIM is a fast sanity estimate, not the final professional full-vector result.

---

### `src/simulation/waveguide_mode_numeric.py`

Numerical MPB/Meep waveguide mode simulation and diagnostics.

Current capabilities:

- defines numerical simulation problem
- builds Meep material and geometry objects
- computes MPB candidate `n_eff`
- suppresses verbose MPB output
- runs resolution convergence sweep
- runs padding/domain convergence sweep
- runs band diagnostic
- extracts field profiles
- saves field arrays to `.npz`
- plots total `|E|^2`
- plots `|Ex|^2`, `|Ey|^2`, `|Ez|^2`
- computes electric-field component fractions
- runs resolution + polarization sweep
- runs padding + polarization sweep
- plots padding field comparison
- runs wavelength sweep
- estimates group index from `n_eff(lambda)`
- saves CSV outputs and plots

Coordinate convention used in MPB mode solving:

```text
x = propagation direction
y = horizontal waveguide-width direction
z = vertical thickness direction
```

For this convention:

```text
TE-like mode -> dominant Ey component
TM-like mode -> dominant Ez component
```

---

## Waveguide mode validation status

Device under study:

```text
500 nm x 220 nm SOI strip waveguide in oxide at 1550 nm
```

### EIM reference

```text
EIM vertical slab n_eff ~= 2.8478
EIM rectangular waveguide n_eff ~= 2.6292
```

EIM is used as a sanity estimate, not the final result.

### MPB band diagnostic

Earlier diagnostic at 1550 nm:

```text
band 1 -> n_eff ~= 2.4355, ok
band 2 -> n_eff ~= 1.7629, ok
band 3 -> n_eff ~= 1.4893, ok
band 4 -> no root found
```

Interpretation:

- band 1 is the strongest core-guided candidate
- band 2 is weaker
- band 3 is close to oxide index and suspicious as a useful guided mode
- band 4 did not have a root within the current search method

### Field diagnostics

Completed MPB band 1 field diagnostics:

- saved total `|E|^2` field plot
- saved `Ex`, `Ey`, `Ez` component plot
- computed component energy fractions
- `Ey` dominates, supporting TE-like classification
- field is centered on the silicon core and decays into oxide
- band 1 is a plausible TE-like core-guided mode candidate

### Resolution + polarization sweep

Completed resolution + polarization sweep for MPB band 1:

```text
resolution   n_eff      Ey fraction   classification
30 px/um     2.434596   ~0.752        TE-like
40 px/um     2.435537   ~0.752        TE-like
50 px/um     2.444373   ~0.752        TE-like
60 px/um     2.442511   ~0.752        TE-like
70 px/um     2.442548   ~0.752        TE-like
80 px/um     2.443276   ~0.752        TE-like
```

Interpretation:

- `Ey` remains dominant across resolution
- classification remains TE-like across resolution
- higher-resolution values cluster around approximately 2.443
- this improves confidence that band 1 is consistently the same TE-like mode across resolution

### Padding/domain diagnostics

Updated numerical diagnostics to use 70 px/um as the base resolution.

Padding field comparison at 70 px/um shows:

- field remains core-confined
- field remains centered on the silicon core
- no obvious boundary/domain-localized mode shape appears
- padding 1.5-3.0 um gives much more stable `n_eff`

Representative result at 70 px/um:

```text
padding 1.5 um -> n_eff ~= 2.4425
padding 2.0 um -> n_eff ~= 2.4425
padding 2.5 um -> n_eff ~= 2.4444
padding 3.0 um -> n_eff ~= 2.4445
```

Current engineering estimate:

```text
MPB band 1 TE-like mode: n_eff ~= 2.444
```

Current caveat:

- This is now a reasonable engineering estimate, not just a rough candidate.
- It is still not a final benchmark-validated value because it has not been compared against an independent trusted mode solver or reference data.
- Remaining numerical spread across padding is about 0.08%.

---

## Wavelength sweep and group index

Added wavelength sweep for the MPB band 1 TE-like mode.

Purpose:

- compute `n_eff(lambda)` around 1550 nm
- estimate `dn_eff/dlambda`
- estimate group index using:

```text
n_g = n_eff - lambda dn_eff/dlambda
```

Important caveat:

- material indices are currently fixed at their 1550 nm values
- therefore the current group-index estimate includes waveguide dispersion only
- material dispersion from `n_Si(lambda)` and `n_SiO2(lambda)` has not been added yet

Teaching point covered:

```text
n_eff -> phase at a wavelength
n_g   -> phase slope, delay, FSR, resonance spacing
```

The user understands that if `dn_eff/dlambda` is negative, then `n_g > n_eff`.

This group-index result is good enough for a first compact-model connection, but should not yet be treated as a fully material-dispersive group index.

---

## `src/compact_models/ring.py`

Added first ring compact-model utility.

Current capabilities:

- defines `RingResonatorSpec`
- estimates ring round-trip length
- estimates FSR from group index
- generates simple all-pass through-port spectrum
- saves ring spectrum CSV and plot
- extracts ring resonance metrics:
  - resonance wavelength
  - mean FSR from adjacent dips
  - extinction ratio
  - linewidth
  - loaded Q
- runs coupling-power sweep around critical coupling
- saves coupling sweep CSV
- plots extinction ratio and loaded Q versus coupling
- plots linewidth versus coupling
- plots minimum through-transmission versus coupling
- plots representative spectra versus coupling

Compact-model chain now demonstrated:

```text
MPB waveguide mode
    ↓
n_eff(lambda)
    ↓
group index
    ↓
ring FSR
    ↓
all-pass ring spectrum
    ↓
resonance metrics
    ↓
coupling-dependent extinction, linewidth, and loaded Q

## Tests

Current tests include:

```text
tests/test_pdk.py
tests/test_waveguide_mode.py
tests/test_ring.py
```

They check:

### PDK tests

- silicon index > oxide index
- oxide index > 1
- SOI thickness is 0.22 um
- WG layer is `(1, 0)`
- `StripWaveguideSpec` defaults are correct

### Waveguide mode tests

- EIM `n_eff` lies between cladding and core index
- EIM `n_eff` increases with waveguide width
- invalid core/cladding ordering raises `ValueError`

### Ring tests

- ring round-trip length is positive
- FSR is positive
- larger radius reduces FSR
- larger group index reduces FSR
- wavelength grid has expected length
- all-pass through transmission stays bounded between 0 and 1
- spectrum varies with wavelength
- invalid coupling values raise errors
- invalid loss values raise errors
- resonance metric extraction finds resonances
- extracted extinction ratio, FSR, linewidth, and loaded Q are positive

Added ring Q decomposition:
- estimates intrinsic Q from round-trip loss
- estimates coupling Q from bus-ring coupling
- estimates analytic loaded Q from 1/Q_loaded = 1/Q_intrinsic + 1/Q_coupling
- compares analytic loaded Q against spectrum-extracted loaded Q in the coupling sweep
- confirms loaded Q decreases as coupling increases

Added ring Q decomposition and comparison diagnostics:
- estimates intrinsic Q from round-trip loss
- estimates coupling Q from bus-ring coupling
- estimates analytic loaded Q using reciprocal-Q addition
- keeps spectrum-extracted loaded Q from linewidth
- compares analytic loaded Q to spectrum-extracted loaded Q versus coupling
- confirms stronger coupling lowers coupling Q and loaded Q

Added add-drop ring compact model:
- computes through-port and drop-port spectra
- saves add-drop spectrum CSV and plot
- added tests for bounded through/drop power and resonant drop peaks
- demonstrated that through port dips and drop port peaks occur at resonance

Added add-drop ring compact model and metrics:
- computes through-port and drop-port spectra
- saves add-drop spectrum CSV and plot
- extracts drop peak wavelength, max drop power, drop insertion loss, through extinction, and mean FSR
- added tests for bounded through/drop power, resonant drop peaks, and add-drop metric extraction


Run:

```bash
pytest
```

The tests are guardrails. They do not prove the full numerical simulation is correct, but they protect important assumptions and trends.

---

## Commands to resume work

```bash
cd /Users/blow/siphotonics-framework
conda activate siphotonics-clean

pytest
python -m src.devices.straight
python -m src.simulation.waveguide_mode
python -m src.simulation.waveguide_mode_numeric
python -m src.compact_models.ring
git status
```

Useful output files to inspect:

```bash
cat data/sweeps/waveguide_mpb_resolution_polarization_sweep.csv
cat data/sweeps/waveguide_mpb_padding_polarization_sweep.csv
cat data/sweeps/waveguide_mpb_wavelength_sweep.csv
cat data/sweeps/ring_all_pass_metrics.csv
```

Useful plots:

```bash
open results/figures/waveguide_mpb_band1_field.png
open results/figures/waveguide_mpb_band1_components.png
open results/figures/waveguide_mpb_padding_field_comparison.png
open results/figures/waveguide_mpb_wavelength_sweep.png
open results/figures/ring_all_pass_spectrum.png
```

---

## Current status

We have a functioning mini-framework:

```text
shared spec
  ↓
layout generation
  ↓
analytic EIM estimate
  ↓
numerical MPB mode solve
  ↓
mode validation diagnostics
  ↓
wavelength sweep and group-index estimate
  ↓
ring FSR compact model
  ↓
all-pass ring spectrum
  ↓
ring resonance metrics
  ↓
tests
  ↓
Git commits
```

Current best waveguide statement:

> MPB band 1 is a plausible TE-like, core-guided mode with engineering-estimate n_eff ~= 2.444 for the 500 nm x 220 nm SOI strip waveguide in oxide at 1550 nm.

Current best compact-model statement:

> The framework now uses the waveguide group-index workflow to estimate ring FSR, generate an all-pass ring spectrum, and extract resonance-level metrics including FSR, extinction ratio, linewidth, and loaded Q.

---

## Current caveats

1. Waveguide `n_eff` has not been benchmarked against an independent trusted mode solver.
2. Current group index includes waveguide dispersion only; material dispersion is not implemented.
3. Ring model is an all-pass model only; no drop port yet.
4. Ring coupling is wavelength independent.
5. Ring propagation loss is represented only as round-trip power loss, not yet derived from waveguide loss in dB/cm.
6. Bend loss is not modeled.
7. Backscattering, resonance splitting, thermal tuning, nonlinear effects, and fabrication variation are not modeled.
8. Field confinement fraction in silicon has not yet been quantified, only visually inspected.
9. S-parameter extraction has not yet started.

---

## Current learning checkpoint

The user has worked through:

1. layout versus simulation separation
2. shared design specs
3. EIM effective-index approximation
4. MPB numerical mode solving
5. resolution convergence
6. padding/domain convergence
7. band diagnostics
8. field-profile inspection
9. polarization/component diagnostics
10. wavelength sweep and group-index estimate
11. ring FSR estimate
12. all-pass ring spectrum generation
13. ring resonance metric extraction

Important conceptual corrections already covered:

- padding is physical cladding-domain size, not mesh resolution
- resolution is pixels per micron
- `n_eff` controls phase at a wavelength
- `n_g` controls phase slope, delay, and FSR
- `n_eff` alone is not enough for ring FSR
- all-pass ring through-port dips are due to destructive interference at resonance
- extinction ratio measures through-port dip contrast, not simply "light entering the ring"
- linewidth measures the width of a resonance
- loaded Q increases as linewidth decreases
- current group index is waveguide-only because material dispersion is not implemented yet

---

## Next recommended technical step

Next step after the break:

Next step after the break:

> Add intrinsic Q, coupling Q, and loaded Q decomposition for the all-pass ring model.

Goal:

Explain the coupling sweep using cavity lifetime/Q language:

- intrinsic Q comes from internal round-trip loss
- coupling Q comes from energy leaving through the bus
- loaded Q combines both loss channels
- stronger coupling lowers coupling Q and therefore lowers loaded Q

Expected learning:

- why loaded Q decreases as coupling increases
- why critical coupling maximizes extinction
- how intrinsic loss and coupling loss determine ring behavior

---

## End-of-session Git habit

At the end of each session:

```bash
pytest
git status
git add AGENT_HANDOFF.md README.md pyproject.toml src tests
git commit -m "Update framework progress"
git status
```

Generated data and figures are normally ignored unless deliberately released.

Update this handoff file every session with:

- what changed
- current numerical results
- unresolved issues
- next recommended step
- quiz topics covered
