# Agent Handoff: Silicon Photonics Simulation Framework

# Project purpose

We are building an open-source silicon photonics device-physics and simulation framework step by step.

The goal is not only to generate a simulations framework, but to create a learning enviroment for  integrated photonic device simulation practice:

- layout-driven design with gdsfactory
- shared specs between layout and simulation
- analytic estimates before numerical simulation
- eigenmode / MPB / Meep workflows
- convergence testing
- S-parameter and compact-model extraction later
- disciplined Git-based engineering workflow

Current focus:

> Modes and effective index of a 500 nm × 220 nm SOI strip waveguide in oxide at 1550 nm.

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

  tests/
    test_pdk.py
    test_waveguide_mode.py

  data/
    sweeps/
    fields/
    sparameters/

  results/
    figures/

  notebooks/
```

Generated files such as `.gds`, `.csv`, `.png`, `.h5`, etc. are generally ignored by Git unless deliberately released.

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
EIM vertical slab n_eff ≈ 2.8478
EIM rectangular waveguide n_eff ≈ 2.6292
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

Numerical MPB/Meep scaffold and candidate eigenmode estimate.

Current capabilities:

- defines numerical simulation problem
- builds Meep material and geometry objects
- computes MPB candidate `n_eff`
- suppresses verbose MPB output
- runs resolution convergence sweep
- runs padding convergence sweep
- runs band diagnostic
- saves CSV outputs
- optionally generates convergence plots if plotting functions have been added

Current MPB candidate:

```text
band 1 n_eff ≈ 2.4355
```

Current interpretation:

> MPB gives a plausible candidate effective index around 2.44, but this is not yet a fully validated TE0 result.

Why not fully validated yet:

- resolution convergence is not clean
- padding convergence is not clean
- mode identity is not fully confirmed
- field profile has not been inspected
- TE-like polarization has not been verified

Current resolution sweep:

```text
20 px/um → 2.426751
30 px/um → 2.434596
40 px/um → 2.435537
50 px/um → 2.444373
```

Current padding sweep:

```text
1.0 um → 2.443246
1.5 um → 2.435537
2.0 um → 2.435154
2.5 um → 2.442268
```

Current band diagnostic:

```text
band 1 → 2.435537, ok
band 2 → 1.762858, ok
band 3 → 1.489270, ok
band 4 → no root found
```

Interpretation:

- band 1 is most likely the strongest core-guided mode
- band 2 is a weaker candidate
- band 3 is suspicious because it is close to the oxide index
- band 4 did not have a root within the current search method

Important concepts already discussed:

- `resolution` = mesh points per micron
- `padding` = physical cladding region around the waveguide
- convergence means result stabilizes as numerical settings improve
- MPB bands are eigenmode branches, not automatically “the mode we want”
- “ok” in the band diagnostic means a root was found, not necessarily that it is the correct physical mode
- field profile inspection is required to verify mode identity
- staircasing = grid approximation of material boundaries

---

## Tests

Current tests:

```text
tests/test_pdk.py
tests/test_waveguide_mode.py
```

They check:

- silicon index > oxide index
- oxide index > 1
- SOI thickness is 0.22 µm
- WG layer is `(1, 0)`
- `StripWaveguideSpec` defaults are correct
- EIM `n_eff` lies between cladding and core index
- EIM `n_eff` increases with waveguide width
- invalid core/cladding ordering raises `ValueError`

Run:

```bash
pytest
```

The tests are guardrails. They do not prove the full numerical simulation is correct.

---

## Commands to resume work

```bash
cd /Users/blow/siphotonics-framework
conda activate siphotonics-clean

python -m src.devices.straight
python -m src.simulation.waveguide_mode
python -m src.simulation.waveguide_mode_numeric
pytest
git status
```

Inspect generated numerical outputs:

```bash
cat data/sweeps/waveguide_mpb_resolution_sweep.csv
cat data/sweeps/waveguide_mpb_padding_sweep.csv
cat data/sweeps/waveguide_mpb_band_diagnostic.csv
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
numerical MPB candidate solve
  ↓
resolution / padding / band diagnostics
  ↓
CSV / plot outputs
  ↓
tests
  ↓
Git commits
```

The current numerical result should be phrased carefully:

> MPB gives a plausible candidate n_eff around 2.44 for the strongest guided mode.

Do not yet say:

> The validated TE0 n_eff is 2.44.

---

## Next technical step

The next major step should be:

> Add field-profile diagnostics for the MPB candidate modes.

Goal:

Confirm whether band 1 is actually the fundamental TE-like core-confined mode.

Questions to answer:

1. Is the field concentrated in the silicon core?
2. Is the mode TE-like or TM-like?
3. Is band 1 the fundamental mode?
4. Does the field shape remain consistent across resolution and padding sweeps?
5. Are bands 2 and 3 weakly guided, cladding-like, or domain modes?

Potential implementation direction:

- Use MPB/Meep field extraction for the selected band.
- Save field profile arrays to `data/fields/`.
- Save field plots to `results/figures/`.
- Plot core outline on top of field intensity.
- Eventually compute confinement fraction in the silicon core.

---

## Next teaching checkpoint

Before moving beyond waveguide modes, the user should be able to explain:

1. Difference between layout width and simulation cross section.
2. Difference between EIM and numerical eigenmode solving.
3. Meaning of effective index.
4. Why `n_eff` should lie between cladding and core index for guided dielectric modes.
5. Difference between resolution and padding.
6. Why convergence is required.
7. What MPB bands are.
8. Why field profile inspection is required for mode identity.
9. Why the current MPB result is plausible but not fully validated.
10. How this waveguide mode work will later feed bends, couplers, rings, MZIs, and compact models.

---

## Git habits

At the end of each session:

```bash
pytest
git status
git add AGENT_HANDOFF.md README.md pyproject.toml src tests results/*.md
git commit -m "Update framework handoff and progress"
git status
```

Generated data and figures are normally ignored unless explicitly released.

Update this handoff file every session with:

- what changed
- current numerical results
- unresolved issues
- next recommended step
- quiz topics covered

---

## Last known unresolved issues

1. MPB candidate `n_eff` is plausible but not validated.
2. Resolution convergence has a jump at 50 px/µm.
3. Padding convergence has a jump at 2.5 µm.
4. Band identity is only partially diagnosed.
5. Field profiles have not yet been extracted.
6. TE-like polarization has not yet been confirmed.
7. Numerical mode tracking should eventually be based on field identity, not just band number.
