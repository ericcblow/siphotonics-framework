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

> Build a practical, layout-aware silicon photonics simulation framework and use it to move from waveguide mode physics to ring-resonator compact modeling, then toward couplers, S-parameters, compact-model extraction, and layout-driven circuit simulation.

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

Important teaching habits:

- Quiz the user after completing each major step.
- Be critical and accurate when reviewing quiz answers.
- Do not simply affirm partially correct answers.
- Correct misconceptions explicitly and explain the physical intuition behind the correction.
- Require the user to distinguish similar concepts carefully, especially:
  - `n_eff` versus `n_g`
  - intrinsic loss versus coupling loss
  - propagation loss versus bend loss
  - all-pass rings versus add-drop rings
  - cascaded rings versus directly coupled/stacked rings
  - field transfer functions versus power spectra
- The goal of quizzes is to verify understanding and intuition, not just to proceed through implementation steps.

The user prefers going slowly and understanding the code before implementing blindly.

---

## Long-term learning artifact

The user plans to eventually organize the completed learning journey into a lecture-like PowerPoint deck.

Purpose:

- consolidate the device-physics intuition learned across sessions
- turn the code framework into a teachable engineering narrative
- create a reusable study/reference tool for future silicon photonics work
- practice explaining devices from physical intuition through simulation workflow and compact-model extraction

The deck should likely be built after major modules are complete, using the handoff file, generated plots, code milestones, and quiz corrections as source material.

Potential lecture structure:

1. Waveguide fundamentals and mode solving
2. EIM versus numerical eigenmode solving
3. MPB convergence and mode validation
4. Group index and dispersion
5. Ring resonator fundamentals
6. All-pass rings, add-drop rings, and Q decomposition
7. Cascaded and stacked rings
8. Loss budgets and intrinsic Q
9. Directional coupler physics
10. S-parameter extraction and compact models
11. Layout-driven simulation workflow

This should be treated as a final learning exercise and possibly a useful reusable teaching/reference artifact.

Current lecture-draft status from the revisit session:

- Rough lecture slides have been brought up through the stacked-ring section.
- Slides now cover, at least conceptually:
  - EIM equations and parameter definitions
  - MPB as a frequency-domain eigenmode solver, not FDTD
  - MPB field plots as eigenvectors of Maxwell's operator
  - resolution and padding convergence, including staircasing and boundary/domain errors
  - wavelength sweep and numerical `dn_eff/dlambda` extraction
  - group index and the caveat that material dispersion is not yet included
  - all-pass ring resonance, FSR, linewidth, loaded Q, and critical coupling
  - loss-budget interpretation and intrinsic Q
  - add-drop spectra and coupling-balance heatmap takeaways
  - cascaded all-pass spectra
  - stacked-ring resonance splitting and detuning
- The lecture pass clarified several concepts and produced slide-ready equations, tables, and takeaways.
- The next lecture module should start with directional couplers once the corresponding simulation work begins.


---

## Environment

Repo:

```bash
cd /Users/blow/siphotonics-framework
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
- Femwell should be considered as an additional open-source FEM/eigenmode option when useful, especially for mode solving, bent/curved cross-section studies, and independent benchmarking against MPB/Meep.
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
      directional_coupler_supermodes.py

    compact_models/
      __init__.py
      ring.py
      stacked_rings.py

  tests/
    test_pdk.py
    test_waveguide_mode.py
    test_ring.py
    test_stacked_rings.py

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

### `src/pdk/layers.py`

Defines GDS layers:

```python
WG = (1, 0)
SLAB = (2, 0)
PORT = (1, 10)
TEXT = (10, 0)
```

### `src/pdk/cross_sections.py`

Defines the gdsfactory strip waveguide cross section.

Important distinction:

- this defines layout width and GDS layer
- it does not define optical thickness, refractive index, wavelength, or mode

### `src/pdk/specs.py`

Defines shared design intent:

```python
StripWaveguideSpec(width_um=0.5, thickness_um=0.22, wavelength_um=1.55)
```

Purpose:

> Prevent layout and simulation from silently using different waveguide dimensions.

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

Concepts already covered:

- gdsfactory components have local coordinates
- built-in components generate ports automatically
- ports are used to connect references in larger layouts
- GDS stores top-view mask geometry, not the full optical simulation stack
- layout object and simulation object are related but distinct

---

## Waveguide simulation status

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

### Lecture-revisit clarifications

The following slide-ready clarifications were developed during the lecture-review pass:

- EIM solves the symmetric TE slab equation by root-finding:

```text
f(n_eff) = h tan(h a) - q = 0
```

  with `h`, `q`, `beta`, `k0`, and half-dimension `a = d/2` explicitly defined for slides.
- EIM is used twice:
  - vertical slab solve gives `n_eff,vertical`
  - lateral slab solve uses `n_eff,vertical` as the lateral core index
- MPB is a frequency-domain eigenmode solver, not an FDTD time-domain propagation simulation.
- MPB solves Maxwell's eigenproblem for a given `kx`; Python then root-finds `kx` until the MPB band frequency equals the target frequency.
- MPB field diagrams come from eigenvectors of the solved mode, not from launched-propagation fields.
- Main MPB numerical pitfalls discussed for slides:
  - staircasing of high-index-contrast geometry at low resolution
  - boundaries too close to evanescent tails when padding is too small
  - mode identity switching if only band number is tracked
  - cladding-like or box modes near oxide index
  - root-finding the wrong branch
  - over-reporting digits beyond convergence support
  - inconsistent color scales or axis limits in field plots
- For lecture plots, fixed physical axis limits are preferred when comparing padding cases.

---

## Ring compact-model status: `src/compact_models/ring.py`

`ring.py` now contains the main passive ring compact-model work.

### Current capabilities

- defines `RingResonatorSpec`
- estimates ring round-trip length
- estimates FSR from group index
- converts physical loss budgets into round-trip power loss:
  - propagation loss in dB/cm
  - bend loss in dB/turn
  - coupler excess loss in dB/round trip
- estimates intrinsic Q from a loss budget
- generates simple all-pass through-port spectrum
- saves ring spectrum CSV and plot
- extracts all-pass ring resonance metrics:
  - resonance wavelength
  - mean FSR from adjacent dips
  - extinction ratio
  - linewidth
  - loaded Q
- runs coupling-power sweep around critical coupling
- estimates intrinsic Q, coupling Q, and analytic loaded Q
- compares analytic loaded Q against spectrum-extracted loaded Q
- saves coupling sweep CSV
- plots:
  - extinction ratio and loaded Q versus coupling
  - linewidth versus coupling
  - minimum through-transmission versus coupling
  - representative spectra versus coupling
  - analytic loaded Q versus spectrum-extracted loaded Q
- computes add-drop ring through-port and drop-port spectra
- extracts add-drop metrics:
  - drop peak wavelength
  - max drop power
  - drop insertion loss
  - through extinction
  - mean FSR
- sweeps add-drop input/drop coupling balance
- plots max-drop-power and insertion-loss heatmaps
- adds complex all-pass field transfer function
- cascades all-pass rings by multiplying field transfer functions
- compares:
  - one ring
  - two identical rings
  - three identical rings
  - three detuned rings
- saves cascaded spectra CSV and plot
- extracts cascade metrics including extinction ratio, linewidth, loaded Q, and FSR
- recent lecture-driven additions/requests for ring visualizations include:
  - estimating lifetime-equivalent number of round trips from loaded Q
  - plotting loaded Q and estimated round trips versus power coupling
  - plotting undercoupled, critically coupled, and overcoupled all-pass spectra as three separate subplots
  - plotting add-drop spectra along the balanced coupling diagonal where `kappa1^2 = kappa2^2`

Note: verify which of the lecture-driven plotting additions have been committed in the local repo before relying on them in tests or generated figures.

### Compact-model chain demonstrated

```text
MPB waveguide mode
    -> n_eff(lambda)
    -> group index
    -> ring FSR
    -> all-pass ring spectrum
    -> resonance metrics
    -> coupling-dependent extinction, linewidth, and loaded Q
    -> add-drop ring spectra and metrics
    -> cascaded multi-ring spectra and metrics
```

### Physical loss-budget model

Added a loss-budget bridge:

```text
propagation loss dB/cm
+ bend loss dB/turn
+ coupler excess loss dB/round trip
    -> round-trip loss dB
    -> round-trip power loss
    -> intrinsic Q
```

Example printed result from current run:

```text
propagation loss:       2.000 dB/cm
bend loss per turn:     0.050 dB/turn
coupler excess loss:    0.020 dB/round trip
round-trip loss:        0.080053 dB
round-trip power loss:  0.018264
intrinsic Q estimate:   45179.7
```

Important interpretation:

- Straight propagation loss alone is not assumed to include bend loss.
- Bend loss and coupler excess loss are separate loss-budget terms.
- For the above example, bend loss and coupler excess dominate over the straight propagation contribution.
- Bus-ring coupling loss is not part of intrinsic loss; it is an external loading channel.

### Important ring concepts covered

- larger radius decreases FSR
- larger group index decreases FSR
- `n_eff` controls resonance locations
- `n_g` controls resonance spacing
- all-pass through-port dips are caused by destructive interference at resonance
- extinction ratio measures through-port dip contrast
- linewidth measures resonance width
- smaller linewidth means larger loaded Q
- stronger bus-ring coupling broadens the resonance
- loaded Q decreases as coupling increases
- extinction peaks near critical coupling
- minimum through power occurs near critical coupling
- intrinsic Q is set by internal loss
- coupling Q is set by energy leaking into the bus
- loaded Q combines intrinsic and coupling loss channels
- add-drop rings require balancing loading into the ring and extraction into the drop bus
- field transfer functions should be cascaded before converting to power
- identical cascaded rings increase rejection
- detuned cascaded rings broaden or split the rejection feature
- linewidth for a through-port dip is measured as full width at half depth:

```text
T_half = T_min + (T_max - T_min)/2
Delta_lambda = lambda_right - lambda_left
Q_loaded = lambda_res / Delta_lambda
```

- lifetime-equivalent number of round trips can be estimated from loaded Q:

```text
N_rt ~= Q_loaded * lambda0 / (2*pi*n_g*L_rt)
```

  This is not a literal integer photon count; it is photon lifetime divided by round-trip time.
- hundreds of round trips are possible only for sufficiently high loaded Q, typically `Q_loaded` in the `1e5` to `1e6` range for the current small-radius example.
- maximum extinction and maximum Q do not occur at the same coupling. Weak coupling can give high Q but shallow dips; critical coupling gives maximum extinction but lower Q.
- add-drop heatmaps should be interpreted as coupling-balance maps: `kappa1^2` loads power into the ring and `kappa2^2` extracts power to the drop bus.
- high through extinction does not automatically imply low drop insertion loss because intrinsic ring loss can dissipate energy before extraction.

---

## Stacked-ring compact-model status: `src/compact_models/stacked_rings.py`

Added a separate module for directly coupled / stacked rings.

This is intentionally separate from `ring.py`.

### Current capabilities

- defines `TwoStackedRingSpec`
- implements normalized temporal-coupled-mode-theory model for two directly coupled rings
- computes through-port spectrum for:
  - uncoupled second ring
  - weak ring-ring coupling
  - strong ring-ring coupling
  - detuned second ring
- saves stacked-ring spectra CSV and plot
- sweeps ring-2 detuning at fixed ring-ring coupling `mu`
- saves detuning sweep CSV and plot
- adds tests for detuning units, nonnegative power, coupling-dependent spectrum changes, invalid parameters, and detuning sweep output

### Important stacked-ring concepts covered

- `mu` is a normalized ring-to-ring coupling rate.
- `mu` controls how strongly optical energy transfers between ring 1 and ring 2.
- Larger `mu` produces stronger resonance splitting.
- `mu` comes physically from evanescent overlap between nearby rings.
- Ring-to-ring coupling depends on gap, coupling arc length, waveguide width, wavelength, polarization, mode confinement, and fabrication variation.
- Detuning one ring shifts its natural resonance relative to the other ring.
- Zero detuning gives the most symmetric split resonances for identical rings.
- Detuning creates asymmetric split resonances and changes modal participation.
- A heater primarily shifts resonance wavelength through thermo-optic index change, while secondary effects can include loss, coupling changes, and thermal crosstalk.

### Cascaded rings versus stacked rings

Cascaded all-pass rings:

```text
bus -> ring 1 -> bus -> ring 2 -> bus -> ring 3
```

- independent ring responses multiply along the bus
- field transfer functions cascade
- no direct resonator-to-resonator energy exchange in the model

Stacked / directly coupled rings:

```text
bus -> ring 1 <-> ring 2
```

- rings directly exchange energy
- resonant modes hybridize
- resonance splitting appears from coupled modes

This distinction is important and should continue to be reinforced.

### Lecture-revisit clarifications for stacked rings

- If two identical rings have the same uncoupled resonance, direct coupling does not simply make one stronger shared resonance.
- Coupling lifts the degeneracy and forms two hybrid modes:
  - symmetric mode
  - antisymmetric mode
- In a simple lossless frequency-domain picture, the hybrid frequencies are approximately:

```text
omega_+ = omega0 + mu
omega_- = omega0 - mu
Delta_omega ~= 2*mu
```

- Clear splitting is visible only when the coupling rate is comparable to or larger than the resonance linewidth/decay rate.
- Detuning one ring breaks the symmetry. One hybrid mode becomes more ring-1-like and the other more ring-2-like; because only ring 1 is directly bus-coupled in the current model, detuning also changes which resonance is more visible in the through spectrum.

---

## Tests

Current tests include:

```text
tests/test_pdk.py
tests/test_waveguide_mode.py
tests/test_ring.py
tests/test_stacked_rings.py
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
- Q decomposition values are positive
- loaded Q is less than intrinsic and coupling Q
- stronger coupling reduces coupling Q and loaded Q
- coupling sweep includes Q-decomposition columns
- add-drop ring power is bounded
- add-drop ring has resonant drop peaks
- add-drop metric extraction works
- add-drop coupling-balance sweep runs
- all-pass field-derived power matches power function
- cascaded identical rings deepen the notch
- cascade metric extraction runs
- loss-budget conversions are bounded and monotonic
- higher loss budget lowers intrinsic Q
- negative loss-budget values raise errors

### Stacked-ring tests

- normalized wavelength detuning has expected sign and center
- stacked-ring through power is nonnegative
- ring-ring coupling changes the spectrum
- invalid stacked-ring parameters raise `ValueError`
- ring-2 detuning sweep returns expected spectra

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
python -m src.compact_models.stacked_rings
python -m src.simulation.directional_coupler_supermodes --solver mpb --resolution 30 --num-bands 4
python -m src.simulation.directional_coupler_supermodes --solver mpb --num-bands 4 --convergence --convergence-resolutions 30 50
git status
```

Useful output files to inspect:

```bash
cat data/sweeps/directional_coupler_gap_sweep_mpb.csv
cat data/sweeps/directional_coupler_resolution_30_vs_50.csv
cat data/sweeps/directional_coupler_design_table_mpb.csv
cat data/sweeps/waveguide_mpb_resolution_polarization_sweep.csv
cat data/sweeps/waveguide_mpb_padding_polarization_sweep.csv
cat data/sweeps/waveguide_mpb_wavelength_sweep.csv
cat data/sweeps/ring_all_pass_metrics.csv
cat data/sweeps/ring_coupling_sweep.csv
cat data/sweeps/ring_add_drop_metrics.csv
cat data/sweeps/ring_add_drop_coupling_balance.csv
cat data/sweeps/ring_cascade_metrics.csv
cat data/sweeps/two_stacked_ring_spectra.csv
cat data/sweeps/two_stacked_ring_detuning_sweep.csv
```

Useful plots:

```bash
open results/figures/directional_coupler_gap_sweep_mpb.png
open results/figures/directional_coupler_resolution_convergence.png
open results/figures/directional_coupler_fields_gap_sweep.png
open results/figures/directional_coupler_fields_gap_sweep_zoom.png
open results/figures/directional_coupler_kappa_vs_length_mpb.png
open results/figures/waveguide_mpb_band1_field.png
open results/figures/waveguide_mpb_band1_components.png
open results/figures/waveguide_mpb_padding_field_comparison.png
open results/figures/waveguide_mpb_wavelength_sweep.png

open results/figures/ring_all_pass_spectrum.png
open results/figures/ring_coupling_sweep.png
open results/figures/ring_spectra_vs_coupling.png
open results/figures/ring_loaded_q_comparison.png
open results/figures/ring_q_and_round_trips_sweep.png
open results/figures/ring_coupling_regime_spectra.png
open results/figures/ring_add_drop_spectrum.png
open results/figures/ring_add_drop_max_drop_power_heatmap.png
open results/figures/ring_add_drop_insertion_loss_heatmap.png
open results/figures/ring_add_drop_balanced_coupling_spectra.png
open results/figures/ring_cascade_spectrum.png
open results/figures/ring_loss_budget_all_pass_spectrum.png

open results/figures/two_stacked_ring_spectra.png
open results/figures/two_stacked_ring_detuning_sweep.png
```

---

---

## Directional coupler supermode status: `src/simulation/directional_coupler_supermodes.py`

A new directional-coupler supermode module has been developed to bridge abstract ring coupling coefficients to physical coupler geometry.

### Current capabilities

- Defines `DirectionalCouplerSpec` for a 500 nm x 220 nm SOI strip directional coupler.
- Defines `MpbCouplerSolveSettings` for MPB resolution, padding, band count, and root-finding settings.
- Supports two solver modes:
  - `mock`: exponential gap-dependence model for quick pipeline testing.
  - `mpb`: Meep/MPB supermode extraction for two parallel waveguides.
- Uses MPB `find_k` to extract guided supermode effective indices at 1550 nm.
- Selects the two largest distinct guided effective indices as first-pass even/odd supermode candidates.
- Computes:
  - `n_even`
  - `n_odd`
  - `delta_neff = n_even - n_odd`
  - `L_full_um = lambda / (2 delta_neff)`
  - `L_3dB_um = L_full_um / 2`
- Adds target-coupling design columns for selected `kappa^2` values.
- Generates ideal `kappa^2` versus coupler-length curves from the extracted supermode splitting.
- Generates practical design tables mapping target `kappa^2` to required coupling length.
- Adds resolution convergence mode comparing MPB resolution 30 versus 50.
- Adds field-validation plots for selected bands across several gaps, including full-view and zoomed versions.

### Important commands

Run the standard MPB gap sweep:

```bash
python -m src.simulation.directional_coupler_supermodes --solver mpb --resolution 50 --num-bands 4
```

Run fast/debug field validation:

```bash
python -m src.simulation.directional_coupler_supermodes --solver mpb --resolution 30 --num-bands 4
```

Run resolution convergence:

```bash
python -m src.simulation.directional_coupler_supermodes \
  --solver mpb \
  --num-bands 4 \
  --convergence \
  --convergence-resolutions 30 50
```

Run design-facing outputs:

```bash
python -m src.simulation.directional_coupler_supermodes \
  --solver mpb \
  --resolution 50 \
  --num-bands 4 \
  --kappa-length-plot \
  --design-table \
  --max-coupler-length 120
```

### Key numerical results: resolution 50 baseline

For two 500 nm x 220 nm SOI strip waveguides in oxide at 1550 nm, MPB resolution 50 produced approximately:

```text
gap_um   delta_neff   L_full_um   L_3dB_um
0.10     0.053810     14.40       7.20
0.15     0.031802     24.37       12.18
0.20     0.020293     38.19       19.10
0.25     0.013339     58.10       29.05
0.30     0.008547     90.68       45.34
0.40     0.003549     218.39      109.20
0.50     0.001526     507.90      253.95
```

Physical trend confirmed:

```text
larger gap -> weaker evanescent overlap -> smaller delta_neff -> longer coupling length
```

### Resolution convergence: 30 versus 50 px/um

The resolution comparison showed that `L_full` differs by roughly 1-6% between 30 and 50 px/um across the swept gaps:

```text
gap_um  delta_neff_pct_diff_vs_high  L_full_pct_diff_vs_high
0.10     3.19                         -3.10
0.15     2.55                         -2.48
0.20    -4.19                          4.37
0.25    -3.97                          4.13
0.30     4.08                         -3.92
0.40    -0.87                          0.88
0.50     6.34                         -5.96
```

Interpretation:

- Resolution 30 is useful for fast debugging and field-plot exploration.
- Resolution 50 is the current trusted baseline for first-pass design.
- These are not final PDK-grade values; they still need field-symmetry validation, padding checks, and independent benchmarking.

### Field-validation plots

Field plots now show selected MPB bands across multiple gaps, instead of all bands for one gap.

Expected outputs:

```text
results/figures/directional_coupler_fields_gap_sweep.png
results/figures/directional_coupler_fields_gap_sweep_zoom.png
```

Purpose:

- Inspect how candidate modes evolve as gap changes.
- Catch mode switching or cladding-like modes.
- Build intuition for even/odd supermodes and TE-like field confinement.

Important implementation lesson:

- The first attempt to call `get_efield()` directly failed because MPB required `get_dfield()` before converting D to E.
- Field extraction was patched by calling `mode_solver.get_dfield(band)` before `mode_solver.get_efield(band, bloch_phase=True)`.
- The field plot was changed from “all bands at one gap” to “selected bands across several gaps,” which is more useful for mode-continuity validation.

### Coupler design outputs

The module can now produce:

```text
data/sweeps/directional_coupler_gap_sweep_mpb.csv
results/figures/directional_coupler_gap_sweep_mpb.png
data/sweeps/directional_coupler_resolution_convergence.csv
data/sweeps/directional_coupler_resolution_30_vs_50.csv
results/figures/directional_coupler_resolution_convergence.png
data/sweeps/directional_coupler_kappa_vs_length_mpb.csv
results/figures/directional_coupler_kappa_vs_length_mpb.png
data/sweeps/directional_coupler_design_table_mpb.csv
results/figures/directional_coupler_design_map_mpb.png
results/figures/directional_coupler_fields_gap_sweep.png
results/figures/directional_coupler_fields_gap_sweep_zoom.png
```

Design interpretation learned:

- Small gaps are compact but fabrication/gap sensitive.
- Large gaps are more tolerant but require long couplers.
- Around 0.15-0.30 um is likely a useful first-pass design region for moderate coupling.
- Around 0.40-0.50 um can be useful for weak coupling but becomes long for stronger coupling.

### Directional coupler lessons learned

Key physical chain:

```text
gap -> even/odd supermode splitting -> coupling length -> target kappa^2 length
```

The central equations used:

```text
delta_neff = n_even - n_odd
L_full = lambda / (2 delta_neff)
L_3dB = L_full / 2
kappa^2(L) = sin^2(pi L / (2 L_full))
L(kappa^2) = 2 L_full / pi * asin(sqrt(kappa^2))
```

Conceptual distinction reinforced:

- The MPB supermode model is an infinite uniform coupler model.
- It predicts ideal beating between even and odd modes.
- It does not yet include finite input/output transitions, reflections, radiation, excess loss, or port-based S-parameters.

### Current directional-coupler caveats

1. Even/odd selection currently uses the two largest distinct guided `n_eff` values; this is useful but not final mode tracking.
2. Field validation is visual/diagnostic rather than fully automated.
3. MPB field plots at fixed `k_guess` are validation plots, not exact field plots at each `find_k` root.
4. Coupler design tables are based on ideal, symmetric, lossless supermode beating.
5. Finite-length coupler transitions and S-parameters have not yet been simulated.
6. Wavelength dependence of coupling is not yet included.
7. Results have not been benchmarked against Femwell, Lumerical MODE/EME, Tidy3D, or another independent solver.

### Open-source solver note

Future simulation lessons should consider additional open-source tools when useful:

- **MPB/Meep**: current eigenmode and future FDTD workflow.
- **Femwell**: useful candidate for FEM eigenmode solving, independent benchmarking of effective index/supermode splitting, and potentially curved/bent waveguide cross-section studies.
- **gdsfactory**: layout/PCell and design-rule-aware workflows.
- **SAX**: circuit-level compact modeling after S-parameter extraction.
- **Tidy3D**: Python-driven cloud FDTD option when appropriate.

Femwell should not replace MPB automatically, but it should be considered when FEM geometry handling, independent benchmarking, or finite-element cross-section studies would strengthen the learning objective.

## Current status

We have a functioning mini-framework:

```text
shared spec
  -> layout generation
  -> analytic EIM estimate
  -> numerical MPB mode solve
  -> mode validation diagnostics
  -> wavelength sweep and group-index estimate
  -> ring FSR compact model
  -> all-pass ring spectrum
  -> ring resonance metrics
  -> ring Q/loss/coupling interpretation
  -> add-drop ring compact model
  -> cascaded ring compact model
  -> stacked-ring coupled-mode model
  -> directional coupler MPB supermode extraction
  -> directional coupler convergence and field validation
  -> directional coupler design-table post-processing
  -> tests
  -> Git commits
```

Current best waveguide statement:

> MPB band 1 is a plausible TE-like, core-guided mode with engineering-estimate n_eff ~= 2.444 for the 500 nm x 220 nm SOI strip waveguide in oxide at 1550 nm.

Current best compact-model statement:

> The framework now uses the waveguide group-index workflow to estimate ring FSR, generate all-pass/add-drop/cascaded/stacked ring spectra, and extract resonance-level metrics including FSR, extinction ratio, linewidth, loaded Q, loss-budget-derived intrinsic Q, coupling-dependent behavior, and detuning-dependent stacked-ring splitting.

Current best directional-coupler statement:

> MPB supermode extraction now gives a first-pass geometry-to-coupling bridge for two parallel 500 nm x 220 nm SOI strip waveguides. The framework extracts `n_even`, `n_odd`, `delta_neff`, `L_full`, target `kappa^2` lengths, convergence comparisons, and field-validation plots across gap.

---

## Current caveats

1. Waveguide `n_eff` has not been benchmarked against an independent trusted mode solver.
2. Current group index includes waveguide dispersion only; material dispersion is not implemented.
3. Ring models use wavelength-independent coupling.
4. Ring propagation/bend/coupler losses are compact loss-budget terms, not EM-simulated loss values.
5. Bend loss is not yet simulated from geometry.
6. Backscattering / CW-CCW doublet splitting is not modeled.
7. Thermal tuning is represented as resonance detuning, not as a full current-power-temperature-index model.
8. Stacked-ring `mu` is normalized and not yet tied to geometry.
9. Fabrication variation / Monte Carlo yield analysis is not implemented.
10. Field confinement fraction in silicon has not yet been quantified, only visually inspected.
11. S-parameter extraction has not yet started.
12. Directional coupler supermode simulation has started and works for first-pass design, but finite-length coupler S-parameter extraction has not yet started.
13. Directional coupler coupling is still treated as ideal lossless supermode beating; finite transitions, reflections, radiation, excess loss, and broadband complex phase are not yet modeled.
14. Femwell has not yet been integrated but should be considered as an additional open-source FEM/eigenmode benchmark where useful.

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
14. coupling-power sweep and critical coupling
15. intrinsic/coupling/loaded Q decomposition
16. add-drop ring spectrum and metrics
17. add-drop coupling-balance sweep
18. cascaded all-pass rings with identical and detuned rings
19. stacked-ring coupled-mode model
20. fixed-`mu` ring detuning sweep as heater-like tuning
21. ring loss-budget conversion from dB/cm + bend/coupler loss to round-trip power loss and intrinsic Q
22. lecture-ready EIM equation derivation and parameter table
23. MPB eigenproblem versus FDTD distinction
24. MPB field plots as eigenmode/eigenvector fields
25. numerical pitfalls including staircasing, padding/domain interaction, and mode identity switching
26. lifetime-equivalent round-trip estimate from loaded Q
27. coupling-regime interpretation: undercoupled, critical, overcoupled
28. add-drop balanced-coupling diagonal spectra and heatmap interpretation
29. stacked-ring splitting as degeneracy lifting and detuning as hybrid-mode unbalancing
30. directional coupler even/odd supermode intuition
31. `delta_neff` as the source of power beating and coupling length
32. gap sweep trend: smaller gap gives larger splitting and shorter coupling length
33. MPB resolution convergence for directional coupler supermodes
34. field-validation plots across selected bands and gaps
35. difference between infinite-supermode coupler estimates and finite-length coupler S-parameters

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
- coupling loss is not intrinsic loss
- propagation loss dB/cm does not automatically include bend loss unless measured on structures containing comparable bends
- field transfer functions should be cascaded before converting to power
- cascaded rings and directly coupled/stacked rings are physically different
- `mu` is a normalized ring-to-ring coupling rate, not the same thing as bus-ring `kappa^2`
- a heater primarily shifts resonance wavelength, but can also cause secondary loss/coupling/crosstalk effects
- current group index is waveguide-only because material dispersion is not implemented yet
- MPB is not FDTD; it solves frequency-domain eigenmodes for a given wavevector
- MPB field plots are eigenmode field profiles, not time-propagated launch simulations
- staircasing is a mesh/geometry representation error, especially important for high-index-contrast silicon boundaries
- fixed physical axes and common color scales are better for comparing field plots in lecture slides
- a lifetime-equivalent number of round trips is estimated from photon lifetime and loaded Q, not directly counted as discrete round trips
- coupling splits identical stacked-ring resonances by lifting degeneracy; detuning then unbalances the hybrid modes
- directional coupler supermode extraction is not the same as finite-device S-parameter extraction
- a design table is useful, but the next simulation-skill step is finite-length coupler simulation with sources, monitors, PML, mesh, and port/modal power extraction

---

## Recommended next technical step

Next step after the break:

> Move from infinite directional-coupler supermode estimates to finite-length directional coupler simulation and S-parameter extraction.

Why:

The current directional-coupler module is useful for first-pass design:

```text
gap -> delta_neff -> L_full -> target kappa^2 length
```

But it does not yet build the next layer of simulation skill. The next learning objective is to simulate a real finite coupler with:

```text
input waveguides
finite coupling region
output waveguides
ports
source setup
monitors
PML
mesh choices
through/cross/reflection/excess-loss extraction
```

### Next Section 1: finite-length directional coupler simulation

Goal:

> Pick a gap, e.g. `g = 0.20 um`, use the supermode result to predict `L_3dB`, then simulate finite couplers with lengths around that value and compare measured through/cross power to the ideal supermode prediction.

Recommended module:

```text
src/simulation/directional_coupler_finite.py
```

Possible outputs:

```text
data/sweeps/directional_coupler_length_sweep_fdtd.csv
results/figures/directional_coupler_length_sweep_fdtd.png
```

Key columns:

```text
gap_um
length_um
through_power
cross_power
reflected_power
excess_loss
ideal_cross_power
ideal_through_power
```

Primary simulation lessons:

- mode source placement and polarization
- output monitor placement
- guided-mode power versus raw field intensity
- PML distance and thickness
- mesh resolution around the gap
- finite transition effects
- comparison between finite-device simulation and infinite-supermode theory

Solver guidance:

- Open-source route: start with Meep FDTD for learning source/monitor/PML/mesh issues.
- Professional/commercial analog: Lumerical MODE EME is often better for finite coupler length sweeps.
- Femwell may be useful for independent FEM eigenmode benchmarking and possibly cross-section studies before/alongside FDTD.

### Next Section 2: S-parameter extraction and compact-model fitting

Goal:

> Extract complex coupler S-parameters versus wavelength and fit a reusable compact model for circuit simulation.

A directional coupler should eventually be represented by complex quantities such as:

```text
S21(lambda): through amplitude
S31(lambda): cross amplitude
S11(lambda): reflection amplitude
loss(lambda): excess/radiated/non-guided power
```

This matters because `kappa^2` alone loses:

- phase
- reflection
- excess loss
- wavelength dependence
- port convention
- mode purity

Recommended modules:

```text
src/simulation/directional_coupler_sparameters.py
src/compact_models/directional_coupler.py
```

Possible outputs:

```text
data/sparameters/directional_coupler_sparams.csv
data/sparameters/directional_coupler_sparams.npz
results/figures/directional_coupler_sparams.png
```

Professional checkpoint:

> The user should be able to explain why an ideal power coupling value `kappa^2` is insufficient for circuit simulation and why complex S-parameters are needed for rings, MZIs, and larger photonic circuits.

### Do not over-polish design tables now

The design-table work is useful and should remain in the module, but the user correctly identified that it is less valuable for simulation-skill development than finite-device simulation and S-parameter extraction.

The next session should start with finite directional-coupler simulation, not more design-table refinement.

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
