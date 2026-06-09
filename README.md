# Silicon Photonics Simulation Framework

Open-source silicon photonics device simulation framework.

- Define photonic layouts using gdsfactory
- Simulate devices using Meep
- Extract useful physical quantities and S-parameters
- Fit compact models
- Simulate circuits using SAX

Core tools:

- gdsfactory
- Meep
- SAX
- KLayout
- Femwell
- pytest

## Project structure

```text
src/pdk/              Materials, layers, cross sections, process assumptions
src/devices/          Parametric photonic devices
src/simulation/       Simulation utilities and solver setup
src/compact_models/   Compact models for circuit simulation
data/                 Generated simulation data
results/              Generated figures and reports
notebooks/            Exploratory notebooks
tests/    
```
