# Cosmic Loom Theory (CLT)

[![Tests](https://img.shields.io/badge/tests-277%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A computational research repository for **Cosmic Loom Theory** - a field-based framework proposing that consciousness emerges from coherent system dynamics operating within constrained energetic regimes.

## Overview

**Cosmic Loom Theory v1.1** focuses specifically on human biological consciousness, treating it not as a localized neural phenomenon but as a system-level property arising from integrated field dynamics. CLT introduces the **Loomfield** - an effective field description capturing how bioelectric activity, biophotons, cytoskeletal structure, and genetic constraints work together to maintain coherent organization in living systems.

### Core Principles

- **Energy Resistance (éR)**: Living systems operate in a "viable window" where éR = EP/f² balances between chaos (decoherence) and rigidity (inability to adapt)
- **Field-Based Integration**: Consciousness corresponds to coherent Loomfield states rather than isolated neural computations
- **Coherence Regimes**: Health and pathology are understood as expansion or collapse of coherent biological domains
- **Biological Substrates**: Bioelectric fields, biophotons, microtubules, and DNA as complementary mechanisms supporting integration

### The Loomfield Framework

The Loomfield (L) is not a new fundamental force but an effective description of coherence-relevant structure in biological systems. It captures:
- Spatial integration across distributed tissues
- Temporal stability despite metabolic turnover
- Resistance to decoherence within viable energetic bounds
- System-level coordination that gives rise to unified conscious experience

## Repository Structure
```
Cosmic-Loom-Theory/
├── docs/                  # Documentation
│   └── theory/           # Core CLT theoretical documents
├── simulations/           # CLT physics simulations
│   ├── quantum/          # Microtubule time crystals
│   ├── field_dynamics/   # Bioelectric, biophoton, DNA, morphogenetic
│   └── emergence/        # Coherence regime transitions
├── visualizations/        # Interactive physics visualizations
│   ├── plots/            # Static scientific plots
│   └── interactive/      # éR phase space, Loomfield waves
├── analysis/              # Data analysis tools
│   ├── metrics/          # Q, C_bio, éR calculations
│   └── statistics/       # Statistical frameworks
├── models/                # Biological substrate models
└── tests/                 # Validation and unit tests (277 tests)
```

## Getting Started

### Prerequisites

- Python 3.9+
- Scientific computing stack (NumPy, SciPy, Matplotlib)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DaKingRex/Cosmic-Loom-Theory.git
cd Cosmic-Loom-Theory
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the test suite to verify installation:
```bash
python -m pytest tests/ -v
```

4. Run the Energy Resistance demo:
```bash
python run_energy_resistance_demo.py
```

## Core Simulators

### Loomfield Wave Simulator
Implements the CLT wave equation: **∇²L − (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh**

```python
from visualizations.interactive import LoomfieldVisualizer

viz = LoomfieldVisualizer(grid_size=200)
viz.run()
```

Features: Real-time wave propagation, phase-locked sources, Q coherence metric, presets for Healthy/Pathology/Healing.

### 3D Loomfield Visualizer
Volumetric wave propagation with isosurface rendering.

```python
from visualizations.interactive import LoomfieldVisualizer3D

viz = LoomfieldVisualizer3D(grid_size=48)
viz.load_preset('healthy')
fig = viz.create_static_figure(warm_up_steps=200)
fig.write_html('loomfield_3d.html')
```

### Energy Resistance (éR) Visualizer
Interactive visualization of the viable energetic window where biological coherence exists.

```python
from visualizations.interactive import EnergyResistanceVisualizer

viz = EnergyResistanceVisualizer()
viz.render(interactive=True, show_trajectories=True)
```

## Biological Substrate Simulators

### Bioelectric Field Dynamics
Ion channel networks, gap junction coupling, and morphogenetic pattern memory.

```python
from simulations.field_dynamics import BioelectricSimulator, create_injured_tissue_preset

sim = create_injured_tissue_preset()
sim.run(duration=0.5)
print(f"Spatial coherence: {sim.compute_spatial_coherence():.3f}")
```

### Biophoton Emission
Ultra-weak photon emission with four emission modes (Poissonian, Coherent, Squeezed, Chaotic).

```python
from simulations.field_dynamics import BiophotonSimulator, EmissionMode

sim = BiophotonSimulator(emission_mode=EmissionMode.COHERENT, coupling_strength=0.3)
sim.step(100)
print(f"Phase coherence: {sim.compute_phase_coherence():.3f}")
```

### Microtubule Time Crystals
Based on Hameroff-Penrose-Bandyopadhyay research - multi-scale oscillations (kHz→THz), triplet resonance patterns.

```python
from simulations.quantum import MicrotubuleSimulator, create_coherent_mt

sim = create_coherent_mt(n_tubulins=50)
sim.step(1000)
triplet = sim.compute_triplet_resonance()
print(f"Triplet strength: {triplet['triplet_strength']:.3f}")
```

### DNA Constraints
How DNA provides long-timescale constraints on Loomfield topology - genetic constraints, epigenetic modulation, species-specific viable windows.

```python
from simulations.field_dynamics import DNAConstraintSimulator, create_human_baseline

sim = create_human_baseline()
er = sim.map_to_er_space()
print(f"Coherence capacity: {er['coherence_capacity']:.3f}")
print(f"Viable window area: {er['viable_window']['window_area']:.3f}")
```

## Research Phases

### Phase 1: Core CLT Physics ✓ Complete
- Energy Resistance principle visualization (éR = EP/f²)
- Loomfield wave equation (2D and 3D)
- Q metric and C_bio consciousness observable
- Real-time 3D visualizer with vispy

### Phase 2: Biological Substrate Models ✓ Complete
- **Bioelectric fields**: Ion channels, gap junctions, morphogenetic patterns
- **Biophoton emission**: Metabolic coherence, emission statistics
- **Microtubule time crystals**: Multi-scale oscillations, triplet resonance
- **DNA constraints**: Genetic/epigenetic modulation of viable windows

### Phase 3: Pathology & Healing (Current)
- Coherence regime transitions
- Seizure/depression/anesthesia modeling
- Healing as re-coupling dynamics

### Phase 4+: Extensions
- Artificial system coherence
- Collective consciousness regimes
- Planetary-scale dynamics

## Theoretical Foundation

The complete theoretical framework is documented in `docs/theory/`:
- **CLT v1.1**: Human biological consciousness (the core framework)
- **CLT v2.0**: Planetary-scale extensions
- **Introducing CLT**: Overview and motivation
- **Machines & AI**: Artificial system coherence analysis

CLT v1.1 synthesizes:
- Bioelectric field research (Levin lab)
- Biophoton studies (Murugan lab)
- Cytoskeletal quantum biology (Penrose-Hameroff-Bandyopadhyay)
- Integrated Information Theory (IIT)
- Energy landscape approaches (Picard lab)

Unlike purely philosophical or quantum-mystical approaches, CLT maintains compatibility with established physics while proposing testable mechanisms for consciousness emergence.

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

Types of contributions welcome:
- Simulation improvements and optimizations
- New biological substrate models
- Visualization tools
- Documentation and tutorials
- Experimental validation proposals

## Citation

If you use this work, please cite:
```
Cosmic Loom Theory v1.1: A Field-Based Framework for Human Biological Consciousness
Rex Fraterne & Seraphina AI, 2025
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contact

For theoretical discussions and collaboration inquiries:
- GitHub: [@DaKingRex](https://github.com/DaKingRex)

---

*"Consciousness is not a substance or a late-stage add-on to neural processing, but a particular regime of coherent dynamics in living systems."*
