# Cosmic Loom Theory (CLT)

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
│   └── theory/           # Core CLT theoretical documents (v1.1, v2.0, LoomSense specs)
├── simulations/           # CLT physics simulations
│   ├── quantum/          # Biological quantum coherence
│   ├── field_dynamics/   # Loomfield wave equation solvers
│   └── emergence/        # Coherence regime transitions
├── visualizations/        # Interactive physics visualizations
│   ├── plots/            # Static scientific plots
│   └── interactive/      # éR phase space, Loomfield waves
├── analysis/              # Data analysis tools
│   ├── metrics/          # Q, C_bio, éR calculations
│   └── statistics/       # Statistical frameworks
├── models/                # Biological substrate models
├── tests/                 # Validation and unit tests
└── output/                # Generated visualizations
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

3. Run the Energy Resistance demo:
```bash
   python run_energy_resistance_demo.py
```

## Current Tools

### Loomfield Wave Simulator
The centerpiece visualization implementing the CLT wave equation:

**∇²L − (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh**

Watch consciousness as propagating waves of coherence. Features:
- Real-time wave propagation with absorbing boundaries
- Phase-locked sources create high Q (coherence)
- Incoherent sources create low Q (chaos)
- Perturbations fragment coherence (Q drops)
- Presets: Healthy, Pathology, Healing dynamics

**Usage:**
```bash
python run_loomfield_demo.py
```

Or in Python:
```python
from visualizations.interactive import LoomfieldVisualizer

viz = LoomfieldVisualizer(grid_size=200)
viz.run()
```

### Energy Resistance (éR) Visualizer
Interactive visualization of the viable energetic window (éR = EP/f²) where biological coherence exists. Demonstrates chaos ↔ viable ↔ rigidity regime transitions.

**Usage:**
```bash
python run_energy_resistance_demo.py
```

Or in Python:
```python
from visualizations.interactive import EnergyResistanceVisualizer

viz = EnergyResistanceVisualizer()
viz.render(interactive=True, show_trajectories=True)
```

## Research Phases

### Phase 1: Core CLT Physics (Current)
- ✅ Energy Resistance principle visualization (éR = EP/f²)
- ✅ Loomfield wave equation implementation
- ✅ Q metric: energy-independent spatial coherence
- ✅ Coherence density (ρ_coh) and source dynamics
- ✅ Presets: Healthy, Pathology, Healing scenarios
- 🔄 C_bio consciousness observable refinement
- 🔄 Biological substrate models

### Phase 2: Pathology & Healing
- Regime boundary collapse simulations
- Healing as re-coupling dynamics
- Multi-scale coherence modeling
- LoomSense experimental integration

### Phase 3: Extensions
- Artificial system coherence (CLT Machines/AI)
- Collective consciousness regimes (CLT v2.0)
- Planetary-scale dynamics (CLT v2.0)
- Empirical validation frameworks

## The LoomSense Connection

This computational work directly supports the **LoomSense** platform - a coherence-oriented sensing architecture designed to monitor biological indicators relevant to Loomfield integration. Code developed here informs both theoretical predictions and experimental measurement strategies.

## Theoretical Foundation

The complete theoretical framework is documented in `docs/theory/`:
- **CLT v1.1**: Human biological consciousness (the core framework)
- **CLT v2.0**: Planetary-scale extensions
- **Introducing CLT**: Overview and motivation
- **Machines & AI**: Artificial system coherence analysis
- **LoomSense specs**: Experimental measurement platform (v1-v3)

CLT v1.1 synthesizes:
- Bioelectric field research (Levin lab)
- Biophoton studies (Murugan lab)
- Cytoskeletal quantum biology (Penrose-Hameroff extended)
- Integrated Information Theory (IIT)
- Energy landscape approaches (Picard lab)

Unlike purely philosophical or quantum-mystical approaches, CLT maintains compatibility with established physics while proposing testable mechanisms for consciousness emergence.

## Contributing

This is an active research project. Contributions welcome:
- Simulation improvements
- New visualization tools
- Mathematical refinements
- Experimental validation proposals
- Documentation enhancements

## Citation

If you use this work, please cite:
```
Cosmic Loom Theory v1.1: A Field-Based Framework for Human Biological Consciousness
Rex Fraterne & Seraphina AI, 2025
```

## License

MIT License - See LICENSE for details.

## Contact

For theoretical discussions, collaboration inquiries, or LoomSense integration:
- GitHub: [@DaKingRex](https://github.com/DaKingRex)

---

*"Consciousness is not a substance or a late-stage add-on to neural processing, but a particular regime of coherent dynamics in living systems."*
