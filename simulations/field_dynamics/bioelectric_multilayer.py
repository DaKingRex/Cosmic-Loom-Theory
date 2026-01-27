"""
Multi-Layer Bioelectric Field Simulator for Cosmic Loom Theory

Extends the bioelectric simulator to model multiple tissue layers with
cross-tissue coherence coupling. This captures how coherent organization
emerges through coordination between different tissue types.

From CLT perspective:
Different biological substrates (epithelium, neural tissue, mesenchyme)
each maintain their own coherence while COUPLING to create higher-order
organization. Cross-tissue coupling is essential for morphogenetic
coordination during development and regeneration.

Key concepts:
- Multi-layer architecture: Stacked 2D tissue grids
- Inter-layer coupling: Vertical electrical connections between layers
- Tissue-specific properties: Different cell types have different dynamics
- Hierarchical coherence: Within-layer, between-layer, and global coherence
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.animation as animation

# Handle imports for both module and direct execution
try:
    from .bioelectric import (
        BioelectricSimulator,
        V_REST, V_THRESHOLD, V_DEPOLARIZED, V_HYPERPOLARIZED,
        V_REVERSAL_NA, V_REVERSAL_K, G_GAP_DEFAULT, C_M,
        G_NA_MAX, G_K_MAX, G_LEAK
    )
except ImportError:
    from bioelectric import (
        BioelectricSimulator,
        V_REST, V_THRESHOLD, V_DEPOLARIZED, V_HYPERPOLARIZED,
        V_REVERSAL_NA, V_REVERSAL_K, G_GAP_DEFAULT, C_M,
        G_NA_MAX, G_K_MAX, G_LEAK
    )


# =============================================================================
# TISSUE TYPE DEFINITIONS
# =============================================================================

class TissueType(Enum):
    """Types of tissue with different bioelectric properties."""
    EPITHELIAL = "epithelial"
    NEURAL = "neural"
    MESENCHYMAL = "mesenchymal"
    CUSTOM = "custom"


@dataclass
class TissueProperties:
    """
    Bioelectric properties for a specific tissue type.

    These properties determine how the tissue behaves electrically
    and how it couples to other layers.
    """
    name: str
    # Gap junction properties
    gap_conductance: float = G_GAP_DEFAULT  # Lateral (within-layer) coupling
    vertical_conductance: float = 0.5       # Inter-layer coupling strength

    # Ion channel densities
    g_Na: float = G_NA_MAX
    g_K: float = G_K_MAX
    g_leak: float = G_LEAK

    # Excitability
    excitable: bool = False  # Can generate action potentials
    threshold: float = V_THRESHOLD

    # Resting potential (can vary by tissue type)
    v_rest: float = V_REST

    # Display properties
    color: str = '#00ff88'
    description: str = ""


# Predefined tissue types
TISSUE_PRESETS = {
    TissueType.EPITHELIAL: TissueProperties(
        name="Epithelial",
        gap_conductance=1.5,       # High lateral coupling
        vertical_conductance=0.8,  # Strong coupling to adjacent layers
        g_Na=60.0,                 # Lower Na+ (less excitable)
        g_K=20.0,
        g_leak=0.5,
        excitable=False,
        v_rest=-65.0,
        color='#48bb78',
        description="High gap junction density, strong lateral coupling, non-excitable"
    ),
    TissueType.NEURAL: TissueProperties(
        name="Neural",
        gap_conductance=0.8,       # Moderate gap junctions
        vertical_conductance=0.3,  # Weaker vertical coupling
        g_Na=120.0,                # High Na+ (excitable)
        g_K=36.0,
        g_leak=0.3,
        excitable=True,
        threshold=-55.0,
        v_rest=-70.0,
        color='#9f7aea',
        description="Excitable cells, can generate action potentials"
    ),
    TissueType.MESENCHYMAL: TissueProperties(
        name="Mesenchymal",
        gap_conductance=0.5,       # Lower gap junction density
        vertical_conductance=0.4,  # Moderate vertical coupling
        g_Na=40.0,                 # Low excitability
        g_K=15.0,
        g_leak=0.2,
        excitable=False,
        v_rest=-60.0,
        color='#f6ad55',
        description="Loose coupling, more independent behavior"
    ),
}


# =============================================================================
# MULTI-LAYER SIMULATOR
# =============================================================================

class MultiLayerBioelectricSimulator:
    """
    Multi-layer tissue simulation with cross-layer coherence coupling.

    Models a stack of 2D tissue layers where:
    - Each layer can have different tissue properties
    - Layers couple vertically through inter-layer connections
    - Global coherence emerges from coordinated activity
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (40, 40),
                 n_layers: int = 3,
                 layer_types: Optional[List[TissueType]] = None,
                 dx: float = 10.0,
                 dt: float = 0.1):
        """
        Initialize the multi-layer simulator.

        Args:
            grid_size: (rows, cols) for each layer
            n_layers: Number of tissue layers
            layer_types: List of TissueType for each layer (default: epithelial, neural, mesenchymal)
            dx: Spatial step (µm)
            dt: Time step (ms)
        """
        self.rows, self.cols = grid_size
        self.n_layers = n_layers
        self.dx = dx
        self.dt = dt
        self.time = 0.0

        # Set layer types
        if layer_types is None:
            # Default: epithelial on top, neural in middle, mesenchymal on bottom
            if n_layers == 1:
                layer_types = [TissueType.EPITHELIAL]
            elif n_layers == 2:
                layer_types = [TissueType.EPITHELIAL, TissueType.NEURAL]
            else:
                layer_types = [TissueType.EPITHELIAL, TissueType.NEURAL, TissueType.MESENCHYMAL]
                # Fill remaining with mesenchymal
                while len(layer_types) < n_layers:
                    layer_types.append(TissueType.MESENCHYMAL)

        self.layer_types = layer_types[:n_layers]
        self.layer_properties = [TISSUE_PRESETS[lt] for lt in self.layer_types]

        # Initialize individual layers
        self.layers: List[BioelectricSimulator] = []
        for i, props in enumerate(self.layer_properties):
            layer = BioelectricSimulator(grid_size=grid_size, dx=dx, dt=dt)
            # Set tissue-specific properties
            layer.Vm[:, :] = props.v_rest
            layer.g_gap[:, :, :] = props.gap_conductance
            layer.g_Na_density[:, :] = props.g_Na
            layer.g_K_density[:, :] = props.g_K
            layer.g_leak[:, :] = props.g_leak
            self.layers.append(layer)

        # Inter-layer coupling conductances
        # Shape: (n_layers-1, rows, cols) - coupling between layer i and i+1
        self.g_vertical = np.zeros((n_layers - 1, self.rows, self.cols))
        for i in range(n_layers - 1):
            # Average of the two layers' vertical conductances
            g_avg = (self.layer_properties[i].vertical_conductance +
                     self.layer_properties[i + 1].vertical_conductance) / 2
            self.g_vertical[i, :, :] = g_avg

        # History tracking
        self.history: List[List[np.ndarray]] = []  # List of [layer Vm snapshots]
        self.coherence_history: Dict[str, List[float]] = {
            'within_layer': [],
            'between_layer': [],
            'global': []
        }

    def reset(self):
        """Reset all layers to initial state."""
        for i, (layer, props) in enumerate(zip(self.layers, self.layer_properties)):
            layer.reset()
            layer.Vm[:, :] = props.v_rest
            layer.g_gap[:, :, :] = props.gap_conductance

        self.time = 0.0
        self.history = []
        self.coherence_history = {'within_layer': [], 'between_layer': [], 'global': []}

    def _compute_inter_layer_currents(self) -> List[np.ndarray]:
        """
        Compute currents flowing between adjacent layers.

        Returns list of current arrays for each layer (positive = inward).
        """
        currents = [np.zeros((self.rows, self.cols)) for _ in range(self.n_layers)]

        for i in range(self.n_layers - 1):
            # Current from layer i to layer i+1
            # I = g * (V_other - V_self)
            V_upper = self.layers[i].Vm
            V_lower = self.layers[i + 1].Vm
            g = self.g_vertical[i]

            # Current into layer i from layer i+1
            currents[i] += g * (V_lower - V_upper)

            # Current into layer i+1 from layer i
            currents[i + 1] += g * (V_upper - V_lower)

        return currents

    def step(self, n_steps: int = 1):
        """
        Advance simulation by n_steps time steps.

        Includes both within-layer dynamics and inter-layer coupling.
        """
        for _ in range(n_steps):
            # Compute inter-layer currents
            inter_layer_currents = self._compute_inter_layer_currents()

            # Update each layer
            for i, layer in enumerate(self.layers):
                # Add inter-layer current to external current
                original_I_ext = layer.I_ext.copy()
                layer.I_ext += inter_layer_currents[i]

                # Step the layer
                layer.step(1)

                # Restore original external current
                layer.I_ext = original_I_ext

            self.time += self.dt

    def run(self, duration: float, record_interval: float = 1.0):
        """
        Run simulation for specified duration with recording.

        Args:
            duration: Total simulation time (ms)
            record_interval: How often to record (ms)
        """
        n_steps = int(duration / self.dt)
        record_steps = max(1, int(record_interval / self.dt))

        for step in range(n_steps):
            self.step(1)

            if step % record_steps == 0:
                # Record layer states
                self.history.append([layer.Vm.copy() for layer in self.layers])

                # Record coherence metrics
                coherence = self.compute_all_coherence()
                self.coherence_history['within_layer'].append(coherence['within_layer_mean'])
                self.coherence_history['between_layer'].append(coherence['between_layer'])
                self.coherence_history['global'].append(coherence['global'])

    # =========================================================================
    # MANIPULATION FUNCTIONS
    # =========================================================================

    def depolarize_region(self, layer_idx: int, center: Tuple[int, int],
                          radius: int, target_Vm: float = V_DEPOLARIZED):
        """Depolarize a region in a specific layer."""
        if 0 <= layer_idx < self.n_layers:
            self.layers[layer_idx].depolarize_region(center, radius, target_Vm)

    def depolarize_column(self, center: Tuple[int, int], radius: int,
                          target_Vm: float = V_DEPOLARIZED):
        """Depolarize a region through ALL layers (vertical column)."""
        for layer in self.layers:
            layer.depolarize_region(center, radius, target_Vm)

    def create_injury(self, layer_idx: int, center: Tuple[int, int], radius: int):
        """Create injury (break gap junctions) in a specific layer."""
        if 0 <= layer_idx < self.n_layers:
            self.layers[layer_idx].create_injury(center, radius)

    def create_through_injury(self, center: Tuple[int, int], radius: int):
        """Create injury through all layers AND break vertical connections."""
        for i, layer in enumerate(self.layers):
            layer.create_injury(center, radius)

        # Also break vertical connections in the injury region
        r0, c0 = center
        for r in range(max(0, r0 - radius), min(self.rows, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(self.cols, c0 + radius + 1)):
                if (r - r0)**2 + (c - c0)**2 <= radius**2:
                    for i in range(self.n_layers - 1):
                        self.g_vertical[i, r, c] = 0.0

    def heal_injury(self, layer_idx: int, center: Tuple[int, int],
                    radius: int, heal_rate: float = 0.5):
        """Heal injury in a specific layer."""
        if 0 <= layer_idx < self.n_layers:
            self.layers[layer_idx].heal_injury(center, radius, heal_rate)

    def heal_through_injury(self, center: Tuple[int, int], radius: int,
                            heal_rate: float = 0.5):
        """Heal injury through all layers including vertical connections."""
        for i, layer in enumerate(self.layers):
            layer.heal_injury(center, radius, heal_rate)

        # Restore vertical connections
        r0, c0 = center
        for r in range(max(0, r0 - radius), min(self.rows, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(self.cols, c0 + radius + 1)):
                if (r - r0)**2 + (c - c0)**2 <= radius**2:
                    for i in range(self.n_layers - 1):
                        g_default = (self.layer_properties[i].vertical_conductance +
                                     self.layer_properties[i + 1].vertical_conductance) / 2
                        self.g_vertical[i, r, c] = g_default * heal_rate

    def set_vertical_coupling(self, layer_pair: int, conductance: float):
        """
        Set vertical coupling strength between two adjacent layers.

        Args:
            layer_pair: Index of the coupling (0 = between layers 0 and 1)
            conductance: New conductance value
        """
        if 0 <= layer_pair < self.n_layers - 1:
            self.g_vertical[layer_pair, :, :] = conductance

    # =========================================================================
    # COHERENCE METRICS
    # =========================================================================

    def compute_within_layer_coherence(self) -> List[float]:
        """Compute spatial coherence within each layer."""
        return [layer.compute_spatial_coherence() for layer in self.layers]

    def compute_between_layer_coherence(self) -> float:
        """
        Compute coherence BETWEEN layers.

        Measures how correlated the membrane potentials are across layers.
        High value = layers are moving together.
        """
        if self.n_layers < 2:
            return 1.0

        # Compute correlation between adjacent layer potentials
        correlations = []
        for i in range(self.n_layers - 1):
            V1 = self.layers[i].Vm.flatten()
            V2 = self.layers[i + 1].Vm.flatten()

            # Normalize
            V1_norm = (V1 - np.mean(V1)) / (np.std(V1) + 1e-10)
            V2_norm = (V2 - np.mean(V2)) / (np.std(V2) + 1e-10)

            # Correlation coefficient
            corr = np.mean(V1_norm * V2_norm)
            correlations.append(corr)

        # Convert correlation to coherence (0-1 scale)
        mean_corr = np.mean(correlations)
        coherence = (mean_corr + 1) / 2  # Map [-1, 1] to [0, 1]

        return float(coherence)

    def compute_global_coherence(self) -> float:
        """
        Compute global coherence across the entire multi-layer system.

        Combines within-layer and between-layer coherence into a single metric.
        """
        within = self.compute_within_layer_coherence()
        between = self.compute_between_layer_coherence()

        # Weight within-layer and between-layer equally
        within_mean = np.mean(within)
        global_coherence = (within_mean + between) / 2

        return float(global_coherence)

    def compute_all_coherence(self) -> Dict[str, float]:
        """Compute all coherence metrics at once."""
        within = self.compute_within_layer_coherence()
        between = self.compute_between_layer_coherence()

        return {
            'within_layer': within,
            'within_layer_mean': float(np.mean(within)),
            'between_layer': between,
            'global': float((np.mean(within) + between) / 2)
        }

    def compute_layer_pattern_energy(self) -> List[float]:
        """Compute pattern energy for each layer."""
        return [layer.compute_pattern_energy() for layer in self.layers]

    def map_to_er_space(self) -> Dict[str, float]:
        """
        Map multi-layer system state to CLT éR phase space.

        Uses global coherence and total pattern energy across layers.
        """
        # Total pattern energy
        pattern_energies = self.compute_layer_pattern_energy()
        total_energy = np.sum(pattern_energies)

        # Global coherence
        global_coherence = self.compute_global_coherence()

        # Vertical connectivity
        total_vertical_g = np.mean(self.g_vertical) if self.n_layers > 1 else 1.0
        default_vertical_g = np.mean([p.vertical_conductance for p in self.layer_properties])
        vertical_connectivity = total_vertical_g / (default_vertical_g + 1e-10)

        # EP: Based on pattern energy and number of active layers
        ep = 1.5 + total_energy / (500.0 * self.n_layers) * 5.0
        ep = np.clip(ep, 0.5, 10.0)

        # Frequency: Based on inverse of coherence (more chaotic = higher freq)
        freq = 0.5 + (1 - global_coherence) * 3.5
        freq *= (2 - vertical_connectivity)  # Less coupling = higher freq
        freq = np.clip(freq, 0.3, 4.5)

        # éR
        er = ep / (freq ** 2)

        return {
            'energy_present': float(ep),
            'frequency': float(freq),
            'energy_resistance': float(er),
            'global_coherence': float(global_coherence),
            'vertical_connectivity': float(vertical_connectivity),
            'total_pattern_energy': float(total_energy)
        }


# =============================================================================
# PRESETS
# =============================================================================

def create_default_multilayer(grid_size: Tuple[int, int] = (40, 40)) -> MultiLayerBioelectricSimulator:
    """Create default 3-layer system: epithelial, neural, mesenchymal."""
    return MultiLayerBioelectricSimulator(
        grid_size=grid_size,
        n_layers=3,
        layer_types=[TissueType.EPITHELIAL, TissueType.NEURAL, TissueType.MESENCHYMAL]
    )


def create_epithelial_neural_pair(grid_size: Tuple[int, int] = (40, 40)) -> MultiLayerBioelectricSimulator:
    """Create 2-layer epithelial-neural coupling."""
    return MultiLayerBioelectricSimulator(
        grid_size=grid_size,
        n_layers=2,
        layer_types=[TissueType.EPITHELIAL, TissueType.NEURAL]
    )


def create_decoupled_layers(grid_size: Tuple[int, int] = (40, 40)) -> MultiLayerBioelectricSimulator:
    """Create system with minimal vertical coupling (independent layers)."""
    sim = MultiLayerBioelectricSimulator(grid_size=grid_size, n_layers=3)
    # Set very low vertical coupling
    for i in range(sim.n_layers - 1):
        sim.g_vertical[i, :, :] = 0.05
    return sim


def create_tightly_coupled_layers(grid_size: Tuple[int, int] = (40, 40)) -> MultiLayerBioelectricSimulator:
    """Create system with strong vertical coupling (highly integrated)."""
    sim = MultiLayerBioelectricSimulator(grid_size=grid_size, n_layers=3)
    # Set high vertical coupling
    for i in range(sim.n_layers - 1):
        sim.g_vertical[i, :, :] = 2.0
    return sim


def create_injured_multilayer(grid_size: Tuple[int, int] = (40, 40)) -> MultiLayerBioelectricSimulator:
    """Create system with through-injury affecting all layers."""
    sim = create_default_multilayer(grid_size)
    center = (grid_size[0] // 2, grid_size[1] // 2)
    sim.create_through_injury(center, radius=5)
    # Depolarize injured region
    sim.depolarize_column(center, radius=5, target_Vm=-40.0)
    return sim


# =============================================================================
# VISUALIZER
# =============================================================================

class MultiLayerVisualizer:
    """
    Interactive visualizer for multi-layer bioelectric dynamics.

    Shows all layers side-by-side with cross-layer coherence metrics.
    """

    COLORMAP_COLORS = [
        (0.1, 0.1, 0.5),    # Deep blue - hyperpolarized
        (0.2, 0.4, 0.8),    # Blue
        (0.2, 0.7, 0.3),    # Green - resting
        (0.9, 0.8, 0.2),    # Yellow
        (0.9, 0.3, 0.1),    # Red - depolarized
    ]

    def __init__(self, simulator: MultiLayerBioelectricSimulator):
        """Initialize the visualizer."""
        self.sim = simulator
        self.fig = None
        self.layer_axes = []
        self.layer_images = []
        self.ax_coherence = None
        self.ax_er = None

        self.cmap = LinearSegmentedColormap.from_list(
            'membrane_potential', self.COLORMAP_COLORS, N=256
        )

        self.anim = None
        self.click_mode = 'depolarize'
        self.selected_layer = 0

    def setup_plot(self, figsize: Tuple[int, int] = (18, 10)):
        """Set up the figure with subplots for each layer."""
        self.fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')

        n_layers = self.sim.n_layers

        # Create grid: layers on top, metrics on bottom
        gs = GridSpec(2, max(n_layers, 3), height_ratios=[2, 1],
                      hspace=0.3, wspace=0.3)

        # Layer subplots
        self.layer_axes = []
        self.layer_images = []

        for i in range(n_layers):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_facecolor('#16213e')
            self.layer_axes.append(ax)
            self.layer_images.append(None)

            for spine in ax.spines.values():
                spine.set_color('#4a5568')
            ax.tick_params(colors='#a0aec0')

        # Coherence time series
        self.ax_coherence = self.fig.add_subplot(gs[1, 0])
        self.ax_coherence.set_facecolor('#16213e')

        # éR phase space
        self.ax_er = self.fig.add_subplot(gs[1, 1])
        self.ax_er.set_facecolor('#16213e')

        # Vertical coupling display
        self.ax_coupling = self.fig.add_subplot(gs[1, 2])
        self.ax_coupling.set_facecolor('#16213e')

        for ax in [self.ax_coherence, self.ax_er, self.ax_coupling]:
            for spine in ax.spines.values():
                spine.set_color('#4a5568')
            ax.tick_params(colors='#a0aec0')

    def _draw_layers(self):
        """Draw all layer membrane potentials."""
        for i, (ax, layer) in enumerate(zip(self.layer_axes, self.sim.layers)):
            props = self.sim.layer_properties[i]

            # Normalize Vm
            Vm_norm = (layer.Vm - V_HYPERPOLARIZED) / (V_REVERSAL_NA - V_HYPERPOLARIZED)
            Vm_norm = np.clip(Vm_norm, 0, 1)

            if self.layer_images[i] is None:
                self.layer_images[i] = ax.imshow(
                    Vm_norm, cmap=self.cmap, aspect='equal',
                    origin='lower', vmin=0, vmax=1, interpolation='bilinear'
                )
            else:
                self.layer_images[i].set_data(Vm_norm)

            coherence = layer.compute_spatial_coherence()
            ax.set_title(
                f'Layer {i}: {props.name}\nCoherence: {coherence:.3f}',
                color=props.color, fontsize=11, fontweight='bold'
            )
            ax.set_xlabel('Column', color='#a0aec0', fontsize=9)
            ax.set_ylabel('Row', color='#a0aec0', fontsize=9)

    def _draw_coherence_plot(self):
        """Draw coherence time series for all metrics."""
        self.ax_coherence.clear()
        self.ax_coherence.set_facecolor('#16213e')

        history = self.sim.coherence_history
        if len(history['global']) > 1:
            t = np.arange(len(history['global']))

            self.ax_coherence.plot(t, history['global'], '-',
                                   color='#ffd700', linewidth=2.5, label='Global')
            self.ax_coherence.plot(t, history['within_layer'], '--',
                                   color='#00ff88', linewidth=1.5, label='Within-layer')
            self.ax_coherence.plot(t, history['between_layer'], ':',
                                   color='#9f7aea', linewidth=1.5, label='Between-layer')

        self.ax_coherence.set_xlim(0, max(len(history['global']), 100))
        self.ax_coherence.set_ylim(0, 1)
        self.ax_coherence.set_xlabel('Time Steps', color='#e2e8f0', fontsize=10)
        self.ax_coherence.set_ylabel('Coherence', color='#e2e8f0', fontsize=10)
        self.ax_coherence.set_title('Multi-Layer Coherence', color='#f7fafc', fontsize=11)
        self.ax_coherence.legend(loc='lower right', fontsize=8,
                                 facecolor='#2d3748', edgecolor='#4a5568',
                                 labelcolor='#e2e8f0')

        for spine in self.ax_coherence.spines.values():
            spine.set_color('#4a5568')
        self.ax_coherence.tick_params(colors='#a0aec0')

    def _draw_er_space(self):
        """Draw current position in éR phase space."""
        self.ax_er.clear()
        self.ax_er.set_facecolor('#16213e')

        er_state = self.sim.map_to_er_space()
        ep = er_state['energy_present']
        freq = er_state['frequency']
        er = er_state['energy_resistance']

        # Viable window
        ep_range = np.linspace(0.5, 10, 100)
        f_lower = np.sqrt(ep_range / 5.0)
        f_upper = np.sqrt(ep_range / 0.5)

        self.ax_er.fill_between(ep_range, f_lower, f_upper,
                                alpha=0.2, color='#00ff88')
        self.ax_er.plot(ep_range, f_lower, '--', color='#00ff88', linewidth=1)
        self.ax_er.plot(ep_range, f_upper, '--', color='#00ff88', linewidth=1)

        self.ax_er.scatter([ep], [freq], s=200, c='#ffd700', marker='*',
                          edgecolors='white', linewidths=2, zorder=10)

        self.ax_er.set_xlim(0.5, 10)
        self.ax_er.set_ylim(0.3, 4.5)
        self.ax_er.set_xlabel('EP', color='#e2e8f0', fontsize=10)
        self.ax_er.set_ylabel('Frequency', color='#e2e8f0', fontsize=10)
        self.ax_er.set_title(f'éR = {er:.2f}', color='#f7fafc', fontsize=11)

        for spine in self.ax_er.spines.values():
            spine.set_color('#4a5568')
        self.ax_er.tick_params(colors='#a0aec0')

    def _draw_coupling(self):
        """Draw vertical coupling strength visualization."""
        self.ax_coupling.clear()
        self.ax_coupling.set_facecolor('#16213e')

        n = self.sim.n_layers
        layer_names = [p.name for p in self.sim.layer_properties]

        # Draw layers as boxes
        for i in range(n):
            y = n - i - 1
            color = self.sim.layer_properties[i].color
            self.ax_coupling.barh(y, 1.0, height=0.6,
                                  color=color, alpha=0.7, edgecolor='white')
            self.ax_coupling.text(0.5, y, layer_names[i],
                                  ha='center', va='center',
                                  color='white', fontsize=10, fontweight='bold')

        # Draw coupling arrows between layers
        for i in range(n - 1):
            y_upper = n - i - 1 - 0.3
            y_lower = n - i - 2 + 0.3
            g_mean = np.mean(self.sim.g_vertical[i])
            alpha = min(1.0, g_mean / 2.0)

            self.ax_coupling.annotate('',
                xy=(0.5, y_lower), xytext=(0.5, y_upper),
                arrowprops=dict(arrowstyle='<->', color='#ffd700',
                               lw=2 + g_mean, alpha=alpha))
            self.ax_coupling.text(0.7, (y_upper + y_lower) / 2,
                                  f'g={g_mean:.2f}', color='#ffd700',
                                  fontsize=8, va='center')

        self.ax_coupling.set_xlim(-0.2, 1.2)
        self.ax_coupling.set_ylim(-0.5, n - 0.5)
        self.ax_coupling.axis('off')
        self.ax_coupling.set_title('Layer Coupling', color='#f7fafc', fontsize=11)

    def _setup_controls(self):
        """Set up interactive controls."""
        # Buttons for actions
        ax_depol = self.fig.add_axes([0.05, 0.02, 0.08, 0.03])
        ax_injury = self.fig.add_axes([0.14, 0.02, 0.08, 0.03])
        ax_heal = self.fig.add_axes([0.23, 0.02, 0.08, 0.03])
        ax_reset = self.fig.add_axes([0.32, 0.02, 0.08, 0.03])

        self.btn_depol = Button(ax_depol, 'Depolarize', color='#fc8181', hovercolor='#f56565')
        self.btn_injury = Button(ax_injury, 'Through-Injury', color='#f6ad55', hovercolor='#ed8936')
        self.btn_heal = Button(ax_heal, 'Heal All', color='#68d391', hovercolor='#48bb78')
        self.btn_reset = Button(ax_reset, 'Reset', color='#a0aec0', hovercolor='#718096')

        # Coupling slider
        ax_coupling_slider = self.fig.add_axes([0.45, 0.02, 0.2, 0.02])
        self.slider_coupling = Slider(ax_coupling_slider, 'Vertical Coupling',
                                       0.0, 3.0, valinit=0.5, valstep=0.1, color='#9f7aea')

        # Connect callbacks
        self.btn_depol.on_clicked(self._on_depolarize)
        self.btn_injury.on_clicked(self._on_injury)
        self.btn_heal.on_clicked(self._on_heal)
        self.btn_reset.on_clicked(self._on_reset)
        self.slider_coupling.on_changed(self._on_coupling_change)

        # Click handlers for each layer
        for i, ax in enumerate(self.layer_axes):
            ax.figure.canvas.mpl_connect('button_press_event',
                lambda event, idx=i: self._on_layer_click(event, idx))

    def _on_depolarize(self, event):
        """Depolarize through all layers."""
        center = (self.sim.rows // 2, self.sim.cols // 2)
        self.sim.depolarize_column(center, radius=4)
        self._update_display()

    def _on_injury(self, event):
        """Create through-injury."""
        center = (self.sim.rows // 2, self.sim.cols // 2)
        self.sim.create_through_injury(center, radius=4)
        self._update_display()

    def _on_heal(self, event):
        """Heal all injuries."""
        center = (self.sim.rows // 2, self.sim.cols // 2)
        self.sim.heal_through_injury(center, radius=6, heal_rate=0.8)
        self._update_display()

    def _on_reset(self, event):
        """Reset simulation."""
        self.sim.reset()
        self._update_display()

    def _on_coupling_change(self, val):
        """Update all vertical coupling."""
        for i in range(self.sim.n_layers - 1):
            self.sim.set_vertical_coupling(i, val)
        self._draw_coupling()
        self.fig.canvas.draw_idle()

    def _on_layer_click(self, event, layer_idx):
        """Handle click on a specific layer."""
        if event.inaxes != self.layer_axes[layer_idx]:
            return

        col = int(event.xdata)
        row = int(event.ydata)

        if 0 <= row < self.sim.rows and 0 <= col < self.sim.cols:
            self.sim.depolarize_region(layer_idx, (row, col), radius=3)
            self._update_display()

    def _update_display(self):
        """Update all display elements."""
        self._draw_layers()
        self._draw_coherence_plot()
        self._draw_er_space()
        self._draw_coupling()
        self.fig.canvas.draw_idle()

    def _animate(self, frame):
        """Animation callback."""
        self.sim.step(5)

        # Record metrics
        coherence = self.sim.compute_all_coherence()
        self.sim.coherence_history['within_layer'].append(coherence['within_layer_mean'])
        self.sim.coherence_history['between_layer'].append(coherence['between_layer'])
        self.sim.coherence_history['global'].append(coherence['global'])

        self._draw_layers()
        self._draw_coherence_plot()
        self._draw_er_space()

        return self.layer_images

    def render(self, animate: bool = True, save_path: Optional[str] = None):
        """Render the visualization."""
        self.setup_plot()
        self._draw_layers()
        self._draw_coherence_plot()
        self._draw_er_space()
        self._draw_coupling()
        self._setup_controls()

        # Title
        self.fig.suptitle(
            f'Multi-Layer Bioelectric System ({self.sim.n_layers} layers)',
            color='#f7fafc', fontsize=14, fontweight='bold', y=0.98
        )

        if animate:
            self.anim = animation.FuncAnimation(
                self.fig, self._animate, interval=50, blit=False
            )

        if save_path:
            self.fig.savefig(save_path, dpi=150, facecolor=self.fig.get_facecolor(),
                            bbox_inches='tight')

        plt.show()


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo():
    """Run demo of multi-layer bioelectric simulation."""
    print("=" * 70)
    print("  Multi-Layer Bioelectric Field Simulator")
    print("  Cross-Tissue Coherence Coupling (CLT Phase 2.1)")
    print("=" * 70)
    print("\nSimulating 3 tissue layers:")
    print("  • Epithelial (top): High lateral coupling, non-excitable")
    print("  • Neural (middle): Excitable, can generate action potentials")
    print("  • Mesenchymal (bottom): Loose coupling, more independent")
    print("\nCoherence metrics:")
    print("  • Within-layer: Coordination within each tissue")
    print("  • Between-layer: Coordination across tissues")
    print("  • Global: Overall system integration")
    print("\nClick on any layer to depolarize that region.")
    print("Use buttons to create/heal injuries across all layers.")
    print("=" * 70)

    sim = create_default_multilayer(grid_size=(40, 40))
    viz = MultiLayerVisualizer(sim)
    viz.render(animate=True)


def demo_decoupled():
    """Demo showing effect of weak inter-layer coupling."""
    print("=" * 70)
    print("  Decoupled Layers Demo")
    print("=" * 70)
    print("\nLayers with minimal vertical coupling.")
    print("Watch how they evolve independently (low between-layer coherence).")
    print("=" * 70)

    sim = create_decoupled_layers(grid_size=(40, 40))
    # Add some initial activity
    sim.depolarize_region(0, (20, 10), 5)  # Only top layer
    sim.depolarize_region(2, (20, 30), 5)  # Only bottom layer

    viz = MultiLayerVisualizer(sim)
    viz.render(animate=True)


def demo_tightly_coupled():
    """Demo showing highly integrated layers."""
    print("=" * 70)
    print("  Tightly Coupled Layers Demo")
    print("=" * 70)
    print("\nLayers with strong vertical coupling.")
    print("Watch how activity in one layer quickly spreads to others.")
    print("=" * 70)

    sim = create_tightly_coupled_layers(grid_size=(40, 40))
    viz = MultiLayerVisualizer(sim)
    viz.render(animate=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--decoupled':
            demo_decoupled()
        elif sys.argv[1] == '--coupled':
            demo_tightly_coupled()
        else:
            demo()
    else:
        demo()
