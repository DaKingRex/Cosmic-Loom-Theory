"""
Morphogenetic Field Simulator for Cosmic Loom Theory

Implements bioelectric control of morphogenesis - demonstrating how
bioelectric patterns encode and maintain morphological "memory".

From CLT perspective:
Morphogenesis IS coherence maintenance at the tissue scale. The bioelectric
field doesn't just correlate with form - it CAUSES form by coordinating
cellular behavior across spatial scales. Pattern memory in the bioelectric
field is a direct manifestation of Loomfield coherence.

Key concepts from Levin lab research:
- Bioelectric prepatterns precede and guide morphological development
- Voltage gradients encode positional information
- Target morphology is stored in bioelectric memory
- Injury triggers regeneration toward the stored pattern
- Manipulating bioelectric state can reprogram morphological outcomes

This simulator demonstrates:
1. Target pattern encoding in bioelectric fields
2. Pattern memory and stability
3. Regeneration after injury (pattern completion)
4. Repatterning (changing the target)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import matplotlib.animation as animation

# Handle imports for both module and direct execution
try:
    from .bioelectric import (
        BioelectricSimulator, V_REST, V_THRESHOLD, V_DEPOLARIZED,
        V_HYPERPOLARIZED, V_REVERSAL_NA, G_GAP_DEFAULT
    )
except ImportError:
    from bioelectric import (
        BioelectricSimulator, V_REST, V_THRESHOLD, V_DEPOLARIZED,
        V_HYPERPOLARIZED, V_REVERSAL_NA, G_GAP_DEFAULT
    )


# =============================================================================
# MORPHOGENETIC PATTERN DEFINITIONS
# =============================================================================

class PatternType(Enum):
    """Types of target morphogenetic patterns."""
    UNIFORM = "uniform"
    LEFT_RIGHT = "left_right"
    ANTERIOR_POSTERIOR = "anterior_posterior"
    RADIAL = "radial"
    STRIPE = "stripe"
    SPOT = "spot"
    CHECKERBOARD = "checkerboard"
    HEAD_TAIL = "head_tail"
    LETTER_T = "letter_t"
    CUSTOM = "custom"


def generate_pattern(pattern_type: PatternType,
                     grid_size: Tuple[int, int],
                     v_low: float = V_REST,
                     v_high: float = V_DEPOLARIZED) -> np.ndarray:
    """
    Generate a target bioelectric pattern.

    Args:
        pattern_type: Type of pattern to generate
        grid_size: (rows, cols) of the grid
        v_low: Low voltage value (typically resting)
        v_high: High voltage value (typically depolarized)

    Returns:
        2D array of target membrane potentials
    """
    rows, cols = grid_size
    pattern = np.ones((rows, cols)) * v_low

    if pattern_type == PatternType.UNIFORM:
        pass  # All at v_low

    elif pattern_type == PatternType.LEFT_RIGHT:
        # Left half depolarized, right half at rest
        pattern[:, :cols//2] = v_high

    elif pattern_type == PatternType.ANTERIOR_POSTERIOR:
        # Top (anterior) depolarized, bottom (posterior) at rest
        pattern[:rows//2, :] = v_high

    elif pattern_type == PatternType.RADIAL:
        # Central region depolarized
        center_r, center_c = rows // 2, cols // 2
        radius = min(rows, cols) // 4
        for r in range(rows):
            for c in range(cols):
                if (r - center_r)**2 + (c - center_c)**2 < radius**2:
                    pattern[r, c] = v_high

    elif pattern_type == PatternType.STRIPE:
        # Horizontal stripes
        n_stripes = 4
        stripe_height = rows // n_stripes
        for i in range(n_stripes):
            if i % 2 == 0:
                pattern[i*stripe_height:(i+1)*stripe_height, :] = v_high

    elif pattern_type == PatternType.SPOT:
        # Multiple spots
        spot_radius = min(rows, cols) // 8
        centers = [
            (rows//4, cols//4),
            (rows//4, 3*cols//4),
            (3*rows//4, cols//2),
        ]
        for cr, cc in centers:
            for r in range(rows):
                for c in range(cols):
                    if (r - cr)**2 + (c - cc)**2 < spot_radius**2:
                        pattern[r, c] = v_high

    elif pattern_type == PatternType.CHECKERBOARD:
        # Checkerboard pattern
        block_size = max(rows, cols) // 6
        for r in range(rows):
            for c in range(cols):
                if ((r // block_size) + (c // block_size)) % 2 == 0:
                    pattern[r, c] = v_high

    elif pattern_type == PatternType.HEAD_TAIL:
        # Head region (top) highly depolarized, gradient toward tail
        for r in range(rows):
            # Gradient from top to bottom
            gradient = 1 - (r / rows)
            pattern[r, :] = v_low + (v_high - v_low) * gradient

    elif pattern_type == PatternType.LETTER_T:
        # Letter T shape
        bar_thickness = max(rows, cols) // 8
        # Top bar
        pattern[:bar_thickness, :] = v_high
        # Vertical stem
        stem_start = cols//2 - bar_thickness//2
        stem_end = cols//2 + bar_thickness//2
        pattern[:, stem_start:stem_end] = v_high

    return pattern


# =============================================================================
# MORPHOGENETIC FIELD SIMULATOR
# =============================================================================

class MorphogeneticSimulator:
    """
    Simulator for bioelectric morphogenetic fields.

    Extends the basic bioelectric simulator with:
    - Target pattern encoding (morphogenetic memory)
    - Pattern attraction dynamics (cells move toward target)
    - Regeneration after injury
    - Repatterning capabilities
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (50, 50),
                 target_pattern: Optional[np.ndarray] = None,
                 pattern_type: PatternType = PatternType.LEFT_RIGHT,
                 attraction_strength: float = 0.01,
                 dx: float = 10.0,
                 dt: float = 0.1):
        """
        Initialize the morphogenetic simulator.

        Args:
            grid_size: (rows, cols) for the tissue
            target_pattern: Target bioelectric pattern (if None, generated from pattern_type)
            pattern_type: Type of pattern to generate if target_pattern is None
            attraction_strength: How strongly cells are attracted to target pattern
            dx: Spatial step (µm)
            dt: Time step (ms)
        """
        self.rows, self.cols = grid_size
        self.dx = dx
        self.dt = dt
        self.time = 0.0
        self.attraction_strength = attraction_strength

        # Create underlying bioelectric simulator
        self.sim = BioelectricSimulator(grid_size=grid_size, dx=dx, dt=dt)

        # Set target pattern
        if target_pattern is not None:
            self.target_pattern = target_pattern.copy()
        else:
            self.target_pattern = generate_pattern(pattern_type, grid_size)

        self.pattern_type = pattern_type

        # Initialize current state to target (stable initial condition)
        self.sim.Vm = self.target_pattern.copy()

        # Pattern fidelity history
        self.fidelity_history: List[float] = []
        self.coherence_history: List[float] = []

        # Injury tracking
        self.injury_mask = np.zeros((self.rows, self.cols), dtype=bool)

    def reset(self, preserve_target: bool = True):
        """
        Reset the simulation.

        Args:
            preserve_target: If True, keep the target pattern
        """
        self.sim.reset()
        if preserve_target:
            self.sim.Vm = self.target_pattern.copy()
        self.time = 0.0
        self.fidelity_history = []
        self.coherence_history = []
        self.injury_mask = np.zeros((self.rows, self.cols), dtype=bool)

    def set_target_pattern(self, pattern_type: PatternType = None,
                           custom_pattern: np.ndarray = None):
        """
        Set a new target pattern.

        This simulates "repatterning" - changing the morphogenetic memory.
        """
        if custom_pattern is not None:
            self.target_pattern = custom_pattern.copy()
            self.pattern_type = PatternType.CUSTOM
        elif pattern_type is not None:
            self.target_pattern = generate_pattern(
                pattern_type, (self.rows, self.cols)
            )
            self.pattern_type = pattern_type

    @property
    def Vm(self) -> np.ndarray:
        """Current membrane potential field."""
        return self.sim.Vm

    @Vm.setter
    def Vm(self, value: np.ndarray):
        """Set membrane potential field."""
        self.sim.Vm = value

    # =========================================================================
    # MORPHOGENETIC DYNAMICS
    # =========================================================================

    def _compute_pattern_attraction(self) -> np.ndarray:
        """
        Compute the "attraction current" that drives cells toward the target.

        This represents the morphogenetic memory - cells have an intrinsic
        tendency to return to the target bioelectric state.
        """
        deviation = self.target_pattern - self.sim.Vm

        # Attraction is proportional to deviation from target
        # But only for cells not in injured regions
        attraction = self.attraction_strength * deviation

        # Injured cells have weakened attraction (damaged memory)
        attraction[self.injury_mask] *= 0.1

        return attraction

    def step(self, n_steps: int = 1):
        """
        Advance simulation by n_steps.

        Combines:
        1. Standard bioelectric dynamics (ion channels, gap junctions)
        2. Morphogenetic attraction toward target pattern
        """
        for _ in range(n_steps):
            # Compute pattern attraction
            attraction = self._compute_pattern_attraction()

            # Add to external current (this drives cells toward target)
            original_I_ext = self.sim.I_ext.copy()
            self.sim.I_ext += attraction * 100  # Scale factor for current

            # Step the underlying simulator
            self.sim.step(1)

            # Restore original external current
            self.sim.I_ext = original_I_ext

            self.time += self.dt

    def run(self, duration: float, record_interval: float = 1.0):
        """Run simulation for specified duration."""
        n_steps = int(duration / self.dt)
        record_steps = max(1, int(record_interval / self.dt))

        for i in range(n_steps):
            self.step(1)

            if i % record_steps == 0:
                self.fidelity_history.append(self.compute_pattern_fidelity())
                self.coherence_history.append(self.sim.compute_spatial_coherence())

    # =========================================================================
    # INJURY AND REGENERATION
    # =========================================================================

    def create_injury(self, center: Tuple[int, int], radius: int,
                      scramble: bool = True):
        """
        Create an injury that disrupts the bioelectric pattern.

        Args:
            center: (row, col) center of injury
            radius: Radius of injury
            scramble: Whether to scramble voltages in injury region
        """
        r0, c0 = center

        for r in range(max(0, r0 - radius), min(self.rows, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(self.cols, c0 + radius + 1)):
                if (r - r0)**2 + (c - c0)**2 <= radius**2:
                    self.injury_mask[r, c] = True

                    if scramble:
                        # Random voltage in injury region
                        self.sim.Vm[r, c] = np.random.uniform(
                            V_HYPERPOLARIZED, V_DEPOLARIZED
                        )

        # Also damage gap junctions in injury region
        self.sim.create_injury(center, radius)

    def remove_region(self, center: Tuple[int, int], radius: int):
        """
        Remove a region entirely (simulate amputation).

        Sets region to resting potential and marks as injured.
        """
        r0, c0 = center

        for r in range(max(0, r0 - radius), min(self.rows, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(self.cols, c0 + radius + 1)):
                if (r - r0)**2 + (c - c0)**2 <= radius**2:
                    self.injury_mask[r, c] = True
                    self.sim.Vm[r, c] = V_REST

        self.sim.create_injury(center, radius)

    def heal_injury(self, center: Tuple[int, int], radius: int,
                    heal_rate: float = 0.8):
        """
        Heal an injury - restore gap junctions and pattern memory.

        This allows regeneration to proceed.
        """
        r0, c0 = center

        for r in range(max(0, r0 - radius), min(self.rows, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(self.cols, c0 + radius + 1)):
                if (r - r0)**2 + (c - c0)**2 <= radius**2:
                    # Partially restore pattern memory
                    if np.random.random() < heal_rate:
                        self.injury_mask[r, c] = False

        self.sim.heal_injury(center, radius, heal_rate)

    def amputate_half(self, side: str = 'right'):
        """
        Amputate half of the tissue.

        This is the classic regeneration experiment - can the remaining
        half regenerate the complete pattern?
        """
        if side == 'right':
            region = (self.rows // 2, 3 * self.cols // 4)
            radius = self.cols // 3
        elif side == 'left':
            region = (self.rows // 2, self.cols // 4)
            radius = self.cols // 3
        elif side == 'top':
            region = (self.rows // 4, self.cols // 2)
            radius = self.rows // 3
        else:  # bottom
            region = (3 * self.rows // 4, self.cols // 2)
            radius = self.rows // 3

        self.remove_region(region, radius)

    # =========================================================================
    # METRICS
    # =========================================================================

    def compute_pattern_fidelity(self) -> float:
        """
        Compute how closely the current pattern matches the target.

        Returns a value between 0 (no match) and 1 (perfect match).
        """
        # Normalize both patterns to [0, 1]
        current_norm = (self.sim.Vm - V_HYPERPOLARIZED) / (V_REVERSAL_NA - V_HYPERPOLARIZED)
        target_norm = (self.target_pattern - V_HYPERPOLARIZED) / (V_REVERSAL_NA - V_HYPERPOLARIZED)

        # Compute mean squared error
        mse = np.mean((current_norm - target_norm) ** 2)

        # Convert to fidelity (1 - normalized error)
        max_mse = 1.0  # Maximum possible MSE for normalized data
        fidelity = 1 - np.sqrt(mse / max_mse)

        return float(np.clip(fidelity, 0, 1))

    def compute_regeneration_progress(self) -> float:
        """
        Compute regeneration progress for injured regions only.

        Returns how much the injured regions have recovered toward target.
        """
        if not np.any(self.injury_mask):
            return 1.0  # No injury = fully regenerated

        injured_current = self.sim.Vm[self.injury_mask]
        injured_target = self.target_pattern[self.injury_mask]

        # Normalize
        current_norm = (injured_current - V_HYPERPOLARIZED) / (V_REVERSAL_NA - V_HYPERPOLARIZED)
        target_norm = (injured_target - V_HYPERPOLARIZED) / (V_REVERSAL_NA - V_HYPERPOLARIZED)

        # Correlation as regeneration metric
        if len(current_norm) < 2:
            return 1.0

        corr = np.corrcoef(current_norm, target_norm)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Map [-1, 1] to [0, 1]
        progress = (corr + 1) / 2

        return float(progress)

    def map_to_er_space(self) -> Dict[str, float]:
        """Map morphogenetic state to éR phase space."""
        base_er = self.sim.map_to_er_space()

        # Modulate by pattern fidelity
        fidelity = self.compute_pattern_fidelity()

        # Higher fidelity = more coherent = lower frequency
        freq = base_er['frequency'] * (2 - fidelity)
        freq = np.clip(freq, 0.3, 4.5)

        er = base_er['energy_present'] / (freq ** 2)

        return {
            'energy_present': base_er['energy_present'],
            'frequency': float(freq),
            'energy_resistance': float(er),
            'pattern_fidelity': float(fidelity),
            'coherence': base_er['coherence']
        }


# =============================================================================
# PRESETS
# =============================================================================

def create_stable_pattern(pattern_type: PatternType = PatternType.LEFT_RIGHT,
                          grid_size: Tuple[int, int] = (50, 50)) -> MorphogeneticSimulator:
    """Create simulator with a stable target pattern."""
    return MorphogeneticSimulator(
        grid_size=grid_size,
        pattern_type=pattern_type,
        attraction_strength=0.02
    )


def create_regeneration_scenario(pattern_type: PatternType = PatternType.LEFT_RIGHT,
                                 grid_size: Tuple[int, int] = (50, 50)) -> MorphogeneticSimulator:
    """
    Create a regeneration scenario.

    Starts with complete pattern, then amputates half.
    """
    sim = MorphogeneticSimulator(
        grid_size=grid_size,
        pattern_type=pattern_type,
        attraction_strength=0.015
    )

    # Amputate right half
    sim.amputate_half('right')

    # Partially heal to allow regeneration
    sim.heal_injury((grid_size[0]//2, 3*grid_size[1]//4), grid_size[1]//3, heal_rate=0.5)

    return sim


def create_repatterning_scenario(grid_size: Tuple[int, int] = (50, 50)) -> MorphogeneticSimulator:
    """
    Create a repatterning scenario.

    Starts with one pattern, then changes the target to see reorganization.
    """
    # Start with left-right pattern
    sim = MorphogeneticSimulator(
        grid_size=grid_size,
        pattern_type=PatternType.LEFT_RIGHT,
        attraction_strength=0.01
    )

    # Will need to call set_target_pattern() to trigger repatterning
    return sim


def create_cancer_scenario(grid_size: Tuple[int, int] = (50, 50)) -> MorphogeneticSimulator:
    """
    Create a "cancer-like" scenario with disrupted pattern memory.

    Pattern memory is weakened, allowing chaotic growth.
    """
    sim = MorphogeneticSimulator(
        grid_size=grid_size,
        pattern_type=PatternType.RADIAL,
        attraction_strength=0.005  # Very weak attraction
    )

    # Create central injury with scrambled voltages
    center = (grid_size[0]//2, grid_size[1]//2)
    sim.create_injury(center, radius=8, scramble=True)

    # Don't heal - keep pattern memory disrupted
    return sim


# =============================================================================
# VISUALIZER
# =============================================================================

class MorphogeneticVisualizer:
    """
    Interactive visualizer for morphogenetic field dynamics.

    Shows:
    - Current bioelectric pattern
    - Target pattern
    - Pattern fidelity over time
    - Regeneration progress
    """

    COLORMAP_COLORS = [
        (0.1, 0.1, 0.5),    # Deep blue - hyperpolarized
        (0.2, 0.4, 0.8),    # Blue
        (0.2, 0.7, 0.3),    # Green - resting
        (0.9, 0.8, 0.2),    # Yellow
        (0.9, 0.3, 0.1),    # Red - depolarized
    ]

    def __init__(self, simulator: MorphogeneticSimulator):
        """Initialize the visualizer."""
        self.sim = simulator
        self.fig = None
        self.ax_current = None
        self.ax_target = None
        self.ax_difference = None
        self.ax_fidelity = None
        self.im_current = None
        self.im_target = None
        self.im_diff = None

        self.cmap = LinearSegmentedColormap.from_list(
            'membrane_potential', self.COLORMAP_COLORS, N=256
        )
        self.diff_cmap = plt.cm.RdBu_r

        self.anim = None

    def setup_plot(self, figsize: Tuple[int, int] = (16, 10)):
        """Set up the figure."""
        self.fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')

        gs = GridSpec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

        # Current pattern
        self.ax_current = self.fig.add_subplot(gs[0, 0])
        self.ax_current.set_facecolor('#16213e')

        # Target pattern
        self.ax_target = self.fig.add_subplot(gs[0, 1])
        self.ax_target.set_facecolor('#16213e')

        # Difference map
        self.ax_difference = self.fig.add_subplot(gs[0, 2])
        self.ax_difference.set_facecolor('#16213e')

        # Fidelity plot
        self.ax_fidelity = self.fig.add_subplot(gs[1, :2])
        self.ax_fidelity.set_facecolor('#16213e')

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[1, 2])
        self.ax_info.set_facecolor('#16213e')
        self.ax_info.axis('off')

        for ax in [self.ax_current, self.ax_target, self.ax_difference, self.ax_fidelity]:
            for spine in ax.spines.values():
                spine.set_color('#4a5568')
            ax.tick_params(colors='#a0aec0')

    def _normalize_Vm(self, Vm: np.ndarray) -> np.ndarray:
        """Normalize Vm to [0, 1] range."""
        Vm_norm = (Vm - V_HYPERPOLARIZED) / (V_REVERSAL_NA - V_HYPERPOLARIZED)
        return np.clip(Vm_norm, 0, 1)

    def _draw_patterns(self):
        """Draw current and target patterns."""
        current_norm = self._normalize_Vm(self.sim.Vm)
        target_norm = self._normalize_Vm(self.sim.target_pattern)
        difference = current_norm - target_norm

        # Current pattern
        if self.im_current is None:
            self.im_current = self.ax_current.imshow(
                current_norm, cmap=self.cmap, aspect='equal',
                origin='lower', vmin=0, vmax=1, interpolation='bilinear'
            )
        else:
            self.im_current.set_data(current_norm)

        fidelity = self.sim.compute_pattern_fidelity()
        self.ax_current.set_title(
            f'Current Pattern\nFidelity: {fidelity:.3f}',
            color='#f7fafc', fontsize=12, fontweight='bold'
        )

        # Overlay injury region
        if np.any(self.sim.injury_mask):
            injury_overlay = np.ma.masked_where(~self.sim.injury_mask, np.ones_like(current_norm))
            self.ax_current.contour(injury_overlay, levels=[0.5], colors='red',
                                    linewidths=2, linestyles='dashed')

        # Target pattern
        if self.im_target is None:
            self.im_target = self.ax_target.imshow(
                target_norm, cmap=self.cmap, aspect='equal',
                origin='lower', vmin=0, vmax=1, interpolation='bilinear'
            )
        else:
            self.im_target.set_data(target_norm)

        self.ax_target.set_title(
            f'Target Pattern\n({self.sim.pattern_type.value})',
            color='#f7fafc', fontsize=12, fontweight='bold'
        )

        # Difference map
        if self.im_diff is None:
            self.im_diff = self.ax_difference.imshow(
                difference, cmap=self.diff_cmap, aspect='equal',
                origin='lower', vmin=-0.5, vmax=0.5, interpolation='bilinear'
            )
        else:
            self.im_diff.set_data(difference)

        regen_progress = self.sim.compute_regeneration_progress()
        self.ax_difference.set_title(
            f'Difference (Current - Target)\nRegeneration: {regen_progress:.1%}',
            color='#f7fafc', fontsize=12, fontweight='bold'
        )

    def _draw_fidelity_plot(self):
        """Draw fidelity time series."""
        self.ax_fidelity.clear()
        self.ax_fidelity.set_facecolor('#16213e')

        if len(self.sim.fidelity_history) > 1:
            t = np.arange(len(self.sim.fidelity_history))

            self.ax_fidelity.plot(t, self.sim.fidelity_history, '-',
                                  color='#ffd700', linewidth=2.5, label='Pattern Fidelity')
            self.ax_fidelity.fill_between(t, self.sim.fidelity_history,
                                          alpha=0.3, color='#ffd700')

            if len(self.sim.coherence_history) == len(self.sim.fidelity_history):
                self.ax_fidelity.plot(t, self.sim.coherence_history, '--',
                                      color='#00ff88', linewidth=1.5, label='Spatial Coherence')

        self.ax_fidelity.axhline(y=0.9, color='#48bb78', linestyle=':',
                                 linewidth=1, alpha=0.7, label='High fidelity threshold')
        self.ax_fidelity.axhline(y=0.5, color='#f6ad55', linestyle=':',
                                 linewidth=1, alpha=0.7, label='Pattern disrupted')

        self.ax_fidelity.set_xlim(0, max(len(self.sim.fidelity_history), 100))
        self.ax_fidelity.set_ylim(0, 1)
        self.ax_fidelity.set_xlabel('Time Steps', color='#e2e8f0', fontsize=11)
        self.ax_fidelity.set_ylabel('Fidelity / Coherence', color='#e2e8f0', fontsize=11)
        self.ax_fidelity.set_title('Pattern Fidelity Over Time', color='#f7fafc', fontsize=12)
        self.ax_fidelity.legend(loc='lower right', fontsize=9,
                               facecolor='#2d3748', edgecolor='#4a5568',
                               labelcolor='#e2e8f0')

        for spine in self.ax_fidelity.spines.values():
            spine.set_color('#4a5568')
        self.ax_fidelity.tick_params(colors='#a0aec0')

    def _draw_info(self):
        """Draw info panel."""
        self.ax_info.clear()
        self.ax_info.set_facecolor('#16213e')
        self.ax_info.axis('off')

        er_state = self.sim.map_to_er_space()
        n_injured = np.sum(self.sim.injury_mask)
        total_cells = self.sim.rows * self.sim.cols

        info_text = (
            f"MORPHOGENETIC FIELD\n"
            f"{'=' * 25}\n\n"
            f"Pattern: {self.sim.pattern_type.value}\n"
            f"Time: {self.sim.time:.1f} ms\n\n"
            f"Fidelity: {er_state['pattern_fidelity']:.3f}\n"
            f"Coherence: {er_state['coherence']:.3f}\n\n"
            f"Injured: {n_injured}/{total_cells}\n"
            f"  ({100*n_injured/total_cells:.1f}%)\n\n"
            f"éR Mapping:\n"
            f"  EP = {er_state['energy_present']:.2f}\n"
            f"  f = {er_state['frequency']:.2f}\n"
            f"  éR = {er_state['energy_resistance']:.2f}"
        )

        self.ax_info.text(0.1, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, color='#e2e8f0', family='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='#2d3748',
                                  edgecolor='#4a5568', alpha=0.9))

    def _setup_controls(self):
        """Set up interactive controls."""
        # Buttons
        ax_injure = self.fig.add_axes([0.05, 0.02, 0.1, 0.03])
        ax_amputate = self.fig.add_axes([0.16, 0.02, 0.1, 0.03])
        ax_heal = self.fig.add_axes([0.27, 0.02, 0.1, 0.03])
        ax_reset = self.fig.add_axes([0.38, 0.02, 0.1, 0.03])
        ax_repattern = self.fig.add_axes([0.49, 0.02, 0.1, 0.03])

        self.btn_injure = Button(ax_injure, 'Injure', color='#fc8181', hovercolor='#f56565')
        self.btn_amputate = Button(ax_amputate, 'Amputate', color='#f6ad55', hovercolor='#ed8936')
        self.btn_heal = Button(ax_heal, 'Heal', color='#68d391', hovercolor='#48bb78')
        self.btn_reset = Button(ax_reset, 'Reset', color='#a0aec0', hovercolor='#718096')
        self.btn_repattern = Button(ax_repattern, 'New Pattern', color='#9f7aea', hovercolor='#805ad5')

        # Pattern selector
        ax_pattern = self.fig.add_axes([0.65, 0.01, 0.15, 0.06])
        ax_pattern.set_facecolor('#2d3748')
        patterns = ['left_right', 'radial', 'stripe', 'head_tail']
        self.radio_pattern = RadioButtons(ax_pattern, patterns, active=0)
        for label in self.radio_pattern.labels:
            label.set_color('#e2e8f0')
            label.set_fontsize(8)

        # Attraction strength slider
        ax_attraction = self.fig.add_axes([0.82, 0.02, 0.15, 0.02])
        self.slider_attraction = Slider(ax_attraction, 'Memory', 0.001, 0.05,
                                         valinit=self.sim.attraction_strength,
                                         valstep=0.001, color='#ffd700')

        # Connect callbacks
        self.btn_injure.on_clicked(self._on_injure)
        self.btn_amputate.on_clicked(self._on_amputate)
        self.btn_heal.on_clicked(self._on_heal)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_repattern.on_clicked(self._on_repattern)
        self.radio_pattern.on_clicked(self._on_pattern_change)
        self.slider_attraction.on_changed(self._on_attraction_change)

    def _on_injure(self, event):
        """Create injury at random location."""
        center = (
            np.random.randint(self.sim.rows // 4, 3 * self.sim.rows // 4),
            np.random.randint(self.sim.cols // 4, 3 * self.sim.cols // 4)
        )
        self.sim.create_injury(center, radius=6, scramble=True)
        self._update_display()

    def _on_amputate(self, event):
        """Amputate right half."""
        self.sim.amputate_half('right')
        self._update_display()

    def _on_heal(self, event):
        """Heal all injuries."""
        # Heal everywhere
        for r in range(self.sim.rows):
            for c in range(self.sim.cols):
                if self.sim.injury_mask[r, c]:
                    self.sim.heal_injury((r, c), radius=1, heal_rate=0.9)
        self._update_display()

    def _on_reset(self, event):
        """Reset simulation."""
        self.sim.reset()
        self._update_display()

    def _on_repattern(self, event):
        """Set random new pattern."""
        patterns = [PatternType.LEFT_RIGHT, PatternType.RADIAL,
                   PatternType.STRIPE, PatternType.HEAD_TAIL, PatternType.SPOT]
        new_pattern = np.random.choice(patterns)
        self.sim.set_target_pattern(new_pattern)
        self._update_display()

    def _on_pattern_change(self, label):
        """Change target pattern based on radio selection."""
        pattern_map = {
            'left_right': PatternType.LEFT_RIGHT,
            'radial': PatternType.RADIAL,
            'stripe': PatternType.STRIPE,
            'head_tail': PatternType.HEAD_TAIL,
        }
        self.sim.set_target_pattern(pattern_map.get(label, PatternType.LEFT_RIGHT))
        self._update_display()

    def _on_attraction_change(self, val):
        """Update pattern memory strength."""
        self.sim.attraction_strength = val

    def _update_display(self):
        """Update all display elements."""
        self._draw_patterns()
        self._draw_fidelity_plot()
        self._draw_info()
        self.fig.canvas.draw_idle()

    def _animate(self, frame):
        """Animation callback."""
        self.sim.step(10)

        # Record metrics
        self.sim.fidelity_history.append(self.sim.compute_pattern_fidelity())
        self.sim.coherence_history.append(self.sim.sim.compute_spatial_coherence())

        self._draw_patterns()
        self._draw_fidelity_plot()
        self._draw_info()

        return [self.im_current, self.im_target, self.im_diff]

    def render(self, animate: bool = True, save_path: Optional[str] = None):
        """Render the visualization."""
        self.setup_plot()
        self._draw_patterns()
        self._draw_fidelity_plot()
        self._draw_info()
        self._setup_controls()

        self.fig.suptitle(
            'Morphogenetic Field Simulator - Bioelectric Pattern Memory',
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
    """Run demo of morphogenetic field simulation."""
    print("=" * 70)
    print("  Morphogenetic Field Simulator")
    print("  Bioelectric Pattern Memory & Regeneration (CLT Phase 2.1)")
    print("=" * 70)
    print("\nThis simulator demonstrates key Levin lab findings:")
    print("  • Bioelectric patterns ENCODE morphological memory")
    print("  • Cells have intrinsic tendency to return to target pattern")
    print("  • Injury disrupts pattern, regeneration restores it")
    print("  • Pattern memory can be reprogrammed")
    print("\nControls:")
    print("  • Injure: Create local pattern disruption")
    print("  • Amputate: Remove half the tissue")
    print("  • Heal: Restore pattern memory/gap junctions")
    print("  • New Pattern: Change the morphogenetic target")
    print("  • Memory slider: Strength of pattern attraction")
    print("\nWatch how the tissue REGENERATES toward the target pattern!")
    print("=" * 70)

    sim = create_stable_pattern(PatternType.LEFT_RIGHT, grid_size=(50, 50))
    viz = MorphogeneticVisualizer(sim)
    viz.render(animate=True)


def demo_regeneration():
    """Demo showing regeneration after amputation."""
    print("=" * 70)
    print("  Regeneration Demo")
    print("=" * 70)
    print("\nStarting with half the pattern removed.")
    print("Watch the tissue regenerate toward the complete pattern.")
    print("=" * 70)

    sim = create_regeneration_scenario(PatternType.RADIAL, grid_size=(50, 50))
    viz = MorphogeneticVisualizer(sim)
    viz.render(animate=True)


def demo_repatterning():
    """Demo showing repatterning when target changes."""
    print("=" * 70)
    print("  Repatterning Demo")
    print("=" * 70)
    print("\nDemonstrating how tissue reorganizes when the")
    print("target pattern (morphogenetic memory) is changed.")
    print("Use the pattern selector to change targets.")
    print("=" * 70)

    sim = create_repatterning_scenario(grid_size=(50, 50))
    viz = MorphogeneticVisualizer(sim)
    viz.render(animate=True)


def demo_cancer():
    """Demo showing disrupted pattern memory (cancer-like)."""
    print("=" * 70)
    print("  Cancer-like Disruption Demo")
    print("=" * 70)
    print("\nSimulating a scenario where pattern memory is weakened.")
    print("The tissue loses its ability to maintain organized form.")
    print("This represents how cancer may involve bioelectric")
    print("pattern memory disruption (Levin hypothesis).")
    print("=" * 70)

    sim = create_cancer_scenario(grid_size=(50, 50))
    viz = MorphogeneticVisualizer(sim)
    viz.render(animate=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--regeneration':
            demo_regeneration()
        elif sys.argv[1] == '--repattern':
            demo_repatterning()
        elif sys.argv[1] == '--cancer':
            demo_cancer()
        else:
            demo()
    else:
        demo()
