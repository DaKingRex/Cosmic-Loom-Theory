"""
Loomfield Wave Propagation Simulator

The centerpiece visualization for Cosmic Loom Theory, implementing the
Loomfield dynamics equation from CLT v1.1 Section 7.3:

    ∇²L - (1/v_L²)(∂²L/∂t²) = κ_L · ρ_coh

Where:
    L(r,t)  = Loomfield amplitude (coherence-relevant field structure)
    v_L     = propagation speed for coherence modes in biological tissue
    κ_L     = coupling constant (how coherence density shapes field dynamics)
    ρ_coh   = local coherence density (acts as source term)

This visualization demonstrates:
    - Real-time wave propagation of the Loomfield
    - Color-coded coherence (bright = high, dark = low)
    - Interactive source placement (click to add coherence sources)
    - Perturbation dynamics (watch the field respond to disruption)
    - Healthy state: stable, breathing oscillation
    - Pathology: boundary collapse, domain fragmentation
    - Healing: re-coupling, coherence restoration

The Loomfield is not a new fundamental force—it is an effective field capturing
how bioelectric activity, biophotons, and structural organization collectively
support large-scale biological integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, List, Callable
import warnings
warnings.filterwarnings('ignore')


class LoomfieldSimulator:
    """
    2D numerical solver for the Loomfield wave equation.

    Uses finite difference method with absorbing boundary conditions
    to simulate wave propagation and coherence dynamics.
    """

    def __init__(self,
                 grid_size: int = 200,
                 domain_size: float = 10.0,
                 v_L: float = 1.0,
                 kappa_L: float = 2.0,
                 damping: float = 0.001):
        """
        Initialize the Loomfield simulator.

        Args:
            grid_size: Number of grid points per dimension
            domain_size: Physical size of the domain
            v_L: Loomfield propagation speed
            kappa_L: Coupling constant for coherence density
            damping: Dissipation rate (biological energy loss)
        """
        self.N = grid_size
        self.L_domain = domain_size
        self.dx = domain_size / grid_size
        self.v_L = v_L
        self.kappa_L = kappa_L
        self.damping = damping

        # CFL condition for stability: dt < dx / (v_L * sqrt(2))
        self.dt = 0.4 * self.dx / (v_L * np.sqrt(2))

        # Field arrays: current, previous, coherence density
        self.L = np.zeros((self.N, self.N))          # Current Loomfield
        self.L_prev = np.zeros((self.N, self.N))     # Previous timestep
        self.rho_coh = np.zeros((self.N, self.N))    # Coherence density sources

        # Persistent sources (active coherence nodes)
        self.sources: List[dict] = []

        # Perturbation tracking for coherence disruption
        self.perturbation_count = 0
        self.last_perturbation_time = -10.0

        # Coordinate grids
        x = np.linspace(-domain_size/2, domain_size/2, self.N)
        y = np.linspace(-domain_size/2, domain_size/2, self.N)
        self.X, self.Y = np.meshgrid(x, y)

        # Time tracking
        self.time = 0.0
        self.step_count = 0

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 2D Laplacian using central differences."""
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        ) / (self.dx ** 2)
        return lap

    def add_source(self, x: float, y: float, strength: float = 1.0,
                   radius: float = 0.3, frequency: float = 2.0,
                   phase: float = 0.0,
                   source_type: str = 'oscillating'):
        """
        Add a coherence source at position (x, y).

        Args:
            x, y: Position in domain coordinates
            strength: Source amplitude
            radius: Spatial extent of source
            frequency: Oscillation frequency (for oscillating sources)
            phase: Initial phase offset (for phase-locking multiple sources)
            source_type: 'oscillating', 'pulse', or 'static'
        """
        self.sources.append({
            'x': x, 'y': y,
            'strength': strength,
            'radius': radius,
            'frequency': frequency,
            'phase': phase,
            'type': source_type,
            'birth_time': self.time
        })

    def clear_sources(self):
        """Remove all coherence sources."""
        self.sources = []
        self.rho_coh = np.zeros((self.N, self.N))

    def add_perturbation(self, x: float, y: float, strength: float = 2.0,
                         radius: float = 1.0):
        """
        Add a perturbation (disruption) to the field.

        Perturbations inject HIGH-FREQUENCY chaotic noise that disrupts
        spatial coherence. This fragments phase relationships, lowering Q
        even though total energy may increase.

        Key insight: Coherence requires smooth, organized structure.
        Perturbations destroy this by adding sharp, uncorrelated variations.
        """
        dist_sq = (self.X - x)**2 + (self.Y - y)**2
        envelope = np.exp(-dist_sq / (2 * radius**2))

        # HIGH-FREQUENCY noise: minimal smoothing = disrupts spatial correlation
        noise = np.random.randn(self.N, self.N)
        # Very light smoothing (sigma=1) keeps high-frequency content
        noise = gaussian_filter(noise, sigma=1)

        # Also scramble the velocity field (L - L_prev) to disrupt wave coherence
        velocity_noise = np.random.randn(self.N, self.N)
        velocity_noise = gaussian_filter(velocity_noise, sigma=1)

        # Apply perturbation to both field and velocity
        self.L += strength * envelope * noise
        self.L_prev += strength * 0.5 * envelope * velocity_noise  # Scramble momentum

        # Track perturbation
        self.perturbation_count += 1
        self.last_perturbation_time = self.time

    def _update_coherence_density(self):
        """
        Update the coherence density field from all sources.

        For true wave propagation, sources must oscillate between positive
        and negative values (like a vibrating membrane or dipole antenna).
        This creates the push-pull dynamics that generate propagating waves.
        """
        self.rho_coh = np.zeros((self.N, self.N))

        for src in self.sources:
            dist_sq = (self.X - src['x'])**2 + (self.Y - src['y'])**2
            spatial = np.exp(-dist_sq / (2 * src['radius']**2))

            if src['type'] == 'oscillating':
                # TRUE OSCILLATION: goes positive AND negative
                # This is essential for generating propagating waves
                phase = src.get('phase', 0.0)
                temporal = np.sin(2 * np.pi * src['frequency'] * self.time + phase)
                amplitude = src['strength'] * temporal
            elif src['type'] == 'pulse':
                # Pulse: bipolar wavelet (Mexican hat style)
                dt_birth = self.time - src['birth_time']
                if dt_birth > 0:
                    # Damped oscillation for pulse
                    temporal = np.cos(4 * np.pi * dt_birth) * np.exp(-dt_birth / 0.3)
                    amplitude = src['strength'] * temporal
                else:
                    amplitude = 0.0
            else:  # static
                amplitude = src['strength']

            self.rho_coh += amplitude * spatial

    def step(self, n_steps: int = 1):
        """
        Advance the simulation by n_steps timesteps.

        Implements the driven wave equation:
        ∂²L/∂t² = v_L² * ∇²L - damping * ∂L/∂t

        Sources inject energy directly into the field velocity, creating
        outward-propagating waves. This is physically equivalent to
        oscillating membranes or antennas driving the Loomfield.
        """
        for _ in range(n_steps):
            self._update_coherence_density()

            # Compute Laplacian
            lap_L = self._laplacian(self.L)

            # Wave equation (propagation + damping)
            c2dt2 = (self.v_L * self.dt) ** 2

            L_new = (
                2 * self.L - self.L_prev +
                c2dt2 * lap_L -
                self.damping * (self.L - self.L_prev)
            )

            # Source injection: directly drive field velocity
            # This creates strong, visible wave emission from sources
            # Using dt (not dt²) gives physically meaningful coupling
            L_new += self.dt * self.kappa_L * self.rho_coh

            # Absorbing boundary conditions (Mur's first-order ABC)
            # Prevents reflections at domain edges
            c_ratio = self.v_L * self.dt / self.dx

            # Left boundary
            L_new[:, 0] = self.L[:, 1] + (c_ratio - 1)/(c_ratio + 1) * (L_new[:, 1] - self.L[:, 0])
            # Right boundary
            L_new[:, -1] = self.L[:, -2] + (c_ratio - 1)/(c_ratio + 1) * (L_new[:, -2] - self.L[:, -1])
            # Bottom boundary
            L_new[0, :] = self.L[1, :] + (c_ratio - 1)/(c_ratio + 1) * (L_new[1, :] - self.L[0, :])
            # Top boundary
            L_new[-1, :] = self.L[-2, :] + (c_ratio - 1)/(c_ratio + 1) * (L_new[-2, :] - self.L[-1, :])

            # Update arrays
            self.L_prev = self.L.copy()
            self.L = L_new

            self.time += self.dt
            self.step_count += 1

    def get_total_coherence(self) -> float:
        """
        Calculate Q: TRUE spatial coherence (ENERGY-INDEPENDENT).

        Coherence measures how ORGANIZED the field is - how well different
        regions maintain consistent phase relationships. This is computed as:

        1. Spatial autocorrelation: How correlated is the field at different distances?
        2. Roughness penalty: High-frequency noise = low coherence

        CRITICAL: Q is NORMALIZED by energy, so:
        - High energy + chaos = LOW Q
        - Low energy + organization = HIGH Q
        - Perturbations add noise → Q DROPS

        Q range: [0, ~2] where 1.0 = baseline organized field
        """
        total_energy = np.sum(self.L**2)
        if total_energy < 1e-10:
            return 0.0

        # Roughness: high-frequency content (Laplacian magnitude)
        # Chaotic/noisy fields have high Laplacian energy relative to total energy
        laplacian = self._laplacian(self.L)
        laplacian_energy = np.sum(laplacian**2)
        roughness = laplacian_energy / (total_energy + 1e-10)

        # Spatial correlation at distance (measures coherence length)
        # Organized waves maintain correlation; noise decorrelates rapidly
        shift = max(1, self.N // 10)  # ~10% of domain
        L_shifted_x = np.roll(self.L, shift, axis=0)
        L_shifted_y = np.roll(self.L, shift, axis=1)
        corr_x = np.sum(self.L * L_shifted_x) / (total_energy + 1e-10)
        corr_y = np.sum(self.L * L_shifted_y) / (total_energy + 1e-10)
        correlation = (corr_x + corr_y) / 2.0

        # Q = coherence metric (ENERGY-INDEPENDENT)
        # High correlation + low roughness = high Q
        # Formula tuned so organized waves → Q ≈ 1-2, noise → Q ≈ 0.1-0.5
        Q = (1.0 + correlation) / (1.0 + 0.001 * roughness)

        return max(0.0, Q)

    def get_consciousness_metric(self) -> float:
        """
        Calculate C_bio from CLT v1.1 Section 7.6:
        C_bio = ∫ ρ_coh * Λ dV

        This measures how well source activity (ρ_coh) couples to
        field dynamics (Λ). High C_bio = sources are effectively
        driving coherent field activity.
        """
        # Approximate Λ as field gradient magnitude (flow indicator)
        grad_x = np.gradient(self.L, self.dx, axis=1)
        grad_y = np.gradient(self.L, self.dx, axis=0)
        lambda_field = np.sqrt(grad_x**2 + grad_y**2)

        # Only count where sources and flow are co-located and in-phase
        # Use absolute values to measure coupling magnitude
        coupling = np.abs(self.rho_coh) * lambda_field

        return np.sum(coupling) * self.dx**2

    def get_field_energy(self) -> float:
        """Get total field energy (for comparison with coherence)."""
        return np.sum(self.L**2) * self.dx**2

    def reset(self):
        """Reset simulation to initial state."""
        self.L = np.zeros((self.N, self.N))
        self.L_prev = np.zeros((self.N, self.N))
        self.rho_coh = np.zeros((self.N, self.N))
        self.sources = []
        self.time = 0.0
        self.step_count = 0


class LoomfieldVisualizer:
    """
    Interactive animated visualization of Loomfield dynamics.

    Features:
        - Real-time wave propagation animation
        - Color-coded coherence intensity
        - Click to add coherence sources
        - Parameter adjustment sliders
        - Preset scenarios (healthy, pathology, healing)
        - Perturbation mode
    """

    # Diverging colormap: negative (blue) -> zero (dark) -> positive (gold/white)
    # This reveals wave structure with clear positive/negative phases
    COLORMAP_COLORS = [
        (0.10, 0.20, 0.60),   # Deep blue - strong negative
        (0.20, 0.40, 0.80),   # Blue - negative
        (0.15, 0.25, 0.45),   # Dark blue - weak negative
        (0.05, 0.05, 0.10),   # Near black - zero/neutral
        (0.25, 0.20, 0.10),   # Dark gold - weak positive
        (0.70, 0.50, 0.20),   # Gold - positive
        (1.00, 0.85, 0.40),   # Bright gold - strong positive
        (1.00, 1.00, 0.90),   # White-gold - peak positive
    ]

    def __init__(self, grid_size: int = 200):
        """Initialize the visualizer with a simulator instance."""
        self.sim = LoomfieldSimulator(grid_size=grid_size)
        self.cmap = self._create_colormap()

        # Animation state
        self.animation = None
        self.running = True
        self.steps_per_frame = 3
        self.mode = 'add_source'  # 'add_source' or 'perturbation'

        # Plot elements
        self.fig = None
        self.ax_main = None
        self.im = None
        self.source_markers = []
        self.info_text = None

    def _create_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for Loomfield visualization."""
        return LinearSegmentedColormap.from_list(
            'loomfield_coherence',
            self.COLORMAP_COLORS,
            N=512
        )

    def setup_plot(self, figsize: Tuple[int, int] = (16, 10)):
        """Set up the matplotlib figure with all UI elements."""

        self.fig = plt.figure(figsize=figsize, facecolor='#0a0a12')
        self.fig.canvas.manager.set_window_title('Loomfield Wave Propagation - Cosmic Loom Theory')

        # Main visualization axis
        self.ax_main = self.fig.add_axes([0.05, 0.22, 0.6, 0.72])
        self.ax_main.set_facecolor('#0a0a12')
        self.ax_main.set_aspect('equal')

        # Remove axis spines and ticks for cleaner look
        for spine in self.ax_main.spines.values():
            spine.set_visible(False)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])

        # Initialize the image
        extent = [-self.sim.L_domain/2, self.sim.L_domain/2,
                  -self.sim.L_domain/2, self.sim.L_domain/2]
        self.im = self.ax_main.imshow(
            self.sim.L,
            extent=extent,
            origin='lower',
            cmap=self.cmap,
            vmin=-1.5,
            vmax=1.5,
            interpolation='bilinear'
        )

        # Title
        self.ax_main.set_title(
            'Loomfield Wave Propagation\n∇²L − (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh',
            fontsize=16, fontweight='bold', color='#e0e0e0', pad=10
        )

        # Info panel on the right
        self._setup_info_panel()

        # Parameter sliders
        self._setup_sliders()

        # Control buttons
        self._setup_buttons()

        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_info_panel(self):
        """Create the information panel on the right side."""

        # Theory box
        theory_text = (
            "COSMIC LOOM THEORY v1.1\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "WAVE VISUALIZATION:\n"
            "• Gold/white: Positive phase\n"
            "• Blue: Negative phase\n"
            "• Dark: Zero/neutral\n"
            "• Rings: Source locations\n\n"
            "Q = SPATIAL COHERENCE:\n"
            "• Measures COORDINATION\n"
            "• Phase-locked → HIGH Q\n"
            "• Chaotic/noisy → LOW Q\n"
            "• Perturbations DISRUPT Q\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "INTERACTIONS:\n"
            "• Click: Add wave source\n"
            "• Perturb: Add chaos\n"
            "• Watch interference!"
        )
        self.fig.text(0.72, 0.92, theory_text, fontsize=9,
                     color='#b0b0b0', family='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='#151525',
                              edgecolor='#303050', alpha=0.95))

        # Metrics display (will be updated in animation)
        self.metrics_text = self.fig.text(
            0.72, 0.38,
            "METRICS\n━━━━━━━━━━━\nInitializing...",
            fontsize=10, color='#80ffaa', family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor='#102010',
                     edgecolor='#305030', alpha=0.95)
        )

    def _setup_sliders(self):
        """Create parameter adjustment sliders."""
        slider_color = '#304060'

        # Propagation speed v_L
        ax_vL = self.fig.add_axes([0.15, 0.12, 0.35, 0.025])
        ax_vL.set_facecolor('#151525')
        self.slider_vL = Slider(
            ax_vL, 'v_L (speed)', 0.2, 3.0,
            valinit=self.sim.v_L, valstep=0.1,
            color='#40a0ff'
        )
        self.slider_vL.label.set_color('#a0a0a0')
        self.slider_vL.valtext.set_color('#80c0ff')

        # Coupling constant κ_L
        ax_kappa = self.fig.add_axes([0.15, 0.08, 0.35, 0.025])
        ax_kappa.set_facecolor('#151525')
        self.slider_kappa = Slider(
            ax_kappa, 'κ_L (coupling)', 0.0, 2.0,
            valinit=self.sim.kappa_L, valstep=0.05,
            color='#ff8040'
        )
        self.slider_kappa.label.set_color('#a0a0a0')
        self.slider_kappa.valtext.set_color('#ffa080')

        # Animation speed
        ax_speed = self.fig.add_axes([0.15, 0.04, 0.35, 0.025])
        ax_speed.set_facecolor('#151525')
        self.slider_speed = Slider(
            ax_speed, 'Speed', 1, 10,
            valinit=self.steps_per_frame, valstep=1,
            color='#40ff80'
        )
        self.slider_speed.label.set_color('#a0a0a0')
        self.slider_speed.valtext.set_color('#80ffa0')

        # Connect callbacks
        self.slider_vL.on_changed(self._update_vL)
        self.slider_kappa.on_changed(self._update_kappa)
        self.slider_speed.on_changed(self._update_speed)

    def _setup_buttons(self):
        """Create control buttons and preset selectors."""
        button_color = '#252540'
        hover_color = '#353560'

        # Preset scenarios
        ax_presets = self.fig.add_axes([0.72, 0.04, 0.12, 0.18])
        ax_presets.set_facecolor('#151525')
        self.radio_presets = RadioButtons(
            ax_presets,
            ('Empty', 'Healthy', 'Pathology', 'Healing'),
            activecolor='#40ff80'
        )
        for label in self.radio_presets.labels:
            label.set_color('#a0a0a0')
            label.set_fontsize(9)
        self.radio_presets.on_clicked(self._apply_preset)
        self.fig.text(0.72, 0.23, 'PRESETS', fontsize=10, color='#808080',
                     fontweight='bold')

        # Mode toggle
        ax_mode = self.fig.add_axes([0.86, 0.04, 0.12, 0.12])
        ax_mode.set_facecolor('#151525')
        self.radio_mode = RadioButtons(
            ax_mode,
            ('Add Source', 'Perturb'),
            activecolor='#ff8040'
        )
        for label in self.radio_mode.labels:
            label.set_color('#a0a0a0')
            label.set_fontsize(9)
        self.radio_mode.on_clicked(self._set_mode)
        self.fig.text(0.86, 0.17, 'CLICK MODE', fontsize=10, color='#808080',
                     fontweight='bold')

        # Play/Pause button
        ax_pause = self.fig.add_axes([0.55, 0.04, 0.06, 0.04])
        self.button_pause = Button(ax_pause, '⏸ Pause',
                                   color=button_color, hovercolor=hover_color)
        self.button_pause.label.set_color('#a0a0a0')
        self.button_pause.on_clicked(self._toggle_pause)

        # Reset button
        ax_reset = self.fig.add_axes([0.55, 0.09, 0.06, 0.04])
        self.button_reset = Button(ax_reset, '↺ Reset',
                                   color=button_color, hovercolor=hover_color)
        self.button_reset.label.set_color('#a0a0a0')
        self.button_reset.on_clicked(self._reset)

        # Clear sources button
        ax_clear = self.fig.add_axes([0.55, 0.14, 0.06, 0.04])
        self.button_clear = Button(ax_clear, '✕ Clear',
                                   color=button_color, hovercolor=hover_color)
        self.button_clear.label.set_color('#a0a0a0')
        self.button_clear.on_clicked(self._clear_sources)

    def _update_vL(self, val):
        """Update propagation speed."""
        self.sim.v_L = val
        # Recalculate dt for stability
        self.sim.dt = 0.4 * self.sim.dx / (val * np.sqrt(2))

    def _update_kappa(self, val):
        """Update coupling constant."""
        self.sim.kappa_L = val

    def _update_speed(self, val):
        """Update animation speed."""
        self.steps_per_frame = int(val)

    def _toggle_pause(self, event):
        """Toggle animation pause state."""
        self.running = not self.running
        self.button_pause.label.set_text('▶ Play' if not self.running else '⏸ Pause')

    def _reset(self, event):
        """Reset simulation to initial state."""
        self.sim.reset()
        self._clear_source_markers()

    def _clear_sources(self, event):
        """Clear all coherence sources."""
        self.sim.clear_sources()
        self._clear_source_markers()

    def _clear_source_markers(self):
        """Remove source marker graphics."""
        for marker in self.source_markers:
            marker.remove()
        self.source_markers = []

    def _set_mode(self, label):
        """Set click mode."""
        self.mode = 'add_source' if label == 'Add Source' else 'perturbation'

    def _on_click(self, event):
        """Handle mouse clicks on the main plot."""
        if event.inaxes != self.ax_main:
            return

        x, y = event.xdata, event.ydata

        if self.mode == 'add_source':
            # Add oscillating coherence source
            self.sim.add_source(x, y, strength=1.0, radius=0.4,
                               frequency=2.0, source_type='oscillating')
            # Visual marker
            marker = Circle((x, y), 0.2, fill=False,
                           edgecolor='#80ffaa', linewidth=2, alpha=0.8)
            self.ax_main.add_patch(marker)
            self.source_markers.append(marker)
        else:
            # Add perturbation (disruption)
            self.sim.add_perturbation(x, y, strength=-1.5, radius=0.6)
            # Visual flash (temporary)
            marker = Circle((x, y), 0.5, fill=True,
                           facecolor='#ff4040', alpha=0.5)
            self.ax_main.add_patch(marker)
            self.source_markers.append(marker)

    def _apply_preset(self, label):
        """Apply a preset scenario."""
        self.sim.reset()
        self._clear_source_markers()

        if label == 'Empty':
            pass  # Already reset

        elif label == 'Healthy':
            # PHASE-LOCKED sources: same frequency, coordinated phases
            # This creates coherent interference patterns (high Q)
            base_freq = 1.5

            # Central dominant source
            self.sim.add_source(0, 0, strength=1.5, radius=0.4,
                               frequency=base_freq, phase=0.0,
                               source_type='oscillating')

            # Ring of supporting sources - ALL SAME FREQUENCY AND PHASE
            # Phase-locking creates constructive interference = coherence
            for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                x = 2.5 * np.cos(angle)
                y = 2.5 * np.sin(angle)
                self.sim.add_source(x, y, strength=0.8, radius=0.3,
                                   frequency=base_freq, phase=0.0,
                                   source_type='oscillating')

        elif label == 'Pathology':
            # INCOHERENT sources: different frequencies AND random phases
            # This creates chaotic interference (low Q despite high energy)
            positions = [
                (-2.5, 2), (2, 2.5), (-1.5, -2.5), (3, -1.5),
                (-3.5, 0), (0.5, -3), (2.5, 0.5)
            ]
            # Different frequencies = can never synchronize
            frequencies = [0.7, 1.8, 2.5, 1.1, 3.0, 0.5, 2.2]
            # Random phases = additional disorder
            phases = [0.0, 1.2, 2.5, 0.8, 3.1, 1.9, 0.4]

            for (x, y), f, p in zip(positions, frequencies, phases):
                self.sim.add_source(x, y, strength=0.8, radius=0.3,
                                   frequency=f, phase=p,
                                   source_type='oscillating')

        elif label == 'Healing':
            # GRADUAL RE-COUPLING: sources with same frequency
            # Phase differences create interesting but organized patterns
            base_freq = 1.2

            # Ring of sources - same frequency creates gradual synchronization
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                r = 2.8
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                # Small phase offsets - will create traveling wave patterns
                phase = angle * 0.5  # Gradual phase gradient
                self.sim.add_source(x, y, strength=0.7, radius=0.35,
                                   frequency=base_freq, phase=phase,
                                   source_type='oscillating')

            # Central healing node - dominant attractor
            self.sim.add_source(0, 0, strength=1.8, radius=0.5,
                               frequency=base_freq, phase=0.0,
                               source_type='oscillating')

        # Add visual markers for sources
        for src in self.sim.sources:
            color = '#ffcc44' if label != 'Pathology' else '#ff6666'
            marker = Circle((src['x'], src['y']), 0.15, fill=False,
                           edgecolor=color, linewidth=2, alpha=0.9)
            self.ax_main.add_patch(marker)
            self.source_markers.append(marker)

    def _update_frame(self, frame):
        """Animation update function."""
        if self.running:
            self.sim.step(self.steps_per_frame)

        # Update the image data
        self.im.set_array(self.sim.L)

        # Auto-scale color range based on field amplitude for better visibility
        field_max = max(0.5, np.abs(self.sim.L).max() * 1.2)
        self.im.set_clim(-field_max, field_max)

        # Update metrics display
        Q = self.sim.get_total_coherence()
        energy = self.sim.get_field_energy()
        c_bio = self.sim.get_consciousness_metric()

        # Color-code Q based on coherence level
        if Q > 1.5:
            q_status = "(HIGH - integrated)"
        elif Q > 0.5:
            q_status = "(moderate)"
        else:
            q_status = "(LOW - fragmented)"

        metrics_str = (
            f"METRICS\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Time: {self.sim.time:.2f}\n\n"
            f"Q (spatial coherence):\n"
            f"  {Q:.3f} {q_status}\n\n"
            f"Field Energy:\n"
            f"  {energy:.3f}\n\n"
            f"C_bio (coupling):\n"
            f"  {c_bio:.4f}\n\n"
            f"Sources: {len(self.sim.sources)}\n"
            f"v_L={self.sim.v_L:.1f}  κ_L={self.sim.kappa_L:.2f}"
        )
        self.metrics_text.set_text(metrics_str)

        return [self.im, self.metrics_text]

    def run(self, interval: int = 30, save_path: Optional[str] = None):
        """
        Run the interactive visualization.

        Args:
            interval: Animation frame interval in milliseconds
            save_path: If provided, save a screenshot to this path
        """
        self.setup_plot()

        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=interval,
            blit=False,
            cache_frame_data=False
        )

        if save_path:
            self.fig.savefig(save_path, dpi=150,
                           facecolor=self.fig.get_facecolor(),
                           bbox_inches='tight', pad_inches=0.2)
            print(f"Screenshot saved to: {save_path}")

        plt.show()


def demo():
    """Run a demonstration of the Loomfield Wave visualizer."""
    print("=" * 70)
    print("  COSMIC LOOM THEORY: Loomfield Wave Propagation Simulator")
    print("=" * 70)
    print("""
The Loomfield L(r,t) is an effective field describing coherence-relevant
structure in biological systems. Its dynamics follow the wave equation:

    ∇²L − (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh

Where:
    • v_L   = propagation speed for coherence modes
    • κ_L   = coupling constant (coherence → field dynamics)
    • ρ_coh = local coherence density (sources of coherent activity)

INTERACTIONS:
    • Click on the field to add coherence sources
    • Use 'Perturb' mode to introduce disruptions
    • Try the presets to see different scenarios:
        - Healthy: Stable, integrated oscillation
        - Pathology: Fragmented, incoherent sources
        - Healing: Re-coupling toward integration
    • Adjust parameters with the sliders

Watch how consciousness might literally be a propagating wave of coherence.
    """)
    print("=" * 70)

    visualizer = LoomfieldVisualizer(grid_size=200)
    visualizer.run(interval=30)


if __name__ == "__main__":
    demo()
