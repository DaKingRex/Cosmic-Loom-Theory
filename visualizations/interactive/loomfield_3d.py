"""
3D Loomfield Wave Propagation Simulator

The centerpiece visualization for Cosmic Loom Theory, extending the 2D
Loomfield dynamics to three-dimensional space.

Design Philosophy:
    "The 3D Loomfield should feel like a calm, living volume whose internal
    order becomes visible as coherence increases — not a collection of objects,
    but a continuous system revealing structure through constraint."

Wave Equation (3D):
    ∇²L - (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh

Visual Principles:
    - Continuity over objects (volumetric field, not particles)
    - Gradient meaning (every visual encodes state)
    - Stillness matters (high coherence = visually calm)
    - Beauty from order, not effects

Three Layers of Engagement:
    Layer 1 - Passive viewer: Auto-rotating, minimal controls
    Layer 2 - Presenter mode: Preset paths, one-click scenarios
    Layer 3 - Explorer mode: Full controls, sliders, metrics
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class LoomfieldSimulator3D:
    """
    3D numerical solver for the Loomfield wave equation.

    Uses finite difference method with absorbing boundary conditions
    on all 6 faces to simulate wave propagation in 3D space.

    The Loomfield L(r,t) is a continuous field - not particles or objects.
    This simulator captures coherence dynamics in biological volumes.
    """

    def __init__(self,
                 grid_size: int = 64,
                 domain_size: float = 10.0,
                 v_L: float = 1.0,
                 kappa_L: float = 2.0,
                 damping: float = 0.002):
        """
        Initialize the 3D Loomfield simulator.

        Args:
            grid_size: Number of grid points per dimension (N×N×N)
            domain_size: Physical size of the cubic domain
            v_L: Loomfield propagation speed
            kappa_L: Coupling constant for coherence density
            damping: Dissipation rate (slightly higher in 3D for stability)
        """
        self.N = grid_size
        self.L_domain = domain_size
        self.dx = domain_size / grid_size
        self.v_L = v_L
        self.kappa_L = kappa_L
        self.damping = damping

        # CFL condition for stability: dt < dx / (v_L * sqrt(3))
        # In 3D, factor is sqrt(3) ≈ 1.73 for the diagonal
        self.dt = 0.4 * self.dx / (self.v_L * np.sqrt(3))

        # Field arrays: current, previous, coherence density
        self.L = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        self.L_prev = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        self.rho_coh = np.zeros((self.N, self.N, self.N), dtype=np.float32)

        # Persistent sources
        self.sources: List[dict] = []

        # Coordinate grids
        coords = np.linspace(-domain_size/2, domain_size/2, self.N)
        self.X, self.Y, self.Z = np.meshgrid(coords, coords, coords, indexing='ij')

        # Time tracking
        self.time = 0.0
        self.step_count = 0

        # Precompute for Laplacian (performance)
        self._dx2_inv = 1.0 / (self.dx ** 2)

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute 3D Laplacian using 7-point stencil finite differences.

        ∇²L = (L[i+1] + L[i-1] + L[j+1] + L[j-1] + L[k+1] + L[k-1] - 6*L[i,j,k]) / dx²
        """
        lap = np.zeros_like(field)

        # Interior points only (boundaries handled separately)
        lap[1:-1, 1:-1, 1:-1] = (
            field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +  # x neighbors
            field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +  # y neighbors
            field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -  # z neighbors
            6 * field[1:-1, 1:-1, 1:-1]
        ) * self._dx2_inv

        return lap

    def add_source(self, x: float, y: float, z: float,
                   strength: float = 1.0,
                   radius: float = 0.5,
                   frequency: float = 1.5,
                   phase: float = 0.0,
                   source_type: str = 'oscillating'):
        """
        Add a coherence source at position (x, y, z).

        In 3D, sources generate spherical waves that propagate outward.
        Phase-locked sources create organized interference patterns.
        """
        self.sources.append({
            'x': x, 'y': y, 'z': z,
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
        self.rho_coh = np.zeros((self.N, self.N, self.N), dtype=np.float32)

    def add_perturbation(self, x: float, y: float, z: float,
                         strength: float = 2.0, radius: float = 1.5):
        """
        Add a spherical perturbation (disruption) to the field.

        Injects high-frequency noise that fragments spatial coherence.
        Uses minimal smoothing to preserve high-frequency content that
        disrupts organized wave patterns.
        """
        dist_sq = (self.X - x)**2 + (self.Y - y)**2 + (self.Z - z)**2
        envelope = np.exp(-dist_sq / (2 * radius**2))

        # High-frequency noise - minimal smoothing to disrupt coherence
        noise = np.random.randn(self.N, self.N, self.N).astype(np.float32)
        noise = gaussian_filter(noise, sigma=0.5)  # Less smoothing = more disruptive

        # Scramble velocity field with uncorrelated noise
        velocity_noise = np.random.randn(self.N, self.N, self.N).astype(np.float32)
        velocity_noise = gaussian_filter(velocity_noise, sigma=0.5)

        # Apply to field and velocity (momentum scrambling)
        self.L += strength * envelope * noise
        self.L_prev += strength * 0.7 * envelope * velocity_noise  # Stronger momentum disruption

    def _update_coherence_density(self):
        """
        Update the coherence density field from all sources.

        Sources oscillate between positive and negative (bipolar)
        to generate propagating spherical waves.
        """
        self.rho_coh = np.zeros((self.N, self.N, self.N), dtype=np.float32)

        for src in self.sources:
            dist_sq = ((self.X - src['x'])**2 +
                       (self.Y - src['y'])**2 +
                       (self.Z - src['z'])**2)
            spatial = np.exp(-dist_sq / (2 * src['radius']**2))

            if src['type'] == 'oscillating':
                temporal = np.sin(2 * np.pi * src['frequency'] * self.time + src['phase'])
                amplitude = src['strength'] * temporal
            elif src['type'] == 'pulse':
                dt_birth = self.time - src['birth_time']
                if dt_birth > 0:
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

        Implements the 3D wave equation with absorbing boundaries.
        """
        c2dt2 = (self.v_L * self.dt) ** 2
        c_ratio = self.v_L * self.dt / self.dx

        for _ in range(n_steps):
            self._update_coherence_density()

            # Compute 3D Laplacian
            lap_L = self._laplacian(self.L)

            # Wave equation update
            L_new = (
                2 * self.L - self.L_prev +
                c2dt2 * lap_L -
                self.damping * (self.L - self.L_prev)
            )

            # Source injection (drives field velocity)
            L_new += self.dt * self.kappa_L * self.rho_coh

            # Absorbing boundary conditions (Mur's first-order ABC)
            # All 6 faces of the cube
            abc = (c_ratio - 1) / (c_ratio + 1)

            # X boundaries
            L_new[0, :, :] = self.L[1, :, :] + abc * (L_new[1, :, :] - self.L[0, :, :])
            L_new[-1, :, :] = self.L[-2, :, :] + abc * (L_new[-2, :, :] - self.L[-1, :, :])

            # Y boundaries
            L_new[:, 0, :] = self.L[:, 1, :] + abc * (L_new[:, 1, :] - self.L[:, 0, :])
            L_new[:, -1, :] = self.L[:, -2, :] + abc * (L_new[:, -2, :] - self.L[:, -1, :])

            # Z boundaries
            L_new[:, :, 0] = self.L[:, :, 1] + abc * (L_new[:, :, 1] - self.L[:, :, 0])
            L_new[:, :, -1] = self.L[:, :, -2] + abc * (L_new[:, :, -2] - self.L[:, :, -1])

            # Update arrays
            self.L_prev = self.L.copy()
            self.L = L_new

            self.time += self.dt
            self.step_count += 1

    def get_total_coherence(self) -> float:
        """
        Calculate Q: spatial coherence metric (3D version).

        Measures how organized/coordinated the field is across the volume.
        """
        total_energy = np.sum(self.L**2)
        if total_energy < 1e-10:
            return 0.0

        # Roughness via Laplacian
        laplacian = self._laplacian(self.L)
        laplacian_energy = np.sum(laplacian**2)
        roughness = laplacian_energy / (total_energy + 1e-10)

        # Spatial correlation at distance
        shift = max(1, self.N // 10)
        L_shifted_x = np.roll(self.L, shift, axis=0)
        L_shifted_y = np.roll(self.L, shift, axis=1)
        L_shifted_z = np.roll(self.L, shift, axis=2)

        corr_x = np.sum(self.L * L_shifted_x) / (total_energy + 1e-10)
        corr_y = np.sum(self.L * L_shifted_y) / (total_energy + 1e-10)
        corr_z = np.sum(self.L * L_shifted_z) / (total_energy + 1e-10)
        correlation = (corr_x + corr_y + corr_z) / 3.0

        Q = (1.0 + correlation) / (1.0 + 0.001 * roughness)
        return max(0.0, Q)

    def get_consciousness_metric(self) -> float:
        """
        Calculate C_bio: consciousness observable (3D version).

        C_bio = Q³ × ∫|ρ_coh|·|∂L/∂t| dV

        Uses Q³ (stronger than 2D's Q²) because 3D chaotic dynamics
        accumulate more activity volume. The cubic penalty ensures
        consciousness requires strong coherence, not just activity.
        """
        Q = self.get_total_coherence()
        if Q < 0.01:
            return 0.0

        dL_dt = (self.L - self.L_prev) / self.dt
        temporal_activity = np.abs(dL_dt)
        local_coupling = np.abs(self.rho_coh) * temporal_activity
        raw_activity = np.sum(local_coupling) * (self.dx ** 3)

        return (Q ** 3) * raw_activity

    def get_field_energy(self) -> float:
        """Get total field energy."""
        return np.sum(self.L**2) * (self.dx ** 3)

    def reset(self):
        """Reset simulation to initial state."""
        self.L = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        self.L_prev = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        self.rho_coh = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        self.sources = []
        self.time = 0.0
        self.step_count = 0

    def get_slice(self, axis: str = 'z', position: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a 2D slice through the 3D field.

        Args:
            axis: 'x', 'y', or 'z' - normal to the slice plane
            position: Position along the axis (-domain/2 to +domain/2)

        Returns:
            (slice_data, coord1, coord2) for plotting
        """
        idx = int((position + self.L_domain/2) / self.dx)
        idx = np.clip(idx, 0, self.N - 1)

        coords = np.linspace(-self.L_domain/2, self.L_domain/2, self.N)

        if axis == 'x':
            return self.L[idx, :, :], coords, coords
        elif axis == 'y':
            return self.L[:, idx, :], coords, coords
        else:  # z
            return self.L[:, :, idx], coords, coords

    def get_isosurface_data(self, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for isosurface rendering at given threshold.

        Returns positive and negative surfaces separately for
        proper coloring (gold vs blue in CLT visualization).
        """
        return self.L >= threshold, self.L <= -threshold


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def create_healthy_preset(sim: LoomfieldSimulator3D):
    """
    Healthy: Phase-locked sources creating coherent interference.

    Central source + shell of synchronized satellites.
    Should produce high Q, high C_bio, calm visual appearance.
    """
    sim.reset()

    base_freq = 1.2

    # Central dominant source
    sim.add_source(0, 0, 0, strength=1.5, radius=0.6,
                   frequency=base_freq, phase=0.0)

    # Octahedral shell of phase-locked sources
    r = 2.5
    for pos in [(r,0,0), (-r,0,0), (0,r,0), (0,-r,0), (0,0,r), (0,0,-r)]:
        sim.add_source(*pos, strength=0.8, radius=0.5,
                       frequency=base_freq, phase=0.0)


def create_pathology_preset(sim: LoomfieldSimulator3D):
    """
    Pathology: Incoherent sources with different frequencies.

    Scattered sources, no phase-locking, chaotic interference.
    Should produce low Q, low C_bio, noisy visual appearance.
    """
    sim.reset()

    # Scattered sources with different frequencies
    positions = [
        (-2, 2, 1), (2.5, -1, 1.5), (-1, -2, -2),
        (1.5, 1.5, -1.5), (-2.5, 0, 2), (0, 2.5, -2)
    ]
    frequencies = [0.7, 1.9, 2.3, 1.1, 2.8, 0.9]
    phases = [0.0, 1.2, 2.5, 0.8, 3.1, 1.9]

    for pos, freq, phase in zip(positions, frequencies, phases):
        sim.add_source(*pos, strength=0.7, radius=0.5,
                       frequency=freq, phase=phase)


def create_healing_preset(sim: LoomfieldSimulator3D):
    """
    Healing: Re-coupling toward coherence.

    Sources with same frequency but phase gradients,
    creating organized traveling waves that converge.
    """
    sim.reset()

    base_freq = 1.0

    # Central attractor
    sim.add_source(0, 0, 0, strength=1.8, radius=0.7,
                   frequency=base_freq, phase=0.0)

    # Ring in xy-plane with phase gradient
    r = 3.0
    n_ring = 8
    for i in range(n_ring):
        angle = 2 * np.pi * i / n_ring
        x, y = r * np.cos(angle), r * np.sin(angle)
        phase = angle * 0.5  # Phase gradient creates spiral convergence
        sim.add_source(x, y, 0, strength=0.6, radius=0.4,
                       frequency=base_freq, phase=phase)


# =============================================================================
# VISUALIZATION (using plotly for web-compatible 3D)
# =============================================================================

def create_volumetric_figure(sim: LoomfieldSimulator3D,
                              render_mode: str = 'volume',
                              opacity_scale: float = 0.5):
    """
    Create a plotly figure with volumetric rendering of the Loomfield.

    Args:
        sim: The 3D Loomfield simulator
        render_mode: 'volume' (continuous 3D volume) or 'isosurface' (nested shells)
        opacity_scale: Overall opacity multiplier (0.0-1.0)

    The field is visualized with a gold colorscale where:
    - Brighter = higher field amplitude (more coherent)
    - Darker = lower field amplitude
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required for 3D visualization. Install with: pip install plotly")

    # Get field data
    L = sim.L

    # Create flattened meshgrid coordinates (required by plotly)
    coords_1d = np.linspace(-sim.L_domain/2, sim.L_domain/2, sim.N)
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')

    # Create figure
    fig = go.Figure()

    L_flat = L.flatten()
    L_min, L_max = L_flat.min(), L_flat.max()
    L_absmax = max(abs(L_min), abs(L_max))

    if render_mode == 'volume' and L_absmax > 0.01:
        # Normalize to [-1, 1] for consistent visualization
        L_norm = L_flat / L_absmax

        # Volume trace with bipolar colorscale: blue (negative) -> dark -> gold (positive)
        # Opacity ensures visibility at ALL field values (no disappearing regions)
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=L_norm,
            isomin=-1.0,
            isomax=1.0,
            opacity=opacity_scale,
            opacityscale=[
                [0.0, 0.9],   # Strong negative = very visible
                [0.25, 0.5],  # Moderate negative = visible
                [0.5, 0.25],  # Zero = still visible (base glow)
                [0.75, 0.5],  # Moderate positive = visible
                [1.0, 0.9],   # Strong positive = very visible
            ],
            surface_count=20,  # More surfaces = smoother volume
            # Bipolar colorscale: blue -> dark -> gold
            colorscale=[
                [0.0, 'rgb(80,140,255)'],   # Strong negative = bright blue
                [0.25, 'rgb(40,70,150)'],   # Weak negative = dim blue
                [0.5, 'rgb(30,30,40)'],     # Zero = dark (but visible via opacity)
                [0.75, 'rgb(180,130,50)'],  # Weak positive = dim gold
                [1.0, 'rgb(255,220,100)'],  # Strong positive = bright gold
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text='L (normalized)', font=dict(color='#a0a0a0')),
                tickfont=dict(color='#a0a0a0'),
                x=0.02, len=0.5
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    elif render_mode == 'isosurface' and L_absmax > 0.01:
        # Isosurface mode - nested shells for both positive and negative regions
        # Positive thresholds (gold)
        pos_thresholds = [0.1 * L_absmax, 0.3 * L_absmax, 0.5 * L_absmax, 0.7 * L_absmax]
        pos_colors = ['rgb(120,90,30)', 'rgb(180,140,50)', 'rgb(220,180,70)', 'rgb(255,220,100)']
        pos_opacities = [0.2, 0.35, 0.5, 0.7]

        for threshold, color, base_opacity in zip(pos_thresholds, pos_colors, pos_opacities):
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=L_flat,
                isomin=threshold,
                isomax=L_absmax,
                opacity=opacity_scale * base_opacity,
                surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=f'Positive {threshold:.2f}',
            ))

        # Negative thresholds (blue)
        neg_thresholds = [-0.1 * L_absmax, -0.3 * L_absmax, -0.5 * L_absmax, -0.7 * L_absmax]
        neg_colors = ['rgb(50,70,120)', 'rgb(60,90,160)', 'rgb(70,110,190)', 'rgb(100,150,230)']

        for threshold, color, base_opacity in zip(neg_thresholds, neg_colors, pos_opacities):
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=L_flat,
                isomin=-L_absmax,
                isomax=threshold,
                opacity=opacity_scale * base_opacity,
                surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=f'Negative {threshold:.2f}',
            ))

    # Add source markers (minimal particles - only for sources)
    for src in sim.sources:
        fig.add_trace(go.Scatter3d(
            x=[src['x']], y=[src['y']], z=[src['z']],
            mode='markers',
            marker=dict(size=8, color='white', opacity=0.8,
                       line=dict(width=2, color='gold')),
            name='Source',
            showlegend=False
        ))

    # Metrics annotation
    Q = sim.get_total_coherence()
    C_bio = sim.get_consciousness_metric()
    energy = sim.get_field_energy()

    q_status = "HIGH" if Q > 1.5 else ("moderate" if Q > 0.5 else "LOW")
    c_status = "conscious" if C_bio > 5 else ("partial" if C_bio > 1 else "minimal")

    # Layout for calm, scientific appearance
    fig.update_layout(
        title=dict(
            text=f"<b>Loomfield 3D</b><br>"
                 f"<span style='font-size:12px'>Q={Q:.2f} ({q_status}) | "
                 f"C_bio={C_bio:.2f} ({c_status}) | t={sim.time:.1f}</span>",
            x=0.5,
            font=dict(size=16, color='#b0b0b0')
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='rgb(10,10,20)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='cube'
        ),
        paper_bgcolor='rgb(10,10,20)',
        plot_bgcolor='rgb(10,10,20)',
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=False
    )

    return fig


def create_slice_figure(sim: LoomfieldSimulator3D,
                        axis: str = 'z',
                        position: float = 0.0):
    """
    Create a 2D slice through the 3D Loomfield.

    Useful for examining interior structure without full volumetric rendering.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required. Install with: pip install plotly")

    slice_data, coords1, coords2 = sim.get_slice(axis, position)

    # CLT colorscale: blue (negative) -> dark (zero) -> gold (positive)
    colorscale = [
        [0.0, 'rgb(30,50,150)'],
        [0.3, 'rgb(40,60,120)'],
        [0.5, 'rgb(20,20,30)'],
        [0.7, 'rgb(120,90,40)'],
        [1.0, 'rgb(255,200,80)']
    ]

    vmax = max(np.abs(slice_data).max(), 0.1)

    fig = go.Figure(data=go.Heatmap(
        z=slice_data,
        x=coords1,
        y=coords2,
        colorscale=colorscale,
        zmin=-vmax,
        zmax=vmax,
        showscale=True,
        colorbar=dict(
            title=dict(text='L', font=dict(color='#a0a0a0')),
            tickfont=dict(color='#a0a0a0')
        )
    ))

    fig.update_layout(
        title=f"Loomfield Slice ({axis}={position:.1f})",
        xaxis=dict(title=f"{'Y' if axis=='x' else 'X'}",
                   scaleanchor='y', scaleratio=1,
                   color='#a0a0a0'),
        yaxis=dict(title=f"{'Z' if axis!='z' else 'Y'}",
                   color='#a0a0a0'),
        paper_bgcolor='rgb(15,15,25)',
        plot_bgcolor='rgb(15,15,25)',
        font=dict(color='#a0a0a0')
    )

    return fig


def create_animated_figure(sim: LoomfieldSimulator3D,
                           n_frames: int = 60,
                           steps_per_frame: int = 5,
                           opacity_scale: float = 0.6):
    """
    Create an animated 3D figure showing wave propagation over time.

    Args:
        sim: Simulator with sources already configured
        n_frames: Number of animation frames
        steps_per_frame: Simulation steps between frames
        opacity_scale: Volume opacity

    Returns:
        Plotly figure with animation frames and play button
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required. Install with: pip install plotly")

    # Create coordinate arrays
    coords_1d = np.linspace(-sim.L_domain/2, sim.L_domain/2, sim.N)
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()

    # Collect frames by running simulation
    print(f"  Generating {n_frames} animation frames...")
    frames_data = []
    metrics_data = []

    for i in range(n_frames):
        sim.step(steps_per_frame)

        # Store frame data
        frames_data.append(sim.L.copy())
        metrics_data.append({
            'Q': sim.get_total_coherence(),
            'C_bio': sim.get_consciousness_metric(),
            'time': sim.time
        })

        if (i + 1) % 20 == 0:
            print(f"    Frame {i+1}/{n_frames}")

    # Use per-frame normalization so field is visible throughout
    # This ensures the "breathing" shows as intensity variation, not on/off toggling

    def normalize_frame(L_frame):
        """Normalize frame values to [-1, 1] range for consistent visibility."""
        L_flat = L_frame.flatten()
        L_absmax = max(abs(L_flat.min()), abs(L_flat.max()), 0.01)
        return L_flat / L_absmax, L_absmax

    # Opacity scale that ensures visibility at ALL values
    # Zero values still have base visibility, peaks are brighter
    visibility_opacityscale = [
        [0.0, 0.9],   # Strong negative = very visible
        [0.25, 0.5],  # Moderate negative = visible
        [0.5, 0.25],  # Zero = still visible (base glow)
        [0.75, 0.5],  # Moderate positive = visible
        [1.0, 0.9],   # Strong positive = very visible
    ]

    # Bipolar colorscale: blue -> dark -> gold
    bipolar_colorscale = [
        [0.0, 'rgb(80,140,255)'],   # Strong negative = bright blue
        [0.25, 'rgb(40,70,150)'],   # Weak negative = dim blue
        [0.5, 'rgb(30,30,40)'],     # Zero = dark (but still visible via opacity)
        [0.75, 'rgb(180,130,50)'],  # Weak positive = dim gold
        [1.0, 'rgb(255,220,100)'],  # Strong positive = bright gold
    ]

    # Create initial figure with first frame (normalized)
    L_norm_0, _ = normalize_frame(frames_data[0])

    fig = go.Figure(
        data=[go.Volume(
            x=X_flat,
            y=Y_flat,
            z=Z_flat,
            value=L_norm_0,
            isomin=-1.0,
            isomax=1.0,
            opacity=opacity_scale,
            opacityscale=visibility_opacityscale,
            surface_count=17,
            colorscale=bipolar_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text='L', font=dict(color='#a0a0a0')),
                tickfont=dict(color='#a0a0a0'),
                x=0.02, len=0.4
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
        )]
    )

    # Add source markers
    for src in sim.sources:
        fig.add_trace(go.Scatter3d(
            x=[src['x']], y=[src['y']], z=[src['z']],
            mode='markers',
            marker=dict(size=6, color='white', opacity=0.8,
                       line=dict(width=1, color='gold')),
            showlegend=False
        ))

    # Create animation frames with per-frame normalization
    frames = []
    for i, (L_frame, metrics) in enumerate(zip(frames_data, metrics_data)):
        # Normalize this frame's values to [-1, 1]
        L_norm, frame_absmax = normalize_frame(L_frame)

        frame = go.Frame(
            data=[go.Volume(
                x=X_flat,
                y=Y_flat,
                z=Z_flat,
                value=L_norm,
                isomin=-1.0,
                isomax=1.0,
                opacity=opacity_scale,
                opacityscale=visibility_opacityscale,
                surface_count=17,
                colorscale=bipolar_colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(text='L', font=dict(color='#a0a0a0')),
                    tickfont=dict(color='#a0a0a0'),
                    x=0.02, len=0.4
                ),
                caps=dict(x_show=False, y_show=False, z_show=False),
            )],
            name=str(i),
            layout=go.Layout(
                title=dict(
                    text=f"<b>Loomfield 3D</b><br>"
                         f"<span style='font-size:12px'>Q={metrics['Q']:.2f} | "
                         f"C_bio={metrics['C_bio']:.1f} | t={metrics['time']:.1f} | "
                         f"max|L|={frame_absmax:.2f}</span>",
                )
            )
        )
        frames.append(frame)

    fig.frames = frames

    # Initial metrics for title
    m0 = metrics_data[0]
    _, L0_absmax = normalize_frame(frames_data[0])

    # Layout with animation controls
    fig.update_layout(
        title=dict(
            text=f"<b>Loomfield 3D - Animated</b><br>"
                 f"<span style='font-size:12px'>Q={m0['Q']:.2f} | "
                 f"C_bio={m0['C_bio']:.1f} | t={m0['time']:.1f} | "
                 f"max|L|={L0_absmax:.2f}</span>",
            x=0.5,
            font=dict(size=16, color='#b0b0b0')
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='rgb(10,10,20)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube'
        ),
        paper_bgcolor='rgb(10,10,20)',
        plot_bgcolor='rgb(10,10,20)',
        margin=dict(l=0, r=0, t=60, b=80),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 80, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ],
                font=dict(color='#b0b0b0'),
                bgcolor='rgb(40,40,50)',
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right",
                "font": {"color": "#a0a0a0"}
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.8,
            "x": 0.1,
            "y": 0,
            "steps": [
                {"args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}],
                 "label": str(i),
                 "method": "animate"}
                for i in range(n_frames)
            ],
            "font": {"color": "#a0a0a0"},
            "bgcolor": "rgb(40,40,50)",
            "bordercolor": "rgb(60,60,70)",
        }]
    )

    return fig


# =============================================================================
# INTERACTIVE APPLICATION
# =============================================================================

class LoomfieldVisualizer3D:
    """
    Interactive 3D visualization of the Loomfield.

    Provides three layers of engagement:
    - Layer 1 (Passive): Auto-rotating, caption-friendly
    - Layer 2 (Presenter): Preset paths, one-click scenarios
    - Layer 3 (Explorer): Full controls, metrics, debug views
    """

    def __init__(self, grid_size: int = 48):
        """
        Initialize the 3D visualizer.

        Args:
            grid_size: Resolution (48 for smooth performance, 64 for detail)
        """
        self.sim = LoomfieldSimulator3D(grid_size=grid_size)
        self.grid_size = grid_size
        self.running = False
        self.current_preset = None

    def load_preset(self, preset: str = 'healthy'):
        """Load a preset configuration."""
        if preset == 'healthy':
            create_healthy_preset(self.sim)
            self.current_preset = 'healthy'
        elif preset == 'pathology':
            create_pathology_preset(self.sim)
            self.current_preset = 'pathology'
        elif preset == 'healing':
            create_healing_preset(self.sim)
            self.current_preset = 'healing'
        else:
            self.sim.reset()
            self.current_preset = 'empty'

    def run_simulation(self, n_steps: int = 100, steps_per_frame: int = 5):
        """
        Run simulation and return frames for animation.

        Returns list of field snapshots for animation.
        """
        frames = []
        for i in range(n_steps):
            self.sim.step(steps_per_frame)
            if i % 2 == 0:  # Sample every 2nd step for performance
                frames.append({
                    'L': self.sim.L.copy(),
                    'Q': self.sim.get_total_coherence(),
                    'C_bio': self.sim.get_consciousness_metric(),
                    'time': self.sim.time
                })
        return frames

    def create_static_figure(self, warm_up_steps: int = 200):
        """
        Create a static figure after warming up the simulation.

        Good for screenshots and embedding.
        """
        self.sim.step(warm_up_steps)
        return create_volumetric_figure(self.sim)

    def create_slice_view(self, axis: str = 'z', position: float = 0.0):
        """Create a 2D slice view."""
        return create_slice_figure(self.sim, axis, position)

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return {
            'Q': self.sim.get_total_coherence(),
            'C_bio': self.sim.get_consciousness_metric(),
            'energy': self.sim.get_field_energy(),
            'time': self.sim.time,
            'sources': len(self.sim.sources)
        }


# =============================================================================
# DEMO / ENTRY POINT
# =============================================================================

def demo():
    """
    Run a demonstration of the 3D Loomfield visualizer.

    Creates figures for each preset and displays them.
    """
    print("=" * 70)
    print("  COSMIC LOOM THEORY: 3D Loomfield Visualizer")
    print("=" * 70)
    print("""
The Loomfield L(r,t) extends into 3D space, creating spherical waves
that propagate outward from coherence sources. This visualization shows
the field as a calm, living volume - not particles, but continuous flow.

Creating visualizations for each preset...
    """)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("ERROR: plotly required for 3D visualization.")
        print("Install with: pip install plotly")
        return

    # Create visualizer
    viz = LoomfieldVisualizer3D(grid_size=48)

    # Generate figures for each preset
    presets = ['healthy', 'pathology', 'healing']

    for preset in presets:
        print(f"\nGenerating {preset.upper()} preset...")
        viz.load_preset(preset)

        # Warm up simulation
        viz.sim.step(150)

        # Get metrics
        metrics = viz.get_metrics()
        print(f"  Q = {metrics['Q']:.3f}")
        print(f"  C_bio = {metrics['C_bio']:.3f}")
        print(f"  Energy = {metrics['energy']:.3f}")

        # Create and save figure (volume mode for visibility)
        fig = create_volumetric_figure(viz.sim, render_mode='volume', opacity_scale=0.6)
        fig.write_html(f"output/loomfield_3d_{preset}.html")
        print(f"  Saved: output/loomfield_3d_{preset}.html")

    # Also create a slice view
    print("\nGenerating slice view...")
    viz.load_preset('healthy')
    viz.sim.step(150)
    fig_slice = create_slice_figure(viz.sim, 'z', 0.0)
    fig_slice.write_html("output/loomfield_3d_slice.html")
    print("  Saved: output/loomfield_3d_slice.html")

    # Create animated visualization (shows wave propagation)
    print("\nGenerating ANIMATED healthy visualization...")
    sim_anim = LoomfieldSimulator3D(grid_size=32)  # Smaller grid for animation performance
    create_healthy_preset(sim_anim)
    fig_anim = create_animated_figure(sim_anim, n_frames=60, steps_per_frame=5, opacity_scale=0.5)
    fig_anim.write_html("output/loomfield_3d_animated.html")
    print("  Saved: output/loomfield_3d_animated.html")

    print("\n" + "=" * 70)
    print("Open the HTML files in a browser to explore the 3D visualizations.")
    print("Files generated:")
    print("  - loomfield_3d_healthy.html    (static volume)")
    print("  - loomfield_3d_pathology.html  (static volume)")
    print("  - loomfield_3d_healing.html    (static volume)")
    print("  - loomfield_3d_slice.html      (2D slice view)")
    print("  - loomfield_3d_animated.html   (animated with play button)")
    print("=" * 70)


if __name__ == "__main__":
    demo()
