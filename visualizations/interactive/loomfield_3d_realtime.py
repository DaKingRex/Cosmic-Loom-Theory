"""
Real-Time 3D Loomfield Simulator

An interactive 3D visualization of Loomfield dynamics with real-time physics
computation and rendering. This is the 3D extension of loomfield_wave.py.

Wave Equation (3D):
    ∇²L - (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh

Features:
    - Real-time 3D wave propagation
    - Volumetric rendering with gold/blue colormap
    - Interactive mouse rotation
    - Sliders for v_L, κ_L, animation speed
    - Preset buttons: Healthy, Pathology, Healing
    - Click modes: Add Source, Add Perturbation
    - Slice plane to view interior structure

Requirements:
    pip install vispy PyQt5 numpy scipy

Usage:
    python run_loomfield_3d_realtime.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import vispy and Qt
try:
    from vispy import app, scene, color
    from vispy.scene import visuals
    from vispy.visuals.transforms import STTransform, MatrixTransform
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False
    print("vispy not found. Install with: pip install vispy PyQt5")

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("PyQt5 not found. Install with: pip install PyQt5")


class LoomfieldSimulator3DRealtime:
    """
    Optimized 3D Loomfield simulator for real-time rendering.

    Uses smaller grid and efficient numpy operations for smooth performance.
    """

    def __init__(self,
                 grid_size: int = 32,
                 domain_size: float = 10.0,
                 v_L: float = 1.0,
                 kappa_L: float = 2.0,
                 damping: float = 0.002):
        """
        Initialize the real-time 3D simulator.

        Args:
            grid_size: Grid points per dimension (32-48 for real-time)
            domain_size: Physical size of cubic domain
            v_L: Loomfield propagation speed
            kappa_L: Coupling constant
            damping: Dissipation rate
        """
        self.N = grid_size
        self.L_domain = domain_size
        self.dx = domain_size / grid_size
        self.v_L = v_L
        self.kappa_L = kappa_L
        self.damping = damping

        # CFL condition for 3D stability
        self.dt = 0.4 * self.dx / (self.v_L * np.sqrt(3))

        # Field arrays (float32 for GPU-friendly rendering)
        self.L = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        self.L_prev = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        self.rho_coh = np.zeros((self.N, self.N, self.N), dtype=np.float32)

        # Sources
        self.sources: List[dict] = []

        # Coordinate grids
        coords = np.linspace(-domain_size/2, domain_size/2, self.N)
        self.X, self.Y, self.Z = np.meshgrid(coords, coords, coords, indexing='ij')

        # Time tracking
        self.time = 0.0
        self.step_count = 0

        # Precompute for efficiency
        self._dx2_inv = 1.0 / (self.dx ** 2)

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """7-point stencil 3D Laplacian."""
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1, 1:-1] = (
            field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
            field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
            field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -
            6 * field[1:-1, 1:-1, 1:-1]
        ) * self._dx2_inv
        return lap

    def add_source(self, x: float, y: float, z: float,
                   strength: float = 1.0,
                   radius: float = 0.5,
                   frequency: float = 1.5,
                   phase: float = 0.0,
                   source_type: str = 'oscillating'):
        """Add a coherence source."""
        self.sources.append({
            'x': x, 'y': y, 'z': z,
            'strength': strength,
            'radius': radius,
            'frequency': frequency,
            'phase': phase,
            'type': source_type,
            'birth_time': self.time
        })
        print(f"[DEBUG] Added source at ({x:.2f}, {y:.2f}, {z:.2f}), strength={strength}, freq={frequency}")

    def clear_sources(self):
        """Remove all sources."""
        self.sources = []
        self.rho_coh.fill(0)

    def add_perturbation(self, x: float, y: float, z: float,
                         strength: float = 2.0, radius: float = 1.5):
        """Add a perturbation that disrupts coherence."""
        print(f"[DEBUG] Adding perturbation at ({x:.2f}, {y:.2f}, {z:.2f}), strength={strength}")
        dist_sq = (self.X - x)**2 + (self.Y - y)**2 + (self.Z - z)**2
        envelope = np.exp(-dist_sq / (2 * radius**2))

        noise = np.random.randn(self.N, self.N, self.N).astype(np.float32)
        noise = gaussian_filter(noise, sigma=0.5)

        velocity_noise = np.random.randn(self.N, self.N, self.N).astype(np.float32)
        velocity_noise = gaussian_filter(velocity_noise, sigma=0.5)

        self.L += strength * envelope * noise
        self.L_prev += strength * 0.7 * envelope * velocity_noise
        print(f"[DEBUG] After perturbation: L range [{self.L.min():.4f}, {self.L.max():.4f}]")

    def _update_coherence_density(self):
        """Update coherence density from all sources."""
        self.rho_coh.fill(0)

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
            else:
                amplitude = src['strength']

            self.rho_coh += amplitude * spatial

    def step(self, n_steps: int = 1):
        """Advance simulation by n_steps."""
        c2dt2 = (self.v_L * self.dt) ** 2
        c_ratio = self.v_L * self.dt / self.dx

        for _ in range(n_steps):
            self._update_coherence_density()

            lap_L = self._laplacian(self.L)
            L_new = (
                2 * self.L - self.L_prev +
                c2dt2 * lap_L +
                c2dt2 * self.kappa_L * self.rho_coh -
                self.damping * (self.L - self.L_prev)
            )

            # Absorbing boundaries (Mur's method on all 6 faces)
            abc = (c_ratio - 1) / (c_ratio + 1)

            L_new[0, :, :] = self.L[1, :, :] + abc * (L_new[1, :, :] - self.L[0, :, :])
            L_new[-1, :, :] = self.L[-2, :, :] + abc * (L_new[-2, :, :] - self.L[-1, :, :])
            L_new[:, 0, :] = self.L[:, 1, :] + abc * (L_new[:, 1, :] - self.L[:, 0, :])
            L_new[:, -1, :] = self.L[:, -2, :] + abc * (L_new[:, -2, :] - self.L[:, -1, :])
            L_new[:, :, 0] = self.L[:, :, 1] + abc * (L_new[:, :, 1] - self.L[:, :, 0])
            L_new[:, :, -1] = self.L[:, :, -2] + abc * (L_new[:, :, -2] - self.L[:, :, -1])

            self.L_prev = self.L.copy()
            self.L = L_new.astype(np.float32)
            self.time += self.dt
            self.step_count += 1

    def get_total_coherence(self) -> float:
        """Calculate Q metric."""
        total_energy = np.sum(self.L**2)
        if total_energy < 1e-10:
            return 0.0

        grad_x = np.diff(self.L, axis=0, append=self.L[-1:, :, :])
        grad_y = np.diff(self.L, axis=1, append=self.L[:, -1:, :])
        grad_z = np.diff(self.L, axis=2, append=self.L[:, :, -1:])
        roughness = np.sum(grad_x**2 + grad_y**2 + grad_z**2)

        L_shifted_x = np.roll(self.L, 2, axis=0)
        L_shifted_y = np.roll(self.L, 2, axis=1)
        L_shifted_z = np.roll(self.L, 2, axis=2)

        corr_x = np.sum(self.L * L_shifted_x) / (total_energy + 1e-10)
        corr_y = np.sum(self.L * L_shifted_y) / (total_energy + 1e-10)
        corr_z = np.sum(self.L * L_shifted_z) / (total_energy + 1e-10)
        correlation = (corr_x + corr_y + corr_z) / 3.0

        Q = (1.0 + correlation) / (1.0 + 0.001 * roughness)
        return max(0.0, Q)

    def get_consciousness_metric(self) -> float:
        """Calculate C_bio = Q³ × activity."""
        Q = self.get_total_coherence()
        if Q < 0.01:
            return 0.0

        dL_dt = (self.L - self.L_prev) / self.dt
        temporal_activity = np.abs(dL_dt)
        local_coupling = np.abs(self.rho_coh) * temporal_activity
        raw_activity = np.sum(local_coupling) * (self.dx ** 3)

        return (Q ** 3) * raw_activity

    def get_field_energy(self) -> float:
        """Total field energy."""
        return np.sum(self.L**2) * (self.dx ** 3)

    def reset(self):
        """Reset simulation."""
        self.L.fill(0)
        self.L_prev.fill(0)
        self.rho_coh.fill(0)
        self.sources = []
        self.time = 0.0
        self.step_count = 0

    def get_slice(self, axis: str = 'z', index: int = None) -> np.ndarray:
        """Get 2D slice through the field."""
        if index is None:
            index = self.N // 2
        index = max(0, min(index, self.N - 1))

        if axis == 'x':
            return self.L[index, :, :]
        elif axis == 'y':
            return self.L[:, index, :]
        else:
            return self.L[:, :, index]


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def create_healthy_preset(sim: LoomfieldSimulator3DRealtime):
    """Healthy: Phase-locked sources."""
    print("[DEBUG] Loading HEALTHY preset...")
    sim.reset()
    base_freq = 1.2

    sim.add_source(0, 0, 0, strength=1.5, radius=0.6, frequency=base_freq, phase=0.0)

    r = 2.5
    for pos in [(r,0,0), (-r,0,0), (0,r,0), (0,-r,0), (0,0,r), (0,0,-r)]:
        sim.add_source(*pos, strength=0.8, radius=0.5, frequency=base_freq, phase=0.0)
    print(f"[DEBUG] Healthy preset loaded: {len(sim.sources)} sources")


def create_pathology_preset(sim: LoomfieldSimulator3DRealtime):
    """Pathology: Incoherent sources with different frequencies."""
    sim.reset()

    positions = [(-2, 2, 1), (2.5, -1, 1.5), (-1, -2, -2),
                 (1.5, 1.5, -1.5), (-2.5, 0, 2), (0, 2.5, -2)]
    frequencies = [0.7, 1.9, 2.3, 1.1, 2.8, 0.9]
    phases = [0.0, 1.2, 2.5, 0.8, 3.1, 1.9]

    for pos, freq, phase in zip(positions, frequencies, phases):
        sim.add_source(*pos, strength=0.7, radius=0.5, frequency=freq, phase=phase)


def create_healing_preset(sim: LoomfieldSimulator3DRealtime):
    """Healing: Re-coupling toward coherence."""
    sim.reset()
    base_freq = 1.0

    sim.add_source(0, 0, 0, strength=1.8, radius=0.7, frequency=base_freq, phase=0.0)

    r = 3.0
    n_ring = 8
    for i in range(n_ring):
        angle = 2 * np.pi * i / n_ring
        x, y = r * np.cos(angle), r * np.sin(angle)
        phase = angle * 0.5
        sim.add_source(x, y, 0, strength=0.6, radius=0.4, frequency=base_freq, phase=phase)


# =============================================================================
# VISPY 3D VISUALIZER
# =============================================================================

if HAS_VISPY and HAS_QT:

    class Loomfield3DVisualizer(QtWidgets.QMainWindow):
        """
        Real-time interactive 3D Loomfield visualizer.

        Uses vispy for fast 3D rendering and Qt for controls.
        """

        def __init__(self, grid_size: int = 32):
            super().__init__()

            self.sim = LoomfieldSimulator3DRealtime(grid_size=grid_size)
            self.grid_size = grid_size

            # Animation state
            self.running = True
            self.steps_per_frame = 3
            self.mode = 'add_source'  # 'add_source' or 'perturbation'
            self.slice_index = grid_size // 2
            self.show_slice = False

            # Render mode: 'slices' or 'isosurface'
            self.render_mode = 'slices'
            self.render_modes = ['slices', 'isosurface']

            # Visual references (set in _create_* methods)
            self.slice_xy = None
            self.slice_xz = None
            self.slice_yz = None
            self.isosurfaces = []

            # Create colormap: blue -> dark -> gold
            self.cmap = self._create_colormap()

            self._setup_ui()
            self._setup_timer()

            # Start with healthy preset and warm up
            create_healthy_preset(self.sim)
            self.sim.step(100)  # Warm up so field is immediately visible

        def _create_colormap(self):
            """Create bipolar colormap for Loomfield with high visibility."""
            # For MIP rendering, we need colors that show up well
            # Strong colors at extremes, visible color at center
            colors = [
                (0.2, 0.4, 1.0, 1.0),    # Strong blue (negative/low)
                (0.4, 0.6, 0.9, 0.6),    # Medium blue
                (0.3, 0.3, 0.4, 0.3),    # Dim gray (center) - still visible
                (0.9, 0.7, 0.3, 0.6),    # Medium gold
                (1.0, 0.85, 0.3, 1.0),   # Strong gold (positive/high)
            ]
            return color.Colormap(colors, controls=[0.0, 0.25, 0.5, 0.75, 1.0])

        def _setup_ui(self):
            """Set up the main window UI."""
            self.setWindowTitle('Loomfield 3D - Real-Time Simulator')
            self.setGeometry(100, 100, 1400, 900)
            self.setStyleSheet("background-color: #0a0a14;")

            # Central widget
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)

            # Main layout
            layout = QtWidgets.QHBoxLayout(central)
            layout.setContentsMargins(10, 10, 10, 10)

            # 3D canvas (left side)
            self.canvas = scene.SceneCanvas(
                keys='interactive',
                bgcolor='#0a0a14',
                show=True,
                size=(900, 800)
            )
            layout.addWidget(self.canvas.native, stretch=3)

            # Create 3D view
            # Camera distance should be ~1.5x the grid size to see the full volume
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = scene.cameras.TurntableCamera(
                fov=45,
                distance=self.grid_size * 1.5,  # Scale with grid size
                elevation=30,
                azimuth=45
            )
            print(f"[DEBUG] Camera distance set to {self.grid_size * 1.5} for grid_size={self.grid_size}")

            # Volume visual
            self._create_volume_visual()

            # Slice visual (initially hidden)
            self._create_slice_visual()

            # Control panel (right side)
            self._setup_controls(layout)

            # Connect canvas events
            self.canvas.events.mouse_press.connect(self._on_mouse_press)
            self.canvas.events.mouse_release.connect(self._on_mouse_release)
            self.canvas.events.key_press.connect(self._on_key_press)

            # Track mouse press position for click vs drag detection
            self._mouse_press_pos = None
            self._click_threshold = 5  # pixels - movement below this is a "click"

        def _create_volume_visual(self):
            """Create all 3D visualization modes.

            Supports two render modes:
            - 'slices': 3 orthogonal Image slices (most compatible)
            - 'isosurface': Nested transparent isosurface shells
            """
            print(f"[DEBUG] Creating 3D visualization modes")

            # Create visual types
            self._create_slice_visuals()
            self._create_isosurface_visuals()

            # Add a wireframe cube to show boundaries
            self._create_boundary_box()

            # Add source markers
            self._create_source_markers()

            # Set initial visibility based on render_mode
            self._set_render_mode(self.render_mode)

            print(f"[DEBUG] Render modes created. Current mode: {self.render_mode}")

        def _create_slice_visuals(self):
            """Create the 3 orthogonal slice plane visuals."""
            N = self.grid_size
            center = N // 2

            # Prepare slice data
            xy_data = self._prepare_slice_for_axis('z', center)
            xz_data = self._prepare_slice_for_axis('y', center)
            yz_data = self._prepare_slice_for_axis('x', center)

            # XY plane (horizontal slice at z=center)
            self.slice_xy = scene.visuals.Image(
                xy_data,
                parent=self.view.scene,
                cmap=self.cmap,
                clim=(0, 1),
            )
            self.slice_xy.transform = STTransform(
                translate=(-N/2, -N/2, 0),
                scale=(1, 1, 1)
            )

            # XZ plane (vertical slice at y=center)
            self.slice_xz = scene.visuals.Image(
                xz_data,
                parent=self.view.scene,
                cmap=self.cmap,
                clim=(0, 1),
            )
            xz_transform = MatrixTransform()
            xz_transform.rotate(90, (1, 0, 0))
            xz_transform.translate((-N/2, 0, -N/2))
            self.slice_xz.transform = xz_transform

            # YZ plane (vertical slice at x=center)
            self.slice_yz = scene.visuals.Image(
                yz_data,
                parent=self.view.scene,
                cmap=self.cmap,
                clim=(0, 1),
            )
            yz_transform = MatrixTransform()
            yz_transform.rotate(90, (0, 1, 0))
            yz_transform.rotate(90, (1, 0, 0))
            yz_transform.translate((0, -N/2, -N/2))
            self.slice_yz.transform = yz_transform

            print(f"[DEBUG] Created slice visuals at center={center}")

        def _create_isosurface_visuals(self):
            """Create nested isosurface shells as an alternative to Volume.

            Creates multiple transparent Isosurface visuals at different
            threshold levels. This often works when Volume rendering fails.
            """
            # Isosurfaces are created dynamically in _update_isosurfaces()
            self.isosurfaces = []
            print(f"[DEBUG] Isosurface mode initialized (surfaces created on update)")

        def _update_isosurfaces(self):
            """Update or recreate isosurface visuals based on current field.

            vispy's Isosurface generates mesh vertices in data index coordinates.
            For a (N, N, N) array, vertices span [0, N-1] in each dimension.
            We transform to match slice coordinates: [-N/2, N/2].
            """
            # Remove old isosurfaces
            for iso in self.isosurfaces:
                iso.parent = None
            self.isosurfaces = []

            try:
                L = self.sim.L
                N = self.grid_size
                L_absmax = max(np.abs(L).max(), 0.1)

                # Debug first time
                if not hasattr(self, '_iso_debug_done'):
                    print(f"[DEBUG] Isosurface: L.shape={L.shape}, N={N}, L_absmax={L_absmax:.3f}")
                    self._iso_debug_done = True

                # Threshold levels (as fractions of max)
                thresholds = [0.3, 0.5, 0.7]  # Slightly higher thresholds for visibility
                colors_pos = [
                    (1.0, 0.9, 0.5, 0.3),   # Light gold
                    (1.0, 0.8, 0.3, 0.5),   # Gold
                    (1.0, 0.7, 0.2, 0.7),   # Deep gold
                ]
                colors_neg = [
                    (0.5, 0.7, 1.0, 0.3),   # Light blue
                    (0.3, 0.5, 1.0, 0.5),   # Blue
                    (0.2, 0.4, 0.9, 0.7),   # Deep blue
                ]

                # Transform: data indices [0, N-1] -> visual coords [-N/2, N/2-1]
                # vispy Isosurface vertices are in data index space
                iso_transform = STTransform(
                    translate=(-N/2, -N/2, -N/2),
                    scale=(1, 1, 1)
                )

                for thresh_frac, color_p, color_n in zip(thresholds, colors_pos, colors_neg):
                    thresh_val = thresh_frac * L_absmax

                    # Positive isosurface (gold)
                    if L.max() > thresh_val:
                        try:
                            iso_pos = scene.visuals.Isosurface(
                                L, level=thresh_val,
                                color=color_p,
                                parent=self.view.scene,
                            )
                            iso_pos.transform = iso_transform
                            self.isosurfaces.append(iso_pos)
                        except Exception as e:
                            if not hasattr(self, '_iso_pos_error'):
                                print(f"[DEBUG] Isosurface+ error: {e}")
                                self._iso_pos_error = True

                    # Negative isosurface (blue)
                    if L.min() < -thresh_val:
                        try:
                            iso_neg = scene.visuals.Isosurface(
                                -L, level=thresh_val,
                                color=color_n,
                                parent=self.view.scene,
                            )
                            iso_neg.transform = iso_transform
                            self.isosurfaces.append(iso_neg)
                        except Exception as e:
                            if not hasattr(self, '_iso_neg_error'):
                                print(f"[DEBUG] Isosurface- error: {e}")
                                self._iso_neg_error = True

            except Exception as e:
                print(f"[DEBUG] Isosurface update error: {e}")

        def _set_render_mode(self, mode: str):
            """Set the active render mode and update visibility."""
            if mode not in self.render_modes:
                print(f"[DEBUG] Unknown render mode: {mode}")
                return

            self.render_mode = mode
            N = self.grid_size
            print(f"[DEBUG] Switching to render mode: {mode}")

            # Reset camera to default distance when switching modes
            # This ensures all modes are viewable at the same scale
            self.view.camera.distance = N * 1.5

            # Hide all visuals first
            if self.slice_xy:
                self.slice_xy.visible = False
            if self.slice_xz:
                self.slice_xz.visible = False
            if self.slice_yz:
                self.slice_yz.visible = False
            for iso in self.isosurfaces:
                iso.visible = False

            # Show visuals for current mode
            if mode == 'slices':
                if self.slice_xy:
                    self.slice_xy.visible = True
                if self.slice_xz:
                    self.slice_xz.visible = True
                if self.slice_yz:
                    self.slice_yz.visible = True

            elif mode == 'isosurface':
                # Isosurfaces are recreated each frame, just update them now
                self._update_isosurfaces()
                for iso in self.isosurfaces:
                    iso.visible = True

            # Update the mode label if it exists
            if hasattr(self, 'mode_label'):
                self.mode_label.setText(f"Render: {mode.upper()}")

        def _cycle_render_mode(self):
            """Cycle to the next render mode (slices <-> isosurface)."""
            current_idx = self.render_modes.index(self.render_mode)
            next_idx = (current_idx + 1) % len(self.render_modes)
            next_mode = self.render_modes[next_idx]
            print(f"[DEBUG] Cycling: {self.render_mode} -> {next_mode}")
            self._set_render_mode(next_mode)

        def _create_boundary_box(self):
            """Create a wireframe box showing the volume boundaries."""
            N = self.grid_size
            # 8 corners of the cube
            vertices = np.array([
                [-N/2, -N/2, -N/2], [N/2, -N/2, -N/2],
                [N/2, N/2, -N/2], [-N/2, N/2, -N/2],
                [-N/2, -N/2, N/2], [N/2, -N/2, N/2],
                [N/2, N/2, N/2], [-N/2, N/2, N/2],
            ], dtype=np.float32)

            # 12 edges of the cube
            edges = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
            ], dtype=np.uint32)

            self.boundary_box = scene.visuals.Line(
                pos=vertices,
                connect=edges,
                color=(0.3, 0.4, 0.5, 0.5),
                parent=self.view.scene,
            )

        def _create_source_markers(self):
            """Create visual markers for source positions."""
            # Create markers visual (spheres at source locations)
            self.source_markers = scene.visuals.Markers(
                parent=self.view.scene,
            )
            self._update_source_markers()

        def _get_source_positions(self) -> np.ndarray:
            """Get array of source positions for visualization."""
            if not self.sim.sources:
                return np.zeros((1, 3), dtype=np.float32)  # Dummy point

            positions = []
            for src in self.sim.sources:
                # Convert from physical coords to grid coords for display
                # Physical domain is [-L/2, L/2], display is [-N/2, N/2]
                scale = self.grid_size / self.sim.L_domain
                x = src['x'] * scale
                y = src['y'] * scale
                z = src['z'] * scale
                positions.append([x, y, z])

            return np.array(positions, dtype=np.float32)

        def _update_source_markers(self):
            """Update source marker positions and appearance."""
            positions = self._get_source_positions()

            if len(self.sim.sources) == 0:
                # Hide markers if no sources
                self.source_markers.set_data(
                    pos=np.zeros((1, 3)),
                    face_color=(0, 0, 0, 0),
                    size=0
                )
            else:
                # Show white markers with gold edge
                self.source_markers.set_data(
                    pos=positions,
                    face_color=(1.0, 1.0, 1.0, 0.9),
                    edge_color=(1.0, 0.8, 0.3, 1.0),
                    edge_width=2,
                    size=12,
                    symbol='o'
                )

        def _prepare_slice_for_axis(self, axis: str, index: int) -> np.ndarray:
            """Prepare a 2D slice with enhanced contrast for the given axis.

            vispy Image expects data[row, col] = data[y, x], so we transpose
            the physics data L[x, y, z] to match the visual coordinate system.
            """
            if axis == 'x':
                # YZ plane: L[x_index, y, z] -> want [z, y] for Image (row=z, col=y)
                slice_data = self.sim.L[index, :, :].T
            elif axis == 'y':
                # XZ plane: L[x, y_index, z] -> want [z, x] for Image (row=z, col=x)
                slice_data = self.sim.L[:, index, :].T
            else:  # z
                # XY plane: L[x, y, z_index] -> want [y, x] for Image (row=y, col=x)
                slice_data = self.sim.L[:, :, index].T

            L_absmax = max(np.abs(slice_data).max(), 0.05)
            slice_norm = np.clip(slice_data / L_absmax, -1, 1)
            slice_enhanced = np.sign(slice_norm) * np.sqrt(np.abs(slice_norm))
            return ((slice_enhanced + 1) / 2).astype(np.float32)

        def _create_slice_visual(self):
            """Create the 2D slice plane visualization."""
            slice_data = self._prepare_slice_data()

            self.slice_image = scene.visuals.Image(
                slice_data,
                parent=self.view.scene,
                cmap=self.cmap,
                clim=(0, 1),  # Match the 0-1 range of our normalized data
            )
            self.slice_image.transform = STTransform(
                translate=(-self.grid_size/2, -self.grid_size/2, self.slice_index - self.grid_size/2),
                scale=(1, 1, 1)
            )
            self.slice_image.visible = self.show_slice

        def _prepare_slice_data(self) -> np.ndarray:
            """Prepare 2D slice data with same normalization as volume."""
            slice_data = self.sim.get_slice('z', self.slice_index)
            L_absmax = max(np.abs(slice_data).max(), 0.05)
            slice_norm = np.clip(slice_data / L_absmax, -1, 1)
            slice_enhanced = np.sign(slice_norm) * np.sqrt(np.abs(slice_norm))
            return ((slice_enhanced + 1) / 2).astype(np.float32)

        def _prepare_volume_data(self) -> np.ndarray:
            """Prepare volume data for rendering (kept for potential future use)."""
            L = self.sim.L
            L_absmax = max(np.abs(L).max(), 0.05)
            L_norm = np.clip(L / L_absmax, -1, 1)
            L_enhanced = np.sign(L_norm) * np.sqrt(np.abs(L_norm))
            volume_data = ((L_enhanced + 1) / 2).astype(np.float32)
            return np.transpose(volume_data, (2, 1, 0))

        def _setup_controls(self, parent_layout):
            """Set up the control panel."""
            control_panel = QtWidgets.QWidget()
            control_panel.setFixedWidth(350)
            control_panel.setStyleSheet("""
                QWidget { color: #b0b0b0; }
                QLabel { font-size: 11px; }
                QPushButton {
                    background-color: #252540;
                    border: 1px solid #404060;
                    border-radius: 4px;
                    padding: 8px;
                    color: #b0b0b0;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #353560; }
                QPushButton:pressed { background-color: #454580; }
                QPushButton:checked { background-color: #406080; border-color: #60a0ff; }
                QSlider::groove:horizontal {
                    background: #252540;
                    height: 8px;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #6080a0;
                    width: 16px;
                    margin: -4px 0;
                    border-radius: 8px;
                }
                QGroupBox {
                    border: 1px solid #404060;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    color: #80a0c0;
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
            """)

            controls_layout = QtWidgets.QVBoxLayout(control_panel)
            controls_layout.setSpacing(15)

            # Title
            title = QtWidgets.QLabel("COSMIC LOOM THEORY")
            title.setStyleSheet("font-size: 16px; font-weight: bold; color: #80a0c0;")
            title.setAlignment(QtCore.Qt.AlignCenter)
            controls_layout.addWidget(title)

            subtitle = QtWidgets.QLabel("3D Loomfield Simulator")
            subtitle.setStyleSheet("font-size: 12px; color: #606080;")
            subtitle.setAlignment(QtCore.Qt.AlignCenter)
            controls_layout.addWidget(subtitle)

            controls_layout.addSpacing(10)

            # Metrics display
            self._setup_metrics(controls_layout)

            # Preset buttons
            self._setup_presets(controls_layout)

            # Mode buttons
            self._setup_modes(controls_layout)

            # Sliders
            self._setup_sliders(controls_layout)

            # Slice controls
            self._setup_slice_controls(controls_layout)

            # Control buttons
            self._setup_buttons(controls_layout)

            controls_layout.addStretch()

            # Instructions
            instructions = QtWidgets.QLabel(
                "CONTROLS:\n"
                "• Click to add source/perturbation\n"
                "• Drag to rotate view\n"
                "• Scroll to zoom\n"
                "• 'V' = toggle Slices/Isosurface\n"
                "• 'R' = reset camera\n"
                "• 'S' = random source\n"
                "• 'P' = random perturbation"
            )
            instructions.setStyleSheet("font-size: 10px; color: #606080; padding: 10px;")
            controls_layout.addWidget(instructions)

            parent_layout.addWidget(control_panel)

        def _setup_metrics(self, layout):
            """Set up metrics display."""
            metrics_group = QtWidgets.QGroupBox("METRICS")
            metrics_layout = QtWidgets.QGridLayout(metrics_group)

            self.q_label = QtWidgets.QLabel("Q: --")
            self.q_label.setStyleSheet("font-size: 14px; color: #80ffaa;")
            metrics_layout.addWidget(QtWidgets.QLabel("Coherence:"), 0, 0)
            metrics_layout.addWidget(self.q_label, 0, 1)

            self.cbio_label = QtWidgets.QLabel("C_bio: --")
            self.cbio_label.setStyleSheet("font-size: 14px; color: #ffaa80;")
            metrics_layout.addWidget(QtWidgets.QLabel("Consciousness:"), 1, 0)
            metrics_layout.addWidget(self.cbio_label, 1, 1)

            self.time_label = QtWidgets.QLabel("t: 0.0")
            self.time_label.setStyleSheet("font-size: 12px; color: #8080a0;")
            metrics_layout.addWidget(QtWidgets.QLabel("Time:"), 2, 0)
            metrics_layout.addWidget(self.time_label, 2, 1)

            self.sources_label = QtWidgets.QLabel("Sources: 0")
            self.sources_label.setStyleSheet("font-size: 12px; color: #8080a0;")
            metrics_layout.addWidget(QtWidgets.QLabel("Active:"), 3, 0)
            metrics_layout.addWidget(self.sources_label, 3, 1)

            # Render mode indicator
            self.mode_label = QtWidgets.QLabel(f"Render: {self.render_mode.upper()}")
            self.mode_label.setStyleSheet("font-size: 12px; color: #a080ff; font-weight: bold;")
            metrics_layout.addWidget(QtWidgets.QLabel("Mode:"), 4, 0)
            metrics_layout.addWidget(self.mode_label, 4, 1)

            layout.addWidget(metrics_group)

        def _setup_presets(self, layout):
            """Set up preset buttons."""
            presets_group = QtWidgets.QGroupBox("PRESETS")
            presets_layout = QtWidgets.QHBoxLayout(presets_group)

            btn_healthy = QtWidgets.QPushButton("Healthy")
            btn_healthy.clicked.connect(lambda: self._load_preset('healthy'))
            btn_healthy.setStyleSheet("QPushButton { background-color: #204020; }")
            presets_layout.addWidget(btn_healthy)

            btn_pathology = QtWidgets.QPushButton("Pathology")
            btn_pathology.clicked.connect(lambda: self._load_preset('pathology'))
            btn_pathology.setStyleSheet("QPushButton { background-color: #402020; }")
            presets_layout.addWidget(btn_pathology)

            btn_healing = QtWidgets.QPushButton("Healing")
            btn_healing.clicked.connect(lambda: self._load_preset('healing'))
            btn_healing.setStyleSheet("QPushButton { background-color: #203040; }")
            presets_layout.addWidget(btn_healing)

            layout.addWidget(presets_group)

        def _setup_modes(self, layout):
            """Set up interaction mode buttons."""
            modes_group = QtWidgets.QGroupBox("CLICK MODE")
            modes_layout = QtWidgets.QHBoxLayout(modes_group)

            self.btn_source = QtWidgets.QPushButton("Add Source")
            self.btn_source.setCheckable(True)
            self.btn_source.setChecked(True)
            self.btn_source.clicked.connect(lambda: self._set_mode('add_source'))
            modes_layout.addWidget(self.btn_source)

            self.btn_perturb = QtWidgets.QPushButton("Perturbation")
            self.btn_perturb.setCheckable(True)
            self.btn_perturb.clicked.connect(lambda: self._set_mode('perturbation'))
            modes_layout.addWidget(self.btn_perturb)

            layout.addWidget(modes_group)

        def _setup_sliders(self, layout):
            """Set up parameter sliders."""
            sliders_group = QtWidgets.QGroupBox("PARAMETERS")
            sliders_layout = QtWidgets.QVBoxLayout(sliders_group)

            # v_L slider
            vl_layout = QtWidgets.QHBoxLayout()
            vl_layout.addWidget(QtWidgets.QLabel("v_L (speed):"))
            self.vl_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.vl_slider.setRange(20, 300)
            self.vl_slider.setValue(100)
            self.vl_slider.valueChanged.connect(self._update_vl)
            vl_layout.addWidget(self.vl_slider)
            self.vl_value = QtWidgets.QLabel("1.0")
            vl_layout.addWidget(self.vl_value)
            sliders_layout.addLayout(vl_layout)

            # kappa_L slider
            kappa_layout = QtWidgets.QHBoxLayout()
            kappa_layout.addWidget(QtWidgets.QLabel("κ_L (coupling):"))
            self.kappa_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.kappa_slider.setRange(0, 50)
            self.kappa_slider.setValue(20)
            self.kappa_slider.valueChanged.connect(self._update_kappa)
            kappa_layout.addWidget(self.kappa_slider)
            self.kappa_value = QtWidgets.QLabel("2.0")
            kappa_layout.addWidget(self.kappa_value)
            sliders_layout.addLayout(kappa_layout)

            # Speed slider
            speed_layout = QtWidgets.QHBoxLayout()
            speed_layout.addWidget(QtWidgets.QLabel("Speed:"))
            self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.speed_slider.setRange(1, 10)
            self.speed_slider.setValue(3)
            self.speed_slider.valueChanged.connect(self._update_speed)
            speed_layout.addWidget(self.speed_slider)
            self.speed_value = QtWidgets.QLabel("3")
            speed_layout.addWidget(self.speed_value)
            sliders_layout.addLayout(speed_layout)

            layout.addWidget(sliders_group)

        def _setup_slice_controls(self, layout):
            """Set up slice plane controls."""
            slice_group = QtWidgets.QGroupBox("SLICE PLANE")
            slice_layout = QtWidgets.QVBoxLayout(slice_group)

            # Toggle slice visibility
            self.slice_checkbox = QtWidgets.QCheckBox("Show Slice")
            self.slice_checkbox.stateChanged.connect(self._toggle_slice)
            slice_layout.addWidget(self.slice_checkbox)

            # Slice position slider
            pos_layout = QtWidgets.QHBoxLayout()
            pos_layout.addWidget(QtWidgets.QLabel("Z Position:"))
            self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slice_slider.setRange(0, self.grid_size - 1)
            self.slice_slider.setValue(self.grid_size // 2)
            self.slice_slider.valueChanged.connect(self._update_slice_position)
            pos_layout.addWidget(self.slice_slider)
            self.slice_pos_label = QtWidgets.QLabel(str(self.grid_size // 2))
            pos_layout.addWidget(self.slice_pos_label)
            slice_layout.addLayout(pos_layout)

            layout.addWidget(slice_group)

        def _setup_buttons(self, layout):
            """Set up control buttons."""
            buttons_layout = QtWidgets.QHBoxLayout()

            self.btn_play = QtWidgets.QPushButton("⏸ Pause")
            self.btn_play.clicked.connect(self._toggle_play)
            buttons_layout.addWidget(self.btn_play)

            btn_clear = QtWidgets.QPushButton("Clear")
            btn_clear.clicked.connect(self._clear)
            buttons_layout.addWidget(btn_clear)

            btn_reset = QtWidgets.QPushButton("Reset")
            btn_reset.clicked.connect(self._reset)
            buttons_layout.addWidget(btn_reset)

            layout.addLayout(buttons_layout)

        def _setup_timer(self):
            """Set up animation timer."""
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self._update)
            self.timer.start(33)  # ~30 FPS

        def _update(self):
            """Animation update callback."""
            if not self.running:
                return

            # Step simulation
            self.sim.step(self.steps_per_frame)

            # Update visuals based on current render mode
            if self.render_mode == 'slices':
                # Update the 3 orthogonal slice planes
                center = self.grid_size // 2
                if self.slice_xy:
                    self.slice_xy.set_data(self._prepare_slice_for_axis('z', center))
                if self.slice_xz:
                    self.slice_xz.set_data(self._prepare_slice_for_axis('y', center))
                if self.slice_yz:
                    self.slice_yz.set_data(self._prepare_slice_for_axis('x', center))

            elif self.render_mode == 'isosurface':
                # Recreate isosurfaces (they need to be rebuilt for new data)
                self._update_isosurfaces()

            # Update source markers (always visible)
            self._update_source_markers()

            # Debug output every 60 frames (reduced frequency)
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 60 == 0:
                L = self.sim.L
                print(f"[DEBUG] Frame {self._frame_count}: L=[{L.min():.4f}, {L.max():.4f}], "
                      f"mode={self.render_mode}, sources={len(self.sim.sources)}")

            # Update movable slice if visible
            if self.show_slice:
                slice_data = self._prepare_slice_data()
                self.slice_image.set_data(slice_data)

            # Update metrics
            Q = self.sim.get_total_coherence()
            C_bio = self.sim.get_consciousness_metric()

            self.q_label.setText(f"Q: {Q:.3f}")
            self.cbio_label.setText(f"C_bio: {C_bio:.2f}")
            self.time_label.setText(f"t: {self.sim.time:.1f}")
            self.sources_label.setText(f"Sources: {len(self.sim.sources)}")

            # Update colors based on values
            q_color = "#80ffaa" if Q > 1.5 else ("#ffff80" if Q > 0.5 else "#ff8080")
            self.q_label.setStyleSheet(f"font-size: 14px; color: {q_color};")

            self.canvas.update()

        def _load_preset(self, preset: str):
            """Load a preset configuration."""
            print(f"[DEBUG] _load_preset called with preset='{preset}'")
            if preset == 'healthy':
                create_healthy_preset(self.sim)
            elif preset == 'pathology':
                create_pathology_preset(self.sim)
            elif preset == 'healing':
                create_healing_preset(self.sim)

            # Warm up the simulation so field is immediately visible
            print(f"[DEBUG] Running 50 warm-up steps...")
            self.sim.step(50)
            print(f"[DEBUG] After warm-up: L range=[{self.sim.L.min():.4f}, {self.sim.L.max():.4f}], "
                  f"sources={len(self.sim.sources)}")

        def _set_mode(self, mode: str):
            """Set interaction mode."""
            self.mode = mode
            self.btn_source.setChecked(mode == 'add_source')
            self.btn_perturb.setChecked(mode == 'perturbation')

        def _update_vl(self, value):
            """Update v_L from slider."""
            v = value / 100.0
            self.sim.v_L = v
            self.sim.dt = 0.4 * self.sim.dx / (v * np.sqrt(3))
            self.vl_value.setText(f"{v:.1f}")

        def _update_kappa(self, value):
            """Update kappa_L from slider."""
            k = value / 10.0
            self.sim.kappa_L = k
            self.kappa_value.setText(f"{k:.1f}")

        def _update_speed(self, value):
            """Update animation speed."""
            self.steps_per_frame = value
            self.speed_value.setText(str(value))

        def _toggle_slice(self, state):
            """Toggle slice visibility."""
            self.show_slice = state == QtCore.Qt.Checked
            self.slice_image.visible = self.show_slice

        def _update_slice_position(self, value):
            """Update slice position."""
            self.slice_index = value
            self.slice_pos_label.setText(str(value))

            # Update slice transform
            self.slice_image.transform = STTransform(
                translate=(-self.grid_size/2, -self.grid_size/2, value - self.grid_size/2),
                scale=(1, 1, 1)
            )

        def _toggle_play(self):
            """Toggle play/pause."""
            self.running = not self.running
            self.btn_play.setText("▶ Play" if not self.running else "⏸ Pause")

        def _clear(self):
            """Clear sources but keep simulation running."""
            self.sim.clear_sources()

        def _reset(self):
            """Reset entire simulation."""
            self.sim.reset()
            create_healthy_preset(self.sim)
            self.sim.step(100)  # Warm up so field is immediately visible

        def _on_mouse_press(self, event):
            """Handle mouse press - store position for click vs drag detection."""
            # Store press position for all buttons
            if event.button in [1, 2]:
                self._mouse_press_pos = event.pos

        def _on_mouse_release(self, event):
            """Handle mouse release - add source only if it was a click (not drag)."""
            if self._mouse_press_pos is None:
                return

            # Calculate movement distance
            dx = event.pos[0] - self._mouse_press_pos[0]
            dy = event.pos[1] - self._mouse_press_pos[1]
            distance = np.sqrt(dx**2 + dy**2)

            # Only add source if movement was below threshold (a "click", not a drag)
            if distance < self._click_threshold and event.button in [1, 2]:
                print(f"[DEBUG] Click detected (moved {distance:.1f}px < {self._click_threshold}px threshold)")
                pos_3d = self._screen_to_world(event.pos)
                if pos_3d is not None:
                    self._add_interaction_at(pos_3d[0], pos_3d[1], pos_3d[2])
            else:
                print(f"[DEBUG] Drag detected (moved {distance:.1f}px) - no source added")

            # Clear press position
            self._mouse_press_pos = None

        def _screen_to_world(self, screen_pos):
            """Convert screen coordinates to world coordinates on the XY plane (z=0)."""
            try:
                # Get the transform from the view
                tr = self.view.scene.transform

                # Create a ray from the camera through the click point
                # Map screen position to normalized device coordinates
                pos = np.array([[screen_pos[0], screen_pos[1], 0, 1],
                                [screen_pos[0], screen_pos[1], 1, 1]], dtype=np.float32)

                # Transform to world coordinates
                # Use the inverse of the scene transform
                world_pos = tr.imap(pos)

                if world_pos is not None and len(world_pos) >= 2:
                    # Get ray direction
                    p0 = world_pos[0][:3] / world_pos[0][3] if world_pos[0][3] != 0 else world_pos[0][:3]
                    p1 = world_pos[1][:3] / world_pos[1][3] if world_pos[1][3] != 0 else world_pos[1][:3]

                    # Find intersection with z=0 plane
                    direction = p1 - p0
                    if abs(direction[2]) > 1e-6:
                        t = -p0[2] / direction[2]
                        intersection = p0 + t * direction

                        # Convert from grid coordinates to physical coordinates
                        scale = self.sim.L_domain / self.grid_size
                        x = intersection[0] * scale
                        y = intersection[1] * scale
                        z = 0  # On the XY plane

                        # Clamp to domain
                        half = self.sim.L_domain / 2
                        x = np.clip(x, -half * 0.9, half * 0.9)
                        y = np.clip(y, -half * 0.9, half * 0.9)

                        print(f"[DEBUG] Screen ({screen_pos[0]:.0f}, {screen_pos[1]:.0f}) -> "
                              f"World ({x:.2f}, {y:.2f}, {z:.2f})")
                        return (x, y, z)
            except Exception as e:
                print(f"[DEBUG] Screen to world conversion failed: {e}")

            # Fallback: random position near center
            print("[DEBUG] Using fallback random position")
            return (np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0)

        def _on_key_press(self, event):
            """Handle key press for adding sources/perturbations and mode changes."""
            print(f"[DEBUG] Key press: key={event.key}")
            if event.key == 'S':
                print("[DEBUG] 'S' key pressed - adding source at random position")
                self._set_mode('add_source')
                self._add_interaction_at_random()
            elif event.key == 'P':
                print("[DEBUG] 'P' key pressed - adding perturbation at random position")
                self._set_mode('perturbation')
                self._add_interaction_at_random()
            elif event.key == 'T':
                # Test: inject a test pattern to verify rendering
                print("[DEBUG] 'T' key pressed - injecting test pattern")
                self._inject_test_pattern()
            elif event.key == 'V':
                # Cycle through render modes: slices -> volume -> isosurface -> slices
                print("[DEBUG] 'V' key pressed - cycling render mode")
                self._cycle_render_mode()
            elif event.key == 'R':
                # Reset camera to default view
                print("[DEBUG] 'R' key pressed - resetting camera")
                self._reset_camera()

        def _reset_camera(self):
            """Reset camera to default viewing position."""
            N = self.grid_size
            self.view.camera.distance = N * 1.5
            self.view.camera.elevation = 30
            self.view.camera.azimuth = 45
            print(f"[DEBUG] Camera reset: distance={N * 1.5}, elevation=30, azimuth=45")

        def _add_interaction_at_random(self):
            """Add a source or perturbation at a random location."""
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(-2, 2)
            self._add_interaction_at(x, y, z)

        def _add_interaction_at(self, x: float, y: float, z: float):
            """Add a source or perturbation at the specified position."""
            print(f"[DEBUG] Adding {self.mode} at ({x:.2f}, {y:.2f}, {z:.2f})")
            if self.mode == 'add_source':
                self.sim.add_source(x, y, z, strength=1.0, frequency=1.5)
            else:
                self.sim.add_perturbation(x, y, z, strength=2.5, radius=1.5)
            # Immediately update markers
            self._update_source_markers()

        def _inject_test_pattern(self):
            """Inject a test pattern directly into the field to verify rendering."""
            # Create a simple sphere pattern
            coords = np.linspace(-self.sim.L_domain/2, self.sim.L_domain/2, self.sim.N)
            X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
            dist = np.sqrt(X**2 + Y**2 + Z**2)

            # Inject sphere: max at center (0), fading outward
            radius = 3.0
            self.sim.L = np.clip(1.0 - dist/radius, 0, 1).astype(np.float32)
            print(f"[DEBUG] Test pattern injected: L range=[{self.sim.L.min():.4f}, {self.sim.L.max():.4f}]")


def demo():
    """Run the real-time 3D Loomfield visualizer."""
    if not HAS_VISPY or not HAS_QT:
        print("\n" + "=" * 60)
        print("  MISSING DEPENDENCIES")
        print("=" * 60)
        print("\nThe real-time 3D visualizer requires vispy and PyQt5.")
        print("\nInstall with:")
        print("  pip install vispy PyQt5")
        print("\nAlternatively, use the plotly-based visualizer:")
        print("  python run_loomfield_3d_demo.py")
        print("=" * 60)
        return

    print("=" * 60)
    print("  COSMIC LOOM THEORY: Real-Time 3D Loomfield Simulator")
    print("=" * 60)
    print("""
Starting real-time 3D visualization...

CONTROLS:
  • LEFT CLICK on slice plane to add source/perturbation
  • Drag to rotate view
  • Scroll to zoom in/out
  • 'V' key = toggle render mode (Slices <-> Isosurface)
  • 'R' key = reset camera view
  • 'S' key = add source at random position
  • 'P' key = add perturbation at random position
  • Use preset buttons to load scenarios
  • Toggle between Add Source / Perturbation modes
  • White markers show source locations

RENDER MODES:
  • Slices: 3 orthogonal slice planes (fast, shows interior)
  • Isosurface: Nested transparent shells (3D structure)
    """)

    # Create Qt application
    qt_app = QtWidgets.QApplication([])
    qt_app.setStyle('Fusion')

    # Create and show visualizer
    viz = Loomfield3DVisualizer(grid_size=32)
    viz.show()

    # Run event loop
    qt_app.exec_()


if __name__ == "__main__":
    demo()
