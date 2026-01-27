"""
Biophoton Emission Simulator for Cosmic Loom Theory

Models ultra-weak photon emission from living tissue, demonstrating how
biophotonic activity relates to metabolic state and coherence.

From CLT v1.1 Section 4.2:
Living systems emit ultra-weak photons (biophotons) as a natural consequence
of metabolic activity, with mitochondria as a major source. Biophotonic
emission exhibits properties that distinguish it from random thermal radiation:
- Non-Poissonian statistics (can show coherent/squeezed states)
- Spatial coherence over short distances
- Sensitivity to physiological state

In CLT: Biophotons are a potential medium for TEMPORAL SYNCHRONIZATION of
coherent dynamics. Optical signaling provides a mechanism by which distant
subsystems may align phase relationships without requiring large energy
expenditure.

Key concepts:
- Mitochondria as primary emission sources
- Emission rate linked to metabolic activity (ATP production)
- Oxidative stress increases emission (reactive oxygen species)
- Coherent vs incoherent emission statistics
- Spatial and temporal coherence metrics
- LoomSense measurement compatibility
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.animation as animation
from scipy import ndimage
from scipy.stats import poisson


# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

# Biophoton wavelength range (nm) - UV to visible
WAVELENGTH_MIN = 200.0   # UV
WAVELENGTH_MAX = 800.0   # Red/near-IR
WAVELENGTH_PEAK = 500.0  # Green (typical peak)

# Emission rates (photons per cell per second)
EMISSION_RATE_BASELINE = 10.0      # Healthy resting cell
EMISSION_RATE_STRESSED = 100.0     # Under oxidative stress
EMISSION_RATE_APOPTOTIC = 500.0    # Cell death burst

# Metabolic parameters
ATP_BASELINE = 1.0         # Normalized ATP level
ATP_STRESSED = 0.5         # Reduced ATP under stress
ROS_BASELINE = 0.1         # Reactive oxygen species (normalized)
ROS_STRESSED = 1.0         # High ROS under oxidative stress

# Coherence parameters
COHERENCE_LENGTH_DEFAULT = 3.0    # Spatial coherence length (cells)
COHERENCE_TIME_DEFAULT = 100.0    # Temporal coherence (ms)
COUPLING_STRENGTH_DEFAULT = 0.3   # Inter-cell phase coupling

# Mitochondrial parameters
MITO_DENSITY_DEFAULT = 100        # Mitochondria per cell
MITO_ACTIVITY_DEFAULT = 1.0       # Normalized activity


# =============================================================================
# EMISSION STATE AND STATISTICS
# =============================================================================

class EmissionMode(Enum):
    """Types of photon emission statistics."""
    POISSONIAN = "poissonian"       # Random/thermal (incoherent)
    COHERENT = "coherent"           # Phase-locked (laser-like)
    SQUEEZED = "squeezed"           # Sub-Poissonian (quantum)
    CHAOTIC = "chaotic"             # Super-Poissonian (thermal)


class TissueState(Enum):
    """Physiological states affecting emission."""
    HEALTHY = 0
    STRESSED = 1
    INFLAMED = 2
    APOPTOTIC = 3
    PROLIFERATING = 4


@dataclass
class CellState:
    """
    State of a single cell affecting biophoton emission.

    Attributes:
        atp_level: Normalized ATP concentration (0-2, baseline 1)
        ros_level: Reactive oxygen species level (0-2, baseline 0.1)
        mito_density: Number of mitochondria (affects emission capacity)
        mito_activity: Mitochondrial activity level (0-2)
        emission_phase: Phase for coherent emission (0-2π)
        tissue_state: Physiological state category
    """
    atp_level: float = ATP_BASELINE
    ros_level: float = ROS_BASELINE
    mito_density: float = MITO_DENSITY_DEFAULT
    mito_activity: float = MITO_ACTIVITY_DEFAULT
    emission_phase: float = 0.0
    tissue_state: TissueState = TissueState.HEALTHY


# =============================================================================
# BIOPHOTON SIMULATOR
# =============================================================================

class BiophotonSimulator:
    """
    Simulates biophoton emission from a 2D tissue grid.

    Models photon emission based on cellular metabolic state, with
    support for coherent and incoherent emission modes. Tracks
    spatial and temporal coherence metrics relevant to CLT.

    Parameters:
        grid_size: (rows, cols) dimensions of tissue grid
        dt: Time step in milliseconds
        emission_mode: Type of emission statistics
        coherence_length: Spatial coherence length in cells
        coherence_time: Temporal coherence in ms
        coupling_strength: Inter-cell phase coupling strength
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (50, 50),
        dt: float = 1.0,  # ms
        emission_mode: EmissionMode = EmissionMode.POISSONIAN,
        coherence_length: float = COHERENCE_LENGTH_DEFAULT,
        coherence_time: float = COHERENCE_TIME_DEFAULT,
        coupling_strength: float = COUPLING_STRENGTH_DEFAULT
    ):
        self.rows, self.cols = grid_size
        self.dt = dt
        self.emission_mode = emission_mode
        self.coherence_length = coherence_length
        self.coherence_time = coherence_time
        self.coupling_strength = coupling_strength

        # Initialize cellular states
        self._init_cells()

        # Emission tracking
        self.emission_counts = np.zeros((self.rows, self.cols))  # Current step
        self.emission_history = []  # Time series of total emission
        self.wavelength_history = []  # Spectrum over time

        # Coherence tracking
        self.spatial_coherence_history = []
        self.temporal_coherence_history = []
        self.phase_coherence_history = []

        # Time tracking
        self.time = 0.0
        self.step_count = 0

        # For temporal coherence calculation
        self._emission_buffer = []
        self._buffer_size = int(coherence_time / dt) if dt > 0 else 100

    def _init_cells(self):
        """Initialize cellular state arrays."""
        # Metabolic state
        self.atp = np.ones((self.rows, self.cols)) * ATP_BASELINE
        self.ros = np.ones((self.rows, self.cols)) * ROS_BASELINE

        # Mitochondrial state
        self.mito_density = np.ones((self.rows, self.cols)) * MITO_DENSITY_DEFAULT
        self.mito_activity = np.ones((self.rows, self.cols)) * MITO_ACTIVITY_DEFAULT

        # Phase for coherent emission (random initial)
        self.emission_phase = np.random.uniform(0, 2*np.pi, (self.rows, self.cols))

        # Tissue state mask (all healthy initially)
        self.tissue_state = np.zeros((self.rows, self.cols), dtype=int)

        # Emission rate (computed from state)
        self._update_emission_rates()

    def _update_emission_rates(self):
        """Compute emission rates from cellular state."""
        # Base rate from mitochondrial activity
        base_rate = self.mito_density * self.mito_activity / MITO_DENSITY_DEFAULT

        # Modulation by ROS (oxidative stress increases emission)
        ros_factor = 1.0 + 5.0 * (self.ros - ROS_BASELINE)
        ros_factor = np.clip(ros_factor, 0.5, 10.0)

        # Modulation by ATP (low ATP can increase emission via stress)
        atp_factor = 1.0 + 0.5 * (1.0 - self.atp)
        atp_factor = np.clip(atp_factor, 0.8, 2.0)

        # Combined emission rate
        self.emission_rate = EMISSION_RATE_BASELINE * base_rate * ros_factor * atp_factor

    def _generate_emissions_poissonian(self) -> np.ndarray:
        """Generate Poissonian (random) emission counts."""
        # Each cell emits independently with Poisson statistics
        mean_counts = self.emission_rate * self.dt / 1000.0  # per timestep
        return np.random.poisson(mean_counts)

    def _generate_emissions_coherent(self) -> np.ndarray:
        """Generate coherent (phase-locked) emission."""
        # Emission modulated by phase
        phase_factor = 0.5 * (1.0 + np.cos(self.emission_phase))
        mean_counts = self.emission_rate * self.dt / 1000.0 * phase_factor

        # Still has some randomness but with phase structure
        counts = np.random.poisson(mean_counts * 2)  # Higher peak
        return counts

    def _generate_emissions_squeezed(self) -> np.ndarray:
        """Generate squeezed (sub-Poissonian) emission."""
        # Reduced variance compared to Poisson
        mean_counts = self.emission_rate * self.dt / 1000.0

        # Sub-Poissonian: variance < mean
        # Use a narrower distribution
        noise = np.random.normal(0, np.sqrt(mean_counts) * 0.5, (self.rows, self.cols))
        counts = np.maximum(0, mean_counts + noise).astype(int)
        return counts

    def _generate_emissions_chaotic(self) -> np.ndarray:
        """Generate chaotic (super-Poissonian) emission."""
        # Increased variance compared to Poisson (thermal bunching)
        mean_counts = self.emission_rate * self.dt / 1000.0

        # Super-Poissonian: variance > mean
        # Negative binomial approximation
        r = 2.0  # Shape parameter (lower = more variance)
        p = r / (r + mean_counts + 0.001)
        counts = np.random.negative_binomial(r, p)
        return counts

    def _update_phases(self):
        """Update emission phases with coupling to neighbors."""
        # Natural frequency (based on mitochondrial activity)
        omega = 2 * np.pi * self.mito_activity / self.coherence_time

        # Kuramoto-like coupling to neighbors
        coupling = np.zeros_like(self.emission_phase)

        # 4-neighbor coupling
        coupling += np.roll(self.emission_phase, 1, axis=0)  # Up
        coupling += np.roll(self.emission_phase, -1, axis=0)  # Down
        coupling += np.roll(self.emission_phase, 1, axis=1)  # Left
        coupling += np.roll(self.emission_phase, -1, axis=1)  # Right
        coupling = coupling / 4.0

        # Phase update with coupling
        phase_diff = np.sin(coupling - self.emission_phase)
        self.emission_phase += self.dt * (omega + self.coupling_strength * phase_diff)

        # Keep in [0, 2π]
        self.emission_phase = self.emission_phase % (2 * np.pi)

    def _generate_wavelengths(self, n_photons: int) -> np.ndarray:
        """Generate wavelengths for emitted photons."""
        if n_photons == 0:
            return np.array([])

        # Biophoton spectrum peaks around 500nm (green)
        # with tails into UV and red
        wavelengths = np.random.normal(WAVELENGTH_PEAK, 80, n_photons)
        wavelengths = np.clip(wavelengths, WAVELENGTH_MIN, WAVELENGTH_MAX)
        return wavelengths

    def step(self, n_steps: int = 1):
        """
        Advance simulation by n_steps.

        Parameters:
            n_steps: Number of time steps to simulate
        """
        for _ in range(n_steps):
            # Update emission rates from current state
            self._update_emission_rates()

            # Generate emissions based on mode
            if self.emission_mode == EmissionMode.POISSONIAN:
                self.emission_counts = self._generate_emissions_poissonian()
            elif self.emission_mode == EmissionMode.COHERENT:
                self.emission_counts = self._generate_emissions_coherent()
            elif self.emission_mode == EmissionMode.SQUEEZED:
                self.emission_counts = self._generate_emissions_squeezed()
            else:  # CHAOTIC
                self.emission_counts = self._generate_emissions_chaotic()

            # Update phases for coherent modes
            if self.emission_mode in [EmissionMode.COHERENT, EmissionMode.SQUEEZED]:
                self._update_phases()

            # Generate wavelengths
            total_photons = int(np.sum(self.emission_counts))
            wavelengths = self._generate_wavelengths(total_photons)

            # Record history
            self.emission_history.append(total_photons)
            if len(wavelengths) > 0:
                self.wavelength_history.append(wavelengths)

            # Update emission buffer for temporal coherence
            self._emission_buffer.append(self.emission_counts.copy())
            if len(self._emission_buffer) > self._buffer_size:
                self._emission_buffer.pop(0)

            # Record coherence metrics
            self.spatial_coherence_history.append(self.compute_spatial_coherence())
            self.temporal_coherence_history.append(self.compute_temporal_coherence())
            self.phase_coherence_history.append(self.compute_phase_coherence())

            # Update time
            self.time += self.dt
            self.step_count += 1

    def run(self, duration: float, record_interval: float = 10.0):
        """
        Run simulation for specified duration.

        Parameters:
            duration: Total simulation time in ms
            record_interval: Interval for recording (not used, records every step)
        """
        n_steps = int(duration / self.dt)
        self.step(n_steps)

    # =========================================================================
    # COHERENCE METRICS
    # =========================================================================

    def compute_spatial_coherence(self) -> float:
        """
        Compute spatial coherence of emission pattern.

        Measures correlation between nearby cells' emission patterns.
        High coherence = spatially organized emission.

        Returns:
            Spatial coherence index (0-1)
        """
        if np.std(self.emission_counts) < 1e-10:
            return 0.5  # Uniform = baseline coherence

        # Compute correlation with shifted versions
        correlations = []

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            shifted = np.roll(np.roll(self.emission_counts, dr, axis=0), dc, axis=1)
            if np.std(shifted) > 1e-10:
                corr = np.corrcoef(self.emission_counts.flatten(), shifted.flatten())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if len(correlations) == 0:
            return 0.5

        # Map correlation (-1 to 1) to coherence (0 to 1)
        mean_corr = np.mean(correlations)
        return 0.5 * (1.0 + mean_corr)

    def compute_temporal_coherence(self) -> float:
        """
        Compute temporal coherence of emission.

        Measures autocorrelation of emission over time.
        High coherence = rhythmic/synchronized emission.

        Returns:
            Temporal coherence index (0-1)
        """
        if len(self._emission_buffer) < 10:
            return 0.5  # Not enough data

        # Total emission over time
        totals = [np.sum(e) for e in self._emission_buffer]

        if np.std(totals) < 1e-10:
            return 0.5  # Constant = baseline

        # Autocorrelation at lag 1
        totals = np.array(totals)
        autocorr = np.corrcoef(totals[:-1], totals[1:])[0, 1]

        if np.isnan(autocorr):
            return 0.5

        # Map to 0-1 range
        return 0.5 * (1.0 + autocorr)

    def compute_phase_coherence(self) -> float:
        """
        Compute phase coherence (Kuramoto order parameter).

        Measures how synchronized the emission phases are.
        High coherence = phase-locked emission across tissue.

        Returns:
            Phase coherence (0-1)
        """
        # Kuramoto order parameter: |<e^{i*phase}>|
        complex_phases = np.exp(1j * self.emission_phase)
        order_param = np.abs(np.mean(complex_phases))
        return order_param

    def compute_emission_statistics(self) -> Dict:
        """
        Compute emission statistics for current state.

        Returns:
            Dictionary with mean, variance, Fano factor, etc.
        """
        counts = self.emission_counts.flatten()

        mean = np.mean(counts)
        var = np.var(counts)

        # Fano factor: variance/mean
        # F = 1 for Poisson, F < 1 for squeezed, F > 1 for chaotic
        fano = var / (mean + 1e-10)

        # Total emission
        total = np.sum(counts)

        # Rate per cell per second
        rate_per_cell = total / (self.rows * self.cols) / (self.dt / 1000.0)

        return {
            'mean': mean,
            'variance': var,
            'fano_factor': fano,
            'total_photons': total,
            'rate_per_cell': rate_per_cell,
            'emission_mode': self.emission_mode.value
        }

    def compute_spectrum(self, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute emission spectrum from recent wavelengths.

        Parameters:
            n_bins: Number of wavelength bins

        Returns:
            (wavelengths, intensities) arrays
        """
        # Combine recent wavelengths
        if len(self.wavelength_history) == 0:
            bins = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            return centers, np.zeros(n_bins)

        # Use last N records
        n_recent = min(100, len(self.wavelength_history))
        recent = np.concatenate(self.wavelength_history[-n_recent:])

        if len(recent) == 0:
            bins = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            return centers, np.zeros(n_bins)

        # Histogram
        counts, bins = np.histogram(recent, bins=n_bins,
                                    range=(WAVELENGTH_MIN, WAVELENGTH_MAX))
        centers = 0.5 * (bins[:-1] + bins[1:])

        # Normalize
        if np.sum(counts) > 0:
            counts = counts / np.sum(counts)

        return centers, counts

    # =========================================================================
    # STATE MANIPULATION
    # =========================================================================

    def set_emission_mode(self, mode: EmissionMode):
        """Change emission statistics mode."""
        self.emission_mode = mode

    def set_metabolic_rate(self, rate: float, region: Optional[Tuple] = None):
        """
        Set metabolic rate (affects mitochondrial activity).

        Parameters:
            rate: Metabolic rate multiplier (1.0 = baseline)
            region: Optional (row_slice, col_slice) to affect only part
        """
        if region is None:
            self.mito_activity[:, :] = MITO_ACTIVITY_DEFAULT * rate
        else:
            r_slice, c_slice = region
            self.mito_activity[r_slice, c_slice] = MITO_ACTIVITY_DEFAULT * rate
        self._update_emission_rates()

    def induce_oxidative_stress(self, center: Tuple[int, int], radius: float,
                                intensity: float = 1.0):
        """
        Induce oxidative stress in a region.

        Parameters:
            center: (row, col) center of stress
            radius: Radius of affected region
            intensity: Stress intensity (0-1, where 1 = max stress)
        """
        y, x = np.ogrid[:self.rows, :self.cols]
        dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        mask = dist <= radius

        # Increase ROS
        stress_level = ROS_BASELINE + intensity * (ROS_STRESSED - ROS_BASELINE)
        self.ros[mask] = stress_level

        # Decrease ATP (stress depletes energy)
        atp_level = ATP_BASELINE - intensity * (ATP_BASELINE - ATP_STRESSED)
        self.atp[mask] = atp_level

        self._update_emission_rates()

    def induce_apoptosis(self, center: Tuple[int, int], radius: float):
        """
        Trigger apoptosis (cell death) in a region.

        Apoptosis causes a burst of biophoton emission.

        Parameters:
            center: (row, col) center
            radius: Radius of affected region
        """
        y, x = np.ogrid[:self.rows, :self.cols]
        dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        mask = dist <= radius

        # Massive ROS increase
        self.ros[mask] = ROS_STRESSED * 2.0

        # ATP depletion
        self.atp[mask] = 0.1

        # Mark tissue state
        self.tissue_state[mask] = TissueState.APOPTOTIC.value

        self._update_emission_rates()

    def set_coupling_strength(self, strength: float):
        """Set inter-cell phase coupling strength."""
        self.coupling_strength = strength

    def synchronize_phases(self, coherence: float = 1.0):
        """
        Artificially synchronize emission phases.

        Parameters:
            coherence: 0 = random phases, 1 = fully synchronized
        """
        if coherence >= 1.0:
            # All same phase
            self.emission_phase[:, :] = 0.0
        else:
            # Partial synchronization
            mean_phase = np.mean(self.emission_phase)
            self.emission_phase = (1 - coherence) * self.emission_phase + coherence * mean_phase

    def reset(self):
        """Reset simulation to initial state."""
        self._init_cells()
        self.emission_counts = np.zeros((self.rows, self.cols))
        self.emission_history = []
        self.wavelength_history = []
        self.spatial_coherence_history = []
        self.temporal_coherence_history = []
        self.phase_coherence_history = []
        self._emission_buffer = []
        self.time = 0.0
        self.step_count = 0

    # =========================================================================
    # CLT / ÉR PHASE SPACE MAPPING
    # =========================================================================

    def map_to_er_space(self) -> Dict:
        """
        Map biophoton emission state to éR phase space.

        In CLT, biophoton coherence relates to the overall coherence
        of the Loomfield. This maps emission metrics to éR components.

        Returns:
            Dictionary with éR mapping:
            - energy_present: Total emission (metabolic energy → photons)
            - frequency: Effective emission frequency
            - energy_resistance: éR = EP / f²
            - coherence_index: Combined coherence measure
        """
        # Energy present: proportional to emission rate
        total_emission = np.sum(self.emission_rate)
        ep = total_emission / (self.rows * self.cols)  # Normalized

        # Frequency: based on temporal dynamics
        # Higher coherence → lower effective frequency (more organized)
        phase_coh = self.compute_phase_coherence()
        base_freq = np.mean(self.mito_activity) * 10.0  # Hz
        freq = base_freq * (2.0 - phase_coh)  # Coherence reduces effective freq
        freq = max(freq, 0.1)  # Prevent division by zero

        # Energy resistance
        er = ep / (freq ** 2)

        # Combined coherence index
        spatial_coh = self.compute_spatial_coherence()
        temporal_coh = self.compute_temporal_coherence()
        coherence_index = (spatial_coh + temporal_coh + phase_coh) / 3.0

        # Emission statistics
        stats = self.compute_emission_statistics()

        return {
            'energy_present': ep,
            'frequency': freq,
            'energy_resistance': er,
            'coherence_index': coherence_index,
            'spatial_coherence': spatial_coh,
            'temporal_coherence': temporal_coh,
            'phase_coherence': phase_coh,
            'fano_factor': stats['fano_factor'],
            'emission_mode': self.emission_mode.value
        }

    # =========================================================================
    # LOOMSENSE OUTPUT
    # =========================================================================

    def get_loomsense_output(self) -> Dict:
        """
        Generate output metrics compatible with LoomSense hardware.

        These are the measurements that photodetector arrays
        would actually record from living tissue.

        Returns:
            Dictionary of measurable quantities
        """
        stats = self.compute_emission_statistics()
        wavelengths, spectrum = self.compute_spectrum()

        # Peak wavelength
        if np.sum(spectrum) > 0:
            peak_idx = np.argmax(spectrum)
            peak_wavelength = wavelengths[peak_idx]
        else:
            peak_wavelength = WAVELENGTH_PEAK

        # Spectral width (FWHM approximation)
        if np.sum(spectrum) > 0:
            half_max = np.max(spectrum) / 2
            above_half = spectrum > half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                spectral_width = wavelengths[indices[-1]] - wavelengths[indices[0]]
            else:
                spectral_width = 0.0
        else:
            spectral_width = 0.0

        return {
            # Intensity metrics
            'total_photon_count': stats['total_photons'],
            'mean_intensity': stats['mean'],
            'emission_rate_per_cell': stats['rate_per_cell'],

            # Statistical metrics
            'fano_factor': stats['fano_factor'],
            'intensity_variance': stats['variance'],

            # Spectral metrics
            'peak_wavelength_nm': peak_wavelength,
            'spectral_width_nm': spectral_width,

            # Coherence metrics
            'spatial_coherence': self.compute_spatial_coherence(),
            'temporal_coherence': self.compute_temporal_coherence(),
            'phase_coherence': self.compute_phase_coherence(),

            # Metabolic indicators
            'mean_ros_level': np.mean(self.ros),
            'mean_atp_level': np.mean(self.atp),
            'metabolic_stress_index': np.mean(self.ros) / np.mean(self.atp),

            # Timestamp
            'time_ms': self.time,
            'sample_count': self.step_count
        }


# =============================================================================
# VISUALIZER
# =============================================================================

class BiophotonVisualizer:
    """
    Interactive visualization for biophoton emission simulation.

    Features:
    - Emission intensity heat map with photon flashes
    - Spectrum display
    - Coherence metrics over time
    - Interactive controls for parameters
    - éR phase space view
    """

    def __init__(self, simulator: BiophotonSimulator):
        self.sim = simulator
        self.running = False
        self.show_flashes = True

        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Biophoton Emission Simulator - CLT Phase 2.2',
                         fontsize=14, fontweight='bold')

        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[2, 1, 1],
                     hspace=0.3, wspace=0.3)

        # Main emission display
        self.ax_emission = self.fig.add_subplot(gs[0, :2])
        self.ax_emission.set_title('Emission Intensity')

        # Spectrum display
        self.ax_spectrum = self.fig.add_subplot(gs[0, 2])
        self.ax_spectrum.set_title('Emission Spectrum')
        self.ax_spectrum.set_xlabel('Wavelength (nm)')
        self.ax_spectrum.set_ylabel('Intensity')

        # Coherence metrics
        self.ax_coherence = self.fig.add_subplot(gs[0, 3])
        self.ax_coherence.set_title('Coherence Metrics')

        # Time series
        self.ax_timeseries = self.fig.add_subplot(gs[1, :2])
        self.ax_timeseries.set_title('Emission History')
        self.ax_timeseries.set_xlabel('Time (ms)')
        self.ax_timeseries.set_ylabel('Total Photons')

        # Statistics display
        self.ax_stats = self.fig.add_subplot(gs[1, 2:])
        self.ax_stats.set_title('LoomSense Output')
        self.ax_stats.axis('off')

        # Controls area
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')

        # Create colormap (dark to bright yellow-green for biophotons)
        colors = ['#000510', '#001030', '#003060', '#006040',
                  '#40A060', '#80FF80', '#FFFF80', '#FFFFFF']
        self.cmap = LinearSegmentedColormap.from_list('biophoton', colors)

        # Initialize plots
        self._init_plots()
        self._add_controls()

        # Animation
        self.anim = None

    def _init_plots(self):
        """Initialize all plot elements."""
        # Emission heat map
        self.emission_img = self.ax_emission.imshow(
            self.sim.emission_counts,
            cmap=self.cmap,
            aspect='equal',
            vmin=0,
            vmax=10,
            interpolation='nearest'
        )
        self.fig.colorbar(self.emission_img, ax=self.ax_emission, label='Photon Count')

        # Flash overlay (scatter plot for individual photons)
        self.flash_scatter = self.ax_emission.scatter([], [], c='white', s=20,
                                                       alpha=0.8, marker='*')

        # Spectrum
        wavelengths, spectrum = self.sim.compute_spectrum()
        self.spectrum_line, = self.ax_spectrum.plot(wavelengths, spectrum,
                                                    color='#40FF80', linewidth=2)
        self.ax_spectrum.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)
        self.ax_spectrum.set_ylim(0, 0.1)
        self.ax_spectrum.axvline(WAVELENGTH_PEAK, color='white', linestyle='--',
                                 alpha=0.5, label=f'Peak ~{WAVELENGTH_PEAK}nm')

        # Coherence bars
        self.coherence_bars = self.ax_coherence.bar(
            ['Spatial', 'Temporal', 'Phase'],
            [0.5, 0.5, 0.5],
            color=['#4080FF', '#FF8040', '#80FF40']
        )
        self.ax_coherence.set_ylim(0, 1)
        self.ax_coherence.set_ylabel('Coherence')

        # Time series
        self.timeseries_line, = self.ax_timeseries.plot([], [], color='#40FF80', linewidth=1)
        self.ax_timeseries.set_xlim(0, 1000)
        self.ax_timeseries.set_ylim(0, 1000)

        # Stats text
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', transform=self.ax_stats.transAxes,
                                             fontsize=9, family='monospace',
                                             verticalalignment='top')

    def _add_controls(self):
        """Add interactive controls."""
        # Control positions
        left_margin = 0.1

        # Emission mode radio buttons
        ax_mode = plt.axes([left_margin, 0.15, 0.12, 0.12])
        self.radio_mode = RadioButtons(ax_mode,
                                       ['Poissonian', 'Coherent', 'Squeezed', 'Chaotic'],
                                       active=0)
        self.radio_mode.on_clicked(self._on_mode_change)

        # Metabolic rate slider
        ax_metabolic = plt.axes([left_margin + 0.18, 0.22, 0.15, 0.02])
        self.slider_metabolic = Slider(ax_metabolic, 'Metabolic', 0.1, 3.0, valinit=1.0)
        self.slider_metabolic.on_changed(self._on_metabolic_change)

        # Coupling strength slider
        ax_coupling = plt.axes([left_margin + 0.18, 0.18, 0.15, 0.02])
        self.slider_coupling = Slider(ax_coupling, 'Coupling', 0.0, 1.0,
                                      valinit=self.sim.coupling_strength)
        self.slider_coupling.on_changed(self._on_coupling_change)

        # Buttons
        ax_stress = plt.axes([left_margin + 0.38, 0.20, 0.08, 0.03])
        self.btn_stress = Button(ax_stress, 'Add Stress')
        self.btn_stress.on_clicked(self._on_stress_click)

        ax_sync = plt.axes([left_margin + 0.48, 0.20, 0.08, 0.03])
        self.btn_sync = Button(ax_sync, 'Synchronize')
        self.btn_sync.on_clicked(self._on_sync_click)

        ax_reset = plt.axes([left_margin + 0.58, 0.20, 0.06, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset_click)

        ax_pause = plt.axes([left_margin + 0.66, 0.20, 0.06, 0.03])
        self.btn_pause = Button(ax_pause, 'Start')
        self.btn_pause.on_clicked(self._on_pause_click)

        # Presets
        ax_preset = plt.axes([left_margin + 0.76, 0.15, 0.12, 0.10])
        self.radio_preset = RadioButtons(ax_preset,
                                         ['Healthy', 'Stressed', 'Coherent', 'Chaotic'],
                                         active=0)
        self.radio_preset.on_clicked(self._on_preset_change)

        # Flash toggle
        ax_flash = plt.axes([left_margin + 0.18, 0.14, 0.1, 0.03])
        self.check_flash = CheckButtons(ax_flash, ['Show Flashes'], [True])
        self.check_flash.on_clicked(self._on_flash_toggle)

    def _on_mode_change(self, label):
        """Handle emission mode change."""
        modes = {
            'Poissonian': EmissionMode.POISSONIAN,
            'Coherent': EmissionMode.COHERENT,
            'Squeezed': EmissionMode.SQUEEZED,
            'Chaotic': EmissionMode.CHAOTIC
        }
        self.sim.set_emission_mode(modes[label])

    def _on_metabolic_change(self, val):
        """Handle metabolic rate change."""
        self.sim.set_metabolic_rate(val)

    def _on_coupling_change(self, val):
        """Handle coupling strength change."""
        self.sim.set_coupling_strength(val)

    def _on_stress_click(self, event):
        """Add oxidative stress at random location."""
        center = (np.random.randint(10, self.sim.rows - 10),
                  np.random.randint(10, self.sim.cols - 10))
        self.sim.induce_oxidative_stress(center, radius=5, intensity=0.8)

    def _on_sync_click(self, event):
        """Synchronize emission phases."""
        self.sim.synchronize_phases(coherence=0.9)

    def _on_reset_click(self, event):
        """Reset simulation."""
        self.sim.reset()

    def _on_pause_click(self, event):
        """Toggle pause/run."""
        self.running = not self.running
        self.btn_pause.label.set_text('Pause' if self.running else 'Start')

    def _on_preset_change(self, label):
        """Apply preset configuration."""
        self.sim.reset()

        if label == 'Healthy':
            self.sim.set_emission_mode(EmissionMode.POISSONIAN)
            self.sim.set_metabolic_rate(1.0)
            self.slider_metabolic.set_val(1.0)

        elif label == 'Stressed':
            self.sim.set_emission_mode(EmissionMode.CHAOTIC)
            self.sim.set_metabolic_rate(1.5)
            self.slider_metabolic.set_val(1.5)
            # Add stress regions
            for _ in range(3):
                center = (np.random.randint(10, self.sim.rows - 10),
                          np.random.randint(10, self.sim.cols - 10))
                self.sim.induce_oxidative_stress(center, radius=8, intensity=0.7)

        elif label == 'Coherent':
            self.sim.set_emission_mode(EmissionMode.COHERENT)
            self.sim.set_coupling_strength(0.8)
            self.slider_coupling.set_val(0.8)
            self.sim.synchronize_phases(coherence=0.8)

        elif label == 'Chaotic':
            self.sim.set_emission_mode(EmissionMode.CHAOTIC)
            self.sim.set_metabolic_rate(2.0)
            self.slider_metabolic.set_val(2.0)
            self.sim.set_coupling_strength(0.0)
            self.slider_coupling.set_val(0.0)

    def _on_flash_toggle(self, label):
        """Toggle flash display."""
        self.show_flashes = not self.show_flashes

    def _update(self, frame):
        """Animation update function."""
        if self.running:
            self.sim.step(5)  # Multiple steps per frame

        # Update emission display
        self.emission_img.set_array(self.sim.emission_counts)
        max_count = max(np.max(self.sim.emission_counts), 1)
        self.emission_img.set_clim(0, max_count)

        # Update flash scatter
        if self.show_flashes and np.sum(self.sim.emission_counts) > 0:
            # Find cells with emission
            emitting = np.where(self.sim.emission_counts > 0)
            if len(emitting[0]) > 0:
                # Random subset for visual effect
                n_show = min(50, len(emitting[0]))
                idx = np.random.choice(len(emitting[0]), n_show, replace=False)
                self.flash_scatter.set_offsets(
                    np.column_stack([emitting[1][idx], emitting[0][idx]])
                )
            else:
                self.flash_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.flash_scatter.set_offsets(np.empty((0, 2)))

        # Update spectrum
        wavelengths, spectrum = self.sim.compute_spectrum()
        self.spectrum_line.set_data(wavelengths, spectrum)
        if np.max(spectrum) > 0:
            self.ax_spectrum.set_ylim(0, np.max(spectrum) * 1.2)

        # Update coherence bars
        spatial = self.sim.compute_spatial_coherence()
        temporal = self.sim.compute_temporal_coherence()
        phase = self.sim.compute_phase_coherence()
        for bar, height in zip(self.coherence_bars, [spatial, temporal, phase]):
            bar.set_height(height)

        # Update time series
        if len(self.sim.emission_history) > 0:
            times = np.arange(len(self.sim.emission_history)) * self.sim.dt
            self.timeseries_line.set_data(times, self.sim.emission_history)
            self.ax_timeseries.set_xlim(0, max(times[-1], 100))
            self.ax_timeseries.set_ylim(0, max(max(self.sim.emission_history), 100) * 1.1)

        # Update stats text
        output = self.sim.get_loomsense_output()
        er = self.sim.map_to_er_space()

        stats_str = (
            f"=== LoomSense Output ===\n"
            f"Time: {output['time_ms']:.0f} ms\n"
            f"\nEmission Metrics:\n"
            f"  Total Photons: {output['total_photon_count']}\n"
            f"  Rate/cell: {output['emission_rate_per_cell']:.1f}/s\n"
            f"  Fano Factor: {output['fano_factor']:.3f}\n"
            f"\nCoherence Metrics:\n"
            f"  Spatial: {output['spatial_coherence']:.3f}\n"
            f"  Temporal: {output['temporal_coherence']:.3f}\n"
            f"  Phase: {output['phase_coherence']:.3f}\n"
            f"\nMetabolic State:\n"
            f"  ROS: {output['mean_ros_level']:.3f}\n"
            f"  ATP: {output['mean_atp_level']:.3f}\n"
            f"  Stress Index: {output['metabolic_stress_index']:.3f}\n"
            f"\néR Phase Space:\n"
            f"  EP: {er['energy_present']:.2f}\n"
            f"  f: {er['frequency']:.2f} Hz\n"
            f"  éR: {er['energy_resistance']:.4f}"
        )
        self.stats_text.set_text(stats_str)

        return [self.emission_img, self.flash_scatter, self.spectrum_line,
                self.stats_text, *self.coherence_bars, self.timeseries_line]

    def run(self, interval: int = 50):
        """
        Start the interactive visualization.

        Parameters:
            interval: Animation interval in milliseconds
        """
        self.anim = animation.FuncAnimation(
            self.fig, self._update, interval=interval, blit=False,
            cache_frame_data=False
        )
        plt.show()


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def create_healthy_tissue(grid_size: Tuple[int, int] = (50, 50)) -> BiophotonSimulator:
    """
    Create simulator with healthy tissue parameters.

    Healthy tissue shows:
    - Baseline emission rates
    - Poissonian statistics
    - Moderate spatial coherence
    """
    sim = BiophotonSimulator(
        grid_size=grid_size,
        emission_mode=EmissionMode.POISSONIAN,
        coupling_strength=0.3
    )
    return sim


def create_stressed_tissue(grid_size: Tuple[int, int] = (50, 50)) -> BiophotonSimulator:
    """
    Create simulator with oxidatively stressed tissue.

    Stressed tissue shows:
    - Elevated emission rates
    - Super-Poissonian (chaotic) statistics
    - Reduced coherence
    """
    sim = BiophotonSimulator(
        grid_size=grid_size,
        emission_mode=EmissionMode.CHAOTIC,
        coupling_strength=0.1
    )

    # Add stress regions
    for _ in range(5):
        center = (np.random.randint(5, grid_size[0] - 5),
                  np.random.randint(5, grid_size[1] - 5))
        sim.induce_oxidative_stress(center, radius=8, intensity=0.8)

    sim.set_metabolic_rate(1.5)
    return sim


def create_coherent_emission(grid_size: Tuple[int, int] = (50, 50)) -> BiophotonSimulator:
    """
    Create simulator demonstrating coherent biophoton emission.

    Coherent emission shows:
    - Phase-locked emission across cells
    - Sub-Poissonian or coherent statistics
    - High spatial and temporal coherence
    """
    sim = BiophotonSimulator(
        grid_size=grid_size,
        emission_mode=EmissionMode.COHERENT,
        coupling_strength=0.8
    )

    # Synchronize phases
    sim.synchronize_phases(coherence=0.9)

    return sim


def create_meditation_state(grid_size: Tuple[int, int] = (50, 50)) -> BiophotonSimulator:
    """
    Create simulator modeling meditation-like coherent state.

    Based on studies showing increased biophoton coherence
    during meditation and focused attention states.
    """
    sim = BiophotonSimulator(
        grid_size=grid_size,
        emission_mode=EmissionMode.SQUEEZED,  # Sub-Poissonian
        coupling_strength=0.7,
        coherence_time=200.0  # Longer coherence time
    )

    # Lower metabolic rate (relaxed state)
    sim.set_metabolic_rate(0.8)

    # Partial phase synchronization
    sim.synchronize_phases(coherence=0.7)

    return sim


def create_inflammation_model(grid_size: Tuple[int, int] = (50, 50)) -> BiophotonSimulator:
    """
    Create simulator modeling inflamed tissue.

    Inflammation shows:
    - Localized high emission
    - Disrupted coherence
    - Elevated ROS
    """
    sim = BiophotonSimulator(
        grid_size=grid_size,
        emission_mode=EmissionMode.CHAOTIC,
        coupling_strength=0.05
    )

    # Create inflammation focus - radius scales with grid size
    # but stays localized (not reaching corners)
    center = (grid_size[0] // 2, grid_size[1] // 2)
    radius = min(grid_size[0], grid_size[1]) // 4  # 1/4 of smallest dimension
    sim.induce_oxidative_stress(center, radius=radius, intensity=1.0)

    return sim


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo():
    """Run the interactive biophoton visualizer with healthy tissue."""
    sim = create_healthy_tissue(grid_size=(50, 50))
    viz = BiophotonVisualizer(sim)
    viz.run()


def demo_stressed():
    """Demo with oxidatively stressed tissue."""
    sim = create_stressed_tissue(grid_size=(50, 50))
    viz = BiophotonVisualizer(sim)
    viz.run()


def demo_coherent():
    """Demo with coherent biophoton emission."""
    sim = create_coherent_emission(grid_size=(50, 50))
    viz = BiophotonVisualizer(sim)
    viz.run()


def demo_meditation():
    """Demo modeling meditation-like coherent state."""
    sim = create_meditation_state(grid_size=(50, 50))
    viz = BiophotonVisualizer(sim)
    viz.run()


def demo_comparison():
    """
    Non-interactive comparison of emission modes.

    Shows how different emission statistics differ.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Biophoton Emission Modes Comparison', fontsize=14, fontweight='bold')

    modes = [
        (EmissionMode.POISSONIAN, 'Poissonian (Random)'),
        (EmissionMode.COHERENT, 'Coherent (Phase-locked)'),
        (EmissionMode.SQUEEZED, 'Squeezed (Sub-Poissonian)'),
        (EmissionMode.CHAOTIC, 'Chaotic (Super-Poissonian)')
    ]

    # Create colormap
    colors = ['#000510', '#001030', '#003060', '#006040',
              '#40A060', '#80FF80', '#FFFF80', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('biophoton', colors)

    for i, (mode, name) in enumerate(modes):
        sim = BiophotonSimulator(grid_size=(40, 40), emission_mode=mode)

        if mode == EmissionMode.COHERENT:
            sim.synchronize_phases(0.8)
            sim.set_coupling_strength(0.8)

        # Run simulation
        sim.run(duration=500)

        # Show emission pattern
        ax = axes[0, i]
        im = ax.imshow(sim.emission_counts, cmap=cmap, aspect='equal')
        ax.set_title(name, fontsize=10)
        ax.axis('off')

        # Show statistics
        ax2 = axes[1, i]
        stats = sim.compute_emission_statistics()

        # Histogram of emission counts
        counts = sim.emission_counts.flatten()
        ax2.hist(counts, bins=20, color='#40FF80', edgecolor='white', alpha=0.7)
        ax2.set_xlabel('Photon Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Fano: {stats["fano_factor"]:.2f}', fontsize=9)

        # Add annotation
        ax2.text(0.95, 0.95, f'F={stats["fano_factor"]:.2f}\nF=1: Poisson',
                transform=ax2.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Biophoton Emission Simulator - CLT Phase 2.2")
    print("=" * 50)
    print("\nAvailable demos:")
    print("  1. demo()           - Interactive healthy tissue")
    print("  2. demo_stressed()  - Oxidatively stressed tissue")
    print("  3. demo_coherent()  - Coherent emission mode")
    print("  4. demo_meditation()- Meditation-like state")
    print("  5. demo_comparison()- Compare emission modes")
    print("\nStarting interactive demo...")
    demo()
