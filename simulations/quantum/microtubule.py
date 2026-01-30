"""
Microtubule Time Crystal Simulator for Cosmic Loom Theory

Models microtubules as fractal time crystals based on Hameroff-Penrose-Bandyopadhyay
research. Implements multi-scale oscillations and the characteristic "triplet of
triplets" resonance pattern.

Scientific Background:
Microtubules are cylindrical polymers of tubulin dimers arranged in 13 protofilaments.
Each tubulin contains 86 aromatic rings that can support quantum coherent oscillations.
Research by Bandyopadhyay's group has shown microtubules exhibit time crystal behavior
with resonance peaks appearing in triplet patterns at multiple frequency scales.

THREE COUPLED OSCILLATING SYSTEMS:
1. C-termini + ordered water layer → KILOHERTZ oscillations
2. Tubulin lattice (phonons, dipoles) → MEGAHERTZ oscillations
3. Internal water channel → GIGAHERTZ oscillations
4. Aromatic ring electrons → TERAHERTZ oscillations

From CLT perspective:
Microtubule coherence represents the FASTEST timescale of biological substrate
coherence. Synchronized microtubule networks may provide the quantum-classical
interface that links molecular dynamics to cellular-scale Loomfield patterns.

Key concepts:
- Tubulin dipole states and neighbor coupling
- Multi-scale oscillation hierarchy
- Triplet resonance (fractal time crystal signature)
- Floquet driving maintains time crystal behavior
- Decoherence from thermal noise
- Connection to biophoton and bioelectric layers
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import matplotlib.animation as animation
from scipy import signal
from scipy.fft import fft, fftfreq


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Microtubule geometry
N_PROTOFILAMENTS = 13          # Number of protofilaments in cylinder
TUBULIN_LENGTH_NM = 8.0        # Length of tubulin dimer (nm)
TUBULIN_WIDTH_NM = 4.0         # Width of tubulin dimer (nm)
MT_INNER_DIAMETER_NM = 12.0    # Inner water channel diameter (nm)
MT_OUTER_DIAMETER_NM = 25.0    # Outer diameter (nm)
AROMATIC_RINGS_PER_TUBULIN = 86  # Quantum-relevant regions

# Oscillation frequencies (Hz) - from Bandyopadhyay experiments
FREQ_CTERMINI = 1e3            # C-termini: kilohertz
FREQ_LATTICE = 1e6             # Lattice phonons: megahertz
FREQ_WATER_CHANNEL = 1e9       # Internal water: gigahertz
FREQ_AROMATIC = 1e12           # Aromatic rings: terahertz

# Triplet frequency ratios (fractal structure)
TRIPLET_RATIO_1 = 1.0          # Base frequency
TRIPLET_RATIO_2 = 1.618        # Golden ratio (φ)
TRIPLET_RATIO_3 = 2.618        # φ² (self-similar)

# Coupling strengths (normalized)
COUPLING_LATERAL = 0.3         # Between adjacent tubulins in same protofilament
COUPLING_LONGITUDINAL = 0.2    # Between protofilaments
COUPLING_SCALE = 0.1           # Between frequency scales

# Thermal parameters
TEMPERATURE_BODY = 310.0       # Body temperature (K)
KB = 1.38e-23                  # Boltzmann constant (J/K)
HBAR = 1.055e-34               # Reduced Planck constant (J·s)

# Decoherence timescales (seconds) - these are model estimates
DECOHERENCE_TIME_AROMATIC = 1e-13   # fs timescale for aromatic
DECOHERENCE_TIME_LATTICE = 1e-9     # ns timescale for lattice
DECOHERENCE_TIME_CTERMINI = 1e-6    # μs timescale for C-termini


# =============================================================================
# OSCILLATION SCALE ENUM
# =============================================================================

class OscillationScale(Enum):
    """Frequency scales in the microtubule time crystal."""
    CTERMINI = "ctermini"           # kHz - C-termini tails
    LATTICE = "lattice"             # MHz - Tubulin lattice phonons
    WATER_CHANNEL = "water_channel" # GHz - Internal water
    AROMATIC = "aromatic"           # THz - Aromatic ring electrons


class MicrotubuleState(Enum):
    """Overall state of microtubule coherence."""
    COHERENT = "coherent"           # Synchronized oscillations
    DECOHERENT = "decoherent"       # Thermal noise dominated
    FLOQUET = "floquet"             # Externally driven time crystal
    ANESTHETIZED = "anesthetized"   # Suppressed (anesthesia model)


# =============================================================================
# TUBULIN DIMER MODEL
# =============================================================================

@dataclass
class TubulinState:
    """
    State of a single tubulin dimer.

    Each tubulin has multiple oscillating subsystems at different
    frequency scales, plus a net dipole moment.
    """
    # Dipole state (simplified: +1 or -1, representing conformational states)
    dipole: float = 1.0

    # Phase of oscillation at each frequency scale (0 to 2π)
    phase_ctermini: float = 0.0
    phase_lattice: float = 0.0
    phase_water: float = 0.0
    phase_aromatic: float = 0.0

    # Amplitude at each scale (0 to 1, where 1 = fully coherent)
    amplitude_ctermini: float = 1.0
    amplitude_lattice: float = 1.0
    amplitude_water: float = 1.0
    amplitude_aromatic: float = 1.0

    # GTP/GDP state affects dynamics
    gtp_bound: bool = True


# =============================================================================
# MICROTUBULE SIMULATOR
# =============================================================================

class MicrotubuleSimulator:
    """
    Simulates microtubule as a fractal time crystal.

    Models the 13-protofilament cylindrical structure with tubulin
    dipole dynamics at multiple frequency scales. Implements the
    "triplet of triplets" resonance pattern observed experimentally.

    Parameters:
        n_tubulins: Number of tubulin dimers per protofilament
        temperature: Temperature in Kelvin
        coupling_lateral: Coupling within protofilament
        coupling_longitudinal: Coupling between protofilaments
        driving_amplitude: External Floquet driving strength
        driving_frequency: External driving frequency (Hz)
    """

    def __init__(
        self,
        n_tubulins: int = 100,
        temperature: float = TEMPERATURE_BODY,
        coupling_lateral: float = COUPLING_LATERAL,
        coupling_longitudinal: float = COUPLING_LONGITUDINAL,
        driving_amplitude: float = 0.0,
        driving_frequency: float = FREQ_LATTICE
    ):
        self.n_tubulins = n_tubulins
        self.n_protofilaments = N_PROTOFILAMENTS
        self.temperature = temperature
        self.coupling_lateral = coupling_lateral
        self.coupling_longitudinal = coupling_longitudinal
        self.driving_amplitude = driving_amplitude
        self.driving_frequency = driving_frequency

        # Time tracking
        self.time = 0.0
        self.dt = 1e-9  # 1 ns timestep (for MHz-scale dynamics)

        # Initialize tubulin lattice
        # Shape: (n_protofilaments, n_tubulins)
        self._init_lattice()

        # History for analysis
        self.dipole_history = []
        self.coherence_history = []
        self.spectrum_history = []

        # State
        self.state = MicrotubuleState.COHERENT

    def _init_lattice(self):
        """Initialize the tubulin lattice arrays."""
        shape = (self.n_protofilaments, self.n_tubulins)

        # Dipole states (+1 or -1)
        self.dipoles = np.ones(shape)

        # Phases at each frequency scale
        self.phase_ctermini = np.random.uniform(0, 2*np.pi, shape)
        self.phase_lattice = np.random.uniform(0, 2*np.pi, shape)
        self.phase_water = np.random.uniform(0, 2*np.pi, shape)
        self.phase_aromatic = np.random.uniform(0, 2*np.pi, shape)

        # Amplitudes (coherence level at each scale)
        self.amp_ctermini = np.ones(shape)
        self.amp_lattice = np.ones(shape)
        self.amp_water = np.ones(shape)
        self.amp_aromatic = np.ones(shape)

        # GTP state (affects stability)
        self.gtp_state = np.ones(shape, dtype=bool)

    def _compute_neighbor_coupling(self, phases: np.ndarray) -> np.ndarray:
        """
        Compute phase coupling from neighbors (Kuramoto-like).

        Each tubulin is coupled to:
        - 2 neighbors in same protofilament (lateral)
        - 2 neighbors in adjacent protofilaments (longitudinal)
        """
        coupling = np.zeros_like(phases)

        # Lateral coupling (within protofilament)
        coupling += self.coupling_lateral * np.sin(
            np.roll(phases, 1, axis=1) - phases
        )
        coupling += self.coupling_lateral * np.sin(
            np.roll(phases, -1, axis=1) - phases
        )

        # Longitudinal coupling (between protofilaments)
        # Note: MT has helical structure, so this is slightly offset
        coupling += self.coupling_longitudinal * np.sin(
            np.roll(phases, 1, axis=0) - phases
        )
        coupling += self.coupling_longitudinal * np.sin(
            np.roll(phases, -1, axis=0) - phases
        )

        return coupling

    def _apply_thermal_noise(self, scale: OscillationScale) -> np.ndarray:
        """
        Apply thermal decoherence based on temperature.

        Higher temperature = more noise = faster decoherence.
        Different scales have different decoherence times.
        """
        # Thermal energy
        thermal_energy = KB * self.temperature

        # Decoherence rate depends on scale
        if scale == OscillationScale.AROMATIC:
            decoherence_rate = 1.0 / DECOHERENCE_TIME_AROMATIC
        elif scale == OscillationScale.WATER_CHANNEL:
            decoherence_rate = 1.0 / (DECOHERENCE_TIME_AROMATIC * 1e4)
        elif scale == OscillationScale.LATTICE:
            decoherence_rate = 1.0 / DECOHERENCE_TIME_LATTICE
        else:  # CTERMINI
            decoherence_rate = 1.0 / DECOHERENCE_TIME_CTERMINI

        # Noise strength proportional to sqrt(T * decoherence_rate * dt)
        noise_strength = np.sqrt(thermal_energy * decoherence_rate * self.dt * 1e20)

        return noise_strength * np.random.randn(self.n_protofilaments, self.n_tubulins)

    def _apply_floquet_driving(self) -> np.ndarray:
        """
        Apply external periodic driving (Floquet).

        This represents ATP hydrolysis / mitochondrial energy input
        that maintains the time crystal behavior.
        """
        if self.driving_amplitude <= 0:
            return np.zeros((self.n_protofilaments, self.n_tubulins))

        # Uniform driving across all tubulins
        driving = self.driving_amplitude * np.sin(
            2 * np.pi * self.driving_frequency * self.time
        )

        return np.full((self.n_protofilaments, self.n_tubulins), driving)

    def _cross_scale_coupling(self):
        """
        Implement coupling between frequency scales.

        The triplet resonance emerges from how oscillations at
        different scales influence each other.
        """
        # Aromatic → Lattice coupling
        aromatic_mean_phase = np.mean(np.cos(self.phase_aromatic))
        lattice_kick = COUPLING_SCALE * aromatic_mean_phase
        self.phase_lattice += lattice_kick * self.dt * FREQ_LATTICE

        # Lattice → C-termini coupling
        lattice_mean_phase = np.mean(np.cos(self.phase_lattice))
        ctermini_kick = COUPLING_SCALE * lattice_mean_phase
        self.phase_ctermini += ctermini_kick * self.dt * FREQ_CTERMINI

        # Water channel couples to both
        water_kick = COUPLING_SCALE * 0.5 * (
            np.mean(np.cos(self.phase_lattice)) +
            np.mean(np.cos(self.phase_aromatic))
        )
        self.phase_water += water_kick * self.dt * FREQ_WATER_CHANNEL

    def step(self, n_steps: int = 1):
        """
        Advance simulation by n_steps.

        Updates all frequency scales with their respective dynamics.
        """
        for _ in range(n_steps):
            # Update each frequency scale

            # C-termini (kHz) - slowest, largest amplitude motion
            omega_ct = 2 * np.pi * FREQ_CTERMINI
            coupling_ct = self._compute_neighbor_coupling(self.phase_ctermini)
            noise_ct = self._apply_thermal_noise(OscillationScale.CTERMINI)
            self.phase_ctermini += self.dt * (
                omega_ct + coupling_ct + noise_ct * 0.1
            )

            # Lattice phonons (MHz)
            omega_lat = 2 * np.pi * FREQ_LATTICE
            coupling_lat = self._compute_neighbor_coupling(self.phase_lattice)
            noise_lat = self._apply_thermal_noise(OscillationScale.LATTICE)
            driving = self._apply_floquet_driving()
            self.phase_lattice += self.dt * (
                omega_lat + coupling_lat + noise_lat + driving
            )

            # Water channel (GHz) - fast internal dynamics
            omega_water = 2 * np.pi * FREQ_WATER_CHANNEL
            coupling_water = self._compute_neighbor_coupling(self.phase_water) * 0.5
            noise_water = self._apply_thermal_noise(OscillationScale.WATER_CHANNEL)
            self.phase_water += self.dt * (
                omega_water + coupling_water + noise_water
            )

            # Aromatic rings (THz) - fastest, quantum-scale
            omega_ar = 2 * np.pi * FREQ_AROMATIC
            coupling_ar = self._compute_neighbor_coupling(self.phase_aromatic) * 0.3
            noise_ar = self._apply_thermal_noise(OscillationScale.AROMATIC)
            self.phase_aromatic += self.dt * (
                omega_ar + coupling_ar + noise_ar
            )

            # Cross-scale coupling (triplet emergence)
            self._cross_scale_coupling()

            # Keep phases in [0, 2π]
            self.phase_ctermini %= (2 * np.pi)
            self.phase_lattice %= (2 * np.pi)
            self.phase_water %= (2 * np.pi)
            self.phase_aromatic %= (2 * np.pi)

            # Update dipoles based on lattice phase
            self.dipoles = np.sign(np.cos(self.phase_lattice))

            # Apply amplitude decay from decoherence
            if self.state != MicrotubuleState.COHERENT:
                self._apply_decoherence()

            # Update time
            self.time += self.dt

            # Record history
            self.dipole_history.append(np.mean(self.dipoles))
            self.coherence_history.append(self.compute_coherence())

    def _apply_decoherence(self):
        """Apply decoherence effects based on state."""
        if self.state == MicrotubuleState.DECOHERENT:
            # Gradual amplitude decay
            decay = 0.999
            self.amp_ctermini *= decay
            self.amp_lattice *= decay
            self.amp_water *= decay
            self.amp_aromatic *= decay

        elif self.state == MicrotubuleState.ANESTHETIZED:
            # Suppress aromatic oscillations specifically
            # (anesthetics bind to aromatic regions)
            self.amp_aromatic *= 0.95
            # Also reduces cross-scale coupling
            self.phase_aromatic += np.random.uniform(
                -0.5, 0.5, self.phase_aromatic.shape
            )

    def run(self, duration: float, record_interval: float = None):
        """
        Run simulation for specified duration.

        Parameters:
            duration: Total time in seconds
            record_interval: Not used (records every step)
        """
        n_steps = int(duration / self.dt)
        self.step(n_steps)

    # =========================================================================
    # COHERENCE METRICS
    # =========================================================================

    def compute_coherence(self, scale: OscillationScale = None) -> float:
        """
        Compute coherence (Kuramoto order parameter) at specified scale.

        If scale is None, returns mean across all scales.

        Returns:
            Order parameter R ∈ [0, 1] where 1 = fully synchronized
        """
        def order_param(phases):
            complex_phases = np.exp(1j * phases)
            return np.abs(np.mean(complex_phases))

        if scale == OscillationScale.CTERMINI:
            return order_param(self.phase_ctermini)
        elif scale == OscillationScale.LATTICE:
            return order_param(self.phase_lattice)
        elif scale == OscillationScale.WATER_CHANNEL:
            return order_param(self.phase_water)
        elif scale == OscillationScale.AROMATIC:
            return order_param(self.phase_aromatic)
        else:
            # Mean across all scales
            coherences = [
                order_param(self.phase_ctermini),
                order_param(self.phase_lattice),
                order_param(self.phase_water),
                order_param(self.phase_aromatic)
            ]
            return np.mean(coherences)

    def compute_all_coherences(self) -> Dict[str, float]:
        """Compute coherence at all scales."""
        return {
            'ctermini': self.compute_coherence(OscillationScale.CTERMINI),
            'lattice': self.compute_coherence(OscillationScale.LATTICE),
            'water_channel': self.compute_coherence(OscillationScale.WATER_CHANNEL),
            'aromatic': self.compute_coherence(OscillationScale.AROMATIC),
            'mean': self.compute_coherence()
        }

    def compute_dipole_correlation(self) -> float:
        """
        Compute spatial correlation of dipole states.

        High correlation = organized pattern.
        """
        # Correlation with neighbors
        corr_lateral = np.mean(
            self.dipoles * np.roll(self.dipoles, 1, axis=1)
        )
        corr_long = np.mean(
            self.dipoles * np.roll(self.dipoles, 1, axis=0)
        )
        return 0.5 * (corr_lateral + corr_long)

    # =========================================================================
    # FREQUENCY SPECTRUM AND TRIPLET ANALYSIS
    # =========================================================================

    def compute_spectrum(self, scale: OscillationScale = OscillationScale.LATTICE,
                        n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency spectrum of oscillations.

        Parameters:
            scale: Which oscillation scale to analyze
            n_samples: Number of time samples (from history)

        Returns:
            (frequencies, power_spectrum) arrays
        """
        # Get phase history for this scale
        if len(self.dipole_history) < n_samples:
            n_samples = len(self.dipole_history)

        if n_samples < 10:
            return np.array([0]), np.array([0])

        # Use dipole mean as proxy for collective oscillation
        signal_data = np.array(self.dipole_history[-n_samples:])

        # FFT
        spectrum = np.abs(fft(signal_data))[:n_samples // 2]
        freqs = fftfreq(n_samples, self.dt)[:n_samples // 2]

        # Normalize
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        return freqs, spectrum

    def compute_triplet_resonance(self) -> Dict[str, float]:
        """
        Analyze triplet resonance pattern.

        The "triplet of triplets" signature is peaks at frequencies
        related by golden ratio φ.

        Returns:
            Dictionary with triplet analysis
        """
        freqs, spectrum = self.compute_spectrum(n_samples=2000)

        if len(freqs) < 10:
            return {
                'triplet_ratio_1': 0.0,
                'triplet_ratio_2': 0.0,
                'triplet_ratio_3': 0.0,
                'triplet_strength': 0.0
            }

        # Find peaks
        peaks, properties = signal.find_peaks(spectrum, height=0.1, distance=5)

        if len(peaks) < 3:
            return {
                'triplet_ratio_1': 1.0 if len(peaks) > 0 else 0.0,
                'triplet_ratio_2': 0.0,
                'triplet_ratio_3': 0.0,
                'triplet_strength': 0.0
            }

        # Get top 3 peaks
        peak_heights = spectrum[peaks]
        top_3_idx = np.argsort(peak_heights)[-3:]
        top_3_freqs = freqs[peaks[top_3_idx]]
        top_3_freqs = np.sort(top_3_freqs)

        # Compute ratios
        if top_3_freqs[0] > 0:
            ratio_2 = top_3_freqs[1] / top_3_freqs[0]
            ratio_3 = top_3_freqs[2] / top_3_freqs[0]
        else:
            ratio_2 = 0.0
            ratio_3 = 0.0

        # How close to golden ratio triplet?
        triplet_error = (
            abs(ratio_2 - TRIPLET_RATIO_2) +
            abs(ratio_3 - TRIPLET_RATIO_3)
        ) / 2.0

        triplet_strength = max(0, 1.0 - triplet_error)

        return {
            'triplet_ratio_1': 1.0,
            'triplet_ratio_2': ratio_2,
            'triplet_ratio_3': ratio_3,
            'triplet_strength': triplet_strength,
            'peak_frequencies': top_3_freqs.tolist()
        }

    # =========================================================================
    # STATE MANIPULATION
    # =========================================================================

    def set_state(self, state: MicrotubuleState):
        """Set the microtubule state."""
        self.state = state

        if state == MicrotubuleState.COHERENT:
            # Restore full amplitudes
            self.amp_ctermini.fill(1.0)
            self.amp_lattice.fill(1.0)
            self.amp_water.fill(1.0)
            self.amp_aromatic.fill(1.0)

        elif state == MicrotubuleState.FLOQUET:
            # Enable driving
            if self.driving_amplitude <= 0:
                self.driving_amplitude = 0.5

    def set_temperature(self, temperature: float):
        """Set temperature (affects decoherence)."""
        self.temperature = temperature

    def set_floquet_driving(self, amplitude: float, frequency: float = None):
        """
        Set external Floquet driving parameters.

        Parameters:
            amplitude: Driving strength (0 = off)
            frequency: Driving frequency in Hz
        """
        self.driving_amplitude = amplitude
        if frequency is not None:
            self.driving_frequency = frequency

    def synchronize_phases(self, scale: OscillationScale = None, coherence: float = 1.0):
        """
        Artificially synchronize phases at specified scale.

        Parameters:
            scale: Which scale to synchronize (None = all)
            coherence: 0 = random, 1 = fully synchronized
        """
        def sync(phases, coherence):
            mean_phase = np.mean(phases)
            return (1 - coherence) * phases + coherence * mean_phase

        if scale is None or scale == OscillationScale.CTERMINI:
            self.phase_ctermini = sync(self.phase_ctermini, coherence)
        if scale is None or scale == OscillationScale.LATTICE:
            self.phase_lattice = sync(self.phase_lattice, coherence)
        if scale is None or scale == OscillationScale.WATER_CHANNEL:
            self.phase_water = sync(self.phase_water, coherence)
        if scale is None or scale == OscillationScale.AROMATIC:
            self.phase_aromatic = sync(self.phase_aromatic, coherence)

    def randomize_phases(self, scale: OscillationScale = None):
        """Randomize phases (induce decoherence)."""
        shape = (self.n_protofilaments, self.n_tubulins)

        if scale is None or scale == OscillationScale.CTERMINI:
            self.phase_ctermini = np.random.uniform(0, 2*np.pi, shape)
        if scale is None or scale == OscillationScale.LATTICE:
            self.phase_lattice = np.random.uniform(0, 2*np.pi, shape)
        if scale is None or scale == OscillationScale.WATER_CHANNEL:
            self.phase_water = np.random.uniform(0, 2*np.pi, shape)
        if scale is None or scale == OscillationScale.AROMATIC:
            self.phase_aromatic = np.random.uniform(0, 2*np.pi, shape)

    def apply_anesthesia(self, concentration: float = 1.0):
        """
        Model anesthetic effect on aromatic oscillations.

        Anesthetics (like propofol) bind to aromatic regions,
        suppressing their oscillations.

        Parameters:
            concentration: 0 = none, 1 = full anesthesia
        """
        self.state = MicrotubuleState.ANESTHETIZED

        # Suppress aromatic amplitude
        self.amp_aromatic *= (1.0 - 0.9 * concentration)

        # Add noise to aromatic phases
        noise = concentration * np.random.uniform(
            -np.pi, np.pi, self.phase_aromatic.shape
        )
        self.phase_aromatic += noise
        self.phase_aromatic %= (2 * np.pi)

    def reset(self):
        """Reset to initial state."""
        self._init_lattice()
        self.time = 0.0
        self.dipole_history = []
        self.coherence_history = []
        self.spectrum_history = []
        self.state = MicrotubuleState.COHERENT

    # =========================================================================
    # CLT / ÉR PHASE SPACE MAPPING
    # =========================================================================

    def map_to_er_space(self) -> Dict:
        """
        Map microtubule state to éR phase space.

        In CLT, microtubule coherence represents the fastest
        biological timescale contributing to the Loomfield.

        Returns:
            Dictionary with éR mapping
        """
        # Get coherences at each scale
        coherences = self.compute_all_coherences()

        # Energy present: proportional to coherent oscillation amplitude
        # Higher coherence = more organized energy
        ep = (
            coherences['aromatic'] * 0.4 +  # THz contributes most
            coherences['lattice'] * 0.3 +
            coherences['water_channel'] * 0.2 +
            coherences['ctermini'] * 0.1
        )

        # Frequency: effective collective frequency
        # Weighted by coherence at each scale
        freq = (
            coherences['aromatic'] * FREQ_AROMATIC +
            coherences['lattice'] * FREQ_LATTICE +
            coherences['water_channel'] * FREQ_WATER_CHANNEL +
            coherences['ctermini'] * FREQ_CTERMINI
        ) / (sum(coherences.values()) + 1e-10)

        # Normalize frequency to reasonable range for éR
        freq_normalized = freq / FREQ_AROMATIC  # 0-1 range
        freq_effective = 0.1 + 9.9 * freq_normalized  # 0.1-10 Hz effective

        # Energy resistance
        er = ep / (freq_effective ** 2 + 0.01)

        # Triplet analysis
        triplet = self.compute_triplet_resonance()

        return {
            'energy_present': ep,
            'frequency': freq_effective,
            'energy_resistance': er,
            'mean_coherence': coherences['mean'],
            'aromatic_coherence': coherences['aromatic'],
            'lattice_coherence': coherences['lattice'],
            'water_coherence': coherences['water_channel'],
            'ctermini_coherence': coherences['ctermini'],
            'triplet_strength': triplet['triplet_strength'],
            'dipole_correlation': self.compute_dipole_correlation(),
            'temperature': self.temperature,
            'state': self.state.value
        }


# =============================================================================
# VISUALIZER
# =============================================================================

class MicrotubuleVisualizer:
    """
    Interactive visualization for microtubule time crystal simulation.

    Features:
    - Cylindrical microtubule display with dipole states
    - Multi-scale coherence display
    - Frequency spectrum with triplet peaks
    - éR phase space mapping
    - Interactive controls
    """

    def __init__(self, simulator: MicrotubuleSimulator):
        self.sim = simulator
        self.running = False

        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Microtubule Time Crystal Simulator - CLT Phase 2.3',
                         fontsize=14, fontweight='bold')

        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[2, 1, 1],
                     hspace=0.3, wspace=0.3)

        # Microtubule lattice display (unrolled cylinder)
        self.ax_mt = self.fig.add_subplot(gs[0, :2])
        self.ax_mt.set_title('Microtubule Dipole States (Unrolled Cylinder)')
        self.ax_mt.set_xlabel('Position along MT (tubulin index)')
        self.ax_mt.set_ylabel('Protofilament')

        # Frequency spectrum
        self.ax_spectrum = self.fig.add_subplot(gs[0, 2])
        self.ax_spectrum.set_title('Frequency Spectrum')
        self.ax_spectrum.set_xlabel('Frequency')
        self.ax_spectrum.set_ylabel('Power')

        # Coherence bars
        self.ax_coherence = self.fig.add_subplot(gs[0, 3])
        self.ax_coherence.set_title('Multi-Scale Coherence')

        # Time series
        self.ax_timeseries = self.fig.add_subplot(gs[1, :2])
        self.ax_timeseries.set_title('Coherence History')
        self.ax_timeseries.set_xlabel('Time (ns)')
        self.ax_timeseries.set_ylabel('Coherence')

        # Info display
        self.ax_info = self.fig.add_subplot(gs[1, 2:])
        self.ax_info.set_title('éR Phase Space & State')
        self.ax_info.axis('off')

        # Controls
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')

        # Colormaps
        self.cmap_dipole = LinearSegmentedColormap.from_list(
            'dipole', ['#0066CC', '#FFFFFF', '#CC6600']
        )

        # Initialize plots
        self._init_plots()
        self._add_controls()

        # Animation
        self.anim = None

    def _init_plots(self):
        """Initialize all plot elements."""
        # Microtubule dipole display
        self.mt_img = self.ax_mt.imshow(
            self.sim.dipoles,
            cmap=self.cmap_dipole,
            aspect='auto',
            vmin=-1, vmax=1
        )
        self.fig.colorbar(self.mt_img, ax=self.ax_mt, label='Dipole State')

        # Spectrum
        self.spectrum_line, = self.ax_spectrum.plot([], [], color='#00FF88', linewidth=1.5)
        self.ax_spectrum.set_xlim(0, 1e8)
        self.ax_spectrum.set_ylim(0, 1)

        # Coherence bars
        scales = ['C-term\n(kHz)', 'Lattice\n(MHz)', 'Water\n(GHz)', 'Aromatic\n(THz)']
        colors = ['#FF6666', '#66FF66', '#6666FF', '#FFFF66']
        self.coherence_bars = self.ax_coherence.bar(
            scales, [0.5, 0.5, 0.5, 0.5], color=colors
        )
        self.ax_coherence.set_ylim(0, 1)
        self.ax_coherence.set_ylabel('Order Parameter')

        # Time series
        self.coherence_line, = self.ax_timeseries.plot([], [], color='#00FF88', linewidth=1)
        self.ax_timeseries.set_xlim(0, 1000)
        self.ax_timeseries.set_ylim(0, 1)

        # Info text
        self.info_text = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            fontsize=9, family='monospace', verticalalignment='top'
        )

    def _add_controls(self):
        """Add interactive controls."""
        left = 0.1

        # State selector
        ax_state = plt.axes([left, 0.12, 0.12, 0.12])
        self.radio_state = RadioButtons(
            ax_state, ['Coherent', 'Decoherent', 'Floquet', 'Anesthesia'],
            active=0
        )
        self.radio_state.on_clicked(self._on_state_change)

        # Temperature slider
        ax_temp = plt.axes([left + 0.18, 0.22, 0.15, 0.02])
        self.slider_temp = Slider(ax_temp, 'Temp (K)', 270, 350, valinit=310)
        self.slider_temp.on_changed(self._on_temp_change)

        # Driving amplitude slider
        ax_drive = plt.axes([left + 0.18, 0.18, 0.15, 0.02])
        self.slider_drive = Slider(ax_drive, 'Driving', 0.0, 1.0, valinit=0.0)
        self.slider_drive.on_changed(self._on_drive_change)

        # Coupling slider
        ax_coupling = plt.axes([left + 0.18, 0.14, 0.15, 0.02])
        self.slider_coupling = Slider(ax_coupling, 'Coupling', 0.0, 1.0, valinit=0.3)
        self.slider_coupling.on_changed(self._on_coupling_change)

        # Buttons
        ax_sync = plt.axes([left + 0.38, 0.20, 0.08, 0.03])
        self.btn_sync = Button(ax_sync, 'Synchronize')
        self.btn_sync.on_clicked(self._on_sync_click)

        ax_random = plt.axes([left + 0.48, 0.20, 0.08, 0.03])
        self.btn_random = Button(ax_random, 'Randomize')
        self.btn_random.on_clicked(self._on_random_click)

        ax_reset = plt.axes([left + 0.58, 0.20, 0.06, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset_click)

        ax_pause = plt.axes([left + 0.66, 0.20, 0.06, 0.03])
        self.btn_pause = Button(ax_pause, 'Start')
        self.btn_pause.on_clicked(self._on_pause_click)

        # Presets
        ax_preset = plt.axes([left + 0.76, 0.12, 0.12, 0.12])
        self.radio_preset = RadioButtons(
            ax_preset, ['Time Crystal', 'Thermal', 'Driven', 'Anesthetized'],
            active=0
        )
        self.radio_preset.on_clicked(self._on_preset_change)

    def _on_state_change(self, label):
        """Handle state change."""
        states = {
            'Coherent': MicrotubuleState.COHERENT,
            'Decoherent': MicrotubuleState.DECOHERENT,
            'Floquet': MicrotubuleState.FLOQUET,
            'Anesthesia': MicrotubuleState.ANESTHETIZED
        }
        self.sim.set_state(states[label])

    def _on_temp_change(self, val):
        """Handle temperature change."""
        self.sim.set_temperature(val)

    def _on_drive_change(self, val):
        """Handle driving amplitude change."""
        self.sim.set_floquet_driving(val)

    def _on_coupling_change(self, val):
        """Handle coupling change."""
        self.sim.coupling_lateral = val
        self.sim.coupling_longitudinal = val * 0.7

    def _on_sync_click(self, event):
        """Synchronize phases."""
        self.sim.synchronize_phases(coherence=0.9)

    def _on_random_click(self, event):
        """Randomize phases."""
        self.sim.randomize_phases()

    def _on_reset_click(self, event):
        """Reset simulation."""
        self.sim.reset()

    def _on_pause_click(self, event):
        """Toggle pause/run."""
        self.running = not self.running
        self.btn_pause.label.set_text('Pause' if self.running else 'Start')

    def _on_preset_change(self, label):
        """Apply preset."""
        self.sim.reset()

        if label == 'Time Crystal':
            self.sim.set_state(MicrotubuleState.COHERENT)
            self.sim.synchronize_phases(coherence=0.8)
            self.slider_temp.set_val(310)
            self.slider_drive.set_val(0.0)

        elif label == 'Thermal':
            self.sim.set_state(MicrotubuleState.DECOHERENT)
            self.sim.randomize_phases()
            self.slider_temp.set_val(340)
            self.slider_drive.set_val(0.0)

        elif label == 'Driven':
            self.sim.set_state(MicrotubuleState.FLOQUET)
            self.sim.set_floquet_driving(0.7, FREQ_LATTICE)
            self.slider_drive.set_val(0.7)

        elif label == 'Anesthetized':
            self.sim.apply_anesthesia(0.8)
            self.slider_temp.set_val(310)

    def _update(self, frame):
        """Animation update."""
        if self.running:
            self.sim.step(50)  # Multiple steps per frame

        # Update microtubule display
        self.mt_img.set_array(self.sim.dipoles)

        # Update spectrum
        freqs, spectrum = self.sim.compute_spectrum()
        self.spectrum_line.set_data(freqs, spectrum)
        if len(freqs) > 0:
            self.ax_spectrum.set_xlim(0, freqs[-1] if freqs[-1] > 0 else 1e8)

        # Update coherence bars
        coherences = self.sim.compute_all_coherences()
        for bar, key in zip(self.coherence_bars,
                          ['ctermini', 'lattice', 'water_channel', 'aromatic']):
            bar.set_height(coherences[key])

        # Update time series
        if len(self.sim.coherence_history) > 0:
            times = np.arange(len(self.sim.coherence_history)) * self.sim.dt * 1e9
            self.coherence_line.set_data(times, self.sim.coherence_history)
            self.ax_timeseries.set_xlim(0, max(times[-1], 100))

        # Update info text
        er = self.sim.map_to_er_space()
        triplet = self.sim.compute_triplet_resonance()

        info_str = (
            f"=== Microtubule Time Crystal ===\n"
            f"Time: {self.sim.time*1e9:.1f} ns\n"
            f"State: {self.sim.state.value}\n"
            f"\nCoherence by Scale:\n"
            f"  C-termini (kHz): {coherences['ctermini']:.3f}\n"
            f"  Lattice (MHz):   {coherences['lattice']:.3f}\n"
            f"  Water (GHz):     {coherences['water_channel']:.3f}\n"
            f"  Aromatic (THz):  {coherences['aromatic']:.3f}\n"
            f"  Mean:            {coherences['mean']:.3f}\n"
            f"\nTriplet Resonance:\n"
            f"  Strength: {triplet['triplet_strength']:.3f}\n"
            f"  Ratio 2/1: {triplet['triplet_ratio_2']:.3f} (φ={TRIPLET_RATIO_2:.3f})\n"
            f"  Ratio 3/1: {triplet['triplet_ratio_3']:.3f} (φ²={TRIPLET_RATIO_3:.3f})\n"
            f"\néR Phase Space:\n"
            f"  EP: {er['energy_present']:.3f}\n"
            f"  f: {er['frequency']:.2f} Hz\n"
            f"  éR: {er['energy_resistance']:.4f}\n"
            f"\nDipole Correlation: {er['dipole_correlation']:.3f}"
        )
        self.info_text.set_text(info_str)

        return [self.mt_img, self.spectrum_line, self.coherence_line,
                self.info_text, *self.coherence_bars]

    def run(self, interval: int = 50, save_path: Optional[str] = None):
        """
        Start the interactive visualization.

        Parameters:
            interval: Animation interval in milliseconds
            save_path: If provided, save a snapshot instead of showing interactively
        """
        if save_path:
            self.running = True
            self._update(0)
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight',
                            facecolor=self.fig.get_facecolor())
            plt.close(self.fig)
        else:
            self.anim = animation.FuncAnimation(
                self.fig, self._update, interval=interval, blit=False,
                cache_frame_data=False
            )
            plt.show()

    @classmethod
    def create_static_figure(cls, save_path: Optional[str] = None):
        """
        Generate a publication-quality 2x2 grid comparing microtubule states.

        Panels: Coherent, Thermal, Floquet-Driven, Anesthetized.
        Each panel shows the dipole heatmap with coherence and éR metrics.

        Parameters:
            save_path: If provided, save figure to this path at 300 DPI.
                       Otherwise display interactively.

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        fig.suptitle('Microtubule Time Crystal States', fontsize=14, fontweight='bold')

        configs = [
            ('Coherent (Body Temp)', create_coherent_mt),
            ('Thermal Noise', create_thermal_mt),
            ('Floquet Driven', create_floquet_driven_mt),
            ('Anesthetized', create_anesthetized_mt),
        ]

        cmap = LinearSegmentedColormap.from_list(
            'dipole', ['#0066CC', '#FFFFFF', '#CC6600']
        )

        for idx, (name, create_fn) in enumerate(configs):
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            sim = create_fn(n_tubulins=60)
            sim.run(duration=1e-7)  # 100 ns

            # Dipole heatmap
            im = ax.imshow(sim.dipoles, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Dipole')

            # Compute metrics
            coherences = sim.compute_all_coherences()
            er = sim.map_to_er_space()

            metrics_str = (
                f"Mean coh: {coherences['mean']:.3f}\n"
                f"Aromatic: {coherences['aromatic']:.3f}\n"
                f"Lattice:  {coherences['lattice']:.3f}\n"
                f"éR: {er['energy_resistance']:.4f}\n"
                f"Dipole corr: {er['dipole_correlation']:.3f}"
            )
            ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.8, edgecolor='gray'),
                    color='black')

            ax.set_title(name, fontsize=11)
            ax.set_xlabel('Position along MT')
            ax.set_ylabel('Protofilament')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        return fig


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def create_coherent_mt(n_tubulins: int = 100) -> MicrotubuleSimulator:
    """
    Create microtubule with coherent oscillations.

    Synchronized phases, clear triplet resonance.
    """
    sim = MicrotubuleSimulator(
        n_tubulins=n_tubulins,
        temperature=TEMPERATURE_BODY,
        coupling_lateral=0.4,
        coupling_longitudinal=0.3
    )
    sim.synchronize_phases(coherence=0.8)
    return sim


def create_thermal_mt(n_tubulins: int = 100) -> MicrotubuleSimulator:
    """
    Create microtubule dominated by thermal noise.

    Random phases, no clear resonance structure.
    """
    sim = MicrotubuleSimulator(
        n_tubulins=n_tubulins,
        temperature=340.0,  # Elevated temperature
        coupling_lateral=0.2,
        coupling_longitudinal=0.1
    )
    sim.set_state(MicrotubuleState.DECOHERENT)
    sim.randomize_phases()
    return sim


def create_floquet_driven_mt(n_tubulins: int = 100) -> MicrotubuleSimulator:
    """
    Create externally driven time crystal.

    Floquet driving maintains coherent oscillation against decoherence.
    This models the effect of ATP/mitochondrial energy input.
    """
    sim = MicrotubuleSimulator(
        n_tubulins=n_tubulins,
        temperature=TEMPERATURE_BODY,
        driving_amplitude=0.6,
        driving_frequency=FREQ_LATTICE
    )
    sim.set_state(MicrotubuleState.FLOQUET)
    return sim


def create_anesthetized_mt(n_tubulins: int = 100,
                          concentration: float = 0.8) -> MicrotubuleSimulator:
    """
    Create anesthetized microtubule.

    Models effect of anesthetics binding to aromatic regions,
    suppressing THz oscillations and cross-scale coupling.
    """
    sim = MicrotubuleSimulator(
        n_tubulins=n_tubulins,
        temperature=TEMPERATURE_BODY
    )
    sim.apply_anesthesia(concentration)
    return sim


def create_cold_mt(n_tubulins: int = 100) -> MicrotubuleSimulator:
    """
    Create cold microtubule with extended coherence.

    Lower temperature = less thermal decoherence.
    """
    sim = MicrotubuleSimulator(
        n_tubulins=n_tubulins,
        temperature=280.0,  # Cold
        coupling_lateral=0.4,
        coupling_longitudinal=0.3
    )
    sim.synchronize_phases(coherence=0.9)
    return sim


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo():
    """Run the interactive microtubule visualizer."""
    sim = create_coherent_mt(n_tubulins=80)
    viz = MicrotubuleVisualizer(sim)
    viz.run()


def demo_thermal():
    """Demo with thermal decoherence."""
    sim = create_thermal_mt(n_tubulins=80)
    viz = MicrotubuleVisualizer(sim)
    viz.run()


def demo_floquet():
    """Demo with Floquet driving."""
    sim = create_floquet_driven_mt(n_tubulins=80)
    viz = MicrotubuleVisualizer(sim)
    viz.run()


def demo_anesthesia():
    """Demo with anesthesia effect."""
    sim = create_anesthetized_mt(n_tubulins=80)
    viz = MicrotubuleVisualizer(sim)
    viz.run()


def demo_comparison():
    """
    Static comparison of different microtubule states.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Microtubule States Comparison', fontsize=14, fontweight='bold')

    configs = [
        ('Coherent (Body Temp)', create_coherent_mt),
        ('Thermal Noise', create_thermal_mt),
        ('Floquet Driven', create_floquet_driven_mt),
        ('Anesthetized', create_anesthetized_mt)
    ]

    cmap = LinearSegmentedColormap.from_list(
        'dipole', ['#0066CC', '#FFFFFF', '#CC6600']
    )

    for i, (name, create_fn) in enumerate(configs):
        sim = create_fn(n_tubulins=60)

        # Run briefly
        sim.run(duration=1e-7)  # 100 ns

        # Show dipole pattern
        ax = axes[0, i]
        ax.imshow(sim.dipoles, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Position')
        ax.set_ylabel('Protofilament')

        # Show coherence
        ax2 = axes[1, i]
        coherences = sim.compute_all_coherences()
        scales = ['CT', 'Lat', 'H2O', 'Ar']
        values = [coherences['ctermini'], coherences['lattice'],
                 coherences['water_channel'], coherences['aromatic']]
        colors = ['#FF6666', '#66FF66', '#6666FF', '#FFFF66']
        ax2.bar(scales, values, color=colors)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Coherence')
        ax2.set_title(f'Mean: {coherences["mean"]:.2f}', fontsize=9)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Microtubule Time Crystal Simulator - CLT Phase 2.3")
    print("=" * 55)
    print("\nBased on Hameroff-Penrose-Bandyopadhyay research")
    print("\nAvailable demos:")
    print("  1. demo()           - Interactive coherent MT")
    print("  2. demo_thermal()   - Thermal noise dominated")
    print("  3. demo_floquet()   - Externally driven time crystal")
    print("  4. demo_anesthesia()- Anesthetic effect")
    print("  5. demo_comparison()- Static comparison")
    print("\nStarting interactive demo...")
    demo()
