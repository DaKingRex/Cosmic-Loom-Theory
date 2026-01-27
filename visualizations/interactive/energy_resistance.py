"""
Energy Resistance (éR) Phase Space Visualizer

A core teaching tool for Cosmic Loom Theory that visualizes the relationship
between Energy Present (EP), frequency (f), and Energy Resistance (éR).

The fundamental equation: éR = EP / f²

The Energy Resistance Principle states that living systems operate within a
specific energetic regime - a "viable window" between chaos and rigidity.

- Too low éR: System collapses into chaos/decoherence
- Too high éR: System becomes rigid, unable to respond adaptively
- Viable window: The "Goldilocks zone" where biological coherence thrives

Enhanced Features (v1.2):
- Biological Parameter Mapping: Maps abstract éR to physiological measurements
- Pathology Signatures: Shows where mental health conditions appear in éR space
- Clinical Trajectories: Decompensation and recovery paths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Polygon, Ellipse, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from typing import Tuple, List, Optional, Dict


# =============================================================================
# BIOLOGICAL PARAMETER MAPPING
# =============================================================================

# Biological state reference points (EP, frequency, label, color)
# These map physiological states to the abstract éR phase space
BIOLOGICAL_STATES = {
    'resting_awake': {
        'ep': 2.5, 'freq': 1.2,
        'label': 'Resting Awake',
        'color': '#00ff88',
        'description': 'Alert but relaxed, EEG alpha (8-12 Hz)',
        'metabolic_rate': 70,  # kcal/hr (basal)
        'hrv_ms': 50,  # RMSSD in milliseconds
    },
    'deep_sleep': {
        'ep': 1.8, 'freq': 0.6,
        'label': 'Deep Sleep (N3)',
        'color': '#4299e1',
        'description': 'Delta waves (0.5-4 Hz), low metabolism',
        'metabolic_rate': 55,
        'hrv_ms': 80,
    },
    'rem_sleep': {
        'ep': 2.2, 'freq': 1.8,
        'label': 'REM Sleep',
        'color': '#9f7aea',
        'description': 'Dream state, mixed frequencies, paralysis',
        'metabolic_rate': 65,
        'hrv_ms': 40,
    },
    'focused_attention': {
        'ep': 3.5, 'freq': 2.0,
        'label': 'Focused Attention',
        'color': '#f6ad55',
        'description': 'Beta waves (12-30 Hz), active cognition',
        'metabolic_rate': 85,
        'hrv_ms': 35,
    },
    'exercise': {
        'ep': 7.0, 'freq': 3.2,
        'label': 'Exercise',
        'color': '#fc8181',
        'description': 'High metabolism, elevated HR, sympathetic',
        'metabolic_rate': 400,
        'hrv_ms': 20,
    },
    'meditation': {
        'ep': 2.0, 'freq': 0.8,
        'label': 'Deep Meditation',
        'color': '#68d391',
        'description': 'Theta/alpha, coherent, parasympathetic',
        'metabolic_rate': 60,
        'hrv_ms': 90,
    },
    'flow_state': {
        'ep': 4.0, 'freq': 1.5,
        'label': 'Flow State',
        'color': '#ffd700',
        'description': 'Optimal performance, effortless focus',
        'metabolic_rate': 100,
        'hrv_ms': 55,
    },
}

# Pathology zones in éR space
# Each pathology has a center, spread, and clinical characteristics
PATHOLOGY_ZONES = {
    'depression': {
        'ep_center': 1.2, 'freq_center': 0.5,
        'ep_spread': 0.6, 'freq_spread': 0.3,
        'label': 'Depression',
        'color': '#4a5568',
        'description': 'Low energy, slowed dynamics, toward rigidity',
        'regime': 'rigidity',
    },
    'anxiety': {
        'ep_center': 2.5, 'freq_center': 3.5,
        'label': 'Anxiety',
        'ep_spread': 0.8, 'freq_spread': 0.6,
        'color': '#f56565',
        'description': 'Normal energy, hyperactive frequency, chaos boundary',
        'regime': 'chaos',
    },
    'mania': {
        'ep_center': 6.0, 'freq_center': 4.0,
        'ep_spread': 1.5, 'freq_spread': 0.8,
        'label': 'Mania',
        'color': '#ed64a6',
        'description': 'Excessive energy, chaotic dynamics',
        'regime': 'chaos',
    },
    'seizure': {
        'ep_center': 8.5, 'freq_center': 1.0,
        'ep_spread': 1.0, 'freq_spread': 0.4,
        'label': 'Seizure',
        'color': '#e53e3e',
        'description': 'Very high energy, hyper-synchronized, extreme rigidity',
        'regime': 'rigidity',
    },
    'dissociation': {
        'ep_center': 1.5, 'freq_center': 2.5,
        'ep_spread': 0.5, 'freq_spread': 0.8,
        'label': 'Dissociation',
        'color': '#a0aec0',
        'description': 'Low energy with fragmented dynamics, boundary state',
        'regime': 'boundary',
    },
    'adhd': {
        'ep_center': 3.0, 'freq_center': 3.0,
        'ep_spread': 1.2, 'freq_spread': 1.0,
        'label': 'ADHD',
        'color': '#f6ad55',
        'description': 'Unstable, oscillating near chaos boundary',
        'regime': 'chaos',
    },
    'ptsd': {
        'ep_center': 4.5, 'freq_center': 3.8,
        'ep_spread': 1.0, 'freq_spread': 0.6,
        'label': 'PTSD',
        'color': '#805ad5',
        'description': 'Hyperarousal, threat detection hyperactive',
        'regime': 'chaos',
    },
}


def map_hrv_to_frequency(hrv_ms: float) -> float:
    """
    Map Heart Rate Variability (RMSSD in ms) to éR frequency parameter.

    Higher HRV generally indicates more parasympathetic activity and
    lower effective frequency in the coherence space.

    Args:
        hrv_ms: RMSSD heart rate variability in milliseconds

    Returns:
        Estimated frequency parameter for éR calculation

    Note:
        This is a theoretical mapping. Empirical calibration with
        LoomSense data will refine these relationships.
    """
    # Inverse relationship: high HRV -> low frequency (calm, coherent)
    # Typical range: HRV 10-100ms -> freq 0.5-4.0
    # Using logarithmic mapping for physiological realism
    hrv_clamped = np.clip(hrv_ms, 10, 150)
    freq = 4.5 - 1.5 * np.log10(hrv_clamped / 10)
    return np.clip(freq, 0.3, 4.5)


def map_metabolic_rate_to_energy(kcal_per_hour: float) -> float:
    """
    Map metabolic rate (kcal/hour) to éR Energy Present parameter.

    Higher metabolic activity corresponds to more energy present
    in the biological coherence field.

    Args:
        kcal_per_hour: Metabolic rate in kilocalories per hour

    Returns:
        Estimated EP parameter for éR calculation

    Note:
        This is a theoretical mapping. Actual values will depend on
        body mass, activity type, and other factors.
    """
    # Linear-ish mapping with saturation
    # Basal: ~60 kcal/hr -> EP ~2.0
    # Heavy exercise: ~800 kcal/hr -> EP ~9.0
    kcal_clamped = np.clip(kcal_per_hour, 40, 1000)
    ep = 1.5 + (kcal_clamped - 40) * (8.0 / 960)
    return np.clip(ep, 0.5, 10.0)


def map_eeg_band_to_frequency(band: str, power_ratio: float = 1.0) -> float:
    """
    Map EEG frequency band to éR frequency parameter.

    Args:
        band: EEG band name ('delta', 'theta', 'alpha', 'beta', 'gamma')
        power_ratio: Relative power in this band (0-1, affects weighting)

    Returns:
        Estimated frequency parameter
    """
    band_frequencies = {
        'delta': 0.5,    # 0.5-4 Hz - deep sleep
        'theta': 0.8,    # 4-8 Hz - drowsy, meditation
        'alpha': 1.2,    # 8-12 Hz - relaxed awake
        'beta': 2.5,     # 12-30 Hz - active, alert
        'gamma': 4.0,    # 30-100 Hz - high cognition
    }
    base_freq = band_frequencies.get(band.lower(), 1.5)
    return base_freq * (0.5 + 0.5 * power_ratio)


def biological_state_to_er(state_name: str) -> Dict:
    """
    Get the éR parameters for a named biological state.

    Args:
        state_name: Name of state (e.g., 'resting_awake', 'deep_sleep')

    Returns:
        Dictionary with EP, frequency, éR, and description
    """
    if state_name not in BIOLOGICAL_STATES:
        raise ValueError(f"Unknown biological state: {state_name}. "
                        f"Available: {list(BIOLOGICAL_STATES.keys())}")

    state = BIOLOGICAL_STATES[state_name]
    er = state['ep'] / (state['freq'] ** 2)

    return {
        'state': state_name,
        'energy_present': state['ep'],
        'frequency': state['freq'],
        'energy_resistance': er,
        'label': state['label'],
        'description': state['description'],
        'metabolic_rate': state.get('metabolic_rate'),
        'hrv_ms': state.get('hrv_ms'),
    }


class EnergyResistanceVisualizer:
    """
    Interactive visualizer for the Energy Resistance phase space.

    Displays the relationship between Energy Present (EP), frequency (f),
    and Energy Resistance (éR), highlighting the viable window for
    biological coherence.

    Enhanced Features:
    - Biological Mode: Maps axes to physiological parameters
    - Pathology Zones: Shows where mental health conditions appear
    - Clinical Trajectories: Decompensation and recovery paths
    """

    # Default viable window boundaries (éR units)
    DEFAULT_ER_MIN = 0.5   # Below this: chaos/decoherence
    DEFAULT_ER_MAX = 5.0   # Above this: rigidity/frozen

    # Custom colormap: chaos (purple) -> viable (green) -> rigid (blue)
    COLORMAP_COLORS = [
        (0.4, 0.0, 0.6),    # Deep purple - chaos
        (0.6, 0.2, 0.8),    # Purple - low éR
        (0.2, 0.8, 0.4),    # Green - viable zone center
        (0.0, 0.6, 0.8),    # Cyan - high éR
        (0.0, 0.2, 0.6),    # Deep blue - rigidity
    ]

    def __init__(self,
                 ep_range: Tuple[float, float] = (0.1, 10.0),
                 freq_range: Tuple[float, float] = (0.1, 5.0),
                 resolution: int = 500):
        """
        Initialize the visualizer.

        Args:
            ep_range: (min, max) range for Energy Present axis
            freq_range: (min, max) range for frequency axis
            resolution: Grid resolution for the phase space
        """
        self.ep_range = ep_range
        self.freq_range = freq_range
        self.resolution = resolution

        # Viable window boundaries
        self.er_min = self.DEFAULT_ER_MIN
        self.er_max = self.DEFAULT_ER_MAX

        # Trajectories storage
        self.trajectories: List[dict] = []

        # Create custom colormap
        self.cmap = self._create_colormap()

        # Display mode flags
        self.biological_mode = False
        self.show_pathology = False
        self.show_biological_states = False

        # Initialize plot elements (will be set in setup)
        self.fig = None
        self.ax_main = None
        self.ax_colorbar = None
        self.im = None
        self.viable_patch = None
        self.trajectory_lines = []
        self.pathology_patches = []
        self.state_markers = []
        self.state_annotations = []

    def _create_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for éR visualization."""
        return LinearSegmentedColormap.from_list(
            'energy_resistance',
            self.COLORMAP_COLORS,
            N=256
        )

    @staticmethod
    def calculate_er(ep: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """
        Calculate Energy Resistance: éR = EP / f²

        Args:
            ep: Energy Present values
            freq: Frequency values

        Returns:
            Energy Resistance values
        """
        return ep / (freq ** 2)

    def _compute_phase_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the éR values across the phase space grid."""
        ep = np.linspace(self.ep_range[0], self.ep_range[1], self.resolution)
        freq = np.linspace(self.freq_range[0], self.freq_range[1], self.resolution)

        EP, F = np.meshgrid(ep, freq)
        ER = self.calculate_er(EP, F)

        return EP, F, ER

    def _get_viable_boundary_coords(self) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
        """
        Calculate the viable window boundary curves.

        For éR = EP / f², the boundaries are:
        - Lower: EP = éR_min * f²
        - Upper: EP = éR_max * f²
        """
        freq = np.linspace(self.freq_range[0], self.freq_range[1], 200)

        ep_lower = self.er_min * freq**2
        ep_upper = self.er_max * freq**2

        # Clip to visible range
        mask_lower = ep_lower <= self.ep_range[1]
        mask_upper = ep_upper <= self.ep_range[1]

        return (freq[mask_lower], ep_lower[mask_lower],
                freq[mask_upper], ep_upper[mask_upper])

    def add_trajectory(self,
                       ep_values: np.ndarray,
                       freq_values: np.ndarray,
                       label: str = "System",
                       color: str = 'white',
                       show_arrow: bool = True):
        """
        Add a system trajectory to the visualization.

        Args:
            ep_values: Array of EP values over time
            freq_values: Array of frequency values over time
            label: Label for the trajectory
            color: Line color
            show_arrow: Whether to show direction arrow
        """
        self.trajectories.append({
            'ep': np.array(ep_values),
            'freq': np.array(freq_values),
            'label': label,
            'color': color,
            'show_arrow': show_arrow
        })

    def add_example_trajectories(self):
        """Add example trajectories demonstrating different system behaviors."""

        # 1. Healthy oscillation within viable window
        t = np.linspace(0, 4*np.pi, 100)
        ep_healthy = 2.5 + 0.8 * np.sin(t)
        freq_healthy = 1.5 + 0.3 * np.cos(t * 1.3)
        self.add_trajectory(ep_healthy, freq_healthy,
                           "Healthy Oscillation", '#00FF88')

        # 2. System falling into chaos (decreasing éR)
        t = np.linspace(0, 1, 50)
        ep_chaos = 3.0 - 2.0 * t
        freq_chaos = 1.0 + 2.5 * t
        self.add_trajectory(ep_chaos, freq_chaos,
                           "Decoherence Path", '#FF6B6B')

        # 3. System becoming rigid (increasing éR)
        t = np.linspace(0, 1, 50)
        ep_rigid = 2.0 + 6.0 * t
        freq_rigid = 1.2 - 0.5 * t
        self.add_trajectory(ep_rigid, freq_rigid,
                           "Rigidity Path", '#6B9FFF')

        # 4. Recovery trajectory - from chaos back to viable
        t = np.linspace(0, 1, 50)
        ep_recovery = 0.5 + 2.5 * t
        freq_recovery = 2.5 - 1.0 * t
        self.add_trajectory(ep_recovery, freq_recovery,
                           "Recovery Path", '#FFD93D')

    def add_clinical_trajectories(self):
        """Add clinically-relevant trajectories showing decompensation and recovery."""

        # 1. Depression onset: gradual energy and frequency decline
        t = np.linspace(0, 1, 60)
        ep_depression = 3.0 - 1.8 * t + 0.3 * np.sin(6 * t)  # Declining with tremor
        freq_depression = 1.5 - 1.0 * t + 0.2 * np.sin(8 * t)
        self.add_trajectory(ep_depression, freq_depression,
                           "Depression Onset", '#4a5568')

        # 2. Anxiety escalation: frequency increases while energy fluctuates
        t = np.linspace(0, 1, 60)
        ep_anxiety = 2.5 + 0.8 * np.sin(10 * t)  # Fluctuating energy
        freq_anxiety = 1.5 + 2.0 * t  # Rising frequency
        self.add_trajectory(ep_anxiety, freq_anxiety,
                           "Anxiety Escalation", '#f56565')

        # 3. Manic episode: energy and frequency spike together
        t = np.linspace(0, 1, 50)
        ep_mania = 2.5 + 4.0 * t * (1 + 0.2 * np.sin(15 * t))
        freq_mania = 1.5 + 2.5 * t
        self.add_trajectory(ep_mania, freq_mania,
                           "Manic Episode", '#ed64a6')

        # 4. Panic attack: rapid spike then crash
        t = np.linspace(0, 1, 80)
        attack_peak = 0.3
        ep_panic = 2.5 + 3.0 * np.exp(-((t - attack_peak) / 0.15)**2)
        freq_panic = 1.5 + 2.5 * np.exp(-((t - attack_peak) / 0.12)**2)
        self.add_trajectory(ep_panic, freq_panic,
                           "Panic Attack", '#e53e3e')

        # 5. Therapeutic recovery: gradual return to viable window
        t = np.linspace(0, 1, 80)
        # Starting from anxiety region, spiraling back to center
        ep_therapy = 2.5 + 1.0 * np.exp(-3 * t) * np.cos(8 * t)
        freq_therapy = 3.5 - 2.0 * t + 0.5 * np.exp(-2 * t) * np.sin(6 * t)
        self.add_trajectory(ep_therapy, freq_therapy,
                           "Therapeutic Recovery", '#48bb78')

        # 6. Meditation deepening: controlled descent into coherent state
        t = np.linspace(0, 1, 50)
        ep_meditation = 2.5 - 0.5 * t
        freq_meditation = 1.2 - 0.4 * t
        self.add_trajectory(ep_meditation, freq_meditation,
                           "Meditation Deepening", '#68d391')

    def _draw_biological_states(self):
        """Draw markers for reference biological states."""
        for state_name, state in BIOLOGICAL_STATES.items():
            ep, freq = state['ep'], state['freq']

            # Skip if outside visible range
            if not (self.ep_range[0] <= ep <= self.ep_range[1] and
                    self.freq_range[0] <= freq <= self.freq_range[1]):
                continue

            # Draw marker
            marker = self.ax_main.scatter(
                ep, freq, s=150, c=state['color'],
                marker='*', edgecolors='white', linewidths=1.5,
                zorder=10, alpha=0.9
            )
            self.state_markers.append(marker)

            # Add label
            offset = (0.2, 0.1) if freq < 3 else (0.2, -0.2)
            ann = self.ax_main.annotate(
                state['label'],
                xy=(ep, freq),
                xytext=(ep + offset[0], freq + offset[1]),
                fontsize=8, color=state['color'],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e',
                         edgecolor=state['color'], alpha=0.8),
                arrowprops=dict(arrowstyle='->', color=state['color'],
                               connectionstyle='arc3,rad=0.1'),
                zorder=11
            )
            self.state_annotations.append(ann)

    def _draw_pathology_zones(self):
        """Draw elliptical zones for pathology regions."""
        for path_name, pathology in PATHOLOGY_ZONES.items():
            ep_c, freq_c = pathology['ep_center'], pathology['freq_center']
            ep_s, freq_s = pathology['ep_spread'], pathology['freq_spread']

            # Skip if center is outside visible range
            if not (self.ep_range[0] <= ep_c <= self.ep_range[1] and
                    self.freq_range[0] <= freq_c <= self.freq_range[1]):
                continue

            # Draw ellipse for pathology zone
            ellipse = Ellipse(
                (ep_c, freq_c),
                width=ep_s * 2, height=freq_s * 2,
                facecolor=pathology['color'], alpha=0.25,
                edgecolor=pathology['color'], linewidth=2,
                linestyle='--', zorder=3
            )
            self.ax_main.add_patch(ellipse)
            self.pathology_patches.append(ellipse)

            # Add label at center
            self.ax_main.text(
                ep_c, freq_c, pathology['label'],
                fontsize=9, fontweight='bold',
                color=pathology['color'], ha='center', va='center',
                alpha=0.9, zorder=4
            )

    def _clear_overlays(self):
        """Clear biological state markers and pathology zones."""
        for marker in self.state_markers:
            marker.remove()
        self.state_markers = []

        for ann in self.state_annotations:
            ann.remove()
        self.state_annotations = []

        for patch in self.pathology_patches:
            patch.remove()
        self.pathology_patches = []

    def setup_plot(self, figsize: Tuple[int, int] = (14, 10)):
        """Set up the matplotlib figure and axes."""

        # Create figure with space for controls
        self.fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')

        # Main phase space plot
        self.ax_main = self.fig.add_axes([0.1, 0.25, 0.65, 0.65])
        self.ax_main.set_facecolor('#16213e')

        # Colorbar axis
        self.ax_colorbar = self.fig.add_axes([0.78, 0.25, 0.03, 0.65])

        # Style the axes
        for spine in self.ax_main.spines.values():
            spine.set_color('#4a5568')
        self.ax_main.tick_params(colors='#a0aec0')

    def _draw_phase_space(self):
        """Draw the main phase space visualization."""
        EP, F, ER = self._compute_phase_space()

        # Normalize éR for color mapping (log scale for better visualization)
        er_log = np.log10(ER + 0.01)
        er_normalized = (er_log - er_log.min()) / (er_log.max() - er_log.min())

        # Draw the phase space
        self.im = self.ax_main.imshow(
            er_normalized,
            extent=[self.ep_range[0], self.ep_range[1],
                   self.freq_range[0], self.freq_range[1]],
            origin='lower',
            aspect='auto',
            cmap=self.cmap,
            alpha=0.9
        )

        # Add colorbar
        cbar = self.fig.colorbar(self.im, cax=self.ax_colorbar)
        cbar.set_label('Energy Resistance (éR)', color='#a0aec0', fontsize=12)
        cbar.ax.yaxis.set_tick_params(color='#a0aec0')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a0aec0')

        # Custom colorbar ticks
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Chaos\n(Low)', '', 'Viable\n(Optimal)', '', 'Rigid\n(High)'])

    def _draw_viable_window(self):
        """Draw the viable window boundaries."""
        f_low, ep_low, f_high, ep_high = self._get_viable_boundary_coords()

        # Draw boundary curves
        self.ax_main.plot(ep_low, f_low, '--', color='#00ff88',
                         linewidth=2.5, label=f'éR = {self.er_min} (Chaos Threshold)')
        self.ax_main.plot(ep_high, f_high, '--', color='#00ff88',
                         linewidth=2.5, label=f'éR = {self.er_max} (Rigidity Threshold)')

        # Create shaded viable region
        # Build polygon vertices
        ep_range_vals = np.linspace(self.ep_range[0], self.ep_range[1], 100)

        # Upper boundary (lower éR = chaos threshold): f = sqrt(EP / éR_min)
        f_upper = np.sqrt(ep_range_vals / self.er_min)
        f_upper = np.clip(f_upper, self.freq_range[0], self.freq_range[1])

        # Lower boundary (higher éR = rigidity threshold): f = sqrt(EP / éR_max)
        f_lower = np.sqrt(ep_range_vals / self.er_max)
        f_lower = np.clip(f_lower, self.freq_range[0], self.freq_range[1])

        # Only shade where both boundaries are in range
        valid_mask = (f_upper > self.freq_range[0]) & (f_lower < self.freq_range[1])

        if np.any(valid_mask):
            self.ax_main.fill_between(
                ep_range_vals[valid_mask],
                f_lower[valid_mask],
                f_upper[valid_mask],
                alpha=0.15,
                color='#00ff88',
                label='Viable Window'
            )

        # Add zone labels
        self.ax_main.text(0.5, 4.2, 'CHAOS\nZONE', fontsize=14, fontweight='bold',
                         color='#9f7aea', ha='center', va='center', alpha=0.8)
        self.ax_main.text(8.0, 0.8, 'RIGIDITY\nZONE', fontsize=14, fontweight='bold',
                         color='#4299e1', ha='center', va='center', alpha=0.8)
        self.ax_main.text(3.5, 2.0, 'VIABLE\nWINDOW', fontsize=16, fontweight='bold',
                         color='#00ff88', ha='center', va='center', alpha=0.9)

    def _draw_trajectories(self):
        """Draw all system trajectories."""
        for traj in self.trajectories:
            # Plot trajectory line
            line, = self.ax_main.plot(
                traj['ep'], traj['freq'],
                '-', color=traj['color'], linewidth=2.5, alpha=0.9,
                label=traj['label']
            )
            self.trajectory_lines.append(line)

            # Add start and end markers
            self.ax_main.scatter(traj['ep'][0], traj['freq'][0],
                               s=80, c=traj['color'], marker='o',
                               edgecolors='white', linewidths=1.5, zorder=5)
            self.ax_main.scatter(traj['ep'][-1], traj['freq'][-1],
                               s=100, c=traj['color'], marker='s',
                               edgecolors='white', linewidths=1.5, zorder=5)

            # Add direction arrow
            if traj['show_arrow'] and len(traj['ep']) > 2:
                mid_idx = len(traj['ep']) // 2
                dx = traj['ep'][mid_idx+1] - traj['ep'][mid_idx-1]
                dy = traj['freq'][mid_idx+1] - traj['freq'][mid_idx-1]
                self.ax_main.annotate('',
                    xy=(traj['ep'][mid_idx] + dx*0.1, traj['freq'][mid_idx] + dy*0.1),
                    xytext=(traj['ep'][mid_idx], traj['freq'][mid_idx]),
                    arrowprops=dict(arrowstyle='->', color=traj['color'], lw=2))

    def _add_labels_and_legend(self):
        """Add axis labels, title, and legend."""
        if self.biological_mode:
            # Biological mode labels
            xlabel = 'Energy Present (EP)\n← Lower Metabolism | Higher Metabolism →'
            ylabel = 'Frequency (f)\n← Slower Rhythms | Faster Rhythms →'
            title = 'Energy Resistance Phase Space\nBiological Mapping Mode'
        else:
            # Abstract mode labels
            xlabel = 'Energy Present (EP)'
            ylabel = 'Frequency (f)'
            title = 'Energy Resistance Phase Space\néR = EP / f²'

        self.ax_main.set_xlabel(xlabel, fontsize=12,
                               color='#e2e8f0', labelpad=10)
        self.ax_main.set_ylabel(ylabel, fontsize=12,
                               color='#e2e8f0', labelpad=10)
        self.ax_main.set_title(title, fontsize=16, fontweight='bold',
                              color='#f7fafc', pad=15)

        # Add legend
        legend = self.ax_main.legend(loc='upper right', fontsize=8,
                                    facecolor='#2d3748', edgecolor='#4a5568',
                                    labelcolor='#e2e8f0')

        # Add equation box - content depends on mode
        if self.biological_mode:
            eq_text = (
                "Biological Mapping\n"
                "━━━━━━━━━━━━━━━━━\n"
                "EP ↔ Metabolic Rate\n"
                "  (ATP, O₂, HR)\n\n"
                "f ↔ Neural Rhythms\n"
                "  (EEG, HRV, Circadian)\n\n"
                "éR = EP / f²\n\n"
                "★ = Reference States\n"
                "⬭ = Pathology Zones"
            )
        else:
            eq_text = (
                "Energy Resistance\n"
                "━━━━━━━━━━━━━━━━━\n"
                "éR = EP / f²\n\n"
                f"Viable Window:\n"
                f"  {self.er_min} < éR < {self.er_max}\n\n"
                "• Low éR → Chaos\n"
                "• High éR → Rigidity\n"
                "• Middle → Life"
            )

        self.fig.text(0.88, 0.60, eq_text, fontsize=9,
                     color='#e2e8f0', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='#2d3748',
                              edgecolor='#4a5568', alpha=0.9),
                     verticalalignment='top')

    def _setup_interactive_controls(self):
        """Set up interactive sliders and buttons."""

        # Slider for éR minimum (chaos threshold)
        ax_er_min = self.fig.add_axes([0.1, 0.14, 0.30, 0.025])
        self.slider_er_min = Slider(
            ax_er_min, 'Chaos (éR min)', 0.1, 2.0,
            valinit=self.er_min, valstep=0.1,
            color='#9f7aea'
        )
        ax_er_min.set_facecolor('#2d3748')

        # Slider for éR maximum (rigidity threshold)
        ax_er_max = self.fig.add_axes([0.1, 0.10, 0.30, 0.025])
        self.slider_er_max = Slider(
            ax_er_max, 'Rigid (éR max)', 2.0, 15.0,
            valinit=self.er_max, valstep=0.5,
            color='#4299e1'
        )
        ax_er_max.set_facecolor('#2d3748')

        # Toggle checkboxes for biological mode and pathology
        ax_toggles = self.fig.add_axes([0.45, 0.06, 0.15, 0.10])
        ax_toggles.set_facecolor('#2d3748')
        self.check_buttons = CheckButtons(
            ax_toggles,
            ['Biological Mode', 'Show Pathology', 'Show States'],
            [self.biological_mode, self.show_pathology, self.show_biological_states]
        )
        # Style the checkboxes
        for text in self.check_buttons.labels:
            text.set_color('#e2e8f0')
            text.set_fontsize(9)

        # Reset button
        ax_reset = self.fig.add_axes([0.63, 0.07, 0.08, 0.035])
        self.button_reset = Button(ax_reset, 'Reset',
                                   color='#4a5568', hovercolor='#718096')

        # Clinical trajectories button
        ax_clinical = self.fig.add_axes([0.63, 0.115, 0.08, 0.035])
        self.button_clinical = Button(ax_clinical, 'Clinical',
                                      color='#553c9a', hovercolor='#6b46c1')

        # Connect callbacks
        self.slider_er_min.on_changed(self._update_boundaries)
        self.slider_er_max.on_changed(self._update_boundaries)
        self.button_reset.on_clicked(self._reset_parameters)
        self.button_clinical.on_clicked(self._toggle_clinical_trajectories)
        self.check_buttons.on_clicked(self._toggle_mode)

    def _update_boundaries(self, val):
        """Callback to update viable window when sliders change."""
        self.er_min = self.slider_er_min.val
        self.er_max = self.slider_er_max.val
        self._redraw_plot()

    def _reset_parameters(self, event):
        """Reset sliders to default values."""
        self.slider_er_min.set_val(self.DEFAULT_ER_MIN)
        self.slider_er_max.set_val(self.DEFAULT_ER_MAX)

    def _toggle_mode(self, label):
        """Handle checkbox toggles for display modes."""
        if label == 'Biological Mode':
            self.biological_mode = not self.biological_mode
        elif label == 'Show Pathology':
            self.show_pathology = not self.show_pathology
        elif label == 'Show States':
            self.show_biological_states = not self.show_biological_states

        # Redraw
        self._redraw_plot()

    def _toggle_clinical_trajectories(self, event):
        """Toggle clinical trajectories on/off."""
        # Check if clinical trajectories are already added
        clinical_labels = {'Depression Onset', 'Anxiety Escalation', 'Manic Episode',
                          'Panic Attack', 'Therapeutic Recovery', 'Meditation Deepening'}

        existing_labels = {t['label'] for t in self.trajectories}

        if clinical_labels & existing_labels:
            # Remove clinical trajectories
            self.trajectories = [t for t in self.trajectories
                                if t['label'] not in clinical_labels]
        else:
            # Add clinical trajectories
            self.add_clinical_trajectories()

        self._redraw_plot()

    def _redraw_plot(self):
        """Redraw the entire plot with current settings."""
        self.ax_main.clear()
        self._clear_overlays()
        self._draw_phase_space()
        self._draw_viable_window()

        if self.show_pathology:
            self._draw_pathology_zones()

        if self.show_biological_states:
            self._draw_biological_states()

        self._draw_trajectories()
        self._add_labels_and_legend()
        self.fig.canvas.draw_idle()

    def render(self,
               interactive: bool = True,
               show_trajectories: bool = True,
               show_biological_states: bool = False,
               show_pathology: bool = False,
               biological_mode: bool = False,
               save_path: Optional[str] = None):
        """
        Render the complete visualization.

        Args:
            interactive: Whether to show interactive controls
            show_trajectories: Whether to display example trajectories
            show_biological_states: Whether to show biological reference states
            show_pathology: Whether to show pathology zones
            biological_mode: Whether to use biological axis labels
            save_path: If provided, save figure to this path
        """
        # Set display modes
        self.biological_mode = biological_mode
        self.show_biological_states = show_biological_states
        self.show_pathology = show_pathology

        self.setup_plot()
        self._draw_phase_space()
        self._draw_viable_window()

        if show_pathology:
            self._draw_pathology_zones()

        if show_biological_states:
            self._draw_biological_states()

        if show_trajectories:
            if not self.trajectories:
                self.add_example_trajectories()
            self._draw_trajectories()

        self._add_labels_and_legend()

        if interactive:
            self._setup_interactive_controls()

        if save_path:
            self.fig.savefig(save_path, dpi=150, facecolor=self.fig.get_facecolor(),
                           bbox_inches='tight', pad_inches=0.3)
            print(f"Figure saved to: {save_path}")

        plt.show()

    def create_static_figure(self, save_path: Optional[str] = None):
        """Create a static (non-interactive) version for publications."""
        self.render(interactive=False, show_trajectories=True, save_path=save_path)


def calculate_system_er(energy_present: float, frequency: float) -> dict:
    """
    Calculate éR for a given system state and determine its regime.

    Args:
        energy_present: Current energy in the system
        frequency: Fundamental frequency of the system

    Returns:
        Dictionary with éR value and regime classification
    """
    er = energy_present / (frequency ** 2)

    if er < EnergyResistanceVisualizer.DEFAULT_ER_MIN:
        regime = "chaos"
        description = "System at risk of decoherence - insufficient energy resistance"
    elif er > EnergyResistanceVisualizer.DEFAULT_ER_MAX:
        regime = "rigidity"
        description = "System becoming rigid - excessive energy resistance"
    else:
        regime = "viable"
        description = "System in viable window - optimal for coherent organization"

    return {
        'energy_present': energy_present,
        'frequency': frequency,
        'energy_resistance': er,
        'regime': regime,
        'description': description
    }


def demo():
    """Run a demonstration of the Energy Resistance visualizer."""
    print("=" * 70)
    print("  Cosmic Loom Theory: Energy Resistance Visualizer (v1.2)")
    print("=" * 70)
    print("\nThe Energy Resistance Principle: éR = EP / f²")
    print("\nThis visualization shows the phase space where:")
    print("  • Purple regions: Low éR (chaos/decoherence)")
    print("  • Green regions: Viable window (biological coherence)")
    print("  • Blue regions: High éR (rigidity)")
    print("\nNEW FEATURES:")
    print("  ☐ Biological Mode - Map axes to physiological parameters")
    print("  ☐ Show Pathology - Display mental health condition zones")
    print("  ☐ Show States - Display biological reference states")
    print("  [Clinical] - Toggle clinical decompensation/recovery paths")
    print("\nUse the sliders to adjust the viable window boundaries.")
    print("=" * 70)

    # Create and show the visualizer
    visualizer = EnergyResistanceVisualizer()
    visualizer.render(interactive=True, show_trajectories=True)


def demo_biological():
    """
    Run a demonstration focused on biological and clinical applications.

    Shows the éR phase space with:
    - Biological reference states (sleep, meditation, exercise, etc.)
    - Pathology zones (depression, anxiety, mania, etc.)
    - Clinical trajectories (decompensation and recovery paths)
    """
    print("=" * 70)
    print("  Cosmic Loom Theory: Biological & Clinical Mode")
    print("=" * 70)
    print("\nThis view maps abstract éR parameters to physiology:")
    print("\n  ENERGY PRESENT (EP) ↔ Metabolic Indicators")
    print("    • ATP production, oxygen consumption")
    print("    • Heart rate, metabolic rate")
    print("    • Autonomic activation level")
    print("\n  FREQUENCY (f) ↔ Neural/Physiological Rhythms")
    print("    • EEG frequency bands (delta to gamma)")
    print("    • Heart rate variability (HRV)")
    print("    • Circadian rhythms")
    print("\n  REFERENCE STATES (★):")
    for name, state in BIOLOGICAL_STATES.items():
        er = state['ep'] / (state['freq'] ** 2)
        print(f"    • {state['label']}: éR = {er:.2f}")
    print("\n  PATHOLOGY ZONES (dashed ellipses):")
    for name, path in PATHOLOGY_ZONES.items():
        print(f"    • {path['label']}: {path['description']}")
    print("=" * 70)

    # Create visualizer with biological features enabled
    visualizer = EnergyResistanceVisualizer()
    visualizer.add_clinical_trajectories()
    visualizer.render(
        interactive=True,
        show_trajectories=True,
        show_biological_states=True,
        show_pathology=True,
        biological_mode=True
    )


def demo_pathology():
    """
    Run a demonstration focused specifically on pathology signatures.

    Shows how different mental health conditions map to éR space
    and the trajectories of decompensation and recovery.
    """
    print("=" * 70)
    print("  Cosmic Loom Theory: Pathology Signatures in éR Space")
    print("=" * 70)
    print("\nMental health conditions as energetic regimes:")
    print("\n  CHAOS BOUNDARY CONDITIONS (low éR):")
    print("    • Anxiety: High frequency, normal energy")
    print("    • Mania: High frequency AND high energy")
    print("    • ADHD: Unstable, oscillating near boundary")
    print("    • PTSD: Hyperarousal, threat detection overdrive")
    print("\n  RIGIDITY BOUNDARY CONDITIONS (high éR):")
    print("    • Depression: Low frequency AND low energy")
    print("    • Seizure: Very high energy, hyper-synchronized")
    print("\n  BOUNDARY/FRAGMENTED STATES:")
    print("    • Dissociation: Low energy, fragmented dynamics")
    print("\n  CLINICAL TRAJECTORIES:")
    print("    • Depression Onset: Gradual decline toward rigidity")
    print("    • Anxiety Escalation: Frequency increases into chaos")
    print("    • Panic Attack: Rapid spike then crash")
    print("    • Therapeutic Recovery: Spiral back to viable window")
    print("=" * 70)

    visualizer = EnergyResistanceVisualizer()
    visualizer.trajectories = []  # Clear default trajectories
    visualizer.add_clinical_trajectories()
    visualizer.render(
        interactive=True,
        show_trajectories=True,
        show_biological_states=False,
        show_pathology=True,
        biological_mode=True
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--bio':
            demo_biological()
        elif sys.argv[1] == '--pathology':
            demo_pathology()
        else:
            demo()
    else:
        demo()
