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
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from typing import Tuple, List, Optional


class EnergyResistanceVisualizer:
    """
    Interactive visualizer for the Energy Resistance phase space.

    Displays the relationship between Energy Present (EP), frequency (f),
    and Energy Resistance (éR), highlighting the viable window for
    biological coherence.
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

        # Initialize plot elements (will be set in setup)
        self.fig = None
        self.ax_main = None
        self.ax_colorbar = None
        self.im = None
        self.viable_patch = None
        self.trajectory_lines = []

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
        self.ax_main.set_xlabel('Energy Present (EP)', fontsize=14,
                               color='#e2e8f0', labelpad=10)
        self.ax_main.set_ylabel('Frequency (f)', fontsize=14,
                               color='#e2e8f0', labelpad=10)
        self.ax_main.set_title('Energy Resistance Phase Space\néR = EP / f²',
                              fontsize=18, fontweight='bold',
                              color='#f7fafc', pad=20)

        # Add legend
        legend = self.ax_main.legend(loc='upper right', fontsize=9,
                                    facecolor='#2d3748', edgecolor='#4a5568',
                                    labelcolor='#e2e8f0')

        # Add equation box
        eq_text = (
            "Energy Resistance Principle\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "éR = EP / f²\n\n"
            f"Viable Window:\n"
            f"  {self.er_min} < éR < {self.er_max}\n\n"
            "• Low éR → Chaos\n"
            "• High éR → Rigidity\n"
            "• Middle → Life thrives"
        )
        self.fig.text(0.88, 0.65, eq_text, fontsize=10,
                     color='#e2e8f0', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='#2d3748',
                              edgecolor='#4a5568', alpha=0.9),
                     verticalalignment='top')

    def _setup_interactive_controls(self):
        """Set up interactive sliders and buttons."""

        # Slider for éR minimum (chaos threshold)
        ax_er_min = self.fig.add_axes([0.15, 0.12, 0.35, 0.03])
        self.slider_er_min = Slider(
            ax_er_min, 'Chaos Threshold (éR min)', 0.1, 2.0,
            valinit=self.er_min, valstep=0.1,
            color='#9f7aea'
        )
        ax_er_min.set_facecolor('#2d3748')

        # Slider for éR maximum (rigidity threshold)
        ax_er_max = self.fig.add_axes([0.15, 0.07, 0.35, 0.03])
        self.slider_er_max = Slider(
            ax_er_max, 'Rigidity Threshold (éR max)', 2.0, 15.0,
            valinit=self.er_max, valstep=0.5,
            color='#4299e1'
        )
        ax_er_max.set_facecolor('#2d3748')

        # Reset button
        ax_reset = self.fig.add_axes([0.6, 0.07, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset',
                                   color='#4a5568', hovercolor='#718096')

        # Connect callbacks
        self.slider_er_min.on_changed(self._update_boundaries)
        self.slider_er_max.on_changed(self._update_boundaries)
        self.button_reset.on_clicked(self._reset_parameters)

    def _update_boundaries(self, val):
        """Callback to update viable window when sliders change."""
        self.er_min = self.slider_er_min.val
        self.er_max = self.slider_er_max.val

        # Redraw the plot
        self.ax_main.clear()
        self._draw_phase_space()
        self._draw_viable_window()
        self._draw_trajectories()
        self._add_labels_and_legend()
        self.fig.canvas.draw_idle()

    def _reset_parameters(self, event):
        """Reset sliders to default values."""
        self.slider_er_min.set_val(self.DEFAULT_ER_MIN)
        self.slider_er_max.set_val(self.DEFAULT_ER_MAX)

    def render(self,
               interactive: bool = True,
               show_trajectories: bool = True,
               save_path: Optional[str] = None):
        """
        Render the complete visualization.

        Args:
            interactive: Whether to show interactive controls
            show_trajectories: Whether to display example trajectories
            save_path: If provided, save figure to this path
        """
        self.setup_plot()
        self._draw_phase_space()
        self._draw_viable_window()

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
    print("=" * 60)
    print("  Cosmic Loom Theory: Energy Resistance Visualizer")
    print("=" * 60)
    print("\nThe Energy Resistance Principle: éR = EP / f²")
    print("\nThis visualization shows the phase space where:")
    print("  • Purple regions: Low éR (chaos/decoherence)")
    print("  • Green regions: Viable window (biological coherence)")
    print("  • Blue regions: High éR (rigidity)")
    print("\nUse the sliders to adjust the viable window boundaries.")
    print("=" * 60)

    # Create and show the visualizer
    visualizer = EnergyResistanceVisualizer()
    visualizer.render(interactive=True, show_trajectories=True)


if __name__ == "__main__":
    demo()
