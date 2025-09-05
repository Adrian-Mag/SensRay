"""
Earth visualization and plotting functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from ..core.earth_models import EarthModelManager


class EarthPlotter:
    """
    Class for creating Earth visualizations and ray path plots.
    """

    def __init__(self, model_name: str = "iasp91"):
        """
        Initialize the Earth plotter.

        Parameters
        ----------
        model_name : str
            Name of the Earth model to use
        """
        self.model_name = model_name
        self.model_manager = EarthModelManager()
        self.earth_structure = self.model_manager.get_earth_structure(model_name)

    def plot_circular_earth(self,
                           ray_coordinates: Dict,
                           source_depth: float,
                           distance_deg: float,
                           fig_size: Tuple[int, int] = (12, 10),
                           show_atmosphere: bool = True) -> plt.Figure:
        """
        Create a circular Earth cross-section with ray paths.

        Parameters
        ----------
        ray_coordinates : Dict
            Dictionary with ray path coordinates from RayPathTracer
        source_depth : float
            Source depth in km
        distance_deg : float
            Epicentral distance in degrees
        fig_size : Tuple[int, int]
            Figure size (width, height)
        show_atmosphere : bool
            Whether to show atmosphere layer

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=fig_size)

        # Earth structure parameters
        earth_radius = self.earth_structure['earth_radius']
        cmb_radius = self.earth_structure['cmb_radius']
        icb_radius = self.earth_structure['icb_radius']

        # Create angular arrays for boundaries
        theta = np.linspace(0, np.pi, 180)  # Semicircle
        theta_full = np.linspace(0, 2*np.pi, 360)  # Full circle for filling

        # Fill Earth layers
        self._fill_earth_layers(ax, theta_full, earth_radius, cmb_radius,
                               icb_radius, show_atmosphere)

        # Plot Earth boundaries
        self._plot_earth_boundaries(ax, theta, earth_radius, cmb_radius, icb_radius)

        # Plot ray paths
        self._plot_ray_paths(ax, ray_coordinates)

        # Mark source and receiver
        self._mark_source_receiver(ax, source_depth, distance_deg, earth_radius)

        # Add distance arc
        self._add_distance_arc(ax, distance_deg, earth_radius)

        # Format plot
        self._format_earth_plot(ax, source_depth, distance_deg, earth_radius)

        return fig

    def _fill_earth_layers(self, ax, theta_full, earth_radius, cmb_radius,
                          icb_radius, show_atmosphere):
        """Fill Earth layers with colors."""
        # Fill atmosphere
        if show_atmosphere:
            atmosphere_radius = earth_radius + 500
            x_atm = atmosphere_radius * np.cos(theta_full)
            y_atm = atmosphere_radius * np.sin(theta_full)
            ax.fill(x_atm, y_atm, color='lightblue', alpha=0.2,
                   label='Atmosphere')

        # Fill mantle
        x_surf = earth_radius * np.cos(theta_full)
        y_surf = earth_radius * np.sin(theta_full)
        ax.fill(x_surf, y_surf, color='saddlebrown', alpha=0.4, label='Mantle')

        # Fill outer core
        x_cmb = cmb_radius * np.cos(theta_full)
        y_cmb = cmb_radius * np.sin(theta_full)
        ax.fill(x_cmb, y_cmb, color='red', alpha=0.5, label='Outer Core')

        # Fill inner core
        x_ic = icb_radius * np.cos(theta_full)
        y_ic = icb_radius * np.sin(theta_full)
        ax.fill(x_ic, y_ic, color='gold', alpha=0.6, label='Inner Core')

    def _plot_earth_boundaries(self, ax, theta, earth_radius, cmb_radius, icb_radius):
        """Plot Earth boundary lines."""
        # Surface
        x_surface = earth_radius * np.cos(theta)
        y_surface = earth_radius * np.sin(theta)
        ax.plot(x_surface, y_surface, 'k-', linewidth=3, label='Surface')

        # Core-mantle boundary
        x_cmb = cmb_radius * np.cos(theta)
        y_cmb = cmb_radius * np.sin(theta)
        ax.plot(x_cmb, y_cmb, 'r--', linewidth=2, alpha=0.8,
               label='Core-Mantle Boundary')

        # Inner core boundary
        x_ic = icb_radius * np.cos(theta)
        y_ic = icb_radius * np.sin(theta)
        ax.plot(x_ic, y_ic, 'orange', linestyle='--', linewidth=2,
               alpha=0.8, label='Inner Core Boundary')

    def _plot_ray_paths(self, ax, ray_coordinates):
        """Plot ray paths."""
        colors = ['blue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan']

        for i, (phase, coords) in enumerate(ray_coordinates.items()):
            color = colors[i % len(colors)]

            ax.plot(coords['x_cartesian'], coords['y_cartesian'],
                   color=color, linewidth=3,
                   label=f"{phase} ({coords['total_time']:.1f}s)")

    def _mark_source_receiver(self, ax, source_depth, distance_deg, earth_radius):
        """Mark source and receiver locations."""
        # Source
        source_radius = earth_radius - source_depth
        ax.plot(source_radius, 0, 'r*', markersize=20,
               markeredgecolor='black', markeredgewidth=1, label='Source')

        # Receiver
        receiver_angle = distance_deg * np.pi / 180.0
        receiver_x = earth_radius * np.cos(receiver_angle)
        receiver_y = earth_radius * np.sin(receiver_angle)
        ax.plot(receiver_x, receiver_y, 'b^', markersize=15,
               markeredgecolor='black', markeredgewidth=1, label='Receiver')

    def _add_distance_arc(self, ax, distance_deg, earth_radius):
        """Add distance arc on surface."""
        receiver_angle = distance_deg * np.pi / 180.0
        arc_angles = np.linspace(0, receiver_angle, 50)
        arc_x = earth_radius * np.cos(arc_angles)
        arc_y = earth_radius * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, 'k-', linewidth=3, alpha=0.7)

        # Add distance label
        mid_angle = receiver_angle / 2
        label_x = (earth_radius + 300) * np.cos(mid_angle)
        label_y = (earth_radius + 300) * np.sin(mid_angle)
        ax.text(label_x, label_y, f'{distance_deg}°', fontsize=14,
               ha='center', va='center', fontweight='bold')

    def _format_earth_plot(self, ax, source_depth, distance_deg, earth_radius):
        """Format the Earth plot."""
        ax.set_xlim(-earth_radius*1.1, earth_radius*1.1)
        ax.set_ylim(-earth_radius*0.15, earth_radius*1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('Height (km)', fontsize=12)
        ax.set_title(f'Seismic Ray Paths Through Earth\\n'
                    f'Source: {source_depth} km depth, Distance: {distance_deg}°, '
                    f'Model: {self.model_name.upper()}', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add center annotation
        ax.text(0, -earth_radius*0.1, 'Center of Earth', ha='center',
               va='center', fontsize=10, style='italic')

    def plot_travel_time_curves(self,
                               travel_time_table: Dict,
                               source_depth: float,
                               fig_size: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot travel time curves.

        Parameters
        ----------
        travel_time_table : Dict
            Travel time table from TravelTimeCalculator
        source_depth : float
            Source depth in km
        fig_size : Tuple[int, int]
            Figure size

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=fig_size)

        distances = travel_time_table['distances']

        # Plot each phase
        for phase, times in travel_time_table.items():
            if phase != 'distances':
                # Remove NaN values for plotting
                valid_mask = ~np.isnan(times)
                if np.any(valid_mask):
                    ax.plot(distances[valid_mask], times[valid_mask],
                           'o-', label=phase, markersize=4, linewidth=2)

        ax.set_xlabel('Distance (degrees)', fontsize=12)
        ax.set_ylabel('Travel Time (seconds)', fontsize=12)
        ax.set_title(f'Travel Time Curves\\n'
                    f'Source depth: {source_depth} km, Model: {self.model_name.upper()}',
                    fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_pierce_points(self,
                          pierce_points: Dict,
                          pierce_depths: List[float],
                          fig_size: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot pierce point distributions.

        Parameters
        ----------
        pierce_points : Dict
            Pierce points from RayPathTracer
        pierce_depths : List[float]
            List of pierce depths
        fig_size : Tuple[int, int]
            Figure size

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object
        """
        fig, axes = plt.subplots(len(pierce_depths), 1, figsize=fig_size,
                                sharex=True)

        if len(pierce_depths) == 1:
            axes = [axes]

        colors = ['blue', 'red', 'green', 'purple']

        for i, depth in enumerate(pierce_depths):
            ax = axes[i]

            for j, (phase, phase_pierces) in enumerate(pierce_points.items()):
                if depth in phase_pierces and phase_pierces[depth]:
                    pierces = phase_pierces[depth]
                    distances = [p['distance_deg'] for p in pierces]
                    times = [p['time'] for p in pierces]

                    color = colors[j % len(colors)]
                    ax.scatter(distances, times, c=color, label=phase, s=50)

            ax.set_ylabel('Time (s)', fontsize=11)
            ax.set_title(f'Pierce Points at {depth} km depth', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Distance (degrees)', fontsize=12)
        fig.suptitle(f'Pierce Point Analysis - Model: {self.model_name.upper()}',
                    fontsize=14)

        return fig

    def plot_model_comparison(self,
                             comparison_data: Dict,
                             fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot comparison between different Earth models.

        Parameters
        ----------
        comparison_data : Dict
            Comparison data from TravelTimeCalculator
        fig_size : Tuple[int, int]
            Figure size

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=fig_size)

        models = list(comparison_data.keys())
        times = [comparison_data[model] for model in models]

        # Create bar plot
        bars = ax.bar(models, times, alpha=0.7)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            if not np.isnan(time):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{time:.2f}s', ha='center', va='bottom')

        ax.set_ylabel('Travel Time (seconds)', fontsize=12)
        ax.set_title('Travel Time Comparison Between Earth Models', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        return fig
