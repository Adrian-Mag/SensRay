"""
Earth visualization and plotting functionality.
"""

from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..utils.earth_models import EarthModelManager


class EarthPlotter:
    """Class for creating Earth visualizations and ray path plots."""

    def __init__(self, model_name: str = "iasp91") -> None:
        self.model_name = model_name
        self.model_manager = EarthModelManager()
        self.earth_structure = self.model_manager.get_earth_structure(
            model_name
        )

    def plot_circular_earth(
        self,
        ray_coordinates: Dict,
        source_depth: float,
        distance_deg: float,
        fig_size: Tuple[int, int] = (12, 10),
        show_atmosphere: bool = True,
        view: str = "upper",
        filter_to_receiver: bool = False,
        filter_tolerance_deg: float = 0.5,
    ) -> Figure:
        """Create a circular Earth cross-section with ray paths.

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
        view : str
            'upper' for upper hemisphere, 'full' for full Earth
        filter_to_receiver : bool
            If True, only plot rays whose end distance matches receiver
        filter_tolerance_deg : float
            Tolerance in degrees for matching receiver distance
        """
        fig, ax = plt.subplots(figsize=fig_size)

        # Earth structure parameters
        earth_radius = self.earth_structure["earth_radius"]
        cmb_radius = self.earth_structure["cmb_radius"]
        icb_radius = self.earth_structure["icb_radius"]

        # Angle arrays
        full = view == "full"
        theta = np.linspace(
            0.0, 2 * np.pi if full else np.pi, 360 if full else 180
        )
        theta_full = np.linspace(0.0, 2 * np.pi, 360)

        # Fill and boundaries
        self._fill_earth_layers(
            ax,
            theta_full,
            earth_radius,
            cmb_radius,
            icb_radius,
            show_atmosphere,
        )
        self._plot_earth_boundaries(
            ax, theta, earth_radius, cmb_radius, icb_radius
        )

        # Optional filtering to receiver
        coords_to_plot = ray_coordinates
        if filter_to_receiver:
            coords_to_plot = {}
            for phase_key, coords in ray_coordinates.items():
                degs = coords.get("distances_deg")
                if degs is None or len(degs) == 0:
                    continue
                end_deg = float(degs[-1]) % 360.0
                diff = min(
                    abs(end_deg - distance_deg),
                    abs((360.0 - end_deg) - distance_deg),
                )
                if diff <= filter_tolerance_deg:
                    coords_to_plot[phase_key] = coords

        self._plot_ray_paths(ax, coords_to_plot)
        self._mark_source_receiver(
            ax, source_depth, distance_deg, earth_radius
        )
        self._add_distance_arc(ax, distance_deg, earth_radius)
        self._format_earth_plot(
            ax, source_depth, distance_deg, earth_radius, view=view
        )
        return fig

    def _fill_earth_layers(
        self,
        ax: plt.Axes,
        theta_full: np.ndarray,
        earth_radius: float,
        cmb_radius: float,
        icb_radius: float,
        show_atmosphere: bool,
    ) -> None:
        # Atmosphere
        if show_atmosphere:
            atmosphere_radius = earth_radius + 500.0
            ax.fill(
                atmosphere_radius * np.cos(theta_full),
                atmosphere_radius * np.sin(theta_full),
                color="lightblue",
                alpha=0.2,
                label="Atmosphere",
            )
        # Mantle (surface disk)
        ax.fill(
            earth_radius * np.cos(theta_full),
            earth_radius * np.sin(theta_full),
            color="saddlebrown",
            alpha=0.4,
            label="Mantle",
        )
        # Outer core
        ax.fill(
            cmb_radius * np.cos(theta_full),
            cmb_radius * np.sin(theta_full),
            color="red",
            alpha=0.5,
            label="Outer Core",
        )
        # Inner core
        ax.fill(
            icb_radius * np.cos(theta_full),
            icb_radius * np.sin(theta_full),
            color="gold",
            alpha=0.6,
            label="Inner Core",
        )

    def _plot_earth_boundaries(
        self,
        ax: plt.Axes,
        theta: np.ndarray,
        earth_radius: float,
        cmb_radius: float,
        icb_radius: float,
    ) -> None:
        ax.plot(
            earth_radius * np.cos(theta),
            earth_radius * np.sin(theta),
            "k-",
            linewidth=3,
            label="Surface",
        )
        ax.plot(
            cmb_radius * np.cos(theta),
            cmb_radius * np.sin(theta),
            "r--",
            linewidth=2,
            alpha=0.8,
            label="Core-Mantle Boundary",
        )
        ax.plot(
            icb_radius * np.cos(theta),
            icb_radius * np.sin(theta),
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Inner Core Boundary",
        )

    def _plot_ray_paths(self, ax: plt.Axes, ray_coordinates: Dict) -> None:
        colors = [
            "blue", "red", "green", "purple",
            "brown", "pink", "gray", "cyan",
        ]
        for i, (phase, coords) in enumerate(ray_coordinates.items()):
            color = colors[i % len(colors)]
            label = phase
            if "total_time" in coords:
                try:
                    label = f"{phase} ({float(coords['total_time']):.1f}s)"
                except Exception:
                    label = phase
            ax.plot(
                coords["x_cartesian"],
                coords["y_cartesian"],
                color=color,
                linewidth=3,
                label=label,
            )

    def _mark_source_receiver(
        self,
        ax: plt.Axes,
        source_depth: float,
        distance_deg: float,
        earth_radius: float,
    ) -> None:
        # Source
        ax.plot(
            earth_radius - source_depth,
            0.0,
            "r*",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Source",
        )
        # Receiver on surface at distance_deg
        angle = np.deg2rad(distance_deg)
        ax.plot(
            earth_radius * np.cos(angle),
            earth_radius * np.sin(angle),
            "b^",
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Receiver",
        )

    def _add_distance_arc(
        self, ax: plt.Axes, distance_deg: float, earth_radius: float
    ) -> None:
        angle = np.deg2rad(distance_deg)
        arc_angles = np.linspace(0.0, angle, 50)
        ax.plot(
            earth_radius * np.cos(arc_angles),
            earth_radius * np.sin(arc_angles),
            "k-",
            linewidth=3,
            alpha=0.7,
        )
        mid = angle / 2.0
        ax.text(
            (earth_radius + 500.0) * np.cos(mid),
            (earth_radius + 500.0) * np.sin(mid),
            f"{distance_deg:.1f}°",
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold",
        )

    def _format_earth_plot(
        self,
        ax: plt.Axes,
        source_depth: float,
        distance_deg: float,
        earth_radius: float,
        view: str = "upper",
    ) -> None:
        ax.set_xlim(-earth_radius * 1.1, earth_radius * 1.1)
        if view == "full":
            ax.set_ylim(-earth_radius * 1.1, earth_radius * 1.1)
        else:
            ax.set_ylim(-earth_radius * 0.15, earth_radius * 1.1)
        ax.set_aspect("equal")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Height (km)")
        ax.set_title(
            "Seismic Ray Paths Through Earth\n"
            f"Source: {source_depth} km depth, Distance: {distance_deg}°, "
            f"Model: {self.model_name.upper()}\n"
            f"View: {'Full' if view=='full' else 'Upper hemisphere'}"
        )
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.0,
            -earth_radius * 0.1,
            "Center of Earth",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
        )

    def plot_travel_time_curves(
        self,
        travel_time_table: Dict,
        source_depth: float,
        fig_size: Tuple[int, int] = (12, 8),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=fig_size)
        distances = travel_time_table["distances"]
        for phase, times in travel_time_table.items():
            if phase == "distances":
                continue
            mask = ~np.isnan(times)
            if np.any(mask):
                ax.plot(
                    distances[mask],
                    times[mask],
                    "o-",
                    label=phase,
                    markersize=4,
                    linewidth=2,
                )
        ax.set_xlabel("Distance (degrees)")
        ax.set_ylabel("Travel Time (seconds)")
        ax.set_title(
            "Travel Time Curves\n"
            f"Source depth: {source_depth} km, Model: "
            f"{self.model_name.upper()}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def plot_pierce_points(
        self,
        pierce_points: Dict,
        pierce_depths: List[float],
        fig_size: Tuple[int, int] = (12, 8),
    ) -> Figure:
        fig, axes = plt.subplots(
            len(pierce_depths), 1, figsize=fig_size, sharex=True
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        colors = ["blue", "red", "green", "purple"]
        for i, depth in enumerate(pierce_depths):
            ax = axes[i]
            for j, (phase, phase_pierces) in enumerate(pierce_points.items()):
                if depth in phase_pierces and phase_pierces[depth]:
                    pts = phase_pierces[depth]
                    distances = [p["distance_deg"] for p in pts]
                    times = [p["time"] for p in pts]
                    color = colors[j % len(colors)]
                    ax.scatter(distances, times, c=color, label=phase, s=50)
            ax.set_ylabel("Time (s)")
            ax.set_title(f"Pierce Points at {depth} km depth")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Distance (degrees)")
        fig.suptitle(
            f"Pierce Point Analysis - Model: {self.model_name.upper()}"
        )
        return fig

    def plot_model_comparison(
        self,
        comparison_data: Dict,
        fig_size: Tuple[int, int] = (10, 6),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=fig_size)
        models = list(comparison_data.keys())
        times = [comparison_data[m] for m in models]
        bars = ax.bar(models, times, alpha=0.7)
        for bar, t in zip(bars, times):
            if not np.isnan(t):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{t:.2f}s",
                    ha="center",
                    va="bottom",
                )
        ax.set_ylabel("Travel Time (seconds)")
        ax.set_title("Travel Time Comparison Between Earth Models")
        ax.grid(True, alpha=0.3, axis="y")
        return fig
