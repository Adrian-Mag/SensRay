#!/usr/bin/env python3
"""
Circular Earth ray path plotting using ObsPy.
This creates a realistic circular cross-section of the Earth showing ray paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel


def plot_circular_earth_rays(source_depth=100.0, distance_deg=60.0,
                           phases=["P", "S", "PP", "SS"], model_name="iasp91"):
    """
    Plot ray paths through a circular Earth cross-section.

    Parameters:
    -----------
    source_depth : float
        Source depth in km
    distance_deg : float
        Epicentral distance in degrees
    phases : list
        List of seismic phases to plot
    model_name : str
        Earth model to use (iasp91, prem, ak135)
    """

    print(f"Plotting ray paths for {phases}")
    print(f"Source depth: {source_depth} km, Distance: {distance_deg}°")

    # Load Earth model
    model = TauPyModel(model=model_name)

    # Get ray paths
    ray_paths = model.get_ray_paths(
        source_depth_in_km=source_depth,
        distance_in_degree=distance_deg,
        phase_list=phases
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Earth parameters
    earth_radius = 6371.0  # km
    cmb_radius = 3480.0    # Core-mantle boundary
    ic_radius = 1220.0     # Inner core boundary

    # Create angular array for Earth boundaries (semicircle)
    theta = np.linspace(0, np.pi, 180)

    # Plot Earth structure
    # Surface
    x_surface = earth_radius * np.cos(theta)
    y_surface = earth_radius * np.sin(theta)
    ax.plot(x_surface, y_surface, 'k-', linewidth=3, label='Surface')

    # Core-mantle boundary
    x_cmb = cmb_radius * np.cos(theta)
    y_cmb = cmb_radius * np.sin(theta)
    ax.plot(x_cmb, y_cmb, 'red', linestyle='--', linewidth=2,
            alpha=0.8, label='Core-Mantle Boundary')

    # Inner core boundary
    x_ic = ic_radius * np.cos(theta)
    y_ic = ic_radius * np.sin(theta)
    ax.plot(x_ic, y_ic, 'orange', linestyle='--', linewidth=2,
            alpha=0.8, label='Inner Core Boundary')

    # Fill Earth layers with colors
    ax.fill_between(x_surface, y_surface, earth_radius + 500,
                    color='lightblue', alpha=0.3, label='Atmosphere')
    ax.fill_between(x_surface, y_surface, x_cmb,
                    color='saddlebrown', alpha=0.4, label='Mantle')
    ax.fill_between(x_cmb, y_cmb, x_ic,
                    color='darkred', alpha=0.5, label='Outer Core')
    ax.fill_between(x_ic, y_ic, 0,
                    color='gold', alpha=0.6, label='Inner Core')

    # Plot ray paths
    colors = ['blue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan']

    for i, ray_path in enumerate(ray_paths):
        path = ray_path.path

        # Get ray path coordinates
        distances_rad = path['dist']  # in radians
        depths = path['depth']
        radius = earth_radius - depths

        # Convert to Cartesian coordinates
        x_ray = radius * np.cos(distances_rad)
        y_ray = radius * np.sin(distances_rad)

        # Plot ray path
        color = colors[i % len(colors)]
        ax.plot(x_ray, y_ray, color=color, linewidth=2.5,
                label=f"{ray_path.name} ({ray_path.time:.1f}s)")

    # Mark source and receiver
    source_radius = earth_radius - source_depth
    source_x = source_radius * np.cos(0)
    source_y = source_radius * np.sin(0)
    ax.plot(source_x, source_y, 'r*', markersize=20,
            markeredgecolor='black', markeredgewidth=1, label='Source')

    receiver_angle = distance_deg * np.pi / 180.0
    receiver_x = earth_radius * np.cos(receiver_angle)
    receiver_y = earth_radius * np.sin(receiver_angle)
    ax.plot(receiver_x, receiver_y, 'b^', markersize=15,
            markeredgecolor='black', markeredgewidth=1, label='Receiver')

    # Add distance arc on surface
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

    # Formatting
    ax.set_xlim(-7200, 7200)
    ax.set_ylim(-1000, 7200)
    ax.set_aspect('equal')
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_title(f'Seismic Ray Paths Through Earth\n'
                f'Source: {source_depth} km depth, Distance: {distance_deg}°, '
                f'Model: {model_name.upper()}', fontsize=14)

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add some annotations
    ax.text(0, -800, 'Center of Earth', ha='center', va='center',
            fontsize=10, style='italic')

    plt.tight_layout()
    plt.show()

    return ray_paths


def plot_multiple_distances():
    """
    Plot ray paths for multiple distances on the same circular Earth.
    """
    print("\n=== Multiple Distance Ray Paths ===")

    model = TauPyModel(model="iasp91")
    source_depth = 50.0
    distances = [30, 60, 90, 120]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Earth structure (same as above)
    earth_radius = 6371.0
    cmb_radius = 3480.0
    ic_radius = 1220.0
    theta = np.linspace(0, np.pi, 180)

    # Plot Earth layers
    x_surface = earth_radius * np.cos(theta)
    y_surface = earth_radius * np.sin(theta)
    ax.plot(x_surface, y_surface, 'k-', linewidth=3)

    x_cmb = cmb_radius * np.cos(theta)
    y_cmb = cmb_radius * np.sin(theta)
    ax.plot(x_cmb, y_cmb, 'red', linestyle='--', linewidth=2, alpha=0.8)

    x_ic = ic_radius * np.cos(theta)
    y_ic = ic_radius * np.sin(theta)
    ax.plot(x_ic, y_ic, 'orange', linestyle='--', linewidth=2, alpha=0.8)

    # Fill Earth
    ax.fill_between(x_surface, y_surface, x_cmb,
                    color='saddlebrown', alpha=0.3)
    ax.fill_between(x_cmb, y_cmb, x_ic,
                    color='darkred', alpha=0.4)
    ax.fill_between(x_ic, y_ic, 0,
                    color='gold', alpha=0.5)

    colors = ['blue', 'red', 'green', 'purple']

    # Plot P-wave ray paths for different distances
    for i, dist in enumerate(distances):
        try:
            ray_paths = model.get_ray_paths(
                source_depth_in_km=source_depth,
                distance_in_degree=dist,
                phase_list=["P"]
            )

            if ray_paths:
                ray_path = ray_paths[0]  # Take first P arrival
                path = ray_path.path

                distances_rad = path['dist']
                depths = path['depth']
                radius = earth_radius - depths

                x_ray = radius * np.cos(distances_rad)
                y_ray = radius * np.sin(distances_rad)

                color = colors[i % len(colors)]
                ax.plot(x_ray, y_ray, color=color, linewidth=2.5,
                        label=f"P at {dist}° ({ray_path.time:.0f}s)")

                # Mark receiver
                receiver_angle = dist * np.pi / 180.0
                receiver_x = earth_radius * np.cos(receiver_angle)
                receiver_y = earth_radius * np.sin(receiver_angle)
                ax.plot(receiver_x, receiver_y, 'o', color=color,
                        markersize=8, markeredgecolor='black')

        except:
            continue

    # Mark source
    source_radius = earth_radius - source_depth
    ax.plot(source_radius, 0, 'r*', markersize=20,
            markeredgecolor='black', markeredgewidth=1, label='Source')

    ax.set_xlim(-7200, 7200)
    ax.set_ylim(-1000, 7200)
    ax.set_aspect('equal')
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_title(f'P-wave Ray Paths at Multiple Distances\n'
                f'Source depth: {source_depth} km', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example 1: Standard ray path plot
    plot_circular_earth_rays(source_depth=100.0, distance_deg=60.0,
                           phases=["P", "S", "PP", "SS"])

    # Example 2: Different configuration
    plot_circular_earth_rays(source_depth=200.0, distance_deg=90.0,
                           phases=["P", "PKP", "PKIKP"], model_name="prem")

    # Example 3: Multiple distances
    plot_multiple_distances()
