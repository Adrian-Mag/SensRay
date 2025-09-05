#!/usr/bin/env python3
"""
Simple circular Earth ray plotting example.
Easy to customize for different scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel


def simple_circular_earth_plot(source_depth=50, distance=45, phases=["P", "S"]):
    """
    Simple function to create a circular Earth ray plot.

    Parameters:
    source_depth: depth in km
    distance: epicentral distance in degrees
    phases: list of seismic phases to plot
    """

    # Load Earth model
    model = TauPyModel(model="iasp91")

    # Get ray paths
    ray_paths = model.get_ray_paths(
        source_depth_in_km=source_depth,
        distance_in_degree=distance,
        phase_list=phases
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Earth parameters
    earth_radius = 6371
    core_radius = 3480
    inner_core_radius = 1220

    # Create semicircle for Earth boundaries
    angles = np.linspace(0, np.pi, 100)

    # Plot Earth structure
    earth_x = earth_radius * np.cos(angles)
    earth_y = earth_radius * np.sin(angles)
    ax.plot(earth_x, earth_y, 'k-', linewidth=3, label='Surface')

    core_x = core_radius * np.cos(angles)
    core_y = core_radius * np.sin(angles)
    ax.plot(core_x, core_y, 'r--', linewidth=2, label='Outer Core')

    inner_x = inner_core_radius * np.cos(angles)
    inner_y = inner_core_radius * np.sin(angles)
    ax.plot(inner_x, inner_y, 'orange', linestyle='--', linewidth=2,
            label='Inner Core')

    # Fill regions
    ax.fill_between(earth_x, earth_y, core_x, color='brown', alpha=0.3,
                    label='Mantle')
    ax.fill_between(core_x, core_y, inner_x, color='red', alpha=0.4,
                    label='Outer Core')
    ax.fill_between(inner_x, inner_y, 0, color='yellow', alpha=0.5,
                    label='Inner Core')

    # Plot ray paths
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, ray_path in enumerate(ray_paths):
        # Get path coordinates
        path_dist = ray_path.path['dist']  # radians
        path_depth = ray_path.path['depth']  # km
        path_radius = earth_radius - path_depth

        # Convert to x,y coordinates
        x = path_radius * np.cos(path_dist)
        y = path_radius * np.sin(path_dist)

        # Plot ray
        color = colors[i % len(colors)]
        ax.plot(x, y, color=color, linewidth=3,
                label=f"{ray_path.name} ({ray_path.time:.1f}s)")

    # Mark source and receiver
    source_radius = earth_radius - source_depth
    ax.plot(source_radius, 0, 'r*', markersize=20,
            markeredgecolor='black', label='Source')

    receiver_angle = distance * np.pi / 180
    receiver_x = earth_radius * np.cos(receiver_angle)
    receiver_y = earth_radius * np.sin(receiver_angle)
    ax.plot(receiver_x, receiver_y, 'b^', markersize=15,
            markeredgecolor='black', label='Receiver')

    # Format plot
    ax.set_xlim(-7000, 7000)
    ax.set_ylim(-1000, 7000)
    ax.set_aspect('equal')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title(f'Ray Paths: {source_depth}km depth, {distance}° distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return ray_paths


if __name__ == "__main__":
    # Example usage
    print("Creating circular Earth ray plot...")

    # Simple P and S waves
    simple_circular_earth_plot(source_depth=50, distance=45, phases=["P", "S"])

    # Deep source with core phases
    simple_circular_earth_plot(source_depth=600, distance=120,
                              phases=["P", "PKP", "PKIKP"])

    # Shallow source, short distance
    simple_circular_earth_plot(source_depth=10, distance=20,
                              phases=["P", "S", "PP", "SS"])
