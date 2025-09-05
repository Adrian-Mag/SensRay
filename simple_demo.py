#!/usr/bin/env python3
"""
Simple demonstration of the seisray package functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from seisray import (TravelTimeCalculator, RayPathTracer, EarthPlotter,
                     EarthModelManager)


def main():
    """Demonstrate basic seisray package functionality."""

    print("🌍 SeisRay Package Simple Demo")
    print("=" * 40)

    # Setup parameters
    source_depth = 10  # km
    distances = [30, 60, 90]  # degrees

    # 1. Earth Model Management
    print("\n1. Earth Model Management")
    print("-" * 30)

    manager = EarthModelManager()
    models = manager.list_available_models()
    print(f"Available models: {models}")

    # 2. Travel Time Calculations
    print("\n2. Travel Time Calculations")
    print("-" * 30)

    calc = TravelTimeCalculator('iasp91')

    for distance in distances:
        times = calc.calculate_travel_times(source_depth, distance)
        print(f"Distance {distance}°:")
        for phase in times[:3]:  # Show first 3 phases
            print(f"  {phase.name}: {phase.time:.1f} s")

    # 3. Ray Path Tracing
    print("\n3. Ray Path Tracing")
    print("-" * 30)

    tracer = RayPathTracer('iasp91')

    # Get P-wave ray paths for one distance
    rays = tracer.get_ray_paths(source_depth, 30, phases=['P'])
    print(f"Extracted {len(rays)} P-wave ray paths for 30°")

    if rays:
        ray = rays[0]
        print(f"Ray path has {len(ray.path['dist'])} points")
        print(f"Maximum depth: {max(ray.path['depth']):.1f} km")

    # 4. Simple Visualization
    print("\n4. Simple Visualization")
    print("-" * 30)

    plotter = EarthPlotter()

    # Create a simple plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot Earth circle
    circle = Circle((0, 0), 6371, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    # Plot a few ray paths
    for dist in [30, 60, 90]:
        rays = tracer.get_ray_paths(source_depth, dist, phases=['P'])
        if rays:
            ray = rays[0]
            # Convert to Cartesian coordinates for plotting
            theta = np.radians(ray.path['dist'])
            r = 6371 - ray.path['depth']
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            ax.plot(x, y, label=f'P-wave {dist}°')

    ax.set_xlim(-7000, 7000)
    ax.set_ylim(-7000, 7000)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Simple Ray Path Visualization')
    ax.grid(True, alpha=0.3)

    plt.savefig('simple_demo.png', dpi=150, bbox_inches='tight')
    print("Saved simple demonstration plot as 'simple_demo.png'")

    plt.show()

    print("\n🎉 Simple demonstration complete!")


if __name__ == "__main__":
    main()
