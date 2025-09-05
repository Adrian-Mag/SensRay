#!/usr/bin/env python3
"""
Example script demonstrating the full functionality of the seisray package.
"""

import numpy as np
import matplotlib.pyplot as plt
from seisray import (TravelTimeCalculator, RayPathTracer, EarthPlotter,
                     EarthModelManager, SensitivityKernel)

def main():
    """Demonstrate seisray package functionality."""

    print("🌍 SeisRay Package Demonstration")
    print("=" * 50)

    # Setup parameters
    source_depth = 10  # km
    distances = np.linspace(10, 90, 9)  # degrees

    # 1. Earth Model Management
    print("\n1. Earth Model Management")
    print("-" * 30)

    manager = EarthModelManager()
    models = manager.list_available_models()
    print(f"Available models: {models}")

    model_info = manager.get_model_info('iasp91')
    print(f"IASP91 model depth range: {model_info['depth_range']}")

    # 2. Travel Time Calculations
    print("\n2. Travel Time Calculations")
    print("-" * 30)

    calc = TravelTimeCalculator('iasp91')

    for distance in [30, 60, 90]:
        times = calc.calculate_travel_times(source_depth, distance)
        print(f"Distance {distance}°:")
        for phase in times:
            print(f"  {phase.name}: {phase.time:.1f} s")

    # 3. Ray Path Tracing
    print("\n3. Ray Path Tracing")
    print("-" * 30)

    tracer = RayPathTracer('iasp91')

    # Get P-wave ray paths for multiple distances
    all_rays = []
    for dist in distances:
        rays = tracer.get_ray_paths(source_depth, dist, phases=['P'])
        if rays:
            all_rays.extend(rays)

    print(f"Extracted {len(all_rays)} P-wave ray paths")

    # Analyze pierce points
    pierce_points = tracer.get_pierce_points(all_rays[0], depths=[410, 660])
    print(f"Pierce points at 410 and 660 km: {len(pierce_points)} points")

    # 4. Visualization
    print("\n4. Visualization")
    print("-" * 30)

    plotter = EarthPlotter()

    # Plot 1: Earth cross-section with ray paths
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Earth with rays
    ax = axes[0, 0]
    plotter.plot_earth_with_rays(all_rays, source_depth, ax=ax)
    ax.set_title('P-wave Ray Paths')

    # Travel time curves
    ax = axes[0, 1]
    all_times = []
    for dist in distances:
        times = calc.calculate_travel_times(source_depth, dist)
        all_times.extend(times)

    plotter.plot_travel_time_curves(all_times, ax=ax)
    ax.set_title('Travel Time Curves')

    # Pierce point density
    ax = axes[1, 0]
    pierce_depth = 400
    density = tracer.get_pierce_point_density(all_rays, pierce_depth,
                                            distance_bins=20)
    plotter.plot_pierce_point_density(density, pierce_depth, ax=ax)
    ax.set_title(f'Pierce Point Density at {pierce_depth} km')

    # Model comparison
    ax = axes[1, 1]
    models_to_compare = ['iasp91', 'prem']
    plotter.compare_models(models_to_compare, source_depth, distances,
                          phase='P', ax=ax)
    ax.set_title('Model Comparison')

    plt.tight_layout()
    plt.savefig('seisray_demo.png', dpi=150, bbox_inches='tight')
    print("Saved demonstration plot as 'seisray_demo.png'")

    # 5. Sensitivity Kernels
    print("\n5. Sensitivity Kernels")
    print("-" * 30)

    # Create a simple sensitivity kernel
    kernel = SensitivityKernel('iasp91')

    # Define a simple velocity grid
    depth_grid = np.linspace(0, 800, 41)
    distance_grid = np.linspace(0, 90, 46)

    # Compute kernel for P-wave at 30 degrees
    kernel_matrix = kernel.compute_kernel(source_depth=10, distance_deg=30,
                                        phase='P', depth_grid=depth_grid,
                                        distance_grid=distance_grid)

    print(f"Computed kernel matrix shape: {kernel_matrix.shape}")
    print(f"Kernel matrix sparsity: {np.count_nonzero(kernel_matrix) / kernel_matrix.size:.1%}")

    # Plot kernel
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(kernel_matrix, aspect='auto', origin='lower',
                   extent=[distance_grid[0], distance_grid[-1],
                          depth_grid[0], depth_grid[-1]])
    ax.set_xlabel('Distance (degrees)')
    ax.set_ylabel('Depth (km)')
    ax.set_title('P-wave Sensitivity Kernel')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='Sensitivity')
    plt.savefig('sensitivity_kernel.png', dpi=150, bbox_inches='tight')
    print("Saved sensitivity kernel plot as 'sensitivity_kernel.png'")

    plt.show()

    print("\n🎉 Demonstration complete!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
