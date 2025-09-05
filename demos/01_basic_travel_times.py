#!/usr/bin/env python3
"""
Demo 1: Basic Travel Time Calculations

This demo shows how to:
- Calculate travel times for different phases
- Plot travel time curves
- Compare P and S wave arrivals
"""

import numpy as np
import matplotlib.pyplot as plt
from seisray import TravelTimeCalculator

def main():
    print("Demo 1: Basic Travel Time Calculations")
    print("=" * 50)

    # Initialize travel time calculator
    calc = TravelTimeCalculator('iasp91')

    # Source parameters
    source_depth = 10  # km
    distances = np.linspace(5, 100, 20)  # degrees

    # Calculate travel times for different distances
    p_times = []
    s_times = []
    p_distances = []
    s_distances = []

    print("Calculating travel times...")
    for distance in distances:
        arrivals = calc.calculate_travel_times(source_depth, distance)

        for arrival in arrivals:
            if arrival.name == 'P':
                p_times.append(arrival.time)
                p_distances.append(distance)
            elif arrival.name == 'S':
                s_times.append(arrival.time)
                s_distances.append(distance)

    print(f"Found {len(p_times)} P arrivals and {len(s_times)} S arrivals")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Travel time curves
    ax1.plot(p_distances, p_times, 'b-', linewidth=2, label='P waves')
    ax1.plot(s_distances, s_times, 'r-', linewidth=2, label='S waves')
    ax1.set_xlabel('Distance (degrees)')
    ax1.set_ylabel('Travel Time (seconds)')
    ax1.set_title('Travel Time Curves (IASP91 model)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: S-P time differences
    # Interpolate to common distance grid for comparison
    common_distances = np.linspace(10, 90, 17)
    p_interp = np.interp(common_distances, p_distances, p_times)
    s_interp = np.interp(common_distances, s_distances, s_times)
    sp_diff = s_interp - p_interp

    ax2.plot(common_distances, sp_diff, 'g-', linewidth=2, marker='o')
    ax2.set_xlabel('Distance (degrees)')
    ax2.set_ylabel('S-P Time Difference (seconds)')
    ax2.set_title('S-P Time Differences')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demos/01_basic_travel_times.png', dpi=150, bbox_inches='tight')
    print("Saved plot as 'demos/01_basic_travel_times.png'")

    # Print some specific examples
    print("\nExample calculations:")
    for dist in [30, 60, 90]:
        arrivals = calc.calculate_travel_times(source_depth, dist)
        print(f"\nDistance: {dist}°")
        for arrival in arrivals:
            print(f"  {arrival.name}: {arrival.time:.1f} seconds")

    plt.show()

if __name__ == "__main__":
    main()
