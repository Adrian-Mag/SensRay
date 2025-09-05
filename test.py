# Python code to compute travel times using ObsPy's TauP module
# This demonstrates how to calculate travel times between source and
# receiver locations using 1D Earth models (spherically symmetric Earth models)

import numpy as np
import matplotlib.pyplot as plt
import math
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, locations2degrees


def calculate_travel_times_geographic():
    """
    Calculate travel times using geographic coordinates (lat, lon, depth).
    This is the most common use case for real seismological applications.
    """
    print("=== Geographic Coordinate Travel Time Calculation ===")

    # Load a standard 1D Earth model
    model = TauPyModel(model="iasp91")  # or "prem", "ak135"

    # Define source location (earthquake)
    source_lat = 35.0      # degrees North
    source_lon = 140.0     # degrees East
    source_depth = 30.0    # km below surface

    # Define receiver location (seismic station)
    receiver_lat = 40.0    # degrees North
    receiver_lon = 145.0   # degrees East
    receiver_depth = 0.0   # km (surface station)

    # Calculate distance and azimuth between source and receiver
    distance_m, azimuth, back_azimuth = gps2dist_azimuth(
        source_lat, source_lon, receiver_lat, receiver_lon
    )
    distance_km = distance_m / 1000.0
    distance_deg = locations2degrees(source_lat, source_lon,
                                   receiver_lat, receiver_lon)

    print(f"Source: {source_lat}°N, {source_lon}°E, {source_depth} km depth")
    print(f"Receiver: {receiver_lat}°N, {receiver_lon}°E, {receiver_depth} km depth")
    print(f"Distance: {distance_km:.1f} km ({distance_deg:.2f}°)")
    print(f"Azimuth: {azimuth:.1f}°")

    # Calculate travel times for different seismic phases
    phases = ["P", "S", "PP", "SS", "PcP", "ScS", "PKP", "SKS"]

    arrivals = model.get_travel_times(
        source_depth_in_km=source_depth,
        distance_in_degree=distance_deg,
        phase_list=phases
    )

    print("\nTravel times:")
    print("Phase    Time (s)   Ray Parameter (s/deg)")
    print("-" * 40)
    for arrival in arrivals:
        print(f"{arrival.name:<8} {arrival.time:<8.2f} {arrival.ray_param:<12.4f}")

    return arrivals, model, distance_deg, source_depth


def calculate_travel_times_simple():
    """
    Simple travel time calculation using distance and depth directly.
    """
    print("\n=== Simple Distance-Depth Travel Time Calculation ===")

    # Load Earth model
    model = TauPyModel(model="iasp91")

    # Simple parameters
    source_depth = 15.0  # km
    distance_deg = 25.0  # degrees

    # Calculate for P and S waves
    arrivals = model.get_travel_times(
        source_depth_in_km=source_depth,
        distance_in_degree=distance_deg,
        phase_list=["P", "S"]
    )

    print(f"Source depth: {source_depth} km")
    print(f"Distance: {distance_deg}°")
    print("\nArrival times:")
    for arrival in arrivals:
        print(f"{arrival.name}: {arrival.time:.2f} seconds")

    return arrivals


def plot_ray_paths():
    """
    Calculate and plot ray paths through a circular Earth slice.
    """
    print("\n=== Ray Path Calculation and Plotting ===")

    model = TauPyModel(model="iasp91")

    source_depth = 100.0  # km
    distance_deg = 60.0   # degrees

    # Get ray paths for different phases
    ray_paths = model.get_ray_paths(
        source_depth_in_km=source_depth,
        distance_in_degree=distance_deg,
        phase_list=["P", "S", "PP", "SS"]
    )

    # Create figure with circular Earth cross-section
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot 1: Circular Earth cross-section with ray paths
    earth_radius = 6371.0  # km

    # Create circular Earth boundaries
    theta_earth = np.linspace(0, np.pi, 180)  # Half circle (0 to 180 degrees)

    # Earth's surface
    x_surface = earth_radius * np.cos(theta_earth)
    y_surface = earth_radius * np.sin(theta_earth)
    ax1.plot(x_surface, y_surface, 'k-', linewidth=2, label='Surface')

    # Core-mantle boundary (CMB)
    cmb_radius = 3480.0
    x_cmb = cmb_radius * np.cos(theta_earth)
    y_cmb = cmb_radius * np.sin(theta_earth)
    ax1.plot(x_cmb, y_cmb, 'r--', linewidth=1.5, alpha=0.7, label='Core-Mantle')

    # Inner core boundary
    ic_radius = 1220.0
    x_ic = ic_radius * np.cos(theta_earth)
    y_ic = ic_radius * np.sin(theta_earth)
    ax1.plot(x_ic, y_ic, 'orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Inner Core')

    # Plot ray paths in polar coordinates
    colors = ['blue', 'red', 'green', 'purple', 'brown', 'pink']
    for i, ray_path in enumerate(ray_paths):
        path = ray_path.path

        # Convert ray path to Cartesian coordinates
        distances_rad = path['dist']  # in radians
        depths = path['depth']
        radius = earth_radius - depths

        # Convert to Cartesian coordinates for the cross-section
        x_ray = radius * np.cos(distances_rad)
        y_ray = radius * np.sin(distances_rad)

        color = colors[i % len(colors)]
        ax1.plot(x_ray, y_ray, color=color, linewidth=2,
                label=f"{ray_path.name} ({ray_path.time:.1f}s)")

    # Mark source and receiver
    source_radius = earth_radius - source_depth
    source_x = source_radius * np.cos(0)  # At 0 degrees
    source_y = source_radius * np.sin(0)
    ax1.plot(source_x, source_y, 'r*', markersize=15, label='Source')

    receiver_radius = earth_radius  # At surface
    receiver_angle = distance_deg * np.pi / 180.0  # Convert to radians
    receiver_x = receiver_radius * np.cos(receiver_angle)
    receiver_y = receiver_radius * np.sin(receiver_angle)
    ax1.plot(receiver_x, receiver_y, 'b^', markersize=10, label='Receiver')

    # Fill Earth regions with proper circular regions
    # Create full circles for proper filling
    theta_full = np.linspace(0, 2*3.14159, 360)

    # Fill atmosphere (above surface)
    atmosphere_radius = earth_radius + 500
    x_atm = atmosphere_radius * np.cos(theta_full)
    y_atm = atmosphere_radius * np.sin(theta_full)
    ax1.fill(x_atm, y_atm, color='lightblue', alpha=0.2, label='Atmosphere')

    # Fill mantle (between surface and CMB)
    x_surf_full = earth_radius * np.cos(theta_full)
    y_surf_full = earth_radius * np.sin(theta_full)
    ax1.fill(x_surf_full, y_surf_full, color='saddlebrown', alpha=0.4,
             label='Mantle')

    # Fill outer core (between CMB and inner core)
    x_cmb_full = cmb_radius * np.cos(theta_full)
    y_cmb_full = cmb_radius * np.sin(theta_full)
    ax1.fill(x_cmb_full, y_cmb_full, color='red', alpha=0.5,
             label='Outer Core')

    # Fill inner core
    x_ic_full = ic_radius * np.cos(theta_full)
    y_ic_full = ic_radius * np.sin(theta_full)
    ax1.fill(x_ic_full, y_ic_full, color='gold', alpha=0.6, label='Inner Core')

    ax1.set_xlim(-7000, 7000)
    ax1.set_ylim(-1000, 7000)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    ax1.set_title(f'Ray Paths Through Earth\n(Source: {source_depth}km, Distance: {distance_deg}°)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Travel time vs distance curves
    distances = np.linspace(5, 180, 36)
    phases = ["P", "S", "PP"]

    for phase in phases:
        times = []
        valid_distances = []

        for dist in distances:
            try:
                arrivals = model.get_travel_times(
                    source_depth_in_km=source_depth,
                    distance_in_degree=dist,
                    phase_list=[phase]
                )
                if arrivals:
                    times.append(arrivals[0].time)
                    valid_distances.append(dist)
            except:
                continue

        if times:
            ax2.plot(valid_distances, times, 'o-', label=phase, markersize=3)

    ax2.set_xlabel('Distance (degrees)')
    ax2.set_ylabel('Travel Time (seconds)')
    ax2.set_title(f'Travel Time Curves (depth = {source_depth} km)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return ray_paths


def compare_earth_models():
    """
    Compare travel times using different Earth models.
    """
    print("\n=== Comparison of Different Earth Models ===")

    models = ["iasp91", "prem", "ak135"]
    source_depth = 50.0
    distance_deg = 40.0
    phase = "P"

    print(f"Comparing P-wave travel times:")
    print(f"Source depth: {source_depth} km, Distance: {distance_deg}°")
    print("\nModel     Travel Time (s)")
    print("-" * 25)

    for model_name in models:
        try:
            model = TauPyModel(model=model_name)
            arrivals = model.get_travel_times(
                source_depth_in_km=source_depth,
                distance_in_degree=distance_deg,
                phase_list=[phase]
            )

            if arrivals:
                time = arrivals[0].time
                print(f"{model_name:<8} {time:>8.2f}")
            else:
                print(f"{model_name:<8} {'No arrival':>8}")
        except Exception as e:
            print(f"{model_name:<8} {'Error':>8}")


def calculate_pierce_points():
    """
    Calculate where rays pierce specific depth levels.
    Useful for understanding ray sampling in tomography.
    """
    print("\n=== Ray Pierce Point Calculation ===")

    model = TauPyModel(model="iasp91")

    source_depth = 10.0
    distance_deg = 50.0
    pierce_depth = 400.0  # km - upper mantle

    ray_paths = model.get_ray_paths(
        source_depth_in_km=source_depth,
        distance_in_degree=distance_deg,
        phase_list=["P", "S"]
    )

    print(f"Pierce points at {pierce_depth} km depth:")
    print("Phase  Pierce Distance (deg)  Pierce Time (s)")
    print("-" * 45)

    for ray_path in ray_paths:
        # Get the ray path points
        path = ray_path.path

        # Find where the ray crosses the target depth
        depths = path['depth']
        distances = path['dist'] * 180.0 / 3.14159  # convert to degrees
        times = path['time']

        # Find pierce points (where ray crosses target depth)
        pierce_indices = []
        for i in range(len(depths) - 1):
            if (depths[i] <= pierce_depth <= depths[i+1] or
                depths[i+1] <= pierce_depth <= depths[i]):
                pierce_indices.append(i)

        if pierce_indices:
            # Take the first pierce point (downgoing)
            idx = pierce_indices[0]
            # Linear interpolation for more accurate pierce point
            f = (pierce_depth - depths[idx]) / (depths[idx+1] - depths[idx])
            pierce_dist = distances[idx] + f * (distances[idx+1] - distances[idx])
            pierce_time = times[idx] + f * (times[idx+1] - times[idx])

            print(f"{ray_path.name:<6} {pierce_dist:>12.2f}          {pierce_time:>8.2f}")
        else:
            print(f"{ray_path.name:<6} {'No pierce':>12}")


if __name__ == "__main__":
    # Run all examples
    arrivals, model, distance_deg, source_depth = calculate_travel_times_geographic()

    calculate_travel_times_simple()

    plot_ray_paths()

    compare_earth_models()

    calculate_pierce_points()

    print("\n=== Summary ===")
    print("This code demonstrates ObsPy's capabilities for:")
    print("1. Travel time calculation with geographic coordinates")
    print("2. Simple distance-based calculations")
    print("3. Ray path plotting and visualization")
    print("4. Comparison between different Earth models")
    print("5. Pierce point calculations for tomography applications")
    print("\nFor custom 1D models, you can create your own .nd files")
    print("and load them with TauPyModel(model='path/to/your/model.nd')")
