#!/usr/bin/env python3
"""
Simple example of computing travel times using ObsPy's TauP module.
This demonstrates the most common use cases for seismic travel time calculations.
"""

from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import numpy as np

def basic_travel_time_example():
    """Basic example of travel time calculation"""
    print("=== Basic Travel Time Calculation ===")

    # Load a standard 1D Earth model
    model = TauPyModel(model="iasp91")

    # Define source and receiver parameters
    source_depth_km = 20.0    # earthquake depth in km
    distance_degrees = 30.0   # epicentral distance in degrees

    # Calculate travel times for P and S waves
    arrivals = model.get_travel_times(
        source_depth_in_km=source_depth_km,
        distance_in_degree=distance_degrees,
        phase_list=["P", "S", "PP", "SS"]
    )

    print(f"Source depth: {source_depth_km} km")
    print(f"Distance: {distance_degrees}°")
    print("\nPhase arrivals:")
    for arrival in arrivals:
        print(f"{arrival.name:>4}: {arrival.time:7.2f} seconds")

    return arrivals

def geographic_coordinates_example():
    """Calculate travel times using lat/lon coordinates"""
    print("\n=== Geographic Coordinates Example ===")

    model = TauPyModel(model="iasp91")

    # Earthquake location (e.g., Japan)
    eq_lat, eq_lon = 35.0, 140.0
    eq_depth = 50.0  # km

    # Station location (e.g., California)
    sta_lat, sta_lon = 37.0, -122.0

    # Calculate distance
    distance_deg = locations2degrees(eq_lat, eq_lon, sta_lat, sta_lon)

    # Get travel times
    arrivals = model.get_travel_times(
        source_depth_in_km=eq_depth,
        distance_in_degree=distance_deg,
        phase_list=["P", "S"]
    )

    print(f"Earthquake: {eq_lat}°N, {eq_lon}°E, {eq_depth} km depth")
    print(f"Station: {sta_lat}°N, {sta_lon}°E")
    print(f"Distance: {distance_deg:.1f}°")
    print("\nTravel times:")
    for arrival in arrivals:
        print(f"{arrival.name}: {arrival.time:.1f} seconds")

def travel_time_curve():
    """Create a simple travel time curve"""
    print("\n=== Travel Time Curve ===")

    model = TauPyModel(model="iasp91")
    source_depth = 10.0  # km

    # Calculate travel times for different distances
    distances = np.arange(10, 100, 10)  # 10° to 90° in 10° steps

    print(f"Travel times for P waves (source depth: {source_depth} km)")
    print("Distance (°)  Time (s)")
    print("-" * 20)

    for dist in distances:
        arrivals = model.get_travel_times(
            source_depth_in_km=source_depth,
            distance_in_degree=dist,
            phase_list=["P"]
        )

        if arrivals:
            time = arrivals[0].time
            print(f"{dist:>8.0f}    {time:>7.1f}")

def compare_models():
    """Compare travel times between different Earth models"""
    print("\n=== Model Comparison ===")

    models = ["iasp91", "prem", "ak135"]
    source_depth = 100.0  # km
    distance = 60.0       # degrees

    print(f"P-wave travel times (depth={source_depth}km, dist={distance}°)")
    print("Model    Time (s)")
    print("-" * 16)

    for model_name in models:
        model = TauPyModel(model=model_name)
        arrivals = model.get_travel_times(
            source_depth_in_km=source_depth,
            distance_in_degree=distance,
            phase_list=["P"]
        )

        if arrivals:
            time = arrivals[0].time
            print(f"{model_name:<8} {time:>6.2f}")

if __name__ == "__main__":
    # Run all examples
    basic_travel_time_example()
    geographic_coordinates_example()
    travel_time_curve()
    compare_models()

    print("\n" + "="*50)
    print("SUMMARY: Key ObsPy TauP Functions")
    print("="*50)
    print("1. TauPyModel(model='iasp91') - Load Earth model")
    print("2. get_travel_times() - Calculate arrival times")
    print("3. get_ray_paths() - Get ray paths through Earth")
    print("4. locations2degrees() - Convert lat/lon to distance")
    print("\nAvailable models: iasp91, prem, ak135, etc.")
    print("Common phases: P, S, PP, SS, PcP, ScS, PKP, SKS")
