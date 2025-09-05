#!/usr/bin/env python3
"""
Simple test script to verify the seisray package functionality.
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from seisray import TravelTimeCalculator, RayPathTracer, EarthPlotter, EarthModelManager
    print("✓ Successfully imported all main classes")

    # Test travel time calculator
    calc = TravelTimeCalculator('iasp91')
    times = calc.calculate_travel_times(source_depth=10, distance_deg=30)
    print(f"✓ Travel time calculation works: {len(times)} phases found")

    # Test ray path tracer
    tracer = RayPathTracer('iasp91')
    rays = tracer.get_ray_paths(source_depth=10, distance_deg=30,
                                phases=['P'])
    print(f"✓ Ray path tracing works: {len(rays)} rays found")

    # Test earth model manager
    manager = EarthModelManager()
    models = manager.list_available_models()
    print(f"✓ Earth model manager works: {len(models)} models available")

    # Test plotter (just initialization)
    plotter = EarthPlotter()
    print("✓ Earth plotter initialization works")

    print("\n🎉 All basic functionality tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime error: {e}")
    sys.exit(1)
