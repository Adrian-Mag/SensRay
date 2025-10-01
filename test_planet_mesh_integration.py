#!/usr/bin/env python3
"""
Test script for PlanetMesh integration.

Tests basic functionality:
- Loading PlanetModel
- Creating PlanetMesh
- Generating octree mesh
- Populating properties
- Computing ray lengths from ObsPy rays
- Basic visualization
"""

import sys
import os

# Add the sensray directory to path
sys.path.insert(0, '/disks/data/PhD/masters/SensRay')

try:
    from sensray import PlanetModel
    print("✓ Successfully imported PlanetModel")
except ImportError as e:
    print(f"✗ Failed to import PlanetModel: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic PlanetMesh functionality."""

    print("\n=== Testing Basic PlanetMesh Functionality ===")

    # 1. Load model
    try:
        print("1. Loading PREM model...")
        model = PlanetModel.from_standard_model("prem")
        print(f"   ✓ Loaded model: {model.name}")
        print(f"   ✓ Radius: {model.radius:.1f} km")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # 2. Access mesh property
    try:
        print("2. Accessing mesh property...")
        mesh = model.mesh
        print(f"   ✓ Mesh object created: {type(mesh).__name__}")
        print(f"   ✓ Mesh type: {mesh.mesh_type}")
    except Exception as e:
        print(f"   ✗ Failed to access mesh: {e}")
        return False

    # 3. Generate octree mesh
    try:
        print("3. Generating octree mesh...")
        mesh.generate_octree_mesh(base_cells=64, buffer_km=1000.0)  # Small for testing
        print(f"   ✓ Generated mesh with {mesh.mesh.n_cells} cells")
    except Exception as e:
        print(f"   ✗ Failed to generate mesh: {e}")
        return False

    # 4. Populate properties
    try:
        print("4. Populating properties...")
        mesh.populate_properties(["vp", "vs"])
        print("   ✓ Populated vp and vs properties")

        # Check property ranges
        vp_range = (mesh.mesh.cell_data["vp"].min(), mesh.mesh.cell_data["vp"].max())
        vs_range = (mesh.mesh.cell_data["vs"].min(), mesh.mesh.cell_data["vs"].max())
        print(f"   ✓ Vp range: {vp_range[0]:.2f} - {vp_range[1]:.2f} km/s")
        print(f"   ✓ Vs range: {vs_range[0]:.2f} - {vs_range[1]:.2f} km/s")
    except Exception as e:
        print(f"   ✗ Failed to populate properties: {e}")
        return False

    # 5. Test ray computation
    try:
        print("5. Testing ray computation...")
        # Get a ray using ObsPy
        rays = model.taupy_model.get_ray_paths_geo(
            source_depth_in_km=100,
            source_latitude_in_deg=0,
            source_longitude_in_deg=0,
            receiver_latitude_in_deg=30,
            receiver_longitude_in_deg=30,
            phase_list=["P"]
        )

        if rays:
            p_ray = rays[0]
            print(f"   ✓ Got P ray with {len(p_ray.path)} path points")

            # Compute per-cell lengths
            lengths = mesh.compute_ray_lengths(p_ray)
            total_length = lengths.sum()
            non_zero_cells = (lengths > 0).sum()

            print(f"   ✓ Ray intersects {non_zero_cells} cells")
            print(f"   ✓ Total path length: {total_length:.1f} km")
        else:
            print("   ⚠ No rays returned from TauP")

    except Exception as e:
        print(f"   ✗ Failed ray computation: {e}")
        return False

    # 6. Test visualization setup (don't actually show)
    try:
        print("6. Testing visualization setup...")
        plotter = mesh.plot_cross_section(
            plane_normal=(0, 1, 0),
            property_name="vp"
        )
        print("   ✓ Created cross-section plotter")
        plotter.close()  # Don't show, just test creation

    except Exception as e:
        print(f"   ✗ Failed visualization setup: {e}")
        return False

    print("\n✓ All basic tests passed!")
    return True

def test_workflow_example():
    """Test the intended user workflow."""

    print("\n=== Testing User Workflow Example ===")

    try:
        # Simple workflow
        print("Creating model and mesh...")
        model = PlanetModel.from_standard_model("prem")

        # Alternative: explicit mesh creation
        mesh = model.create_mesh(
            mesh_type="octree",
            base_cells=32,  # Very small for testing
            buffer_km=500
        )

        print("Populating properties...")
        mesh.populate_properties(["vp"])

        print("Getting rays...")
        rays = model.taupy_model.get_ray_paths_geo(
            source_depth_in_km=50,
            source_latitude_in_deg=0, source_longitude_in_deg=0,
            receiver_latitude_in_deg=45, receiver_longitude_in_deg=45,
            phase_list=["P", "S"]
        )

        if len(rays) >= 2:
            print("Computing ray lengths...")
            p_lengths = mesh.compute_ray_lengths(rays[0])  # P
            s_lengths = mesh.compute_ray_lengths(rays[1])  # S

            print(f"P wave path: {p_lengths.sum():.1f} km")
            print(f"S wave path: {s_lengths.sum():.1f} km")

        print("✓ Workflow example completed successfully!")

    except Exception as e:
        print(f"✗ Workflow example failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Testing PlanetMesh Integration")
    print("=" * 40)

    success = True
    success &= test_basic_functionality()
    success &= test_workflow_example()

    if success:
        print("\n🎉 All tests passed! PlanetMesh integration is working.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)