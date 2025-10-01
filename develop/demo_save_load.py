#!/usr/bin/env python3
"""
Demo script showing PlanetMesh save/load functionality via create_mesh.
"""

from sensray import PlanetModel
import os
import tempfile

def demo_save_load():
    """Demonstrate the save/load workflow."""

    # Create a temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_path = os.path.join(temp_dir, "demo_mesh")

        print("=== Creating and saving mesh ===")

        # Create model and generate mesh
        model = PlanetModel.from_standard_model('prem')

        # Generate a small mesh for demo
        mesh = model.create_mesh(
            mesh_type="octree",
            base_cells=32,  # Small for demo
            buffer_km=500.0
        )

        print(f"Generated mesh: {mesh.mesh.n_cells} cells, {mesh.mesh.n_points} points")

        # Populate some properties
        mesh.populate_properties(['vp', 'vs'])
        print(f"Populated properties: {list(mesh.mesh.cell_data.keys())}")

        # Save the mesh
        mesh.save(mesh_path)

        print(f"\n=== Loading mesh from file ===")

        # Create a new model instance and load the saved mesh
        model2 = PlanetModel.from_standard_model('prem')

        # Load mesh using create_mesh with from_file parameter
        loaded_mesh = model2.create_mesh(from_file=mesh_path)

        print(f"Loaded mesh: {loaded_mesh.mesh.n_cells} cells, {loaded_mesh.mesh.n_points} points")
        print(f"Loaded properties: {list(loaded_mesh.mesh.cell_data.keys())}")

        # Verify the meshes have the same structure
        assert mesh.mesh.n_cells == loaded_mesh.mesh.n_cells
        assert mesh.mesh.n_points == loaded_mesh.mesh.n_points
        assert set(mesh.mesh.cell_data.keys()) == set(loaded_mesh.mesh.cell_data.keys())

        print("\n✅ Save/load workflow completed successfully!")
        print("\nUsage patterns:")
        print("1. Generate and save: mesh = model.create_mesh('octree', base_cells=64)")
        print("                      mesh.save('my_mesh')")
        print("2. Load from file:    mesh = model.create_mesh(from_file='my_mesh')")

if __name__ == "__main__":
    demo_save_load()