"""
Minimal demo for MeshEarthModel: create a tetra sphere, save, and visualize.

Run: python sensray/mesh/test_mesh_model.py
"""

from sensray.mesh.earth_model import MeshEarthModel


def main():
    # Create a coarse tetrahedral sphere for quick demo
    model = MeshEarthModel.from_pygmsh_sphere(
        radius_km=6371.0, mesh_size_km=500.0
    )

    # Save to disk
    out_path = "earth_sphere_demo.vtu"
    model.save(out_path)
    print(f"Saved mesh to {out_path}")

    # Plot
    plotter = model.plot(show_edges=True, color="lightgray", opacity=1.0)
    plotter.show()


if __name__ == "__main__":
    main()
