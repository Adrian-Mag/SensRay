#!/usr/bin/env python3
"""
Test the new discontinuity-aware mesh generation.
"""

import os
import sys

# Add the package to the path
sys.path.insert(0, '/home/adrian/PhD/masters/SensRay')

from sensray import PlanetModel
from sensray.planet_mesh import PlanetMesh

def test_mesh_generation():
    """Test the new tetrahedral mesh generation with discontinuities."""
    print("Loading PREM model...")
    model = PlanetModel.from_standard_model("prem")

    print(f"Model radius: {model.radius:.1f} km")

    # Get discontinuities
    discontinuities = model.get_discontinuities(as_depths=False)
    print(f"Discontinuities at radii: {sorted(discontinuities)}")

    # Create mesh
    print("\nCreating mesh...")
    mesh = PlanetMesh(model)

    # Generate mesh with the new method
    mesh.generate_tetrahedral_mesh(
        mesh_size_km=500.0,
        fine_size_km=300.0,
        coarse_size_km=800.0,
        transition_width_km=200.0
    )

    print(f"Mesh generated successfully!")
    print(f"Number of cells: {mesh.mesh.n_cells}")
    print(f"Number of points: {mesh.mesh.n_points}")

    # Check region IDs
    if 'region_id' in mesh.mesh.cell_data:
        region_ids = mesh.mesh.cell_data['region_id']
        unique_regions = sorted(set(region_ids))
        print(f"Unique region IDs: {unique_regions}")

        # Count cells per region
        for region_id in unique_regions:
            count = sum(1 for r in region_ids if r == region_id)
            print(f"  Region {region_id}: {count} cells")

    print("\nTest completed successfully!")

    # --- Clip the mesh with a plane and save the result ---
    try:
        import pyvista as pv
        clip_origin = (0.0, 0.0, 0.0)  # center
        clip_normal = (0.0, 1.0, 0.0)  # Y-up plane

        print("\nClipping mesh with plane (origin=%s, normal=%s)" % (clip_origin, clip_normal))
        clipped = mesh.mesh.clip(origin=clip_origin, normal=clip_normal)

        out_vtu = os.path.join(os.getcwd(), "mesh_clipped.vtu")
        print(f"Saving clipped mesh to: {out_vtu}")
        clipped.save(out_vtu)

        # Try saving a quick screenshot of the clipped mesh
        try:
            p = pv.Plotter()
            p.add_mesh(clipped, scalars="region_id" if "region_id" in clipped.cell_data else None)
            p.show()
        except Exception as e:
            print(f"Warning: could not save screenshot: {e}")
    except Exception as e:
        print(f"Warning: clipping requires pyvista: {e}")

if __name__ == "__main__":
    test_mesh_generation()