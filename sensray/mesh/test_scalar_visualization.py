"""Demo: assign per-cell 'vs' and visualize on slice and sphere surface."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from sensray.mesh.earth_model import MeshEarthModel


def main() -> None:
    # Build a coarse volumetric Earth mesh for speed
    model = MeshEarthModel.from_pygmsh_sphere(
        radius_km=6371.0, mesh_size_km=1000.0, name="demo-sphere"
    )

    # Create a synthetic per-cell Vs field (e.g., increasing with depth)
    # Approximate cell depth from cell centers' radii
    centers = model.mesh.cell_centers().points  # (n_cells, 3)
    radii = np.linalg.norm(centers, axis=1)
    depth = 6371.0 - radii
    vs = 4.5 + 0.0005 * depth  # km/s: shallow ~4.5, deeper slightly higher
    model.set_cell_data("vs", vs)

    # Choose a source/receiver to define a slice plane
    src_lat, src_lon = 10.0, 20.0
    rec_lat, rec_lon = -20.0, 100.0

    # Plot a slice (with scalar)
    pv.set_plot_theme("document")
    p1 = model.plot_slice(
        src_lat,
        src_lon,
        rec_lat,
        rec_lon,
        scalar_name="vs",
        cmap="viridis",
        show_edges=False,
        wireframe=True,
        wireframe_line_width=2,
        clim=(4.5, 7.5),
        opacity=0.8,
    )
    p1.add_text("Slice colored by Vs", position="upper_left", font_size=10)
    p1.show()

    # Build a spherical surface at a chosen radius and sample Vs onto it
    # Plot a spherical surface (with scalar)
    p2 = model.plot_sphere(
        radius_km=4200.0,
        scalar_name="vs",
        cmap="plasma",
        wireframe=True,
        wireframe_line_width=2,
        clim=(4.5, 7.5),
        opacity=1.0,
        nan_opacity=0.0,
    )
    p2.add_text(
        "Sphere at r=4200 km (Vs)", position="upper_left", font_size=10
    )
    p2.show()


if __name__ == "__main__":
    main()
