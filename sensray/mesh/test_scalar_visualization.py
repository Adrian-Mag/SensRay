"""Demo: assign per-cell scalars ('vs' and 'vp') and visualize.

Run with:
    python test_scalar_visualization.py --scalar vs
    python test_scalar_visualization.py --scalar vp
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from sensray.mesh.earth_model import MeshEarthModel
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize slice and spherical shell colored by a scalar"
    )
    parser.add_argument(
        "--scalar",
        choices=["vs", "vp"],
        default="vs",
        help="Which scalar to visualize",
    )
    args = parser.parse_args()
    scalar = args.scalar
    # Build a coarse volumetric Earth mesh for speed
    model = MeshEarthModel.from_pygmsh_sphere(
        radius_km=6371.0, mesh_size_km=1000.0, name="demo-sphere"
    )

    # Create synthetic per-cell fields (increasing slightly with depth)
    # Approximate cell depth from cell centers' radii
    centers = model.mesh.cell_centers().points  # (n_cells, 3)
    radii = np.linalg.norm(centers, axis=1)
    depth = 6371.0 - radii
    vs = 4.5 + 0.0005 * depth  # km/s: shallow ~4.5, deeper slightly higher
    vp = 1.73 * vs  # simple relation: Vp ~ 1.73 * Vs
    model.set_cell_data("vs", vs)
    model.set_cell_data("vp", vp)

    # Pick defaults per scalar
    if scalar == "vs":
        clim = (4.5, 7.5)
        cmap = "RdBu"
        label = "Vs"
    else:
        # Rough Vp range, consistent with vp above
        clim = (7.8, 13.0)
        cmap = "RdBu"
        label = "Vp"

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
        scalar_name=scalar,
        cmap=cmap,
        show_edges=False,
        wireframe=True,
        wireframe_line_width=2,
        clim=clim,
        opacity=0.8,
    )
    p1.add_text(
        f"Slice colored by {label}", position="upper_left", font_size=10
    )
    p1.show()

    # Build a spherical surface at a chosen radius and sample Vs onto it
    # Plot a spherical surface (with scalar)
    p2 = model.plot_sphere(
        radius_km=4200.0,
        scalar_name=scalar,
        cmap=cmap,
        wireframe=True,
        wireframe_line_width=2,
        clim=clim,
        opacity=1.0,
        nan_opacity=0.0,
    )
    p2.add_text(
        f"Sphere at r=4200 km ({label})", position="upper_left", font_size=10
    )
    p2.show()


if __name__ == "__main__":
    main()
