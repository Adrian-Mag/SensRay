# assign_vp_from_1d.py
# pip install pyvista numpy

import numpy as np
import pyvista as pv

# ---- set your mesh path here ----
VTU = "nlayer.vtu"   # or "two_layer_sphere_two_sizes_smooth.vtu", etc.

# ---- define a 1D profile vp(r) ----
# Provide radii (km) and vp (km/s) at those radii. Radii must be ascending.
# You can fill this with your own model. Example (toy numbers):
r_nodes = np.array([0.0, 1221.5, 3480.0, 5701.0, 6331.0, 6371.0], float)
vp_nodes = np.array([11.0, 11.0,  9.0,   13.5,   6.8,    6.2   ], float)

# Choose mapping mode:
#   "linear"     -> linearly interpolate between nodes (smooth)
#   "step_left"  -> piecewise-constant: value of the left node on [r_i, r_{i+1})
#   "step_right" -> piecewise-constant: value of the right node on (r_{i-1}, r_i]
MODE = "step_left"

def assign_cell_property_from_1d(grid: pv.UnstructuredGrid,
                                 r_nodes: np.ndarray,
                                 v_nodes: np.ndarray,
                                 name: str = "vp",
                                 mode: str = "linear") -> None:
    """Attach per-cell values from a radial 1D profile v(r)."""
    # ensure ascending nodes
    order = np.argsort(r_nodes)
    rN = np.asarray(r_nodes, float)[order]
    vN = np.asarray(v_nodes, float)[order]

    # radii of cell centers
    centers = grid.cell_centers().points
    rc = np.linalg.norm(centers, axis=1)

    if mode == "linear":
        vals = np.interp(rc, rN, vN)  # clamps outside to endpoints
    else:
        # piecewise-constant
        # bins: r in [r_i, r_{i+1}) -> index i (step_left)
        #       r in (r_{i-1}, r_i] -> index i (step_right)
        right = (mode == "step_right")
        idx = np.digitize(rc, rN, right=right)  # 0..len(rN)
        idx = np.clip(idx, 1, len(rN)-1)
        # pick left or right value
        pick = idx if right else (idx-1)
        vals = vN[pick]

    grid.cell_data[name] = vals

# ---- load mesh, assign, and show ----
grid = pv.read(VTU)

assign_cell_property_from_1d(grid, r_nodes, vp_nodes, name="vp", mode=MODE)

# Quick QA: print min/max and a few sample values
vp = grid.cell_data["vp"]
print(f"Assigned vp to {vp.size} cells. min={vp.min():.3f}, max={vp.max():.3f}")

# Clip to view inside; color by CELL data (no smoothing across cells)
clip = grid.clip(normal=(1, 0, 0), origin=(0, 0, 0))

p = pv.Plotter()
p.add_mesh(grid.extract_surface(), color="lightgray", opacity=0.12)
p.add_mesh(
    clip,
    scalars="vp",
    preference="cell",            # use cell_data, constant per tet
    interpolate_before_map=False, # ensure no pre-interp
    cmap="viridis",
    show_edges=True,
    edge_color="black",
)
p.add_scalar_bar(title="vp (km/s)")
p.show()
