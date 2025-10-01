# octree_layers_uniform.py
# pip install discretize pyvista numpy

import numpy as np
import discretize
import pyvista as pv

# ---------------- params ----------------
R = 6371.0  # sphere radius (km)

# Concentric shells (inner->outer) with target octree levels.
# Level = number of binary splits from the root in that band.
SHELLS = [
    ("inner-core", 1221.5, 3),
    ("outer-core", 3480.0, 4),
    ("mantle",     6331.0, 5),
    ("crust",      R,      7),
]

# Optional: extra sharpening of the very outer surface
OUTER_BAND_THICKNESS = 0.0  # set e.g. 200.0 to add one more level in the last 200 km

# -------------- build & refine --------------
# Start with a SINGLE root cube [-R, R]^3
mesh = discretize.TreeMesh(
    [[(2*R, 1)], [(2*R, 1)], [(2*R, 1)]],
    origin=(-R, -R, -R),
    diagonal_balance=True,
)

# Refinement helper: refine an annulus (r in (r_in, r_out]) by 1 level
def refine_band(mesh, r_in, r_out):
    mesh.finalize()
    cc = mesh.cell_centers
    r = np.linalg.norm(cc, axis=1)
    mask = (r > r_in) & (r <= r_out)
    mesh.refine(mask)

# Refine each shell to its target level using annuli so levels don’t “bleed” inward
prev_r = 0.0
for name, r_out, level in SHELLS:
    for _ in range(level):
        refine_band(mesh, prev_r, r_out)
    prev_r = r_out

# Optional extra sharpening near the surface
if OUTER_BAND_THICKNESS > 0:
    refine_band(mesh, R - OUTER_BAND_THICKNESS, R)

# Balance the tree (enforces 2:1 neighbors, may add a few cells)
mesh.finalize()

# -------------- extract spherical volume --------------
# Convert to VTK grid of hexahedra
grid_all = pv.wrap(mesh.to_vtk())

# Keep cells that INTERSECT the sphere: r_center - half_diagonal <= R
centers = grid_all.cell_centers().points
r_c = np.linalg.norm(centers, axis=1)

half_diag = np.empty(grid_all.n_cells, float)
for i in range(grid_all.n_cells):
    xmin, xmax, ymin, ymax, zmin, zmax = grid_all.get_cell(i).bounds
    dx = xmax - xmin; dy = ymax - ymin; dz = zmax - zmin
    half_diag[i] = 0.5 * np.sqrt(dx*dx + dy*dy + dz*dz)

keep = (r_c - half_diag) <= R + 1e-9  # tiny tolerance
grid = grid_all.extract_cells(np.where(keep)[0])

# Label cells by shell index (0..len(SHELLS)-1) using center radius
shell_radii = np.array([r for _, r, _ in SHELLS], float)
def shell_idx(rc):
    return int(np.searchsorted(shell_radii, rc, side="right") - 1)
centers_in = pv.wrap(grid).cell_centers().points
grid.cell_data["region"] = np.array([shell_idx(np.linalg.norm(p)) for p in centers_in], np.int32)

print(f"VTK cells total: {grid_all.n_cells}  -> kept in sphere: {grid.n_cells}")

# -------------- visualize --------------
DO_CLIP = True
if DO_CLIP:
    clipped = pv.wrap(grid).clip(normal=(1,0,0), origin=(0,0,0))
    surf = pv.wrap(clipped).extract_surface()
else:
    surf = pv.wrap(grid).extract_surface()

p = pv.Plotter()
p.add_mesh(surf, scalars="region", cmap="tab10", show_edges=True, interpolate_before_map=False)
p.add_scalar_bar(title="region (shell index)")
p.show()

# -------------- save (optional) --------------
try: grid.save("octree_sphere_volume.vtu")
except Exception: pass
try: pv.wrap(grid).extract_surface().save("octree_sphere_surface.vtp")
except Exception: pass
