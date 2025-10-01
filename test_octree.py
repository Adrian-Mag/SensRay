# octree_sphere_min_fixed.py
# pip install discretize pyvista numpy

import numpy as np
import discretize
import pyvista as pv
from obspy.taup import TauPyModel
from sensray.utils.coordinates import CoordinateConverter

# ---- params ----
R = 6371.0       # target sphere radius (km) to extract
N0 = 256          # base cells per axis (power of 2 recommended: 2^Lmax)
# Add an outer buffer halo so boundary artifacts occur outside the target
# sphere
PAD_KM = 2500.0   # thickness of the outer buffer (km). Adjust if needed.
# Build on larger cube [-R_DOMAIN, R_DOMAIN]^3
R_DOMAIN = R + PAD_KM

# TauP ray configuration (enable to use a seismic ray instead of a straight
# line)
DO_TAUP = True
TAUP_MODEL = "iasp91"
PHASES = ["P"]  # e.g., ["P"], ["S"], ["PP"], ["SKS"], etc.
SRC = {"lat": 0.0, "lon": 0.0, "depth_km": 50.0}
STA = {"lat": 0.0, "lon": 60.0}  # receiver at surface

# User-defined concentric shells (inner to outer) with target octree levels.
# Feel free to change radii/levels and add more shells. The last shell should
# have radius == R to cover the full sphere.
# Format per entry: (name, radius_km, level)
SHELLS = [
    ("inner-core", 1221.5, 4),
    ("outer-core", 3480.0, 4),
    ("660", 5711.0, 4),
    ("410", 5961.0, 4),
    ("moho", 6331.0, 5),
    ("surface", R, 5),
]

# 1) start with a single cube [-R_DOMAIN, R_DOMAIN]^3
hx = [(2 * R_DOMAIN / N0, N0)]
mesh = discretize.TreeMesh(
    [hx, hx, hx],
    origin=(-R_DOMAIN, -R_DOMAIN, -R_DOMAIN),
    diagonal_balance=True,
)

# 2) refine using a callable: arbitrary concentric shells with target levels
LMAX = int(np.log2(N0))

"""Build refinement layers from SHELLS and add a buffer halo outside the target
radius so any boundary balancing occurs outside the desired sphere and is later
cut away."""
h0 = 2 * R_DOMAIN / N0

# Clamp shells to mesh max level and sort by radius ascending
shell_names = [s[0] for s in SHELLS]
shell_radii = [float(s[1]) for s in SHELLS]
shell_levels = [min(int(s[2]), LMAX) for s in SHELLS]
order = np.argsort(shell_radii)
shell_names = [shell_names[i] for i in order]
shell_radii = [shell_radii[i] for i in order]
shell_levels = [shell_levels[i] for i in order]

# Preserve the original outer radius as the extraction radius (target Earth)
EXTRACT_R = float(shell_radii[-1])

# Append a buffer shell up to R + PAD_KM with the same refinement as the crust
buffer_radius = EXTRACT_R + PAD_KM
buffer_level = shell_levels[-1]
# match crust level to avoid jumps at EXTRACT_R
shell_names.append("buffer")
shell_radii.append(buffer_radius)
shell_levels.append(min(int(buffer_level), LMAX))
print(EXTRACT_R)
# Print shell configuration
desc_shells = ", ".join([
    f"{nm}: r<= {rad:.1f} km -> L{lvl}"
    for nm, rad, lvl in zip(shell_names, shell_radii, shell_levels)
])
print(f"Shell refinement levels (LMAX={LMAX}): {desc_shells}")

# Use callable refinement for proper shell levels (no ramp for symmetry)


def shell_level(cell):
    cx, cy, cz = cell.center
    rr = (cx * cx + cy * cy + cz * cz) ** 0.5
    for i, (rad, lvl) in enumerate(zip(shell_radii, shell_levels)):
        if rr <= rad:
            return int(lvl)
    return 0  # outside all shells


mesh.refine(shell_level)
print("Applied shell-specific refinement levels")

# balance the tree (valid hanging nodes)
mesh.finalize()

# 3) convert to VTK and keep only cells whose centers are inside the sphere
# Wrap the VTK object with PyVista so we can use its convenience methods
grid_all = pv.wrap(mesh.to_vtk())  # pyvista.UnstructuredGrid
centers = grid_all.cell_centers().points
r_all = np.linalg.norm(centers, axis=1)
# Select strictly inside the sphere; refinement halo is only to improve balance
inside = r_all <= EXTRACT_R
idx = np.where(inside)[0]
print(f"Total cells in VTK grid: {grid_all.n_cells}")
print(f"Cells with centers inside sphere: {idx.size}")

# extract_cells expects cell indices (relative to this grid). Using the wrapped
# grid ensures indices align with PyVista's ordering. The result is a volume
# containing only hexahedral cells whose centers lie inside the sphere.
grid = grid_all.extract_cells(idx)
centers_in = pv.wrap(grid).cell_centers().points
r_in = np.linalg.norm(centers_in, axis=1)

# region index by shell: 0..len(shells)-1. Ramp layers are outside EXTRACT_R.


def shell_index(r):
    for i, rad in enumerate(shell_radii):
        if r <= rad:
            return i
    return len(shell_radii) - 1


region = np.array([shell_index(r) for r in r_in], dtype=np.int32)
grid.cell_data["region"] = region
try:
    # Optional string names (may not render everywhere but saved to VTU)
    grid.cell_data["region_name"] = np.array([shell_names[i] for i in region])
except Exception:
    pass

# 4) Extract surface from the selected volume (should be symmetric now)
surf = pv.wrap(grid).extract_surface().clean()

print(f"Extracted volume: cells={grid.n_cells}, points={grid.n_points}")
print(f"Surface: cells={surf.n_cells}, points={surf.n_points}")

# Sanity check: average cell volumes per shell
grid_with_vol = pv.wrap(grid).compute_cell_sizes(
    length=False, area=False, volume=True
)
vol = grid_with_vol.cell_data["Volume"]
for i, nm in enumerate(shell_names):
    mask = region == i
    if mask.any():
        avg = float(vol[mask].mean())
        print(f"Shell {i} ({nm}) avg cell volume: {avg:.3e}")

# --- Simple straight-line ray: per-cell path lengths ---


def _aabb_segment_length(bounds, p0, p1):
    """Length of the segment inside an axis-aligned box.

    For our hexahedral cells, this AABB intersection is exact.

    bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    p0, p1: endpoints of the segment (3,)
    returns: float length (>=0)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    d = p1 - p0
    seg_len = float(np.linalg.norm(d))
    if seg_len == 0.0:
        # Degenerate segment; test if point is inside box
        inside = (
            (xmin <= p0[0] <= xmax)
            and (ymin <= p0[1] <= ymax)
            and (zmin <= p0[2] <= zmax)
        )
        return 0.0 if not inside else 0.0

    tmin, tmax = 0.0, 1.0
    for axis, mn, mx in ((0, xmin, xmax), (1, ymin, ymax), (2, zmin, zmax)):
        if d[axis] != 0.0:
            t1 = (mn - p0[axis]) / d[axis]
            t2 = (mx - p0[axis]) / d[axis]
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)
            tmin = max(tmin, t_enter)
            tmax = min(tmax, t_exit)
            if tmax <= tmin:
                return 0.0
        else:
            # Parallel to this axis: must lie within slab.
            # Use half-open inclusion [mn, mx) to avoid counting shared faces
            # twice across neighboring cells.
            if not (mn <= p0[axis] < mx):
                return 0.0
            # else: no constraint on t from this axis
    if tmax <= 0.0 or tmin >= 1.0:
        return 0.0
    t0 = max(tmin, 0.0)
    t1c = min(tmax, 1.0)
    if t1c <= t0:
        return 0.0
    return seg_len * (t1c - t0)


def compute_ray_cell_lengths(grid, p0, p1):
    """Compute per-cell intersection lengths of a line segment with the grid.

    returns: lengths (n_cells,), total_length
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    n = grid.n_cells
    lengths = np.zeros(n, dtype=float)
    pts = grid.points
    # Iterate cells: get per-cell bounds via point ids
    for cid in range(n):
        cell = grid.get_cell(cid)
        ids = cell.point_ids
        # point_ids can be vtkIdList; fetch coordinates
        coords = pts[np.array(ids, dtype=int)]
        xmin, ymin, zmin = coords.min(axis=0)
        xmax, ymax, zmax = coords.max(axis=0)
        length = _aabb_segment_length(
            (xmin, xmax, ymin, ymax, zmin, zmax), p0, p1
        )
        if length > 0.0:
            lengths[cid] = length
    return lengths, float(lengths.sum())


def _build_cell_bounds_for_ids(grid, ids):
    """Compute AABB bounds for a subset of cells.

    Returns array of shape (k, 6): [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    pts = grid.points
    k = len(ids)
    bounds = np.empty((k, 6), dtype=float)
    for j, cid in enumerate(ids):
        cell = grid.get_cell(int(cid))
        cids = np.array(cell.point_ids, dtype=int)
        coords = pts[cids]
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        bounds[j, 0] = mn[0]
        bounds[j, 1] = mx[0]
        bounds[j, 2] = mn[1]
        bounds[j, 3] = mx[1]
        bounds[j, 4] = mn[2]
        bounds[j, 5] = mx[2]
    return bounds


def _aabb_segment_length_vec(bounds, p0, p1):
    """Vectorized segment length inside many AABBs.

    bounds: (n,6), p0,p1: (3,)
    returns: lengths (n,)
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    seg_len = float(np.linalg.norm(d))
    if seg_len == 0.0:
        return np.zeros(bounds.shape[0], dtype=float)

    # Unpack
    xmin, xmax = bounds[:, 0], bounds[:, 1]
    ymin, ymax = bounds[:, 2], bounds[:, 3]
    zmin, zmax = bounds[:, 4], bounds[:, 5]

    # Initialize t-intervals
    tmin = np.zeros_like(xmin)
    tmax = np.ones_like(xmin)

    # Vectorized per-axis handling (slab method with half-open parallel check)
    def slab(comp0, dd, mn, mx, tmin, tmax):
        if dd == 0.0:
            mask_inside = (mn <= comp0) & (comp0 < mx)
            tmin_new = np.where(mask_inside, tmin, 1.0)
            tmax_new = np.where(mask_inside, tmax, 0.0)
            return tmin_new, tmax_new
        t1 = (mn - comp0) / dd
        t2 = (mx - comp0) / dd
        t_enter = np.minimum(t1, t2)
        t_exit = np.maximum(t1, t2)
        tmin_new = np.maximum(tmin, t_enter)
        tmax_new = np.minimum(tmax, t_exit)
        return tmin_new, tmax_new

    tmin, tmax = slab(p0[0], d[0], xmin, xmax, tmin, tmax)
    tmin, tmax = slab(p0[1], d[1], ymin, ymax, tmin, tmax)
    tmin, tmax = slab(p0[2], d[2], zmin, zmax, tmin, tmax)

    # Valid intersections within [0,1]
    valid = (tmax > tmin) & (tmax > 0.0) & (tmin < 1.0)
    t0 = np.maximum(tmin, 0.0)
    t1c = np.minimum(tmax, 1.0)
    lengths = np.where(valid, seg_len * (t1c - t0), 0.0)
    return lengths


def compute_polyline_cell_lengths(grid, points, centers=None, margin=None):
    """Compute per-cell intersection lengths for a polyline.

    points: (m,3) vertices
    centers: optional (n,3) cell centers to speed candidate selection
    margin: km to expand polyline bounding box for candidate cells
    """
    pts = np.asarray(points, float)
    if pts.shape[0] < 2:
        return np.zeros(grid.n_cells, float)
    if centers is None:
        centers = pv.wrap(grid).cell_centers().points
    if margin is None:
        # Use ~2 coarse cells as margin
        h0 = 2 * R_DOMAIN / N0
        margin = 2.0 * h0
    bb_min = pts.min(axis=0) - margin
    bb_max = pts.max(axis=0) + margin
    # Debug info
    print(
        "Polyline points: {}".format(pts.shape[0])
        + ", x[{:.1f},{:.1f}]".format(pts[:, 0].min(), pts[:, 0].max())
        + ", y[{:.1f},{:.1f}]".format(pts[:, 1].min(), pts[:, 1].max())
        + ", z[{:.1f},{:.1f}]".format(pts[:, 2].min(), pts[:, 2].max())
    )
    mask = (
        (centers[:, 0] >= bb_min[0]) & (centers[:, 0] <= bb_max[0]) &
        (centers[:, 1] >= bb_min[1]) & (centers[:, 1] <= bb_max[1]) &
        (centers[:, 2] >= bb_min[2]) & (centers[:, 2] <= bb_max[2])
    )
    cand_ids = np.nonzero(mask)[0]
    print(
        "Candidate cells near polyline: {} (margin ~{:.1f} km)".format(
            cand_ids.size, margin
        )
    )
    if cand_ids.size == 0:
        # Fallback: scan all cells (grid is already extracted volume)
        print("No candidates in bbox; falling back to all cells.")
        cand_ids = np.arange(grid.n_cells, dtype=int)
    bounds = _build_cell_bounds_for_ids(grid, cand_ids)

    lengths_sub = np.zeros(len(cand_ids), float)
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        lengths_sub += _aabb_segment_length_vec(bounds, p0, p1)

    lengths = np.zeros(grid.n_cells, float)
    lengths[cand_ids] = lengths_sub
    return lengths


def taup_polyline_points(src, sta, model_name, phase_list, earth_radius=R):
    """Build 3D polyline points from TauP ray path in an x-z great-circle
    plane.

    src: dict with keys lat, lon, depth_km
    sta: dict with keys lat, lon (surface)
    """
    # Epicentral distance in degrees
    _, dist_deg, _ = CoordinateConverter.geographic_to_distance(
        src["lat"], src["lon"], sta["lat"], sta["lon"]
    )
    model = TauPyModel(model=model_name)
    rays = model.get_ray_paths(
        distance_in_degree=dist_deg,
        source_depth_in_km=src["depth_km"],
        phase_list=phase_list,
    )
    if not rays:
        return None
    # Choose the arrival whose endpoint distance is closest to requested
    best = None
    best_err = float("inf")
    for arr in rays:
        arr_dist = getattr(arr, "distance", None)
        if arr_dist is None:
            continue
        err = abs(float(arr_dist) - float(dist_deg))
        if best is None or err < best_err:
            best = arr
            best_err = err
    if best is None:
        best = rays[0]
    # structured array with fields like 'dist' (deg) and 'depth' (km)
    path = getattr(best, "path", None)
    if path is None:
        return None
    dist = np.asarray(path["dist"], float)
    depth = np.asarray(path["depth"], float)
    try:
        print(
            "TauP dist range: {:.2f}..{:.2f} deg (target {:.2f} deg)".format(
                float(dist.min()), float(dist.max()), float(dist_deg)
            )
        )
    except Exception:
        pass
    r = earth_radius - depth
    alpha = np.deg2rad(dist)
    x = r * np.sin(alpha)
    z = r * np.cos(alpha)
    y = np.zeros_like(x)
    pts = np.column_stack([x, y, z])
    return pts

# --- Build ray (TauP or straight) and compute per-cell distances ---


ray_poly = None
if DO_TAUP:
    pts = taup_polyline_points(SRC, STA, TAUP_MODEL, PHASES, earth_radius=R)
    if pts is None:
        print("TauP returned no rays; falling back to straight chord.")
    else:
        ray_lengths = compute_polyline_cell_lengths(
            grid, pts, centers=centers_in
        )
        grid.cell_data["ray_length"] = ray_lengths
        total_L = float(ray_lengths.sum())
        print(f"TauP ray total path length across cells: {total_L:.3f} km")
        # Build a polyline for plotting
        try:
            ray_poly = pv.lines_from_points(pts)
            ray_poly.save("ray_polyline.vtk")
        except Exception:
            ray_poly = None

if ray_poly is None and "ray_length" not in grid.cell_data:
    # Straight chord through the center along x-axis
    p0 = np.array([-EXTRACT_R, 0.0, 0.0])
    p1 = np.array([EXTRACT_R, 0.0, 0.0])
    ray_lengths, total_L = compute_ray_cell_lengths(grid, p0, p1)
    grid.cell_data["ray_length"] = ray_lengths
    print(f"Ray total path length across included cells: {total_L:.3f} km")
    try:
        ray_poly = pv.Line(p0, p1)
    except Exception:
        ray_poly = None

# If running in a headless environment, set this to False and the script will
# save the surface to a file instead of opening an interactive window.
DO_PLOT = True
DO_CLIP = True  # clip with a plane to see inside
CLIP_NORMAL = (0.0, 1.0, 0.0)  # plane normal (x-cut)
CLIP_ORIGIN = (0.0, 0.0, 0.0)  # plane point (through center)
if DO_PLOT:
    if DO_CLIP:
        grid_clip = pv.wrap(grid).clip(
            normal=CLIP_NORMAL, origin=CLIP_ORIGIN
        )
        surf_clip = pv.wrap(grid_clip).extract_surface()
        # label surface cells by shell index
        centers_s = pv.wrap(surf_clip).cell_centers().points
        r_s = np.linalg.norm(centers_s, axis=1)
        region_surf = np.array([shell_index(r) for r in r_s], dtype=np.int32)
        surf_clip.cell_data["region"] = region_surf
        # Save clipped surface for inspection
        try:
            surf_clip.save("sphere_surface_clipped.vtk")
        except Exception:
            pass
        # Plot the clipped volume colored by per-cell ray length if present
        p = pv.Plotter()
        scalars_name = (
            "ray_length" if "ray_length" in grid_clip.cell_data else "region"
        )
        cmap_name = "viridis" if scalars_name == "ray_length" else "tab10"
        p.add_mesh(
            grid_clip,
            show_edges=True,
            scalars=scalars_name,
            cmap=cmap_name,
        )
        if ray_poly is not None:
            p.add_mesh(ray_poly, color="red", line_width=3)
        p.show()
    else:
        # Non-clipped view: overlay ray and color by ray_length if desired
        p = pv.Plotter()
        scalars_name = (
            "ray_length" if "ray_length" in grid.cell_data else "region"
        )
        cmap_name = "viridis" if scalars_name == "ray_length" else "tab10"
        # Plot the surface for lighter visualization
        centers_s = pv.wrap(surf).cell_centers().points
        r_s = np.linalg.norm(centers_s, axis=1)
        region_surf = np.array([shell_index(r) for r in r_s], dtype=np.int32)
        surf.cell_data["region"] = region_surf
        # If ray_length coloring requested, transfer to surface by sampling
        if scalars_name == "ray_length":
            # Map cell-centered data to surface via interpolation of cell
            # centers
            # Fallback: color by region if mapping is unavailable
            try:
                # Use pyvista's probe to sample cell data onto surface points
                surf2 = surf.sample(pv.wrap(grid))
                p.add_mesh(
                    surf2,
                    show_edges=True,
                    scalars=scalars_name,
                    cmap=cmap_name,
                )
            except Exception:
                p.add_mesh(
                    surf,
                    show_edges=True,
                    scalars="region",
                    cmap="tab10",
                )
        else:
            p.add_mesh(
                surf,
                show_edges=True,
                scalars="region",
                cmap="tab10",
            )
        if ray_poly is not None:
            p.add_mesh(ray_poly, color="red", line_width=3)
        p.show()
else:
    out = "sphere_surface.vtk"
    surf.save(out)
    print(f"Saved surface to {out}")

# Always save the surface for inspection
try:
    surf.save("sphere_surface.vtk")
except Exception:
    pass

# Save volume grid with region labels
try:
    grid.save("sphere_volume.vtu")
except Exception:
    pass
