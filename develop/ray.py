# plot_ray_lengths.py
# pip install pyvista numpy

from __future__ import annotations
import os, sys, pathlib
import numpy as np
import pyvista as pv
from pyvista import _vtk as vtk

EARTH_RADIUS_KM = 6371.0

# ---------------------- helpers ----------------------
def latlon_depth_to_xyz(lat_deg, lon_deg, depth_km, r_earth=EARTH_RADIUS_KM):
    r = float(r_earth) - float(depth_km)
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    return np.array([r*np.cos(lat)*np.cos(lon),
                     r*np.cos(lat)*np.sin(lon),
                     r*np.sin(lat)], float)

def densify_polyline(points, max_seg_len):
    """Subdivide segments so each subsegment <= max_seg_len (same units as points)."""
    points = np.asarray(points, float)
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        L = np.linalg.norm(b - a)
        if L == 0:
            continue
        n = max(1, int(np.ceil(L / max_seg_len)))
        for i in range(1, n + 1):
            t = i / n
            out.append(a*(1-t) + b*t)
    return np.asarray(out)

def compute_ray_cell_lengths(grid: pv.UnstructuredGrid,
                             ray_xyz: np.ndarray,
                             step_km: float = 8.0,
                             merge_tol: float = 1e-8) -> np.ndarray:
    """
    Accumulate ray length inside each cell by densifying segments and
    binning by midpoint with CellLocator.FindCell (no IntersectWithLine).
    """
    assert ray_xyz.ndim == 2 and ray_xyz.shape[1] == 3
    if grid.n_cells == 0:
        return np.zeros(0, float)

    # densify and prepare midpoints
    pts = densify_polyline(ray_xyz, max_seg_len=step_km)
    seg_vecs = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    keep = seg_len > merge_tol
    if not np.any(keep):
        return np.zeros(grid.n_cells, float)
    mids = 0.5*(pts[:-1] + pts[1:])[keep]
    seg_vecs = seg_vecs[keep]
    seg_len = seg_len[keep]

    # fast locator
    loc = vtk.vtkStaticCellLocator()
    loc.SetDataSet(grid)
    loc.BuildLocator()

    def find_cell(point, v):
        cid = loc.FindCell(point)
        if cid < 0:
            eps = 1e-7  # nudge off faces/edges
            cid = loc.FindCell(point + eps*v)
            if cid < 0:
                cid = loc.FindCell(point - eps*v)
        return int(cid) if cid is not None and cid >= 0 else -1

    cell_length = np.zeros(grid.n_cells, float)
    for m, v, L in zip(mids, seg_vecs, seg_len):
        cid = find_cell(m, v)
        if cid >= 0:
            cell_length[cid] += float(L)
    return cell_length

def build_demo_ray(radius_km=EARTH_RADIUS_KM) -> np.ndarray:
    """
    Simple great-circle-ish demo ray: from (lat,lon,depth) = (0,0,50km)
    to (30,40,300km) via a mid control point to bend inward.
    Replace this with your own coordinates or load from .npy file.
    """
    p0 = latlon_depth_to_xyz(0.0, 0.0, 50.0, radius_km)
    p1 = latlon_depth_to_xyz(15.0, 20.0, 400.0, radius_km)  # interior
    p2 = latlon_depth_to_xyz(30.0, 40.0, 300.0, radius_km)
    pts = np.vstack([p0, p1, p2])
    return densify_polyline(pts, max_seg_len=25.0)

# ---------------------- main ----------------------
def main():
    # Try to make headless environments work
    if os.environ.get("PYVISTA_USE_XVFB", "1") == "1":
        try:
            pv.start_xvfb()
        except Exception:
            pass

    if len(sys.argv) < 2:
        print("Usage: python plot_ray_lengths.py /path/to/mesh.vtu [optional: /path/to/ray_xyz.npy] [--screenshot out.png]")
        sys.exit(2)

    mesh_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    ray_path = None
    screenshot = None
    for arg in sys.argv[2:]:
        if arg.endswith(".npy"):
            ray_path = pathlib.Path(arg).expanduser().resolve()
        if arg.startswith("--screenshot"):
            parts = arg.split("=", 1)
            if len(parts) == 2:
                screenshot = parts[1]
            else:
                screenshot = "ray_lengths.png"

    grid = pv.read(str(mesh_path))
    print(f"\nLoaded mesh: {mesh_path.name}")
    print(f"  n_points: {grid.n_points}, n_cells: {grid.n_cells}")

    # Load or build a ray
    if ray_path and ray_path.exists():
        ray_xyz = np.load(str(ray_path))
        if ray_xyz.ndim == 1 and ray_xyz.size % 3 == 0:
            ray_xyz = ray_xyz.reshape(-1, 3)
        if ray_xyz.shape[1] != 3:
            raise ValueError("ray_xyz must have shape (N,3) or flat of length 3N.")
        print(f"Loaded ray: {ray_path.name} with {len(ray_xyz)} points")
    else:
        ray_xyz = build_demo_ray(radius_km=float(np.max(np.linalg.norm(grid.points, axis=1))))
        print(f"Using demo ray with {len(ray_xyz)} points")

    # Compute per-cell lengths
    lengths = compute_ray_cell_lengths(grid, ray_xyz, step_km=8.0)
    grid = grid.copy()  # don’t mutate disk-loaded object
    grid.cell_data["ray_len"] = lengths.astype(np.float32)

    # Summaries
    total_poly_len = float(np.sum(np.linalg.norm(ray_xyz[1:] - ray_xyz[:-1], axis=1)))
    inside_len = float(lengths.sum())
    hit_cells = int(np.count_nonzero(lengths > 0))
    print(f"Total polyline length (km): {total_poly_len:.3f}")
    print(f"Accumulated inside-mesh length (km): {inside_len:.3f}")
    print(f"Cells intersected: {hit_cells}")

    # Prepare geometry to plot
    # 1) cells with >0 length
    if hit_cells > 0:
        g_hit = grid.threshold((1e-10, float(lengths.max())), scalars="ray_len")
    else:
        g_hit = None

    # 2) ray as a polyline
    ray_poly = pv.lines_from_points(ray_xyz)

    # Plot
    p = pv.Plotter(window_size=(1100, 800))
    p.add_axes()
    p.show_grid()

    # Base mesh (wireframe) for context
    if grid.n_cells > 0:
        p.add_mesh(grid, color="lightgray", style="wireframe", opacity=0.25)

    # Intersected cells colored by length
    if g_hit is not None and g_hit.n_cells > 0:
        p.add_mesh(g_hit, scalars="ray_len", cmap="viridis", show_edges=True)
    else:
        print("No intersected cells to color; showing only the base mesh and ray.")

    # Ray line
    p.add_mesh(ray_poly, line_width=4)

    # View
    try:
        if screenshot:
            p.show(screenshot=screenshot, auto_close=True, interactive=False)
            print(f"Saved screenshot to: {screenshot}")
        else:
            p.camera_position = "iso"
            p.show()
    except Exception as e:
        # Fallback to headless screenshot if GUI can't open
        print(f"Interactive show failed ({e}); saving screenshot instead.")
        out = screenshot or "ray_lengths.png"
        try:
            pv.start_xvfb()
        except Exception:
            pass
        p.show(screenshot=out, auto_close=True, interactive=False)
        print(f"Saved screenshot to: {out}")

if __name__ == "__main__":
    main()
