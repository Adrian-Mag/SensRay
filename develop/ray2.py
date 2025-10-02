# plot_ray_lengths.py
# pip install pyvista numpy

import os, sys, pathlib
import numpy as np
import pyvista as pv
from pyvista import _vtk as vtk


# -----------------------------
# Tetra helpers
# -----------------------------
def tetrahedralize_if_needed(grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Return a tetra-only grid; converts mixed cells with vtkDataSetTriangleFilter."""
    if grid is None or grid.n_cells == 0:
        return grid
    try:
        cts = grid.celltypes
        if (cts == vtk.VTK_TETRA).all():
            return grid
    except Exception:
        pass
    tri = vtk.vtkDataSetTriangleFilter()
    tri.SetInputData(grid)
    tri.Update()
    return pv.wrap(tri.GetOutput())


def _clip_segment_by_tetra(tet_pts: np.ndarray, p0: np.ndarray, p1: np.ndarray, tol=1e-8) -> float:
    """Length of segment p0->p1 inside a tetra (tet_pts: 4x3)."""
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L == 0.0:
        return 0.0
    tmin, tmax = 0.0, 1.0
    cen = tet_pts.mean(axis=0)
    for ia, ib, ic in faces:
        a, b, c = tet_pts[ia], tet_pts[ib], tet_pts[ic]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn == 0.0:
            continue
        n /= nn
        # inward normal -> inside means n·(x-a) <= 0
        if np.dot(n, cen - a) > 0:
            n = -n
        num = -np.dot(n, p0 - a)
        den =  np.dot(n, d)
        if abs(den) < tol:
            if np.dot(n, p0 - a) > tol:
                return 0.0  # parallel & outside
            continue        # parallel & inside
        th = num / den
        if den > 0:  # entering: t <= th
            tmax = min(tmax, th)
        else:        # leaving:  t >= th
            tmin = max(tmin, th)
        if tmin - tmax > tol:
            return 0.0
    t0 = max(0.0, min(1.0, tmin))
    t1 = max(0.0, min(1.0, tmax))
    if t1 <= t0:
        return 0.0
    return (t1 - t0) * L


def compute_ray_cell_lengths_tetra(
    grid: pv.UnstructuredGrid,
    ray_xyz: np.ndarray,
    *,
    tol: float = 1e-8,          # slightly tighter to avoid face double-counts
    ensure_tetra: bool = True,
) -> tuple[np.ndarray, pv.UnstructuredGrid]:
    """
    Compute per-cell path length (km) of a ray polyline (Nx3) through a tetra mesh.
    Returns (lengths, grid_used). If ensure_tetra=True, grid_used may be a tetra-ized copy.
    """
    if grid is None or grid.n_cells == 0:
        return np.zeros(0, float), grid

    if ray_xyz.ndim != 2 or ray_xyz.shape[1] != 3 or len(ray_xyz) < 2:
        raise ValueError("ray_xyz must be shape (N,3) with N>=2.")

    g = tetrahedralize_if_needed(grid) if ensure_tetra else grid

    try:
        cts = np.asarray(g.celltypes)
        tet_mask = (cts == vtk.VTK_TETRA)
    except Exception:
        tet_mask = None

    out = np.zeros(g.n_cells, float)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(g)
    locator.BuildLocator()

    idlist = vtk.vtkIdList()
    cache: dict[int, np.ndarray] = {}

    for p0, p1 in zip(ray_xyz[:-1], ray_xyz[1:]):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        if np.allclose(p0, p1):
            continue

        idlist.Reset()
        locator.FindCellsAlongLine(p0, p1, tol, idlist)

        for k in range(idlist.GetNumberOfIds()):
            cid = int(idlist.GetId(k))
            if cid < 0 or cid >= g.n_cells:
                continue
            if tet_mask is not None and not tet_mask[cid]:
                continue

            tet = cache.get(cid)
            if tet is None:
                cell = g.extract_cells(cid)
                tet_pts = np.asarray(cell.points, float)
                if tet_pts.shape[0] > 4:
                    _, idx = np.unique(tet_pts, axis=0, return_index=True)
                    tet_pts = tet_pts[np.sort(idx)[:4]]
                if tet_pts.shape[0] != 4:
                    continue
                cache[cid] = tet_pts
                tet = tet_pts

            seg_len = _clip_segment_by_tetra(tet, p0, p1, tol=tol)
            if seg_len > 0.0:
                out[cid] += seg_len

    return out, g


# -----------------------------
# Demo ray builder
# -----------------------------
def build_demo_ray(radius_km: float, kind: str = "chord", npts: int = 1201) -> np.ndarray:
    """
    Build a demo ray in Cartesian km.
      kind="chord": straight line from +x surface to -x surface
      kind="arc":   shallow arc near surface along +x -> +y quadrant
    """
    R = float(radius_km)
    if kind == "arc":
        theta = np.linspace(0.0, 0.5*np.pi, npts)
        x = R*np.cos(theta); y = R*np.sin(theta); z = np.zeros_like(theta)
        return np.column_stack([x, y, z]).astype(float)
    p0 = np.array([ R, 0.0, 0.0], float)
    p1 = np.array([-R, 0.0, 0.0], float)
    t  = np.linspace(0.0, 1.0, npts)[:, None]
    return (1.0 - t) * p0 + t * p1


# -----------------------------
# Pretty plotting
# -----------------------------
def make_plot(grid: pv.UnstructuredGrid, ray_xyz: np.ndarray, lengths: np.ndarray, screenshot: str | None):
    p = pv.Plotter(window_size=(1200, 900))
    p.set_background("white")
    p.add_axes()
    p.show_grid(color="lightgray")  # <- no opacity kwarg in recent PyVista

    # Base mesh (wireframe) for context
    p.add_mesh(grid, color="#d9d9d9", style="wireframe", opacity=0.25)

    # Intersected cells colored by per-cell length
    if lengths.size and np.max(lengths) > 0:
        g_hit = grid.copy()
        g_hit.cell_data["ray_len"] = lengths.astype(np.float32)
        g_hit = g_hit.threshold((1e-12, float(np.max(lengths))), scalars="ray_len")
        if g_hit.n_cells > 0:
            p.add_mesh(
                g_hit,
                scalars="ray_len",
                cmap="viridis",
                show_edges=True,
                scalar_bar_args=dict(
                    title="Ray length per cell (km)",
                    italic=False, bold=False, fmt="%.3f"
                ),
            )
        else:
            print("No intersected cells after threshold; plotting only mesh & ray.")
    else:
        print("All per-cell lengths are zero; plotting only mesh & ray.")

    # Ray line
    p.add_mesh(pv.lines_from_points(ray_xyz), line_width=5, color="black")

    p.camera_position = "iso"

    if screenshot:
        try:
            p.show(screenshot=screenshot, auto_close=True, interactive=False)
            print(f"Saved screenshot to: {screenshot}")
        except Exception as e:
            print(f"Interactive show failed ({e}); attempting headless screenshot...")
            try:
                pv.start_xvfb()
            except Exception:
                pass
            p.show(screenshot=screenshot, auto_close=True, interactive=False)
            print(f"Saved screenshot to: {screenshot}")
    else:
        p.show()


# -----------------------------
# main
# -----------------------------
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

    ray_file = None
    screenshot = None
    args = sys.argv[2:]

    # accept "--screenshot out.png" or "--screenshot=out.png"
    if "--screenshot" in args:
        i = args.index("--screenshot")
        if i + 1 < len(args):
            screenshot = args[i + 1]
            args = args[:i] + args[i+2:]
        else:
            screenshot = "ray_lengths.png"
            args = args[:i] + args[i+1:]
    for a in list(args):
        if a.startswith("--screenshot="):
            screenshot = a.split("=", 1)[1] or "ray_lengths.png"
            args.remove(a)
    for a in args:
        if a.endswith(".npy"):
            ray_file = pathlib.Path(a).expanduser().resolve()

    grid_in = pv.read(str(mesh_path))
    print(f"\nLoaded mesh: {mesh_path.name}")
    print(f"  n_points: {grid_in.n_points}, n_cells: {grid_in.n_cells}")

    # Load ray or build a demo one
    if ray_file and ray_file.exists():
        ray_xyz = np.load(str(ray_file))
        if ray_xyz.ndim == 1 and ray_xyz.size % 3 == 0:
            ray_xyz = ray_xyz.reshape(-1, 3)
        if ray_xyz.shape[1] != 3:
            raise ValueError("ray_xyz must have shape (N,3) or flat of length 3N.")
        print(f"Loaded ray: {ray_file.name} with {len(ray_xyz)} points")
    else:
        R_est = float(np.max(np.linalg.norm(grid_in.points, axis=1))) or 6371.0
        ray_xyz = build_demo_ray(R_est, kind="chord", npts=1201)
        print(f"Using demo ray with {len(ray_xyz)} points (chord, R~{R_est:.1f} km)")

    # Compute per-cell lengths
    lengths, grid = compute_ray_cell_lengths_tetra(grid_in, ray_xyz, tol=1e-8, ensure_tetra=True)

    # Summaries
    total_poly_len = float(np.sum(np.linalg.norm(ray_xyz[1:] - ray_xyz[:-1], axis=1)))
    inside_len = float(np.sum(lengths)) if lengths.size else 0.0
    hit_cells = int(np.count_nonzero(lengths > 0))
    print(f"Total polyline length (km): {total_poly_len:.3f}")
    print(f"Accumulated inside-mesh length (km): {inside_len:.3f}")
    print(f"Cells intersected: {hit_cells}")

    # Attach for plotting
    grid = grid.copy()
    if lengths.size == grid.n_cells:
        grid.cell_data["ray_len"] = lengths.astype(np.float32)

    # Plot
    make_plot(grid, ray_xyz, lengths, screenshot)


if __name__ == "__main__":
    main()
