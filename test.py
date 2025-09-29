# pip install pyvista numpy
import numpy as np
import pyvista as pv
from pyvista import _vtk as vtk

def latlon_to_xyz(lat_deg, lon_deg, r=1.0):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    return np.array([r*np.cos(lat)*np.cos(lon),
                     r*np.cos(lat)*np.sin(lon),
                     r*np.sin(lat)], dtype=float)

def refine_radii(radii_km,
                 subdivs=0,
                 max_dr=None):
    """
    Return a refined, sorted, unique list of radii including the originals.

    Parameters
    ----------
    radii_km : sequence[float]
        Monotonic non-decreasing radii (discontinuities), e.g. [0, 1221.5, 3480, 5701, 6371].
    subdivs : int | sequence[int]
        If int N>0, insert N **uniform** internal cuts in every [r_i, r_{i+1}].
        If sequence, length must be len(radii_km)-1; per-interval N.
    max_dr : float | None
        If set (km), each interval is split into ceil((r2-r1)/max_dr) equal sublayers.

    Notes
    -----
    - If both `subdivs` and `max_dr` are given, `max_dr` takes precedence.
    """
    radii = np.asarray(radii_km, float)
    assert np.all(np.diff(radii) >= 0), "radii_km must be non-decreasing"

    refined = [radii[0]]
    L = len(radii) - 1

    if max_dr is not None:
        n_per = [int(np.ceil((radii[i+1]-radii[i]) / float(max_dr))) for i in range(L)]
    else:
        if isinstance(subdivs, (list, tuple, np.ndarray)):
            assert len(subdivs) == L
            n_per = [int(n) for n in subdivs]
        else:
            n_per = [int(subdivs)] * L

    for i in range(L):
        r1, r2 = radii[i], radii[i+1]
        n = max(0, n_per[i])
        if n == 0:
            refined.extend([r2])
        else:
            mids = np.linspace(r1, r2, n+2)[1:-1]
            refined.extend(mids)
            refined.extend([r2])

    # dedupe + sort (just in case)
    out = np.unique(np.array(refined, float))
    return out.tolist()

def build_layered_wedge_mesh(
    radii_km, theta_res=96, phi_res=48
) -> pv.UnstructuredGrid:
    """
    Create a layered spherical mesh of VTK WEDGE (triangular prism) cells.
    `radii_km` is an ascending list of shell radii [r0, r1, ..., rN].
    """
    # 1) Triangulate the unit sphere (surface)
    surf = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res)
    dirs = surf.points
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # unit directions
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    nV = dirs.shape[0]

    # 2) Build 3D vertices for each radial level by scaling the unit directions
    levels = [dirs * r for r in radii_km]   # list of (nV, 3)
    points = np.vstack(levels)

    # 3) For each layer and each triangle, make a WEDGE cell
    cell_types = []
    cell_conn = []
    for li in range(len(radii_km) - 1):
        off_bot = li * nV
        off_top = (li + 1) * nV
        for i0, i1, i2 in faces:
            b0, b1, b2 = off_bot + i0, off_bot + i1, off_bot + i2
            t0, t1, t2 = off_top + i0, off_top + i1, off_top + i2
            cell_conn.extend([6, b0, b1, b2, t0, t1, t2])
            cell_types.append(pv.CellType.WEDGE)

    cells = np.asarray(cell_conn, dtype=np.int64)
    cell_types = np.asarray(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    return grid

def assign_cell_scalar_by_layer(grid: pv.UnstructuredGrid, radii_km, name="vs", f=None):
    """
    Assign a cell scalar that's constant within each refined shell.
    - If f is None: simple demo function of layer mid-radius.
    - Else: f(r_inner, r_outer, r_mid, layer_index) -> value
    """
    centers = grid.cell_centers().points
    r = np.linalg.norm(centers, axis=1)

    radii = np.asarray(radii_km, float)
    layer_idx = np.searchsorted(radii, r, side="right") - 1
    layer_idx = np.clip(layer_idx, 0, len(radii) - 2)

    r_in  = radii[layer_idx]
    r_out = radii[layer_idx + 1]
    r_mid = 0.5 * (r_in + r_out)

    if f is None:
        vals = 4.5 + 0.0002 * (r_mid - radii[0])  # demo (km/s)
    else:
        vals = np.array([f(a, b, m, k) for a, b, m, k in zip(r_in, r_out, r_mid, layer_idx)], float)

    grid.cell_data[name] = vals
    grid.cell_data["layer_id"] = layer_idx.astype(np.int32)

def add_example_cell_scalar(grid: pv.UnstructuredGrid, radii_km, name="vs"):
    """
    Attach a per-cell scalar that's constant within each spherical shell.
    Uses cell centers to determine the layer index.
    """
    centers = grid.cell_centers().points            # (ncells, 3)
    r = np.linalg.norm(centers, axis=1)             # radius of each cell center

    radii = np.asarray(radii_km, float)
    # layer index: 0..len(radii)-2  (between radii[i], radii[i+1])
    layer_idx = np.searchsorted(radii, r, side="right") - 1
    layer_idx = np.clip(layer_idx, 0, len(radii) - 2)

    rmid = 0.5 * (radii[layer_idx] + radii[layer_idx + 1])

    # Example model: vary with layer mid-radius (replace with your own)
    vals = 4.5 + 0.0002 * (rmid - radii[0])         # e.g., km/s
    grid.cell_data[name] = vals.astype(float)

def spherical_slice_with_cell_scalars(grid: pv.UnstructuredGrid,
                                      scalar_name: str,
                                      radius: float,
                                      eps_rel: float = 1e-6) -> pv.PolyData:
    """
    Extract a spherical slice (isosurface at ||x|| = radius) and color it by
    the parent cell's scalar so each triangle has a constant color.
    """
    # Ensure the scalar lives on CELLS (no smoothing)
    g = grid
    if scalar_name in g.point_data and scalar_name not in g.cell_data:
        g = g.point_data_to_cell_data(pass_point_data=False)

    # Add a pointwise radius field (km) and extract the isosurface r=radius
    if "radius" not in g.point_data:
        g = g.copy()
        g.point_data["radius"] = np.linalg.norm(g.points, axis=1)
    sph = g.contour(isosurfaces=[radius], scalars="radius")  # -> PolyData

    # Map each sphere-triangle back to its parent cell using a locator
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(g)
    locator.BuildLocator()

    centers = sph.cell_centers().points
    parent_ids = np.empty(sph.n_cells, dtype=int)

    # Small nudge along radial direction to avoid sitting exactly on faces
    for i, c in enumerate(centers):
        r = np.linalg.norm(c)
        u = c / (r + 1e-15)
        step = max(radius * eps_rel, 1e-9)  # absolute fallback
        p_try = (c + step * u).astype(float)

        try:
            cid = locator.FindCell(p_try)  # some VTK builds expose this overload
        except TypeError:
            # Use the long signature if needed
            gcell = vtk.vtkGenericCell(); pcoords = [0.0, 0.0, 0.0]
            weights = [0.0] * g.GetMaxCellSize()
            cid = locator.FindCell(p_try, 0.0, gcell, pcoords, weights)

        if cid < 0:
            p_try = (c - step * u).astype(float)  # try inward
            try:
                cid = locator.FindCell(p_try)
            except TypeError:
                gcell = vtk.vtkGenericCell(); pcoords = [0.0, 0.0, 0.0]
                weights = [0.0] * g.GetMaxCellSize()
                cid = locator.FindCell(p_try, 0.0, gcell, pcoords, weights)

        if cid < 0:
            raise RuntimeError("Could not map spherical slice triangle to a parent cell.")
        parent_ids[i] = int(cid)

    # Copy parent **cell** scalar to the slice polygons (constant color per poly)
    sph.cell_data[scalar_name] = np.asarray(g.cell_data[scalar_name])[parent_ids]
    return sph

def slice_great_circle_with_cell_scalars(grid, scalar_name,
                                         src_lat, src_lon, rec_lat, rec_lon,
                                         eps=1e-6):
    # --- define the great-circle plane ---
    A = latlon_to_xyz(src_lat, src_lon, r=1.0)
    B = latlon_to_xyz(rec_lat, rec_lon, r=1.0)
    n = np.cross(A, B); n /= np.linalg.norm(n)

    # --- make the slice (polygons on the plane) ---
    sl = grid.slice(normal=tuple(n), origin=(0.0, 0.0, 0.0))

    # --- try the fast path: OriginalCellIds present? ---
    key = next((k for k in sl.cell_data.keys() if "OriginalCellIds" in k), None)
    if key is not None:
        parent_ids = sl.cell_data[key].astype(int)
    else:
        # --- robust fallback: use a cell locator on the volume ---
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(grid)
        locator.BuildLocator()

        centers = sl.cell_centers().points
        parent_ids = np.empty(sl.n_cells, dtype=int)

        # Allocate work buffers for FindCell signature variant
        gcell = vtk.vtkGenericCell()
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0] * grid.GetMaxCellSize()

        for i, c in enumerate(centers):
            # nudge off the plane to avoid boundary ambiguity
            p_try = (c + eps * n).astype(float)
            try:
                cid = locator.FindCell(p_try, 0.0, gcell, pcoords, weights)
            except TypeError:
                # some VTK wrappers expose FindCell(point) overload
                cid = locator.FindCell(p_try)

            if cid < 0:
                p_try = (c - eps * n).astype(float)
                try:
                    cid = locator.FindCell(p_try, 0.0, gcell, pcoords, weights)
                except TypeError:
                    cid = locator.FindCell(p_try)

            if cid < 0:
                raise RuntimeError("Could not map slice polygon to a parent cell.")
            parent_ids[i] = int(cid)

    # copy the parent **cell** scalar (constant per polygon)
    sl.cell_data[scalar_name] = np.asarray(grid.cell_data[scalar_name])[parent_ids]
    return sl

if __name__ == "__main__":
     # Discontinuities (km): inner core, CMB, 660, Moho, surface
    base = [0.0, 1221.5, 3480.0, 5701.0, 6371.0]

    # Option A: same number of extra cuts per shell (e.g., 3 interior splits)
    # refined = refine_radii(base, subdivs=3)

    # Option B: enforce a maximum Δr (e.g., ~50 km) everywhere
    refined = refine_radii(base, max_dr=500.0)

    print(f"{len(base)} original radii -> {len(refined)} refined levels")

    grid = build_layered_wedge_mesh(refined, theta_res=96, phi_res=48)
    assign_cell_scalar_by_layer(grid, refined, name="vs")

    # Quick check: great-circle slice colored by **cell** scalar (constant per polygon)
    from pyvista import _vtk as vtk

    def slice_gc_cell_scalars(grid, scalar_name, src_lat, src_lon, rec_lat, rec_lon, eps=1e-6):
        A = latlon_to_xyz(src_lat, src_lon, 1.0)
        B = latlon_to_xyz(rec_lat, rec_lon, 1.0)
        n = np.cross(A, B); n /= np.linalg.norm(n)
        sl = grid.slice(normal=tuple(n), origin=(0.0, 0.0, 0.0))

        # map polygons back to parent cells (robust fallback)
        locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(grid); locator.BuildLocator()
        centers = sl.cell_centers().points
        parent_ids = np.empty(sl.n_cells, dtype=int)
        for i, c in enumerate(centers):
            p_try = (c + eps * n).astype(float)
            cid = locator.FindCell(p_try) if hasattr(locator, "FindCell") else -1
            if cid < 0:
                gcell = vtk.vtkGenericCell(); pcoords=[0,0,0]; weights=[0.0]*grid.GetMaxCellSize()
                cid = locator.FindCell(p_try, 0.0, gcell, pcoords, weights)
            if cid < 0:
                p_try = (c - eps * n).astype(float)
                try:
                    cid = locator.FindCell(p_try)
                except TypeError:
                    gcell = vtk.vtkGenericCell(); pcoords=[0,0,0]; weights=[0.0]*grid.GetMaxCellSize()
                    cid = locator.FindCell(p_try, 0.0, gcell, pcoords, weights)
            if cid < 0:
                raise RuntimeError("Could not map slice polygon to a parent cell.")
            parent_ids[i] = int(cid)

        sl.cell_data[scalar_name] = np.asarray(grid.cell_data[scalar_name])[parent_ids]
        return sl

    sl = slice_gc_cell_scalars(grid, "vs", 0.0, 0.0, 60.0, 90.0)

    p = pv.Plotter()
    p.add_mesh(grid.extract_surface(), color="lightgray", opacity=0.15)
    p.add_mesh(sl, scalars="vs", preference="cell",
               interpolate_before_map=False, show_edges=True, edge_color="black")
    p.add_scalar_bar(title="vs"); p.show()

    # pick any radius within your mesh range
    r = 1701.0  # e.g., just below the 660

    sph = spherical_slice_with_cell_scalars(grid, "vs", radius=r)

    p = pv.Plotter()
    p.add_mesh(
        sph,
        scalars="vs",
        preference="cell",            # color by CELL data (one color per triangle)
        interpolate_before_map=False, # no smoothing
        show_edges=True,
        edge_color="black",
    )
    p.add_scalar_bar(title=f"vs @ r={r} km")
    p.show()