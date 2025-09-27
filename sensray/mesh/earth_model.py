"""
Mesh-backed Earth model for SensRay.

A minimal class that wraps a PyVista mesh so it can be created,
saved, loaded, and visualized. This is a foundation for custom
3D Earth models based on unstructured meshes.
"""

from __future__ import annotations

from typing import Optional, Dict, Iterable, Any, Union

import numpy as np

try:  # Optional heavy deps
    import pyvista as pv
except Exception as exc:  # pragma: no cover - optional
    pv = None  # type: ignore[assignment]
    _pv_err = exc  # noqa: F841

# No TYPE_CHECKING block needed; use loose typing (Any) to avoid heavy imports


class MeshEarthModel:
    """A simple mesh-backed Earth model.

    Parameters
    ----------
    mesh : pyvista.DataSet
        A PyVista mesh (e.g., UnstructuredGrid or PolyData).
    name : str
        Optional model name.
    metadata : dict, optional
        Arbitrary metadata dictionary.
    """

    def __init__(
        self,
        mesh: Any,
        name: str = "custom-earth",
        metadata: Optional[Dict] = None,
        radius_km: float = 6371.0,
        cell_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if pv is None:  # pragma: no cover
            raise ImportError(
                "PyVista is required. Install with `pip install pyvista`."
            )
        self.mesh = mesh
        self.name = name
        self.metadata = metadata or {}
        self.radius_km = float(radius_km)
        # Optionally attach initial cell-data arrays
        if cell_data:
            for key, arr in cell_data.items():
                self.set_cell_data(key, np.asarray(arr))

    # ----- Construction helpers ---------------------------------------------
    @classmethod
    def from_pygmsh_sphere(
        cls,
        radius_km: float = 6371.0,
        mesh_size_km: float = 200.0,
        name: str = "sphere",
        cell_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> "MeshEarthModel":
        """Create a tetrahedral sphere using pygmsh -> PyVista.

        Notes
        -----
        Requires pygmsh and meshio. The mesh is volumetric (tets).
        """
        try:
            import pygmsh  # type: ignore
            import meshio  # type: ignore
            import tempfile
            import os
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Creating meshes requires `pygmsh` and `meshio`."
            ) from exc

        if pv is None:  # pragma: no cover
            raise ImportError(
                "PyVista is required to construct meshes. Install pyvista."
            )

        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_min = mesh_size_km
            geom.characteristic_length_max = mesh_size_km
            geom.add_ball([0.0, 0.0, 0.0], radius_km)
            gmsh_mesh = geom.generate_mesh()

            # Convert gmsh -> vtu -> pyvista
            tmp = tempfile.NamedTemporaryFile(suffix=".vtu", delete=False)
            tmp.close()
            meshio.write(tmp.name, gmsh_mesh)
            grid = pv.read(tmp.name)
            os.remove(tmp.name)

        return cls(
            mesh=grid,
            name=name,
            metadata={
                "radius_km": radius_km,
                "mesh_size_km": mesh_size_km,
            },
            radius_km=radius_km,
            cell_data=cell_data,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        name: Optional[str] = None,
        radius_km: float = 6371.0,
        cell_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> "MeshEarthModel":
        """Load a mesh from a file (e.g., .vtu, .vtk, .ply).

        If the file doesn't encode radius, pass `radius_km` explicitly.
        """
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to read mesh files.")
        grid = pv.read(path)
        return cls(
            mesh=grid,
            name=name or "loaded-earth",
            radius_km=radius_km,
            cell_data=cell_data,
        )

    # ----- IO ----------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the mesh to disk.

        Path extension determines the writer; .vtu/.vtk work well.
        """
        # Prefer PyVista's native writer for simplicity
        self.mesh.save(path)  # type: ignore[attr-defined]

    # ----- Scalars (cell data) ----------------------------------------------
    def set_cell_data(self, name: str, values: np.ndarray) -> None:
        """Attach a scalar array to cells (e.g., vs per cell).

        Parameters
        ----------
        name : str
            Array name (e.g., "vs").
        values : np.ndarray
            1D array of length equal to number of cells.
        """
        arr = np.asarray(values)
        n_cells = self.mesh.n_cells  # type: ignore[attr-defined]
        if arr.ndim != 1 or arr.shape[0] != n_cells:
            raise ValueError(
                f"Cell data must be 1D with length {n_cells}, got {arr.shape}"
            )
        self.mesh.cell_data[name] = arr  # type: ignore[attr-defined]

    def get_cell_data(self, name: str) -> np.ndarray:
        """Return a named cell-data array."""
        return self.mesh.cell_data[name]  # type: ignore[attr-defined]

    def list_cell_data(self) -> Iterable[str]:
        """List available cell-data array names."""
        return list(self.mesh.cell_data.keys())  # type: ignore[attr-defined]

    # ----- Visualization -----------------------------------------------------
    def plot(
        self,
        show_edges: bool = False,
        color: str = "lightgray",
        opacity: float = 1.0,
        notebook: bool = False,
    ) -> Any:
        """Plot the mesh in 3D and return the Plotter."""
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to plot meshes.")
        plotter = pv.Plotter(notebook=notebook)
        plotter.add_mesh(
            self.mesh,
            color=color,
            show_edges=show_edges,
            opacity=opacity,
        )
        return plotter

    # ----- Geometry utilities ---------------------------------------------
    def cell_type_summary(self) -> Dict[int, int]:
        """Return a mapping of VTK cell type id -> count.

        Useful to see what kinds of elements are present (e.g., tets, tris).
        """
        try:
            import numpy as _np
            types = _np.asarray(
                self.mesh.celltypes
            )  # type: ignore[attr-defined]
        except Exception:
            return {}
        out: Dict[int, int] = {}
        for t in _np.unique(types):
            out[int(t)] = int((types == t).sum())
        return out

    def filter_to_tetrahedra(self, inplace: bool = True) -> Any:
        """Keep only tetrahedral cells from the mesh.

        Parameters
        ----------
        inplace : bool
            If True, updates this model's mesh in place and returns it.
            If False, returns a new PyVista dataset with only tets.
        """
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required for mesh operations.")
        from pyvista import _vtk as vtk  # type: ignore
        import numpy as _np

        try:
            types = _np.asarray(
                self.mesh.celltypes
            )  # type: ignore[attr-defined]
        except Exception:
            # Nothing to do
            return self.mesh
        tet_ids = _np.where(types == vtk.VTK_TETRA)[0]
        if tet_ids.size == 0:
            return self.mesh
        tet_only = self.mesh.extract_cells(
            tet_ids
        )  # type: ignore[attr-defined]
        if inplace:
            self.mesh = tet_only
            return self.mesh
        return tet_only

    @staticmethod
    def _normalize_points_array(points_xyz: Any) -> np.ndarray:
        """Return an (N, 3) float array from various point representations.

        Accepts Nx3, 3xN, or 1D 3N arrays. Does not handle dicts here.
        """
        arr = np.asarray(points_xyz)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise ValueError(
                    "1D points array length must be "
                    "multiple of 3"
                )
            return arr.reshape(-1, 3).astype(float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(float, copy=False)
        if arr.ndim == 2 and arr.shape[0] == 3:
            return arr.T.astype(float, copy=False)
        raise ValueError(
            "Points must have shape (N,3) or (3,N) or flat 1D of 3N"
        )

    # ----- 1D model mapping -------------------------------------------------
    def _depths_from_points(self, pts: np.ndarray) -> np.ndarray:
        """Compute depth (km) from Cartesian points using model radius."""
        r = np.linalg.norm(pts, axis=1)
        return self.radius_km - r

    def _cell_centers_points(self) -> np.ndarray:
        """Return (n_cells, 3) array of cell centers (km)."""
        return self.mesh.cell_centers().points  # type: ignore[attr-defined]

    def _sample_1d_property_at_depths(
        self,
        model_name: str,
        property_name: str,
        depths_km: np.ndarray,
        max_depth_km: Optional[float] = None,
    ) -> np.ndarray:
        """Sample a TauP 1D property at given depths via linear interp."""
        from sensray.core.earth_models import EarthModelManager

        mgr = EarthModelManager()
        prof = mgr.get_1d_profile(
            model_name, properties=[property_name], max_depth_km=max_depth_km
        )
        d = np.asarray(prof["depth_km"], dtype=float)
        v = np.asarray(prof[property_name], dtype=float)
        # Clamp depths to profile bounds
        depths = np.asarray(depths_km, dtype=float)
        depths = np.clip(depths, float(np.nanmin(d)), float(np.nanmax(d)))
        return np.interp(depths, d, v)

    def add_scalar_from_1d_model(
        self,
        model_name: str,
        property_name: str,
        where: str = "cell",
        method: str = "center",
        max_depth_km: Optional[float] = None,
    ) -> np.ndarray:
        """Attach a scalar from a TauP 1D Earth model onto this mesh.

        Parameters
        ----------
        model_name : str
            TauP model name (e.g., 'prem', 'iasp91', 'ak135').
        property_name : str
            One of 'vp','vs','rho','density','qp','qs'.
        where : str
            'cell' (default) or 'point' to choose data location.
        method : str
            For 'cell': 'center' (sample at cell centers) or 'average'
            (average property at cell's corner depths). For 'point':
            'point' (sample at point depths).
        max_depth_km : float, optional
            Clip the 1D model to this depth.
        """
        where_l = where.lower()
        method_l = method.lower()

        if where_l not in ("cell", "point"):
            raise ValueError("where must be 'cell' or 'point'")

        if where_l == "point":
            pts = np.asarray(self.mesh.points, dtype=float)  # type: ignore
            depths = self._depths_from_points(pts)
            vals = self._sample_1d_property_at_depths(
                model_name, property_name, depths, max_depth_km
            )
            self.mesh.point_data[property_name] = vals  # type: ignore
            return vals

        # Cell data
        centers = self._cell_centers_points()
        if method_l == "center":
            depths = self._depths_from_points(centers)
            vals = self._sample_1d_property_at_depths(
                model_name, property_name, depths, max_depth_km
            )
        elif method_l == "average":
            # Approximate by averaging values at cell corner depths
            # Get connectivity to fetch cell point ids
            # Using PyVista convenience: extract cell points per id
            vals = np.zeros(self.mesh.n_cells, dtype=float)  # type: ignore

            for cid in range(self.mesh.n_cells):  # type: ignore
                cell = self.mesh.extract_cells(cid)
                cpts = np.asarray(cell.points, dtype=float)  # type: ignore
                d = self._depths_from_points(cpts)
                v = self._sample_1d_property_at_depths(
                    model_name, property_name, d, max_depth_km
                )
                vals[cid] = float(np.nanmean(v))
        else:
            raise ValueError("Unsupported method for cell mapping")

        self.mesh.cell_data[property_name] = vals  # type: ignore
        return vals

    def add_scalars_from_1d_model(
        self,
        model_name: str,
        properties: Iterable[str] = ("vp", "vs", "rho"),
        where: str = "cell",
        method: str = "center",
        max_depth_km: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Attach multiple scalars from a TauP 1D model to mesh."""
        out: Dict[str, np.ndarray] = {}
        for p in properties:
            out[p] = self.add_scalar_from_1d_model(
                model_name,
                p,
                where=where,
                method=method,
                max_depth_km=max_depth_km,
            )
        return out

    def ensure_cell_scalar(
        self,
        name: str,
        strategy: str = "point_to_cell",
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Ensure a cell_data scalar exists; create if necessary.

        Order:
        - If already in cell_data, return it.
        - If in point_data and strategy='point_to_cell', convert using
          point_data_to_cell_data and copy that array.
        - Else, if model_name provided, populate from 1D model using
          add_scalar_from_1d_model(model_name, name, where='cell').
        - Else, raise KeyError.
        """
        try:
            return self.mesh.cell_data[name]  # type: ignore
        except Exception:
            pass

        if strategy == "point_to_cell" and name in getattr(
            self.mesh, "point_data", {}
        ):
            ds = self.mesh.point_data_to_cell_data()  # type: ignore
            arr = np.asarray(ds.cell_data[name], dtype=float)  # type: ignore
            self.mesh.cell_data[name] = arr  # type: ignore
            return arr

        if model_name is not None:
            arr = self.add_scalar_from_1d_model(
                model_name, name, where="cell", method="center"
            )
            return arr

        raise KeyError(
            f"Cell scalar '{name}' not found and cannot be created."
        )

    # ----- Sensitivity kernels ---------------------------------------------
    def compute_sensitivity_kernel(
        self,
        ray_points_xyz: Any,
        property_name: str,
        attach_name: Optional[str] = None,
        epsilon: float = 0.0,
        tol: float = 1e-6,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute sensitivity kernel K = -L / (prop^2 + epsilon) per cell.

        Parameters
        ----------
        ray_points_xyz : array-like
            Ray polyline points (N,3) in km Cartesian.
        property_name : str
            'vp' or 'vs' (or other scalar) expected in cell_data.
        attach_name : str, optional
            If provided, store result as this cell_data name (default
            is f"K_{property_name}").
        epsilon : float
            Small regularizer added to denominator to avoid division by zero.
        tol : float
            Geometric tolerance for path length calculation.
        model_name : str, optional
            If given and property is missing, populate from this 1D model.
        """
        lengths = self.compute_ray_cell_path_lengths(ray_points_xyz, tol=tol)
        prop = self.ensure_cell_scalar(
            property_name, strategy="point_to_cell", model_name=model_name
        ).astype(float, copy=False)

        denom = prop * prop
        if epsilon > 0.0:
            denom = denom + float(epsilon)

        # Mask non-finite or non-positive denominators
        valid = np.isfinite(denom) & (denom > 0.0)
        kernel = np.zeros_like(lengths, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel[valid] = -lengths[valid] / denom[valid]

        if attach_name is None:
            attach_name = f"K_{property_name}"
        self.mesh.cell_data[attach_name] = kernel  # type: ignore
        return kernel

    def compute_sensitivity_kernels_for_rays(
        self,
        rays_points_list: Iterable[Any],
        property_name: str,
        attach_name: Optional[str] = None,
        accumulate: Optional[str] = "sum",
        epsilon: float = 0.0,
        tol: float = 1e-6,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute kernels for multiple rays and optionally accumulate.

        Parameters mirror compute_sensitivity_kernel, but accept a list of
        ray polylines. If accumulate='sum' (default), returns a single
        1D array and stores under attach_name or 'Ksum_{property}'. If
        accumulate is None, returns a (n_rays, n_cells) stack and stores
        per-ray arrays with suffixes if attach_name is provided.
        """
        # Ensure property exists once
        _ = self.ensure_cell_scalar(
            property_name, strategy="point_to_cell", model_name=model_name
        )

        kernels = []
        for i, pts in enumerate(rays_points_list):
            k = self.compute_sensitivity_kernel(
                pts,
                property_name,
                attach_name=None,
                epsilon=epsilon,
                tol=tol,
                model_name=None,
            )
            kernels.append(k)

        K = np.vstack(kernels)
        if accumulate == "sum":
            Ksum = K.sum(axis=0)
            name = attach_name or f"Ksum_{property_name}"
            self.mesh.cell_data[name] = Ksum  # type: ignore
            return Ksum

        if attach_name is not None:
            for i, arr in enumerate(kernels):
                self.mesh.cell_data[f"{attach_name}_{i}"] = arr  # type: ignore
        return K

    @staticmethod
    def _clip_segment_by_tetra(
        tet_pts: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        tol: float = 1e-8,
    ) -> float:
        """Length of segment inside a tetrahedron.

        Computes the length of segment p0->p1 inside the tetra defined by
        tet_pts (4x3), using half-space clipping against the four faces.

        Returns 0.0 if no intersection.
        """
        # Define 4 faces by vertex indices
        faces = (
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        )
        d = p1 - p0
        seg_len = float(np.linalg.norm(d))
        if seg_len == 0.0:
            return 0.0
        # Work with param t in [0,1]
        t_min, t_max = 0.0, 1.0
        centroid = tet_pts.mean(axis=0)
        for ia, ib, ic in faces:
            a, b, c = tet_pts[ia], tet_pts[ib], tet_pts[ic]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm == 0.0:
                # Degenerate face; skip
                continue
            n = n / n_norm
            # Ensure outward such that centroid is inside:
            # we want inside as n·(x-a) <= 0
            if np.dot(n, centroid - a) > 0:
                n = -n
            num = -np.dot(n, p0 - a)
            den = np.dot(n, d)
            if abs(den) < tol:
                # Segment parallel to plane; reject if outside
                if np.dot(n, p0 - a) > tol:
                    return 0.0
                # Else, inequality holds for entire segment for this plane
                continue
            t_hit = num / den
            if den > 0:
                # entering constraint: t <= t_hit
                t_max = min(t_max, t_hit)
            else:
                # leaving constraint: t >= t_hit
                t_min = max(t_min, t_hit)
            if t_min - t_max > tol:
                return 0.0
        # Clamp to [0,1]
        t0 = max(0.0, min(1.0, t_min))
        t1 = max(0.0, min(1.0, t_max))
        if t1 <= t0:
            return 0.0
        return (t1 - t0) * seg_len

    def compute_ray_cell_path_lengths(
        self,
        ray_points_xyz: Any,
        attach_name: Optional[str] = None,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Compute per-cell path length of a ray polyline through a tet mesh.

        Parameters
        ----------
        ray_points_xyz : array-like
            Sequence of points (N,3) defining the ray polyline in Cartesian km.
            If provided as 3xN or flat 1D 3N, it will be normalized.
        attach_name : str, optional
            If given, stores the resulting 1D array to cell_data[attach_name].
        tol : float
            Geometric tolerance for locator queries and clipping checks.

        Returns
        -------
        lengths : np.ndarray
            1D array of length n_cells with accumulated path length per cell.

        Notes
        -----
        - Currently supports tetrahedral meshes. For other cell types, emits a
          warning and returns zeros.
        - Efficient for long rays: uses vtkStaticCellLocator to prune tests.
        """
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required for geometric queries.")
        from pyvista import _vtk as vtk  # type: ignore
        import warnings

        # Validate mesh: proceed if there are any tetrahedra; skip others
        try:
            celltypes = np.asarray(
                self.mesh.celltypes  # type: ignore[attr-defined]
            )
        except Exception:
            celltypes = None
        if celltypes is None:
            warnings.warn(
                "Mesh has no celltypes information; cannot compute path "
                "lengths.",
                UserWarning,
            )
            zeros = np.zeros(getattr(self.mesh, "n_cells", 0), dtype=float)
            if attach_name:
                self.mesh.cell_data[
                    attach_name
                ] = zeros  # type: ignore[attr-defined]
            return zeros
        tet_mask = celltypes == vtk.VTK_TETRA
        if not np.any(tet_mask):
            warnings.warn(
                "Mesh contains no tetrahedral cells; returning zeros.",
                UserWarning,
            )
            zeros = np.zeros(getattr(self.mesh, "n_cells", 0), dtype=float)
            if attach_name:
                self.mesh.cell_data[
                    attach_name
                ] = zeros  # type: ignore[attr-defined]
            return zeros

        # Normalize points to (N,3)
        pts = self._normalize_points_array(ray_points_xyz)
        n = len(pts)
        n_cells = self.mesh.n_cells  # type: ignore[attr-defined]
        lengths = np.zeros(n_cells, dtype=float)
        if n < 2:
            if attach_name:
                self.mesh.cell_data[
                    attach_name
                ] = lengths  # type: ignore[attr-defined]
            return lengths

        # Build spatial locator
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(self.mesh)
        locator.BuildLocator()

        id_list = vtk.vtkIdList()
        # Cache for tetra points to avoid repeated extraction cost
        tet_pts_cache: Dict[int, np.ndarray] = {}
        for i in range(n - 1):
            p0 = np.asarray(pts[i], dtype=float)
            p1 = np.asarray(pts[i + 1], dtype=float)
            if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
                continue
            if np.allclose(p0, p1):
                continue
            id_list.Reset()
            # Find candidate cells intersected by the segment
            locator.FindCellsAlongLine(p0, p1, tol, id_list)
            # For each candidate, clip and accumulate
            for k in range(id_list.GetNumberOfIds()):
                cid = id_list.GetId(k)
                # Skip non-tetrahedral candidate cells
                if cid < 0 or cid >= n_cells or not tet_mask[cid]:
                    continue
                # Extract cell points (should be 4 points for tetra)
                tet_pts = tet_pts_cache.get(cid)
                if tet_pts is None:
                    cell = self.mesh.extract_cells(
                        cid
                    )  # type: ignore[attr-defined]
                    tet_pts = np.asarray(
                        cell.points, dtype=float
                    )  # type: ignore[attr-defined]
                    # Defensive: some extra points can be present;
                    # try to select unique
                    if tet_pts.shape[0] > 4:
                        # Use the first 4 unique rows
                        _, idx = np.unique(
                            tet_pts, axis=0, return_index=True
                        )
                        tet_pts = tet_pts[np.sort(idx)[:4]]
                    tet_pts_cache[cid] = tet_pts
                if tet_pts.shape[0] != 4:
                    continue
                seg_len_in = self._clip_segment_by_tetra(tet_pts, p0, p1, tol)
                if seg_len_in > 0:
                    lengths[cid] += seg_len_in

        if attach_name:
            self.mesh.cell_data[
                attach_name
            ] = lengths  # type: ignore[attr-defined]
        return lengths

    def add_points(
        self,
        plotter: Any,
        points_xyz: Union[
            np.ndarray,
            Iterable[Iterable[float]],
            Dict[str, Any],
        ],
        color: str = "red",
        point_size: float = 8.0,
    ) -> None:
        """Overlay points on an existing plotter.

        Accepts:
        - Nx3 array-like of Cartesian XYZ (km)
        - 3xN array-like (auto-transposed)
        - 1D flat array of length 3N (reshaped)
        - dict with keys {"x","y","z"}
        - dict with geo keys {"lat","lon"} and either "r"/"radius[_km]"
          or "depth[_km]" (converted to Cartesian using model radius)
        """
        pts: np.ndarray

        # Dict inputs: x/y/z or lat/lon + r/depth
        if isinstance(points_xyz, dict):  # type: ignore[unreachable]
            d: Dict[str, Any] = points_xyz  # type: ignore[assignment]
            # Cartesian components
            if all(k in d for k in ("x", "y", "z")):
                x = np.asarray(d["x"], dtype=float)
                y = np.asarray(d["y"], dtype=float)
                z = np.asarray(d["z"], dtype=float)
                if not (x.shape == y.shape == z.shape):
                    raise ValueError("x, y, z must have the same shape")
                pts = np.column_stack([x, y, z])
            # Geographic components -> Cartesian
            elif ("lat" in d or "latitude" in d) and (
                "lon" in d or "longitude" in d or "long" in d
            ):
                lat_val = d.get("lat", d.get("latitude"))
                lon_val = d.get("lon", d.get("longitude", d.get("long")))
                lat = np.asarray(lat_val, dtype=float)
                lon = np.asarray(lon_val, dtype=float)
                r = d.get("r", d.get("radius", d.get("radius_km")))
                depth = d.get("depth", d.get("depth_km"))
                if r is not None:
                    rad = np.asarray(r, dtype=float)
                elif depth is not None:
                    rad = self.radius_km - np.asarray(depth, dtype=float)
                else:
                    rad = np.full_like(lat, self.radius_km, dtype=float)
                # Convert to Cartesian (loop to use lightweight converter)
                from sensray.utils.coordinates import CoordinateConverter

                pts = np.vstack(
                    [
                        CoordinateConverter.earth_to_cartesian(
                            float(la), float(lo), float(rr),
                            earth_radius=self.radius_km,
                        )
                        for la, lo, rr in zip(lat, lon, rad)
                    ]
                ).astype(float)
            else:
                raise ValueError(
                    "Unsupported dict keys for points: expected x/y/z or "
                    "lat/lon with r or depth"
                )
        else:
            # General array-like handling
            arr = np.asarray(points_xyz)
            # If it's object-dtype of 3-tuples, coerce via list()
            if arr.dtype == object and arr.ndim == 1:
                arr = np.asarray(list(arr), dtype=float)
            if arr.ndim == 1:
                if arr.size % 3 != 0:
                    raise ValueError(
                        "1D points array length must be a multiple of 3"
                    )
                pts = arr.reshape(-1, 3).astype(float)
            elif arr.ndim == 2:
                if arr.shape[1] == 3:
                    pts = arr.astype(float, copy=False)
                elif arr.shape[0] == 3:
                    pts = arr.T.astype(float, copy=False)
                else:
                    raise ValueError(
                        "Points array must have shape (N,3) or (3,N)"
                    )
            else:
                raise ValueError("Points must be 1D or 2D array-like")

        plotter.add_points(pts, color=color, point_size=point_size)

    def slice_great_circle(
        self,
        source_lat: float,
        source_lon: float,
        receiver_lat: float,
        receiver_lon: float,
        scalar_name: Optional[str] = None,
    ) -> Any:
        """Slice the mesh by the plane of the source-receiver great circle.

        If scalar_name is provided, convert cell data to point data so the
        resulting slice carries interpolated point scalars for coloring.
        """
        # Use coordinate utility instead of visualizer helper
        from sensray.utils.coordinates import CoordinateConverter

        src = np.asarray(
            CoordinateConverter.earth_to_cartesian(
                source_lat, source_lon, 0.0, earth_radius=self.radius_km
            )
        )
        rec = np.asarray(
            CoordinateConverter.earth_to_cartesian(
                receiver_lat, receiver_lon, 0.0, earth_radius=self.radius_km
            )
        )
        normal = np.cross(src, rec)
        normal = normal / np.linalg.norm(normal)
        # Choose dataset: convert to point data if a scalar is requested
        ds = self.mesh
        if (
            scalar_name is not None
            and scalar_name not in ds.point_data  # type: ignore[attr-defined]
        ):
            ds = ds.cell_data_to_point_data()  # type: ignore[attr-defined]

        # Slice through origin
        return ds.slice(normal=normal, origin=(0.0, 0.0, 0.0))

    def slice_great_circle_with_scalar(
        self,
        source_lat: float,
        source_lon: float,
        receiver_lat: float,
        receiver_lon: float,
        scalar_name: str,
    ) -> Any:
        """Deprecated: use slice_great_circle(..., scalar_name=...)."""
        import warnings

        warnings.warn(
            "slice_great_circle_with_scalar is deprecated; "
            "use slice_great_circle(..., scalar_name=...) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.slice_great_circle(
            source_lat,
            source_lon,
            receiver_lat,
            receiver_lon,
            scalar_name=scalar_name,
        )

    def sphere_surface_with_scalar(
        self,
        radius_km: float,
        scalar_name: str,
        theta_res: int = 90,
        phi_res: int = 180,
    ) -> Any:
        """Sample volumetric scalar values onto a spherical surface.

        Returns a PolyData sphere with the requested scalar as point data.
        """
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to build surfaces.")

        # Surface geometry
        sphere = pv.Sphere(
            radius=radius_km,
            theta_resolution=theta_res,
            phi_resolution=phi_res,
        )

        # Ensure we have point data for sampling
        ds = self.mesh
        if scalar_name not in ds.point_data:  # type: ignore[attr-defined]
            ds = ds.cell_data_to_point_data()  # type: ignore[attr-defined]

        # Sample/interpolate scalar onto the sphere's points
        sampled = sphere.sample(ds)  # type: ignore[attr-defined]
        return sampled

    def plot_surface(
        self,
        surface: Any,
        scalar_name: str,
        cmap: str = "viridis",
        clim: Optional[tuple] = None,
        notebook: bool = False,
        show_edges: bool = False,
        show_wireframe: bool = False,
        wireframe_color: str = "black",
        wireframe_line_width: float = 1.0,
        opacity: float = 1.0,
        nan_opacity: float = 0.0,
    ) -> Any:
        """Plot a surface with a scalar colormap and return the Plotter."""
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to plot surfaces.")
        # Derive color limits from data if not provided
        use_clim = clim
        if use_clim is None and scalar_name in getattr(
            surface, "point_data", {}
        ):
            try:
                arr = np.asarray(
                    surface.point_data[scalar_name]
                )  # type: ignore[attr-defined]
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))
                if np.isfinite(vmin) and np.isfinite(vmax):
                    if vmin == vmax:
                        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
                        use_clim = (vmin, vmin + eps)
                    else:
                        use_clim = (vmin, vmax)
            except Exception:
                use_clim = clim

        plotter = pv.Plotter(notebook=notebook)
        # Use numeric array to avoid ambiguity with active scalars
        scalar_array = None
        try:
            scalar_array = surface.point_data[scalar_name]
        except Exception:
            scalar_array = scalar_name  # fallback to name
        plotter.add_mesh(
            surface,
            scalars=scalar_array,
            cmap=cmap,
            clim=use_clim,
            show_edges=show_edges,
            opacity=opacity,
            nan_opacity=nan_opacity,
            show_scalar_bar=False,
        )
        # Add the scalar bar now so it's linked to the scalar-mapped surface,
        # not the later wireframe actor (which would yield a 0–1 default).
        plotter.add_scalar_bar(title=scalar_name)  # type: ignore[attr-defined]
        if show_wireframe:
            plotter.add_mesh(
                surface,
                style="wireframe",
                color=wireframe_color,
                line_width=wireframe_line_width,
                opacity=1.0,
            )
        return plotter

    # ----- High-level plotting helpers -------------------------------------
    def plot_slice(
        self,
        source_lat: float,
        source_lon: float,
        receiver_lat: float,
        receiver_lon: float,
        scalar_name: Optional[str] = None,
        cmap: str = "viridis",
        clim: Optional[tuple] = None,
        notebook: bool = False,
        show_edges: bool = False,
        wireframe: bool = False,
        wireframe_color: str = "black",
        wireframe_line_width: float = 1.0,
        opacity: float = 1.0,
    ) -> Any:
        """Compute a great-circle slice and plot it.

        If `scalar_name` is provided the slice will carry interpolated point
        scalars and the plot will use a colormap; otherwise the surface is
        plotted with a single color.
        """
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to plot surfaces.")

        surface = self.slice_great_circle(
            source_lat,
            source_lon,
            receiver_lat,
            receiver_lon,
            scalar_name=scalar_name,
        )

        if scalar_name is not None:
            return self.plot_surface(
                surface,
                scalar_name=scalar_name,
                cmap=cmap,
                clim=clim,
                notebook=notebook,
                show_edges=show_edges,
                show_wireframe=wireframe,
                wireframe_color=wireframe_color,
                wireframe_line_width=wireframe_line_width,
                opacity=opacity,
            )

        # No scalar: render wireframe-only
        plotter = pv.Plotter(notebook=notebook)
        plotter.add_mesh(
            surface,
            style="wireframe",
            color=wireframe_color,
            line_width=wireframe_line_width,
        )
        return plotter

    def plot_sphere(
        self,
        radius_km: float,
        scalar_name: Optional[str] = None,
        cmap: str = "viridis",
        clim: Optional[tuple] = None,
        notebook: bool = False,
        show_edges: bool = False,
        wireframe: bool = False,
        wireframe_color: str = "black",
        wireframe_line_width: float = 1.0,
        opacity: float = 1.0,
        nan_opacity: float = 1.0,
    ) -> Any:
        """Plot the model on a spherical shell derived from the mesh itself.

        This uses an iso-surface (contour) of radial distance on the
        volumetric mesh so both the surface geometry and any optional
        wireframe reflect the underlying mesh discretization.
        """
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to plot surfaces.")

        # Prepare a working copy and a per-point radius array
        ds = self.mesh
        if (
            scalar_name is not None
            and scalar_name not in ds.point_data  # type: ignore[attr-defined]
        ):
            ds = ds.cell_data_to_point_data()  # type: ignore[attr-defined]
        base = ds.copy()
        base["__radius__"] = np.linalg.norm(base.points, axis=1)

        # Extract the spherical shell surface from the underlying mesh
        shell = base.contour(isosurfaces=[radius_km], scalars="__radius__")

        # If the extracted shell doesn't already carry the requested
        # scalar as point-data, sample the volumetric dataset so the
        # shell gets interpolated scalar values. This prevents
        # `nan_opacity` from making the entire shell transparent when
        # the scalar is missing on the contour geometry.
        if (
            scalar_name is not None
            and scalar_name not in getattr(shell, "point_data", {})
        ):
            try:
                shell = shell.sample(ds)  # type: ignore[attr-defined]
            except Exception:
                # Sampling failed; continue and let downstream logic
                # handle missing data (will result in NaNs).
                pass

        if scalar_name is not None:
            # If no clim provided, derive it from the shell's scalar values
            local_clim = None
            try:
                arr = np.asarray(
                    shell.point_data[scalar_name]
                )  # type: ignore[attr-defined]
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))
                if np.isfinite(vmin) and np.isfinite(vmax):
                    if vmin == vmax:
                        # Avoid degenerate range; widen slightly
                        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
                        local_clim = (vmin, vmin + eps)
                    else:
                        local_clim = (vmin, vmax)
            except Exception:
                local_clim = None

            return self.plot_surface(
                shell,
                scalar_name=scalar_name,
                cmap=cmap,
                clim=clim if clim is not None else local_clim,
                notebook=notebook,
                show_edges=show_edges,
                show_wireframe=wireframe,
                wireframe_color=wireframe_color,
                wireframe_line_width=wireframe_line_width,
                opacity=opacity,
                nan_opacity=nan_opacity,
            )

        # No scalar: draw mesh-based wireframe only
        plotter = pv.Plotter(notebook=notebook)
        plotter.add_mesh(
            shell,
            style="wireframe",
            color=wireframe_color,
            line_width=wireframe_line_width,
        )
        return plotter
