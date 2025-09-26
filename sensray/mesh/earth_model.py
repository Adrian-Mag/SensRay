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

    def add_points(
        self,
        plotter: Any,
        points_xyz: Union[np.ndarray, Iterable[Iterable[float]]],
        color: str = "red",
        point_size: float = 8.0,
    ) -> None:
        """Overlay points (x, y, z) on an existing plotter."""
        pts = np.asarray(points_xyz, dtype=float)
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
        )
        if show_wireframe:
            plotter.add_mesh(
                surface,
                style="wireframe",
                color=wireframe_color,
                line_width=wireframe_line_width,
                opacity=1.0,
            )
        plotter.add_scalar_bar(title=scalar_name)  # type: ignore[attr-defined]
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
