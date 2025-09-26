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
    ) -> None:
        if pv is None:  # pragma: no cover
            raise ImportError(
                "PyVista is required. Install with `pip install pyvista`."
            )
        self.mesh = mesh
        self.name = name
        self.metadata = metadata or {}

    # ----- Construction helpers ---------------------------------------------
    @classmethod
    def from_pygmsh_sphere(
        cls, radius_km: float = 6371.0, mesh_size_km: float = 200.0,
        name: str = "sphere"
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
        )

    @classmethod
    def from_file(
        cls, path: str, name: Optional[str] = None
    ) -> "MeshEarthModel":
        """Load a mesh from a file (e.g., .vtu, .vtk, .ply)."""
        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to read mesh files.")
        grid = pv.read(path)
        return cls(mesh=grid, name=name or "loaded-earth")

    # ----- IO ----------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the mesh to disk.

        Path extension determines the writer; .vtu/.vtk work well.
        """
        # Prefer PyVista's native writer for simplicity
        self.mesh.save(path)  # type: ignore[attr-defined]

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
        plotter.show_axes()
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
        earth_radius_km: float = 6371.0,
    ) -> Any:
        """Slice the mesh by the plane of the source-receiver great circle."""
        # Use coordinate utility instead of visualizer helper
        from sensray.utils.coordinates import CoordinateConverter

        src = np.asarray(
            CoordinateConverter.earth_to_cartesian(
                source_lat, source_lon, 0.0, earth_radius=earth_radius_km
            )
        )
        rec = np.asarray(
            CoordinateConverter.earth_to_cartesian(
                receiver_lat, receiver_lon, 0.0, earth_radius=earth_radius_km
            )
        )
        normal = np.cross(src, rec)
        normal = normal / np.linalg.norm(normal)
        # Slice through origin
        return self.mesh.slice(normal=normal, origin=(0.0, 0.0, 0.0))
