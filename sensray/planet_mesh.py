"""
PlanetMesh class for SensRay.

Provides mesh generation and visualization for PlanetModel instances.
Supports octree and tetrahedral mesh generation with discontinuity-aware
refinement and unified visualization methods.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import PlanetModel
import numpy as np
import warnings

try:  # Optional heavy deps
    import pyvista as pv
    import discretize
except Exception as exc:  # pragma: no cover - optional
    pv = None  # type: ignore[assignment]
    discretize = None  # type: ignore[assignment]
    _mesh_err = exc  # noqa: F841


class PlanetMesh:
    """
    Mesh representation of a PlanetModel.

    Handles octree and tetrahedral mesh generation based on model
    discontinuities, property mapping from model to mesh cells,
    and unified visualization with plane-based clipping.

    Parameters
    ----------
    planet_model : PlanetModel
        The planet model to create a mesh for
    mesh_type : str
        Type of mesh to generate: "octree" or "tetrahedral"

    Examples
    --------
    >>> model = PlanetModel.from_standard_model("prem")
    >>> mesh = PlanetMesh(model)
    >>> mesh.generate_octree_mesh()
    >>> mesh.populate_properties(["vp", "vs"])
    >>> mesh.plot_cross_section(property_name="vp")
    """

    def __init__(self, planet_model, mesh_type: str = "octree"):
        if pv is None or discretize is None:  # pragma: no cover
            raise ImportError(
                "PyVista and discretize are required for mesh operations. "
                "Install with `pip install pyvista discretize`."
            )

        self.planet_model = planet_model
        self.mesh_type = mesh_type
        self.mesh = None  # PyVista UnstructuredGrid
        self._tree_mesh = None  # discretize.TreeMesh (for octree only)

    # ===== Mesh Generation =====

    def generate_octree_mesh(self,
                             base_cells: int = 256,
                             max_level: Optional[int] = None,
                             buffer_km: float = 2500.0,
                             refinement_levels: Optional[Dict[float, int]] = None
                             ) -> None:
        """
        Generate octree mesh based on user-defined depth ranges.

        Parameters
        ----------
        base_cells : int
            Base number of cells per axis (should be power of 2)
        max_level : int, optional
            Maximum refinement level. If None, uses log2(base_cells)
        buffer_km : float
            Buffer zone around planet to avoid boundary artifacts
        refinement_levels : dict, optional
            Custom refinement levels for depth ranges.
            Keys are depths in km from surface (float), values are refinement levels (int).
            Example: {0: 6, 50: 5, 670: 4, 2891: 4} for Earth-like structure.
            If None, uses reasonable Earth-like defaults.
        """
        if discretize is None:
            raise ImportError(
                "discretize is required for octree mesh generation"
            )

        radius = self.planet_model.radius
        domain_size = radius + buffer_km

        if max_level is None:
            max_level = int(np.log2(base_cells))

        # Create base mesh
        hx = [(2 * domain_size / base_cells, base_cells)]
        tree_mesh = discretize.TreeMesh(
            [hx, hx, hx],
            origin=(-domain_size, -domain_size, -domain_size),
            diagonal_balance=True,
        )

        # Generate shells from discontinuities
        shells = self._generate_discontinuity_shells(
            max_level, buffer_km, refinement_levels
        )

        # Refine using shell-based callable
        def shell_level(cell):
            cx, cy, cz = cell.center
            r = (cx*cx + cy*cy + cz*cz)**0.5
            for name, rad, level in shells:
                if r <= rad:
                    return int(level)
            return 0  # outside all shells

        tree_mesh.refine(shell_level)
        tree_mesh.finalize()

        # Convert to PyVista and extract cells inside planet
        grid_all = pv.wrap(tree_mesh.to_vtk())
        centers = grid_all.cell_centers().points
        r_all = np.linalg.norm(centers, axis=1)
        inside = r_all <= radius
        idx = np.where(inside)[0]

        self.mesh = grid_all.extract_cells(idx)
        self._tree_mesh = tree_mesh

        # Add region labels
        centers_in = self.mesh.cell_centers().points
        r_in = np.linalg.norm(centers_in, axis=1)
        regions = np.array(
            [self._get_shell_index(r, shells) for r in r_in], dtype=np.int32
        )
        self.mesh.cell_data["region"] = regions

        print(
            f"Generated octree mesh: {self.mesh.n_cells} cells, "
            f"{self.mesh.n_points} points"
        )

    def generate_tetrahedral_mesh(self,
                                  mesh_size_km: float = 200.0,
                                  **kwargs) -> None:
        """
        Generate tetrahedral mesh using pygmsh.

        Parameters
        ----------
        mesh_size_km : float
            Characteristic mesh size in km
        **kwargs
            Additional arguments passed to pygmsh
        """
        try:
            import pygmsh  # type: ignore
            import meshio  # type: ignore
            import tempfile
            import os
        except ImportError as exc:
            raise ImportError(
                "Tetrahedral mesh generation requires `pygmsh` and `meshio`. "
                "Install with `pip install pygmsh meshio`."
            ) from exc

        radius = self.planet_model.radius

        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_min = mesh_size_km
            geom.characteristic_length_max = mesh_size_km
            geom.add_ball([0.0, 0.0, 0.0], radius)
            gmsh_mesh = geom.generate_mesh()

            # Convert via temporary file
            tmp = tempfile.NamedTemporaryFile(suffix=".vtu", delete=False)
            tmp.close()
            meshio.write(tmp.name, gmsh_mesh)
            grid = pv.read(tmp.name)
            os.remove(tmp.name)

        self.mesh = grid
        print(
            f"Generated tetrahedral mesh: {self.mesh.n_cells} cells, "
            f"{self.mesh.n_points} points"
        )

    # ===== Property Mapping =====

    def populate_properties(self,
                            properties: List[str] = ["vp", "vs", "rho"]
                            ) -> None:
        """
        Populate mesh cells with properties from the planet model.

        Parameters
        ----------
        properties : List[str]
            Properties to sample from the model
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        centers = self.mesh.cell_centers().points
        depths = self.planet_model.radius - np.linalg.norm(centers, axis=1)

        for prop in properties:
            values = np.array([
                self.planet_model.get_property_at_depth(prop, depth)
                for depth in depths
            ], dtype=np.float32)
            self.mesh.cell_data[prop] = values

        print(f"Populated properties: {properties}")

    # ===== Ray Tracing =====

    def compute_ray_lengths(self, ray: Any) -> np.ndarray:
        """
        Compute per-cell path lengths from an ObsPy ray object.

        Parameters
        ----------
        ray : obspy.taup.ray_paths.RayPath
            Ray from model.get_ray_paths_geo() containing path with
            depth, lat, lon

        Returns
        -------
        np.ndarray
            Per-cell path lengths in km
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Extract xyz coordinates from ray path
        path_points = []
        for point in ray.path:
            depth_km = point['depth']  # km below surface
            lat_deg = point['lat']     # degrees
            lon_deg = point['lon']     # degrees

            # Convert to Cartesian coordinates
            from .coordinates import CoordinateConverter
            xyz = CoordinateConverter.earth_to_cartesian(
                lat_deg, lon_deg, depth_km,
                earth_radius=self.planet_model.radius
            )
            path_points.append(xyz)

        path_points = np.array(path_points)

        # Compute per-cell intersections
        if self.mesh_type == "octree":
            return self._compute_polyline_cell_lengths_octree(path_points)
        else:
            return self._compute_polyline_cell_lengths_tetrahedral(path_points)

    def compute_multiple_ray_lengths(self, rays: List[Any]) -> np.ndarray:
        """
        Compute per-cell lengths for multiple rays.

        Parameters
        ----------
        rays : List[RayPath]
            List of ObsPy ray objects

        Returns
        -------
        np.ndarray
            Array of shape (n_rays, n_cells) with per-cell lengths
        """
        lengths = []
        for ray in rays:
            lengths.append(self.compute_ray_lengths(ray))
        return np.array(lengths)

    def compute_ray_lengths_from_arrival(
        self,
        arrival: Any,
        store_as: Optional[str] = None,
        replace_existing: bool = True
    ) -> np.ndarray:
        """
        Compute per-cell path lengths from an ObsPy Arrival object and
        optionally store as mesh cell data.

        Parameters
        ----------
        arrival : obspy.core.event.origin.Arrival or ray object
            Arrival object or ray object with 'path' attribute containing
            ray path points with 'depth', 'lat', 'lon' keys
        store_as : str, optional
            If provided, store the computed lengths as cell data with this name.
            If None, lengths are computed but not stored.
        replace_existing : bool
            If True, replace existing cell data with the same name.
            If False and cell data exists, raise ValueError.

        Returns
        -------
        np.ndarray
            Per-cell path lengths in km

        Examples
        --------
        >>> rays = model.taupy_model.get_ray_paths_geo(
        ...     source_depth_in_km=100, source_latitude_in_deg=0,
        ...     source_longitude_in_deg=0, receiver_latitude_in_deg=30,
        ...     receiver_longitude_in_deg=30, phase_list=["P"])
        >>> if rays:
        ...     lengths = mesh.compute_ray_lengths_from_arrival(
        ...         rays[0], store_as="P_ray_lengths")
        ...     # Now you can visualize with property_name="P_ray_lengths"
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Extract xyz coordinates from arrival/ray path
        path_points = []

        # Handle different types of input objects
        if hasattr(arrival, 'path'):
            # This is a ray object with path attribute
            ray_path = arrival.path
        elif hasattr(arrival, 'ray'):
            # This might be an arrival with a ray attribute
            ray_path = arrival.ray.path
        else:
            # Try to treat as path directly
            ray_path = arrival

        for point in ray_path:
            if isinstance(point, dict):
                depth_km = point['depth']  # km below surface
                lat_deg = point['lat']     # degrees
                lon_deg = point['lon']     # degrees
            else:
                # Handle numpy structured arrays from TauP
                try:
                    depth_km = float(point['depth'])  # km below surface
                    lat_deg = float(point['lat'])     # degrees
                    lon_deg = float(point['lon'])     # degrees
                except (KeyError, TypeError):
                    # Fallback: assume point has attributes depth, lat, lon
                    depth_km = point.depth
                    lat_deg = point.lat
                    lon_deg = point.lon

            # Convert to Cartesian coordinates
            from .coordinates import CoordinateConverter
            xyz = CoordinateConverter.earth_to_cartesian(
                lat_deg, lon_deg, depth_km,
                earth_radius=self.planet_model.radius
            )
            path_points.append(xyz)

        path_points = np.array(path_points)

        # Compute per-cell intersections using appropriate method
        if self.mesh_type == "octree":
            lengths = self._compute_polyline_cell_lengths_octree(path_points)
        else:
            lengths = self._compute_polyline_cell_lengths_tetrahedral(path_points)

        # Store as cell data if requested
        if store_as is not None:
            if store_as in self.mesh.cell_data and not replace_existing:
                raise ValueError(
                    f"Cell data '{store_as}' already exists. "
                    "Set replace_existing=True to overwrite."
                )
            self.mesh.cell_data[store_as] = lengths.astype(np.float32)
            print(f"Stored ray path lengths as cell data: '{store_as}'")

        return lengths

    def add_ray_to_mesh(
        self,
        arrival: Any,
        ray_name: str,
        phase_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Convenience method to compute and store ray path lengths with a
        standardized naming convention.

        Parameters
        ----------
        arrival : obspy ray object
            Ray object from TauP with path information
        ray_name : str
            Base name for the ray (e.g., "P_wave", "S_wave", "ray_1")
        phase_name : str, optional
            Phase name to include in cell data name. If None, tries to
            extract from arrival object.

        Returns
        -------
        np.ndarray
            Per-cell path lengths in km

        Examples
        --------
        >>> rays = model.taupy_model.get_ray_paths_geo(..., phase_list=["P", "S"])
        >>> mesh.add_ray_to_mesh(rays[0], "primary")  # Stores as "ray_primary_lengths"
        >>> mesh.add_ray_to_mesh(rays[1], "secondary", "S")  # Stores as "ray_secondary_S_lengths"
        """
        # Try to extract phase name if not provided
        if phase_name is None:
            if hasattr(arrival, 'phase'):
                if hasattr(arrival.phase, 'name'):
                    phase_name = str(arrival.phase.name)
                else:
                    phase_name = str(arrival.phase)
            elif hasattr(arrival, 'name'):
                phase_name = str(arrival.name)

        # Build cell data name
        if phase_name:
            cell_data_name = f"ray_{ray_name}_{phase_name}_lengths"
        else:
            cell_data_name = f"ray_{ray_name}_lengths"

        # Compute and store
        lengths = self.compute_ray_lengths_from_arrival(
            arrival, store_as=cell_data_name, replace_existing=True
        )

        return lengths

    # ===== Sensitivity Kernels =====

    def compute_sensitivity_kernel(
        self,
        arrival: Any,
        property_name: str,
        attach_name: Optional[str] = None,
        epsilon: float = 1e-6,
        replace_existing: bool = True
    ) -> np.ndarray:
        """
        Compute sensitivity kernel K = -L / (prop^2 + epsilon) per cell.

        Parameters
        ----------
        arrival : obspy ray object or array-like
            Ray object from TauP with path information, or array of
            Cartesian points (N,3)
        property_name : str
            Name of seismic property ('vp', 'vs', 'rho', etc.)
            Must exist in mesh.cell_data or will be populated from model
        attach_name : str, optional
            Name to store kernel as cell data. If None, uses f"K_{property_name}"
        epsilon : float
            Small regularizer added to denominator to avoid division by zero
        replace_existing : bool
            Whether to replace existing cell data with same name

        Returns
        -------
        np.ndarray
            Sensitivity kernel array (length n_cells)

        Examples
        --------
        >>> rays = model.taupy_model.get_ray_paths_geo(..., phase_list=["P"])
        >>> kernel = mesh.compute_sensitivity_kernel(rays[0], 'vp', attach_name='K_P')
        >>> # Visualize with: mesh.plot_cross_section(property_name='K_P')
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Ensure property exists in cell data
        if property_name not in self.mesh.cell_data:
            self.populate_properties([property_name])

        # Compute ray path lengths
        lengths = self.compute_ray_lengths_from_arrival(
            arrival, store_as=None, replace_existing=False
        )

        # Get property values (per cell)
        prop = np.asarray(self.mesh.cell_data[property_name], dtype=float)

        # Compute kernel: K = -L / (prop^2 + epsilon)
        denom = prop * prop + float(epsilon)
        valid = np.isfinite(denom) & (denom > 0.0)

        kernel = np.zeros_like(lengths, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel[valid] = -lengths[valid] / denom[valid]

        # Store as cell data if requested
        if attach_name is None:
            attach_name = f"K_{property_name}"

        if attach_name in self.mesh.cell_data and not replace_existing:
            raise ValueError(
                f"Cell data '{attach_name}' already exists. "
                "Set replace_existing=True to overwrite."
            )

        self.mesh.cell_data[attach_name] = kernel.astype(np.float32)
        print(f"Stored sensitivity kernel as cell data: '{attach_name}'")

        return kernel

    def compute_sensitivity_kernels_for_rays(
        self,
        arrivals: List[Any],
        property_name: str,
        attach_name: Optional[str] = None,
        accumulate: Optional[str] = "sum",
        epsilon: float = 1e-6,
        replace_existing: bool = True
    ) -> np.ndarray:
        """
        Compute sensitivity kernels for multiple rays and optionally accumulate.

        Parameters
        ----------
        arrivals : List[ray objects]
            List of ray objects from TauP with path information
        property_name : str
            Name of seismic property ('vp', 'vs', 'rho', etc.)
        attach_name : str, optional
            Base name for storing kernels. If accumulate='sum', stores single
            array. If accumulate=None, stores individual arrays with suffix.
            If None, uses f"Ksum_{property_name}" or f"K_{property_name}_{{i}}"
        accumulate : str or None
            If "sum", return and store sum of all kernels.
            If None, return (n_rays, n_cells) array and store individual kernels
        epsilon : float
            Small regularizer for denominator
        replace_existing : bool
            Whether to replace existing cell data

        Returns
        -------
        np.ndarray
            If accumulate='sum': 1D array (n_cells,) with summed kernels
            If accumulate=None: 2D array (n_rays, n_cells)

        Examples
        --------
        >>> rays = model.taupy_model.get_ray_paths_geo(..., phase_list=["P", "S"])
        >>> # Sum all kernels
        >>> Ksum = mesh.compute_sensitivity_kernels_for_rays(
        ...     rays, 'vp', attach_name='Ksum_P_S')
        >>> # Keep individual kernels
        >>> K_all = mesh.compute_sensitivity_kernels_for_rays(
        ...     rays, 'vp', accumulate=None, attach_name='K_ray')
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Ensure property exists once
        if property_name not in self.mesh.cell_data:
            self.populate_properties([property_name])

        # Compute kernels for each ray (don't store individual kernels)
        kernels = []
        for i, arrival in enumerate(arrivals):
            # Compute ray path lengths
            lengths = self.compute_ray_lengths_from_arrival(
                arrival, store_as=None, replace_existing=False
            )

            # Get property values (per cell)
            prop = np.asarray(self.mesh.cell_data[property_name], dtype=float)

            # Compute kernel: K = -L / (prop^2 + epsilon)
            denom = prop * prop + float(epsilon)
            valid = np.isfinite(denom) & (denom > 0.0)

            kernel = np.zeros_like(lengths, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                kernel[valid] = -lengths[valid] / denom[valid]

            kernels.append(kernel)

        K_array = np.array(kernels)  # Shape: (n_rays, n_cells)

        if accumulate == "sum":
            Ksum = K_array.sum(axis=0)
            name = attach_name or f"Ksum_{property_name}"
            if name in self.mesh.cell_data and not replace_existing:
                raise ValueError(
                    f"Cell data '{name}' already exists. "
                    "Set replace_existing=True to overwrite."
                )
            self.mesh.cell_data[name] = Ksum.astype(np.float32)
            print(f"Stored summed sensitivity kernel as cell data: '{name}'")
            return Ksum
        else:
            # Store individual kernels if attach_name provided
            if attach_name is not None:
                base_name = attach_name or f"K_{property_name}"
                for i, kernel in enumerate(kernels):
                    name = f"{base_name}_{i}"
                    if name in self.mesh.cell_data and not replace_existing:
                        continue  # Skip if exists and not replacing
                    self.mesh.cell_data[name] = kernel.astype(np.float32)
                print(f"Stored {len(kernels)} individual kernels with base name: '{base_name}_*'")
            return K_array

    # ===== Visualization =====

    def plot_cross_section(self,
                           plane_normal: Tuple[float, float, float] = (0, 1, 0),
                           plane_origin: Tuple[float, float, float] = (0, 0, 0),
                           property_name: str = "vp",
                           show_rays: Optional[List[Any]] = None,
                           **kwargs) -> Any:
        """
        Plot cross-section using plane-based clipping.

        Parameters
        ----------
        plane_normal : tuple
            Normal vector of clipping plane
        plane_origin : tuple
            Point on clipping plane
        property_name : str
            Property to color by
        show_rays : list, optional
            List of ObsPy rays to overlay
        **kwargs
            Additional plotting arguments

        Returns
        -------
        pyvista.Plotter
            Plotter object for further customization
        """
        if self.mesh is None:
            raise RuntimeError("No mesh generated. Call generate_*_mesh() first.")

        # Ensure property exists
        if property_name not in self.mesh.cell_data:
            self.populate_properties([property_name])

        # Clip mesh
        clipped = self.mesh.clip(normal=plane_normal, origin=plane_origin)

        # Create plotter
        plotter = pv.Plotter()

        # Add clipped mesh
        plotter.add_mesh(
            clipped,
            scalars=property_name,
            show_edges=kwargs.get('show_edges', True),
            cmap=kwargs.get('cmap', 'viridis'),
            **{k: v for k, v in kwargs.items()
               if k not in ['show_edges', 'cmap', 'show_rays']}
        )

        # Overlay rays if provided
        if show_rays:
            for ray in show_rays:
                ray_polyline = self._create_ray_polyline(ray)
                if ray_polyline is not None:
                    plotter.add_mesh(ray_polyline, color='red', line_width=3)

        return plotter

    def plot_spherical_shell(self,
                             radius_km: float,
                             property_name: str = "vp",
                             **kwargs) -> Any:
        """
        Plot spherical shell at given radius.

        Parameters
        ----------
        radius_km : float
            Radius of shell to extract
        property_name : str
            Property to color by
        **kwargs
            Additional plotting arguments

        Returns
        -------
        pyvista.Plotter
            Plotter object
        """
        if self.mesh is None:
            raise RuntimeError("No mesh generated. Call generate_*_mesh() first.")

        # Ensure property exists
        if property_name not in self.mesh.cell_data:
            self.populate_properties([property_name])

        # Add radius field for contouring
        points = self.mesh.points
        radii = np.linalg.norm(points, axis=1)
        mesh_with_radius = self.mesh.copy()
        mesh_with_radius.point_data["radius"] = radii

        # Extract isosurface
        shell = mesh_with_radius.contour(isosurfaces=[radius_km], scalars="radius")

        # Sample property onto shell
        shell_sampled = shell.sample(self.mesh)

        # Plot
        plotter = pv.Plotter()
        plotter.add_mesh(
            shell_sampled,
            scalars=property_name,
            cmap=kwargs.get('cmap', 'viridis'),
            **{k: v for k, v in kwargs.items() if k != 'cmap'}
        )

        return plotter

    def list_properties(self,
                        include_point_data: bool = False,
                        show_stats: bool = False,
                        top_n: int = 10) -> Dict[str, Any]:
        """
        List stored properties on the mesh and optionally print basic stats.

        Parameters
        ----------
        include_point_data : bool
            If True, include point (vertex) data keys in the listing.
        show_stats : bool
            If True, print min/max/sum and non-zero counts for cell properties.
        top_n : int
            When printing, shows up to top_n entries for arrays.

        Returns
        -------
        dict
            Dictionary with keys 'cell_data' and optionally 'point_data' mapping
            property names to small summary dicts (or None if not computed).
        """
        if self.mesh is None:
            raise RuntimeError("No mesh generated. Call generate_*_mesh() first.")

        result: Dict[str, Any] = {}

        # Cell data
        cell_keys = list(self.mesh.cell_data.keys())
        result['cell_data'] = {k: None for k in cell_keys}

        if show_stats and cell_keys:
            for k in cell_keys:
                try:
                    arr = np.asarray(self.mesh.cell_data[k])
                    summary = {
                        'shape': arr.shape,
                        'dtype': str(arr.dtype),
                        'min': float(np.nanmin(arr)),
                        'max': float(np.nanmax(arr)),
                        'sum': float(np.nansum(arr)),
                        'non_zero': int((arr != 0).sum()),
                    }
                    # small preview
                    preview_n = min(top_n, arr.size)
                    summary['preview'] = arr.flat[:preview_n].tolist()
                except Exception as e:  # pragma: no cover - defensive
                    summary = {'error': str(e)}
                result['cell_data'][k] = summary

        # Point data (optional)
        if include_point_data:
            point_keys = list(self.mesh.point_data.keys())
            result['point_data'] = {k: None for k in point_keys}
            if show_stats and point_keys:
                for k in point_keys:
                    try:
                        arr = np.asarray(self.mesh.point_data[k])
                        summary = {
                            'shape': arr.shape,
                            'dtype': str(arr.dtype),
                            'min': float(np.nanmin(arr)),
                            'max': float(np.nanmax(arr)),
                            'sum': float(np.nansum(arr)),
                            'non_zero': int((arr != 0).sum()),
                        }
                        preview_n = min(top_n, arr.size)
                        summary['preview'] = arr.flat[:preview_n].tolist()
                    except Exception as e:  # pragma: no cover - defensive
                        summary = {'error': str(e)}
                    result['point_data'][k] = summary

        # Print a compact summary for quick inspection
        print("Mesh properties summary:")
        print(f"  cell_data keys: {cell_keys}")
        if include_point_data:
            print(f"  point_data keys: {list(self.mesh.point_data.keys())}")

        if show_stats and cell_keys:
            print("\nCell property summaries (first entries):")
            for k, v in result['cell_data'].items():
                if isinstance(v, dict):
                    print(f" - {k}: min={v.get('min')}, max={v.get('max')}, non_zero={v.get('non_zero')}")
                else:
                    print(f" - {k}: {v}")

        return result

    # ===== Private Helper Methods =====

    def _generate_discontinuity_shells(self,
                                     max_level: int,
                                     buffer_km: float,
                                     refinement_levels: Optional[Dict[float, int]] = None) -> List[Tuple[str, float, int]]:
        """
        Generate shell specifications from user-defined depth ranges.

        Parameters
        ----------
        max_level : int
            Maximum refinement level allowed
        buffer_km : float
            Buffer zone around planet
        refinement_levels : dict, optional
            Dictionary with depth_km -> refinement_level mapping.
            Keys are depths in km from surface, values are refinement levels.
            Example: {0: 6, 50: 5, 670: 4, 2891: 4}

        Returns
        -------
        List[Tuple[str, float, int]]
            List of (name, radius_km, refinement_level) tuples
        """
        radius = self.planet_model.radius
        shells = []

        if refinement_levels is None:
            # Default depth-based refinement (Earth-like but user can override)
            refinement_levels = {
                0: 6,      # surface to 50km: fine (crust)
                50: 5,     # 50-670km: medium (upper mantle)
                670: 4,    # 670-2891km: coarse (lower mantle)
                2891: 4,   # 2891-5150km: coarse (outer core)
                5150: 3    # 5150km+: very coarse (inner core)
            }

        # Sort depths from surface to center
        sorted_depths = sorted(refinement_levels.keys())

        for i, depth_km in enumerate(sorted_depths):
            level = min(refinement_levels[depth_km], max_level)
            radius_km = radius - depth_km

            # Ensure radius is positive (don't go below center)
            if radius_km <= 0:
                radius_km = 0.1  # small positive value near center

            shell_name = f"depth_{depth_km:.0f}km_level_{level}"
            shells.append((shell_name, radius_km, level))

        # Add buffer zone with same level as outermost shell
        buffer_radius = radius + buffer_km
        buffer_level = shells[-1][2] if shells else 3
        shells.append(("buffer", buffer_radius, buffer_level))

        return shells

    def _get_shell_index(self, r: float, shells: List[Tuple[str, float, int]]) -> int:
        """Get shell index for given radius."""
        for i, (name, rad, level) in enumerate(shells):
            if r <= rad:
                return i
        return len(shells) - 1

    def _compute_polyline_cell_lengths_octree(self, points: np.ndarray) -> np.ndarray:
        """Compute per-cell lengths for octree mesh using AABB intersection."""
        if len(points) < 2:
            return np.zeros(self.mesh.n_cells, dtype=float)

        lengths = np.zeros(self.mesh.n_cells, dtype=float)

        # Get cell bounds for vectorized intersection
        cell_bounds = self._get_cell_bounds()

        # Compute intersection for each segment
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            segment_lengths = self._aabb_segment_length_vectorized(cell_bounds, p0, p1)
            lengths += segment_lengths

        return lengths

    def _compute_polyline_cell_lengths_tetrahedral(self, points: np.ndarray) -> np.ndarray:
        """Compute per-cell lengths for tetrahedral mesh using VTK locator."""
        if len(points) < 2:
            return np.zeros(self.mesh.n_cells, dtype=float)

        # Use internal tetrahedral implementation
        return self._compute_ray_cell_path_lengths_internal(points)

    def _get_cell_bounds(self) -> np.ndarray:
        """Get AABB bounds for all cells."""
        if self.mesh is None:
            raise RuntimeError("No mesh generated. Call generate_*_mesh() first.")

        n_cells = self.mesh.n_cells
        bounds = np.empty((n_cells, 6), dtype=float)

        for i in range(n_cells):
            cell = self.mesh.get_cell(i)
            point_ids = np.array(cell.point_ids, dtype=int)
            coords = self.mesh.points[point_ids]

            bounds[i, 0] = coords[:, 0].min()  # xmin
            bounds[i, 1] = coords[:, 0].max()  # xmax
            bounds[i, 2] = coords[:, 1].min()  # ymin
            bounds[i, 3] = coords[:, 1].max()  # ymax
            bounds[i, 4] = coords[:, 2].min()  # zmin
            bounds[i, 5] = coords[:, 2].max()  # zmax

        return bounds

    def _aabb_segment_length_vectorized(
        self, bounds: np.ndarray, p0: np.ndarray, p1: np.ndarray
    ) -> np.ndarray:
        """Vectorized AABB-segment intersection."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = p1 - p0
        seg_len = float(np.linalg.norm(d))

        if seg_len == 0.0:
            return np.zeros(bounds.shape[0], dtype=float)

        # Unpack bounds
        xmin, xmax = bounds[:, 0], bounds[:, 1]
        ymin, ymax = bounds[:, 2], bounds[:, 3]
        zmin, zmax = bounds[:, 4], bounds[:, 5]

        # Slab method intersection
        tmin = np.zeros_like(xmin)
        tmax = np.ones_like(xmin)

        # X axis
        if d[0] != 0.0:
            tx1 = (xmin - p0[0]) / d[0]
            tx2 = (xmax - p0[0]) / d[0]
            tmin = np.maximum(tmin, np.minimum(tx1, tx2))
            tmax = np.minimum(tmax, np.maximum(tx1, tx2))
        else:
            # Parallel to x-axis: use half-open inclusion
            mask = (xmin <= p0[0]) & (p0[0] < xmax)
            tmax = np.where(mask, tmax, 0.0)

        # Y axis
        if d[1] != 0.0:
            ty1 = (ymin - p0[1]) / d[1]
            ty2 = (ymax - p0[1]) / d[1]
            tmin = np.maximum(tmin, np.minimum(ty1, ty2))
            tmax = np.minimum(tmax, np.maximum(ty1, ty2))
        else:
            mask = (ymin <= p0[1]) & (p0[1] < ymax)
            tmax = np.where(mask, tmax, 0.0)

        # Z axis
        if d[2] != 0.0:
            tz1 = (zmin - p0[2]) / d[2]
            tz2 = (zmax - p0[2]) / d[2]
            tmin = np.maximum(tmin, np.minimum(tz1, tz2))
            tmax = np.minimum(tmax, np.maximum(tz1, tz2))
        else:
            mask = (zmin <= p0[2]) & (p0[2] < zmax)
            tmax = np.where(mask, tmax, 0.0)

        # Valid intersections
        valid = (tmax > tmin) & (tmax > 0.0) & (tmin < 1.0)
        t0 = np.maximum(tmin, 0.0)
        t1 = np.minimum(tmax, 1.0)

        lengths = np.where(valid, seg_len * (t1 - t0), 0.0)
        return lengths

    def _normalize_points_array(self, points_xyz: Any) -> np.ndarray:
        """Return an (N, 3) float array from various point representations.

        Supported inputs:
        - ndarray with shape (N,3), (3,N), or flat 1D of length multiple of 3
        - dict with keys {"x","y","z"} as 1D arrays
        - dict with geographic keys {"lat"/"latitude",
              "lon"/"longitude"/"long"} and one of
              {"r"/"radius"/"radius_km"}
            or {"depth"/"depth_km"}; converted to Cartesian using model
            radius when needed.
        """
        # Dict inputs
        if isinstance(points_xyz, dict):
            d: Dict[str, Any] = points_xyz
            # Cartesian x/y/z arrays
            if all(k in d for k in ("x", "y", "z")):
                x = np.asarray(d["x"], dtype=float)
                y = np.asarray(d["y"], dtype=float)
                z = np.asarray(d["z"], dtype=float)
                if not (x.shape == y.shape == z.shape):
                    raise ValueError("x, y, z must have the same shape")
                return np.column_stack([x, y, z]).astype(float, copy=False)

            # Geographic inputs -> Cartesian
            has_lat = ("lat" in d) or ("latitude" in d)
            has_lon = ("lon" in d) or ("longitude" in d) or ("long" in d)
            if has_lat and has_lon:
                lat_val = d.get("lat", d.get("latitude"))
                lon_val = d.get("lon", d.get("longitude", d.get("long")))
                lat = np.asarray(lat_val, dtype=float)
                lon = np.asarray(lon_val, dtype=float)
                r = d.get("r", d.get("radius", d.get("radius_km")))
                depth = d.get("depth", d.get("depth_km"))
                if r is not None:
                    rad = np.asarray(r, dtype=float)
                elif depth is not None:
                    rad = (
                        self.planet_model.radius - np.asarray(depth, dtype=float)
                    )
                else:
                    rad = np.full_like(
                        lat, self.planet_model.radius, dtype=float
                    )

                from .coordinates import CoordinateConverter

                # Convert radius -> depth for converter
                depth_arr = self.planet_model.radius - rad
                pts = np.vstack(
                    [
                        CoordinateConverter.earth_to_cartesian(
                            float(la), float(lo), float(dd),
                            earth_radius=self.planet_model.radius,
                        )
                        for la, lo, dd in zip(lat, lon, depth_arr)
                    ]
                ).astype(float)
                return pts

            raise ValueError(
                "Unsupported dict for points: expected x/y/z or lat/lon "
                "with r or depth"
            )

        # Array-like inputs
        arr = np.asarray(points_xyz)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise ValueError(
                    "1D points array length must be multiple of 3"
                )
            return arr.reshape(-1, 3).astype(float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(float, copy=False)
        if arr.ndim == 2 and arr.shape[0] == 3:
            return arr.T.astype(float, copy=False)
        raise ValueError(
            "Points must have shape (N,3) or (3,N) or flat 1D of 3N, or a "
            "supported dict"
        )

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

    def _compute_ray_cell_path_lengths_internal(
        self,
        ray_points_xyz: Any,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Compute per-cell path length of a ray polyline through a tet mesh.

        Parameters
        ----------
        ray_points_xyz : array-like
            Ray polyline in Cartesian km. Accepted formats:
            - ndarray with shape (N,3), (3,N), or flat 1D of length 3N
            - dict with x/y/z arrays
            - dict with geo lat/lon + depth or radius
            These inputs are normalized internally to (N,3).
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

        # Validate mesh exists
        if self.mesh is None:
            raise RuntimeError("No mesh generated. Call generate_*_mesh() first.")

        # Validate mesh: proceed if there are any tetrahedra; skip others
        try:
            celltypes = np.asarray(self.mesh.celltypes)
        except Exception:
            celltypes = None
        if celltypes is None:
            warnings.warn(
                "Mesh has no celltypes information; cannot compute path "
                "lengths.",
                UserWarning,
            )
            return np.zeros(self.mesh.n_cells, dtype=float)
        tet_mask = celltypes == vtk.VTK_TETRA
        if not np.any(tet_mask):
            warnings.warn(
                "Mesh contains no tetrahedral cells; returning zeros.",
                UserWarning,
            )
            return np.zeros(self.mesh.n_cells, dtype=float)

        # Normalize points to (N,3)
        pts = self._normalize_points_array(ray_points_xyz)
        n = len(pts)
        n_cells = self.mesh.n_cells
        lengths = np.zeros(n_cells, dtype=float)
        if n < 2:
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
                    cell = self.mesh.extract_cells(cid)
                    tet_pts = np.asarray(cell.points, dtype=float)
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

        return lengths

    def _create_ray_polyline(self, ray: Any) -> Optional[Any]:
        """Create PyVista polyline from ObsPy ray."""
        try:
            # Extract points
            points = []
            for point in ray.path:
                depth_km = point['depth']
                lat_deg = point['lat']
                lon_deg = point['lon']

                from .coordinates import CoordinateConverter
                xyz = CoordinateConverter.earth_to_cartesian(
                    lat_deg, lon_deg, depth_km,
                    earth_radius=self.planet_model.radius
                )
                points.append(xyz)

            if len(points) < 2:
                return None

            if pv is None:
                return None
            return pv.lines_from_points(np.array(points))

        except Exception as e:
            warnings.warn(f"Failed to create ray polyline: {e}")
            return None

    # ----- Save/Load Methods -------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the mesh to disk with metadata.

        Parameters
        ----------
        path : str
            Base path for saving (without extension). Will create:
            - {path}.vtu - the mesh data
            - {path}_metadata.json - mesh generation parameters and metadata

        Examples
        --------
        >>> mesh.save("my_mesh")  # Creates .vtu and _metadata.json files
        """
        if self.mesh is None:
            raise ValueError("No mesh to save. Generate a mesh first.")

        import json

        # Save mesh data as VTU (VTK Unstructured Grid format)
        mesh_path = f"{path}.vtu"
        self.mesh.save(mesh_path)

        # Save metadata as JSON sidecar
        metadata = {
            "mesh_type": self.mesh_type,
            "planet_model_name": getattr(self.planet_model, 'name', 'unknown'),
            "planet_model_radius": self.planet_model.radius,
            "generation_parameters": getattr(self, '_generation_params', {}),
            "properties": (
                list(self.mesh.cell_data.keys()) if self.mesh.cell_data else []
            ),
            "n_cells": self.mesh.n_cells,
            "n_points": self.mesh.n_points,
        }

        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved mesh to {mesh_path}")
        print(f"Saved metadata to {metadata_path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        planet_model: Optional[PlanetModel] = None,
    ) -> 'PlanetMesh':
        """
        Load a mesh from disk with metadata.

        Parameters
        ----------
        path : str
            Base path for loading (without extension). Expects:
            - {path}.vtu - the mesh data
            - {path}_metadata.json - mesh generation parameters and metadata
        planet_model : PlanetModel, optional
            Planet model to associate with the loaded mesh. If None,
            attempts to recreate from metadata.

        Returns
        -------
        PlanetMesh
            Loaded mesh instance

        Examples
        --------
        >>> mesh = PlanetMesh.from_file("my_mesh", planet_model)
        """
        import json
        import os

        if pv is None:  # pragma: no cover
            raise ImportError("PyVista is required to load mesh files.")

        # Load mesh data
        mesh_path = f"{path}.vtu"
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        grid = pv.read(mesh_path)

        # Load metadata
        metadata_path = f"{path}_metadata.json"
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            warnings.warn(f"Metadata file not found: {metadata_path}")

        # Create or validate planet model
        if planet_model is None:
            # Try to recreate from metadata
            if 'planet_model_name' in metadata:
                from .model import PlanetModel
                try:
                    planet_model = PlanetModel.from_standard_model(
                        metadata['planet_model_name']
                    )
                except Exception:
                    warnings.warn(
                        f"Failed to load standard model "
                        f"'{metadata['planet_model_name']}'. "
                        "Using default PREM."
                    )
                    planet_model = PlanetModel.from_standard_model('prem')
            else:
                from .model import PlanetModel
                planet_model = PlanetModel.from_standard_model('prem')
                warnings.warn(
                    "No planet model metadata found. Using default PREM."
                )

        # Create mesh instance
        mesh_type = metadata.get('mesh_type', 'octree')
        instance = cls(planet_model, mesh_type=mesh_type)
        instance.mesh = grid

        print(f"Loaded mesh from {mesh_path}")
        if metadata:
            n_cells = metadata.get('n_cells', 'unknown')
            n_points = metadata.get('n_points', 'unknown')
            print(f"Loaded metadata: {n_cells} cells, {n_points} points")

        return instance
