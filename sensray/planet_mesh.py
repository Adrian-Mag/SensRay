"""
PlanetMesh class for SensRay.

Provides mesh generation and visualization for PlanetModel instances.
Supports tetrahedral mesh generation with discontinuity-aware
refinement and unified visualization methods.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any, TYPE_CHECKING, Callable
import quadpy
if TYPE_CHECKING:
    from .model import PlanetModel
import numpy as np
import warnings

try:  # Optional heavy deps
    import pyvista as pv
except Exception as exc:  # pragma: no cover - optional
    pv = None  # type: ignore[assignment]
    _mesh_err = exc  # noqa: F841


class PlanetMesh:
    """
    Mesh representation of a PlanetModel.

    Handles tetrahedral mesh generation based on model
    discontinuities, property mapping from model to mesh cells,
    and unified visualization with plane-based clipping.

    Parameters
    ----------
    planet_model : PlanetModel
        The planet model to create a mesh for
    mesh_type : str
        Type of mesh to generate: "tetrahedral"

    Examples
    --------
    >>> model = PlanetModel.from_standard_model("prem")
    >>> mesh = PlanetMesh(model)
    >>> mesh.generate_tetrahedral_mesh()
    >>> mesh.populate_properties(["vp", "vs"])
    >>> mesh.plot_cross_section(property_name="vp")
    """

    def __init__(self, planet_model, mesh_type: str = "tetrahedral"):
        if pv is None:  # pragma: no cover
            raise ImportError(
                "PyVista is required for mesh operations. "
                "Install with `pip install pyvista`."
            )

        if mesh_type != "tetrahedral":
            raise ValueError("Only tetrahedral mesh type is supported")

        self.planet_model = planet_model
        self.mesh_type = mesh_type
        self.mesh = None  # PyVista UnstructuredGrid

    # ===== Mesh Generation =====

    def generate_tetrahedral_mesh(self,
                                  mesh_size_km: float = 200.0,
                                  radii: Optional[List[float]] = None,
                                  H_layers: Optional[List[float]] = None,
                                  W_trans: Optional[List[float]] = None,
                                  do_optimize: bool = True) -> None:
        """
        Generate a layered tetrahedral mesh using gmsh with smooth
        size transitions across spherical interfaces.

        Parameters
        ----------
        mesh_size_km : float
            Default characteristic size used if H_layers is not provided.
        radii : list[float], optional
            Ascending list of interface radii in km, including the outer
            radius as the last value. Layers are (0..r1], (r1..r2], ...
        H_layers : list[float], optional
            Target mesh size per layer (len == len(radii)). If None, uses
            [mesh_size_km] * len(radii).
        W_trans : list[float], optional
            Half-widths for smooth transitions at each internal interface
            (len == len(radii)-1). If None, picks 0.2 * layer thickness.
        do_optimize : bool
            Whether to run gmsh mesh optimization.
        """
        try:
            import gmsh  # type: ignore
            import meshio  # type: ignore
            import tempfile
            import os
            import numpy as _np
        except ImportError as exc:
            raise ImportError(
                "Layered tetrahedral mesh generation requires `gmsh` and "
                "`meshio`. Install with `pip install gmsh meshio`."
            ) from exc

        # Build default radii. If radii is None, create a single-layer
        # spherical domain (uniform tetrahedral size set by mesh_size_km).
        if radii is None:
            R_out = float(self.planet_model.radius)
            radii = [R_out]
            # For single-layer uniform sphere, default H_layers to mesh_size_km
            if H_layers is None:
                H_layers = [float(mesh_size_km)]
            # No transition widths necessary for a single layer
            if W_trans is None:
                W_trans = []

        # Validate radii
        r = list(map(float, radii))
        if not all(r[i] < r[i + 1] for i in range(len(r) - 1)):
            raise ValueError(
                "radii must be strictly ascending and include outer"
            )

        # Defaults for sizes
        if H_layers is None:
            H_layers = [float(mesh_size_km)] * len(r)
        if len(H_layers) != len(r):
            raise ValueError(
                "H_layers must have one entry per layer (len(radii))"
            )

        # Defaults for transition widths
        if W_trans is None:
            W_trans = []
            for i in range(len(r)-1):
                thick = r[i+1] - (r[i] if i >= 0 else 0.0)
                W_trans.append(max(1e-3, 0.2 * float(thick)))
        if len(W_trans) != len(r) - 1:
            raise ValueError(
                "W_trans must have one value per internal interface"
            )

        # ---- Gmsh generation (adapted from develop/tets_combined.py) ----
        gmsh.initialize()
        gmsh.model.add("sensray_nlayer_sphere")

        # Disable heuristic size controls, use background field only
        for name in (
            "Mesh.MeshSizeFromPoints",
            "Mesh.MeshSizeFromCurvature",
            "Mesh.MeshSizeExtendFromBoundary",
            "Mesh.CharacteristicLengthFromPoints",
            "Mesh.CharacteristicLengthFromCurvature",
            "Mesh.CharacteristicLengthExtendFromBoundary",
        ):
            try:
                gmsh.option.setNumber(name, 0)
            except Exception:
                pass
        for name in (
            "Mesh.MeshSizeMin",
            "Mesh.MeshSizeMax",
            "Mesh.CharacteristicLengthMin",
            "Mesh.CharacteristicLengthMax",
        ):
            try:
                gmsh.option.setNumber(name, 1e-9 if "Min" in name else 1e12)
            except Exception:
                pass

        # Geometry: inner spheres + outer sphere
        inner_tags = []
        for Ri in r[:-1]:
            inner_tags.append(gmsh.model.occ.addSphere(0, 0, 0, Ri))
        tag_outer = gmsh.model.occ.addSphere(0, 0, 0, r[-1])

        gmsh.model.occ.fragment([(3, tag_outer)], [(3, t) for t in inner_tags])
        gmsh.model.occ.synchronize()

        # Physical groups for volumes ordered by radius
        def extent(tag):
            x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(3, tag)
            return max(
                abs(x0), abs(x1), abs(y0), abs(y1), abs(z0), abs(z1)
            )
        vols = gmsh.model.getEntities(3)
        vols_sorted = [
            t for (_, t) in sorted(
                vols, key=lambda e: extent(e[1])
            )
        ]
        if len(vols_sorted) != len(r):
            gmsh.finalize()
            raise RuntimeError("Unexpected number of volumes after fragment")
        for i, vt in enumerate(vols_sorted, start=1):
            gmsh.model.addPhysicalGroup(3, [vt], i)
            gmsh.model.setPhysicalName(3, i, f"layer_{i-1}")

        # Background size field H(r) via MathEval
        r_expr = "sqrt(x*x+y*y+z*z)"
        s_exprs = [
            f"(0.5*(1+tanh(({r_expr}-{r[i]})/{W_trans[i]})))"
            for i in range(len(r)-1)
        ]

        def prod_expr(seq):
            if not seq:
                return "1"
            out = seq[0]
            for e in seq[1:]:
                out = f"({out})*({e})"
            return out

        windows = []
        windows.append(f"(1-({s_exprs[0]}))" if s_exprs else "1")
        for i in range(1, len(r)-1):
            left = prod_expr(s_exprs[:i])
            windows.append(f"({left})*(1-({s_exprs[i]}))")
        if s_exprs:
            windows.append(prod_expr(s_exprs))

        terms = [
            f"({float(H_layers[i])})*({windows[i]})"
            for i in range(len(H_layers))
        ]
        H_expr = terms[0]
        for t in terms[1:]:
            H_expr = f"({H_expr})+({t})"

        fid = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(fid, "F", H_expr)
        gmsh.model.mesh.field.setAsBackgroundMesh(fid)

        # Mesh + optional optimize
        gmsh.model.mesh.generate(3)
        if do_optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                try:
                    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
                except Exception:
                    pass

        # Write to temp .msh, convert to .vtu, read into PyVista
        tmp_msh = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
        tmp_msh.close()
        msh_path = tmp_msh.name
        gmsh.write(msh_path)
        gmsh.finalize()

        tmp_vtu = tempfile.NamedTemporaryFile(suffix=".vtu", delete=False)
        tmp_vtu.close()
        vtu_path = tmp_vtu.name
        try:
            mesh = meshio.read(msh_path)
            meshio.write(vtu_path, mesh)
            import pyvista as _pv  # local import for type checkers
            grid = _pv.read(vtu_path)
        finally:
            try:
                os.remove(msh_path)
            except Exception:
                pass
            try:
                os.remove(vtu_path)
            except Exception:
                pass

        # Pull region ids from gmsh cell data if present
        region_ids = None
        for key in ("gmsh:physical", "physical"):
            if key in grid.cell_data:
                data = grid.cell_data[key]
                if hasattr(data, "size") and data.size == grid.n_cells:
                    region_ids = _np.asarray(data, dtype=_np.int32)
                    break
                if isinstance(data, (list, tuple)):
                    for arr in data:
                        if hasattr(arr, "size") and arr.size == grid.n_cells:
                            region_ids = _np.asarray(arr, dtype=_np.int32)
                            break
                    if region_ids is not None:
                        break
        if region_ids is None:
            # Fallback by radius of cell centers
            rc = _np.linalg.norm(grid.cell_centers().points, axis=1)
            bins = _np.r_[0.0, _np.array(r, float)]
            region_ids = _np.digitize(rc, bins, right=True)
        grid.cell_data["region"] = _np.asarray(region_ids, dtype=_np.int32)

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

        for prop in properties:
            # Compute and store property averages per cell via quadrature.
            # project_function_on_mesh expects (function, property_name) and
            # stores the result into mesh.cell_data[property_name].
            self.project_function_on_mesh(
                lambda pts: self.planet_model.get_property_at_3d_points(
                    prop, pts
                ),
                prop,
            )

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
        path_points = np.asarray(path_points, dtype=float)
        # Compute per-cell intersections using midpoint-densify method
        return self._compute_ray_cell_lengths_midpoint(path_points)

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
            If provided, store as cell data with this name.
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
    This method accepts an ObsPy ray or arrival with a `.path` list of
    dicts having keys 'lat', 'lon', 'depth'.
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
        path_points = np.asarray(path_points, dtype=float)
        # Compute per-cell intersections using midpoint-densify method
        lengths = self._compute_ray_cell_lengths_midpoint(path_points)

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
    mesh.add_ray_to_mesh(ray_obj, "primary")
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
            Name to store kernel as cell data. If None, uses default.
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
    kernel = mesh.compute_sensitivity_kernel(ray_obj, 'vp')
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
    Compute sensitivity kernels for multiple rays and optionally
    accumulate.

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
            If None, return (n_rays, n_cells) array and store individual
            kernels
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
    Ksum = mesh.compute_sensitivity_kernels_for_rays(rays, 'vp')
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
                msg = (
                    f"Stored {len(kernels)} individual kernels with base "
                    f"name: '{base_name}_*'"
                )
                print(msg)
            return K_array

    # ===== Visualization =====

    def plot_cross_section(self,
                           plane_normal=(0, 1, 0),
                           plane_origin=(0, 0, 0),
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
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Ensure property exists
        if property_name not in self.mesh.cell_data:
            self.populate_properties([property_name])

        # Clip mesh
        clipped = self.mesh.clip(normal=plane_normal, origin=plane_origin)

        # Create plotter
        import pyvista as _pv
        plotter = _pv.Plotter()

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
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Ensure property exists
        if property_name not in self.mesh.cell_data:
            self.populate_properties([property_name])

        # Add radius field for contouring
        points = self.mesh.points
        radii = np.linalg.norm(points, axis=1)
        mesh_with_radius = self.mesh.copy()
        mesh_with_radius.point_data["radius"] = radii

        # Extract isosurface
        shell = mesh_with_radius.contour(
            isosurfaces=[radius_km], scalars="radius"
        )

        # Sample property onto shell
        shell_sampled = shell.sample(self.mesh)

        # Plot
        import pyvista as _pv
        plotter = _pv.Plotter()
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
            Dictionary with keys 'cell_data' and optionally 'point_data'
            mapping
            property names to small summary dicts (or None if not computed).
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        result = {}

        # Cell data
        cell_keys = list(self.mesh.cell_data.keys())
        result['cell_data'] = {}
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
                    preview_n = min(top_n, arr.size)
                    summary['preview'] = arr.flat[:preview_n].tolist()
                except Exception as e:  # pragma: no cover
                    summary = {'error': str(e)}
                result['cell_data'][k] = summary
        else:
            # Map keys to None when not computing stats
            result['cell_data'] = {k: None for k in cell_keys}

        # Point data (optional)
        if include_point_data:
            point_keys = list(self.mesh.point_data.keys())
            if show_stats and point_keys:
                result['point_data'] = {}
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
                    except Exception as e:  # pragma: no cover
                        summary = {'error': str(e)}
                    result['point_data'][k] = summary
            else:
                result['point_data'] = {k: None for k in point_keys}

        # Print a compact summary for quick inspection
        print("Mesh properties summary:")
        print(f"  cell_data keys: {cell_keys}")
        if include_point_data:
            print(f"  point_data keys: {list(self.mesh.point_data.keys())}")

        if show_stats and cell_keys:
            print("\nCell property summaries (first entries):")
            for k, v in result['cell_data'].items():
                if isinstance(v, dict):
                    print(
                        f" - {k}: min={v.get('min')}, max={v.get('max')}, "
                        f"non_zero={v.get('non_zero')}"
                    )
                else:
                    print(f" - {k}: {v}")

        return result

    def project_function_on_mesh(
        self, function: Callable[[np.ndarray], np.ndarray],
        property_name: str,
    ) -> None:
        # Extract tetrahedral cells and points for integration
        grid = self.mesh
        if grid is None:
            raise RuntimeError(
                "No mesh available. Call generate_*_mesh() first."
            )
        cells = grid.cells
        points = grid.points

        # Reshape flat cell array: each row [4, i0, i1, i2, i3]
        n_tets = len(cells) // 5
        tetra_cells = cells.reshape((n_tets, 5))
        tetra_indices = tetra_cells[:, 1:]  # skip the leading 4

        # Extract coordinates
        tetra_points = points[tetra_indices]  # shape (N_tets, 4, 3)

        n_tets = tetra_points.shape[0]
        scheme: Any = quadpy.t3.get_good_scheme(5)

        # QuadPy expects shape (4, n_tets, 3)
        tetra_qp = np.transpose(tetra_points, (1, 0, 2))  # (4, n_tets, 3)

        # Compute volumes for each tetra: V = |det([p1-p0, p2-p0, p3-p0])| / 6
        # tetra_points shape: (n_tets, 4, 3)
        p0 = tetra_points[:, 0, :]
        p1 = tetra_points[:, 1, :]
        p2 = tetra_points[:, 2, :]
        p3 = tetra_points[:, 3, :]
        # Vectorized cross and dot to get 6*volume
        cross = np.cross(p1 - p0, p2 - p0)
        six_vol = np.einsum('ij,ij->i', cross, p3 - p0)
        volumes = np.abs(six_vol) / 6.0

        if np.any(volumes <= 0.0):
            warnings.warn(
                "Detected zero or negative tetrahedron volume(s) during "
                "projection; corresponding cell averages will be set to 0.",
                UserWarning,
            )

        integrals = np.zeros(n_tets, dtype=float)
        for i in range(n_tets):
            tet = tetra_qp[:, i, :]  # shape (4, 3)
            integrals[i] = scheme.integrate(function, tet)

        # Convert integrals -> averages by dividing by volume (defensive)
        with np.errstate(divide='ignore', invalid='ignore'):
            averages = np.where(volumes > 0.0, integrals / volumes, 0.0)

        grid.cell_data[property_name] = averages.astype(np.float32)

    # ===== Private Helper Methods =====

    def _compute_polyline_cell_lengths_tetrahedral(
        self, points: np.ndarray
    ) -> np.ndarray:
        """Compute per-cell lengths for tetrahedral mesh.

        This now uses the midpoint-densify binning approach by default
        for robustness. The old clipping-based method remains available
        as `_compute_ray_cell_path_lengths_internal` if needed.
        """
        if len(points) < 2:
            n = self.mesh.n_cells if self.mesh is not None else 0
            return np.zeros(n, dtype=float)
        return self._compute_ray_cell_lengths_midpoint(points)

    # ---- Ray lengths via densify + midpoint binning ----
    @staticmethod
    def _densify_polyline(
        points: np.ndarray,
        max_seg_len: float
    ) -> np.ndarray:
        pts = np.asarray(points, float)
        if len(pts) == 0:
            return pts
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            if L == 0.0:
                continue
            n = max(1, int(np.ceil(L / float(max_seg_len))))
            for i in range(1, n + 1):
                t = i / float(n)
                out.append(a * (1.0 - t) + b * t)
        return np.asarray(out, float)

    def _compute_ray_cell_lengths_midpoint(self,
                                           ray_xyz: np.ndarray,
                                           step_km: float = 8.0,
                                           merge_tol: float = 1e-8
                                           ) -> np.ndarray:
        """
        Accumulate ray length inside each cell by densifying segments and
        binning by midpoint cell id using vtkStaticCellLocator.FindCell.
        """
        if self.mesh is None:
            return np.zeros(0, float)
        if ray_xyz.ndim != 2 or ray_xyz.shape[1] != 3:
            raise ValueError("ray_xyz must have shape (N,3)")

        if self.mesh.n_cells == 0:
            return np.zeros(0, float)

        # Densify and compute segment midpoints
        pts = self._densify_polyline(ray_xyz, max_seg_len=float(step_km))
        if len(pts) < 2:
            return np.zeros(self.mesh.n_cells, float)
        seg_vecs = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg_vecs, axis=1)
        keep = seg_len > float(merge_tol)
        if not np.any(keep):
            return np.zeros(self.mesh.n_cells, float)
        mids = 0.5 * (pts[:-1] + pts[1:])[keep]
        seg_vecs = seg_vecs[keep]
        seg_len = seg_len[keep]

        # Locator
        from pyvista import _vtk as vtk  # type: ignore
        loc = vtk.vtkStaticCellLocator()
        loc.SetDataSet(self.mesh)
        loc.BuildLocator()

        def find_cell(point, v):
            cid = loc.FindCell(point)
            if cid is None or cid < 0:
                eps = 1e-7
                cid = loc.FindCell(point + eps * v)
                if cid is None or cid < 0:
                    cid = loc.FindCell(point - eps * v)
            return int(cid) if cid is not None and cid >= 0 else -1

        out = np.zeros(self.mesh.n_cells, float)
        for m, v, L in zip(mids, seg_vecs, seg_len):
            cid = find_cell(m, v)
            if cid >= 0:
                out[cid] += float(L)
        return out

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
                        self.planet_model.radius -
                        np.asarray(depth, dtype=float)
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
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

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
        mesh_type = metadata.get('mesh_type', 'tetrahedral')
        if mesh_type != 'tetrahedral':
            raise ValueError(
                f"Unsupported mesh type: {mesh_type}. "
                "Only tetrahedral meshes are supported."
            )
        instance = cls(planet_model, mesh_type=mesh_type)
        instance.mesh = grid

        print(f"Loaded mesh from {mesh_path}")
        if metadata:
            n_cells = metadata.get('n_cells', 'unknown')
            n_points = metadata.get('n_points', 'unknown')
            print(f"Loaded metadata: {n_cells} cells, {n_points} points")

        return instance
