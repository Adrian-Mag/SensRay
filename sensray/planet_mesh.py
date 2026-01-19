"""
PlanetMesh class for SensRay.

Provides mesh generation and visualization for PlanetModel instances.
Supports tetrahedral mesh generation with discontinuity-aware
refinement and unified visualization methods.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any, TYPE_CHECKING, Callable
from . import quadrature
if TYPE_CHECKING:
    from .planet_model import PlanetModel
import numpy as np
import warnings
from scipy import integrate
import matplotlib.pyplot as plt

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
        if mesh_type == "tetrahedral":
            if pv is None:  # pragma: no cover
                raise ImportError(
                    "PyVista is required for tetrahedral mesh operations. "
                    "Install with `pip install pyvista`."
                )
        elif mesh_type not in ["tetrahedral", "spherical"]:
            raise ValueError(
                f"Unsupported mesh type: {mesh_type}. "
                "Supported types: 'tetrahedral', 'spherical'"
            )

        self.planet_model = planet_model
        self.mesh_type = mesh_type
        self.mesh = None  # PyVista UnstructuredGrid or SphericalPlanetMesh

    # ===== Mesh Generation =====

    def generate_spherical_mesh(self, radii: List[float]) -> None:
        """
        Generate a simple 1D spherical mesh with radial layers.

        Parameters
        ----------
        radii : list of float
            List of layer boundary radii in km (e.g., [6371, 3480, 1220])
        """
        self.mesh_type = "spherical"
        self.mesh = SphericalPlanetMesh(radii=radii)
        print(
            f"Generated spherical mesh: {self.mesh.n_cells} layers, "
            f"{self.mesh.n_points} boundaries"
        )

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

        if isinstance(self.mesh, SphericalPlanetMesh):
            # For spherical mesh: use proper radial integration projection
            for prop in properties:
                # Project via volume-weighted integration: (1/V) ∫ f(r) r^2 dr
                self.project_function_on_mesh(
                    lambda r: self.planet_model.get_property_at_radius(
                        prop, r
                    ),
                    prop,
                )
        else:
            # For tetrahedral mesh: use quadrature integration
            for prop in properties:
                # Compute and store property averages per cell via quadrature
                # project_function_on_mesh expects (function, property_name)
                # and stores the result into mesh.cell_data[property_name].
                self.project_function_on_mesh(
                    lambda pts: self.planet_model.get_property_at_3d_points(
                        prop, pts
                    ),
                    prop,
                )

        print(f"Populated properties: {properties}")

    # ===== Ray Tracing =====

    def compute_ray_lengths(
        self,
        arrival: Any,
        store_as: Optional[str] = None,
        replace_existing: bool = True
    ) -> np.ndarray:
        """
        Compute per-cell path lengths from ObsPy ray object(s).

        Works for both tetrahedral (3D) and spherical (1D) meshes.
        Handles single rays or multiple rays automatically.

        Parameters
        ----------
        arrival : ray object or List[ray objects]
            Single ray or list of rays from TauP with 'path' attribute.
            - For spherical meshes: only 'depth' field required
            - For tetrahedral meshes: 'lat', 'lon', 'depth' needed
        store_as : str, optional
            If provided, store as cell data with this name.
            Only valid for single rays (raises ValueError for multiple rays).
            If None, lengths are computed but not stored.
        replace_existing : bool
            If True, replace existing cell data with the same name.
            If False and cell data exists, raise ValueError.
            Only used when store_as is not None.

        Returns
        -------
        np.ndarray
            - Single ray: 1D array of shape (n_cells,) with per-cell lengths
            - Multiple rays: 2D array of shape (n_rays, n_cells)

        Raises
        ------
        ValueError
            If store_as is provided for multiple rays (cell_data storage
            only supported for single rays)

        Examples
        --------
        # Single ray
        rays = model.get_ray_paths(...)
        lengths = mesh.compute_ray_lengths(rays[0])

        # Multiple rays
        all_lengths = mesh.compute_ray_lengths(rays)  # (n_rays, n_cells)

        # Store single ray in cell_data
        mesh.compute_ray_lengths(rays[0], store_as='P_wave_lengths')

        Notes
        -----
        - For tetrahedral meshes: requires lat/lon/depth from
          get_ray_paths_geo()
        - For spherical meshes: works with depth-only from get_ray_paths()
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Detect if we have multiple rays
        is_multiple = isinstance(arrival, (list, tuple))

        if is_multiple and store_as is not None:
            raise ValueError(
                "store_as parameter not supported for multiple rays. "
                "To store multiple ray lengths, compute them individually:\n"
                "  for i, ray in enumerate(rays):\n"
                "      mesh.compute_ray_lengths(ray, store_as=f'ray_{i}')"
            )

        if is_multiple:
            # Handle multiple rays - return 2D array
            lengths = []
            for ray in arrival:
                # Dispatch based on mesh type
                if isinstance(self.mesh, SphericalPlanetMesh):
                    ray_lengths = self._compute_ray_lengths_spherical(ray)
                else:
                    ray_lengths = self._compute_ray_lengths_tetrahedral(ray)
                lengths.append(ray_lengths)
            return np.array(lengths)
        else:
            # Handle single ray
            # Dispatch based on mesh type
            if isinstance(self.mesh, SphericalPlanetMesh):
                lengths = self._compute_ray_lengths_spherical(arrival)
            else:
                lengths = self._compute_ray_lengths_tetrahedral(arrival)

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
        lengths = self.compute_ray_lengths(
            arrival, store_as=cell_data_name, replace_existing=True
        )

        return lengths

    # ===== Sensitivity Kernels =====

    def compute_sensitivity_kernel(
        self,
        arrival: Any,
        property_name: str,
        attach_name: Optional[str] = None,
        accumulate: Optional[str] = None,
        epsilon: float = 1e-6,
        replace_existing: bool = True
    ) -> np.ndarray:
        """
        Compute sensitivity kernel(s) K = -L / (prop^2 + epsilon) per cell.

        Works for both tetrahedral and spherical meshes. Handles single rays
        or multiple rays automatically.

        Parameters
        ----------
        arrival : ray object or List[ray objects]
            Single ray or list of rays from TauP with path information.
            Works for both mesh types.
        property_name : str
            Name of seismic property ('vp', 'vs', 'rho', etc.)
            Must exist in mesh.cell_data or will be populated from model
        attach_name : str, optional
            Name to store kernel as cell data.
            - Single ray: stores as-is (default: f"K_{property_name}")
            - Multiple rays with accumulate='sum': stores sum
              (default: f"Ksum_{property_name}")
            - Multiple rays with accumulate=None: stores with _i suffix
              (default: f"K_{property_name}_{{i}}")
        accumulate : str or None, optional
            Only used for multiple rays. If "sum", returns and stores sum of
            all kernels. If None, returns (n_rays, n_cells) array.
            Ignored for single ray.
        epsilon : float
            Small regularizer added to denominator to avoid division by zero
        replace_existing : bool
            Whether to replace existing cell data with same name

        Returns
        -------
        np.ndarray
            - Single ray: 1D array of shape (n_cells,) with kernel values
            - Multiple rays with accumulate='sum': 1D array (n_cells,)
            - Multiple rays with accumulate=None: 2D array (n_rays, n_cells)

        Examples
        --------
        # Single ray
        kernel = mesh.compute_sensitivity_kernel(ray, 'vp')

        # Multiple rays - return sum
        K_sum = mesh.compute_sensitivity_kernel(rays, 'vp', accumulate='sum')

        # Multiple rays - return individual kernels
        K_all = mesh.compute_sensitivity_kernel(rays, 'vp', accumulate=None)

        # Store with custom name
        mesh.compute_sensitivity_kernel(ray, 'vp', attach_name='my_kernel')

        Notes
        -----
        The sensitivity kernel relates travel time perturbations to velocity
        perturbations: δt = ∫ K δv dx, where K = -L/v² and L is path length.
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        # Ensure property exists in cell data
        if property_name not in self.mesh.cell_data:
            self.populate_properties([property_name])

        # Get property values (per cell) - same for all rays
        prop = np.asarray(self.mesh.cell_data[property_name], dtype=float)

        # Pre-compute denominator for efficiency
        denom = prop * prop + float(epsilon)
        valid = np.isfinite(denom) & (denom > 0.0)

        # Detect if we have multiple rays
        is_multiple = isinstance(arrival, (list, tuple))

        if is_multiple:
            # Handle multiple rays
            kernels = []
            for ray in arrival:
                # Compute ray path lengths
                lengths = self.compute_ray_lengths(ray)

                # Compute kernel: K = -L / (prop^2 + epsilon)
                kernel = np.zeros_like(lengths, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    kernel[valid] = -lengths[valid] / denom[valid]

                kernels.append(kernel)

            K_array = np.array(kernels)  # Shape: (n_rays, n_cells)

            if accumulate == "sum":
                # Sum all kernels
                K_sum = K_array.sum(axis=0)
                name = attach_name or f"Ksum_{property_name}"
                if name in self.mesh.cell_data and not replace_existing:
                    raise ValueError(
                        f"Cell data '{name}' already exists. "
                        "Set replace_existing=True to overwrite."
                    )
                self.mesh.cell_data[name] = K_sum.astype(np.float32)
                print(
                    f"Stored summed sensitivity kernel as cell data: '{name}'"
                )
                return K_sum
            else:
                # Return individual kernels, optionally store
                if attach_name is not None:
                    base_name = attach_name or f"K_{property_name}"
                    for i, kernel in enumerate(kernels):
                        name = f"{base_name}_{i}"
                        if (name in self.mesh.cell_data and
                                not replace_existing):
                            continue  # Skip if exists and not replacing
                        self.mesh.cell_data[name] = kernel.astype(np.float32)
                    msg = (
                        f"Stored {len(kernels)} individual kernels with "
                        f"base name: '{base_name}_*'"
                    )
                    print(msg)
                return K_array
        else:
            # Handle single ray
            # Compute ray path lengths
            lengths = self.compute_ray_lengths(arrival)

            # Compute kernel: K = -L / (prop^2 + epsilon)
            kernel = np.zeros_like(lengths, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                kernel[valid] = -lengths[valid] / denom[valid]

            # Store as cell data
            name = attach_name or f"K_{property_name}"
            if name in self.mesh.cell_data and not replace_existing:
                raise ValueError(
                    f"Cell data '{name}' already exists. "
                    "Set replace_existing=True to overwrite."
                )
            self.mesh.cell_data[name] = kernel.astype(np.float32)
            print(f"Stored sensitivity kernel as cell data: '{name}'")

            return kernel

    # ===== Visualization =====

    def plot_cross_section(self,
                           plane_normal=(0, 1, 0),
                           plane_origin=(0, 0, 0),
                           property_name: str = "vp",
                           show_rays: Optional[List[Any]] = None,
                           **kwargs) -> Any:
        """
        Plot cross-section using plane-based clipping.

        Note: Only works with tetrahedral meshes. For spherical (1D) meshes,
        use plot_shell_property() instead.

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

        if isinstance(self.mesh, SphericalPlanetMesh):
            raise TypeError(
                "plot_cross_section() only works with tetrahedral meshes. "
                "For spherical (1D) meshes, use plot_shell_property() instead."
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

        Note: Only works with tetrahedral meshes. For spherical (1D) meshes,
        use plot_shell_property() instead.

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

        if isinstance(self.mesh, SphericalPlanetMesh):
            raise TypeError(
                "plot_spherical_shell() only works with tetrahedral meshes. "
                "For spherical (1D) meshes, use plot_shell_property() instead."
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

    def plot_shell_property(self,
                            property_name: str, *,
                            show_shading=True,
                            show_centers=False,
                            annotate_radii=False,
                            figsize=(8, 4),
                            title=None):
        """
        Plot a property defined per-shell on a 1D spherical radial
        discretization.

        Parameters
        ----------
        property_name : str
            Name of the property to plot (must exist in mesh.cell_data)
        show_shading : bool, default=True
            Shade shells to make discretization clearer
        show_centers : bool, default=False
            Show markers at shell centers
        annotate_radii : bool, default=False
            Annotate radii values on the plot (may clutter if many shells)
        figsize : tuple, default=(8, 4)
            Figure size (width, height) in inches
        title : str, optional
            Plot title. If None, uses f"{property_name} on 1D spherical mesh"

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            The created figure and axes objects

        Examples
        --------
        >>> model.mesh.plot_shell_property('vp')
        >>> model.mesh.plot_shell_property('lengths', show_centers=True)
        """
        if self.mesh_type != 'spherical':
            raise TypeError(
                f"plot_shell_property only works with spherical meshes, "
                f"got mesh_type='{self.mesh_type}'"
            )

        if property_name not in self.mesh.cell_data:
            raise ValueError(
                f"Property '{property_name}' not found in mesh.cell_data. "
                f"Available properties: {list(self.mesh.cell_data.keys())}"
            )

        radii = np.asarray(self.mesh.radii, dtype=float)
        f_j = np.asarray(self.mesh.cell_data[property_name], dtype=float)

        # Prepare step plot data: piecewise-constant
        # Use stairs/step plotting: create repeated arrays for step plotting
        # Approach: r_steps of length 2N, v_steps length 2N with v_j repeated
        N = self.mesh.n_cells
        r_steps = np.empty(2 * N)
        v_steps = np.empty(2 * N)
        r_steps[0::2] = radii[:-1]
        r_steps[1::2] = radii[1:]
        v_steps[0::2] = f_j
        v_steps[1::2] = f_j

        centers = 0.5 * (radii[:-1] + radii[1:])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # Step plot: use drawstyle='steps-post' or stairs if available Using
        # stairs (clear intent) if matplotlib >= 3.4; otherwise fallback to
        # step
        try:
            ax.stairs(
                f_j, radii, baseline=None,
                label=property_name
            )
        except Exception:
            ax.plot(
                r_steps, v_steps, drawstyle="steps-post",
                label=property_name
            )

        # Vertical lines for shell edges
        for r in radii:
            ax.axvline(r, color="0.85", linewidth=0.8, zorder=0)

        # Optional shading of shells for visibility
        if show_shading:
            cmap = plt.get_cmap("Blues")
            # Normalize to property range for shading intensity (avoid division
            # by zero)
            vmin, vmax = np.min(f_j), np.max(f_j)
            rng = vmax - vmin if vmax != vmin else 1.0
            for i in range(N):
                r0, r1 = radii[i], radii[i+1]
                # shade using normalized value -> subtle color
                norm_val = (f_j[i] - vmin) / rng
                ax.axvspan(
                    r0, r1, facecolor=cmap(0.2 + 0.6 * norm_val),
                    alpha=0.08, zorder=0
                )

        # Centers markers
        if show_centers:
            ax.plot(centers, f_j, "o", markersize=4, label="shell centers")

        # Annotate radii if requested
        if annotate_radii:
            ylim = ax.get_ylim()
            y_offset = ylim[0] - 0.05 * (ylim[1] - ylim[0])
            for r in radii:
                ax.text(
                    r, y_offset, f"{r:.3g}",
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=8,
                )

        # Set labels and title
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel(property_name)
        if title is None:
            title = f"{property_name} on 1D spherical mesh"
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        plt.show()
        return fig, ax

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
        self,
        function: Callable[[np.ndarray], np.ndarray],
        property_name: str,
    ) -> None:
        """
        Project a scalar function onto the mesh cells via quadrature.

        Parameters
        ----------
        function : Callable[[np.ndarray], np.ndarray]
            Function that takes an array of shape (N,3) and returns
            an array of shape (N,) with scalar values.
        property_name : str
            Name to store the projected property in mesh.cell_data.
        """
        if isinstance(self.mesh, SphericalPlanetMesh):
            self._project_function_on_spherical_mesh(
                function, property_name
            )
        else:
            self._project_function_on_tetrahedra_mesh(
                function, property_name
            )

    def _project_function_on_spherical_mesh(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        property_name: str,
        epsabs: float = 1e-9,
        epsrel: float = 1e-9,
        limit: int = 50,
    ) -> None:
        """
        Adaptive quad per shell. f_scalar should accept float and return float.
        Returns (integrals, denom) where integrals are ∫ f(r) r^2 dr.
        """
        radii = np.asarray(self.mesh.radii, dtype=float)
        rL = radii[:-1]
        rR = radii[1:]
        N = rL.size
        integrals = np.empty(N, dtype=float)

        for i in range(N):
            a = rL[i]
            b = rR[i]
            if b <= a:
                integrals[i] = 0.0
                continue

            def integrand(r):
                return float(function(r)) * (r ** 2)

            val, err = integrate.quad(
                integrand, a, b,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit
            )
            integrals[i] = val

        denom = (rR ** 3) - (rL ** 3)

        # Place the computed projections into cell data
        with np.errstate(divide='ignore', invalid='ignore'):
            averages = np.where(denom > 0.0, (3.0 / denom) * integrals, 0.0)

        self.mesh.cell_data[property_name] = averages.astype(np.float32)

    def _project_function_on_tetrahedra_mesh(
        self, function: Callable[[np.ndarray], np.ndarray],
        property_name: str,
    ) -> None:
        # Extract tetrahedral cells and points for integration
        grid = self.mesh
        if grid is None:
            raise RuntimeError(
                "No mesh available. Call generate_*_mesh() first."
            )
        cells = grid.cells  #
        points = grid.points

        # Reshape flat cell array: each row [4, i0, i1, i2, i3]
        n_tets = len(cells) // 5
        tetra_cells = cells.reshape((n_tets, 5))
        tetra_indices = tetra_cells[:, 1:]  # skip the leading 4

        # Extract coordinates
        tetra_points = points[tetra_indices]  # shape (N_tets, 4, 3)

        n_tets = tetra_points.shape[0]
        # Use order 3 quadrature (exact for cubic polynomials)
        scheme: Any = quadrature.t3.get_good_scheme(3)

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

    def _compute_ray_lengths_tetrahedral(self, arrival: Any) -> np.ndarray:
        """
        Compute ray path lengths for tetrahedral mesh.

        Requires lat/lon/depth from get_ray_paths_geo.
        """
        path_points = []

        # Handle different types of input objects
        if hasattr(arrival, 'path'):
            ray_path = arrival.path
        elif hasattr(arrival, 'ray'):
            ray_path = arrival.ray.path
        else:
            ray_path = arrival

        for point in ray_path:
            if isinstance(point, dict):
                depth_km = point['depth']
                lat_deg = point['lat']
                lon_deg = point['lon']
            else:
                # Handle numpy structured arrays from TauP
                try:
                    depth_km = float(point['depth'])
                    lat_deg = float(point['lat'])
                    lon_deg = float(point['lon'])
                except (KeyError, TypeError):
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
        return self._compute_ray_cell_lengths_midpoint(path_points)

    def _compute_ray_lengths_spherical(self, arrival: Any) -> np.ndarray:
        """
        Compute ray path lengths for spherical (1D) mesh using exact
        geometric intersection with shell boundaries.

        Algorithm:
        For each segment between consecutive ray points:
        1. Form parametric line R(s) = P0 + s*(P1 - P0), s in [0,1]
        2. Find intersections with shell boundaries: ||R(s)|| = R_shell
        3. Solve resulting quadratic equation for each boundary
        4. Distribute segment length proportionally to layers

        Parameters
        ----------
        arrival : ray object
            Ray with 'path' attribute containing depth information

        Returns
        -------
        np.ndarray
            Per-layer path lengths in km
        """
        # Extract path
        if hasattr(arrival, 'path'):
            ray_path = arrival.path
        elif hasattr(arrival, 'ray'):
            ray_path = arrival.ray.path
        else:
            ray_path = arrival

        # Extract depths and convert to radii
        depths = np.array([float(p['depth']) for p in ray_path])
        radii = self.planet_model.radius - depths

        # Get Cartesian positions (we need 3D positions for the math)
        # For 1D rays, we can place them in an arbitrary plane
        # Use lat=0, lon varies to create a 2D path in the equatorial plane
        if 'dist' in ray_path.dtype.names:
            # Use angular distance to compute positions
            dists_rad = np.array([float(p['dist']) for p in ray_path])
            # Convert to Cartesian in equatorial plane (z=0)
            # x = r * cos(theta), y = r * sin(theta)
            x = radii * np.cos(dists_rad)
            y = radii * np.sin(dists_rad)
            z = np.zeros_like(x)
            points = np.column_stack([x, y, z])
        else:
            # Fallback: assume points lie along x-axis
            # This is less accurate but works for vertical rays
            points = np.column_stack([radii, np.zeros_like(radii),
                                     np.zeros_like(radii)])

        # Get shell boundary radii (sorted descending)
        shell_radii = np.array(self.mesh.radii, dtype=float)

        # Initialize lengths array
        lengths = np.zeros(self.mesh.n_cells, dtype=float)

        # Process each segment
        for i in range(len(points) - 1):
            P0 = points[i]
            P1 = points[i + 1]
            r0 = radii[i]
            r1 = radii[i + 1]

            # Segment vector and length
            D = P1 - P0
            seg_length = np.linalg.norm(D)

            if seg_length < 1e-10:
                continue

            # Find which layers this segment crosses
            r_min = min(r0, r1)
            r_max = max(r0, r1)

            # Find all shell boundaries in the range [r_min, r_max]
            # These are the boundaries we need to find intersections with
            boundaries_in_range = []
            for j, R_shell in enumerate(shell_radii):
                if r_min < R_shell < r_max:
                    boundaries_in_range.append((j, R_shell))

            # Find intersection parameters s for each boundary
            # ||P0 + s*D||^2 = R_shell^2
            # ||P0||^2 + 2*s*(P0·D) + s^2*||D||^2 = R_shell^2
            # s^2*||D||^2 + 2*s*(P0·D) + (||P0||^2 - R_shell^2) = 0

            D_sq = np.dot(D, D)
            P0_dot_D = np.dot(P0, D)
            P0_sq = np.dot(P0, P0)

            s_values = [0.0]  # Start of segment

            for _, R_shell in boundaries_in_range:
                # Quadratic coefficients
                a = D_sq
                b = 2.0 * P0_dot_D
                c = P0_sq - R_shell * R_shell

                discriminant = b * b - 4 * a * c

                if discriminant >= 0 and abs(a) > 1e-14:
                    sqrt_disc = np.sqrt(discriminant)
                    s1 = (-b - sqrt_disc) / (2 * a)
                    s2 = (-b + sqrt_disc) / (2 * a)

                    # Keep solutions in [0, 1]
                    for s in [s1, s2]:
                        if 0 < s < 1:
                            s_values.append(s)

            s_values.append(1.0)  # End of segment
            s_values = sorted(set(s_values))  # Remove duplicates and sort

            # Distribute segment length to layers
            for j in range(len(s_values) - 1):
                s_start = s_values[j]
                s_end = s_values[j + 1]
                s_mid = 0.5 * (s_start + s_end)

                # Find which layer this sub-segment belongs to
                P_mid = P0 + s_mid * D
                r_mid = np.linalg.norm(P_mid)

                # Find layer index for this radius
                layer_idx = None
                for k in range(self.mesh.n_cells):
                    r_inner = shell_radii[k]
                    r_outer = shell_radii[k + 1]
                    if r_inner <= r_mid <= r_outer:
                        layer_idx = k
                        break

                if layer_idx is not None:
                    # Add proportional length to this layer
                    sub_length = (s_end - s_start) * seg_length
                    lengths[layer_idx] += sub_length

        return lengths

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

    def project_point_onto_mesh(
        self,
        point: np.ndarray,
        max_distance_km: float = 500.0,
        inward_offset_km: float = 0.1,
    ) -> Optional[np.ndarray]:
        """
        Project a point onto the mesh surface if it's outside but within
        planet radius.

        This method is useful for handling surface points that are
        geometrically outside the tetrahedral mesh due to the faceted
        approximation of the spherical surface.

        Parameters
        ----------
        point : np.ndarray
            3D point coordinates (x, y, z) in km
        max_distance_km : float, optional
            Maximum distance from mesh surface to attempt projection
            (default: 500 km)
        inward_offset_km : float, optional
            Small offset to move projected point inside mesh
            (default: 0.1 km)

        Returns
        -------
        np.ndarray or None
            Projected point coordinates, or None if projection failed

        Examples
        --------
        >>> surface_point = np.array([1737.4, 0, 0])  # At Moon surface
        >>> projected = mesh.project_point_onto_mesh(surface_point)
        >>> if projected is not None:
        ...     print(f"Projected point is now inside mesh")

        Notes
        -----
        - Only projects points within planet radius but outside mesh
        - Uses VTK's FindClosestPoint for exact surface projection
        - Projects point slightly inward to ensure it's inside a cell
        """
        if self.mesh is None:
            raise RuntimeError(
                "No mesh generated. Call generate_*_mesh() first."
            )

        from pyvista import _vtk as vtk  # type: ignore

        r = np.linalg.norm(point)

        # Only project if point is within planet radius
        if r > self.planet_model.radius:
            return None

        # Use VTK to find exact projection onto mesh surface
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(self.mesh)
        cell_locator.BuildLocator()

        closest_point = np.zeros(3)
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        cell_locator.FindClosestPoint(
            point, closest_point, cell_id, sub_id, dist2
        )

        dist = np.sqrt(float(dist2))

        # Only proceed if within reasonable distance
        if dist >= max_distance_km:
            warnings.warn(
                f"Point at radius {r:.2f} km is {dist:.2f} km from mesh "
                f"surface (exceeds max_distance_km={max_distance_km:.2f}). "
                "Projection failed.",
                UserWarning,
            )
            return None

        # Project point slightly inside from the surface
        if dist > 1e-10:
            # Direction: from point toward closest surface point
            direction = closest_point - point
            direction = direction / np.linalg.norm(direction)
            # Move to surface + small inward offset
            projected = closest_point + direction * inward_offset_km
        else:
            # Point is already very close, nudge inward radially
            projected = closest_point - point / (r + 1e-10) * inward_offset_km

        return projected

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

                # If still not found, try projecting onto mesh surface
                if cid is None or cid < 0:
                    projected = self.project_point_onto_mesh(point)
                    if projected is not None:
                        cid = loc.FindCell(projected)

                    # If still not found, warn about the issue
                    if cid is None or cid < 0:
                        r = np.linalg.norm(point)
                        if r <= self.planet_model.radius:
                            warnings.warn(
                                f"Ray point at radius {r:.2f} km "
                                f"(within planet radius "
                                f"{self.planet_model.radius:.2f} km) could "
                                "not be located in mesh. Segment ignored.",
                                UserWarning,
                            )
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
            - For tetrahedral: {path}.vtu + {path}_metadata.json
            - For spherical: {path}.npz + {path}_metadata.json

        Examples
        --------
        >>> mesh.save("my_mesh")  # Creates files based on mesh type
        """
        if self.mesh is None:
            raise ValueError("No mesh to save. Generate a mesh first.")

        import json
        import os

        base_path = os.path.splitext(path)[0]  # Remove extension if present

        # Save mesh data based on type
        if isinstance(self.mesh, SphericalPlanetMesh):
            mesh_path = self.mesh.save(base_path)
        else:
            # PyVista tetrahedral mesh
            mesh_path = f"{base_path}.vtu"
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

        metadata_path = f"{base_path}_metadata.json"
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
            - Tetrahedral: {path}.vtu + {path}_metadata.json
            - Spherical: {path}.npz + {path}_metadata.json
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

        base_path = os.path.splitext(path)[0]  # Remove extension if present

        # Load metadata first to determine mesh type
        metadata_path = f"{base_path}_metadata.json"
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            warnings.warn(f"Metadata file not found: {metadata_path}")

        # Determine mesh type
        mesh_type = metadata.get('mesh_type', 'tetrahedral')

        # Create or validate planet model
        if planet_model is None:
            # Try to recreate from metadata
            if 'planet_model_name' in metadata:
                from .planet_model import PlanetModel
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
                from .planet_model import PlanetModel
                planet_model = PlanetModel.from_standard_model('prem')
                warnings.warn(
                    "No planet model metadata found. Using default PREM."
                )

        # Load mesh data based on type
        if mesh_type == 'spherical':
            mesh_path = f"{base_path}.npz"
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

            grid = SphericalPlanetMesh.load(base_path)

        elif mesh_type == 'tetrahedral':
            if pv is None:  # pragma: no cover
                raise ImportError(
                    "PyVista is required to load tetrahedral meshes."
                )

            mesh_path = f"{base_path}.vtu"
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

            grid = pv.read(mesh_path)

        else:
            raise ValueError(
                f"Unsupported mesh type in metadata: {mesh_type}. "
                "Supported types: 'tetrahedral', 'spherical'"
            )

        # Create mesh instance
        instance = cls(planet_model, mesh_type=mesh_type)
        instance.mesh = grid

        print(f"Loaded mesh from {mesh_path}")
        if metadata:
            n_cells = metadata.get('n_cells', 'unknown')
            n_points = metadata.get('n_points', 'unknown')
            print(f"Loaded metadata: {n_cells} cells, {n_points} points")

        return instance


class SphericalPlanetMesh():
    """
    Simple 1D spherical mesh for radially-symmetric planet models.

    Stores properties at discrete radial layers. Compatible with
    PlanetMesh interface for unified save/load operations.

    Parameters
    ----------
    radii : list of float
        List of layer boundary radii in km, from surface to center.

    Examples
    --------
    >>> mesh = SphericalPlanetMesh(radii=[6371, 3480, 1220])
    >>> mesh.cell_data['vp'] = np.array([8.0, 13.0])  # 2 layers
    """

    def __init__(self, radii: List[float]):
        """Initialize a spherical planet mesh with given layer radii.

        Parameters
        ----------
        radii : list of float
            List of layer boundary radii in km, from surface to center.
        """
        self.radii = sorted(radii)  # Ensure descending order
        self._cell_data = {}
        self._point_data = {}

    @property
    def n_cells(self) -> int:
        """Number of cells (layers) in the mesh."""
        return max(0, len(self.radii) - 1)

    @property
    def n_points(self) -> int:
        """Number of points (layer boundaries) in the mesh."""
        return len(self.radii)

    def get_layer_radius(self, index: int) -> float:
        """Get the radius of the layer boundary at the given index.

        Parameters
        ----------
        index : int
            Index of the layer boundary (0 = outermost).

        Returns
        -------
        float
            Radius of the layer boundary in km.
        """
        if index < 0 or index >= len(self.radii):
            raise IndexError("Layer index out of range.")
        return self.radii[index]

    @property
    def cell_data(self) -> Dict[str, np.ndarray]:
        """Dictionary to store cell data properties."""
        return self._cell_data

    @cell_data.setter
    def cell_data(self, value: Dict[str, np.ndarray]) -> None:
        """Set cell data properties."""
        if not isinstance(value, dict):
            raise TypeError("cell_data must be a dictionary")
        self._cell_data = value

    @property
    def point_data(self) -> Dict[str, np.ndarray]:
        """Dictionary to store point data properties."""
        return self._point_data

    @point_data.setter
    def point_data(self, value: Dict[str, np.ndarray]) -> None:
        """Set point data properties."""
        if not isinstance(value, dict):
            raise TypeError("point_data must be a dictionary")
        self._point_data = value

    def save(self, path: str) -> str:
        """
        Save spherical mesh to numpy format.

        Parameters
        ----------
        path : str
            Path to save file (will use .npz extension)

        Returns
        -------
        str
            Path to saved file
        """
        import os
        base_path = os.path.splitext(path)[0]  # Remove extension if present
        save_path = f"{base_path}.npz"

        # Prepare data dictionary
        data = {
            'radii': np.array(self.radii),
        }

        # Add cell data with prefix
        for key, val in self._cell_data.items():
            data[f'cell_data_{key}'] = np.asarray(val)

        # Add point data with prefix
        for key, val in self._point_data.items():
            data[f'point_data_{key}'] = np.asarray(val)

        np.savez_compressed(save_path, **data)
        return save_path

    @classmethod
    def load(cls, path: str) -> 'SphericalPlanetMesh':
        """
        Load spherical mesh from numpy format.

        Parameters
        ----------
        path : str
            Path to load file (.npz)

        Returns
        -------
        SphericalPlanetMesh
            Loaded mesh instance
        """
        import os
        base_path = os.path.splitext(path)[0]  # Remove extension if present
        load_path = f"{base_path}.npz"

        data = np.load(load_path)
        radii = data['radii'].tolist()

        mesh = cls(radii=radii)

        # Load cell data
        for key in data.keys():
            if key.startswith('cell_data_'):
                prop_name = key[len('cell_data_'):]
                mesh._cell_data[prop_name] = data[key]
            elif key.startswith('point_data_'):
                prop_name = key[len('point_data_'):]
                mesh._point_data[prop_name] = data[key]

        return mesh

