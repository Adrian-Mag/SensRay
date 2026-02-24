"""
Planet model class for SensRay.

This module provides a read-only class to represent planet models
loaded directly from TauP .nd files. The model preserves the exact
layered structure and discontinuities as defined in the .nd format.
"""

from __future__ import annotations  # PEP 563: lazy evaluation — allows dict[str, ...]
                                     # on Python 3.8 where built-in generics are not yet
                                     # subscriptable at runtime.

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
import tempfile
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
import warnings

class PlanetModel:
    """
    A read-only planet model loaded directly from TauP .nd files.

    This class parses .nd files and provides access to the layered
    seismic structure exactly as stored in the file format. The model
    maintains the natural discontinuities and layer boundaries that
    are essential for accurate seismic ray tracing.

    Parameters
    ----------
    nd_file_path : str
        Path to the .nd file containing the model data
    name : str, optional
        Custom name for the model. If None, uses filename or model
        name from the .nd file header.

    Examples
    --------
    >>> # Load a custom model
    >>> model = PlanetModel("mars_model.nd")
    >>>
    >>> # Load a standard model by name
    >>> prem = PlanetModel.from_standard_model("prem")
    >>> ak135 = PlanetModel.from_standard_model("ak135")
    >>>
    >>> # Access model properties
    >>> vp_at_cmb = model.get_property_at_depth("vp", 2891)
    >>> layers = model.get_layer_info()
    """

    def __init__(self, nd_file_path: str, name: Optional[str] = None):
        """
        Load planet model from .nd file.

        Parameters
        ----------
        nd_file_path : str
            Path to the .nd file (can be relative or absolute)
        name : str, optional
            Custom name for the model
        """

        if not os.path.exists(nd_file_path):
            raise FileNotFoundError(f"ND file not found: {nd_file_path}")

        self.nd_file_path = nd_file_path
        self._parse_nd_file()

        # Set model name
        if name is not None:
            self.name = name
        elif hasattr(self, '_parsed_name') and self._parsed_name:
            self.name = self._parsed_name
        else:
            # Use filename without extension
            self.name = os.path.splitext(os.path.basename(nd_file_path))[0]

    @classmethod
    def from_standard_model(cls, model_name: str) -> 'PlanetModel':
        """
        Load a standard Earth model by name from the models directory.

        Parameters
        ----------
        model_name : str
            Name of the standard model. Available models:
            'prem', 'ak135favg', 'ak135fcont', 'ak135fsyngine',
            '1066a', '1066b', 'alfs', 'herrin', 'jb', 'pwdk'

        Returns
        -------
        PlanetModel
            Loaded standard model

        Examples
        --------
        >>> prem = PlanetModel.from_standard_model('prem')
        >>> ak135 = PlanetModel.from_standard_model('ak135favg')

        Raises
        ------
        ValueError
            If the model name is not found in the standard models directory
        """
        # Get the directory where this module is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(module_dir, 'models')
        models_dir = os.path.normpath(models_dir)

        # Construct the full path to the .nd file
        nd_file_path = os.path.join(models_dir, f"{model_name}.nd")

        if not os.path.exists(nd_file_path):
            # List available models for the error message
            available_models = []
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith('.nd'):
                        available_models.append(file[:-3])

            raise ValueError(
                f"Standard model '{model_name}' not found. "
                f"Available models: {', '.join(sorted(available_models))}"
            )

        # Create the model with a clean name
        clean_name = model_name.upper()
        return cls(nd_file_path, name=clean_name)

    @classmethod
    def list_standard_models(cls) -> List[str]:
        """
        List all available standard models.

        Returns
        -------
        List[str]
            List of available standard model names
        """
        # Get the directory where this module is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(module_dir, 'models')
        models_dir = os.path.normpath(models_dir)

        available_models = []
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.nd'):
                    available_models.append(file[:-3])  # Remove .nd extension

        return sorted(available_models)

    def _parse_nd_file(self) -> None:
        """
        Parse the .nd file and extract model structure.

        Each layer is stored as a dict of named NumPy arrays:
            self.layers[layer_name] = {
                'depth':  np.ndarray,   # km, ascending
                'vp':     np.ndarray,   # km/s
                'vs':     np.ndarray,   # km/s
                'rho':    np.ndarray,   # g/cm³
                'radius': np.ndarray,   # km  (= planet_radius - depth)
            }

        The first layer is always named 'surface' (data before the first
        discontinuity label).  Subsequent layers take their names from the
        discontinuity labels in the file.  Duplicate layer names raise
        ValueError.
        """
        self.radius = 0.0
        self._parsed_name = None
        current_layer_name = "surface"
        props = ['depth', 'vp', 'vs', 'rho']
        self.layers = {"surface": {prop: [] for prop in props}}

        with open(self.nd_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Handle comments and extract model name
                if line.startswith('#'):
                    if 'Model:' in line or 'model:' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            self._parsed_name = parts[1].strip()
                    continue

                # Check if this is a discontinuity label
                if self._is_discontinuity_label(line):
                    current_layer_name = line.split('#')[0].strip()
                    if not current_layer_name:
                        current_layer_name = 'unnamed_layer'

                    if current_layer_name in self.layers:
                        raise ValueError(f"Duplicate layer name '{current_layer_name}' in {self.nd_file_path}")

                    self.layers[current_layer_name] = {prop: [] for prop in props}
                    continue

                # Parse data line: depth vp vs rho
                try:
                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    depth = float(parts[0])
                    vp = float(parts[1])
                    vs = float(parts[2])
                    rho = float(parts[3])
                    self.layers[current_layer_name]["depth"].append(depth)
                    self.layers[current_layer_name]["vp"].append(vp)
                    self.layers[current_layer_name]["vs"].append(vs)
                    self.layers[current_layer_name]["rho"].append(rho)

                    # Update radius (maximum depth)
                    if depth > self.radius:
                        self.radius = depth

                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"Error parsing line {line_num} in "
                        f"{self.nd_file_path}: '{line}'. "
                        f"Expected format: depth vp vs rho"
                    ) from e

        # Convert depths to radii for internal consistency
        for layer_dict in self.layers.values():
            # sort by depth just in case, and convert to radius
            idx = np.argsort(layer_dict['depth'])
            for prop, vals in layer_dict.items():
                layer_dict[prop] = np.asarray(vals)[idx]
            layer_dict["radius"] = self.radius - layer_dict["depth"]

        # Validate model
        if self.radius <= 0:
            raise ValueError(f"Invalid radius in {self.nd_file_path}")

        if not self.layers:
            raise ValueError(f"No valid data found in {self.nd_file_path}")

    def _is_discontinuity_label(self, line: str) -> bool:
        """Check if a line is a discontinuity label."""
        # Lines that are just text (no numbers) are discontinuity labels
        try:
            # Try to parse as data line
            parts = line.split()
            if len(parts) >= 4:
                # Try to convert first 4 parts to float
                for i in range(4):
                    float(parts[i])
                return False  # Successfully parsed as data
        except (ValueError, IndexError):
            pass

        # If we can't parse as data and it's not empty, it's a label
        return bool(line.strip()) and not line.startswith('#')

    # ========== Read-Only Property Access ========== #
    def layerwise_linear_interp(self, query, prop='vp'):
        """
        Interpolate a property at one or more depths across all layers.

        Layers are stored depth-ascending per layer and concatenated in
        file order (shallowest layer first), so at every discontinuity the
        shallower-side entry always appears before the deeper-side entry in
        the sorted table.  This gives ``np.interp`` a well-defined
        piecewise-constant profile with sharp jumps at boundaries.

        Parameters
        ----------
        query : float or array-like
            Depth(s) in km at which to evaluate the property.
        prop : str
            Property name: 'vp', 'vs', or 'rho'.
        """
        query = np.asarray(query, dtype=float)

        if prop not in ['vp', 'vs', 'rho']:
            raise ValueError(f"Unknown property: {prop}")
        if np.any((query < 0) | (query > self.radius)):
            raise ValueError(
                f"One or more values outside valid range [0, {self.radius}]"
            )

        # Concatenate depth and property arrays across all layers.
        # Layers are ordered shallowest-first (as in the .nd file), and within
        # each layer depths are already sorted ascending by the parser, so the
        # globally concatenated array is non-decreasing with the correct
        # boundary-value ordering at every discontinuity.
        all_depths = np.concatenate([layer["depth"] for layer in self.layers.values()])
        all_values = np.concatenate([layer[prop]   for layer in self.layers.values()])

        return np.interp(query, all_depths, all_values)

    def get_property_at_depth(
        self, property_name: str, depth: Union[float, "np.ndarray"]
    ) -> Union[float, "np.ndarray"]:
        """Get property value at one or more depths (km)."""
        interpolated = self.layerwise_linear_interp(depth, prop=property_name)
        return float(interpolated) if np.isscalar(depth) else interpolated

    def get_property_at_radius(
        self, property_name: str, radius: Union[float, "np.ndarray"]
    ) -> Union[float, "np.ndarray"]:
        """Get property value at one or more radii (km)."""
        depth = self.radius - np.asarray(radius, dtype=float)
        interpolated = self.layerwise_linear_interp(depth, prop=property_name)
        return float(interpolated) if np.isscalar(radius) else interpolated

    def get_property_at_3d_points(
        self, property_name: str, points: np.ndarray
    ) -> np.ndarray:
        """
        Get property values at multiple 3D points.

        Parameters
        ----------
        property_name : str
            Property name ('vp', 'vs', 'rho')
        points : np.ndarray
            Array of points in Cartesian coordinates (x, y, z) in km.
            Accepts either shape (N, 3) or (3, N).

        Returns
        -------
        np.ndarray
            Array of property values at the given points

        Notes
        -----
        Automatically handles both common coordinate conventions:
        - (N, 3): points as rows [point1, point2, ...] (standard)
        - (3, N): coordinates as rows [x_coords, y_coords, z_coords]
                 (quadpy convention)
        """
        if points.ndim != 2:
            raise ValueError("Points array must be 2-dimensional")

        # Handle both (N, 3) and (3, N) shapes
        if points.shape[0] == 3 and points.shape[1] != 3:
            # Shape is (3, N) - transpose to (N, 3)
            points = points.T
        elif points.shape[1] != 3:
            raise ValueError(
                "Points array must have shape (N, 3) or (3, N)"
            )

        # Convert Cartesian to radius (vectorized)
        radii = np.linalg.norm(points, axis=1)

        # Use the vectorized depth lookup and return as ndarray
        return np.asarray(self.get_property_at_radius(property_name, radii))

    def get_property_profile(self, names: Union[str, List[str]], asradius: bool = True) -> Dict[str, np.ndarray]:
        """
        Get the full profile for one or more properties as depth/value arrays.
        If a list of property names is given, returns a dict mapping each property name to its profile array.
        """
        valid_props = {'vp', 'vs', 'rho'}
        if isinstance(names, str):
            names = [names]

        result = {k: [] for k in ["radius" if asradius else "depth"]+names}
        for layer in self.layers.values():
            if asradius:
                result["radius"].append(layer["radius"])
            else:
                result["depth"].append(layer["depth"])
            for n in names:
                if n not in valid_props:
                    raise ValueError(f"Unknown property: {n}")
                result[n].append(layer[n])
        for prop in result.keys():
            result[prop] = np.concatenate(result[prop])
        return result

    # ========== Model Information ========== #

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns
        -------
        info : Dict
            Dictionary containing model information
        """
        properties = ['vp', 'vs', 'rho']
        info = {
            'name': self.name,
            'radius_km': self.radius,
            'properties': properties,
            'n_properties': len(properties),
            'n_layers': len(self.layers),
            'n_discontinuities': len(self.layers) - 1,  # discontinuities between layers so not include surface
            'discontinuities': list(self.layers.keys())[1:],  # exclude surface
            'layers': self.get_layer_info(properties=properties),
        }

        return info

    # ========== TauPy Integration ========== #

    @property
    def taupy_model(self) -> TauPyModel:
        """
        Lazily build or return a cached ObsPy TauPyModel for this
        PlanetModel.

        Behavior:
        - If the model contains metadata with 'original_model', that
          name is passed to TauPyModel(model=...) and returned.
        - Otherwise, the PlanetModel's `nd_file_path` is used with
          ObsPy's `build_taup_model` to create a model file in the
          system temp directory, and a TauPyModel is loaded from that
          file. The built model is cached on the instance as
          `_taupy_model` to avoid rebuilding.

        Raises
        ------
        ValueError
            If the nd_file_path is missing or the file cannot be found.
        """
        # Return cached model if already built
        if hasattr(self, '_taupy_model') and self._taupy_model is not None:
            return self._taupy_model

        metadata = getattr(self, 'metadata', {})
        original_model = metadata.get('original_model')
        if original_model:
            self._taupy_model = TauPyModel(model=original_model)
            return self._taupy_model

        nd_path = getattr(self, 'nd_file_path', None)
        if not nd_path or not os.path.exists(nd_path):
            raise ValueError('PlanetModel must expose a valid nd_file_path')

        model_name = os.path.splitext(os.path.basename(nd_path))[0]
        try:
            build_taup_model(nd_path, output_folder=tempfile.gettempdir())
        except Exception:
            # Let TauPy/TaupCreate raise more informative errors; wrap if
            # desired for friendly messaging.
            raise

        model_path = os.path.join(tempfile.gettempdir(), f"{model_name}.npz")
        self._taupy_model = TauPyModel(model=model_path)
        return self._taupy_model

    def get_layer_info(self, properties: Optional[List[str]] = None, outwards: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about each layer.
        Parameters
        ----------
        properties : List[str], optional
            List of properties to include stats for.
            Default: ['vp', 'vs', 'rho']

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of layer information dictionaries
        """
        layer_info = {}
        if properties is None:
            properties = ['vp', 'vs', 'rho']

        layer_names = list(self.layers.keys())
        if outwards:
            layer_names.reverse()

        for layer_name in layer_names:
            data = self.layers[layer_name]
            layer_info[layer_name] = {
                'name': layer_name,
                'n_points': len(data["depth"]),
                'depth_range': (
                    min(data["depth"]),
                    max(data["depth"])
                ),
                'radius_range': (
                    min(data["radius"]),
                    max(data["radius"])
                ),
                'properties': {},
            }

            for prop in properties:
                values = data[prop]
                layer_info[layer_name]['properties'][prop] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values)
                }

        return layer_info

    def get_discontinuities(self,
                            include_radius: bool = True,
                            outwards: bool = True
                            ) -> dict[str, dict[str, float]]:
        """
        Get discontinuity locations from the surface inward.

        Parameters
        ----------
        as_depths : bool
            If True, return as depths from surface.
            If False, return as radii from center.
        include_radius : bool
            If False, exclude the outer radius from the list.
        outwards : bool
            If True, return discontinuities from center outwards.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Discontinuity locations by layer name, ordered from surface inward.
        """
        props = ["vp", "vs", "rho"]
        discontinuities = {}

        layer_names = list(self.layers.keys())
        for i in range(len(layer_names)):
            if outwards:
                i = len(layer_names) - 1 - i  # reverse order if outwards
            upper_layer = self.layers[layer_names[i-1]] if i > 0 else None
            lower_layer = self.layers[layer_names[i]]
            if lower_layer["depth"][0] == 0 and not include_radius:
                continue  # skip the discontinuity a depth=0 if not including radius
            discontinuities[layer_names[i]] = {
                "upper": {prop: upper_layer[prop][-1] if upper_layer else None for prop in props},
                "lower": {prop: lower_layer[prop][0] for prop in props},
                "depth": lower_layer["depth"][0],
                "radius": lower_layer["radius"][0],
            }

        return discontinuities

    # ========== Visualization ========== #

    def plot_profiles(
        self,
        properties: Optional[List[str]] = None,
        max_depth_km: Optional[float] = None,
        ax: Optional[Axes] = None,
        show_discontinuities: bool = True,
        colors: Optional[Dict[str, str]] = None,
    ) -> Tuple[Any, Axes]:
        """
        Plot 1D profiles of seismic properties.

        Parameters
        ----------
        properties : List[str], optional
            Properties to plot. Default: ['vp', 'vs', 'rho']
        max_depth_km : float, optional
            Maximum depth to plot in km
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        show_discontinuities : bool
            Whether to show discontinuity lines
        colors : Dict[str, str], optional
            Colors for each property

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        """
        if properties is None:
            properties = ['vp', 'vs', 'rho']

        if colors is None:
            colors = {'vp': 'blue', 'vs': 'red', 'rho': 'green'}

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        max_depth = max_depth_km or self.radius

        # Plot each property
        available_properties = ['vp', 'vs', 'rho']
        for prop in properties:
            if prop not in available_properties:
                continue

            # get profile data of {depths, values}
            profile = self.get_property_profile(prop, asradius=False)
            depths = profile['depth']
            values = profile[prop]

            # Filter by depth
            mask = depths <= max_depth
            depths_plot = depths[mask]
            values_plot = values[mask]

            prop_label = "$v_{" + prop[1] + "}$ (km/s)" if prop[0].lower() == "v" else "\u03C1 (g/cm$^3$)"
            ax.plot(values_plot, depths_plot,
                    color=colors.get(prop, 'black'),
                    label=prop_label, linewidth=2)

        # Add discontinuities
        if show_discontinuities:
            disc_depths = np.array(list(d["depth"] for d in self.get_discontinuities().values()), dtype=float)
            # disc_depths = np.array(list(self.get_discontinuities(as_depths=True).keys()), dtype=float)
            for depth in disc_depths:
                if depth <= max_depth:
                    ax.axhline(depth, color='gray', linestyle='--', alpha=0.7)

        ax.set_ylabel('Depth (km)')
        ax.set_xlabel('Property Value')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'{self.name} - 1D Profiles')

        plt.tight_layout()
        return fig, ax

    # ===== Mesh Integration =====

    @property
    def mesh(self):
        """
        Lazy-loaded mesh property.

        Returns
        -------
        PlanetMesh
            Mesh representation of this planet model
        """
        if not hasattr(self, '_mesh') or self._mesh is None:
            from .planet_mesh import PlanetMesh
            self._mesh = PlanetMesh(self)
        return self._mesh

    def create_mesh(
        self,
        mesh_type: str = "tetrahedral",
        from_file: Optional[str] = None,
        populate_properties: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Create a mesh with specific configuration or load from file.

        Parameters
        ----------
        mesh_type : str
            Type of mesh to create: "tetrahedral" or "spherical".
            Ignored if from_file is provided.
        from_file : str, optional
            Path to load pre-saved mesh from (without extension).
            If provided, loads mesh from {from_file}.vtu and
            {from_file}_metadata.json instead of generating new mesh.
        **kwargs
            Additional arguments passed to mesh generation.
            Ignored if from_file is provided.
        Additional kwargs for tetrahedral mesh generation:
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

        Returns
        -------
        PlanetMesh
            Configured mesh instance

        Examples
        --------
        >>> # Generate new mesh
        >>> mesh = model.create_mesh("tetrahedral", mesh_size_km=100)
        >>>
        >>> # Load pre-saved mesh
        >>> mesh = model.create_mesh(from_file="saved_earth_mesh")
        """
        from .planet_mesh import PlanetMesh

        if from_file is not None:
            # Load mesh from file (mesh_type auto-detected from metadata)
            self._mesh = PlanetMesh.from_file(from_file, planet_model=self)
        else:
            # Generate new mesh
            self._mesh = PlanetMesh(self, mesh_type=mesh_type)

            if mesh_type == "tetrahedral":
                self._mesh.generate_tetrahedral_mesh(**kwargs)
            elif mesh_type == "spherical":
                self._mesh.generate_spherical_mesh(**kwargs)
            else:
                raise ValueError(
                    f"Unknown mesh type: {mesh_type}. "
                    "Only 'tetrahedral' and 'spherical' are supported."
                )

        # Optionally populate properties (after generation or load)
        if populate_properties:
            try:
                self._mesh.populate_properties(populate_properties)
            except Exception as e:
                warnings.warn(f"Failed to populate properties: {e}")

        return self._mesh
