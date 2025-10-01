"""
Planet model class for SensRay.

This module provides a read-only class to represent planet models
loaded directly from TauP .nd files. The model preserves the exact
layered structure and discontinuities as defined in the .nd format.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os


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
        models_dir = os.path.join(module_dir, '..', 'models')
        models_dir = os.path.normpath(models_dir)

        # Handle common aliases and case variations
        model_aliases = {
            'ak135': 'ak135favg',
            'AK135': 'ak135favg',
            'PREM': 'prem',
            'Prem': 'prem'
        }

        # Use alias if available, otherwise use the name as-is
        actual_model_name = model_aliases.get(model_name, model_name.lower())

        # Construct the full path to the .nd file
        nd_file_path = os.path.join(models_dir, f"{actual_model_name}.nd")

        if not os.path.exists(nd_file_path):
            # List available models for the error message
            available_models = []
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith('.nd'):
                        available_models.append(file[:-3])  # Remove .nd extension

            raise ValueError(
                f"Standard model '{model_name}' not found. "
                f"Available models: {', '.join(sorted(available_models))}"
            )

        # Create the model with a clean name
        clean_name = actual_model_name.upper()
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
        models_dir = os.path.join(module_dir, '..', 'models')
        models_dir = os.path.normpath(models_dir)

        available_models = []
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.nd'):
                    available_models.append(file[:-3])  # Remove .nd extension

        return sorted(available_models)

    def _parse_nd_file(self) -> None:
        """Parse the .nd file and extract model structure."""
        self.layers = []
        self.discontinuities = []
        self.radius = 0.0
        self._parsed_name = None

        layer_points = []
        layer_count = 0

        with open(self.nd_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Handle comments and extract model name
                if line.startswith('#'):
                    if 'Model:' in line or 'model:' in line:
                        # Extract model name from comment
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            self._parsed_name = parts[1].strip()
                    continue

                # Check if this is a discontinuity label
                if self._is_discontinuity_label(line):
                    # End current layer if we have points
                    if layer_points:
                        # Create layer with previous points
                        if layer_count == 0:
                            layer_name = 'surface'
                        else:
                            layer_name = f'layer_{layer_count}'

                        self.layers.append({
                            'name': layer_name,
                            'points': layer_points.copy()
                        })
                        layer_points = []
                        layer_count += 1

                    # Store discontinuity name for next layer
                    disc_name = line.split('#')[0].strip()
                    self.discontinuities.append(disc_name)
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

                    point = {
                        'depth': depth,
                        'vp': vp,
                        'vs': vs,
                        'rho': rho
                    }

                    layer_points.append(point)

                    # Update radius (maximum depth)
                    if depth > self.radius:
                        self.radius = depth

                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"Error parsing line {line_num} in "
                        f"{self.nd_file_path}: '{line}'. "
                        f"Expected format: depth vp vs rho"
                    ) from e

        # Add final layer if we have remaining points
        if layer_points:
            if layer_count == 0:
                layer_name = 'surface'
            elif (self.discontinuities and
                  layer_count <= len(self.discontinuities)):
                # Use the last discontinuity name for the final layer
                if self.discontinuities:
                    layer_name = self.discontinuities[-1]
                else:
                    layer_name = f'layer_{layer_count}'
            else:
                layer_name = f'layer_{layer_count}'

            self.layers.append({
                'name': layer_name,
                'points': layer_points
            })

        # If no layers were created but we have discontinuities,
        # something went wrong
        if not self.layers and layer_points:
            self.layers.append({
                'name': 'default',
                'points': layer_points
            })

        # Convert depths to radii for internal consistency
        for layer in self.layers:
            for point in layer['points']:
                point['radius'] = self.radius - point['depth']

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

    def get_property_at_depth(self, property_name: str, depth: float) -> float:
        """
        Get property value at a specific depth.

        For depths at discontinuities, returns the value from the layer
        that the depth falls into based on the depth ordering in the file.

        Parameters
        ----------
        property_name : str
            Property name ('vp', 'vs', 'rho')
        depth : float
            Depth in km from surface

        Returns
        -------
        float
            Interpolated property value
        """
        if property_name not in ['vp', 'vs', 'rho']:
            raise ValueError(f"Unknown property: {property_name}")

        if depth < 0 or depth > self.radius:
            raise ValueError(
                f"Depth {depth} outside valid range [0, {self.radius}]"
            )

        # Use the same ordering as get_property_profile to ensure consistency
        all_depths = []
        all_values = []

        for layer in self.layers:
            for point in layer['points']:
                all_depths.append(point['depth'])
                all_values.append(point[property_name])

        # Sort by depth to maintain discontinuity structure
        sort_idx = np.argsort(all_depths)
        sorted_depths = np.array(all_depths)[sort_idx]
        sorted_values = np.array(all_values)[sort_idx]

        return float(np.interp(depth, sorted_depths, sorted_values))

    def get_property_at_radius(
        self, property_name: str, radius: float
    ) -> float:
        """Get property value at a specific radius."""
        depth = self.radius - radius
        return self.get_property_at_depth(property_name, depth)

    def get_property_profile(self, name: str) -> Dict[str, np.ndarray]:
        """
        Get the full profile for a property as depth/value arrays.

        Parameters
        ----------
        name : str
            Property name ('vp', 'vs', 'rho')

        Returns
        -------
        profile : Dict[str, np.ndarray]
            Dictionary with 'radius' and 'value' arrays, preserving
            the original depth ordering from the .nd file to maintain
            discontinuity structure
        """
        if name not in ['vp', 'vs', 'rho']:
            raise ValueError(f"Unknown property: {name}")

        all_radii = []
        all_values = []
        all_depths = []

        for layer in self.layers:
            for point in layer['points']:
                all_radii.append(point['radius'])
                all_values.append(point[name])
                all_depths.append(point['depth'])

        # Sort by depth (NOT radius) to preserve discontinuity ordering
        sort_idx = np.argsort(all_depths)

        return {
            'radius': np.array(all_radii)[sort_idx],
            'value': np.array(all_values)[sort_idx]
        }

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
            'n_discontinuities': len(self.discontinuities),
            'discontinuities': self.discontinuities.copy()
        }

        # Add layer information
        info['layers'] = []
        for layer in self.layers:
            layer_info = {
                'name': layer['name'],
                'n_points': len(layer['points']),
                'depth_range': (
                    min(p['depth'] for p in layer['points']),
                    max(p['depth'] for p in layer['points'])
                )
            }
            info['layers'].append(layer_info)

        return info

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about each layer.

        Returns
        -------
        List[Dict]
            List of layer information dictionaries
        """
        layer_info = []

        for layer in self.layers:
            depths = [p['depth'] for p in layer['points']]
            radii = [p['radius'] for p in layer['points']]

            info = {
                'name': layer['name'],
                'n_points': len(layer['points']),
                'depth_range': (min(depths), max(depths)),
                'radius_range': (min(radii), max(radii)),
                'properties': {}
            }

            # Add property ranges for this layer
            for prop in ['vp', 'vs', 'rho']:
                values = [p[prop] for p in layer['points']]
                info['properties'][prop] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values)
                }

            layer_info.append(info)

        return layer_info

    def get_discontinuities(self, as_depths: bool = False) -> List[float]:
        """
        Get discontinuity locations.

        Parameters
        ----------
        as_depths : bool
            If True, return as depths from surface.
            If False, return as radii from center.

        Returns
        -------
        List[float]
            Discontinuity locations
        """
        if not as_depths:
            # Return as radii - need to extract from layer boundaries
            boundaries = set()
            for layer in self.layers:
                radii = [p['radius'] for p in layer['points']]
                boundaries.add(min(radii))
                boundaries.add(max(radii))

            # Remove center and surface
            boundaries.discard(0.0)
            boundaries.discard(self.radius)

            return sorted(boundaries)
        else:
            # Return as depths
            radii = self.get_discontinuities(as_depths=False)
            return sorted([self.radius - r for r in radii])

    # ========== Visualization ========== #

    def plot_profiles(
        self,
        properties: Optional[List[str]] = None,
        max_depth_km: Optional[float] = None,
        ax: Optional[Axes] = None,
        show_discontinuities: bool = True,
        colors: Optional[Dict[str, str]] = None
    ) -> Tuple[Figure, Axes]:
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

            profile = self.get_property_profile(prop)
            depths = self.radius - profile['radius']
            values = profile['value']

            # Filter by depth
            mask = depths <= max_depth
            depths_plot = depths[mask]
            values_plot = values[mask]

            ax.plot(values_plot, depths_plot,
                    color=colors.get(prop, 'black'),
                    label=prop, linewidth=2)

        # Add discontinuities
        if show_discontinuities:
            disc_depths = self.get_discontinuities(as_depths=True)
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
