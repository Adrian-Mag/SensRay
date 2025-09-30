"""
Planet model class for SensRay.

This module provides a class to represent a planet model with
1D profiles for seismic properties, internal discontinuities, and basic
visualization capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class PlanetModel:
    """
    A class representing a Planet model with 1D profiles.

    This class stores the basic structure of a planetary model including:
    - Planet radius
    - Internal discontinuities (as radii from center)
    - 1D profiles for seismic properties (vp, vs, rho, etc.)

    The model is independent of any mesh and focuses on the 1D radial
    structure of the planet.

    Parameters
    ----------
    radius : float
        Planet radius in km
    name : str, optional
        Name of the model (default: "Custom Planet Model")
    discontinuities : List[float], optional
        List of discontinuity radii in km from planet center
    properties : Dict[str, Dict[str, np.ndarray]], optional
        Dictionary of property profiles. Each property should have
        'radius' and 'value' arrays of the same length.
    metadata : Dict, optional
        Additional metadata about the model
    """

    def __init__(
        self,
        radius: float,
        name: str = "Custom Planet Model",
        discontinuities: Optional[List[float]] = None,
        properties: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        metadata: Optional[Dict] = None
    ):
        self.radius = float(radius)
        self.name = str(name)
        self.discontinuities = discontinuities or []
        self.metadata = metadata or {}

        # Store 1D profiles - each property has 'radius' and 'value' arrays
        self.properties: Dict[str, Dict[str, np.ndarray]] = {}

        if properties:
            for prop_name, profile in properties.items():
                self.add_property(
                    prop_name, profile['radius'], profile['value']
                )

    # ========== Basic Property Management ========== #

    def add_property(
        self,
        name: str,
        radius: Union[np.ndarray, List[float]],
        values: Union[np.ndarray, List[float]]
    ) -> None:
        """
        Add a 1D property profile to the model.

        Parameters
        ----------
        name : str
            Name of the property (e.g., 'vp', 'vs', 'rho')
        radius : array-like
            Radius values in km from planet center
        values : array-like
            Property values at corresponding radii
        """
        radius_arr = np.asarray(radius, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        if radius_arr.shape != values_arr.shape:
            raise ValueError(
                f"Radius and values arrays must have same shape. "
                f"Got {radius_arr.shape} and {values_arr.shape}"
            )

        if radius_arr.ndim != 1:
            raise ValueError("Radius and values must be 1D arrays")

        # Validate radius values
        if np.any(radius_arr < 0) or np.any(radius_arr > self.radius):
            raise ValueError(
                f"Radius values must be between 0 and {self.radius} km"
            )

        # Sort by radius for proper interpolation
        sort_idx = np.argsort(radius_arr)

        self.properties[name] = {
            'radius': radius_arr[sort_idx],
            'value': values_arr[sort_idx]
        }

    def remove_property(self, name: str) -> None:
        """Remove a property from the model."""
        if name not in self.properties:
            raise KeyError(f"Property '{name}' not found in model")
        del self.properties[name]

    def get_property_names(self) -> List[str]:
        """Get list of available property names."""
        return list(self.properties.keys())

    def has_property(self, name: str) -> bool:
        """Check if model has a specific property."""
        return name in self.properties

    def get_property_profile(self, name: str) -> Dict[str, np.ndarray]:
        """
        Get the full profile for a property.

        Parameters
        ----------
        name : str
            Property name

        Returns
        -------
        profile : Dict[str, np.ndarray]
            Dictionary with 'radius' and 'value' arrays
        """
        if name not in self.properties:
            raise KeyError(f"Property '{name}' not found in model")
        return self.properties[name].copy()

    # ========== Model Information ========== #

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns
        -------
        info : Dict
            Dictionary containing model information
        """
        info = {
            'name': self.name,
            'radius_km': self.radius,
            'properties': list(self.properties.keys()),
            'n_properties': len(self.properties),
            'discontinuities': sorted(self.discontinuities),
            'n_discontinuities': len(self.discontinuities),
            'metadata': self.metadata.copy()
        }

        # Add discontinuity depths
        if self.discontinuities:
            info['discontinuity_depths_km'] = [
                self.radius - r for r in sorted(self.discontinuities)
            ]

        # Add property statistics
        prop_stats = {}
        for prop_name, profile in self.properties.items():
            values = profile['value']
            prop_stats[prop_name] = {
                'n_points': len(values),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)),
                'mean_value': float(np.mean(values)),
                'radius_range_km': [
                    float(np.min(profile['radius'])),
                    float(np.max(profile['radius']))
                ],
                'depth_range_km': [
                    float(self.radius - np.max(profile['radius'])),
                    float(self.radius - np.min(profile['radius']))
                ]
            }
        info['property_statistics'] = prop_stats

        return info

    def get_discontinuities(self, as_depths: bool = False) -> List[float]:
        """
        Get list of discontinuities.

        Parameters
        ----------
        as_depths : bool
            If True, return as depths from surface.
            If False, return as radii from center.

        Returns
        -------
        discontinuities : List[float]
            Sorted list of discontinuities
        """
        if as_depths:
            return [self.radius - r for r in sorted(self.discontinuities)]
        return sorted(self.discontinuities)

    def add_discontinuity(self, radius: float) -> None:
        """Add a discontinuity at given radius."""
        if radius < 0 or radius > self.radius:
            raise ValueError(
                f"Discontinuity radius must be between 0 and {self.radius}"
            )
        if radius not in self.discontinuities:
            self.discontinuities.append(radius)

    def remove_discontinuity(self, radius: float, tolerance: float = 1e-6):
        """Remove a discontinuity at given radius (within tolerance)."""
        to_remove = []
        for i, disc in enumerate(self.discontinuities):
            if abs(disc - radius) <= tolerance:
                to_remove.append(i)

        for i in reversed(to_remove):
            self.discontinuities.pop(i)

    # ========== String Representations ========== #

    def __repr__(self) -> str:
        """String representation of the model."""
        n_props = len(self.properties)
        n_disc = len(self.discontinuities)
        return (f"PlanetModel(name='{self.name}', "
                f"radius={self.radius:.1f} km, "
                f"properties={n_props}, discontinuities={n_disc})")

    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [f"Planet Model: {self.name}"]
        lines.append(f"Radius: {self.radius:.1f} km")
        lines.append(f"Discontinuities: {len(self.discontinuities)}")
        if self.discontinuities:
            for i, d in enumerate(sorted(self.discontinuities)):
                depth = self.radius - d
                lines.append(
                    f"  {i+1:2d}: r={d:.1f} km (depth={depth:.1f} km)"
                )
        lines.append(f"Properties: {len(self.properties)}")
        for prop in sorted(self.properties.keys()):
            n_points = len(self.properties[prop]['radius'])
            lines.append(f"  {prop}: {n_points} points")
        return "\n".join(lines)

    # ========== Property Interpolation ========== #

    def interpolate_property(
        self,
        name: str,
        radius: Union[float, np.ndarray, List[float]] = None,
        depth: Union[float, np.ndarray, List[float]] = None
    ) -> Union[float, np.ndarray]:
        """
        Interpolate a property at specific radius/depth points.

        Parameters
        ----------
        name : str
            Property name
        radius : float or array-like, optional
            Radius values in km (provide either radius or depth)
        depth : float or array-like, optional
            Depth values in km from surface (provide either radius or depth)

        Returns
        -------
        values : float or np.ndarray
            Interpolated property values
        """
        if name not in self.properties:
            raise KeyError(f"Property '{name}' not found in model")

        if radius is not None and depth is not None:
            raise ValueError("Provide either radius or depth, not both")

        if radius is None and depth is None:
            raise ValueError("Must provide either radius or depth")

        profile = self.properties[name]

        if radius is not None:
            query_radius = np.asarray(radius)
        else:
            query_depth = np.asarray(depth)
            query_radius = self.radius - query_depth

        # Interpolate
        interp_values = np.interp(
            query_radius, profile['radius'], profile['value']
        )

        # Return scalar if input was scalar
        if np.isscalar(radius) or np.isscalar(depth):
            return float(interp_values)
        return interp_values

    def get_property_at_depth(
        self, name: str, depth: Union[float, np.ndarray, List[float]]
    ) -> Union[float, np.ndarray]:
        """Convenience method to get property at specific depth(s)."""
        return self.interpolate_property(name, depth=depth)

    def get_property_at_radius(
        self, name: str, radius: Union[float, np.ndarray, List[float]]
    ) -> Union[float, np.ndarray]:
        """Convenience method to get property at specific radius/radii."""
        return self.interpolate_property(name, radius=radius)

    # ========== Visualization ========== #

    def plot_profiles(
        self,
        properties: Optional[List[str]] = None,
        use_depth: bool = True,
        max_depth_km: Optional[float] = None,
        max_radius_km: Optional[float] = None,
        fig_size: Tuple[float, float] = (10, 8),
        ax: Optional[Axes] = None,
        show_discontinuities: bool = True,
        colors: Optional[Dict[str, str]] = None,
        line_styles: Optional[Dict[str, str]] = None,
        line_widths: Optional[Dict[str, float]] = None
    ) -> Figure:
        """
        Plot 1D property profiles.

        Parameters
        ----------
        properties : List[str], optional
            Properties to plot. If None, plots all available properties.
        use_depth : bool
            If True, plot vs depth from surface. If False, plot vs radius.
        max_depth_km : float, optional
            Maximum depth to show (only used if use_depth=True)
        max_radius_km : float, optional
            Maximum radius to show (only used if use_depth=False)
        fig_size : tuple
            Figure size (width, height)
        ax : matplotlib.axes.Axes, optional
            Existing axis to plot on
        show_discontinuities : bool
            Whether to show discontinuities as vertical lines
        colors : Dict[str, str], optional
            Custom colors for each property
        line_styles : Dict[str, str], optional
            Custom line styles for each property
        line_widths : Dict[str, float], optional
            Custom line widths for each property

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        if not self.properties:
            raise ValueError("No properties available to plot")

        if properties is None:
            properties = list(self.properties.keys())

        # Validate properties
        for prop in properties:
            if prop not in self.properties:
                raise KeyError(f"Property '{prop}' not found in model")

        # Create figure and axis if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure

        # Default colors and styles
        default_colors = {
            'vp': 'tab:blue',
            'vs': 'tab:orange',
            'rho': 'tab:green',
            'density': 'tab:green',
            'qp': 'tab:red',
            'qs': 'tab:purple'
        }
        default_colors.update(colors or {})

        default_styles = {}
        default_styles.update(line_styles or {})

        default_widths = {}
        default_widths.update(line_widths or {})

        # Plot each property
        for i, prop_name in enumerate(properties):
            profile = self.properties[prop_name]

            if use_depth:
                x_values = self.radius - profile['radius']
                x_label = 'Depth (km)'
            else:
                x_values = profile['radius']
                x_label = 'Radius (km)'

            y_values = profile['value']

            # Apply limits
            if use_depth and max_depth_km is not None:
                mask = x_values <= max_depth_km
                x_values = x_values[mask]
                y_values = y_values[mask]
            elif not use_depth and max_radius_km is not None:
                mask = x_values <= max_radius_km
                x_values = x_values[mask]
                y_values = y_values[mask]

            # Plot styling
            color = default_colors.get(prop_name, f'C{i}')
            style = default_styles.get(prop_name, '-')
            width = default_widths.get(prop_name, 2.0)

            ax.plot(
                y_values, x_values,
                label=self._get_property_label(prop_name),
                color=color,
                linestyle=style,
                linewidth=width
            )

        # Show discontinuities
        if show_discontinuities and self.discontinuities:
            y_min, y_max = ax.get_ylim()
            for disc in self.discontinuities:
                if use_depth:
                    disc_val = self.radius - disc
                    if max_depth_km is None or disc_val <= max_depth_km:
                        ax.axhline(
                            disc_val, color='black', linestyle='--',
                            alpha=0.5, linewidth=1
                        )
                else:
                    if max_radius_km is None or disc <= max_radius_km:
                        ax.axhline(
                            disc, color='black', linestyle='--',
                            alpha=0.5, linewidth=1
                        )

        # Formatting
        if use_depth:
            ax.invert_yaxis()
        ax.set_ylabel(x_label)
        ax.set_xlabel('Property Value')
        ax.set_title(f'1D Property Profiles - {self.name}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def _get_property_label(self, prop_name: str) -> str:
        """Get formatted label for property."""
        labels = {
            'vp': 'Vp (km/s)',
            'vs': 'Vs (km/s)',
            'rho': 'Density (g/cm³)',
            'density': 'Density (g/cm³)',
            'qp': 'Qp',
            'qs': 'Qs'
        }
        return labels.get(prop_name, prop_name.upper())

    def plot_property(
        self,
        property_name: str,
        use_depth: bool = True,
        max_depth_km: Optional[float] = None,
        max_radius_km: Optional[float] = None,
        fig_size: Tuple[float, float] = (8, 6),
        ax: Optional[Axes] = None,
        show_discontinuities: bool = True,
        color: str = 'tab:blue',
        line_style: str = '-',
        line_width: float = 2.0
    ) -> Figure:
        """
        Plot a single property profile.

        This is a convenience method that calls plot_profiles for a single
        property with simplified parameters.
        """
        return self.plot_profiles(
            properties=[property_name],
            use_depth=use_depth,
            max_depth_km=max_depth_km,
            max_radius_km=max_radius_km,
            fig_size=fig_size,
            ax=ax,
            show_discontinuities=show_discontinuities,
            colors={property_name: color},
            line_styles={property_name: line_style},
            line_widths={property_name: line_width}
        )

    # ========== Class Methods for Standard Models ========== #

    @classmethod
    def from_standard_model(
        cls,
        model_name: str,
        properties: Optional[List[str]] = None,
        max_depth_km: Optional[float] = None
    ) -> 'PlanetModel':
        """
        Create a PlanetModel from a standard seismic model (PREM, IASP91, etc).

        This method uses ObsPy TauP to extract 1D profiles directly from
        standard Earth models and creates a PlanetModel instance.

        Parameters
        ----------
        model_name : str
            Name of standard model ('prem', 'iasp91', 'ak135')
        properties : List[str], optional
            Properties to extract (default: ['vp', 'vs', 'rho'])
        max_depth_km : float, optional
            Maximum depth to extract

        Returns
        -------
        model : PlanetModel
            New PlanetModel instance
        """
        try:
            from obspy.taup import TauPyModel
            import numpy as np
        except ImportError as e:
            raise ImportError(
                f"Cannot load standard models: {e}. "
                "This requires ObsPy TauP to be available."
            )

        if properties is None:
            properties = ['vp', 'vs', 'rho']

        # Load TauP model directly
        taup_model = TauPyModel(model=model_name)

        # Extract velocity layers from TauP model internals
        try:
            layers = taup_model.model.s_mod.v_mod.layers
        except Exception as exc:
            raise RuntimeError(
                f"Unable to access velocity layers for model "
                f"'{model_name}': {exc}"
            )

        # Property mapping from TauP layer fields
        prop_map = {
            'vp': ('top_p_velocity', 'bot_p_velocity'),
            'vs': ('top_s_velocity', 'bot_s_velocity'),
            'rho': ('top_density', 'bot_density'),
            'density': ('top_density', 'bot_density'),
            'qp': ('top_qp', 'bot_qp'),
            'qs': ('top_qs', 'bot_qs'),
        }

        # Build piecewise-linear profiles
        depths = []
        series = {p: [] for p in properties}

        for rec in layers:
            top_d = float(rec['top_depth'])
            bot_d = float(rec['bot_depth'])

            # Apply depth limit if specified
            if max_depth_km is not None and top_d > max_depth_km:
                break

            # Add top values
            depths.append(top_d)
            for p in properties:
                fields = prop_map.get(p)
                if fields and fields[0] in layers.dtype.names:
                    series[p].append(float(rec[fields[0]]))
                else:
                    series[p].append(float('nan'))

            # Add bottom values
            if max_depth_km is not None and bot_d > max_depth_km:
                bot_d = float(max_depth_km)
            depths.append(bot_d)
            for p in properties:
                fields = prop_map.get(p)
                if fields and fields[1] in layers.dtype.names:
                    series[p].append(float(rec[fields[1]]))
                else:
                    series[p].append(float('nan'))

            if max_depth_km is not None and bot_d >= max_depth_km:
                break

        # Remove duplicates and average values at same depth
        depths = np.array(depths)
        unique_depths = []
        idx_groups = {}

        for i, val in enumerate(depths):
            if (len(unique_depths) == 0 or
                    not np.isclose(unique_depths[-1], val, atol=1e-10)):
                unique_depths.append(float(val))
            idx_groups.setdefault(float(val), []).append(i)

        depths = np.array(unique_depths)

        # Average duplicate values
        for p in properties:
            arr = np.array(series[p], dtype=float)
            vals = []
            for val in unique_depths:
                inds = idx_groups[val]
                vals.append(float(np.nanmean(arr[inds])))
            series[p] = np.array(vals)

        # Standard Earth radius and discontinuities
        earth_radius = 6371.0
        radii = earth_radius - depths

        # Create property profiles
        model_properties = {}
        for prop in properties:
            if prop in series:
                model_properties[prop] = {
                    'radius': radii,
                    'value': series[prop]
                }

        # Standard discontinuities (approximate depths)
        standard_discontinuities = {
            'moho': 35.0,
            'cmb': 2891.0,
            'icb': 5150.0
        }

        discontinuities = []
        for name, depth in standard_discontinuities.items():
            discontinuities.append(earth_radius - depth)

        # Model descriptions
        descriptions = {
            'iasp91': ('IASP91 (International Association of Seismology '
                       'and Physics of the Earth Interior)'),
            'prem': 'PREM (Preliminary Reference Earth Model)',
            'ak135': 'AK135 (Kennett & Engdahl 1995)',
        }

        # Create metadata
        metadata = {
            'source': 'obspy_taup',
            'original_model': model_name,
            'description': descriptions.get(
                model_name.lower(), f'TauP model: {model_name}'
            ),
            'extracted_properties': properties,
            'earth_radius_km': earth_radius
        }

        return cls(
            radius=earth_radius,
            name=model_name.upper(),
            discontinuities=discontinuities,
            properties=model_properties,
            metadata=metadata
        )

    @classmethod
    def create_simple_earth(
        cls,
        radius: float = 6371.0,
        name: str = "Simple Earth"
    ) -> 'PlanetModel':
        """
        Create a simple Earth model with basic structure.

        Creates a model with standard discontinuities but no property profiles.
        Properties can be added later using add_property().

        Parameters
        ----------
        radius : float
            Earth radius in km (default: 6371 km)
        name : str
            Model name

        Returns
        -------
        model : PlanetModel
            New PlanetModel with standard discontinuities
        """
        # Standard Earth discontinuities (approximate)
        discontinuities = [
            radius - 35.0,    # Moho (approximate)
            radius - 660.0,   # 660 km discontinuity
            radius - 2891.0,  # Core-mantle boundary
            radius - 5150.0   # Inner-outer core boundary
        ]

        metadata = {
            'source': 'simple_earth',
            'description': 'Simple Earth model with standard discontinuities'
        }

        return cls(
            radius=radius,
            name=name,
            discontinuities=discontinuities,
            metadata=metadata
        )

    # ========== Utility Methods ========== #

    def copy(self) -> 'PlanetModel':
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for serialization.

        Returns
        -------
        data : Dict
            Dictionary representation of the model
        """
        data = {
            'radius': self.radius,
            'name': self.name,
            'discontinuities': self.discontinuities.copy(),
            'metadata': self.metadata.copy(),
            'properties': {}
        }

        for prop_name, profile in self.properties.items():
            data['properties'][prop_name] = {
                'radius': profile['radius'].tolist(),
                'value': profile['value'].tolist()
            }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanetModel':
        """
        Create model from dictionary.

        Parameters
        ----------
        data : Dict
            Dictionary representation (from to_dict())

        Returns
        -------
        model : PlanetModel
            Reconstructed model
        """
        properties = {}
        for prop_name, profile in data['properties'].items():
            properties[prop_name] = {
                'radius': np.array(profile['radius']),
                'value': np.array(profile['value'])
            }

        return cls(
            radius=data['radius'],
            name=data['name'],
            discontinuities=data['discontinuities'],
            properties=properties,
            metadata=data['metadata']
        )

    def resample_property(
        self,
        property_name: str,
        new_radius: Union[np.ndarray, List[float]],
        method: str = 'linear'
    ) -> None:
        """
        Resample a property onto new radius points.

        Parameters
        ----------
        property_name : str
            Name of property to resample
        new_radius : array-like
            New radius points (must be within existing range)
        method : str
            Interpolation method ('linear', 'cubic', etc.)
        """
        if property_name not in self.properties:
            raise KeyError(f"Property '{property_name}' not found")

        new_radius = np.asarray(new_radius)

        if method == 'linear':
            new_values = self.interpolate_property(
                property_name, radius=new_radius
            )
        elif method == 'cubic':
            from scipy.interpolate import interp1d
            profile = self.properties[property_name]
            f = interp1d(
                profile['radius'], profile['value'],
                kind='cubic', bounds_error=True
            )
            new_values = f(new_radius)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

        # Update the property
        self.properties[property_name] = {
            'radius': new_radius,
            'value': new_values
        }

    def get_depth_range(self) -> Tuple[float, float]:
        """Get depth range covered by all properties."""
        if not self.properties:
            return (0.0, self.radius)

        min_radius = self.radius
        max_radius = 0.0

        for profile in self.properties.values():
            min_radius = min(min_radius, np.min(profile['radius']))
            max_radius = max(max_radius, np.max(profile['radius']))

        min_depth = self.radius - max_radius
        max_depth = self.radius - min_radius

        return (min_depth, max_depth)

    def get_radius_range(self) -> Tuple[float, float]:
        """Get radius range covered by all properties."""
        if not self.properties:
            return (0.0, self.radius)

        min_radius = self.radius
        max_radius = 0.0

        for profile in self.properties.values():
            min_radius = min(min_radius, np.min(profile['radius']))
            max_radius = max(max_radius, np.max(profile['radius']))

        return (min_radius, max_radius)
