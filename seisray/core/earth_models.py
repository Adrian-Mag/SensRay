"""
Earth model management and utilities.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class EarthModelManager:
    """
    Manager for Earth models and their properties.
    """

    # Standard Earth model parameters
    STANDARD_MODELS = {
        'iasp91': {
            'name': 'IASP91',
            'description': (
                'International Assoc. of Seismology & Physics of the Earth '
                'Interior (1991)'
            ),
            'earth_radius': 6371.0,
            'cmb_depth': 2891.0,
            'icb_depth': 5150.0
        },
        'prem': {
            'name': 'PREM',
            'description': 'Preliminary Reference Earth Model',
            'earth_radius': 6371.0,
            'cmb_depth': 2891.0,
            'icb_depth': 5150.0
        },
        'ak135': {
            'name': 'AK135',
            'description': 'Kennett & Engdahl 1995 model',
            'earth_radius': 6371.0,
            'cmb_depth': 2891.0,
            'icb_depth': 5150.0
        }
    }

    def __init__(self):
        """Initialize the Earth model manager."""
        self.loaded_models = {}

    def get_model(self, model_name: str) -> TauPyModel:
        """
        Get or load an Earth model.

        Parameters
        ----------
        model_name : str
            Name of the Earth model

        Returns
        -------
        model : TauPyModel
            The loaded Earth model
        """
        if model_name not in self.loaded_models:
            self.loaded_models[model_name] = TauPyModel(model=model_name)

        return self.loaded_models[model_name]

    # --------------------- 1D profile utilities --------------------- #
    def _get_velocity_layers(self, model_name: str) -> Any:
        """Return the structured array of velocity layers from TauP.

        Notes
        -----
    TauPyModel exposes an inner `model` with `s_mod.v_mod.layers` that
    holds a structured numpy array with fields like:
        ('top_depth','bot_depth','top_p_velocity','bot_p_velocity',
         'top_s_velocity','bot_s_velocity','top_density','bot_density',
         'top_qp','bot_qp','top_qs','bot_qs')
        """
        model = self.get_model(model_name)
        try:
            # Access the structured array of layers from TauP internals.
            layers = (
                model.model.s_mod.v_mod.layers  # type: ignore[attr-defined]
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to access velocity layers for model '"
                f"{model_name}': {exc}"
            )
        return layers

    def get_1d_profile(
        self,
        model_name: str,
        properties: Optional[List[str]] = None,
        max_depth_km: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract a 1D profile (depth vs properties) from a TauP Earth model.

        Parameters
        ----------
        model_name : str
            Name of the Earth model (e.g., 'iasp91', 'prem', 'ak135').
        properties : List[str], optional
            Properties to extract. Supported (if present in the model):
            'vp', 'vs', 'rho', 'qp', 'qs'. Defaults to ['vp','vs'].
        max_depth_km : float, optional
            If provided, clip the profile to this maximum depth.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary with key 'depth_km' and one array per requested
            property.
        """
        if properties is None:
            properties = ['vp', 'vs']

        prop_map: Dict[str, tuple] = {
            'vp': ('top_p_velocity', 'bot_p_velocity'),
            'vs': ('top_s_velocity', 'bot_s_velocity'),
            'rho': ('top_density', 'bot_density'),
            'qp': ('top_qp', 'bot_qp'),
            'qs': ('top_qs', 'bot_qs'),
            # aliases
            'density': ('top_density', 'bot_density'),
        }

        layers = self._get_velocity_layers(model_name)

        depths: List[float] = []
        series: Dict[str, List[float]] = {p: [] for p in properties}

        # Build piecewise-linear profiles by adding the top and bottom of every
        # layer
        for rec in layers:
            top_d = float(rec['top_depth'])
            bot_d = float(rec['bot_depth'])

            # optionally skip beyond max depth
            if max_depth_km is not None and top_d > max_depth_km:
                break

            # Append top
            depths.append(top_d)
            for p in properties:
                fields = prop_map.get(p)
                if fields and fields[0] in layers.dtype.names:
                    series[p].append(float(rec[fields[0]]))
                else:
                    series[p].append(float('nan'))

            # Append bottom
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

        # Deduplicate depths while preserving order; average duplicate entries
        # per depth
        d = np.array(depths)
        unique_depths: List[float] = []
        idx_groups: Dict[float, List[int]] = {}
        for i, val in enumerate(d):
            if (
                len(unique_depths) == 0
                or not np.isclose(unique_depths[-1], val)
            ):
                unique_depths.append(float(val))
            idx_groups.setdefault(float(val), []).append(i)

        out: Dict[str, np.ndarray] = {'depth_km': np.array(unique_depths)}
        for p in properties:
            arr = np.array(series[p], dtype=float)
            # average duplicates at the same depth
            vals: List[float] = []
            for val in unique_depths:
                inds = idx_groups[val]
                vals.append(float(np.nanmean(arr[inds])))
            out[p] = np.array(vals)

        return out

    def plot_1d_profile(
        self,
        model_name: str,
        properties: Optional[List[str]] = None,
        max_depth_km: Optional[float] = None,
        fig_size=(8, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Plot 1D Earth model profiles (e.g., Vp, Vs, density) versus depth.

        Parameters
        ----------
        model_name : str
            Earth model name.
        properties : List[str], optional
            Which properties to plot. Defaults to ['vp','vs'].
        max_depth_km : float, optional
            Clip at this depth for readability.
        fig_size : tuple, optional
            Matplotlib figure size.
        ax : matplotlib.axes.Axes, optional
            Existing axis to plot into; if None, a new figure is created.

        Returns
        -------
        matplotlib.figure.Figure
            The created or associated figure.
        """
        if properties is None:
            properties = ['vp', 'vs']

        profile = self.get_1d_profile(model_name, properties, max_depth_km)
        depths = profile['depth_km']

        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure

        colors = {
            'vp': 'tab:blue',
            'vs': 'tab:orange',
            'rho': 'tab:green',
            'qp': 'tab:red',
            'qs': 'tab:purple',
            'density': 'tab:green',
        }
        labels = {
            'vp': 'Vp (km/s)',
            'vs': 'Vs (km/s)',
            'rho': 'Density (g/cc)',
            'qp': 'Qp',
            'qs': 'Qs',
            'density': 'Density (g/cc)',
        }

        for p in properties:
            if p not in profile:
                continue
            ax.plot(
                profile[p],
                depths,
                label=labels.get(p, p.upper()),
                color=colors.get(p),
            )

        ax.invert_yaxis()
        ax.set_xlabel('Value')
        ax.set_ylabel('Depth (km)')
        ax.set_title(f'1D Earth Model Profile - {model_name.upper()}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about an Earth model.

        Parameters
        ----------
        model_name : str
            Name of the Earth model

        Returns
        -------
        info : Dict
            Dictionary with model information
        """
        if model_name.lower() in self.STANDARD_MODELS:
            return self.STANDARD_MODELS[model_name.lower()].copy()
        else:
            return {
                'name': model_name.upper(),
                'description': f'Custom model: {model_name}',
                'earth_radius': 6371.0,
                'cmb_depth': 2891.0,
                'icb_depth': 5150.0
            }

    def list_available_models(self) -> List[str]:
        """
        List all available Earth models.

        Returns
        -------
        models : List[str]
            List of available model names
        """
        return list(self.STANDARD_MODELS.keys())

    def get_earth_structure(self, model_name: str) -> Dict:
        """
        Get Earth structure parameters for plotting.

        Parameters
        ----------
        model_name : str
            Name of the Earth model

        Returns
        -------
        structure : Dict
            Dictionary with Earth structure parameters
        """
        info = self.get_model_info(model_name)

        earth_radius = info['earth_radius']
        cmb_depth = info['cmb_depth']
        icb_depth = info['icb_depth']

        structure = {
            'earth_radius': earth_radius,
            'cmb_radius': earth_radius - cmb_depth,
            'icb_radius': earth_radius - icb_depth,
            'surface_depth': 0.0,
            'cmb_depth': cmb_depth,
            'icb_depth': icb_depth,
            'boundaries': {
                'surface': 0.0,
                'moho': 35.0,  # approximate
                'cmb': cmb_depth,
                'icb': icb_depth
            },
            'layers': {
                'crust': (0.0, 35.0),
                'mantle': (35.0, cmb_depth),
                'outer_core': (cmb_depth, icb_depth),
                'inner_core': (icb_depth, earth_radius)
            }
        }

        return structure

    def compare_models_structure(self, models: List[str]) -> Dict:
        """
        Compare structural parameters between models.

        Parameters
        ----------
        models : List[str]
            List of model names to compare

        Returns
        -------
        comparison : Dict
            Dictionary with comparison data
        """
        comparison = {}

        for model_name in models:
            structure = self.get_earth_structure(model_name)
            comparison[model_name] = {
                'earth_radius': structure['earth_radius'],
                'cmb_radius': structure['cmb_radius'],
                'icb_radius': structure['icb_radius'],
                'cmb_depth': structure['cmb_depth'],
                'icb_depth': structure['icb_depth']
            }

        return comparison

    def validate_model(self, model_name: str) -> bool:
        """
        Validate that a model can be loaded.

        Parameters
        ----------
        model_name : str
            Name of the model to validate

        Returns
        -------
        valid : bool
            True if model can be loaded, False otherwise
        """
        try:
            self.get_model(model_name)
            return True
        except Exception:
            return False

    def get_phase_availability(self, model_name: str,
                               phases: List[str]) -> Dict:
        """
        Check which phases are available in a model.

        Parameters
        ----------
        model_name : str
            Name of the Earth model
        phases : List[str]
            List of phases to check

        Returns
        -------
        availability : Dict
            Dictionary showing phase availability
        """
        availability = {}
        model = self.get_model(model_name)

        # Test with a standard configuration
        test_depth = 100.0
        test_distance = 60.0

        for phase in phases:
            try:
                arrivals = model.get_travel_times(
                    source_depth_in_km=test_depth,
                    distance_in_degree=test_distance,
                    phase_list=[phase]
                )
                availability[phase] = len(arrivals) > 0
            except Exception:
                availability[phase] = False

        return availability
