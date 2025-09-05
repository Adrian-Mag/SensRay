"""
Earth model management and utilities.
"""

import numpy as np
from typing import List, Dict, Optional
from obspy.taup import TauPyModel


class EarthModelManager:
    """
    Manager for Earth models and their properties.
    """

    # Standard Earth model parameters
    STANDARD_MODELS = {
        'iasp91': {
            'name': 'IASP91',
            'description': 'International Association of Seismology and Physics of the Earth Interior 1991',
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

    def get_phase_availability(self, model_name: str, phases: List[str]) -> Dict:
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
