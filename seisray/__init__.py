"""
SeisRay: A Python package for seismic ray tracing and travel time calculations.

This package provides tools for:
- Computing travel times using 1D Earth models
- Ray path visualization with circular Earth cross-sections
- Sensitivity kernel calculations
- Geographic coordinate handling
- Model comparisons and analysis

Authors: PhD Project
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "PhD Project"

# Import main classes and functions for easy access
from .core.travel_times import TravelTimeCalculator
from .core.ray_paths import RayPathTracer
from .core.earth_models import EarthModelManager
from .visualization.earth_plots import EarthPlotter
from .kernels.sensitivity import SensitivityKernel
from .utils.coordinates import CoordinateConverter

__all__ = [
    'TravelTimeCalculator',
    'RayPathTracer',
    'EarthModelManager',
    'EarthPlotter',
    'SensitivityKernel',
    'CoordinateConverter'
]
