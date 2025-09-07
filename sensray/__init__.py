"""
SensRay: Seismic Ray Tracing, Travel Times, and 3D Visualization

This package provides comprehensive tools for seismic ray path analysis:
- Travel time calculations for P, S, and other seismic phases
- Ray path extraction and coordinate conversion
- Earth model comparison (IASP91, PREM, AK135)
- 2D circular Earth cross-section visualization
- Interactive 3D visualization with PyVista
- Statistical analysis of model differences

Key Classes:
- TravelTimeCalculator: Compute travel times for seismic phases
- RayPathTracer: Extract and analyze ray paths with geographic coordinates
- EarthModelManager: Manage Earth models and plot velocity profiles
- EarthPlotter: Create publication-quality 2D ray path visualizations
- Earth3DVisualizer: Interactive 3D visualization
- SensitivityKernel: Ray-theoretical sensitivity kernels for tomography

Authors: PhD Project
Version: 0.3.0
"""

__version__ = "0.3.0"
__author__ = "PhD Project"

# Import unique functionality only
from .visualization.earth_3d import Earth3DVisualizer
from .kernels.sensitivity import SensitivityKernel

# Core API convenience exports
from .core.travel_times import TravelTimeCalculator
from .core.earth_models import EarthModelManager
from .core.ray_paths import RayPathTracer
from .visualization.earth_plots import EarthPlotter

__all__ = [
    'Earth3DVisualizer',
    'SensitivityKernel',
    'TravelTimeCalculator',
    'EarthModelManager',
    'RayPathTracer',
    'EarthPlotter'
]
