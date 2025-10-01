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
- PlanetModel: 1D planetary models with seismic property profiles
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
# Earth3DVisualizer requires PyVista - import only if available
try:
    from .visualization.earth_3d import Earth3DVisualizer
    _has_earth3d = True
except ImportError:
    _has_earth3d = False

from .mesh.earth_model import MeshEarthModel
from .core.model import PlanetModel

# Core API convenience exports (these may require ObsPy)
try:
    from .core.travel_times import TravelTimeCalculator
    from .core.ray_paths import RayPathTracer
    _has_core = True
except ImportError:
    _has_core = False

from .visualization.earth_plots import EarthPlotter

# Dynamic __all__ based on available imports
__all__ = ['MeshEarthModel', 'PlanetModel', 'EarthPlotter']

if _has_earth3d:
    __all__.append('Earth3DVisualizer')

if _has_core:
    __all__.extend(['TravelTimeCalculator', 'RayPathTracer'])
