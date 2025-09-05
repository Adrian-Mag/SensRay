"""
SeisRay: Sensitivity Kernels and 3D Visualization for Seismic Tomography

This package provides specialized tools for ray-theoretical seismic tomography:
- Ray-theoretical sensitivity kernel computation for tomographic inversions
- Interactive 3D visualization of kernels, ray paths, and Earth structure
- Integration with ObsPy for seismic data processing

For basic seismic analysis, use ObsPy directly:
- obspy.taup.TauPyModel for travel times and ray paths
- arrivals.plot_rays() for 2D ray path visualization
- ObsPy has excellent coverage of standard seismological tasks

SeisRay focuses on advanced tomography applications and 3D visualization.

Authors: PhD Project
Version: 0.2.0
"""

__version__ = "0.2.0"
__author__ = "PhD Project"

# Import unique functionality only
from .visualization.earth_3d import Earth3DVisualizer
from .kernels.sensitivity import SensitivityKernel

__all__ = [
    'Earth3DVisualizer',
    'SensitivityKernel'
]
