# SeisRay

A Python package for **ray-theoretical sensitivity kernel computation** and **3D visualization** for seismic tomography.

## Overview

SeisRay focuses on specialized tools for seismic tomography that complement ObsPy:

- **Sensitivity Kernels**: Compute ray-theoretical sensitivity kernels for tomographic inversions
- **3D Visualization**: Interactive 3D visualization of kernels, ray paths, and Earth structure using PyVista
- **Tomography Workflows**: Tools specifically designed for tomographic applications

**For basic seismology tasks**, use ObsPy directly:
- `obspy.taup.TauPyModel` for travel times and ray paths
- `arrivals.plot_rays()` for excellent 2D visualization
- ObsPy provides comprehensive coverage of standard seismological analysis

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev,notebooks]"
```

## Quick Start

```python
from seisray import SensitivityKernel, Earth3DVisualizer
from obspy.taup import TauPyModel

# Compute sensitivity kernels for tomography
domain_bounds = (-500, 500, 0, 1000)  # x_min, x_max, y_min, y_max (km)
grid_size = (50, 50)  # nx, ny cells

kernel_calc = SensitivityKernel(
    domain_bounds=domain_bounds,
    grid_size=grid_size
)

# Compute kernel for a source-receiver pair
kernel = kernel_calc.compute_ray_kernel(
    source_pos=(0, 200),     # source at (x, y) in km
    receiver_pos=(300, 50),  # receiver at (x, y) in km
    ray_type='straight'
)

# 3D visualization
viz3d = Earth3DVisualizer()
# ... create interactive 3D plots
```

## Demos

- `demos/01_sensitivity_kernels.ipynb`: Sensitivity kernel computation for tomography
- `demos/02_3d_visualization.ipynb`: 3D visualization of kernels and ray paths

## Core Modules

- **kernels.sensitivity**: Ray-theoretical sensitivity kernel computation
- **visualization.earth_3d**: Interactive 3D visualization with PyVista

## Key Applications

1. **Local Earthquake Tomography**: Compute kernels for velocity inversions
2. **Resolution Assessment**: Visualize ray coverage and model resolution
3. **Experiment Design**: Plan optimal source-receiver geometries
4. **Result Visualization**: Interactive 3D exploration of tomographic models

## Dependencies

- NumPy, SciPy, Matplotlib
- **PyVista**: For 3D interactive visualization
- **ObsPy**: For seismic data processing and ray calculations
- **Cartopy**: For geographic data (optional)

## License

MIT License
