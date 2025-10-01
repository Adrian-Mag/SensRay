# SensRay

A convenience Python package for computing and visualizing ray theoretical sensitivity kernels based on a 1D background model. The core functionality for ray tracing and visualization is basically a wrapper of Obspy.Taup with some extra convenience methods.

## Overview

SensRay provides modern tools for seismic ray path analysis and Earth model studies:

- **Travel Time Calculations**: Based on Obspy.taup.
- **Ray Path Tracing**: Based on Obspy.taup
- **Earth Model Comparison**: For now it only deals with the three models available in Obspy.taup, but I will add the option to use custom models, including for other planets/satelites.
- **2D Visualization**: Circular Earth cross-sections with ray plotting. It uses the ray information from Obspy.taup but then the plotting is done separately to add more custom features that are not available in the minimalist plotting of Obspy.
- **3D Visualization**: This uses the geographic ray tracing of Obspy.taup to get ray paths in geographic coordinates and plots them in an interactive 3D Earth.

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
from sensray import TravelTimeCalculator, RayPathTracer, EarthPlotter
import matplotlib.pyplot as plt

# Calculate travel times for different phases
calc = TravelTimeCalculator('iasp91')
arrivals = calc.calculate_travel_times(
    source_depth=10,  # km
    distance=60       # degrees
)

print(f"Found {len(arrivals)} seismic phases")
for arrival in arrivals:
    print(f"{arrival.name}: {arrival.time:.1f} seconds")

# Extract and visualize ray paths
tracer = RayPathTracer('prem')
ray_paths, info = tracer.get_ray_paths(
    source_depth=10,
    distance_deg=60,
    phases=['P', 'S', 'PP']
)

# Create publication-quality 2D visualization
plotter = EarthPlotter()
ray_coordinates = tracer.extract_ray_coordinates(ray_paths)

fig = plotter.plot_circular_earth(
    ray_coordinates=ray_coordinates,
    source_depth=10,
    distance_deg=60,
    fig_size=(10, 10)
)
plt.show()
```

## Demos

Explore the package capabilities through interactive Jupyter notebooks:

- **[01_basic_travel_times.ipynb](demos/01_basic_travel_times.ipynb)**: Start here! Learn travel time calculations and Earth models
- **[02_ray_path_visualization.ipynb](demos/02_ray_path_visualization.ipynb)**: Extract and visualize ray paths in 2D cross-sections
- **[03_earth_model_comparison.ipynb](demos/03_earth_model_comparison.ipynb)**: Compare IASP91, PREM, and AK135 models.
- **[04_3d_plots.ipynb](demos/04_3d_plots.ipynb)**: Interactive 3D visualization with PyVista

Run the demos: `jupyter notebook demos/00_index.ipynb`

<p align="center">
    <img src="docs/screenshot.png" width="800" alt="SensRay demo screenshot" />
</p>

## Core Modules

- **core.travel_times**: Travel time calculation with multiple Earth models
- **core.ray_paths**: Ray path extraction with geographic coordinate support
- **core.earth_models**: Earth model management and 1D profile visualization
- **visualization.earth_plots**: Circular Earth cross-sections and ray plotting
- **visualization.earth_3d**: Interactive 3D visualization with PyVista
- **kernels.sensitivity**: NOT DONE!

## Dependencies

- NumPy, SciPy, Matplotlib
- **PyVista**: For 3D interactive visualization
- **ObsPy**: For seismic data processing and ray calculations
- **Cartopy**: For geographic data (optional)

## License

MIT License
