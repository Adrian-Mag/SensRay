# SensRay

A Python package for **seismic ray tracing**, **travel time calculations**, and **3D visualization** for seismological research.

## Overview

SensRay provides modern tools for seismic ray path analysis and Earth model studies:

- **Travel Time Calculations**: Fast, accurate travel time computations for P and S waves
- **Ray Path Tracing**: Extract and analyze seismic ray paths with geographic coordinates
- **Earth Model Comparison**: Compare velocity structures between IASP91, PREM, and AK135
- **2D Visualization**: Circular Earth cross-sections with professional ray path plotting
- **3D Visualization**: Interactive 3D visualization using PyVista for publication-quality graphics
- **Model Analysis**: Statistical comparison and validation of Earth models

Built on **ObsPy** foundation with enhanced visualization and analysis capabilities.

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
- **[03_earth_model_comparison.ipynb](demos/03_earth_model_comparison.ipynb)**: Compare IASP91, PREM, and AK135 models with statistical analysis
- **[04_3d_plots.ipynb](demos/04_3d_plots.ipynb)**: Interactive 3D visualization with PyVista

Run the demos: `jupyter notebook demos/00_index.ipynb`

## Core Modules

- **core.travel_times**: Travel time calculation with multiple Earth models
- **core.ray_paths**: Ray path extraction with geographic coordinate support
- **core.earth_models**: Earth model management and 1D profile visualization
- **visualization.earth_plots**: Circular Earth cross-sections and ray plotting
- **visualization.earth_3d**: Interactive 3D visualization with PyVista
- **kernels.sensitivity**: Ray-theoretical sensitivity kernel computation

## Key Applications

1. **Seismic Phase Analysis**: Identify and analyze P, S, PP, SS, and other seismic phases
2. **Earth Model Studies**: Compare velocity structures and understand model limitations
3. **Ray Path Analysis**: Study wave propagation paths through Earth's interior
4. **Educational Tools**: Teach seismology concepts with interactive visualizations
5. **Research Applications**: Support earthquake location, velocity structure studies
6. **Quality Control**: Validate seismic data and model predictions

## Dependencies

- NumPy, SciPy, Matplotlib
- **PyVista**: For 3D interactive visualization
- **ObsPy**: For seismic data processing and ray calculations
- **Cartopy**: For geographic data (optional)

## License

MIT License
