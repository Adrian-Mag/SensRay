# SeisRay

A Python package for seismic ray tracing and travel time calculations in 1D Earth models.

## Features

- **Travel Time Calculations**: Compute P and S wave travel times using various 1D Earth models
- **Ray Path Tracing**: Extract and analyze seismic ray paths through the Earth
- **Earth Model Management**: Work with standard models (iasp91, prem, ak135) and custom models
- **Visualization**: Create circular Earth cross-sections with ray paths and travel time curves
- **Sensitivity Kernels**: Compute discretized sensitivity kernels for tomographic applications

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
from seisray import TravelTimeCalculator, RayPathTracer, EarthPlotter

# Calculate travel times
calc = TravelTimeCalculator(model='iasp91')
times = calc.calculate_travel_times(source_depth=10, distance_deg=30)

# Extract ray paths
tracer = RayPathTracer(model='iasp91')
rays = tracer.get_ray_paths(source_depth=10, distance_deg=30, phase_list=['P', 'S'])

# Plot results
plotter = EarthPlotter()
plotter.plot_earth_with_rays(rays, source_depth=10)
```

## Tutorial

See the included Jupyter notebook `ray_tracing_tutorial.ipynb` for a comprehensive tutorial.

## Demos

There is an interactive 3D demonstration available in `demos/04_3d_plots.ipynb` showcasing
geographic ray path plotting on a PyVista globe with continent outlines and interactive controls.

## Modules

- **core.travel_times**: Travel time calculations
- **core.ray_paths**: Ray path extraction and analysis
- **core.earth_models**: Earth model management
- **visualization.earth_plots**: Plotting utilities
- **kernels.sensitivity**: Sensitivity kernel computation
- **utils.coordinates**: Coordinate conversion utilities

## Dependencies

- NumPy
- Matplotlib
- SciPy
- ObsPy

## License

MIT License
