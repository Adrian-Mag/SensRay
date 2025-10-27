
# SensRay

Lightweight utilities for building planet (1D) models, generating meshes, computing per-cell ray path lengths and simple sensitivity kernels. SensRay intentionally provides a small, stable public API and delegates ray tracing to ObsPy/TauP and 3D rendering to PyVista.

## Highlights

- **PlanetModel**: load .nd TauP-style 1D models, inspect profiles, and build a TauPyModel when needed.
- **PlanetMesh**: generate **1D spherical** or **3D tetrahedral** meshes from a PlanetModel, map properties onto mesh cells, compute per-cell ray lengths and sensitivity kernels, and visualize results.
- **1D Spherical Meshes**: Fast, simple meshes for radially-symmetric models with variable layer spacing (perfect for testing and global-scale studies).
- **3D Tetrahedral Meshes**: Full 3D meshes for laterally-varying models and complex geometries.
- **Sensitivity Kernels**: Compute travel-time sensitivity kernels (K = -L/v²) for single or multiple rays with automatic accumulation.
- **CoordinateConverter**: simple geographic/cartesian helpers used internally and available for convenience.

This README documents the current public surface of the package and quick examples. For detailed workflows see the notebooks in `demos/`.

## Installation

Install in editable/development mode (recommended for contributors):

```bash
pip install -e .
```

Optional extras for demos and development:

```bash
pip install -e "[dev,notebooks]"
```

Dependencies you will likely need for full functionality:

- obsPy (TauP) — ray calculation
- pyvista — mesh I/O & visualization
- pygmsh + meshio — tetrahedral mesh generation

Only core operations (PlanetModel parsing) work without these optional heavy deps.

## Quick API overview

Import the small public API:

```py
from sensray import PlanetModel

# Load a standard model
model = PlanetModel.from_standard_model('prem')

# or load a custom .nd file
model = PlanetModel('path/to/model.nd')

# === 1D Spherical Mesh (recommended for radially-symmetric models) ===
# Generate a 1D spherical mesh with variable layer spacing
import numpy as np

# Define custom radii (descending order: surface to center)
# Example: variable spacing with finer resolution in upper mantle
radii = np.linspace(0, model.radius, 50)[::-1]
model.mesh.generate_spherical_mesh(radii=radii)

# Populate mesh with properties from the model
model.mesh.populate_properties(['vp', 'vs', 'rho'])

# Compute ray paths using TauP (for 1D models)
rays = model.taupy_model.get_ray_paths(
    source_depth_in_km=10.0,
    distance_in_degree=60.0,
    phase_list=['P']
)

# Compute ray lengths through each spherical shell
lengths = model.mesh.compute_ray_lengths(rays[0], store_as='P_lengths')

# Compute sensitivity kernel K = -L/v²
K = model.mesh.compute_sensitivity_kernel(
    rays[0],
    property_name='vp',
    attach_name='K_P'
)

# For multiple rays, use accumulate='sum' to get total kernel
multiple_rays = [...]  # list of rays
K_total = model.mesh.compute_sensitivity_kernel(
    multiple_rays,
    property_name='vp',
    accumulate='sum'
)

# Save mesh with all computed data
model.mesh.save('output_mesh')

# === 3D Tetrahedral Mesh (for laterally-varying models) ===
# Generate a 3D tetrahedral mesh
model.mesh.generate_tetrahedral_mesh(mesh_size_km=200.0)

# Populate properties using volume-weighted integration
model.mesh.populate_properties(['vp', 'vs', 'rho'])

# Compute ray paths in geographic coordinates
rays = model.taupy_model.get_ray_paths_geo(
    source_depth_in_km=10.0,
    source_latitude_in_deg=0.0,
    source_longitude_in_deg=0.0,
    receiver_latitude_in_deg=30.0,
    receiver_longitude_in_deg=60.0,
    phase_list=['P']
)

# Same API works for both mesh types!
lengths = model.mesh.compute_ray_lengths(rays[0], store_as='P_lengths')
K = model.mesh.compute_sensitivity_kernel(rays[0], 'vp', attach_name='K_P')

# Visualize a cross section (3D meshes only)
plotter = model.mesh.plot_cross_section(property_name='K_P')
plotter.show()
```

Notes on API behavior
- **PlanetModel** is read-only: it parses .nd files and provides get_property_at_depth/profile helpers and a lazy TauPyModel accessor.
- **PlanetMesh** supports both **1D spherical** and **3D tetrahedral** mesh generation:
  - **1D spherical meshes**: Fast, simple, perfect for radially-symmetric models. Supports variable layer spacing (uniform in cores, increasing towards surface).
  - **3D tetrahedral meshes**: Full 3D geometry for laterally-varying models. Requires `pygmsh` + `meshio`.
- **Unified API**: `compute_ray_lengths()` and `compute_sensitivity_kernel()` work identically for both mesh types.
- **Ray tracing**: 1D meshes use `model.taupy_model.get_ray_paths()`, 3D meshes use `model.taupy_model.get_ray_paths_geo()`.
- **Sensitivity kernels**: Implemented as K = -L / v². Supports single rays or multiple rays with `accumulate='sum'` or `accumulate=None` (returns 2D array).
- **Property projection**: Uses volume-weighted radial integration for accurate mapping from continuous model to discrete mesh cells.

## Demos

See the notebooks in `demos/` for runnable examples:

- **`01_basic_usage.ipynb`** — Load models, create tetrahedral mesh, populate properties, save/load mesh
- **`02_ray_tracing_kernels.ipynb`** — Trace rays using ObsPy/TauP on 3D tetrahedral meshes, compute per-cell ray lengths, compute sensitivity kernels, and visualize results
- **`03_1D_meshes.ipynb`** — Introduction to 1D spherical meshes: generation, property mapping, and basic operations
- **`04_spherical_ray_tracing_kernels.ipynb`** — Complete 1D workflow: ray tracing on spherical meshes, path lengths, sensitivity kernels (single and multiple rays)
- **`05_spherical_mesh_advanced.ipynb`** — Advanced 1D features: projection accuracy tests, API validation, performance comparisons, variable layer spacing

Pre-generated demo meshes are present as VTU files in `demos/` for quick testing (e.g. `prem_mesh.vtu`, `prem_mesh_with_rays_kernels.vtu`).

## Recommended workflow

### For 1D Spherical Meshes (recommended for radially-symmetric models):

1. Create or load a PlanetModel: `PlanetModel.from_standard_model('prem')` or `PlanetModel('file.nd')`.
2. Define radii with variable spacing (e.g., uniform in cores, increasing in mantle).
3. Generate spherical mesh: `model.mesh.generate_spherical_mesh(radii=radii)`.
4. Populate properties: `model.mesh.populate_properties(['vp', 'vs'])`.
5. Compute ray paths with TauP: `model.taupy_model.get_ray_paths(source_depth_in_km=..., distance_in_degree=..., phase_list=[...])`.
6. Compute ray lengths: `model.mesh.compute_ray_lengths(ray)`.
7. Compute sensitivity kernels: `model.mesh.compute_sensitivity_kernel(ray, 'vp')`.
8. Save mesh: `model.mesh.save('my_mesh')`.

### For 3D Tetrahedral Meshes (for laterally-varying models):

1. Create or load a PlanetModel: `PlanetModel.from_standard_model('prem')` or `PlanetModel('file.nd')`.
2. Generate tetrahedral mesh: `model.mesh.generate_tetrahedral_mesh(mesh_size_km=200.0)`.
3. Populate properties: `model.mesh.populate_properties(['vp', 'vs'])`.
4. Compute ray paths with geographic coordinates: `model.taupy_model.get_ray_paths_geo(...)`.
5. Compute ray lengths: `model.mesh.compute_ray_lengths(ray)`.
6. Compute sensitivity kernels: `model.mesh.compute_sensitivity_kernel(ray, 'vp')`.
7. Visualize: `model.mesh.plot_cross_section(property_name='vp').show()`.
8. Save mesh: `model.mesh.save('my_mesh')`.

**Note**: The same API (`compute_ray_lengths`, `compute_sensitivity_kernel`) works for both mesh types!

## Notes for contributors

- Keep the public API small: the package intentionally exposes only `PlanetModel`, `PlanetMesh` and `CoordinateConverter`.
- Heavy dependencies (pyvista, pygmsh) are optional and should be imported lazily where possible. Unit tests should skip features requiring optional heavy deps when not available.

## Troubleshooting & tips

- If `PlanetMesh` raises import errors, install the optional visualization/mesh dependencies: `pip install pyvista pygmsh meshio`.
- **1D vs 3D meshes**: Use 1D spherical meshes for radially-symmetric models (much faster, simpler visualization). Use 3D tetrahedral meshes for laterally-varying models.
- **Variable layer spacing**: For 1D meshes, use smaller spacing where you need higher resolution (e.g., upper mantle, crust) and larger spacing where gradients are smooth (e.g., lower mantle, cores).
- **Ray tracing**: 1D meshes use `get_ray_paths(source_depth_in_km, distance_in_degree)`, 3D meshes use `get_ray_paths_geo()` with lat/lon coordinates.
- **Multiple rays**: Use `accumulate='sum'` to get a single summed kernel, or `accumulate=None` to get individual kernels as a 2D array.
- For large images generated by demos, keep PNG widths around 800–1200px to render nicely in the README.

## License

MIT


## License

MIT

```
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

Explore the package capabilities through concise interactive Jupyter notebooks (found in `demos/`):

- **[01_basic_usage.ipynb](demos/01_basic_usage.ipynb)** — basic usage: load a 1D Earth model, create a tetrahedral mesh, populate simple properties and visualize them.
- **[02_ray_tracing_kernels.ipynb](demos/02_ray_tracing_kernels.ipynb)** — ray tracing and sensitivity kernels: compute per-cell ray lengths and sensitivity kernels, visualize and save results.

There are also pre-generated demo mesh files in `demos/` (VTU + metadata): `prem_tet_demo.vtu`, `prem_tet_rays_kernels_demo.vtu`.

Run the demos locally with Jupyter, for example:

```
jupyter notebook demos/01_basic_usage.ipynb
```

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
- **kernels.sensitivity**: Basic sensitivity-kernel helpers are implemented on `PlanetMesh` (see `demos/02_ray_tracing_kernels.ipynb`).

What's new
-- Tetrahedral mesh generation and demo notebooks with mesh export.
-- Per-ray and multi-ray sensitivity kernel computation implemented and exposed on `PlanetMesh`.

## Dependencies

- NumPy, SciPy, Matplotlib
- **PyVista**: For 3D interactive visualization
- **ObsPy**: For seismic data processing and ray calculations
- **Cartopy**: For geographic data (optional)

## License

MIT License
