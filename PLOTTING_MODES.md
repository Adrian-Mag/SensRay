# SensRay Plotting Modes: Static vs Interactive

SensRay provides 3D visualization capabilities using PyVista. You can choose between **static** (image-based) and **interactive** (widget-based) plotting modes depending on your needs and environment. Plotting mode is controlled globally via PyVista configuration and by how you call `show()`. There is no per-call `static` parameter.

## Quick Start

### For Jupyter Notebooks (Recommended)

We recommend setting a single INTERACTIVE flag in the first cell of your notebook and configuring PyVista accordingly before importing SensRay. This avoids ambiguous behavior mid-notebook.

```python
# Cell 1: Choose a plotting mode before importing SensRay
INTERACTIVE = True  # Set to False for reproducible/static images (headless)

import os
if INTERACTIVE:
    # Prefer widget-based backends in notebooks
    os.environ.pop('PYVISTA_OFF_SCREEN', None)
    os.environ['PYVISTA_USE_IPYVTK'] = 'true'
else:
    # Force static/off-screen rendering
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['PYVISTA_USE_IPYVTK'] = 'false'

import pyvista as pv
if INTERACTIVE:
    # Try widget backends (order of preference)
    try:
        pv.set_jupyter_backend('ipyvtklink')
    except Exception:
        try:
            pv.set_jupyter_backend('panel')
        except Exception as e:
            print('Could not set interactive backend:', e)
else:
    pv.set_jupyter_backend('static')

from sensray import PlanetModel

# Cell 2: Create plots as usual
model = PlanetModel.from_standard_model('prem')
model.create_mesh(mesh_size_km=1000)

# Use plotter.show(interactive=INTERACTIVE) for consistent behavior
plotter = model.mesh.plot_cross_section(property_name='vp')
plotter.show(interactive=INTERACTIVE, screenshot=('output.png' if not INTERACTIVE else None))
```

### For Python Scripts

```python
import os
# Configure for headless operation
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

from sensray import PlanetModel

model = PlanetModel.from_standard_model('prem')
model.create_mesh(mesh_size_km=1000)

# Generate static image
plotter = model.mesh.plot_cross_section(property_name='vp')
plotter.show(screenshot="output.png", auto_close=True, interactive=False)
```

## When to Use Each Mode

### ✅ Use Static Mode When:
- Running on systems without dedicated GPUs
- Working on remote servers (SSH, clusters)
- Creating figures for publications
- Running automated scripts
- Experiencing crashes with interactive plots
- Want consistent, reproducible outputs

### ✅ Use Interactive Mode When:
- Working locally with a GUI environment
- Need to explore data interactively
- Want to rotate, zoom, and inspect 3D structures
- Debugging visualization code
- Working with small datasets

## Detailed Configuration Guide

### 1. Jupyter Notebook Setup

#### Method A: Global Configuration (Recommended)
Configure PyVista at the very beginning of your notebook:

```python
# First cell - MUST be run before any SensRay imports
import os
# Force static mode for all PyVista operations
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTK'] = 'false'

# Import and configure PyVista backend
import pyvista as pv
pv.set_jupyter_backend('static')
print("✓ Configured PyVista for static plotting")

# Now import SensRay
import numpy as np
from sensray import PlanetModel, CoordinateConverter
```

#### Method B: Switching Modes Within a Notebook
If you want to mix static and interactive plots, switch PyVista's Jupyter backend between cells:

```python
# For static plots
import pyvista as pv
pv.set_jupyter_backend('static')
plotter = model.mesh.plot_cross_section(property_name='vp')
plotter.show(screenshot="static.png", auto_close=True, interactive=False)

# For interactive plots (if your system supports it)
pv.set_jupyter_backend('trame')  # or 'ipyvtklink'
plotter = model.mesh.plot_cross_section(property_name='vp')
plotter.show()
```

### 2. Python Script Setup

#### For Headless Environments (Servers, Clusters)
```python
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure PyVista for headless operation
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['DISPLAY'] = ':99'  # Virtual display if needed

from sensray import PlanetModel

# All plotting will be static
model = PlanetModel.from_standard_model('prem')
model.create_mesh(mesh_size_km=1000)

plotter = model.mesh.plot_cross_section(property_name='vp')
plotter.show(screenshot="headless_output.png", auto_close=True, interactive=False)
```

#### For Desktop Environments
```python
from sensray import PlanetModel

model = PlanetModel.from_standard_model('prem')
model.create_mesh(mesh_size_km=1000)

# Interactive plotting (opens window)
plotter = model.mesh.plot_cross_section(property_name='vp')
plotter.show()

# Or save to file
plotter.show(screenshot="desktop_output.png")
```

## Available Plotting Methods

All SensRay plotting methods return a `pyvista.Plotter`. Control static vs. interactive behavior via PyVista configuration and `show()` parameters.

### Cross-Section Plots
```python
plotter = model.mesh.plot_cross_section(
    plane_normal=(0, 1, 0),
    property_name='vp',
    show_rays=rays,
)
# Static file
plotter.show(screenshot="cross_section_vp.png", auto_close=True, interactive=False)
# Or interactive (if supported)
# plotter.show()
```

### Spherical Shell Plots
```python
plotter = model.mesh.plot_spherical_shell(
    radius_km=3480,  # Core-mantle boundary
    property_name='vs',
)
plotter.show(screenshot="shell_vs.png", auto_close=True, interactive=False)
```

### Ray Length Visualization
```python
plotter = model.mesh.plot_ray_lengths(
    arrival_or_xyz=ray,
    store_as='ray_lengths',
    screenshot='ray_plot.png'  # Optional: write image directly
)
# If you didn't pass screenshot=, you can still render manually:
# plotter.show(screenshot='ray_plot.png', auto_close=True, interactive=False)
```

## Common Issues and Solutions

### ❌ Problem: "Widget still appears despite static images"
**Solution**: Configure PyVista BEFORE importing SensRay:
```python
# Do this FIRST
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTK'] = 'false'
import pyvista as pv
pv.set_jupyter_backend('static')

# Then import SensRay
from sensray import PlanetModel
```

### ❌ Problem: "Plotting crashes on headless system"
**Solution**: Ensure proper environment variables:
```python
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['DISPLAY'] = ':99'  # Use virtual display
```

### ❌ Problem: "No images generated in static mode"
**Solution**: Use proper show() parameters:
```python
plotter.show(
    screenshot="output.png",  # Specify filename
    auto_close=True,          # Close after rendering
    interactive=False         # Disable interaction
)
```

### ❌ Problem: "Interactive mode not working"
**Solution**: Check your environment supports widgets and backends. Start by checking the INTERACTIVE flag and attempting to set backends in order of preference:

```python
import pyvista as pv
INTERACTIVE = True
try:
    pv.set_jupyter_backend('ipyvtklink')
    print('Set ipyvtklink')
except Exception:
    try:
        pv.set_jupyter_backend('panel')
        print('Set panel')
    except Exception as e:
        print('Interactive backends unavailable:', e)
        pv.set_jupyter_backend('static')
```

Note: `trame` is web/server-based and requires additional setup (server process) and is not a direct widget replacement in all notebook setups.

## Best Practices

### 1. Notebook Organization
```python
# Cell 1: Environment setup
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_IPYVTK'] = 'false'

# Cell 2: Import and configure
import pyvista as pv
pv.set_jupyter_backend('static')
from sensray import PlanetModel

# Cell 3+: Your analysis
model = PlanetModel.from_standard_model('prem')
# ... rest of your code
```

### 2. Consistent File Naming
```python
# Use descriptive names for static outputs
plotter.show(screenshot=f"cross_section_{property_name}.png")
plotter.show(screenshot=f"shell_r{radius_km}km_{property_name}.png")
plotter.show(screenshot=f"rays_{phase_name}.png")
```

### 3. Error Handling
```python
try:
    plotter = model.mesh.plot_cross_section(property_name='vp')
    plotter.show(screenshot="output.png", auto_close=True, interactive=False)
    print("✓ Plot saved successfully")
except Exception as e:
    print(f"❌ Plotting failed: {e}")
    # Fallback to data export
    model.mesh.save("data_backup.vtu")
```

## Environment-Specific Examples

### Docker Containers
```dockerfile
# Dockerfile
FROM python:3.11
RUN apt-get update && apt-get install -y \
    xvfb \
    libgl1-mesa-glx \
    libglib2.0-0
ENV DISPLAY=:99
```

```python
# Python code in container
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
from sensray import PlanetModel
# ... static plotting only
```

### HPC Clusters
```bash
#!/bin/bash
#SBATCH --job-name=sensray_viz
export PYVISTA_OFF_SCREEN=true
export DISPLAY=:99
module load python/3.11
python my_sensray_script.py
```

### GitHub Actions / CI
```yaml
# .github/workflows/test.yml
- name: Test plotting
  run: |
    export PYVISTA_OFF_SCREEN=true
    python -c "
    import os
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    from sensray import PlanetModel
    # Test static plotting
    "
```

## Summary

- Static mode: Configure PyVista for off-screen/static and use `show(screenshot=..., auto_close=True, interactive=False)`
- Interactive mode: Use a GUI-capable environment or `trame` in notebooks, then call `plotter.show()`
- Key rule: Configure PyVista BEFORE importing SensRay
- Jupyter: Set environment variables and backend in the first cell
- Scripts: Set environment variables before imports
- Always: Use proper `show()` parameters for static images

For more examples, see the demo notebooks in `demos/` directory.