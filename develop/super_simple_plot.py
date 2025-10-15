#!/usr/bin/env python3
"""Tiny PyVista demo: sphere + clip (≈15 lines).

Run: python3 develop/super_simple_plot.py
Requires: pip install pyvista
"""
try:
    import pyvista as pv
except Exception as e:
    print('PyVista required: pip install pyvista', e)
    raise SystemExit(1)

# Create sphere mesh
sphere = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64)

# Clip with plane (example: z=0 plane)
plane_normal = (0, 0, 1)
plane_origin = (0, 0, 0)
clipped = sphere.clip(normal=plane_normal, origin=plane_origin)

# Show both: transparent original and solid clipped
pl = pv.Plotter()
pl.add_mesh(sphere, color='lightgray', opacity=0.3)
pl.add_mesh(clipped, color='tomato')
pl.add_axes()
pl.show()
