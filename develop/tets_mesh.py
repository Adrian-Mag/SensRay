# sphere_tets_min.py
# pip install pygmsh gmsh meshio pyvista numpy

import numpy as np
import pygmsh, meshio, pyvista as pv

R = 6371.0      # sphere radius (km)
H = 1000.0       # target edge length (~controls tet size). Smaller -> finer.

# --- build tets with Gmsh via pygmsh (super simple) ---
with pygmsh.occ.Geometry() as geom:
    # global target sizes
    geom.characteristic_length_min = H
    geom.characteristic_length_max = H

    geom.add_ball([0.0, 0.0, 0.0], R)     # one solid sphere
    m = geom.generate_mesh(dim=3)         # tetra volume mesh

# write once; .vtu is easy to read in PyVista/ParaView
meshio.write("sphere_tets.vtu", m)

# --- quick peek: clip the volume to see tets inside ---
grid = pv.read("sphere_tets.vtu")               # UnstructuredGrid
clip = grid.clip(normal=(1, 0, 0), origin=(0, 0, 0))

p = pv.Plotter()
# show clipped tets with edges
p.add_mesh(clip, show_edges=False, style='wireframe')
p.add_mesh(clip, style='surface', opacity=0.5)
p.add_scalar_bar(title="cell ids")
p.show()
