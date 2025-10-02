# shell_tets_min.py
# pip install pygmsh gmsh meshio pyvista numpy

import pygmsh
import meshio
import pyvista as pv

R_IN  = 3480.0    # inner radius (km)
R_OUT = 6371.0    # outer radius (km)
H     = 1000.0     # target edge length (km): smaller -> finer

# --- generate tetra mesh of a spherical shell ---
with pygmsh.occ.Geometry() as geom:
    # global size target
    geom.characteristic_length_min = H
    geom.characteristic_length_max = H

    outer = geom.add_ball([0.0, 0.0, 0.0], R_OUT)
    inner = geom.add_ball([0.0, 0.0, 0.0], R_IN)

    shell = geom.boolean_difference(outer, inner)  # shell = outer \ inner
    mesh = geom.generate_mesh(dim=3)

# save to VTU (easy to load in PyVista / ParaView)
meshio.write("sphere_shell_tets.vtu", mesh)

# --- quick visualization: clip to see the interior ---
grid = pv.read("sphere_shell_tets.vtu")          # UnstructuredGrid
clip = grid.clip(normal=(1, 0, 0), origin=(0, 0, 0))

p = pv.Plotter()
# clipped volume with edges
p.add_mesh(clip, style='wireframe')
p.add_mesh(clip, style='surface')
p.add_scalar_bar(title="tet cells")
p.show()
