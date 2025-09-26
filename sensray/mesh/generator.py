# generator.py

import pygmsh
import pyvista as pv
import meshio
import tempfile, os


def spherical_tetra_mesh(r_planet=6371.0, mesh_size=100.0):
    """
    Generate a tetrahedral mesh of a sphere using pygmsh.
    Returns a PyVista UnstructuredGrid without any cell data.
    """
    with pygmsh.geo.Geometry() as geom:
        # pass mesh_size explicitly to the geometry primitive
        ball = geom.add_ball([0, 0, 0], r_planet, mesh_size=mesh_size)
        # generate the 3D tetrahedral mesh
        gmsh_mesh = geom.generate_mesh(dim=3)

    # convert gmsh -> pyvista
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as tmp:
        meshio.write(tmp.name, gmsh_mesh)
        pv_mesh = pv.read(tmp.name)
        os.remove(tmp.name)

    return pv_mesh


def spherical_shell_mesh(r_planet=6371.0, dr=100.0, dtheta=10.0, dphi=10.0):
    """
    Generate a structured spherical shell mesh (approximate hexahedral/wedge cells).
    Returns a PyVista UnstructuredGrid without any cell data.
    """
    import numpy as np

    r_edges = np.arange(0, r_planet+dr, dr)
    theta_edges = np.radians(np.arange(0, 180+dtheta, dtheta))   # colatitude
    phi_edges   = np.radians(np.arange(0, 360+dphi, dphi))       # longitude

    points = []
    for r in r_edges:
        for th in theta_edges:
            for ph in phi_edges:
                x = r * np.sin(th) * np.cos(ph)
                y = r * np.sin(th) * np.sin(ph)
                z = r * np.cos(th)
                points.append([x,y,z])
    points = np.array(points)

    # At this stage: points exist, but connectivity for hex cells needs to be constructed.
    # For now, return just the cloud
    cloud = pv.PolyData(points)
    return cloud
