import logging
from typing import Any, Iterable, List, Tuple

import numpy as np
from itertools import product
from sensray import PlanetModel, CoordinateConverter
from GFwdOpClass import GFwdOp
from GFwdOpClassLinOp import GFwdOp as GFwdOpLin
from SphericalFunc import make_scalar_field
from quadpy_integral import integrate_over_tetrahedra


# Load model and create mesh
model = PlanetModel.from_standard_model('prem')
# Create mesh and save if not exist, otherwise load existing
mesh_path = "prem_mesh"
try:
    model.create_mesh(from_file=mesh_path)
    print(f"Loaded existing mesh from {mesh_path}")
except FileNotFoundError:
    print("Creating new mesh...")
    radii = [1221.5, 3480.0, 6371]
    H_layers = [1000, 1000, 600]
    model.create_mesh(mesh_size_km=1000, radii=radii, H_layers=H_layers)
    model.mesh.populate_properties(['vp', 'vs', 'rho'])
    model.mesh.save("prem_mesh")  # Save mesh to VT
print(f"Created mesh: {model.mesh.mesh.n_cells} cells")

# function to make points
def point(pointType="Source", minLat=-90, maxLat=90, minLon=-180, maxLon=180, minDepth=0, maxDepth=700):
    if pointType == "Source":
        lat = np.random.uniform(minLat, maxLat)
        lon = np.random.uniform(minLon, maxLon)
        depth = np.random.uniform(minDepth, maxDepth)  # depth in km
        return (lat, lon, depth)
    elif pointType == "Receiver":
        lat = np.random.uniform(minLat, maxLat)
        lon = np.random.uniform(minLon, maxLon)
        return (lat, lon)
    else:
        raise ValueError("pointType must be 'Source' or 'Receiver'")

def get_rays(srp):
    srr_lst = []
    for (source, receiver, phases) in srp:
        rays = model.taupy_model.get_ray_paths_geo(
            source_depth_in_km=source[2],
            source_latitude_in_deg=source[0],
            source_longitude_in_deg=source[1],
            receiver_latitude_in_deg=receiver[0],
            receiver_longitude_in_deg=receiver[1],
            phase_list=phases,
        )
        for ray in rays:
            srr_list.append((source, receiver, ray))

    return np.array(srr_list, dtype=object)

# make function for velocity perturbation
# Define R(r) and T(theta, phi)
R = lambda r: r**2 * np.exp(-r/100000)              # simple radial function
T = lambda theta, phi: np.cos(theta)         # angular dependence

f = make_scalar_field(R, T)

# Apply the perturbation function to the mesh cell centers
model.mesh.mesh.cell_data["dv"] = f(model.mesh.mesh.cell_centers().points)
print(model.mesh.mesh.cell_data["dv"])


# Generate source and receiver points and combinations
# sources = [point("Source", minDepth=150, maxDepth=150) for _ in range(2)]
# receivers = [point("Receiver", maxDepth=0) for _ in range(5)]
# phases = ["P", "S", "ScS"]
# srp = [prod + tuple([phases]) for prod in product(sources, receivers)]

# testing with one source-receiver pair - same as initial test
source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
receiver_lat, receiver_lon = 30.0, 45.0  # Surface station
srp = [((source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), ["P", "S", "ScS"])]


srr = get_rays(srp)


def display_dv(source_lat, source_lon, receiver_lat, receiver_lon):
    plane_normal = CoordinateConverter.compute_gc_plane_normal(
        source_lat, source_lon, receiver_lat, receiver_lon
    )

    # Cross-section showing background Vp
    print("Background P-wave velocity:")
    plane_normal = plane_normal
    plotter1 = model.mesh.plot_cross_section(
        plane_normal=plane_normal,
        property_name='dv',
    )
    plotter1.camera.position = (8000, 6000, 10000)
    plotter1.show()

# display dv using first source-receiver pair
display_dv(srr[0,0][0], srr[0,0][1], srr[0,1][0], srr[0,1][1])

print("Calculate travel time kernels and residuals...")
appl = GFwdOp(model, srr[:,2])
travel_times = appl.__apply__(model.mesh.mesh.cell_data["dv"])
print(travel_times)

print("Calculate travel time kernels and residuals...")
appl = GFwdOpLin(model, srr[:,2])
travel_times = appl.__apply__(model.mesh.mesh.cell_data["dv"])
print(travel_times)


# integrate a function over a cell from the mesh
# Extract the cell as a new UnstructuredGrid
cell_id = 0  # Example cell ID
cell_grid = model.mesh.mesh.extract_cells(cell_id)
# Points of the cell (N_points, 3) array
pts = cell_grid.points
# Cell connectivity / indices: cell_grid.cells is a flat array encoding types & offsets
# For convenience you can convert to numpy or inspect celltypes
ctypes = cell_grid.celltypes
print("Cell types:", ctypes)
# Assuming we have only one cell here, get the indices of the points forming the cell
cell_indices = cell_grid.cells[1:]  # Skip the first element which is the number of points
print("Cell indices:", cell_indices)
cell_pts = pts[cell_indices]
print("Cell points:\n", cell_pts)


print("Integrating over cell...")
print(integrate_over_tetrahedra(f, cell_pts))
