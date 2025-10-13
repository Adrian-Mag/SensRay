import numpy as np
from random import seed
from itertools import product
from sensray import PlanetModel
from G_FwdOp import G

seed(0)

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

# Generate source and receiver points and combinations
sources = [point("Source", minDepth=150, maxDepth=150) for _ in range(2)]
receivers = [point("Receiver", maxDepth=0) for _ in range(5)]
phases = ["P", "S", "ScS"]

srp = [prod + tuple([phases]) for prod in product(sources, receivers)]

# testing with one source-receiver pair - same as initial test
source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
receiver_lat, receiver_lon = 30.0, 45.0  # Surface station
srp = [((source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), ["P", "S", "ScS"])]

appl = G(model, srp)
# get voxel count
voxel_count = appl.get_voxelNum()
# grid dims (nx, ny, nz), xyz extent, use linspace perhaps?
# velocity from surf V and sph eqs possibly?


travel_times = appl.__apply__(model)
print(travel_times)