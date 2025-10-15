import numpy as np
from random import seed
from itertools import product
from sensray import PlanetModel
from GFwdOpClass import GFwdOp

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

def pointVel(*args):
    # Allow either 3 separate values or a single tuple/list of length 3
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        x, y, z = args[0]
    elif len(args) == 3:
        x, y, z = args
    else:
        raise ValueError("pointVel() expects either 3 values or a tuple/list of length 3")
    
    #x, y, z = unpackVals(args)
    
    return (x**2 + y**2 + z**2)**0.5 + 4.5 + 0.0003 * z



# print(model.mesh.mesh.cell_centers().points)
# print(model.mesh.mesh.cell_data["vp"])
# apply velocity model to all points in the cell centres of the initial mesh
model.mesh.mesh.cell_data["dv"] = np.apply_along_axis(lambda r: pointVel(*r), axis=1, arr=model.mesh.mesh.cell_centers().points)
print(model.mesh.mesh.cell_data["dv"])

def get_rays(srp):
    srr_lst = []
    for (source, receiver, phases) in srp:
        rays = model.taupy_model.get_ray_paths_geo(
    source_depth_in_km=source[2],
    source_latitude_in_deg=source[0],
    source_longitude_in_deg=source[1],
    receiver_latitude_in_deg=receiver[0],
    receiver_longitude_in_deg=receiver[1],
    phase_list=phases
)
        for ray in rays:
            srr_lst.append((source, receiver, ray))
    return np.array(srr_lst, dtype=object)


# Generate source and receiver points and combinations
sources = [point("Source", minDepth=150, maxDepth=150) for _ in range(2)]
receivers = [point("Receiver", maxDepth=0) for _ in range(5)]
phases = ["P", "S", "ScS"]

# For G_FwdOp
srp = [prod + tuple([phases]) for prod in product(sources, receivers)]

# # testing with one source-receiver pair - same as initial test
# source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
# receiver_lat, receiver_lon = 30.0, 45.0  # Surface station
# srp = [((source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), ["P", "S", "ScS"])]
srr = get_rays(srp)

# Cross-section showing background Vp
print("Background P-wave velocity:")
plane_normal = np.array([1.0, 0.0, 1.0])
plotter1 = model.mesh.plot_cross_section(
    plane_normal=plane_normal,
    property_name='dv',
)
plotter1.camera.position = (8000, 6000, 10000)
plotter1.show()

print("Calculate travel time kernels and residuals...")
appl = GFwdOp(model, srr[:,2])
travel_times = appl.__apply__(model.mesh.mesh.cell_data["dv"])
print(travel_times)
