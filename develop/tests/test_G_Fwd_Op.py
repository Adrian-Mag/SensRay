import numpy as np
from random import seed
from itertools import product
from sensray import PlanetModel
from G_Fwd_Op import GFwdOp

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
    
def pointVel(row): 
    x, y, z = row
    # Simple smooth velocity model: velocity increases with depth
    return (x**2 + y**2 + z**2)**0.5 + 4.5 + 0.0003 * z
    # return 4.5 + 0.0003 * z  # Example: Vp in km/s

def renderField(points=None, scalars=None):
    # Create point cloud
    cloud = pv.PolyData(points)
    cloud['velocity'] = scalars  # attach scalar data to points

    # Plot
    p = pv.Plotter()
    p.add_mesh(
        cloud, 
        scalars='velocity', 
        cmap='viridis', 
        point_size=10, 
        render_points_as_spheres=True
    )
    p.add_scalar_bar(title='Velocity')
    # View the slice from the top (normal to z)
    p.view_xy()  # if slicing normal='z'
    p.show_axes()  # optional for orientation
    p.show(screenshot="modelled_vp.png")

def renderFieldMatplotlib(points=None, scalars=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=scalars, cmap='viridis')

    plt.colorbar(sc, label='Velocity')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.show()

def renderFieldSliceMatplotlib(points=None, scalars=None, slice_axis='z', slice_value=0, slice_thickness=1):
    import numpy as np
    import matplotlib.pyplot as plt    
    
    slice_mask = np.abs(points[:,2] - slice_value) < slice_thickness

    x_slice = points[:,0][slice_mask]
    y_slice = points[:,1][slice_mask]
    speed_slice = scalars[slice_mask]

    # ------------------------------------------------------
    # 4. Plot the scalar velocity field and vectors
    # ------------------------------------------------------
    plt.figure(figsize=(7,6))
    sc = plt.scatter(x_slice, y_slice, c=speed_slice, cmap='viridis', s=20)
    plt.colorbar(sc, label='Velocity magnitude')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Slice through mesh at z = {slice_value:.2f}')
    plt.axis('equal')
    plt.show()

def renderFieldSliceProjectedMatplotlib(points=None, scalars=None, projection_axis='z'):
    import numpy as np
    import matplotlib.pyplot as plt
    # ------------------------------------------------------
    # 2. Define arbitrary slicing plane
    # ------------------------------------------------------
    p0 = np.array([0.0, 0.0, 0.0])             # point on plane
    normal = np.array([1.0, 1.0, 1.0])         # normal vector
    normal = normal / np.linalg.norm(normal)

    # ------------------------------------------------------
    # 3. Project points onto plane
    # ------------------------------------------------------
    p = points
    dist = (p - p0) @ normal                   # distance along normal
    p_proj = p - np.outer(dist, normal)        # projected points

    # ------------------------------------------------------
    # 4. Build local basis on the plane (t1, t2)
    # ------------------------------------------------------
    ref = np.array([0,0,1]) if abs(normal[2]) < 0.9 else np.array([1,0,0])
    t1 = np.cross(normal, ref)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    # Project positions onto plane coordinates
    proj_rel = p_proj - p0
    X2 = proj_rel @ t1
    Y2 = proj_rel @ t2

    # ------------------------------------------------------
    # 6. Plot projected scalar velocity field
    # ------------------------------------------------------
    plt.figure(figsize=(7,6))
    sc = plt.scatter(X2, Y2, c=scalars, cmap='viridis', s=10)
    plt.colorbar(sc, label='Velocity magnitude')

    plt.xlabel('in-plane X')
    plt.ylabel('in-plane Y')
    plt.title(f'Projection of scalar velocity onto plane\nnormal={normal}')
    plt.axis('equal')
    plt.show()




print(model.mesh.mesh)
print(model.mesh.mesh.points)
# apply velocity model to all points in the initial mesh
velocity = np.apply_along_axis(pointVel, axis=1, arr=model.mesh.mesh.points)
print(velocity.shape)
# model.mesh.mesh["dv"] = velocity
# p.add_mesh(model.mesh.mesh, scalars='velocity_magnitude', cmap='plasma')
renderFieldSliceProjectedMatplotlib(model.mesh.mesh.points, velocity)

'''
# Generate source and receiver points and combinations
sources = [point("Source", minDepth=150, maxDepth=150) for _ in range(2)]
receivers = [point("Receiver", maxDepth=0) for _ in range(5)]
phases = ["P", "S", "ScS"]

# For G_FwdOp
# srp = [prod + tuple([phases]) for prod in product(sources, receivers)]
# For G_FwdOp_SinglePhase
srp_onephase = product(sources, receivers, phases)

# # testing with one source-receiver pair - same as initial test
# source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
# receiver_lat, receiver_lon = 30.0, 45.0  # Surface station
# srp = [((source_lat, source_lon, source_depth), (receiver_lat, receiver_lon), ["P", "S", "ScS"])]

# appl = G(model, srp)
appl = GFwdOp(model, srp_onephase)
travel_times = appl.__apply__(model)
print(travel_times)
'''