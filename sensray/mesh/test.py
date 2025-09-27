import numpy as np
import matplotlib
from pathlib import Path
from sensray.core.ray_paths import RayPathTracer
from sensray.mesh.earth_model import MeshEarthModel
from sensray.utils.coordinates import CoordinateConverter

# If you later need pyplot, set backend before importing it:
matplotlib.use('TkAgg')

model = 'prem'
tracer = RayPathTracer(model_name=model)

source_lat, source_lon = 10.0, 20.0
source_depth = 30.0
receiver_lat, receiver_lon = -20.0, 100.0

phases = ['PKKP']

rays, info = tracer.get_ray_paths(
    source_lat, source_lon, source_depth,
    receiver_lat, receiver_lon, phases=phases
)
ray_coordinates = tracer.extract_ray_coordinates(rays)

""" plotter = EarthPlotter(model_name=model)

plotter.plot_circular_earth(
    ray_coordinates,
    source_depth=source_depth,
    distance_deg=info['distance_deg']
)
plt.show() """

#######################################################
# Create 3D earth mesh
#######################################################
mesh_size_km = 300
mesh_file_name = 'demo-sphere_' + str(mesh_size_km) + '.vtk'
if Path(mesh_file_name).exists():
    earth_mesh_model = MeshEarthModel.from_file(mesh_file_name)
else:
    earth_mesh_model = MeshEarthModel.from_pygmsh_sphere(
        mesh_size_km=mesh_size_km,
        name='demo-sphere'
    )
    earth_mesh_model.save('demo-sphere.vtk')

three_d_plot = earth_mesh_model.plot_slice(
    source_lat=source_lat,
    source_lon=source_lon,
    receiver_lat=receiver_lat,
    receiver_lon=receiver_lon,
    scalar_name=None,
    show_edges=False,
    wireframe=True
)


# Convert ray path (lat, lon, depth) to Cartesian Nx3 (km)
x, y, z = CoordinateConverter.earth_to_cartesian(
    rays[0].path['lat'],
    rays[0].path['lon'],
    rays[0].path['depth']
)
points_xyz = np.column_stack([x, y, z]).astype(float)

earth_mesh_model.add_points(
    three_d_plot,
    points_xyz=points_xyz,
    color='red',
    point_size=10,
)

three_d_plot.show()

# --- Simple per-cell path-length example ---------------------------------
# Compute path length inside each tetrahedral cell and attach as 'ray_length'
lengths = earth_mesh_model.compute_ray_cell_path_lengths(
    points_xyz, attach_name="ray_length"
)

# Quick sanity: total length vs per-cell sum
seg = points_xyz[1:] - points_xyz[:-1]
total_len = float(np.linalg.norm(seg, axis=1).sum())
print(
    f"Ray length: {total_len:.2f} km | "
    f"Sum per-cell: {lengths.sum():.2f} km | "
    f"Cells traversed: {(lengths > 0).sum()}"
)

# Optional: visualize the slice colored by per-cell path length
p_len = earth_mesh_model.plot_slice(
    source_lat=source_lat,
    source_lon=source_lon,
    receiver_lat=receiver_lat,
    receiver_lon=receiver_lon,
    scalar_name="ray_length",
    cmap="magma",
    wireframe=True,
)
p_len.add_text(
    "Slice colored by per-cell ray length",
    position="upper_left",
    font_size=10,
)
p_len.show()

# --- Sensitivity kernels (vp, vs) using PREM mapping ---------------------
# Map 1D PREM properties to mesh cells
earth_mesh_model.add_scalars_from_1d_model(
    model_name=model,
    properties=("vp", "vs"),
    where="cell",
    method="center",
)

# Compute kernels for the current ray
K_vp = earth_mesh_model.compute_sensitivity_kernel(
    points_xyz, "vp", attach_name="K_vp", epsilon=0.0, tol=1e-6,
    model_name=None,
)
K_vs = earth_mesh_model.compute_sensitivity_kernel(
    points_xyz, "vs", attach_name="K_vs", epsilon=0.0, tol=1e-6,
    model_name=None,
)

print(
    f"K_vp nonzero cells: {(K_vp != 0).sum()} | "
    f"K_vs nonzero cells: {(K_vs != 0).sum()}"
)

# Visualize kernels on the great-circle slice
p_kvp = earth_mesh_model.plot_slice(
    source_lat=source_lat,
    source_lon=source_lon,
    receiver_lat=receiver_lat,
    receiver_lon=receiver_lon,
    scalar_name="K_vp",
    cmap="seismic",
    wireframe=True,
)
p_kvp.add_text("Slice colored by K_vp", position="upper_left", font_size=10)
p_kvp.show()

p_kvs = earth_mesh_model.plot_slice(
    source_lat=source_lat,
    source_lon=source_lon,
    receiver_lat=receiver_lat,
    receiver_lon=receiver_lon,
    scalar_name="K_vs",
    cmap="seismic",
    wireframe=True,
)
p_kvs.add_text("Slice colored by K_vs", position="upper_left", font_size=10)
p_kvs.show()
