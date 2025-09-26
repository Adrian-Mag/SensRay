# file: examples/point_on_slice.py
import numpy as np
import pyvista as pv

from sensray.mesh.earth_model import MeshEarthModel
from sensray.utils.coordinates import CoordinateConverter
from sensray.core.ray_paths import RayPathTracer

# ---- create model (coarse for speed) ----
model = MeshEarthModel.from_pygmsh_sphere(radius_km=6371.0, mesh_size_km=500.0)

# source/receiver used to define the slice plane
source_lat, source_lon = 10.0, 20.0
receiver_lat, receiver_lon = 30.0, 40.0

# get the slice surface (PyVista mesh)
slice_surface = model.slice_great_circle(
    source_lat, source_lon, receiver_lat, receiver_lon, earth_radius_km=6371.0
)

###############################
# Compute a ray path and plot its points on the slice
###############################

# Trace a geographic ray path between source and receiver
tracer = RayPathTracer(model_name="iasp91")
ray_paths, _ = tracer.get_ray_paths(
    source_lat=source_lat,
    source_lon=source_lon,
    source_depth=15.0,
    receiver_lat=receiver_lat,
    receiver_lon=receiver_lon,
    phases=["P"],
    output_geographic=True,
)

if not ray_paths:
    raise SystemExit("No ray paths found for the chosen configuration.")

# Take first arrival and extract lat/lon/depth arrays
rp = ray_paths[0]
path = rp.path
lats = path["lat"]
lons = path["lon"]
depths = path["depth"]

# Convert all ray sample points to Cartesian
ray_pts = np.array([
    CoordinateConverter.earth_to_cartesian(lat, lon, depth)
    for lat, lon, depth in zip(lats, lons, depths)
])

# compute plane normal using the same great-circle definition
src_xyz = np.array(
    CoordinateConverter.earth_to_cartesian(source_lat, source_lon, 0.0)
)
rec_xyz = np.array(
    CoordinateConverter.earth_to_cartesian(receiver_lat, receiver_lon, 0.0)
)

# ---- visualize ----
pv.set_plot_theme("document")
p = pv.Plotter()
p.add_mesh(
    slice_surface,
    color="lightgray",
    show_edges=False,
    opacity=1.0,
    style="wireframe",
)

# Plot projected ray points
p.add_points(ray_pts, color="red", point_size=8)

# Optional: mark source/receiver on the surface for context
p.add_mesh(
    pv.Sphere(radius=100.0, center=np.array(
        CoordinateConverter.earth_to_cartesian(source_lat, source_lon, 0.0)
    )),
    color="blue",
    label="Source",
)
p.add_mesh(
    pv.Sphere(radius=100.0, center=np.array(
        CoordinateConverter.earth_to_cartesian(receiver_lat, receiver_lon, 0.0)
    )),
    color="green",
    label="Receiver",
)

p.add_text(
    "Ray: red  |  Source: blue  |  Receiver: green",
    position="upper_left",
    font_size=10,
)
p.show()
