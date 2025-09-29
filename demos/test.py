from sensray import MeshEarthModel, RayPathTracer


earth_mesh_model = MeshEarthModel.from_pygmsh_sphere(mesh_size_km=500)


source_lat, source_lon, source_depth = 0, 0,  20.0
receiver_lat, receiver_lon = 0, 170.0

model_name = 'iasp91'

earth_mesh_model.add_scalar_from_1d_model(
    model_name=model_name,
    property_name='vs'
)
earth_mesh_model.add_scalar_from_1d_model(
    model_name=model_name,
    property_name='vp'
)

tracer = RayPathTracer(model_name=model_name)

phases = ['PKP']

rays, info = tracer.get_ray_paths(
    source_lat=source_lat,
    source_lon=source_lon,
    source_depth=source_depth,
    receiver_lat=receiver_lat,
    receiver_lon=receiver_lon,
    phases=phases
)

print(rays)

ray_coords = tracer.extract_ray_coordinates(rays)
print(ray_coords.keys())  # Dict with keys for each phase
chosen_key = 'PKP'

earth_mesh_model.compute_sensitivity_kernel(
    ray_points_xyz=ray_coords[chosen_key],
    model_name=model_name,
    property_name='vp',
    attach_name='K_vp',
    epsilon=1e-6
)

p = earth_mesh_model.plot_slice(
    source_lat=source_lat,
    source_lon=source_lon,
    receiver_lat=receiver_lat,
    receiver_lon=receiver_lon,
    scalar_name='K_vp',
    cmap='plasma',
)

earth_mesh_model.add_polyline(
    p,
    ray_coords[chosen_key],
    color='yellow',
    line_width=4.0
)
p.show()