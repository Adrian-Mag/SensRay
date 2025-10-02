from sensray import PlanetModel, PlanetMesh, CoordinateConverter

my_model = PlanetModel.from_standard_model('prem')
print(my_model.get_discontinuities())

# Create mesh and save if not exist, otherwise load existing
mesh_path = "prem_mesh.vtu"
try:
    my_model.create_mesh(from_file=mesh_path)
    print(f"Loaded existing mesh from {mesh_path}")
except FileNotFoundError:
    print("Creating new mesh...")
    radii = [1221.5, 3480.0, 6371]
    H_layers = [500, 500, 300]
    my_model.create_mesh(mesh_size_km=1000, radii=radii, H_layers=H_layers)
    my_model.mesh.populate_properties(['vp', 'vs', 'rho'])
    my_model.mesh.save("prem_mesh.vtu")  # Save mesh to VT

source_lat, source_lon, source_depth = 0.0, 0.0, 100.0
receiver_lat, receiver_lon = 30.0, 40.0,

plane_normal = CoordinateConverter.compute_gc_plane_normal(
    source_lat,
    source_lon,
    receiver_lat,
    receiver_lon,
    radius_km=my_model.radius
)

my_model.mesh.plot_cross_section(plane_normal, property_name="vp").show()