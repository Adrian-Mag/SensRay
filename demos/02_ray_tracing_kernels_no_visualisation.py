import numpy as np
from sensray import PlanetModel, CoordinateConverter

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

# Define source (earthquake) and receiver (seismic station) locations
source_lat, source_lon, source_depth = 0.0, 0.0, 150.0  # Equator, 150 km depth
receiver_lat, receiver_lon = 30.0, 45.0  # Surface station

# Compute great-circle plane normal for cross-sections
plane_normal = CoordinateConverter.compute_gc_plane_normal(
    source_lat, source_lon, receiver_lat, receiver_lon
)
print(f"Source: ({source_lat}°, {source_lon}°, {source_depth} km)")
print(f"Receiver: ({receiver_lat}°, {receiver_lon}°, 0 km)")
print(f"Great-circle plane normal: {plane_normal}")

# Get ray paths for P and S waves
rays = model.taupy_model.get_ray_paths_geo(
    source_depth_in_km=source_depth,
    source_latitude_in_deg=source_lat,
    source_longitude_in_deg=source_lon,
    receiver_latitude_in_deg=receiver_lat,
    receiver_longitude_in_deg=receiver_lon,
    phase_list=["P", "S", "ScS"]
)

# rays.plot_rays()

print(f"Found {len(rays)} ray paths:")
for i, ray in enumerate(rays):
    print(f"  {i+1}. {ray.phase.name}: {ray.time:.2f} s, {len(ray.path)} points")

# Compute and store path lengths for each ray
P_ray = rays[0]  # First ray (P wave)
S_ray = rays[1] if len(rays) > 1 else rays[0]  # Second ray (S wave)
ScS_ray = rays[2]

# Method 1: Simple computation and storage
P_lengths = model.mesh.add_ray_to_mesh(P_ray, "P_wave")
S_lengths = model.mesh.add_ray_to_mesh(S_ray, "S_wave")
ScS_lengths = model.mesh.add_ray_to_mesh(ScS_ray, "ScS_wave")

print(f"P wave: {P_lengths.sum():.1f} km total, {np.count_nonzero(P_lengths)} cells")
print(f"S wave: {S_lengths.sum():.1f} km total, {np.count_nonzero(S_lengths)} cells")
print(f"ScS wave: {ScS_lengths.sum():.1f} km total, {np.count_nonzero(ScS_lengths)} cells")

# Show stored properties
ray_keys = [k for k in model.mesh.mesh.cell_data.keys() if 'ray_' in k]
print(f"Stored ray properties: {ray_keys}")

# Compute sensitivity kernels for P and S waves
P_kernel = model.mesh.compute_sensitivity_kernel(
    P_ray, property_name='vp', attach_name='K_P_vp', epsilon=1e-6
)
S_kernel = model.mesh.compute_sensitivity_kernel(
    S_ray, property_name='vs', attach_name='K_S_vs', epsilon=1e-6
)
ScS_kernel = model.mesh.compute_sensitivity_kernel(
    ScS_ray, property_name='vs', attach_name='K_ScS_vs', epsilon=1e-6
)
print(f"P kernel range: {P_kernel.min():.6f} to {P_kernel.max():.6f} s²/km³")
print(f"S kernel range: {S_kernel.min():.6f} to {S_kernel.max():.6f} s²/km³")
print(f"ScS kernel range: {ScS_kernel.min():.6f} to {ScS_kernel.max():.6f} s²/km³")
print(f"Non-zero P kernel cells: {np.count_nonzero(P_kernel)}")
print(f"Non-zero S kernel cells: {np.count_nonzero(S_kernel)}")
print(f"Non-zero ScS kernel cells: {np.count_nonzero(ScS_kernel)}")

# Sum kernels from multiple rays
if len(rays) >= 2:
    combined_kernel = model.mesh.compute_sensitivity_kernels_for_rays(
        rays[1:],  # Use first two rays
        property_name='vs',
        attach_name='K_combined_vs',
        accumulate='sum'
    )
    print(f"Combined kernel range: {combined_kernel.min():.6f} to {combined_kernel.max():.6f}")
    print(f"Combined kernel non-zero cells: {np.count_nonzero(combined_kernel)}")

# Save mesh with rays and kernels
model.mesh.save('prem_mesh_with_rays_kernels')

# Show what was saved
info = model.mesh.list_properties(show_stats=False)
print(f"Saved {len(info['cell_data'])} properties to VTU file:")
for prop in info['cell_data'].keys():
    print(f"  - {prop}")

print("\nFiles created:")
print("  - prem_mesh_with_rays_kernels.vtu (mesh + all data)")
print("  - prem_mesh_with_rays_kernels_metadata.json (property list)")

# calculate travel times
print(P_kernel.dot(model.mesh.mesh.cell_data['vp']))