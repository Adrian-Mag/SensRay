from sensray import PlanetModel, CoordinateConverter
from pathlib import Path
from G_FwdOp import G

model = PlanetModel.from_standard_model('prem')

print(model.get_discontinuities())

mesh_path = 'test'
try:
    model.create_mesh(from_file=mesh_path)
    print(f"Loaded existing mesh from {mesh_path}")
except FileNotFoundError:
    print("Creating new mesh...")
    radii = [1221.5, 3480.0, 6371.0]
    model.create_mesh(mesh_size_km=500, radii=radii)
    model.mesh.populate_properties(['vp', 'vs'])
    model.mesh.save(mesh_path)  # Save mesh to VT


source_lat, source_lon, source_depth = 0.0, 0.0, 100.0  # degrees, degrees, km
receiver_lat, receiver_lon = 30.0, 30.0  # degrees, degrees
phases = ['P', 'S', 'PP', 'SS', 'PcP', 'ScS']

""" plane_normal = CoordinateConverter.compute_gc_plane_normal(
    source_lat, source_lon, receiver_lat, receiver_lon
)

p = model.mesh.plot_cross_section(plane_normal)
p.show() """

G_op = G(
    model,
    [
        (
            (source_lat, source_lon, source_depth),
            (receiver_lat, receiver_lon),
            phases
        )
    ]
)