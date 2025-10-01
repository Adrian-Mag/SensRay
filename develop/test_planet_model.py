from sensray import PlanetModel, PlanetMesh, CoordinateConverter

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for compatibility


my_model = PlanetModel.from_standard_model('prem')

#my_model.create_mesh(from_file='prem_tetrahedral_mesh')
my_model.create_mesh(mesh_type='tetrahedral', mesh_size_km=500.0, populate_properties=['vp', 'vs', 'rho'])

# my_model.mesh.save('prem_tetrahedral_mesh')

source_lat, source_lon, source_depth_km = 0.0, 0.0, 100.0
receiver_lat, receiver_lon = 40.0, 80.0

# compute plane normal for the great-circle through source and receiver
plane_normal = CoordinateConverter.compute_gc_plane_normal(
    source_lat, source_lon, receiver_lat, receiver_lon
)

p = my_model.mesh.plot_cross_section(plane_normal=plane_normal)

p.show()
