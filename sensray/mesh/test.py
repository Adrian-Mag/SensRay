from sensray.mesh.earth_model import MeshEarthModel


model = MeshEarthModel.from_pygmsh_sphere(radius_km=6371.0, mesh_size_km=1000.0)

model.slice_great_circle(10, 20, 30, 40).plot(style="wireframe")