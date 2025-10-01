from sensray import PlanetModel, RayPathTracer

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for compatibility

my_model = PlanetModel.from_standard_model('prem')
""" print(my_model.get_property_at_depth('vp', 100))
my_model.plot_profiles()
plt.show() """

rays = my_model.taupy_model.get_ray_paths_geo(source_depth_in_km=100, source_latitude_in_deg=0,
                                       source_longitude_in_deg=0,
                                       receiver_latitude_in_deg=30,
                                       receiver_longitude_in_deg=30,
                                       phase_list=['P', 'S', 'PP', 'SS', 'PcP'])

print(rays[0].path)