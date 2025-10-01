from sensray import PlanetModel

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for compatibility

my_model = PlanetModel.from_standard_model('prem')
print(my_model.get_property_at_depth('vp', 100))
my_model.plot_profiles()
plt.show()