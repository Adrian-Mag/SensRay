from obspy.taup.taup_create import build_taup_model
from obspy.taup import TauPyModel

# 1) Convert your velocity model text file (".nd" or ".tvel") to .npz once
build_taup_model("mymodel.nd", output_folder=".")

# 2) Use it like any built-in model
mod = TauPyModel(model="mymodel.npz")   # or just "mymodel" if cwd

mod.plot()

ray = mod.get_ray_paths_geo(source_depth_in_km=10,
                            source_latitude_in_deg=0,
                            source_longitude_in_deg=0,
                            receiver_latitude_in_deg=10,
                            receiver_longitude_in_deg=10,
                            phase_list=["P", "S", "PcP"])

print(ray[0].path)