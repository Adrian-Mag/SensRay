import numpy as np
from sensray import PlanetModel
from fwdop import GFwdOp, make_scalar_field
from pygeoinf.linear_solvers import LUSolver, CholeskySolver
# from ray_and_point_generation import get_rays, fibonacci_sphere_points
from itertools import product
from random import randint
from obspy.geodetics import locations2degrees

from pandas import read_csv
lander_locs = read_csv("LanderLocs.csv")
nunn_filtered = read_csv("Nunn_catalog_filtered.csv")  # Nunn picks with S-P calculations
nunn_picks = nunn_filtered[nunn_filtered["Phase"] != "S-P"]  # filter out S-P calculations
weber_core = read_csv("Weber Core - picks catalog.csv")  # Weber picks

# Load model and create mesh
model_name = "M1"
mesh_size_km = 1000

# Create mesh and save if not exist, otherwise load existing
mesh_path = "M1_mesh"

# Load model and create mesh
model = PlanetModel.from_standard_model('M1')


try:
    model.create_mesh(from_file=mesh_path)
    print(f"Loaded existing mesh from {mesh_path}")
except FileNotFoundError:
    print("Creating new mesh...")
    radii = model.get_discontinuities()
    H_layers = [1000, 600]
    model.create_mesh(mesh_size_km=mesh_size_km, radii=radii, H_layers=H_layers)
    model.mesh.populate_properties(model.get_info()["properties"])
    model.mesh.save(f"{model_name}_mesh")  # Save mesh to VT


def compute_travel_time(row):
    # Extract event and lander coordinates
    lat, lon = lander_locs[
        (lander_locs["Lander"] == row["Station"]) & 
        (lander_locs["Element"] == "ALSEP")].iloc[0][["Latitude", "Longitude"]]
    
    deg = locations2degrees(
        float(row["Lat"]), float(row["Lon"]),
        float(lat), float(lon)
    )

    arrivals = model.taupy_model.get_travel_times(source_depth_in_km=row["Depth"],
                                        distance_in_degree=deg,
                                        phase_list=[row["Phase"]])
    return arrivals[0].time if arrivals else np.nan


def compute_ray(row):
    # Extract event and lander coordinates
    lat, lon = lander_locs[
    (lander_locs["Lander"] == row["Station"]) & 
    (lander_locs["Element"] == "ALSEP")].iloc[0][["Latitude", "Longitude"]]
    
    rays = model.taupy_model.get_ray_paths_geo(
                    source_depth_in_km=row["Depth"],
                    source_latitude_in_deg=row["Lat"],
                    source_longitude_in_deg=row["Lon"],
                    receiver_latitude_in_deg=lat,
                    receiver_longitude_in_deg=lon,
                    phase_list=row["Phase"],
                )
    return rays[0] if rays else np.nan


# T_obs = nunn_picks["mean arrival time"]
# T_0 = nunn_picks["predicted arrival time"]
result = nunn_picks.apply(compute_travel_time, axis=1)
nunn_picks["predicted arrival time"] = result
# dT_obs = nunn_picks["dT_obs"]
nunn_picks["dT_obs"] = nunn_picks["mean arrival time"] - nunn_picks["predicted arrival time"]


nunn_picks_rays = nunn_picks[nunn_picks["predicted arrival time"].notna()]
nunn_picks_rays["ray"] = nunn_picks_rays.apply(compute_ray, axis=1)
G = GFwdOp(model=model, rays=nunn_picks_rays["ray"])

# Generate different models and calculate dv
functions = {
    "simple": {"R": lambda r: np.ones_like(r), "T": lambda theta, phi: np.ones_like(theta)},
    "complex": {"R": lambda r: r**2 * np.exp(-r/100000), "T": lambda theta, phi: np.cos(theta)},
    "harmonic": {"R": lambda r: 0.1 * model.get_property_at_radius(radius=r, property_name="vp"), "T": lambda theta, phi: 0.5 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)},
}

func = "harmonic"
f = make_scalar_field(functions[func]["R"], functions[func]["T"])

model.mesh.project_function_on_mesh(f, property_name="dm")
print("Cell data 'dm':", model.mesh.mesh.cell_data["dm"])

dT = G(model.mesh.mesh.cell_data["dm"])

# find dm so dT_obs - dT ~ 0
print(nunn_picks_rays["dT_obs"] - dT)
