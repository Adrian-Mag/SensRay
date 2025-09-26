"""
Minimal example: compute ray path between two geographic points using SensRay.
Run: python sensray/mesh/test2.py (from repo root)

Requires: sensray package in PYTHONPATH and obsPy installed.
"""

from sensray.core.ray_paths import RayPathTracer
from sensray.visualization.earth_plots import EarthPlotter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use a non-interactive backend for environments without display
import pygmsh, meshio, pyvista as pv

def main():
    model_name = "iasp91"
    tracer = RayPathTracer(model_name=model_name)

    # Define source and receiver geographic coordinates and source depth (km)
    source_lat, source_lon, source_depth = 10.0, 20.0, 15.0   # degrees, km
    receiver_lat, receiver_lon = 30.0, 40.0                   # degrees

    phases = ["P"]

    ray_paths, info = tracer.get_ray_paths(
        source_lat=source_lat,
        source_lon=source_lon,
        source_depth=source_depth,
        receiver_lat=receiver_lat,
        receiver_lon=receiver_lon,
        phases=phases,
        output_geographic=True,
    )

    print("Computation info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if not ray_paths:
        print("No ray paths found for requested phases.")
        return

    coords = tracer.extract_ray_coordinates(ray_paths)

    # Optional plotting (2D cross-section)
    try:
        plotter = EarthPlotter(model_name=model_name)
        distance_deg_plot = float(info.get("distance_deg") or 0.0)
        plotter.plot_circular_earth(
            ray_coordinates=coords,
            source_depth=source_depth,
            distance_deg=distance_deg_plot,
            fig_size=(8, 6),
            view="full",
            show_atmosphere=False,
        )
        plt.show()
    except Exception as e:
        print("Plotting failed or display not available:", e)

    grid = pv.read("sphere.vtu")

    # Slice with a plane through the origin, normal along +X (x=0 plane)
    sl = grid.slice(normal=(1, 0, 0), origin=(0, 0, 0))

    # Show just the slice (triangles along the cut)
    sl.plot(style='wireframe', color='lightblue', line_width=1)


if __name__ == "__main__":
    main()
