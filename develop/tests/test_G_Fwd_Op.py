import logging
from typing import Any, Iterable, List, Tuple

import numpy as np
from itertools import product
from random import seed

from sensray import PlanetModel
from GFwdOpClass import GFwdOp

LOGGER = logging.getLogger(__name__)


def make_point(
    point_type: str = "Source",
    min_lat: float = -90,
    max_lat: float = 90,
    min_lon: float = -180,
    max_lon: float = 180,
    min_depth: float = 0,
    max_depth: float = 700,
) -> Tuple[float, ...]:
    """Generate a random source or receiver point.

    Returns a (lat, lon, depth) tuple for sources and (lat, lon) for receivers.
    """
    if point_type == "Source":
        lat = float(np.random.uniform(min_lat, max_lat))
        lon = float(np.random.uniform(min_lon, max_lon))
        depth = float(np.random.uniform(min_depth, max_depth))
        return lat, lon, depth

    if point_type == "Receiver":
        lat = float(np.random.uniform(min_lat, max_lat))
        lon = float(np.random.uniform(min_lon, max_lon))
        return lat, lon

    raise ValueError("point_type must be 'Source' or 'Receiver'")


def point_vel(values: np.ndarray) -> float:
    """Compute a toy velocity perturbation for a point (x, y, z).

    Accepts a numpy array of shape (3,).
    """
    if values.shape[0] != 3:
        raise ValueError(
            "point_vel expects an array-like with three numeric values"
        )

    x, y, z = float(values[0]), float(values[1]), float(values[2])
    return (x ** 2 + y ** 2 + z ** 2) ** 0.5 + 4.5 + 0.0003 * z


def get_rays(
    srp: Iterable[Tuple[Tuple[float, ...], Tuple[float, ...], List[str]]],
    model: Any,
) -> np.ndarray:
    """Return an array of (source, receiver, ray) tuples.

    One tuple per ray returned by taupy for each source/receiver pair.
    """
    srr_list: List[Tuple[object, object, object]] = []
    for source, receiver, phases in srp:
        rays = model.taupy_model.get_ray_paths_geo(
            source_depth_in_km=source[2],
            source_latitude_in_deg=source[0],
            source_longitude_in_deg=source[1],
            receiver_latitude_in_deg=receiver[0],
            receiver_longitude_in_deg=receiver[1],
            phase_list=phases,
        )
        for ray in rays:
            srr_list.append((source, receiver, ray))

    return np.array(srr_list, dtype=object)


def main() -> None:
    seed(0)

    # Load model and create mesh
    model = PlanetModel.from_standard_model("prem")

    # Create mesh and save if not exist, otherwise load existing
    mesh_path = "prem_mesh"
    try:
        model.create_mesh(from_file=mesh_path)
        LOGGER.info("Loaded existing mesh from %s", mesh_path)
    except FileNotFoundError:
        LOGGER.info("Creating new mesh...")
        radii = [1221.5, 3480.0, 6371]
        h_layers = [1000, 1000, 600]
        model.create_mesh(mesh_size_km=1000, radii=radii, H_layers=h_layers)
        model.mesh.populate_properties(["vp", "vs", "rho"])
        model.mesh.save(mesh_path)

    mesh_obj: Any = model.mesh.mesh
    LOGGER.info("Created mesh: %s cells", mesh_obj.n_cells)

    # apply velocity model to all cell-centre points
    points = mesh_obj.cell_centers().points
    # use list comprehension to avoid complex type inference in the checker
    dv = np.array([point_vel(r) for r in points])
    mesh_obj.cell_data["dv"] = dv
    LOGGER.debug("dv array: %s", mesh_obj.cell_data["dv"])

    # Generate source and receiver points and combinations
    sources = [
        make_point("Source", min_depth=150, max_depth=150)
        for _ in range(2)
    ]
    receivers = [make_point("Receiver", max_depth=0) for _ in range(5)]
    phases = ["P", "S", "ScS"]

    # For G_FwdOp: build source-receiver-phase combinations
    srp = [
        pair + (phases,)
        for pair in product(sources, receivers)
    ]
    srr = get_rays(srp, model)

    # Cross-section showing background Vp
    LOGGER.info("Background P-wave velocity:")
    plane_normal = np.array([1.0, 0.0, 1.0])
    plotter1 = model.mesh.plot_cross_section(
        plane_normal=plane_normal, property_name="dv"
    )
    plotter1.camera.position = (8000, 6000, 10000)
    plotter1.show()

    LOGGER.info("Calculate travel time kernels and residuals...")
    appl = GFwdOp(model, srr[:, 2])
    travel_times = appl.__apply__(mesh_obj.cell_data["dv"])
    print(travel_times)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
