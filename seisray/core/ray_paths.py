"""
Ray path tracing and analysis functionality.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from obspy.taup import TauPyModel


class RayPathTracer:
    """
    Class for computing and analyzing seismic ray paths through Earth models.
    """

    def __init__(self, model_name: str = "iasp91"):
        """
        Initialize the ray path tracer.

        Parameters
        ----------
        model_name : str
            Name of the Earth model to use
        """
        self.model_name = model_name
        self.model = TauPyModel(model=model_name)

    def get_ray_paths(self,
                     source_depth: float,
                     distance_deg: float,
                     phases: List[str] = ["P", "S"]) -> List:
        """
        Get ray paths for specified phases.

        Parameters
        ----------
        source_depth : float
            Source depth in km
        distance_deg : float
            Epicentral distance in degrees
        phases : List[str]
            List of seismic phases

        Returns
        -------
        ray_paths : List
            List of ray path objects from ObsPy
        """
        ray_paths = self.model.get_ray_paths(
            source_depth_in_km=source_depth,
            distance_in_degree=distance_deg,
            phase_list=phases
        )
        return ray_paths

    def extract_ray_coordinates(self, ray_paths: List) -> Dict:
        """
        Extract coordinates from ray path objects.

        Parameters
        ----------
        ray_paths : List
            List of ray path objects

        Returns
        -------
        coordinates : Dict
            Dictionary with ray path coordinates for each phase
        """
        coordinates = {}
        earth_radius = 6371.0  # km

        for ray_path in ray_paths:
            path = ray_path.path

            # Extract path coordinates
            distances_rad = path['dist']  # radians
            depths = path['depth']  # km
            times = path['time']  # seconds

            # Convert to various coordinate systems
            distances_deg = distances_rad * 180.0 / np.pi
            radius = earth_radius - depths

            # Cartesian coordinates for cross-section plotting
            x_cart = radius * np.cos(distances_rad)
            y_cart = radius * np.sin(distances_rad)

            coordinates[ray_path.name] = {
                'distances_rad': distances_rad,
                'distances_deg': distances_deg,
                'depths': depths,
                'radius': radius,
                'times': times,
                'x_cartesian': x_cart,
                'y_cartesian': y_cart,
                'total_time': ray_path.time,
                'ray_param': ray_path.ray_param
            }

        return coordinates

    def calculate_pierce_points(self,
                               ray_paths: List,
                               pierce_depths: List[float]) -> Dict:
        """
        Calculate where rays pierce specific depth levels.

        Parameters
        ----------
        ray_paths : List
            List of ray path objects
        pierce_depths : List[float]
            List of depths to find pierce points for

        Returns
        -------
        pierce_points : Dict
            Dictionary with pierce point information
        """
        pierce_points = {}

        for ray_path in ray_paths:
            path = ray_path.path
            depths = path['depth']
            distances = path['dist'] * 180.0 / np.pi
            times = path['time']

            phase_pierces = {}

            for target_depth in pierce_depths:
                pierces = []

                # Find all pierce points for this depth
                for i in range(len(depths) - 1):
                    if ((depths[i] <= target_depth <= depths[i+1]) or
                        (depths[i+1] <= target_depth <= depths[i])):

                        # Linear interpolation for accurate pierce point
                        if depths[i+1] != depths[i]:
                            f = ((target_depth - depths[i]) /
                                 (depths[i+1] - depths[i]))
                            pierce_dist = (distances[i] +
                                         f * (distances[i+1] - distances[i]))
                            pierce_time = times[i] + f * (times[i+1] - times[i])
                        else:
                            pierce_dist = distances[i]
                            pierce_time = times[i]

                        pierces.append({
                            'distance_deg': pierce_dist,
                            'time': pierce_time,
                            'depth': target_depth
                        })

                phase_pierces[target_depth] = pierces

            pierce_points[ray_path.name] = phase_pierces

        return pierce_points

    def calculate_ray_density(self,
                             source_depth: float,
                             distances: np.ndarray,
                             phases: List[str],
                             depth_range: Tuple[float, float],
                             grid_size: Tuple[int, int]) -> Dict:
        """
        Calculate ray density in a 2D grid for visualization.

        Parameters
        ----------
        source_depth : float
            Source depth in km
        distances : np.ndarray
            Array of distances in degrees
        phases : List[str]
            List of seismic phases
        depth_range : Tuple[float, float]
            (min_depth, max_depth) in km
        grid_size : Tuple[int, int]
            (n_distance, n_depth) grid dimensions

        Returns
        -------
        density : Dict
            Dictionary with ray density information
        """
        min_dist, max_dist = distances.min(), distances.max()
        min_depth, max_depth = depth_range

        # Create grid
        dist_grid = np.linspace(min_dist, max_dist, grid_size[0])
        depth_grid = np.linspace(min_depth, max_depth, grid_size[1])

        density = {
            'distance_grid': dist_grid,
            'depth_grid': depth_grid,
            'phases': {}
        }

        for phase in phases:
            phase_density = np.zeros(grid_size)

            for distance in distances:
                try:
                    ray_paths = self.get_ray_paths(source_depth, distance, [phase])

                    if ray_paths:
                        path = ray_paths[0].path
                        path_distances = path['dist'] * 180.0 / np.pi
                        path_depths = path['depth']

                        # Interpolate ray path onto grid
                        for i in range(len(path_distances) - 1):
                            # Find grid cells that this ray segment passes through
                            d1, d2 = path_distances[i], path_distances[i+1]
                            z1, z2 = path_depths[i], path_depths[i+1]

                            # Simple ray density calculation
                            if min_dist <= d1 <= max_dist and min_depth <= z1 <= max_depth:
                                dist_idx = int((d1 - min_dist) / (max_dist - min_dist) * (grid_size[0] - 1))
                                depth_idx = int((z1 - min_depth) / (max_depth - min_depth) * (grid_size[1] - 1))

                                dist_idx = max(0, min(grid_size[0] - 1, dist_idx))
                                depth_idx = max(0, min(grid_size[1] - 1, depth_idx))

                                phase_density[dist_idx, depth_idx] += 1

                except Exception:
                    continue

            density['phases'][phase] = phase_density

        return density

    def get_turning_points(self, ray_paths: List) -> Dict:
        """
        Find turning points (maximum depth) for each ray path.

        Parameters
        ----------
        ray_paths : List
            List of ray path objects

        Returns
        -------
        turning_points : Dict
            Dictionary with turning point information
        """
        turning_points = {}

        for ray_path in ray_paths:
            path = ray_path.path
            depths = path['depth']
            distances = path['dist'] * 180.0 / np.pi
            times = path['time']

            # Find maximum depth
            max_depth_idx = np.argmax(depths)
            max_depth = depths[max_depth_idx]
            turning_distance = distances[max_depth_idx]
            turning_time = times[max_depth_idx]

            turning_points[ray_path.name] = {
                'depth': max_depth,
                'distance_deg': turning_distance,
                'time': turning_time,
                'ray_param': ray_path.ray_param
            }

        return turning_points
