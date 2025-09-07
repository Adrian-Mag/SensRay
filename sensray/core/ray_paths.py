"""
Ray path tracing and analysis functionality.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy import geodetics


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

    def get_ray_paths(
        self,
        source_lat: Optional[float] = None,
        source_lon: Optional[float] = None,
        source_depth: float = 0.0,
        receiver_lat: Optional[float] = None,
        receiver_lon: Optional[float] = None,
        distance_deg: Optional[float] = None,
        phases: List[str] = ["P", "S"],
        output_geographic: bool = True
    ) -> Tuple[List, Dict]:
        """
        Flexible ray path calculation supporting both geographic and
        distance-based input.

        This method allows users to specify source/receiver in multiple ways:
        1. Geographic coordinates (lat/lon) for both source and receiver
        2. Geographic coordinates for source + distance in degrees
        3. Distance in degrees only (requires source_depth)

        Ray path output can be either geographic coordinates or distance-based.

        Parameters
        ----------
        source_lat, source_lon : float, optional
            Source latitude and longitude in degrees
        source_depth : float
            Source depth in km (default: 0.0)
        receiver_lat, receiver_lon : float, optional
            Receiver latitude and longitude in degrees
        distance_deg : float, optional
            Epicentral distance in degrees (alternative to receiver coords)
        phases : List[str]
            List of seismic phases to compute
        output_geographic : bool
            If True, return ray paths with geographic coordinates
            If False, return standard distance-based ray paths

        Returns
        -------
        ray_paths : List
            List of ray path objects from ObsPy
        info : Dict
            Dictionary with coordinate and calculation information

        Raises
        ------
        ValueError
            If insufficient coordinate information is provided

        Examples
        --------
        # Geographic input and output
        >>> ray_paths, info = tracer.get_ray_paths_flexible(
        ...     source_lat=10.0, source_lon=20.0, source_depth=15.0,
        ...     receiver_lat=30.0, receiver_lon=40.0,
        ...     phases=['P', 'S'], output_geographic=True)

        # Geographic input, distance-based output
        >>> ray_paths, info = tracer.get_ray_paths_flexible(
        ...     source_lat=10.0, source_lon=20.0, source_depth=15.0,
        ...     receiver_lat=30.0, receiver_lon=40.0,
        ...     phases=['P'], output_geographic=False)

        # Distance-based input and output
        >>> ray_paths, info = tracer.get_ray_paths_flexible(
        ...     source_depth=15.0, distance_deg=45.0,
        ...     phases=['P'], output_geographic=False)

        # Mixed: geographic source + distance
        >>> ray_paths, info = tracer.get_ray_paths_flexible(
        ...     source_lat=10.0, source_lon=20.0, source_depth=15.0,
        ...     distance_deg=45.0, phases=['P'], output_geographic=True)
        """

        # Validate input combinations
        has_source_geo = source_lat is not None and source_lon is not None
        has_receiver_geo = receiver_lat is not None and receiver_lon is not None
        has_distance = distance_deg is not None

        # Check for valid input combinations
        if not has_source_geo and not has_distance:
            raise ValueError("Must provide either source geographic coordinates or distance")

        if has_receiver_geo and has_distance:
            raise ValueError("Cannot specify both receiver coordinates and distance")

        if not has_receiver_geo and not has_distance:
            raise ValueError("Must provide either receiver coordinates or distance")

        # Initialize info dictionary
        info = {
            'input_type': None,
            'source_coords': None,
            'receiver_coords': None,
            'distance_deg': None,
            'distance_km': None,
            'azimuth': None,
            'back_azimuth': None,
            'output_geographic': output_geographic
        }

        # Case 1: Full geographic coordinates for source and receiver
        if has_source_geo and has_receiver_geo:
            info['input_type'] = 'geographic_full'
            info['source_coords'] = (source_lat, source_lon, source_depth)
            info['receiver_coords'] = (receiver_lat, receiver_lon, 0.0)

            # Calculate distance and azimuth
            distance_m, azimuth, back_azimuth = gps2dist_azimuth(
                source_lat, source_lon, receiver_lat, receiver_lon
            )
            distance_km = distance_m / 1000.0
            distance_deg_calc = float(locations2degrees(
                source_lat, source_lon, receiver_lat, receiver_lon
            ))

            info.update({
                'distance_deg': distance_deg_calc,
                'distance_km': distance_km,
                'azimuth': azimuth,
                'back_azimuth': back_azimuth
            })

            # Get ray paths based on output preference
            if output_geographic:
                ray_paths = self.model.get_ray_paths_geo(
                    source_depth_in_km=source_depth,
                    source_latitude_in_deg=source_lat,
                    source_longitude_in_deg=source_lon,
                    receiver_latitude_in_deg=receiver_lat,
                    receiver_longitude_in_deg=receiver_lon,
                    phase_list=phases
                )
            else:
                ray_paths = self.get_ray_paths(source_depth, distance_deg_calc, phases)

        # Case 2: Geographic source + distance
        elif has_source_geo and has_distance:
            assert source_lat is not None and source_lon is not None  # Type check
            info['input_type'] = 'geographic_source_distance'
            info['source_coords'] = (source_lat, source_lon, source_depth)
            info['distance_deg'] = distance_deg
            info['distance_km'] = distance_deg * 111.32

            if output_geographic:
                # Calculate receiver coordinates using great circle
                azimuth = 90.0  # Default eastward direction

                # Great circle calculation
                lat_rad = np.radians(source_lat)
                lon_rad = np.radians(source_lon)
                dist_rad = np.radians(distance_deg)
                azimuth_rad = np.radians(azimuth)

                receiver_lat_calc = np.degrees(np.arcsin(
                    np.sin(lat_rad) * np.cos(dist_rad) +
                    np.cos(lat_rad) * np.sin(dist_rad) * np.cos(azimuth_rad)
                ))
                receiver_lon_calc = np.degrees(lon_rad + np.arctan2(
                    np.sin(azimuth_rad) * np.sin(dist_rad) * np.cos(lat_rad),
                    np.cos(dist_rad) - np.sin(lat_rad) *
                    np.sin(np.radians(receiver_lat_calc))
                ))

                info['receiver_coords'] = (receiver_lat_calc,
                                         receiver_lon_calc, 0.0)
                info['azimuth'] = azimuth
                info['back_azimuth'] = (azimuth + 180.0 if azimuth < 180.0
                                       else azimuth - 180.0)

                ray_paths = self.model.get_ray_paths_geo(
                    source_depth_in_km=source_depth,
                    source_latitude_in_deg=source_lat,
                    source_longitude_in_deg=source_lon,
                    receiver_latitude_in_deg=receiver_lat_calc,
                    receiver_longitude_in_deg=receiver_lon_calc,
                    phase_list=phases
                )
            else:
                ray_paths = self.get_ray_paths(source_depth, distance_deg,
                                             phases)

        # Case 3: Distance-based only
        elif has_distance and not has_source_geo:
            info['input_type'] = 'distance_only'
            info['source_coords'] = (None, None, source_depth)
            info['distance_deg'] = distance_deg
            info['distance_km'] = distance_deg * 111.32

            if output_geographic:
                # Use default coordinates (0, 0) for source and calculate receiver
                source_lat_default, source_lon_default = 0.0, 0.0
                azimuth = 90.0  # Default eastward

                receiver_lat_calc = source_lat_default + (distance_deg * np.cos(np.radians(azimuth)))
                receiver_lon_calc = source_lon_default + (distance_deg * np.sin(np.radians(azimuth)))

                info['source_coords'] = (source_lat_default, source_lon_default, source_depth)
                info['receiver_coords'] = (receiver_lat_calc, receiver_lon_calc, 0.0)
                info['azimuth'] = azimuth
                info['back_azimuth'] = azimuth + 180.0

                ray_paths = self.model.get_ray_paths_geo(
                    source_depth_in_km=source_depth,
                    source_latitude_in_deg=source_lat_default,
                    source_longitude_in_deg=source_lon_default,
                    receiver_latitude_in_deg=receiver_lat_calc,
                    receiver_longitude_in_deg=receiver_lon_calc,
                    phase_list=phases
                )
            else:
                ray_paths = self.get_ray_paths(source_depth, distance_deg, phases)

        return ray_paths, info

    def filter_arrivals(self, ray_paths: List, method: str = 'first',
                       **kwargs) -> List:
        """
        Filter multiple arrivals using various criteria.

        Parameters
        ----------
        ray_paths : List
            List of ray path arrivals
        method : str
            Filtering method:
            - 'first': Keep only the first (fastest) arrival
            - 'ray_param': Filter by ray parameter range
            - 'depth': Filter by maximum depth range
            - 'time': Filter by travel time range
            - 'manual': Manually select by indices
        **kwargs : additional arguments for specific methods

        Returns
        -------
        filtered_paths : List
            Filtered list of arrivals
        """

        if not ray_paths:
            return []

        if method == 'first':
            # Return only the fastest arrival
            return [min(ray_paths, key=lambda x: x.time)]

        elif method == 'ray_param':
            # Filter by ray parameter range
            min_p = kwargs.get('min_ray_param', 0)
            max_p = kwargs.get('max_ray_param', float('inf'))
            return [rp for rp in ray_paths if min_p <= rp.ray_param <= max_p]

        elif method == 'depth':
            # Filter by maximum depth range
            min_depth = kwargs.get('min_depth', 0)
            max_depth = kwargs.get('max_depth', float('inf'))
            filtered = []
            for rp in ray_paths:
                max_ray_depth = max(rp.path['depth'])
                if min_depth <= max_ray_depth <= max_depth:
                    filtered.append(rp)
            return filtered

        elif method == 'time':
            # Filter by travel time range
            min_time = kwargs.get('min_time', 0)
            max_time = kwargs.get('max_time', float('inf'))
            return [rp for rp in ray_paths if min_time <= rp.time <= max_time]

        elif method == 'manual':
            # Manual selection by indices
            indices = kwargs.get('indices', [0])
            return [ray_paths[i] for i in indices if i < len(ray_paths)]

        else:
            return ray_paths

    def extract_ray_coordinates(self, ray_paths: List,
                                filter_by_distance: bool = True,
                                target_distance: Optional[float] = None,
                                distance_tolerance: float = 5.0) -> Dict:
        """
        Extract coordinates from ray path objects.

        Parameters
        ----------
        ray_paths : List
            List of ray path objects
        filter_by_distance : bool, optional
            If True, filter arrivals to only include those ending near
            target_distance
        target_distance : float, optional
            Target distance in degrees. If None, use the first arrival's
            distance
        distance_tolerance : float, optional
            Tolerance in degrees for distance filtering (default: 5.0)

        Returns
        -------
        coordinates : Dict
            Dictionary with ray path coordinates for each phase
        """
        coordinates = {}
        earth_radius = 6371.0  # km
        phase_counters = {}  # Track how many arrivals per phase

        # Determine target distance if not provided
        if (filter_by_distance and target_distance is None and
                len(ray_paths) > 0):
            first_path = ray_paths[0].path
            target_distance = first_path['dist'][-1] * 180.0 / np.pi

        for ray_path in ray_paths:
            path = ray_path.path

            # Extract path coordinates
            distances_rad = path['dist']  # radians
            depths = path['depth']  # km
            times = path['time']  # seconds

            # Convert to various coordinate systems
            distances_deg = distances_rad * 180.0 / np.pi
            radius = earth_radius - depths

            # Check if this arrival ends at the target distance
            final_distance = distances_deg[-1]
            if filter_by_distance and target_distance is not None:
                distance_diff = abs(final_distance - target_distance)
                if distance_diff > distance_tolerance:
                    continue  # Skip this arrival

            # Cartesian coordinates for cross-section plotting
            x_cart = radius * np.cos(distances_rad)
            y_cart = radius * np.sin(distances_rad)

            # Create unique key for multiple arrivals of same phase
            phase_name = ray_path.name
            if phase_name in phase_counters:
                phase_counters[phase_name] += 1
                unique_key = f"{phase_name}_{phase_counters[phase_name]}"
            else:
                phase_counters[phase_name] = 1
                unique_key = phase_name

            coordinates[unique_key] = {
                'distances_rad': distances_rad,
                'distances_deg': distances_deg,
                'depths': depths,
                'radius': radius,
                'times': times,
                'x_cartesian': x_cart,
                'y_cartesian': y_cart,
                'total_time': ray_path.time,
                'ray_param': ray_path.ray_param,
                'phase_name': phase_name  # Keep original phase name
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
