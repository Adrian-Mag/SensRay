"""
Ray path tracing and analysis functionality.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from obspy.geodetics import gps2dist_azimuth, locations2degrees
import tempfile
import os


class RayPathTracer:
    """
    Class for computing and analyzing seismic ray paths through planetary
    models.

    The ray tracer now works exclusively with PlanetModel instances, which
    handle all model file I/O and provide a unified interface for both
    standard Earth models and custom planetary models.
    """

    def __init__(self, planet_model):
        """
        Initialize the ray path tracer with a PlanetModel.

        Parameters
        ----------
        planet_model : PlanetModel
            The planetary model instance to use for ray tracing
        """
        from .model import PlanetModel

        if not isinstance(planet_model, PlanetModel):
            raise TypeError("planet_model must be a PlanetModel instance")

        self.planet_model = planet_model
        self.temp_nd_file = None

        # Create TauP model from PlanetModel
        self.model = self._create_taup_model_from_planet_model()

    def _create_taup_model_from_planet_model(self) -> TauPyModel:
        """Create a TauP model from the PlanetModel instance."""
        # Check if we need to create a standard Earth model
        metadata = getattr(self.planet_model, 'metadata', {})
        if metadata.get('source') == 'obspy_taup':
            # This is already a standard Earth model, use directly
            original_model = metadata.get('original_model', 'prem')
            return TauPyModel(model=original_model)

        # For custom models, create temporary .nd file
        self.temp_nd_file = self.planet_model.create_temp_nd_file()

        # Build TauP model from the .nd file
        model_name = self.planet_model.name.replace(' ', '_').lower()
        build_taup_model(
            self.temp_nd_file,
            output_folder=tempfile.gettempdir()
        )

        # Load the built model
        model_path = os.path.join(
            tempfile.gettempdir(),
            f"{model_name}.npz"
        )
        return TauPyModel(model=model_path)

    @classmethod
    def from_standard_model(cls, model_name: str = "prem"):
        """
        Create RayPathTracer from a standard Earth model.

        Parameters
        ----------
        model_name : str
            Name of standard Earth model ('prem', 'iasp91', 'ak135')

        Returns
        -------
        RayPathTracer
            Tracer configured with the standard Earth model
        """
        from .model import PlanetModel

        planet_model = PlanetModel.from_standard_model(model_name)
        return cls(planet_model)

    @classmethod
    def from_nd_file(cls, filepath: str, name: Optional[str] = None):
        """
        Create RayPathTracer from a .nd format file.

        Parameters
        ----------
        filepath : str
            Path to the .nd format file
        name : str, optional
            Name for the model

        Returns
        -------
        RayPathTracer
            Tracer configured with the model from file
        """
        from .model import PlanetModel

        planet_model = PlanetModel.from_nd_file(filepath, name)
        return cls(planet_model)

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_nd_file and os.path.exists(self.temp_nd_file):
            try:
                os.unlink(self.temp_nd_file)
                self.temp_nd_file = None
            except OSError:
                pass  # File might already be cleaned up

    def __del__(self):
        """Destructor to clean up temporary files."""
        self.cleanup()

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
        has_receiver_geo = (
            receiver_lat is not None and receiver_lon is not None
        )
        has_distance = distance_deg is not None

        # Check for valid input combinations
        if not has_source_geo and not has_distance:
            raise ValueError(
                "Must provide either source geographic coordinates or distance"
            )

        if has_receiver_geo and has_distance:
            raise ValueError(
                "Cannot specify both receiver coordinates and distance"
            )

        if not has_receiver_geo and not has_distance:
            raise ValueError(
                "Must provide either receiver coordinates or distance"
            )

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
                ray_paths = self.model.get_ray_paths(
                    distance_in_degree=distance_deg_calc,
                    source_depth_in_km=source_depth,
                    phase_list=phases,
                )

        # Case 2: Geographic source + distance
        elif has_source_geo and has_distance:
            # Type check
            assert (
                source_lat is not None and source_lon is not None
            )
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

                info['receiver_coords'] = (
                    receiver_lat_calc,
                    receiver_lon_calc,
                    0.0,
                )
                info['azimuth'] = azimuth
                info['back_azimuth'] = (
                    azimuth + 180.0 if azimuth < 180.0 else azimuth - 180.0
                )

                ray_paths = self.model.get_ray_paths_geo(
                    source_depth_in_km=source_depth,
                    source_latitude_in_deg=source_lat,
                    source_longitude_in_deg=source_lon,
                    receiver_latitude_in_deg=receiver_lat_calc,
                    receiver_longitude_in_deg=receiver_lon_calc,
                    phase_list=phases
                )
            else:
                ray_paths = self.model.get_ray_paths(
                    distance_in_degree=distance_deg,
                    source_depth_in_km=source_depth,
                    phase_list=phases,
                )

        # Case 3: Distance-based only
        elif has_distance and not has_source_geo:
            info['input_type'] = 'distance_only'
            info['source_coords'] = (None, None, source_depth)
            info['distance_deg'] = distance_deg
            info['distance_km'] = distance_deg * 111.32

            if output_geographic:
                # Use default coordinates (0, 0) for source and
                # compute receiver
                source_lat_default, source_lon_default = 0.0, 0.0
                azimuth = 90.0  # Default eastward

                receiver_lat_calc = source_lat_default + (
                    distance_deg * np.cos(np.radians(azimuth))
                )
                receiver_lon_calc = source_lon_default + (
                    distance_deg * np.sin(np.radians(azimuth))
                )

                info['source_coords'] = (
                    source_lat_default,
                    source_lon_default,
                    source_depth,
                )
                info['receiver_coords'] = (
                    receiver_lat_calc,
                    receiver_lon_calc,
                    0.0,
                )
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
                ray_paths = self.model.get_ray_paths(
                    distance_in_degree=distance_deg,
                    source_depth_in_km=source_depth,
                    phase_list=phases,
                )

        return ray_paths, info

    def filter_arrivals(
        self,
        ray_paths: List,
        method: str = 'first',
        **kwargs
    ) -> List:
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

    def extract_in_plane_ray_coordinates(
            self,
            ray_paths: List,
            filter_by_distance: bool = True,
            target_distance: Optional[float] = None,
            distance_tolerance: float = 5.0
    ) -> Dict:
        """
        Extract coordinates from ray path objects in plane for plotting.

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

    def extract_ray_coordinates(
            self,
            ray_paths: List,
            filter_by_distance: bool = False,
            target_distance: Optional[float] = None,
            distance_tolerance: float = 5.0
    ) -> Dict:
        """
        Extract 3D Cartesian (x, y, z) coordinates for all ray path points.

        This works with both geographic ray paths (from get_ray_paths_geo)
        that include latitude/longitude and with standard paths that only
        include epicentral distance and depth. For the latter, a default
        great-circle along the equator (eastward from (0, 0)) is assumed
        to derive latitude/longitude for visualization purposes.

        Parameters
        ----------
        ray_paths : List
            List of ObsPy ray path arrivals
        filter_by_distance : bool, optional
            If True, keep only arrivals whose final distance is within
            distance_tolerance of target_distance (in degrees)
        target_distance : float, optional
            Target epicentral distance in degrees. If None and filtering is
            enabled, the final distance of the first path is used.
        distance_tolerance : float, optional
            Tolerance in degrees for distance-based filtering (default 5.0)

        Returns
        -------
        Dict
            Mapping from unique phase key to a dictionary containing:
            - latitude, longitude, depths
            - distances_rad, distances_deg, times
            - x, y, z (Cartesian km)
            - total_time, ray_param, phase_name
        """
        from sensray.utils.coordinates import CoordinateConverter

        coordinates: Dict[str, Dict] = {}
        phase_counters: Dict[str, int] = {}

        if not ray_paths:
            return coordinates

        # Determine target distance (deg) if filtering and not provided
        if filter_by_distance and target_distance is None:
            first_path = ray_paths[0].path
            # Try to read final distance; handle structured arrays gracefully
            target_distance = None
            try:
                target_distance = float(
                    np.asarray(first_path['dist'], dtype=float)[-1]
                    * 180.0
                    / np.pi
                )
            except Exception:
                # Fallback: try longitude progression, else leave None
                try:
                    lon0 = np.asarray(first_path['longitude'], dtype=float)
                    target_distance = float(lon0[-1] - lon0[0])
                except Exception:
                    target_distance = None

        for rp in ray_paths:
            path = rp.path

            # Common arrays
            depths = np.asarray(path['depth'], dtype=float)
            times = np.asarray(path['time'], dtype=float)

            # Prefer provided distances in radians if available
            # Prefer provided distances in radians if available. Use try/except
            # because `path` can be a numpy structured array where `in` checks
            # produce elementwise comparisons and raise TypeError.
            try:
                distances_rad = np.asarray(path['dist'], dtype=float)
                distances_deg = distances_rad * 180.0 / np.pi
            except Exception:
                # Raise error otherwise
                raise ValueError(
                    "Ray path lacks distance information."
                )

            # Optional distance filtering based on final distance
            if (
                filter_by_distance and target_distance is not None and
                len(distances_deg) > 0
            ):
                final_distance = float(distances_deg[-1])
                if abs(final_distance - target_distance) > distance_tolerance:
                    continue

            # Gather/construct geographic coordinates
            try:
                lat_arr = np.asarray(path['lat'], dtype=float)
                lon_arr = np.asarray(path['lon'], dtype=float)
            except Exception:
                # Throw error message
                raise ValueError(
                    "Ray path lacks latitude/longitude information."
                )
                # --- IGNORE ---

            # Convert to Cartesian
            x_list: List[float] = []
            y_list: List[float] = []
            z_list: List[float] = []
            for lat, lon, dep in zip(lat_arr, lon_arr, depths):
                x, y, z = CoordinateConverter.earth_to_cartesian(
                    lat, lon, float(dep)
                )
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)

            # Unique key per phase/arrival
            phase_name = rp.name
            if phase_name in phase_counters:
                phase_counters[phase_name] += 1
                unique_key = f"{phase_name}_{phase_counters[phase_name]}"
            else:
                phase_counters[phase_name] = 1
                unique_key = phase_name

            coordinates[unique_key] = {
                'latitude': lat_arr,
                'longitude': lon_arr,
                'depths': depths,
                'distances_rad': distances_rad,
                'distances_deg': distances_deg,
                'times': times,
                'x': np.asarray(x_list),
                'y': np.asarray(y_list),
                'z': np.asarray(z_list),
                'xyz': np.column_stack(
                    (
                        np.asarray(x_list),
                        np.asarray(y_list),
                        np.asarray(z_list),
                    )
                ),
                'total_time': getattr(rp, 'time', None),
                'ray_param': getattr(rp, 'ray_param', None),
                'phase_name': phase_name
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
                    cond1 = depths[i] <= target_depth <= depths[i + 1]
                    cond2 = depths[i + 1] <= target_depth <= depths[i]
                    if cond1 or cond2:

                        # Linear interpolation for accurate pierce point
                        if depths[i + 1] != depths[i]:
                            f = (
                                (target_depth - depths[i]) /
                                (depths[i + 1] - depths[i])
                            )
                            pierce_dist = (
                                distances[i] +
                                f * (distances[i + 1] - distances[i])
                            )
                            pierce_time = (
                                times[i] + f * (times[i + 1] - times[i])
                            )
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

    def calculate_ray_density(
        self,
        source_depth: float,
        distances: np.ndarray,
        phases: List[str],
        depth_range: Tuple[float, float],
        grid_size: Tuple[int, int]
    ) -> Dict:
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
                    ray_paths = self.model.get_ray_paths(
                        distance_in_degree=float(distance),
                        source_depth_in_km=float(source_depth),
                        phase_list=[phase],
                    )

                    first_arrival = next(iter(ray_paths), None)
                    if first_arrival is not None:
                        path = first_arrival.path
                        path_distances = path['dist'] * 180.0 / np.pi
                        path_depths = path['depth']

                        # Interpolate ray path onto grid
                        for i in range(len(path_distances) - 1):
                            # Grid cell for this ray point
                            d1 = path_distances[i]
                            z1 = path_depths[i]

                            # Simple ray density calculation
                            if (
                                min_dist <= d1 <= max_dist and
                                min_depth <= z1 <= max_depth
                            ):
                                dist_idx = int(
                                    (d1 - min_dist) /
                                    (max_dist - min_dist) *
                                    (grid_size[0] - 1)
                                )
                                depth_idx = int(
                                    (z1 - min_depth) /
                                    (max_depth - min_depth) *
                                    (grid_size[1] - 1)
                                )

                                dist_idx = max(
                                    0, min(grid_size[0] - 1, dist_idx)
                                )
                                depth_idx = max(
                                    0, min(grid_size[1] - 1, depth_idx)
                                )

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
