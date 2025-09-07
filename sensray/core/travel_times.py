"""
Core travel time calculation functionality.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, locations2degrees


class TravelTimeCalculator:
    """
    Main class for computing seismic travel times using 1D Earth models.

    This class provides methods for calculating travel times between
    earthquake sources and seismic receivers using various Earth models.
    """

    def __init__(self, model_name: str = "iasp91"):
        """
        Initialize the travel time calculator.

        Parameters
        ----------
        model_name : str
            Name of the Earth model to use (iasp91, prem, ak135, etc.)
        """
        self.model_name = model_name
        self.model = TauPyModel(model=model_name)

    def calculate_travel_times(self,
                             source_depth: float,
                             distance_deg: float,
                             phases: List[str] = ["P", "S"]) -> List:
        """
        Calculate travel times for specified phases.

        Parameters
        ----------
        source_depth : float
            Source depth in km
        distance_deg : float
            Epicentral distance in degrees
        phases : List[str]
            List of seismic phases to compute

        Returns
        -------
        arrivals : List
            List of arrival objects from ObsPy
        """
        arrivals = self.model.get_travel_times(
            source_depth_in_km=source_depth,
            distance_in_degree=distance_deg,
            phase_list=phases
        )
        return arrivals

    def calculate_from_coordinates(self,
                                 eq_lat: float, eq_lon: float, eq_depth: float,
                                 sta_lat: float, sta_lon: float,
                                 phases: List[str] = ["P", "S"]) -> Tuple[List, Dict]:
        """
        Calculate travel times using geographic coordinates.

        Parameters
        ----------
        eq_lat, eq_lon : float
            Earthquake latitude and longitude in degrees
        eq_depth : float
            Earthquake depth in km
        sta_lat, sta_lon : float
            Station latitude and longitude in degrees
        phases : List[str]
            List of seismic phases to compute

        Returns
        -------
        arrivals : List
            List of arrival objects
        info : Dict
            Dictionary with distance, azimuth, and other information
        """
        # Calculate distance and azimuth
        distance_m, azimuth, back_azimuth = gps2dist_azimuth(
            eq_lat, eq_lon, sta_lat, sta_lon
        )
        distance_km = distance_m / 1000.0
        distance_deg = locations2degrees(eq_lat, eq_lon, sta_lat, sta_lon)

        # Calculate travel times
        arrivals = self.calculate_travel_times(eq_depth, distance_deg, phases)

        # Package information
        info = {
            'distance_km': distance_km,
            'distance_deg': distance_deg,
            'azimuth': azimuth,
            'back_azimuth': back_azimuth,
            'eq_coords': (eq_lat, eq_lon, eq_depth),
            'sta_coords': (sta_lat, sta_lon)
        }

        return arrivals, info

    def calculate_travel_times_flexible(self,
                                       source_lat: Optional[float] = None,
                                       source_lon: Optional[float] = None,
                                       source_depth: float = 0.0,
                                       receiver_lat: Optional[float] = None,
                                       receiver_lon: Optional[float] = None,
                                       distance_deg: Optional[float] = None,
                                       phases: List[str] = ["P", "S"]) -> Tuple[List, Dict]:
        """
        Flexible travel time calculation supporting both geographic and distance-based input.

        This method mirrors the flexibility of RayPathTracer.get_ray_paths_flexible()
        for consistent API across the package.

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

        Returns
        -------
        arrivals : List
            List of arrival objects
        info : Dict
            Dictionary with coordinate and calculation information

        Examples
        --------
        # Geographic coordinates
        >>> arrivals, info = calc.calculate_travel_times_flexible(
        ...     source_lat=10.0, source_lon=20.0, source_depth=15.0,
        ...     receiver_lat=30.0, receiver_lon=40.0, phases=['P', 'S'])

        # Distance-based
        >>> arrivals, info = calc.calculate_travel_times_flexible(
        ...     source_depth=15.0, distance_deg=45.0, phases=['P'])
        """

        # Validate input combinations
        has_source_geo = source_lat is not None and source_lon is not None
        has_receiver_geo = receiver_lat is not None and receiver_lon is not None
        has_distance = distance_deg is not None

        if not has_source_geo and not has_distance:
            raise ValueError("Must provide either source geographic coordinates or distance")

        if has_receiver_geo and has_distance:
            raise ValueError("Cannot specify both receiver coordinates and distance")

        if not has_receiver_geo and not has_distance:
            raise ValueError("Must provide either receiver coordinates or distance")

        # Geographic coordinates provided
        if has_source_geo and has_receiver_geo:
            assert source_lat is not None and source_lon is not None
            assert receiver_lat is not None and receiver_lon is not None
            return self.calculate_from_coordinates(
                eq_lat=source_lat, eq_lon=source_lon, eq_depth=source_depth,
                sta_lat=receiver_lat, sta_lon=receiver_lon, phases=phases
            )

        # Distance provided
        elif has_distance:
            arrivals = self.calculate_travel_times(source_depth, distance_deg, phases)
            info = {
                'input_type': 'distance_only',
                'distance_deg': distance_deg,
                'distance_km': distance_deg * 111.32,
                'source_coords': (source_lat, source_lon, source_depth) if has_source_geo else (None, None, source_depth),
                'receiver_coords': None
            }
            return arrivals, info

        # Should not reach here due to validation
        raise ValueError("Invalid input combination")

    def create_travel_time_table(self,
                               source_depth: float,
                               distances: np.ndarray,
                               phases: List[str] = ["P", "S"]) -> Dict:
        """
        Create a travel time table for multiple distances.

        Parameters
        ----------
        source_depth : float
            Source depth in km
        distances : np.ndarray
            Array of distances in degrees
        phases : List[str]
            List of seismic phases

        Returns
        -------
        table : Dict
            Dictionary with phases as keys and travel time arrays as values
        """
        table = {phase: [] for phase in phases}
        valid_distances = []

        for dist in distances:
            try:
                arrivals = self.calculate_travel_times(source_depth, dist, phases)

                # Create arrival dictionary for this distance
                dist_arrivals = {arr.name: arr.time for arr in arrivals}

                # Add times for each phase (NaN if not present)
                distance_valid = False
                for phase in phases:
                    if phase in dist_arrivals:
                        table[phase].append(dist_arrivals[phase])
                        distance_valid = True
                    else:
                        table[phase].append(np.nan)

                if distance_valid:
                    valid_distances.append(dist)
                else:
                    valid_distances.append(np.nan)

            except Exception:
                # Add NaN for all phases if calculation fails
                for phase in phases:
                    table[phase].append(np.nan)
                valid_distances.append(np.nan)

        table['distances'] = np.array(valid_distances)

        # Convert to numpy arrays
        for phase in phases:
            table[phase] = np.array(table[phase])

        return table

    def compare_models(self,
                      source_depth: float,
                      distance_deg: float,
                      phase: str = "P",
                      models: List[str] = ["iasp91", "prem", "ak135"]) -> Dict:
        """
        Compare travel times between different Earth models.

        Parameters
        ----------
        source_depth : float
            Source depth in km
        distance_deg : float
            Distance in degrees
        phase : str
            Seismic phase to compare
        models : List[str]
            List of model names to compare

        Returns
        -------
        comparison : Dict
            Dictionary with model names as keys and travel times as values
        """
        comparison = {}

        for model_name in models:
            try:
                temp_model = TauPyModel(model=model_name)
                arrivals = temp_model.get_travel_times(
                    source_depth_in_km=source_depth,
                    distance_in_degree=distance_deg,
                    phase_list=[phase]
                )

                if arrivals:
                    comparison[model_name] = arrivals[0].time
                else:
                    comparison[model_name] = np.nan

            except Exception:
                comparison[model_name] = np.nan

        return comparison

    def get_phase_info(self, arrivals: List) -> Dict:
        """
        Extract detailed information from arrivals.

        Parameters
        ----------
        arrivals : List
            List of arrival objects from ObsPy

        Returns
        -------
        info : Dict
            Dictionary with phase information
        """
        info = {}

        for arrival in arrivals:
            phase_info = {
                'time': arrival.time,
                'ray_param': arrival.ray_param,
                'takeoff_angle': arrival.takeoff_angle,
                'incident_angle': arrival.incident_angle,
                'purist_distance': arrival.purist_distance,
                'purist_name': arrival.purist_name
            }

            # Handle multiple arrivals of the same phase
            if arrival.name in info:
                if not isinstance(info[arrival.name], list):
                    info[arrival.name] = [info[arrival.name]]
                info[arrival.name].append(phase_info)
            else:
                info[arrival.name] = phase_info

        return info
