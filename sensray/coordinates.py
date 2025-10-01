
"""
Coordinate conversion and geographic utilities.
"""

import numpy as np
from typing import Tuple, List
from obspy.geodetics import gps2dist_azimuth, locations2degrees


class CoordinateConverter:
    """
    Utilities for coordinate conversions and geographic calculations.
    """

    @staticmethod
    def geographic_to_distance(
        eq_lat: float,
        eq_lon: float,
        sta_lat: float,
        sta_lon: float,
    ) -> Tuple[float, float, float]:
        """
        Convert geographic coordinates to distance and azimuth.

        Parameters
        ----------
        eq_lat, eq_lon : float
            Earthquake latitude and longitude in degrees
        sta_lat, sta_lon : float
            Station latitude and longitude in degrees

        Returns
        -------
        distance_km : float
            Distance in kilometers
        distance_deg : float
            Distance in degrees
        azimuth : float
            Azimuth in degrees
        """
        distance_m, azimuth, _ = gps2dist_azimuth(
            eq_lat, eq_lon, sta_lat, sta_lon
        )
        distance_km = distance_m / 1000.0
        # Cast to float in case the library returns a numpy scalar/ndarray
        distance_deg = float(
            locations2degrees(eq_lat, eq_lon, sta_lat, sta_lon)
        )

        return distance_km, distance_deg, azimuth

    @staticmethod
    def degrees_to_km(
        distance_deg: float,
        earth_radius: float = 6371.0,
    ) -> float:
        """
        Convert distance from degrees to kilometers.

        Parameters
        ----------
        distance_deg : float
            Distance in degrees
        earth_radius : float
            Earth radius in km

        Returns
        -------
        distance_km : float
            Distance in kilometers
        """
        return distance_deg * np.pi * earth_radius / 180.0

    @staticmethod
    def km_to_degrees(
        distance_km: float,
        earth_radius: float = 6371.0,
    ) -> float:
        """
        Convert distance from kilometers to degrees.

        Parameters
        ----------
        distance_km : float
            Distance in kilometers
        earth_radius : float
            Earth radius in km

        Returns
        -------
        distance_deg : float
            Distance in degrees
        """
        return distance_km * 180.0 / (np.pi * earth_radius)

    @staticmethod
    def cartesian_to_polar(
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Cartesian coordinates to polar.

        Parameters
        ----------
        x, y : np.ndarray
            Cartesian coordinates

        Returns
        -------
        r, theta : np.ndarray
            Polar coordinates (radius, angle in radians)
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

    @staticmethod
    def polar_to_cartesian(
        r: np.ndarray,
        theta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert polar coordinates to Cartesian.

        Parameters
        ----------
        r, theta : np.ndarray
            Polar coordinates (radius, angle in radians)

        Returns
        -------
        x, y : np.ndarray
            Cartesian coordinates
        """
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    @staticmethod
    def create_station_array(
        center_lat: float,
        center_lon: float,
        radius_km: float,
        n_stations: int,
    ) -> List[Tuple[float, float]]:
        """
        Create a circular array of stations around a center point.

        Parameters
        ----------
        center_lat, center_lon : float
            Center coordinates in degrees
        radius_km : float
            Radius of array in kilometers
        n_stations : int
            Number of stations

        Returns
        -------
        stations : List[Tuple[float, float]]
            List of (latitude, longitude) pairs
        """
        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.0  # Rough conversion

        stations = []
        for i in range(n_stations):
            angle = 2 * np.pi * i / n_stations

            # Approximate offset in lat/lon
            lat_offset = radius_deg * np.cos(angle)
            lon_offset = (
                radius_deg
                * np.sin(angle)
                / np.cos(np.radians(center_lat))
            )

            station_lat = center_lat + lat_offset
            station_lon = center_lon + lon_offset

            stations.append((station_lat, station_lon))

        return stations

    @staticmethod
    def earth_to_cartesian(
        lat: float,
        lon: float,
        depth: float,
        earth_radius: float = 6371.0,
    ) -> Tuple[float, float, float]:
        """
        Convert Earth coordinates to 3D Cartesian.

        Parameters
        ----------
        lat, lon : float
            Latitude and longitude in degrees
        depth : float
            Depth below surface in km
        earth_radius : float
            Earth radius in km

        Returns
        -------
        x, y, z : float
            Cartesian coordinates in km
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        r = earth_radius - depth

        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        return x, y, z

    @staticmethod
    def compute_gc_plane_normal(
        lat1_deg: float,
        lon1_deg: float,
        lat2_deg: float,
        lon2_deg: float,
        radius_km: float = 6371.0,
    ) -> Tuple[float, float, float]:
        """
        Compute a unit normal vector (x, y, z) to the great-circle plane
        defined by two geographic points (lat1, lon1) and (lat2, lon2).

        The normal is the cross product of the two position vectors from
        the center of the Earth to the surface points. Returns a 3-tuple.
        """
        a_x, a_y, a_z = CoordinateConverter.earth_to_cartesian(
            lat1_deg, lon1_deg, depth=0.0, earth_radius=radius_km
        )
        b_x, b_y, b_z = CoordinateConverter.earth_to_cartesian(
            lat2_deg, lon2_deg, depth=0.0, earth_radius=radius_km
        )

        a = np.array((a_x, a_y, a_z), dtype=float)
        b = np.array((b_x, b_y, b_z), dtype=float)

        n = np.cross(a, b)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError(
                "Points are co-linear or identical; normal is undefined"
            )
        return tuple((n / norm).tolist())

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """
        Validate latitude and longitude coordinates.

        Parameters
        ----------
        lat, lon : float
            Latitude and longitude in degrees

        Returns
        -------
        valid : bool
            True if coordinates are valid
        """
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
