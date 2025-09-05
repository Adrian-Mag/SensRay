"""
3D interactive Earth visualization using PyVista for seismic ray paths and Earth structure.
Supports ray paths, meshes, and volumetric property visualization.
"""

import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Union
from obspy import geodetics
import matplotlib.cm as cm
import matplotlib.colors as colors


class Earth3DVisualizer:
    """
    Interactive 3D visualization of Earth structure and seismic ray paths.

    Features:
    - 3D ray path visualization with geographic coordinates
    - Earth sphere with continental outlines
    - Interior structure visualization (velocity, density)
    - Mesh overlay capabilities
    - Interactive controls
    """

    def __init__(self, earth_radius_km: float = 6371.0):
        """
        Initialize the 3D Earth visualizer.

        Parameters
        ----------
        earth_radius_km : float
            Earth radius in kilometers (default: 6371.0)
        """
        self.earth_radius = earth_radius_km
        self.plotter = None

    def lat_lon_depth_to_cartesian(self, lat: float, lon: float,
                                   depth: float) -> Tuple[float, float, float]:
        """
        Convert geographic coordinates to 3D Cartesian coordinates.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        depth : float
            Depth in kilometers

        Returns
        -------
        x, y, z : float
            3D Cartesian coordinates
        """
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Calculate radius from center (Earth radius - depth)
        r = self.earth_radius - depth

        # Convert to Cartesian coordinates
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        return x, y, z

    def convert_geographic_ray_paths_to_3d(self, ray_paths_geo: List) -> Dict:
        """
        Convert ObsPy ray paths with geographic coordinates to 3D coordinates.

        Parameters
        ----------
        ray_paths_geo : List
            ObsPy ray paths with geographic coordinates

        Returns
        -------
        ray_paths_3d : Dict
            3D ray path coordinates for each phase
        """
        ray_paths_3d = {}

        for rp in ray_paths_geo:
            if not hasattr(rp, 'path') or rp.path is None:
                continue

            # Extract path data
            path = rp.path
            depths = path['depth']    # Depths (km)
            lats = path['lat']        # Latitudes (degrees)
            lons = path['lon']        # Longitudes (degrees)

            # Convert geographic coordinates to 3D Cartesian coordinates
            x_coords = []
            y_coords = []
            z_coords = []

            for lat, lon, depth in zip(lats, lons, depths):
                x, y, z = self.lat_lon_depth_to_cartesian(lat, lon, depth)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)

            ray_paths_3d[rp.name] = {
                'x': np.array(x_coords),
                'y': np.array(y_coords),
                'z': np.array(z_coords),
                'lats': np.array(lats),
                'lons': np.array(lons),
                'depths': np.array(depths),
                'travel_time': rp.time
            }

        return ray_paths_3d

    def create_earth_sphere(self, resolution: int = 50,
                          show_continents: bool = True) -> pv.PolyData:
        """
        Create a 3D Earth sphere mesh.

        Parameters
        ----------
        resolution : int
            Sphere resolution (default: 50)
        show_continents : bool
            Whether to add continent textures/outlines

        Returns
        -------
        earth_mesh : pv.PolyData
            PyVista mesh of Earth sphere
        """
        # Create sphere
        sphere = pv.Sphere(radius=self.earth_radius,
                          theta_resolution=resolution,
                          phi_resolution=resolution)

        if show_continents:
            # Add basic continent coloring (simplified)
            # In future versions, could add actual Earth textures
            pass

        return sphere

    def plot_3d_earth_and_rays(self, ray_paths,
                              source_lat: float, source_lon: float,
                              source_depth: float,
                              receiver_lat: float, receiver_lon: float,
                              show_earth: bool = True,
                              ray_colors: Optional[Dict] = None,
                              notebook: bool = True) -> pv.Plotter:
        """
        Create interactive 3D plot of Earth and ray paths.

        Parameters
        ----------
        ray_paths : List or Dict
            Either ObsPy ray paths (geographic) or pre-converted 3D coordinates dict
        source_lat, source_lon, source_depth : float
            Source coordinates
        receiver_lat, receiver_lon : float
            Receiver coordinates
        show_earth : bool
            Whether to show Earth sphere
        ray_colors : Dict, optional
            Custom colors for each phase
        notebook : bool
            Whether running in Jupyter notebook

        Returns
        -------
        plotter : pv.Plotter
            PyVista plotter object
        """
        # Initialize plotter
        if notebook:
            self.plotter = pv.Plotter(notebook=True)
        else:
            self.plotter = pv.Plotter()

        # Determine input type and convert if necessary
        if isinstance(ray_paths, list):
            # Input is ObsPy ray paths (geographic) - convert to 3D
            ray_paths_3d = self.convert_geographic_ray_paths_to_3d(ray_paths)
        elif isinstance(ray_paths, dict):
            # Input is already 3D coordinates dict
            ray_paths_3d = ray_paths
        else:
            raise ValueError(
                "ray_paths must be either ObsPy ray paths (list) or "
                "3D coordinates (dict)")

        # Add Earth sphere
        if show_earth:
            earth = self.create_earth_sphere()
            self.plotter.add_mesh(earth, color='lightblue', opacity=0.6)

        # Default colors for ray paths
        if ray_colors is None:
            ray_colors = {}
            color_cycle = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            for i, phase in enumerate(ray_paths_3d.keys()):
                ray_colors[phase] = color_cycle[i % len(color_cycle)]

        # Add ray paths
        for phase_name, coords_3d in ray_paths_3d.items():
            # Create line/tube for ray path
            points = np.column_stack([
                coords_3d['x'],
                coords_3d['y'],
                coords_3d['z']
            ])

            # Create spline through points
            spline = pv.Spline(points, n_points=len(points)*2)

            # Add to plot
            color = ray_colors.get(phase_name, 'red')
            self.plotter.add_mesh(
                spline, color=color, line_width=3,
                label=f"{phase_name} ({coords_3d['travel_time']:.1f}s)"
            )

        # Add source point
        source_x, source_y, source_z = self.lat_lon_depth_to_cartesian(
            source_lat, source_lon, source_depth)
        source_center = [source_x, source_y, source_z]
        self.plotter.add_mesh(
            pv.Sphere(radius=50, center=source_center),
            color='red', label='Source'
        )

        # Add receiver point
        receiver_x, receiver_y, receiver_z = self.lat_lon_depth_to_cartesian(
            receiver_lat, receiver_lon, 0)
        receiver_center = [receiver_x, receiver_y, receiver_z]
        self.plotter.add_mesh(
            pv.Sphere(radius=50, center=receiver_center),
            color='blue', label='Receiver'
        )

        # Set up the scene
        self.plotter.add_legend()
        self.plotter.show_axes()
        self.plotter.camera.zoom(1.2)

        return self.plotter

    def add_velocity_structure(self, model_data: Dict,
                             property_name: str = 'vp',
                             depth_slice: Optional[float] = None):
        """
        Add velocity or density structure visualization.

        Parameters
        ----------
        model_data : Dict
            Earth model data with depth profiles
        property_name : str
            Property to visualize ('vp', 'vs', 'rho')
        depth_slice : float, optional
            Specific depth slice to show
        """
        # Implementation for adding velocity structure
        # This will be expanded based on your Earth model format
        pass

    def add_mesh_overlay(self, mesh_data: Union[pv.PolyData, str],
                        color: str = 'yellow', opacity: float = 0.7):
        """
        Add custom mesh overlay (for your future mesh visualizations).

        Parameters
        ----------
        mesh_data : pv.PolyData or str
            Mesh data or path to mesh file
        color : str
            Mesh color
        opacity : float
            Mesh transparency
        """
        if isinstance(mesh_data, str):
            mesh = pv.read(mesh_data)
        else:
            mesh = mesh_data

        if self.plotter is not None:
            self.plotter.add_mesh(mesh, color=color, opacity=opacity)

    def show(self):
        """Show the interactive plot."""
        if self.plotter is not None:
            return self.plotter.show()
