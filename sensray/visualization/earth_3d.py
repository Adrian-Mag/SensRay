"""
3D interactive Earth visualization using PyVista.

Features include seismic ray paths, Earth structure, meshes, and
volumetric property visualization.
"""

import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Union
try:
    import cartopy.io.shapereader as shpreader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shpreader = None


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
        phase_counters = {}  # Track how many rays of each phase we've seen

        for rp in ray_paths_geo:
            if not hasattr(rp, 'path') or rp.path is None:
                continue

            # Create unique key for each ray path
            phase_name = rp.name
            if phase_name not in phase_counters:
                phase_counters[phase_name] = 0
            phase_counters[phase_name] += 1

            # Create unique key: "P_1", "P_2", "PP_1", "PP_2", etc.
            unique_key = f"{phase_name}_{phase_counters[phase_name]}"

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

            ray_paths_3d[unique_key] = {
                'x': np.array(x_coords),
                'y': np.array(y_coords),
                'z': np.array(z_coords),
                'lats': np.array(lats),
                'lons': np.array(lons),
                'depths': np.array(depths),
                'travel_time': rp.time,
                'phase_name': phase_name  # Keep original phase name
            }

        return ray_paths_3d

    def create_earth_sphere(self, resolution: int = 50,
                            show_continents: bool = False) -> pv.PolyData:
        """
        Create a 3D Earth sphere mesh.

        Parameters
        ----------
        resolution : int
            Sphere resolution (default: 50)
        show_continents : bool
            Deprecated here. Use `show_continents` in
            `plot_3d_earth_and_rays` to draw continent outlines.

        Returns
        -------
        earth_mesh : pv.PolyData
            PyVista mesh of Earth sphere
        """
        # Create sphere
        sphere = pv.Sphere(
            radius=self.earth_radius,
            theta_resolution=resolution,
            phi_resolution=resolution,
        )

        return sphere

    def plot_3d_earth_and_rays(
        self,
        ray_paths,
        source_lat: float,
        source_lon: float,
        source_depth: float,
        receiver_lat: float,
        receiver_lon: float,
        show_earth: bool = True,
        show_continents: bool = False,
        ray_colors: Optional[Dict] = None,
        notebook: bool = True,
        show_endpoints: bool = True,
        ray_line_width: float = 2.0,
        continents_line_width: float = 0.8,
    ) -> pv.Plotter:
        """
        Create interactive 3D plot of Earth and ray paths.

        Parameters
        ----------
        ray_paths : List or Dict
            Either ObsPy ray paths (geographic) or pre-converted 3D
            coordinates dict
        source_lat, source_lon, source_depth : float
            Source coordinates
        receiver_lat, receiver_lon : float
            Receiver coordinates
        show_earth : bool
            Whether to show Earth sphere
        show_continents : bool
            If True and cartopy is available, draw continent outlines on the
            sphere surface using Natural Earth coastlines.
        ray_colors : Dict, optional
            Custom colors for each phase
        notebook : bool
            Whether running in Jupyter notebook
        show_endpoints : bool
            If True, draw source/receiver markers. If False, hide them.
        ray_line_width : float
            Line width for ray paths.
        continents_line_width : float
            Line width for continent outlines.

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

            # Optionally add continent outlines
            if show_continents:
                self._add_continent_outlines(
                    color="white",
                    line_width=continents_line_width,
                    opacity=0.9,
                    resolution='110m',
                )

        # Default colors for ray paths based on phase types
        if ray_colors is None:
            ray_colors = {}
            color_cycle = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            # Get unique phase names for coloring
            unique_phases = set()
            for unique_key, coords_3d in ray_paths_3d.items():
                if 'phase_name' in coords_3d:
                    unique_phases.add(coords_3d['phase_name'])
                else:
                    # Backward compatibility: extract phase from key
                    unique_phases.add(unique_key.split('_')[0])

            for i, phase in enumerate(sorted(unique_phases)):
                ray_colors[phase] = color_cycle[i % len(color_cycle)]

        # Add ray paths
        for unique_key, coords_3d in ray_paths_3d.items():
            # Determine the base phase name for coloring
            if 'phase_name' in coords_3d:
                base_phase = coords_3d['phase_name']
            else:
                # Backward compatibility: extract phase from key
                base_phase = unique_key.split('_')[0]

            # Create line/tube for ray path
            points = np.column_stack([
                coords_3d['x'],
                coords_3d['y'],
                coords_3d['z']
            ])

            # Create spline through points
            spline = pv.Spline(points, n_points=len(points)*2)

            # Add to plot
            color = ray_colors.get(base_phase, 'red')
            self.plotter.add_mesh(
                spline,
                color=color,
                line_width=ray_line_width,
                style='wireframe',
                render_points_as_spheres=False,
                point_size=0.1,
                label=f"{unique_key} ({coords_3d['travel_time']:.1f}s)"
            )

        if show_endpoints:
            # Add source point
            source_x, source_y, source_z = self.lat_lon_depth_to_cartesian(
                source_lat, source_lon, source_depth
            )
            source_center = [source_x, source_y, source_z]
            self.plotter.add_mesh(
                pv.Sphere(radius=50, center=source_center),
                color='red',
                label='Source',
            )

            # Add receiver point
            receiver_x, receiver_y, receiver_z = \
                self.lat_lon_depth_to_cartesian(
                    receiver_lat, receiver_lon, 0
                )
            receiver_center = [receiver_x, receiver_y, receiver_z]
            self.plotter.add_mesh(
                pv.Sphere(radius=50, center=receiver_center),
                color='blue',
                label='Receiver',
            )

        # Set up the scene
        self.plotter.add_legend()
        self.plotter.show_axes()
        self.plotter.camera.zoom(1.2)

        return self.plotter

    def _add_continent_outlines(
        self,
        color: Union[str, Tuple[float, float, float]] = "white",
        line_width: float = 1.0,
        opacity: float = 0.9,
        resolution: str = '110m',
    ) -> None:
        """
        Add continent coastlines as polylines on the Earth sphere.

        This uses cartopy's Natural Earth coastline dataset if available.
        If cartopy is not installed, this becomes a no-op.

        Parameters
        ----------
        color : str or 3-tuple
            Line color for the coastlines.
        line_width : float
            Line width for the coastlines.
        opacity : float
            Line opacity for the coastlines.
        resolution : str
            Natural Earth resolution: one of '110m', '50m', or '10m'.
        """
        if self.plotter is None:
            return
        if shpreader is None:  # cartopy not available
            return

        try:
            shpfile = shpreader.natural_earth(
                resolution=resolution,
                category='physical',
                name='coastline',
            )
            reader = shpreader.Reader(shpfile)
        except Exception:
            return

        for rec in reader.records():
            geom = rec.geometry
            # Handle LineString and MultiLineString
            geoms = []
            gtype = getattr(geom, 'geom_type', '')
            if gtype == 'LineString':
                geoms = [geom]
            elif gtype == 'MultiLineString':
                geoms = list(getattr(geom, 'geoms', []))
            else:
                continue

            for line in geoms:
                coords = list(getattr(line, 'coords', []))
                if len(coords) < 2:
                    continue

                # Convert (lon, lat) to 3D points on the sphere surface
                pts = []
                for lon, lat in coords:
                    x, y, z = self.lat_lon_depth_to_cartesian(lat, lon, 0.0)
                    pts.append([x, y, z])

                if len(pts) < 2:
                    continue

                points = np.asarray(pts, dtype=float)
                npts = points.shape[0]
                # VTK polyline connectivity: [N, 0, 1, ..., N-1]
                poly = pv.PolyData(points)
                poly.lines = np.hstack((np.array([npts]),
                                        np.arange(npts, dtype=np.int64)))
                self.plotter.add_mesh(
                    poly,
                    color=color,
                    line_width=line_width,
                    opacity=opacity,
                    style='wireframe',
                    render_points_as_spheres=False,
                    point_size=0.1,
                )

    def add_velocity_structure(
        self,
        model_data: Dict,
        property_name: str = 'vp',
        depth_slice: Optional[float] = None,
    ):
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

    def add_mesh_overlay(
        self,
        mesh_data: Union[pv.PolyData, str],
        color: str = 'yellow',
        opacity: float = 0.7,
    ):
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
