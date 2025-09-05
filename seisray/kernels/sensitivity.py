"""
Sensitivity kernel calculations for seismic tomography.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class SensitivityKernel:
    """
    Class for computing discretized sensitivity kernels for seismic waves.
    """

    def __init__(self,
                 domain_bounds: Tuple[float, float, float, float],
                 grid_size: Tuple[int, int],
                 regularization: float = 1e-4):
        """
        Initialize the sensitivity kernel calculator.

        Parameters
        ----------
        domain_bounds : Tuple[float, float, float, float]
            Domain bounds (x_min, x_max, y_min, y_max) in km
        grid_size : Tuple[int, int]
            Grid dimensions (nx, ny)
        regularization : float
            Regularization parameter for smoothing
        """
        self.domain_bounds = domain_bounds
        self.grid_size = grid_size
        self.regularization = regularization

        # Create grid
        self.x_min, self.x_max, self.y_min, self.y_max = domain_bounds
        self.nx, self.ny = grid_size

        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.y = np.linspace(self.y_min, self.y_max, self.ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.cell_area = self.dx * self.dy

        # Create coordinate grids
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='xy')

    def compute_ray_kernel(self,
                          source_pos: Tuple[float, float],
                          receiver_pos: Tuple[float, float],
                          background_velocity: np.ndarray,
                          kernel_type: str = "travel_time") -> np.ndarray:
        """
        Compute sensitivity kernel for a ray path.

        Parameters
        ----------
        source_pos : Tuple[float, float]
            Source position (x, y) in km
        receiver_pos : Tuple[float, float]
            Receiver position (x, y) in km
        background_velocity : np.ndarray
            Background velocity field on grid
        kernel_type : str
            Type of kernel ("travel_time", "amplitude")

        Returns
        -------
        kernel : np.ndarray
            Sensitivity kernel on grid
        """
        if kernel_type == "travel_time":
            return self._compute_travel_time_kernel(source_pos, receiver_pos,
                                                   background_velocity)
        elif kernel_type == "amplitude":
            return self._compute_amplitude_kernel(source_pos, receiver_pos,
                                                 background_velocity)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def _compute_travel_time_kernel(self,
                                   source_pos: Tuple[float, float],
                                   receiver_pos: Tuple[float, float],
                                   background_velocity: np.ndarray) -> np.ndarray:
        """
        Compute travel time sensitivity kernel using straight ray approximation.
        """
        # Sample points along straight-line ray
        n_samples = 1000
        t = np.linspace(0, 1, n_samples)

        # Ray coordinates
        ray_x = source_pos[0] + (receiver_pos[0] - source_pos[0]) * t
        ray_y = source_pos[1] + (receiver_pos[1] - source_pos[1]) * t

        # Compute segment lengths
        dx_seg = np.diff(ray_x)
        dy_seg = np.diff(ray_y)
        ds = np.sqrt(dx_seg**2 + dy_seg**2)
        ds = np.concatenate(([ds[0]], ds))  # Length n_samples

        # Interpolate background velocity along ray
        from scipy.interpolate import RegularGridInterpolator
        interp_v = RegularGridInterpolator((self.x, self.y), background_velocity.T,
                                          bounds_error=False, fill_value=None)
        ray_points = np.stack([ray_x, ray_y], axis=1)
        v_on_ray = interp_v(ray_points)

        # Build source distribution on grid
        source_grid = np.zeros((self.ny, self.nx))

        for k in range(n_samples):
            rx, ry = ray_x[k], ray_y[k]

            # Find nearest grid cell
            ix = np.searchsorted(self.x, rx) - 1
            iy = np.searchsorted(self.y, ry) - 1

            # Clip to valid indices
            ix = np.clip(ix, 0, self.nx - 1)
            iy = np.clip(iy, 0, self.ny - 1)

            # Add contribution (ds / v^2) as density
            if not np.isnan(v_on_ray[k]) and v_on_ray[k] > 0:
                val = (ds[k] / (v_on_ray[k]**2)) / self.cell_area
                source_grid[iy, ix] += val

        # Solve (I - alpha * Laplacian) g = S for Riesz representer
        kernel = self._solve_riesz_representer(source_grid)

        return kernel

    def _compute_amplitude_kernel(self,
                                 source_pos: Tuple[float, float],
                                 receiver_pos: Tuple[float, float],
                                 background_velocity: np.ndarray) -> np.ndarray:
        """
        Compute amplitude sensitivity kernel (Fresnel zone approximation).
        """
        # For now, use same as travel time kernel but with different weighting
        # This is a simplified implementation
        kernel = self._compute_travel_time_kernel(source_pos, receiver_pos,
                                                 background_velocity)

        # Apply Fresnel zone weighting (simplified)
        distance = np.sqrt((receiver_pos[0] - source_pos[0])**2 +
                          (receiver_pos[1] - source_pos[1])**2)

        # Create Gaussian weighting around ray path
        ray_direction = np.array([receiver_pos[0] - source_pos[0],
                                 receiver_pos[1] - source_pos[1]]) / distance

        # Distance from each grid point to ray path
        for i in range(self.ny):
            for j in range(self.nx):
                point = np.array([self.x[j], self.y[i]])
                to_source = point - np.array(source_pos)

                # Project onto ray direction
                proj_length = np.dot(to_source, ray_direction)
                proj_point = np.array(source_pos) + proj_length * ray_direction

                # Distance from point to ray
                ray_distance = np.linalg.norm(point - proj_point)

                # Fresnel zone radius (simplified)
                fresnel_radius = np.sqrt(0.6 * proj_length *
                                       (distance - proj_length) / distance) if proj_length > 0 and proj_length < distance else 0

                if fresnel_radius > 0:
                    fresnel_weight = np.exp(-(ray_distance / fresnel_radius)**2)
                    kernel[i, j] *= fresnel_weight

        return kernel

    def _solve_riesz_representer(self, source_grid: np.ndarray) -> np.ndarray:
        """
        Solve the Riesz representer equation (I - alpha * Laplacian) g = S.
        """
        N = self.nx * self.ny

        # Build 2D Laplacian with Dirichlet boundary conditions
        rows, cols, vals = [], [], []

        def idx(i, j):
            return i * self.nx + j

        # Build sparse matrix
        for i in range(self.ny):
            for j in range(self.nx):
                p = idx(i, j)

                if i == 0 or i == self.ny-1 or j == 0 or j == self.nx-1:
                    # Dirichlet boundary: u = 0
                    rows.append(p)
                    cols.append(p)
                    vals.append(1.0)
                else:
                    # Interior: 5-point Laplacian
                    rows.extend([p, p, p, p, p])
                    cols.extend([p, idx(i-1, j), idx(i+1, j),
                               idx(i, j-1), idx(i, j+1)])
                    vals.extend([-4.0/(self.dx**2), 1.0/(self.dx**2),
                               1.0/(self.dx**2), 1.0/(self.dx**2), 1.0/(self.dx**2)])

        # Create sparse matrix
        A_lap = csr_matrix((vals, (rows, cols)), shape=(N, N))

        # Form system matrix: I - alpha * Laplacian
        I = csr_matrix((np.ones(N), (range(N), range(N))), shape=(N, N))
        A = I - self.regularization * A_lap

        # Prepare right-hand side
        S_flat = source_grid.ravel(order='C').astype(float)

        # Enforce boundary conditions on RHS
        for i in range(self.ny):
            for j in range(self.nx):
                p = idx(i, j)
                if i == 0 or i == self.ny-1 or j == 0 or j == self.nx-1:
                    S_flat[p] = 0.0

        # Solve sparse linear system
        g_flat = spsolve(A, S_flat)
        g = g_flat.reshape((self.ny, self.nx))

        return g

    def compute_multiple_kernels(self,
                                source_positions: List[Tuple[float, float]],
                                receiver_positions: List[Tuple[float, float]],
                                background_velocity: np.ndarray,
                                kernel_type: str = "travel_time") -> List[np.ndarray]:
        """
        Compute multiple sensitivity kernels.

        Parameters
        ----------
        source_positions : List[Tuple[float, float]]
            List of source positions
        receiver_positions : List[Tuple[float, float]]
            List of receiver positions
        background_velocity : np.ndarray
            Background velocity field
        kernel_type : str
            Type of kernel

        Returns
        -------
        kernels : List[np.ndarray]
            List of sensitivity kernels
        """
        kernels = []

        for src_pos, rcv_pos in zip(source_positions, receiver_positions):
            kernel = self.compute_ray_kernel(src_pos, rcv_pos,
                                           background_velocity, kernel_type)
            kernels.append(kernel)

        return kernels

    def stack_kernels(self, kernels: List[np.ndarray],
                     weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Stack multiple kernels with optional weighting.

        Parameters
        ----------
        kernels : List[np.ndarray]
            List of sensitivity kernels
        weights : Optional[List[float]]
            Optional weights for each kernel

        Returns
        -------
        stacked_kernel : np.ndarray
            Stacked sensitivity kernel
        """
        if weights is None:
            weights = [1.0] * len(kernels)

        if len(kernels) != len(weights):
            raise ValueError("Number of kernels and weights must match")

        stacked = np.zeros_like(kernels[0])

        for kernel, weight in zip(kernels, weights):
            stacked += weight * kernel

        return stacked / len(kernels)

    def compute_resolution_matrix(self,
                                 kernels: List[np.ndarray]) -> np.ndarray:
        """
        Compute resolution matrix for tomographic inversion.

        Parameters
        ----------
        kernels : List[np.ndarray]
            List of sensitivity kernels

        Returns
        -------
        resolution : np.ndarray
            Resolution matrix
        """
        # Flatten kernels into design matrix
        n_data = len(kernels)
        n_model = kernels[0].size

        G = np.zeros((n_data, n_model))
        for i, kernel in enumerate(kernels):
            G[i, :] = kernel.ravel()

        # Compute generalized inverse (with regularization)
        GTG = G.T @ G
        reg_matrix = self.regularization * np.eye(n_model)

        try:
            G_inv = np.linalg.solve(GTG + reg_matrix, G.T)
            resolution = G_inv @ G
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            resolution = np.linalg.pinv(G) @ G

        return resolution
