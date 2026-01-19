"""
Numerical quadrature for tetrahedral integration.

This module provides quadrature rules for integrating functions over
tetrahedral domains. It includes in-house implementations using well-established 
Gauss quadrature rules, with optional fallback to quadpy if available.

References
----------
- Keast, P. (1986). "Moderate-degree tetrahedral quadrature formulas."
  Computer Methods in Applied Mechanics and Engineering, 55(3), 339-348.
- Shunn, L., & Ham, F. (2012). "Symmetric quadrature rules for tetrahedra
  based on a cubic close-packed lattice arrangement."
  Journal of Computational and Applied Mathematics, 236(17), 4348-4564.
"""

import numpy as np
from typing import Callable, Union
import os

# Try to import quadpy (optional)
_QUADPY_AVAILABLE = False
try:
    import quadpy
    _QUADPY_AVAILABLE = True
except ImportError:
    quadpy = None

# Default to in-house implementation
_USE_QUADPY = os.environ.get('SENSRAY_USE_QUADPY', 'false').lower() == 'true'


class TetrahedralQuadrature:
    """
    Quadrature scheme for tetrahedral integration.

    Provides integration points and weights for numerically integrating
    functions over tetrahedral domains.

    Parameters
    ----------
    order : int
        Order of accuracy (1, 2, or 3)

    Examples
    --------
    >>> scheme = TetrahedralQuadrature(order=3)
    >>> def f(x): return x[:, 0]**2  # Example function
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> result = scheme.integrate(f, vertices.T)
    """

    def __init__(self, order: int = 3):
        """Initialize quadrature scheme with specified order."""
        self.order = order
        self.points, self.weights = self._get_quadrature_rule(order)

    def _get_quadrature_rule(self, order: int) -> tuple:
        """
        Get quadrature points and weights for specified order.

        Returns points in barycentric coordinates and corresponding weights.

        Parameters
        ----------
        order : int
            Order of accuracy

        Returns
        -------
        points : ndarray, shape (n_points, 4)
            Quadrature points in barycentric coordinates
        weights : ndarray, shape (n_points,)
            Quadrature weights (sum to 1.0)
        """
        if order <= 1:
            # 1-point rule (centroid, exact for linear)
            points = np.array([[0.25, 0.25, 0.25, 0.25]])
            weights = np.array([1.0])

        elif order == 2:
            # 4-point rule (exact for quadratic)
            # Vertices at (a, b, b, b) and permutations
            a = 0.5854101966249685
            b = 0.1381966011250105
            points = np.array([
                [a, b, b, b],
                [b, a, b, b],
                [b, b, a, b],
                [b, b, b, a]
            ])
            weights = np.array([0.25, 0.25, 0.25, 0.25])

        elif order == 3:
            # 5-point rule (exact for cubic)
            points = np.array([
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 1/6, 1/6, 1/6],
                [1/6, 0.5, 1/6, 1/6],
                [1/6, 1/6, 0.5, 1/6],
                [1/6, 1/6, 1/6, 0.5]
            ])
            weights = np.array([-0.8, 0.45, 0.45, 0.45, 0.45])

        else:
            raise ValueError(f"Order {order} not supported. Use 1, 2, or 3.")

        return points, weights

    def integrate(
        self,
        func: Callable[[np.ndarray], Union[float, np.ndarray]],
        vertices: np.ndarray
    ) -> float:
        """
        Integrate function over a tetrahedron.

        Parameters
        ----------
        func : callable
            Function to integrate. Should accept array of shape (n_points, 3)
            and return array of shape (n_points,) or scalar
        vertices : ndarray
            Tetrahedron vertices. Can be either:
            - Shape (4, 3): standard format [v0, v1, v2, v3]
            - Shape (3, 4): transposed format (quadpy convention)

        Returns
        -------
        float
            Integrated value over the tetrahedron

        Examples
        --------
        >>> scheme = TetrahedralQuadrature(5)
        >>> vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        >>> result = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
        """
        # Handle both (4, 3) and (3, 4) vertex formats
        if vertices.shape == (3, 4):
            # Transpose to (4, 3) format
            vertices = vertices.T
        elif vertices.shape != (4, 3):
            raise ValueError(
                f"Vertices must have shape (4, 3) or (3, 4), got {vertices.shape}"
            )

        # Extract vertices
        v0, v1, v2, v3 = vertices

        # Compute volume using cross product
        # V = |det([v1-v0, v2-v0, v3-v0])| / 6
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0
        volume = abs(np.dot(np.cross(edge1, edge2), edge3)) / 6.0

        # Transform barycentric coordinates to Cartesian
        # x = λ0*v0 + λ1*v1 + λ2*v2 + λ3*v3
        # Vectorized: quad_points_cartesian = self.points @ vertices
        quad_points_cartesian = self.points.dot(vertices)

        # Evaluate function at quadrature points
        func_values = func(quad_points_cartesian)

        # Ensure func_values is array-like
        if np.isscalar(func_values):
            func_values = np.array([func_values])
        else:
            func_values = np.asarray(func_values)

        # Compute weighted sum and scale by actual volume
        # Quadrature weights are normalized such that they sum to 1
        # For a tetrahedron, we need to scale by the volume
        integral = volume * np.sum(self.weights * func_values)

        return float(integral)


def get_good_scheme(order: int, use_quadpy: bool = None) -> TetrahedralQuadrature:
    """
    Get a good quadrature scheme for tetrahedra.

    This function mimics the quadpy API: quadpy.t3.get_good_scheme(order)

    Parameters
    ----------
    order : int
        Desired order of accuracy (1-3 for in-house, 1-10+ for quadpy)
    use_quadpy : bool, optional
        If True, use quadpy if available. If False, use in-house implementation.
        If None (default), uses environment variable SENSRAY_USE_QUADPY or 
        defaults to in-house implementation.
        Note: quadpy uses different vertex ordering - see quadpy documentation.

    Returns
    -------
    TetrahedralQuadrature or quadpy scheme
        Quadrature scheme object with integrate() method

    Raises
    ------
    ImportError
        If use_quadpy=True but quadpy is not installed
    ValueError
        If order not supported by the chosen implementation

    Examples
    --------
    >>> # Use in-house implementation (default)
    >>> scheme = get_good_scheme(3)
    >>> vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    >>> result = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
    
    >>> # Explicitly use quadpy if available
    >>> scheme = get_good_scheme(5, use_quadpy=True)
    
    Notes
    -----
    Set environment variable SENSRAY_USE_QUADPY=true to default to quadpy.
    """
    # Determine which implementation to use
    if use_quadpy is None:
        use_quadpy = _USE_QUADPY and _QUADPY_AVAILABLE
    
    if use_quadpy:
        if not _QUADPY_AVAILABLE:
            raise ImportError(
                "quadpy is not installed. Install with 'pip install quadpy' "
                "or use the in-house implementation (use_quadpy=False)."
            )
        return quadpy.t3.get_good_scheme(order)
    else:
        return TetrahedralQuadrature(order=order)


# Create a namespace that mimics quadpy.t3 for drop-in compatibility
class _T3Namespace:
    """Namespace to mimic quadpy.t3 module."""
    get_good_scheme = staticmethod(get_good_scheme)

t3 = _T3Namespace()
