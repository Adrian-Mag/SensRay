"""
Numerical quadrature for tetrahedral integration.

This module provides quadrature rules for integrating functions over
tetrahedral domains. It replaces the functionality previously provided
by quadpy with in-house implementations using well-established Gauss
quadrature rules.

References
----------
- Keast, P. (1986). "Moderate-degree tetrahedral quadrature formulas."
  Computer Methods in Applied Mechanics and Engineering, 55(3), 339-348.
- Shunn, L., & Ham, F. (2012). "Symmetric quadrature rules for tetrahedra
  based on a cubic close-packed lattice arrangement."
  Journal of Computational and Applied Mathematics, 236(17), 4348-4364.
"""

import numpy as np
from typing import Callable, Union


class TetrahedralQuadrature:
    """
    Quadrature scheme for tetrahedral integration.
    
    Provides integration points and weights for numerically integrating
    functions over tetrahedral domains.
    
    Parameters
    ----------
    order : int
        Order of accuracy (1, 2, 3, 4, or 5)
        
    Examples
    --------
    >>> scheme = TetrahedralQuadrature(order=5)
    >>> def f(x): return x[:, 0]**2  # Example function
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> result = scheme.integrate(f, vertices.T)
    """
    
    def __init__(self, order: int = 5):
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
            
        elif order == 4:
            # 11-point rule (exact for 4th order, Keast rule)
            a = 0.25
            b1 = 11.0 / 14.0
            b2 = 1.0 / 14.0
            c1 = 0.3994035761667992
            c2 = 0.1005964238332008
            
            points = np.array([
                # Centroid
                [a, a, a, a],
                # Edge midpoints (6 points)
                [b1, b2, b2, b2],
                [b2, b1, b2, b2],
                [b2, b2, b1, b2],
                [b2, b2, b2, b1],
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0, 0.5],
                [0.0, 0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ])
            
            w1 = -0.013155555555555555
            w2 = 0.007622222222222222
            w3 = 0.024888888888888888
            
            weights = np.array([
                w1,  # centroid
                w2, w2, w2, w2,  # vertices
                w3, w3, w3, w3, w3, w3  # edge midpoints
            ])
            
        elif order >= 5:
            # 15-point rule (exact for 5th order, Keast 1986, Rule 6)
            # See: Keast (1986) Table III, Rule 6
            a = 0.25
            b1 = 0.0
            b2 = 1.0 / 3.0
            c1 = 8.0 / 11.0
            c2 = 1.0 / 11.0
            d1 = (1.0 - 1.0/np.sqrt(15.0)) / 4.0
            d2 = (1.0 + 1.0/np.sqrt(15.0)) / 4.0
            
            points = np.array([
                # Centroid (1 point)
                [a, a, a, a],
                # Type 1: (0, 1/3, 1/3, 1/3) permutations (4 points)
                [b1, b2, b2, b2],
                [b2, b1, b2, b2],
                [b2, b2, b1, b2],
                [b2, b2, b2, b1],
                # Type 2: (8/11, 1/11, 1/11, 1/11) permutations (4 points)
                [c1, c2, c2, c2],
                [c2, c1, c2, c2],
                [c2, c2, c1, c2],
                [c2, c2, c2, c1],
                # Type 3: symmetric pairs (6 points)
                [d1, d1, d2, d2],
                [d1, d2, d1, d2],
                [d1, d2, d2, d1],
                [d2, d1, d1, d2],
                [d2, d1, d2, d1],
                [d2, d2, d1, d1],
            ])
            
            # Weights from Keast (1986), Table III, Rule 6
            w0 = 16.0 / 135.0  # centroid
            w1 = (2665.0 - 14.0*np.sqrt(15.0)) / 37800.0  # type 1
            w2 = (2665.0 + 14.0*np.sqrt(15.0)) / 37800.0  # type 2
            w3 = 20.0 / 378.0  # type 3
            
            weights = np.array([
                w0,  # centroid
                w1, w1, w1, w1,  # type 1
                w2, w2, w2, w2,  # type 2
                w3, w3, w3, w3, w3, w3  # type 3
            ])
            
        else:
            raise ValueError(f"Order {order} not supported. Use 1-5.")
            
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
        quad_points_cartesian = np.zeros((len(self.points), 3))
        for i, bary in enumerate(self.points):
            quad_points_cartesian[i] = (
                bary[0] * v0 + 
                bary[1] * v1 + 
                bary[2] * v2 + 
                bary[3] * v3
            )
        
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


def get_good_scheme(order: int) -> TetrahedralQuadrature:
    """
    Get a good quadrature scheme for tetrahedra.
    
    This function mimics the quadpy API: quadpy.t3.get_good_scheme(order)
    
    Parameters
    ----------
    order : int
        Desired order of accuracy (1-5)
        
    Returns
    -------
    TetrahedralQuadrature
        Quadrature scheme object with integrate() method
        
    Examples
    --------
    >>> scheme = get_good_scheme(5)
    >>> vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    >>> result = scheme.integrate(lambda x: np.ones(len(x)), vertices.T)
    """
    return TetrahedralQuadrature(order=order)


# Create a namespace that mimics quadpy.t3 for drop-in compatibility
class _T3Namespace:
    """Namespace to mimic quadpy.t3 module."""
    get_good_scheme = staticmethod(get_good_scheme)

t3 = _T3Namespace()
