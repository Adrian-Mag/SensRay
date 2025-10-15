import numpy as np
from typing import Callable, Union

Number = Union[float, int]

def make_scalar_field(
    R: Callable[[np.ndarray], np.ndarray],
    T: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Callable[[Union[np.ndarray, list, tuple, Number], Union[Number, None], Union[Number, None]], float]:
    """
    Creates a scalar field f(x, y, z) = R(r) * T(theta, phi)

    Parameters
    ----------
    R : callable
        Radial function R(r). Example: lambda r: r**2 * np.exp(-r)
    T : callable
        Angular function T(theta, phi). Example: lambda theta, phi: np.cos(theta)

    Returns
    -------
    f : callable
        Function f(x, y, z) -> scalar (or array if x,y,z arrays are passed)
    """

    def f(x, y=None, z=None):
        # Allow both f([x,y,z]) and f(x,y,z)
        if y is None and z is None:
            x = np.asarray(x)
            if x.shape[-1] != 3:
                raise ValueError("Input must be [x, y, z] or three separate numbers.")
            X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        else:
            X = np.asarray(x)
            Y = np.asarray(y)
            Z = np.asarray(z)

        # Convert Cartesian to spherical
        r = np.sqrt(X**2 + Y**2 + Z**2)
        # avoid division by zero for theta
        theta = np.where(r == 0, 0.0, np.arccos(np.clip(Z / r, -1.0, 1.0)))
        phi = np.mod(np.arctan2(Y, X), 2*np.pi)

        return R(r) * T(theta, phi)

    return f


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Define R(r) and T(theta, phi)
    R = lambda r: r**2 * np.exp(-r)              # simple radial function
    T = lambda theta, phi: np.cos(theta)         # angular dependence

    f = make_scalar_field(R, T)

    # Single point
    val = f(1.0, 0.0, 0.0)
    print("f(1,0,0) =", val)

    # Point as a vector
    val_vec = f([0.0, 0.0, 1.0])
    print("f([0,0,1]) =", val_vec)

    # Multiple points
    pts = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    vals = f(pts)
    print("Multiple values:", vals)
