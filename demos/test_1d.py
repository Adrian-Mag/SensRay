import numpy as np
import warnings
from typing import Callable, Tuple
import numpy as np
from sensray import PlanetModel
from fwdop import GFwdOp, make_scalar_field
from pygeoinf.linear_solvers import LUSolver, CholeskySolver
from ray_and_point_generation import get_rays, fibonacci_sphere_points
from itertools import product
from random import randint
import math
from itertools import chain
from obspy.geodetics import locations2degrees

try:
    from scipy import integrate
except Exception as e:
    raise ImportError("scipy is required: pip install scipy") from e


model = PlanetModel.from_standard_model('M1')

r_max = model.radius
r_min = 0
n_shells = 10
shell_radii = np.linspace(r_max, r_min, n_shells + 1)
print(shell_radii)

# ---------------------------------------------------------


FOUR_PI = 4.0 * np.pi
THREE = 3.0

def _gauss_legendre_shell_integrals(
    f_vec: Callable[[np.ndarray], np.ndarray],
    radii: np.ndarray,
    npts: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Gauss-Legendre evaluation of I_j = ∫_{r_j}^{r_{j+1}} f(r) r^2 dr
    Returns (I_j array, denom array where denom = r_{j+1}^3 - r_j^3)
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("radii must be a 1D array of length >= 2")

    xi, wi = np.polynomial.legendre.leggauss(npts)  # nodes & weights on [-1,1]
    rL = radii[:-1]
    rR = radii[1:]
    J = 0.5 * (rR - rL)       # mapping scale
    r_mid = 0.5 * (rR + rL)

    # r_eval shape (N_shells, npts)
    r_eval = r_mid[:, None] + J[:, None] * xi[None, :]

    # evaluate f at all points (expects vectorized f)
    fvals = f_vec(r_eval)
    fvals = np.asarray(fvals)
    if fvals.shape != r_eval.shape:
        # try broadcast shapes (npts,) -> (N, npts) etc.
        try:
            fvals = np.broadcast_to(fvals, r_eval.shape)
        except Exception:
            raise ValueError("f_vec did not return an array of shape matching r_eval")

    # integral per shell
    weighted = (wi[None, :] * fvals * (r_eval ** 2)) * J[:, None]
    integrals = np.sum(weighted, axis=1)  # ∫ f(r) r^2 dr over each shell (no 4π factor)
    denom = (rR ** 3) - (rL ** 3)
    return integrals, denom


def _quad_shell_integrals(
    f_scalar: Callable[[float], float],
    radii: np.ndarray,
    epsabs: float = 1e-9,
    epsrel: float = 1e-9,
    limit: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive quad per shell. f_scalar should accept float and return float.
    Returns (integrals, denom) where integrals are ∫ f(r) r^2 dr.
    """
    radii = np.asarray(model.mesh.radii, dtype=float)
    rL = radii[:-1]
    rR = radii[1:]
    N = rL.size
    integrals = np.empty(N, dtype=float)

    for i in range(N):
        a = rL[i]
        b = rR[i]
        if b <= a:
            integrals[i] = 0.0
            continue
        val, err = integrate.quad(lambda r: float(f_scalar(r)) * (r ** 2), a, b,
                                  epsabs=epsabs, epsrel=epsrel, limit=limit)
        integrals[i] = val

    denom = (rR ** 3) - (rL ** 3)
    return integrals, denom


# === Method to include in your class ===
def project_to_spherical_shells(
    self,
    function: Callable,
    property_name: str,
    method: str = "gauss",  # 'gauss' | 'quad'
    npts: int = 12,
    quad_opts: dict = None,
):
    """
    Project a radial function onto 1D spherical shells using formula:
      f_j = 3 * ∫_{r_j}^{r_{j+1}} f(r) r^2 dr / (r_{j+1}^3 - r_j^3)

    - function: callable, ideally vectorized f(r) -> shape-matching array
    - property_name: name to store in shell_grid.cell_data[property_name]
    - radii: optional array of radii length N+1; if None, tries self.shell_radii
    - method: 'gauss' for vectorized Gauss-Legendre, 'quad' for scipy.quad
    - npts: Gauss points for 'gauss' (or fixed_quad fallback)
    - quad_opts: dict passed to _quad_shell_integrals (epsabs, epsrel, limit)
    """
    # 1) Compute integrals per shell ∫ f(r) r^2 dr and denom = r_{j+1}^3 - r_j^3
    if method == "gauss":
        # prefer vectorized evaluation
        try:
            integrals, denom = _gauss_legendre_shell_integrals(function, self.mesh.radii, npts=npts)
        except Exception as e:
            # if vectorized evaluation fails, fall back to scalar quad with warning
            warnings.warn(f"Gauss vectorized evaluation failed ({e}); falling back to adaptive quad.", UserWarning)
            quad_opts = quad_opts or {}
            integrals, denom = _quad_shell_integrals(function, self.mesh.radii, **quad_opts)
    elif method == "quad":
        quad_opts = quad_opts or {}
        integrals, denom = _quad_shell_integrals(function, self.mesh.radii, **quad_opts)
    else:
        raise ValueError("method must be 'gauss' or 'quad'")

    # 2) compute shell-averaged f_j
    # handle degenerate denom (zero-volume shells)
    small = denom == 0.0
    f_j = np.empty_like(denom, dtype=float)
    # formula: f_j = 3 * integral / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        f_j[~small] = THREE * integrals[~small] / denom[~small]

    if np.any(small):
        # For zero-thickness shells fallback to midpoint value of function
        mids = 0.5 * (self.mesh.radii[:-1][small] + self.mesh.radii[1:][small])
        # attempt vectorized call; if it fails fall back to scalar
        try:
            f_j[small] = np.asarray(function(mids))
        except Exception:
            f_j[small] = np.array([function(float(m)) for m in mids])

    '''
    # 4) Optionally compute volumes (useful if you want cell volumes like tetra code)
    volumes = FOUR_PI * (denom / 3.0)  # since V_shell = 4π/3 (rR^3 - rL^3)

    # 5) Store into grid-like object similar to your tetra code:
    #    try to find self.shell_grid (preferred), else self.grid_shells, else attach attribute
    target_grid = None
    for attr in ("shell_grid", "shells", "shell_grid_obj", "grid_shells"):
        if hasattr(self, attr):
            target_grid = getattr(self, attr)
            break

    if target_grid is None:
        # fallback: attach to 'self' as a simple container with cell_data dict
        # Create a minimal object to hold cell_data if needed
        class _Tmp:
            pass
        target_grid = getattr(self, "shell_grid", None)
        if target_grid is None:
            tmp = _Tmp()
            tmp.cell_data = {}
            target_grid = tmp
            # attach back so subsequent calls reuse it
            self.shell_grid = target_grid
    '''
    # store arrays as float32 to match your tetra code
    # f_j has length N (number of shells)
    model.mesh.cell_data[property_name] = np.asarray(f_j, dtype=np.float32)
    '''
    # optionally also store volumes and denom if desired
    target_grid.cell_data[property_name + "_volume"] = np.asarray(volumes, dtype=np.float64)
    target_grid.cell_data[property_name + "_denom"] = np.asarray(denom, dtype=np.float64)

    # return computed arrays for immediate use
    return f_j, denom, volumes
    '''


# Generate sources and receivers
setup_info = {
    "source": {"N": 5, "min depth": 150, "max depth": 150},
    "receiver": {"N": 5, "min depth": 0, "max depth": 0},
}
depth = randint(setup_info["source"]["min depth"], setup_info["source"]["max depth"])
sources = fibonacci_sphere_points(setup_info["source"]["N"], radius=model.radius-depth, latlon=True)  # 20 sources at 150km depth
receivers = fibonacci_sphere_points(setup_info["source"]["N"], radius=model.radius, latlon=True)  # 20 stations on Earth radius
phases = ["P"]

srr = get_rays(model=model, srp=product(sources, receivers, phases), radius=True)
print(srr[:, 2])

G = GFwdOp(srr[:,2])

# Generate different models and calculate dv
functions = {
    "radial": {"R": lambda r: r, "T": lambda theta, phi: np.ones_like(theta)},
    "simple": {"R": lambda r: np.ones_like(r), "T": lambda theta, phi: np.ones_like(theta)},
    "complex": {"R": lambda r: r**2 * np.exp(-r/100000), "T": lambda theta, phi: np.cos(theta)},
    "harmonic": {"R": lambda r: 0.1 * model.get_property_at_radius(radius=r, property_name="vp"), "T": lambda theta, phi: 0.5 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)},
}

func = "radial"
f = make_scalar_field(functions[func]["R"], functions[func]["T"])

model.mesh.project_function_on_mesh(f, property_name="dv")
print("Cell data 'dv':", model.mesh.mesh.cell_data["dv"])

travel_times = G(model.mesh.mesh.cell_data["dv"])
print(travel_times)

# Inverse in a single operation
M_tilde = (G.adjoint@LUSolver()(G@G.adjoint))(G(model.mesh.mesh.cell_data["dv"]))
model.mesh.mesh.cell_data["solution"] = M_tilde
print(M_tilde)

print("Solution visualization...")
plotter1 = model.mesh.plot_cross_section(plane_normal=(0, 1, 0), property_name="solution")

plotter1.camera.position = (8000, 6000, 10000)

plotter1.show()


# ----------------------------------------------------
# Shell marching kernel
# ----------------------------------------------------
def find_shell_index(r, shell_radii):
    for i in range(len(shell_radii) - 1):
        if shell_radii[i] >= r > shell_radii[i + 1]:
            return i
    return None

def interpolate_t(r0, r1, target):
    if r1 == r0:
        return 0.0
    return (target - r0) / (r1 - r0)

def compute_shell_contributions(r0, th0, r1, th1, contrib):
    shell0 = find_shell_index(r0, shell_radii)
    shell1 = find_shell_index(r1, shell_radii)

    if shell0 is None or shell1 is None:
        return

    L_full = np.sqrt((r1 - r0)**2 + (model.radius * (th1 - th0))**2)

    if shell0 == shell1:
        contrib[shell0] += L_full
        return

    direction = int(np.sign(r0 - r1))
    boundaries = []

    if direction > 0:  # inward
        for rad in shell_radii[shell0 + 1:shell1 + 1]:
            boundaries.append(rad)
    else:  # outward
        for rad in shell_radii[shell1:shell0]:
            boundaries.append(rad)

    boundaries.append(r1)

    prev_r, prev_th = r0, th0
    current_shell = shell0

    for boundary_r in boundaries:
        t = np.clip(interpolate_t(r0, r1, boundary_r), 0, 1)
        interp_r = r0 + (r1 - r0) * t
        interp_th = th0 + (th1 - th0) * t

        L = np.sqrt((interp_r - prev_r)**2 +
                    (model.radius * (interp_th - prev_th))**2)

        contrib[current_shell] += L

        prev_r, prev_th = interp_r, interp_th
        current_shell += direction


lengths = np.zeros(len(shell_radii) - 1)
for i in range(len(radius) - 1):
        compute_shell_contributions(radius[i], dist[i],
                                    radius[i + 1], dist[i + 1], lengths)