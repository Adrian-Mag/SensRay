"""Minimal spherical cross-section plotting.

Defines a function f(x,y,z) that depends on radius r and plots
the function values along a chosen great circle on the sphere of radius R.

Outputs:
- sphere_cross_section_3d.png : small 3D scatter of sampled points on the sphere, colored by f
- sphere_cross_section_profile.png : 1D plot of f along the great circle

This file is intentionally minimal and focused on the request.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable
matplotlib.use("TkAgg")


def f_xyz(x, y, z):
    """Example scalar function of (x,y,z) that depends on radius r.

    You can replace this with any f(x,y,z). Here we use a simple radial
    attenuation times a directional factor:
        f = exp(-((r-R0)/sigma)^2) * (cos(theta) + 0.2*sin(3*phi))
    where r = sqrt(x^2+y^2+z^2).
    """
    r = np.sqrt(x * x + y * y + z * z)
    # spherical angles
    theta = np.arccos(np.clip(z / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(y, x)
    # parameters
    R0 = 1.0
    sigma = 0.25
    radial = np.exp(-((r - R0) ** 2) / (2 * sigma * sigma))
    directional = np.cos(theta) + 0.2 * np.sin(2 * phi)
    return radial * directional


def sample_circle_on_sphere(R, normal, n_points=50):
    normal = normal / np.linalg.norm(normal)
    # pick a vector not parallel to normal
    a = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, normal)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    s = np.linspace(-R, R, n_points)
    pts = []
    for si in s:
        for sj in s:
            if si * si + sj * sj <= R * R:
                pts.append(u * si + v * sj)

    return pts


def plot_on_sphere_cross_section(
    R: float,
    normal: np.ndarray,
    func: Callable,
    cmap: str = 'viridis',
    out: str = 'sphere_cross_section_profile.png',
    show: bool = False
):

    # Sample points on the great circle
    points = sample_circle_on_sphere(R, normal)
    points = np.array(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Evaluate f at these points
    f_values = func(x, y, z)

    # 3D scatter plot of the sampled points on the sphere
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=f_values, cmap=cmap)
    plt.colorbar(sc, ax=ax, label='f(x,y,z)')
    ax.set_title('3D Scatter of f on Sphere Cross-Section')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    if show:
        plt.show()
    else:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_on_sphere(
    R: float,
    function: Callable,
    n_theta: int = 120,
    n_phi: int = 240,
    cmap: str = 'viridis',
    out: str = 'sphere_surface.png',
    show: bool = False
):
    """
    Render f(x,y,z) on the spherical surface r=R.

    If show is True the figure will be displayed interactively (plt.show())
    and NOT saved. If show is False the figure will be saved to `out` and
    closed.

    f_xyz: function f(x,y,z) -> scalar (should accept numpy arrays)
    """
    theta = np.linspace(0.0, np.pi, n_theta)        # colatitude: 0..pi
    phi = np.linspace(0.0, 2*np.pi, n_phi)          # longitude: 0..2pi
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    # Evaluate function (vectorized) on the grid
    vals = function(X, Y, Z)

    # Normalize to [0,1] for facecolors (avoid division by zero)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    norm = (vals - vmin) / (vmax - vmin + 1e-15)
    facecolors = plt.cm.get_cmap(cmap)(norm)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=False
    )
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    if show:
        # Interactive display requested; do not save, leave control to caller
        plt.show()
    else:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    normal = np.array([1.0, 0.0, 1.0])  # normal vector of the great circle plane
    plot_on_sphere(
        1.0,
        f_xyz,
        n_theta=20,
        n_phi=40,
        cmap='viridis',
        out='sphere_surface.png',
        show=True
    )

    plot_on_sphere_cross_section(
        1.0,
        normal,
        f_xyz,
        cmap='viridis',
        out='sphere_cross_section_profile.png',
        show=True
    )