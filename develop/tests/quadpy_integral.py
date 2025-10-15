import numpy as np
import pygmsh
import quadpy

def integrate_over_tetrahedra(f, tetra_vertices, scheme_order=5):
    scheme = quadpy.t3.get_good_scheme(scheme_order)
    element_integrals = scheme.integrate(f, tetra_vertices)   # array of shape (n_tetra,)
    total_integral = element_integrals.sum()
    return total_integral

if __name__ == "__main__":
    # ------------------------------------------------------
    # 1. Create a 3D mesh using pygmsh (unit cube)
    # ------------------------------------------------------
    with pygmsh.geo.Geometry() as geom:
        box = geom.add_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        mesh = geom.generate_mesh()

    # ------------------------------------------------------
    # 2. Extract points and tetrahedral cells
    # ------------------------------------------------------
    points = mesh.points
    cells = mesh.get_cells_type("tetra")

    # ------------------------------------------------------
    # 3. Build tetrahedra in quadpy format (4, 3, n_tetra)
    # ------------------------------------------------------
    tetra_vert = points[cells]                        # (n_tetra, 4, 3)
    tetra_vert = np.transpose(tetra_vert, (1, 2, 0))  # (4, 3, n_tetra)

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])  # shape (4, 3)

    # ------------------------------------------------------
    # 4. Define smooth function to integrate
    # ------------------------------------------------------
    def f(x):
        return np.sin(x[0]) + x[1]**2 + np.exp(x[2])

    # ------------------------------------------------------
    # 6. Integrate over all tetrahedra (vectorized)
    # ------------------------------------------------------
    print("Integrating over tetrahedra...")
    total_integral = integrate_over_tetrahedra(f, vertices, scheme_order=5)
    print(f"Total integral over mesh: {total_integral}")
    print("Integrating over tetrahedra 2...")
    total_integral_2 = integrate_over_tetrahedra(f, tetra_vert, scheme_order=5)
    print(f"Total integral over mesh 2: {total_integral_2}")