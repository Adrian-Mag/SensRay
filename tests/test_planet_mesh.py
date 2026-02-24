"""Tests for sensray.planet_mesh — spherical mesh, cell_volumes, kernel coefficients."""

import numpy as np
import pytest

from sensray import PlanetModel
from sensray.planet_mesh import PlanetMesh, SphericalPlanetMesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spherical_model():
    """Weber core model with a simple spherical mesh (4 boundary radii → 3 shells)."""
    model = PlanetModel.from_standard_model("weber_core")
    # Radii [0, 500, 1000, 1738] km → 3 concentric shells
    model.create_mesh(mesh_type="spherical", radii=[0.0, 500.0, 1000.0, 1738.0])
    return model


@pytest.fixture(scope="module")
def mesh(spherical_model):
    return spherical_model.mesh


# ---------------------------------------------------------------------------
# SphericalPlanetMesh basics
# ---------------------------------------------------------------------------

class TestSphericalPlanetMesh:
    def test_sorted_radii(self):
        """Constructor must sort radii ascending."""
        m = SphericalPlanetMesh(radii=[1738.0, 0.0, 500.0])
        assert m.radii == [0.0, 500.0, 1738.0]

    def test_n_cells(self):
        m = SphericalPlanetMesh(radii=[0.0, 500.0, 1000.0, 1738.0])
        assert m.n_cells == 3

    def test_n_points(self):
        m = SphericalPlanetMesh(radii=[0.0, 500.0, 1000.0, 1738.0])
        assert m.n_points == 4

    def test_cell_and_point_data_start_empty(self):
        m = SphericalPlanetMesh(radii=[0.0, 1738.0])
        assert m.cell_data == {}
        assert m.point_data == {}


# ---------------------------------------------------------------------------
# PlanetMesh.cell_volumes
# ---------------------------------------------------------------------------

class TestCellVolumes:
    def test_shape(self, mesh):
        vols = mesh.cell_volumes
        assert vols.ndim == 1
        assert vols.shape[0] == mesh.mesh.n_cells

    def test_all_positive(self, mesh):
        assert np.all(mesh.cell_volumes > 0)

    def test_single_shell_whole_moon(self):
        """A single shell 0→1738 km should have V = (4π/3) 1738³."""
        model = PlanetModel.from_standard_model("weber_core")
        model.create_mesh(mesh_type="spherical", radii=[0.0, 1738.0])
        vol = model.mesh.cell_volumes
        expected = (4.0 * np.pi / 3.0) * 1738.0 ** 3
        assert vol.shape == (1,)
        assert vol[0] == pytest.approx(expected, rel=1e-10)

    def test_sum_equals_total_planet_volume(self, mesh):
        """Sum of all shell volumes must equal the sphere volume for R=1738 km."""
        R = 1738.0
        expected_total = (4.0 * np.pi / 3.0) * R ** 3
        np.testing.assert_allclose(
            mesh.cell_volumes.sum(), expected_total, rtol=1e-10
        )

    def test_additivity(self):
        """Splitting a shell at an intermediate radius preserves total volume."""
        r_inner, r_mid, r_outer = 500.0, 1000.0, 1738.0
        model_split = PlanetModel.from_standard_model("weber_core")
        model_split.create_mesh(
            mesh_type="spherical", radii=[r_inner, r_mid, r_outer]
        )
        v1, v2 = model_split.mesh.cell_volumes

        model_whole = PlanetModel.from_standard_model("weber_core")
        model_whole.create_mesh(
            mesh_type="spherical", radii=[r_inner, r_outer]
        )
        v_whole = model_whole.mesh.cell_volumes[0]

        assert v1 + v2 == pytest.approx(v_whole, rel=1e-10)

    def test_no_mesh_raises(self):
        model = PlanetModel.from_standard_model("weber_core")
        pm = PlanetMesh(planet_model=model)  # no mesh generated yet
        with pytest.raises(RuntimeError, match="No mesh generated"):
            _ = pm.cell_volumes


# ---------------------------------------------------------------------------
# PlanetMesh.to_kernel_coefficients
# ---------------------------------------------------------------------------

class TestToKernelCoefficients:
    def test_1d_divides_by_volumes(self, mesh):
        """to_kernel_coefficients(ones) == 1 / cell_volumes."""
        vols = mesh.cell_volumes
        K_tilde = np.ones(mesh.mesh.n_cells)
        K_coeff = mesh.to_kernel_coefficients(K_tilde)
        np.testing.assert_allclose(K_coeff, 1.0 / vols, rtol=1e-12)

    def test_2d_divides_row_wise(self, mesh):
        """2D input (n_rays, n_cells): each row divided independently."""
        n_cells = mesh.mesh.n_cells
        vols = mesh.cell_volumes
        # Use two different scaling factors per row
        K_tilde = np.vstack([
            np.ones(n_cells),
            2.0 * np.ones(n_cells),
        ])
        K_coeff = mesh.to_kernel_coefficients(K_tilde)
        assert K_coeff.shape == (2, n_cells)
        np.testing.assert_allclose(K_coeff[0], 1.0 / vols, rtol=1e-12)
        np.testing.assert_allclose(K_coeff[1], 2.0 / vols, rtol=1e-12)

    def test_output_shape_1d(self, mesh):
        K = np.random.rand(mesh.mesh.n_cells)
        assert mesh.to_kernel_coefficients(K).shape == (mesh.mesh.n_cells,)

    def test_output_shape_2d(self, mesh):
        n = mesh.mesh.n_cells
        K = np.random.rand(5, n)
        assert mesh.to_kernel_coefficients(K).shape == (5, n)

    def test_bad_shape_raises(self, mesh):
        K = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="1D or 2D"):
            mesh.to_kernel_coefficients(K)

    def test_no_mesh_raises(self):
        model = PlanetModel.from_standard_model("weber_core")
        pm = PlanetMesh(planet_model=model)
        with pytest.raises(RuntimeError, match="No mesh generated"):
            pm.to_kernel_coefficients(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# populate_properties integration smoke test
# ---------------------------------------------------------------------------

class TestPopulateProperties:
    def test_populate_vp_shape(self, mesh):
        """After populate_properties, vp array length == n_cells."""
        mesh.populate_properties(["vp"])
        assert "vp" in mesh.mesh.cell_data
        assert len(mesh.mesh.cell_data["vp"]) == mesh.mesh.n_cells

    def test_populated_vp_values_positive(self, mesh):
        mesh.populate_properties(["vp"])
        assert np.all(mesh.mesh.cell_data["vp"] > 0)
