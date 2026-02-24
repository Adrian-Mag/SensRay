"""Tests for sensray.planet_model — parsing, interpolation, discontinuities."""

import numpy as np
import pytest

from sensray import PlanetModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def weber_core():
    """Weber core model with known, simple step-function velocity profile."""
    return PlanetModel.from_standard_model("weber_core")


@pytest.fixture(scope="module")
def prem():
    return PlanetModel.from_standard_model("prem")


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_from_standard_model_loads(self, prem):
        """PlanetModel.from_standard_model returns a PlanetModel instance."""
        assert isinstance(prem, PlanetModel)

    def test_from_standard_model_unknown_raises(self):
        with pytest.raises(ValueError, match="not found"):
            PlanetModel.from_standard_model("__nonexistent_model__")

    def test_weber_core_radius(self, weber_core):
        """Weber core planet radius is read from the last depth entry = 1738 km."""
        assert weber_core.radius == pytest.approx(1738.0)

    def test_layers_are_populated(self, weber_core):
        """Parser must produce at least one named layer."""
        assert len(weber_core.layers) >= 1

    def test_list_standard_models_returns_list(self):
        models = PlanetModel.list_standard_models()
        assert isinstance(models, list)
        assert "weber_core" in models
        assert "prem" in models


# ---------------------------------------------------------------------------
# Property interpolation — weber_core.nd has a simple step-function profile
#
# Layer structure (depth-ascending):
#   "surface" : 0–3 km  Vp=1.00  Vs=0.50  rho=2.585
#                3–15 km  Vp=3.20  Vs=1.80  rho=2.679
#               15–38 km  Vp=5.50  Vs=3.20  rho=2.830
#   "mantle"  : 38–238 km Vp=7.65  Vs=4.44  rho=3.321
#   "outer-core": 1408–1498 km  Vp=4.10  Vs=0.00  rho=5.094
#   "inner-core": 1498–1738 km  Vp=4.30  Vs=2.20  rho=8.000
# ---------------------------------------------------------------------------

class TestPropertyAtDepth:
    @pytest.mark.parametrize("depth,prop,expected", [
        (0.0,   "vp",  1.00),   # surface at depth=0
        (0.0,   "vs",  0.50),
        (0.0,   "rho", 2.585),
        (1.5,   "vp",  1.00),   # interior of 0–3 km constant layer
        (1.5,   "vs",  0.50),
        (10.0,  "vp",  3.20),   # interior of 3–15 km constant layer
        (10.0,  "vs",  1.80),
        (25.0,  "vp",  5.50),   # interior of 15–38 km constant layer
        (100.0, "vp",  7.65),   # interior of mantle 38–238 km constant layer
        (1450.0,"vp",  4.10),   # outer-core
        (1450.0,"vs",  0.00),
        (1600.0,"vp",  4.30),   # inner-core
        (1600.0,"vs",  2.20),
        (1738.0,"vp",  4.30),   # centre of the moon
    ])
    def test_get_property_at_depth_constant_layers(
        self, weber_core, depth, prop, expected
    ):
        result = weber_core.get_property_at_depth(prop, depth)
        assert result == pytest.approx(expected, rel=1e-4), (
            f"get_property_at_depth({prop!r}, {depth}) = {result}, expected {expected}"
        )

    def test_vectorised_query(self, weber_core):
        """Passing an array returns an array of same length."""
        depths = np.array([1.5, 10.0, 100.0])
        result = weber_core.get_property_at_depth("vp", depths)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [1.0, 3.2, 7.65], rtol=1e-4)

    def test_out_of_range_raises(self, weber_core):
        with pytest.raises(ValueError):
            weber_core.get_property_at_depth("vp", -1.0)
        with pytest.raises(ValueError):
            weber_core.get_property_at_depth("vp", 99999.0)


class TestPropertyAtRadius:
    def test_radius_matches_depth(self, weber_core):
        """get_property_at_radius(r) == get_property_at_depth(R - r)."""
        R = weber_core.radius
        for r, expected_vp in [
            (R,       1.00),   # surface (depth 0)
            (R - 1.5, 1.00),   # depth 1.5
            (R - 10,  3.20),   # depth 10
            (0.0,     4.30),   # centre (depth = R)
        ]:
            result = weber_core.get_property_at_radius("vp", r)
            assert result == pytest.approx(expected_vp, rel=1e-4), (
                f"radius={r}: got {result}, expected {expected_vp}"
            )

    def test_radius_vectorised(self, weber_core):
        R = weber_core.radius
        radii = np.array([R, R - 10.0, 0.0])
        result = weber_core.get_property_at_radius("vp", radii)
        np.testing.assert_allclose(result, [1.0, 3.2, 4.3], rtol=1e-4)


# ---------------------------------------------------------------------------
# get_property_profile
# ---------------------------------------------------------------------------

class TestGetPropertyProfile:
    def test_single_property_keys(self, weber_core):
        profile = weber_core.get_property_profile("vp")
        assert "radius" in profile
        assert "vp" in profile

    def test_profile_arrays_same_length(self, weber_core):
        profile = weber_core.get_property_profile(["vp", "vs"])
        assert profile["radius"].shape == profile["vp"].shape
        assert profile["radius"].shape == profile["vs"].shape

    def test_profile_asradius_false(self, weber_core):
        profile = weber_core.get_property_profile("vp", asradius=False)
        assert "depth" in profile
        assert "vp" in profile

    def test_profile_surface_value(self, weber_core):
        """First point in outwards-sorted profile should be deepest (radius=0)."""
        profile = weber_core.get_property_profile("vp")
        radii = profile["radius"]
        vp = profile["vp"]
        # Check that radius 0 maps to inner-core Vp ~ 4.30
        idx = np.argmin(radii)
        assert vp[idx] == pytest.approx(4.30, rel=1e-3)


# ---------------------------------------------------------------------------
# get_discontinuities
# ---------------------------------------------------------------------------

class TestGetDiscontinuities:
    def test_returns_dict(self, weber_core):
        disc = weber_core.get_discontinuities()
        assert isinstance(disc, dict)

    def test_expected_layer_names_present(self, weber_core):
        """Weber core has surface / mantle / outer-core / inner-core layers."""
        disc = weber_core.get_discontinuities(include_radius=True)
        for expected_key in ("surface", "mantle", "outer-core", "inner-core"):
            assert expected_key in disc, f"Missing key: {expected_key!r}"

    def test_each_entry_has_required_fields(self, weber_core):
        disc = weber_core.get_discontinuities()
        for name, entry in disc.items():
            for field in ("depth", "radius", "upper", "lower"):
                assert field in entry, f"Entry {name!r} missing field {field!r}"

    def test_depth_radius_sum_to_planet_radius(self, weber_core):
        """depth + radius == planet radius for every discontinuity."""
        R = weber_core.radius
        disc = weber_core.get_discontinuities(include_radius=True)
        for name, entry in disc.items():
            total = entry["depth"] + entry["radius"]
            assert total == pytest.approx(R, rel=1e-6), (
                f"Layer {name!r}: depth+radius={total} != {R}"
            )

    def test_inner_core_boundary_depth(self, weber_core):
        """Inner core starts at 1498 km depth in weber_core.nd."""
        disc = weber_core.get_discontinuities(include_radius=True)
        assert disc["inner-core"]["depth"] == pytest.approx(1498.0)

    def test_outer_core_has_zero_vs(self, weber_core):
        """Outer core has Vs=0 (fluid)."""
        disc = weber_core.get_discontinuities(include_radius=True)
        assert disc["outer-core"]["lower"]["vs"] == pytest.approx(0.0, abs=1e-6)

    def test_exclude_radius(self, weber_core):
        """include_radius=False should omit the outermost 'surface' entry at depth=0."""
        disc_full = weber_core.get_discontinuities(include_radius=True)
        disc_no_r = weber_core.get_discontinuities(include_radius=False)
        assert len(disc_no_r) == len(disc_full) - 1
        # The surface entry (depth=0) should be absent
        surface_entries_at_zero = [
            k for k, v in disc_no_r.items() if v["depth"] == 0.0
        ]
        assert len(surface_entries_at_zero) == 0
