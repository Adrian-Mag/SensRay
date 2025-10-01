"""
SensRay

Lightweight public package API. This module intentionally exposes only a
small set of stable, well-maintained entry points: `PlanetModel`, the
mesh helper `MeshEarthModel`, and utilities like `CoordinateConverter`.

The package previously included higher-level wrappers for ray tracing,
travel-times and plotting. Those were intentionally removed in favor of
using ObsPy/TauP and dedicated visualization tools directly.
"""

__version__ = "0.3.0"
__author__ = "PhD Project"

# Public API: import only stable, present modules
from .core.model import PlanetModel
from .mesh.earth_model import MeshEarthModel
from .utils import CoordinateConverter

__all__ = ["MeshEarthModel", "PlanetModel", "CoordinateConverter"]
