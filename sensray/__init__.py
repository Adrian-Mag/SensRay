"""
SensRay

Lightweight public package API. This module intentionally exposes only a
small set of stable, well-maintained entry points: `PlanetModel`,
`PlanetMesh`, and utilities like `CoordinateConverter`.

The package previously included higher-level wrappers for ray tracing,
travel-times and plotting. Those were intentionally removed in favor of
using ObsPy/TauP and dedicated visualization tools directly.
"""

__version__ = "0.5.0"
__author__ = "PhD Project"

# Public API: import only stable, present modules
from .planet_model import PlanetModel
from .planet_mesh import PlanetMesh
from .coordinates import CoordinateConverter

__all__ = [
    "PlanetModel",
    "PlanetMesh",
    "CoordinateConverter"
]
