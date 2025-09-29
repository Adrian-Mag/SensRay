"""Compatibility shims for the historical 'sensray.core' package.

This module re-exports classes from their new locations to preserve
imports like:
    from sensray.core.earth_models import EarthModelManager
    from sensray.core.ray_paths import RayPathTracer
    from sensray.core.travel_times import TravelTimeCalculator
"""

from .earth_models import EarthModelManager  # type: ignore[F401]
from .ray_paths import RayPathTracer  # type: ignore[F401]
from .travel_times import TravelTimeCalculator  # type: ignore[F401]

__all__ = [
    "EarthModelManager",
    "RayPathTracer",
    "TravelTimeCalculator",
]
