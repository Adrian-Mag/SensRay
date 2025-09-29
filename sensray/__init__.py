"""
SensRay: Seismic ray tracing, travel times, Earth models, and visualization.

Public API highlights (lazy-loaded):
- TravelTimeCalculator, RayPathTracer, EarthModelManager
- MeshEarthModel
- EarthPlotter, Earth3DVisualizer
- CoordinateConverter

Import examples:
    from sensray import TravelTimeCalculator, EarthPlotter
    from sensray.mesh import MeshEarthModel
    from sensray.core import EarthModelManager

This module uses lazy attribute loading to avoid importing heavy
dependencies until actually needed.
"""

from importlib import import_module
from typing import Any, Dict

__all__ = [
    "TravelTimeCalculator",
    "RayPathTracer",
    "EarthModelManager",
    "MeshEarthModel",
    "EarthPlotter",
    "Earth3DVisualizer",
    "CoordinateConverter",
]

# Keep in sync with setup.py version
__version__ = "0.3.0"

_LAZY_ATTRS: Dict[str, str] = {
    # name: "module_path:attribute"
    "TravelTimeCalculator": "sensray.other.travel_times:TravelTimeCalculator",
    "RayPathTracer": "sensray.rays.ray_paths:RayPathTracer",
    "EarthModelManager": "sensray.utils.earth_models:EarthModelManager",
    "MeshEarthModel": "sensray.mesh.earth_model:MeshEarthModel",
    "EarthPlotter": "sensray.visualization.earth_2d_plot:EarthPlotter",
    "Earth3DVisualizer": (
        "sensray.visualization.earth_3d_plot:Earth3DVisualizer"
    ),
    "CoordinateConverter": "sensray.utils.coordinates:CoordinateConverter",
}


class _LazyAttr:
    """Descriptor-like callable proxy that loads the real target on demand.

    This helps static linters see the attribute while deferring heavy imports
    until the object is actually used (called or attribute accessed).
    """

    __slots__ = ("_target", "_cached")

    def __init__(self, target: str) -> None:
        self._target = target  # module:attr
        self._cached: Any = None

    def _load(self) -> Any:
        if self._cached is None:
            mod_path, attr = self._target.split(":", 1)
            mod = import_module(mod_path)
            self._cached = getattr(mod, attr)
        return self._cached

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<lazy {self._target}>"


# Pre-populate top-level names with lazy proxies to satisfy linters
TravelTimeCalculator = _LazyAttr(
    _LAZY_ATTRS["TravelTimeCalculator"]
)  # type: ignore[var-annotated]
RayPathTracer = _LazyAttr(
    _LAZY_ATTRS["RayPathTracer"]
)  # type: ignore[var-annotated]
EarthModelManager = _LazyAttr(
    _LAZY_ATTRS["EarthModelManager"]
)  # type: ignore[var-annotated]
MeshEarthModel = _LazyAttr(
    _LAZY_ATTRS["MeshEarthModel"]
)  # type: ignore[var-annotated]
EarthPlotter = _LazyAttr(
    _LAZY_ATTRS["EarthPlotter"]
)  # type: ignore[var-annotated]
Earth3DVisualizer = _LazyAttr(
    _LAZY_ATTRS["Earth3DVisualizer"]
)  # type: ignore[var-annotated]
CoordinateConverter = _LazyAttr(
    _LAZY_ATTRS["CoordinateConverter"]
)  # type: ignore[var-annotated]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute loading
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'sensray' has no attribute {name!r}")
    mod_path, attr = target.split(":", 1)
    mod = import_module(mod_path)
    obj = getattr(mod, attr)
    globals()[name] = obj  # cache for future access
    return obj


def __dir__() -> list:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
