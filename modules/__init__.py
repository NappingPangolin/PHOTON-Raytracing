"""
modules/__init__.py
--------------------
Auto-discovers all OpticalModule subclasses and provides a registry.
"""

from .base import OpticalModule, ParamSpec
from .flat_mirror import FlatMirror
from .parabolic_mirror import ParabolicMirror
from .beam_splitter import BeamSplitter
from .compound_lens import CompoundLens
from .gradient_medium import GradientMedium
from .misc_modules import PhotonSource, FreePropagate, Detector, UploadPhotons
from .substrate import Substrate
from .dielectric_mirror import DielectricMirror

MODULE_REGISTRY: dict[str, type[OpticalModule]] = {
    cls.MODULE_TYPE: cls
    for cls in [
        PhotonSource,
        FlatMirror,
        ParabolicMirror,
        BeamSplitter,
        FreePropagate,
        GradientMedium,
        Detector,
        UploadPhotons,
        CompoundLens,
        Substrate,
        DielectricMirror,
    ]
}


def create_module(module_type: str, params: dict | None = None) -> OpticalModule:
    cls = MODULE_REGISTRY.get(module_type)
    if cls is None:
        raise ValueError(f"Unknown module type: {module_type!r}. "
                         f"Available: {list(MODULE_REGISTRY)}")
    return cls(params)


def module_catalog() -> list[dict]:
    return [
        {
            "type":  cls.MODULE_TYPE,
            "label": cls.LABEL,
            "color": cls.COLOR,
            "icon":  cls.ICON,
            "params": [
                {
                    "name":    s.name,
                    "label":   s.label,
                    "type":    s.type,
                    "default": s.default,
                    "unit":    s.unit,
                    "min":     s.min,
                    "max":     s.max,
                    "options": s.options,
                    "tooltip": s.tooltip,
                }
                for s in cls.param_specs()
            ],
        }
        for cls in MODULE_REGISTRY.values()
    ]