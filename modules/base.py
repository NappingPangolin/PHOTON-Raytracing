"""
modules/base.py
----------------
Abstract base class for all optical modules.

Every module:
  - Has a unique type name and human-readable label
  - Has a parameter schema (for UI rendering)
  - Accepts one or more Photon objects and returns one or more Photons
  - Can serialise/deserialise itself to/from a dict

LnkMixin
--------
Optional mixin for modules that expose n / k parameters.  Adding it gives
the module:

  • Two extra param specs (via ``lnk_param_specs()``) that should be appended
    to the module's own ``param_specs()`` list:
        - ``nk_mode``    : "manual" | "lnk_table"
        - ``lnk_data``   : list of [wl_um, n, k] rows (type "lnk_table")

  • A helper ``_resolve_nk(wavelength_m)`` that returns (n, k) using either
    the manual fields or the uploaded table, depending on ``nk_mode``.

Usage
-----
    class FlatMirror(LnkMixin, OpticalModule):
        @classmethod
        def param_specs(cls):
            return [
                ParamSpec("n", ...),
                ParamSpec("k", ...),
                ...
                *cls.lnk_param_specs(),
            ]

        def process(self, photons):
            ...
            n, k = self._resolve_nk(photon.wavelength)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from optics_sim import Photon, Material


@dataclass
class ParamSpec:
    """Describes a single configurable parameter for the UI."""
    name: str
    label: str
    type: str          # 'float', 'int', 'vec3', 'select', 'bool', 'lnk_table'
    default: Any
    unit: str = ""
    min: float | None = None
    max: float | None = None
    options: list | None = None   # for 'select' type
    tooltip: str = ""


class OpticalModule(ABC):
    MODULE_TYPE: str = "base"
    LABEL: str = "Optical Element"
    COLOR: str = "#4a9eff"        # colour for the UI node
    ICON: str = "◈"

    # Sided modules (BeamSplitter, DielectricMirror, Substrate, CompoundLens)
    # have a physically meaningful front face and back face, each with its
    # own input branch(es) and output branch - see scene.py's module
    # docstring. process() for a sided module accepts a `side` kwarg
    # ("front" or "back") telling it which face the given photons are
    # arriving at; process() for a non-sided module (the default) takes no
    # such argument.
    SIDED: bool = False

    def __init__(self, params: dict | None = None):
        self.id: str = ""         # set by the scene manager
        self.params: dict = self.default_params()
        if params:
            self.params.update(params)

    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def param_specs(cls) -> list[ParamSpec]:
        """Return the parameter schema for this module."""

    @classmethod
    def default_params(cls) -> dict:
        return {s.name: s.default for s in cls.param_specs()}

    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        """
        Process a list of photons through this element.

        Returns
        -------
        out_photons : list of output Photons
        info        : dict of diagnostic information (for the UI)
        """

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type":   self.MODULE_TYPE,
            "id":     self.id,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpticalModule":
        m = cls(d.get("params", {}))
        m.id = d.get("id", "")
        return m

    # ------------------------------------------------------------------

    def _vacuum(self) -> Material:
        lam = float(self.params.get("wavelength", 800e-9))
        return Material.vacuum(lam)

    def __repr__(self):
        return f"{self.MODULE_TYPE}(id={self.id!r}, params={self.params})"


# ══════════════════════════════════════════════════════════════════════════════
# LnkMixin
# ══════════════════════════════════════════════════════════════════════════════

class LnkMixin:
    """
    Mixin that adds wavelength-dependent n/k support to any OpticalModule.

    The mixin is intentionally dependency-light: it imports LnkTable lazily
    so that modules without lnk data continue to work even if the file is
    not present.

    Concrete subclasses must:
      1. Include ``*cls.lnk_param_specs()`` in their ``param_specs()`` list.
      2. Call ``self._resolve_nk(wavelength_m)`` instead of reading n/k
         params directly, passing the field names of the manual n/k params
         as ``n_key`` / ``k_key`` (default: "n" / "k").
    """

    @classmethod
    def lnk_param_specs(cls) -> list[ParamSpec]:
        """Extra param specs to append to the subclass's param_specs()."""
        return [
            ParamSpec(
                "nk_mode", "n/k Source", "select", "manual",
                options=["manual", "lnk_table"],
                tooltip=(
                    "manual: use the n and k fields above directly.  "
                    "lnk_table: interpolate from an uploaded wavelength table."
                ),
            ),
            ParamSpec(
                "lnk_data", "n/k Table (lnk)", "lnk_table", [],
                tooltip=(
                    "Upload a two-block lnk file (wl[µm] | n then wl[µm] | k).  "
                    "The table is embedded in the scene file so it is preserved "
                    "across save/load without re-uploading."
                ),
            ),
        ]

    def _resolve_nk(
        self,
        wavelength_m: float,
        n_key: str = "n",
        k_key: str = "k",
    ) -> tuple[float, float]:
        """
        Return (n, k) appropriate for *wavelength_m* (SI metres).

        In 'manual' mode the values are read directly from ``self.params``.
        In 'lnk_table' mode the stored table is interpolated.  Falls back to
        manual values if the table is empty or cannot be parsed.
        """
        mode = self.params.get("nk_mode", "manual")

        if mode == "lnk_table":
            rows = self.params.get("lnk_data", [])
            if rows:
                try:
                    from .lnk_table import LnkTable
                    tbl = LnkTable.from_rows(rows)
                    return tbl.get_nk(wavelength_m)
                except Exception:
                    pass  # fall through to manual

        # Manual (or fallback)
        return float(self.params.get(n_key, 1.0)), float(self.params.get(k_key, 0.0))