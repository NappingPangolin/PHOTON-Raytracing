"""
Chamber: container that sequences reflections from a list of mirrors/surfaces.
"""

from __future__ import annotations

import numpy as np
from .photon import Photon
from .material import Material
from .surface import Surface
from .parabola import OffAxisParaboloid


class Chamber:
    """
    An ordered collection of optical elements (mirrors, paraboloids).

    Photons are propagated through the elements in insertion order.

    Parameters
    ----------
    wavelength : free-space wavelength (m) – used to create the internal vacuum.
    """

    def __init__(self, wavelength: float = 800e-9):
        self.wavelength = float(wavelength)
        self.elements: list = []
        self.vacuum = Material.vacuum(wavelength)

    # ------------------------------------------------------------------

    def add(self, element) -> "Chamber":
        """
        Add an optical element (Surface or ParabolaV3) to the chamber.

        Returns self so calls can be chained::

            ch = Chamber(800e-9).add(mirror1).add(mirror2)
        """
        if isinstance(element, (Surface, ParabolaV3)):
            self.elements.append(element)
        else:
            raise TypeError(
                f"Expected Surface or ParabolaV3, got {type(element).__name__}. "
                "The old MATLAB Parabola class is not supported; use ParabolaV3."
            )
        return self

    # ------------------------------------------------------------------

    def reflect(self, photon: Photon, d_out: float = 0.0) -> tuple[Photon, list]:
        """
        Propagate *photon* through all elements in the chamber.

        Parameters
        ----------
        photon : initial Photon
        d_out  : extra travel distance *after* the last element

        Returns
        -------
        photon_out   : final Photon after d_out extra travel
        photon_bunch : list of all intermediate Photon states
        """
        bunch = [photon, photon]  # mimic MATLAB's initial duplication

        for element in self.elements:
            current = bunch[-1]

            if isinstance(element, Surface):
                dist = element.calc_distance_to_surface([current])[0]
                at_surface = current.travel_distance(dist)
                bunch.append(at_surface)
                reflected, _ = element.reflect(at_surface, self.vacuum)
                bunch.append(reflected)

            elif isinstance(element, ParabolaV3):
                reflected, _ = element.reflect(current, self.vacuum)
                bunch.append(reflected)

        photon_out = bunch[-1]
        if d_out != 0:
            bunch.append(photon_out.travel_distance(d_out))

        return photon_out, bunch

    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Chamber(elements={len(self.elements)}, λ={self.wavelength*1e9:.1f} nm)"
