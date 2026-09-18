"""
Material class: optical and magnetic properties of a medium.
"""

from __future__ import annotations
import numpy as np


MU0 = 4 * np.pi * 1e-7  # H/m


class Material:
    """
    Optical material / medium.

    Parameters
    ----------
    n         : real part of refractive index
    k         : imaginary part (extinction coefficient)
    mu_r      : relative magnetic permeability (unitless)
    density   : arbitrary density parameter (unused in Fresnel, reserved)
    wavelength: free-space wavelength (m)
    """

    def __init__(
        self,
        n: float = 1.0,
        k: float = 0.0,
        mu_r: float = 1.0,
        density: float = 0.0,
        wavelength: float = 800e-9,
    ):
        self.n = float(n)
        self.k = float(k)
        self.mu_r = float(mu_r)
        self.density = float(density)
        self.wavelength = float(wavelength)
        self.mu = self.mu_r * MU0
        self.nk: complex = complex(n, -k)   # complex refractive index (n - ik convention)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def custom(cls, wavelength: float = 800e-9, n: float = 1.0, k: float = 0.0, mu_r: float = 1.0, density: float = 0.0) -> "Material":
        return cls(n=n, k=k, mu_r=mu_r, density=density, wavelength=wavelength)
    
    @classmethod
    def vacuum(cls, wavelength: float = 800e-9) -> "Material":
        """Return a vacuum (air) material."""
        return cls(n=1.0, k=0.0, mu_r=1.0, density=0.0, wavelength=wavelength)

    @classmethod
    def silver(cls, wavelength: float = 800e-9) -> "Material":
        """Return approximate silver optical constants at 800 nm."""
        return cls(n=0.036759, k=5.5698, mu_r=1.0, density=1.0, wavelength=wavelength)

    @classmethod
    def gold(cls, wavelength: float = 800e-9) -> "Material":
        """Return approximate gold optical constants at 800 nm."""
        return cls(n=0.181, k=5.126, mu_r=1.0, density=1.0, wavelength=wavelength)

    @classmethod
    def bk7(cls, wavelength: float = 800e-9) -> "Material":
        """Return approximate BK7 glass constants at 800 nm."""
        return cls(n=1.5108, k=9.2656e-9, mu_r=1.0000004, density=1.0, wavelength=wavelength)

    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Material(n={self.n}, k={self.k}, mu_r={self.mu_r}, "
            f"λ={self.wavelength*1e9:.1f} nm)"
        )
