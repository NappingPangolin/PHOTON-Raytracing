"""
OffAxisParaboloid: physically correct off-axis paraboloidal mirror.

Geometry
--------
The paraboloid is defined in its canonical frame by::

    y² + z² = 4·f·(x + f)          (axis along +x, focus at origin, vertex at (-f,0,0))

In this frame:
  - The focus is at the origin  (0, 0, 0).
  - The vertex is at  (-f, 0, 0).
  - Parallel rays along -x are ALL focused to the origin.
  - The **paraboloid axis** is the +x direction = -k̂_in.

Off-axis use
~~~~~~~~~~~~
Rather than using the paraboloid symmetrically around its axis we take
a lateral patch.  For a desired output direction k̂_out the mirror centre
lies at the point where the parabola surface normal bisects  (-k̂_in, k̂_out).
In the canonical frame that point has coordinates::

    y_c = 2·f·sin(θ) / (1 + cos θ)   (conic section formula)
    z_c = 0   (we choose the deflection plane as the xz-plane by picking the
               right e1/e2 frame orientation — see _build_frame)
    x_c = y_c²/(4f) - f

where θ is the off-axis angle = angle between -k̂_in and k̂_out.

Per-photon algorithm
~~~~~~~~~~~~~~~~~~~~
1. Transform photon (position, direction) into the canonical frame.
   - Origin of canonical frame = focal point F (global)
   - Rotation: canonical +x = -k̂_in,  canonical +y = chosen so k̂_out
     lies in the canonical x-y plane.
2. Intersect ray with  y²+z² - 4f·x - 4f²=0  (quadratic in t).
3. Compute the analytic normal at the intersection:
       ∇F = (-4f,  2y,  2z)       →   normalise, orient toward incoming.
4. Transform normal back to global frame.
5. Travel photon to intersection and call Surface.reflect() with that normal.

Module parameters (via parabolic_mirror.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  r_x/r_y/r_z      : position of the **mirror centre** (global, metres)
  k_in_x/y/z       : incoming beam direction
  k_out_x/y/z      : desired output beam direction
  focal_length      : f  (metres)
  n, k_ext          : mirror coating optical constants
  wavelength        : (metres)
"""

from __future__ import annotations
import numpy as np
from .photon import Photon
from .material import Material, MU0
from .surface import Surface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_frame(
    k_in_hat: np.ndarray,
    k_out_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a right-handed orthonormal frame (ex, ey, ez) for the canonical frame.

    Conventions:
      ex = -k̂_in         (paraboloid axis, parallel rays arrive along -ex)
      ey                  chosen so k̂_out lies in the ex-ey plane (with ey·k̂_out > 0)
      ez = ex × ey
    """
    ex = -k_in_hat / np.linalg.norm(k_in_hat)

    # k̂_out has a component in the ex direction and a transverse component.
    # We define ey to be along the transverse component of k̂_out.
    k_out_hat = k_out_hat / np.linalg.norm(k_out_hat)
    k_out_perp = k_out_hat - np.dot(k_out_hat, ex) * ex   # remove axis component

    norm_perp = np.linalg.norm(k_out_perp)
    if norm_perp < 1e-10:
        # k_out is anti-parallel to k_in (retro-reflection) — degenerate.
        # Pick any perpendicular as ey.
        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(ex, ref)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        ey = np.cross(ex, ref)
        ey /= np.linalg.norm(ey)
    else:
        ey = k_out_perp / norm_perp

    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)
    return ex, ey, ez   # columns of R_canonical_to_global


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OffAxisParaboloid:
    """
    Off-axis paraboloidal mirror.

    Parameters
    ----------
    n, k_ext   : real / imaginary refractive index of the mirror coating
    wavelength : free-space wavelength [m]
    focal_length : f  [m]
    mirror_center : (3,) global position of the mirror centre [m]
    k_in_dir   : (3,) direction of the *incoming* beam (not normalised)
    k_out_dir  : (3,) desired output beam direction (not normalised)
    beam_shifts : enable Goos-Hänchen / Imbert-Fedorov shifts
    """

    def __init__(
        self,
        n: float,
        k_ext: float,
        wavelength: float,
        focal_length: float,
        mirror_center: np.ndarray,
        k_in_dir: np.ndarray,
        k_out_dir: np.ndarray,
        *,
        beam_shifts: bool = False,
    ):
        self.n           = float(n)
        self.k_ext       = float(k_ext)
        self.wavelength  = float(wavelength)
        self.f           = float(focal_length)
        self.beam_shifts = beam_shifts

        k_in_hat  = np.asarray(k_in_dir,  dtype=float)
        k_out_hat = np.asarray(k_out_dir, dtype=float)
        k_in_hat  /= np.linalg.norm(k_in_hat)
        k_out_hat /= np.linalg.norm(k_out_hat)

        self.k_in_hat  = k_in_hat
        self.k_out_hat = k_out_hat

        # Off-axis angle (angle between -k_in and k_out)
        self.off_axis_angle = float(
            np.arccos(np.clip(np.dot(-k_in_hat, k_out_hat), -1.0, 1.0))
        )

        # Build canonical frame (columns of R = global←canonical rotation)
        ex, ey, ez = _build_frame(k_in_hat, k_out_hat)
        # R maps canonical → global:  v_global = R @ v_canonical
        self.R = np.column_stack([ex, ey, ez])  # (3,3), orthonormal

        # Mirror centre in canonical frame
        r_m_c = self.R.T @ np.asarray(mirror_center, dtype=float)
        self.F = np.asarray(mirror_center, dtype=float) - self.R @ r_m_c
        d = 2.0 * self.f / (1.0 + np.cos(self.off_axis_angle))
        k_out_c = self.R.T @ k_out_hat   # k_out in canonical frame
        r_mc_canonical = -d * k_out_c     # mirror centre in canonical frame (from focus)

        # Focus in global frame
        self.F = np.asarray(mirror_center, dtype=float) - self.R @ r_mc_canonical

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _to_canonical_pos(self, pos: np.ndarray) -> np.ndarray:
        """Global position → canonical position (origin = focus F)."""
        return self.R.T @ (pos - self.F)

    def _to_canonical_dir(self, vec: np.ndarray) -> np.ndarray:
        """Global direction vector → canonical direction vector."""
        return self.R.T @ vec

    def _to_global_dir(self, vec: np.ndarray) -> np.ndarray:
        """Canonical direction vector → global direction vector."""
        return self.R @ vec

    def _to_global_pos(self, pos_c: np.ndarray) -> np.ndarray:
        """Canonical position → global position."""
        return self.R @ pos_c + self.F

    # ------------------------------------------------------------------
    # Intersection  (canonical frame)
    # ------------------------------------------------------------------

    def _intersect_canonical(
        self,
        r_c: np.ndarray,
        khat_c: np.ndarray,
    ) -> float:
        """
        Intersect ray  r_c + t·khat_c  with the paraboloid
            F(x,y,z) = y² + z² - 4f·x - 4f²  =  0

        Substituting the ray:
            A·t² + B·t + C = 0
        where
            A = ky² + kz²
            B = 2(ry·ky + rz·kz - 2f·kx)
            C = ry² + rz² - 4f·(rx + f)

        Returns the best (smallest positive) root.
        """
        rx, ry, rz = r_c
        kx, ky, kz = khat_c
        f = self.f

        A = ky**2 + kz**2
        B = 2.0 * (ry*ky + rz*kz - 2.0*f*kx)
        C = ry**2 + rz**2 - 4.0*f*(rx + f)

        if abs(A) < 1e-30:
            # Ray parallel to paraboloid axis (ky=kz=0): one solution
            if abs(B) < 1e-30:
                raise RuntimeError(
                    "Ray is on the paraboloid axis and parallel to it — no unique intersection."
                )
            return float(-C / B)

        discriminant = B**2 - 4.0*A*C
        if discriminant < 0.0:
            discriminant = 0.0   # numerical tolerance
        sq = np.sqrt(discriminant)
        t1 = (-B - sq) / (2.0*A)
        t2 = (-B + sq) / (2.0*A)

        # Prefer smallest positive t (ray hitting the mirror in front)
        pos_roots = [t for t in (t1, t2) if t > -1e-9]
        if pos_roots:
            return float(min(pos_roots))
        # Both negative — take largest (least negative); warn
        return float(max(t1, t2))

    # ------------------------------------------------------------------
    # Surface normal  (canonical frame)
    # ------------------------------------------------------------------

    def _normal_canonical(self, pt_c: np.ndarray, khat_c: np.ndarray) -> np.ndarray:
        """
        Analytic outward normal to  y²+z²-4fx-4f²=0  at point pt_c.
            ∇F = (-4f,  2y,  2z)
        Oriented so that  n̂ · (-khat_c) > 0  (faces incoming beam).
        """
        _, y, z = pt_c
        grad = np.array([-4.0*self.f, 2.0*y, 2.0*z], dtype=float)
        n = grad / np.linalg.norm(grad)
        if np.dot(n, -khat_c) < 0:
            n = -n
        return n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(
        self,
        photon_in: Photon,
        medium: Material,
    ) -> tuple[Photon, Photon]:
        """
        Reflect *photon_in* off the paraboloid.

        Returns
        -------
        photon_r  : reflected Photon in the global frame
        photon_at : Photon at the surface (for diagnostics)
        """
        r_g   = np.real(photon_in.r).astype(float)
        kv_g  = np.real(photon_in.k).astype(float)
        khat_g = kv_g / np.linalg.norm(kv_g)

        # --- transform to canonical frame ---------------------------------
        r_c    = self._to_canonical_pos(r_g)
        khat_c = self._to_canonical_dir(khat_g)

        # --- intersection -------------------------------------------------
        t = self._intersect_canonical(r_c, khat_c)
        pt_c = r_c + t * khat_c
        pt_g = self._to_global_pos(pt_c)

        # --- normal -------------------------------------------------------
        n_c = self._normal_canonical(pt_c, khat_c)
        n_g = self._to_global_dir(n_c)
        n_g /= np.linalg.norm(n_g)

        # --- travel photon to surface and reflect -------------------------
        photon_at = photon_in.travel_to_point(pt_g)

        surf = Surface(
            self.n, self.k_ext, 1.0, 1.0, self.wavelength,
            n_g,
            r=pt_g,
            beam_shifts=self.beam_shifts,
        )
        photon_r, photon_t = surf.reflect(photon_at, medium)
        return photon_r, photon_at

    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"OffAxisParaboloid(f={self.f:.4g} m, "
            f"off_axis={np.degrees(self.off_axis_angle):.1f}°, "
            f"F_global={np.round(self.F,4)})"
        )