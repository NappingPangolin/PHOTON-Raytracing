"""
modules/beam_splitter.py
-----------------------
Beam splitter with optional wavelength-dependent reflectance via an lnk table.

When nk_mode = "lnk_table" the reflectance R is NOT used; instead the Fresnel
equations are applied using the interpolated (n, k) at the photon's wavelength.
When nk_mode = "manual" the fixed reflectance R splits intensity directly
(as before), and the n/k fields are ignored for the physics.
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon
from .base import OpticalModule, LnkMixin, ParamSpec


class BeamSplitter(LnkMixin, OpticalModule):
    MODULE_TYPE = "beamsplitter"
    LABEL = "Beam Splitter"
    COLOR = "#f472b6"
    ICON = "⊕"
    SIDED = True

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            # Position
            ParamSpec("r_x", "R_x", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Beam splitter position X"),
            ParamSpec("r_y", "R_y", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Beam splitter position Y"),
            ParamSpec("r_z", "R_z", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Beam splitter position Z"),

            # Orientation
            ParamSpec("normal_x", "N_x", "float", -1.0, "",
                      min=-1.0, max=1.0),
            ParamSpec("normal_y", "N_y", "float",  0.0, "",
                      min=-1.0, max=1.0),
            ParamSpec("normal_z", "N_z", "float", -1.0, "",
                      min=-1.0, max=1.0),

            # Manual-mode reflectance
            ParamSpec("R", "Reflectance R", "float", 0.5, "",
                      min=0.0, max=1.0,
                      tooltip="Fraction of intensity reflected (manual mode only; 1-R transmitted)"),

            # Aperture / visualisation
            ParamSpec("radius", "Aperture Radius", "float", 0.05, "m",
                      min=0.0, max=100.0,
                      tooltip="Beam splitter aperture radius (also used for the "
                              "3-D viewer disc, drawn with 50% opacity). Rays "
                              "landing outside this radius are blocked. "
                              "0 = unbounded (no clipping)."),

            # Wavelength
            ParamSpec("wavelength", "Wavelength", "float", 800e-9, "m",
                      min=200e-9, max=10_000e-9),

            # ── wavelength-dependent n/k (LnkMixin) ──────────────────────
            *cls.lnk_param_specs(),
        ]

    def _fresnel_R(self, n: float, k: float, cos_theta_i: float) -> float:
        """
        Compute power reflectance using the Fresnel equations for an absorbing
        medium (complex index  ñ = n + ik).

        Assumes unpolarised light (average of s and p).
        Incident medium is vacuum (n_i = 1).
        """
        cos_i = abs(cos_theta_i)
        sin_i = np.sqrt(max(0.0, 1.0 - cos_i ** 2))

        nc = complex(n, k)
        # Snell (complex)
        sin_t = sin_i / nc
        cos_t = np.sqrt(1.0 - sin_t ** 2 + 0j)

        # Fresnel amplitude coefficients
        rs = (cos_i - nc * cos_t) / (cos_i + nc * cos_t)
        rp = (nc * cos_i - cos_t) / (nc * cos_i + cos_t)

        Rs = abs(rs) ** 2
        Rp = abs(rp) ** 2
        return float(0.5 * (Rs + Rp))

    def process(self, photons: list[Photon], side: str = "front") -> tuple[list[Photon], dict]:
        # A single flat interface reflects/transmits identically regardless
        # of which side the beam approaches from - the Fresnel formula and
        # the plane-intersection math below never assume a particular sign
        # for the angle of incidence, so `side` only matters to scene.py
        # (which uses it to route the reflected/transmitted pair to the
        # correct front/back output branch) and is otherwise unused here.
        lam_ref = float(self.params["wavelength"])
        mode    = self.params.get("nk_mode", "manual")

        r = np.array([
            float(self.params.get("r_x", 0.0)),
            float(self.params.get("r_y", 0.0)),
            float(self.params.get("r_z", 0.0)),
        ], dtype=float)

        normal = np.array([
            float(self.params["normal_x"]),
            float(self.params["normal_y"]),
            float(self.params["normal_z"]),
        ], dtype=float)
        n_hat = normal / np.linalg.norm(normal)

        radius = float(self.params.get("radius", 0.0))

        out  = []
        info = []
        n_blocked = 0

        for ph in photons:
            lam   = float(ph.wavelength)
            k_hat = np.real(ph.k) / np.linalg.norm(np.real(ph.k))

            # ── travel to beam-splitter plane ────────────────────────────
            denominator = np.dot(k_hat, n_hat)
            if abs(denominator) < 1e-12:
                dist  = 0.0
                ph_at = ph
            else:
                numerator = np.dot(r - np.real(ph.r), n_hat)
                dist  = numerator / denominator
                # Signed distance: may be negative if the splitter plane
                # ends up behind the photon (e.g. after a free_propagate
                # overshoot). travel_distance() supports this directly - it
                # walks the photon back in time to the plane, correctly
                # advancing phase/time and attenuating for |distance|
                # rather than silently clamping to "already there".
                ph_at = ph.travel_distance(dist)

            # ── aperture clipping ────────────────────────────────────────
            if radius > 0:
                d_to_axis = float(np.linalg.norm(np.real(ph_at.r).astype(float) - r))
                if d_to_axis > radius:
                    n_blocked += 1
                    info.append({
                        "blocked": True,
                        "distance_from_center": round(d_to_axis, 6),
                        "radius": radius,
                    })
                    continue

            # ── compute reflectance ──────────────────────────────────────
            if mode == "lnk_table":
                n_val, k_val = self._resolve_nk(lam)
                cos_theta = abs(np.dot(np.real(ph_at.k) / np.linalg.norm(np.real(ph_at.k)), n_hat))
                R = self._fresnel_R(n_val, k_val, cos_theta)
            else:
                R     = float(self.params["R"])
                n_val = float(self.params.get("n", 1.0))
                k_val = float(self.params.get("k", 0.0))

            # ── reflected branch ─────────────────────────────────────────
            k_r = np.real(ph_at.k) - 2 * np.dot(np.real(ph_at.k), n_hat) * n_hat
            ph_r = ph_at._copy()
            k_norm = np.linalg.norm(np.real(ph_at.k))
            if k_norm > 0:
                ph_r.k = (k_r / np.linalg.norm(k_r)) * k_norm

            E_reflected = ph_at.E - 2 * np.dot(ph_at.E, n_hat) * n_hat
            k_r_hat = k_r / np.linalg.norm(k_r)
            E_reflected = E_reflected - np.dot(E_reflected, k_r_hat) * k_r_hat
            e_norm = np.linalg.norm(E_reflected)
            if e_norm > 0:
                E_reflected = E_reflected / e_norm
            ph_r.E = E_reflected
            ph_r.I = ph_at.I * R

            # ── transmitted branch ───────────────────────────────────────
            ph_t   = ph_at._copy()
            ph_t.I = ph_at.I * (1 - R)

            out.extend([ph_r, ph_t])
            info.append({
                "R":             round(R, 6),
                "T":             round(1 - R, 6),
                "n_used":        round(n_val, 6),
                "k_used":        round(k_val, 6),
                "wavelength_nm": round(lam * 1e9, 3),
                "distance":      round(float(dist), 6),
            })

        return out, {
            "nk_mode":    mode,
            "position":   r.tolist(),
            "normal":     n_hat.tolist(),
            "radius":     radius,
            "n_blocked":  n_blocked,
            "reflections": info,
        }