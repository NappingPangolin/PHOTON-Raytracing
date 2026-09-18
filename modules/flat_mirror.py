"""
modules/flat_mirror.py
-----------------------
Flat reflective mirror.

n/k can be set manually or interpolated from an uploaded lnk table;
the active wavelength of each photon is used automatically when
nk_mode = "lnk_table".
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon, Material, Surface
from .base import OpticalModule, LnkMixin, ParamSpec

# Material presets at 800 nm
MATERIAL_PRESETS = {
    "Ag":      (0.036759, 5.5698),
    "Au":      (0.181,    5.126),
    "Al":      (1.664,    7.617),
    "Cu":      (0.212,    4.956),
    "Ideal":   (0.0,      1e6),    # near-perfect reflector
    "Custom":  (None,     None),
}


class FlatMirror(LnkMixin, OpticalModule):
    MODULE_TYPE = "flat_mirror"
    LABEL = "Flat Mirror"
    COLOR = "#5b8dee"
    ICON = "▱"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            # Position
            ParamSpec("r_x", "R_x", "float", 0.0, "m",
                    min=-100.0, max=100.0,
                    tooltip="Mirror center position X"),
            ParamSpec("r_y", "R_y", "float", 0.0, "m",
                    min=-100.0, max=100.0,
                    tooltip="Mirror center position Y"),
            ParamSpec("r_z", "R_z", "float", 0.0, "m",
                    min=-100.0, max=100.0,
                    tooltip="Mirror center position Z"),

            # Orientation
            ParamSpec("normal_x", "N_x", "float", -1.0, "",
                    min=-1.0, max=1.0),
            ParamSpec("normal_y", "N_y", "float",  0.0, "",
                    min=-1.0, max=1.0),
            ParamSpec("normal_z", "N_z", "float", -1.0, "",
                    min=-1.0, max=1.0),

            # Manual optical constants
            ParamSpec("n", "n (real)", "float", 0.036759, "",
                    min=0.0, max=10.0,
                    tooltip="Real part of refractive index (used when n/k Source = manual)"),
            ParamSpec("k", "k (extinction)", "float", 5.5698, "",
                    min=0.0, max=30.0,
                    tooltip="Extinction coefficient (used when n/k Source = manual)"),
            ParamSpec("wavelength", "Wavelength", "float", 800e-9, "m",
                    min=1e-12, max=10_000e-9,
                    tooltip="Reference wavelength (only used for manual mode)"),

            # Goos-Hänchen / Imbert-Fedorov
            ParamSpec("beam_shifts", "GH/IF Beam Shifts", "bool", False,
                    tooltip="Enable Goos-Hänchen and Imbert-Fedorov shifts"),

            # Aperture / visualisation
            ParamSpec("radius", "Aperture Radius", "float", 0.05, "m",
                    min=0.0, max=100.0,
                    tooltip="Mirror aperture radius (also used for the 3-D "
                            "viewer disc). Rays landing outside this radius "
                            "are blocked. 0 = unbounded (no clipping)."),

            # ── wavelength-dependent n/k (LnkMixin) ──────────────────────
            *cls.lnk_param_specs(),
        ]

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        # Position
        r = np.array([
            float(self.params.get("r_x", 0.0)),
            float(self.params.get("r_y", 0.0)),
            float(self.params.get("r_z", 0.0)),
        ], dtype=float)

        # Normal
        normal = np.array([
            float(self.params["normal_x"]),
            float(self.params["normal_y"]),
            float(self.params["normal_z"]),
        ], dtype=float)
        normal = normal / np.linalg.norm(normal)

        beam_shifts = bool(self.params.get("beam_shifts", False))

        radius = float(self.params.get("radius", 0.0))

        out_photons = []
        info_list = []
        n_blocked = 0

        for ph in photons:
            lam = float(ph.wavelength)

            # Resolve n/k — uses the photon's own wavelength when in lnk_table mode
            n, k = self._resolve_nk(lam)

            surf = Surface(n, k, 1.0, 1.0, lam, normal, r=r,
                           beam_shifts=beam_shifts)
            vacuum = Material.vacuum(lam)

            dist = surf.calc_distance_to_surface([ph])[0]
            ph_at = ph.travel_distance(dist)

            # ── aperture clipping ────────────────────────────────────────
            if radius > 0:
                d_to_axis = float(np.linalg.norm(np.real(ph_at.r).astype(float) - r))
                if d_to_axis > radius:
                    n_blocked += 1
                    info_list.append({
                        "blocked": True,
                        "distance_from_center": round(d_to_axis, 6),
                        "radius": radius,
                    })
                    continue

            ph_r, _ = surf.reflect(ph_at, vacuum)
            out_photons.append(ph_r)

            k_hat = np.real(ph.k) / np.linalg.norm(np.real(ph.k))
            theta = np.degrees(np.arccos(np.clip(-np.dot(np.real(k_hat), normal), -1.0, 1.0)))
            if theta > 90:
                normal_local = -normal
                theta = np.degrees(np.arccos(np.clip(-np.dot(np.real(k_hat), normal_local), -1.0, 1.0)))

            info_list.append({
                "theta_i_deg": round(float(theta), 3),
                "R":           round(float(ph_r.I), 4),
                "distance":    round(float(dist), 6),
                "n_used":      round(n, 6),
                "k_used":      round(k, 6),
                "wavelength_nm": round(lam * 1e9, 3),
                "E_out":       ph_r.E.tolist(),
            })

        return out_photons, {
            "nk_mode":    self.params.get("nk_mode", "manual"),
            "position":   r.tolist(),
            "normal":     normal.tolist(),
            "radius":     radius,
            "n_blocked":  n_blocked,
            "reflections": info_list,
        }