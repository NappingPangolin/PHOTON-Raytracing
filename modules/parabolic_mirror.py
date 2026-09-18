"""
modules/parabolic_mirror.py
----------------------------
Off-axis paraboloidal focusing / collimating mirror.

n/k can be set manually or interpolated from an uploaded lnk table;
each photon's own wavelength is used automatically in lnk_table mode.
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon, Material
from optics_sim.parabola import OffAxisParaboloid
from .base import OpticalModule, LnkMixin, ParamSpec


class ParabolicMirror(LnkMixin, OpticalModule):
    MODULE_TYPE = "parabolic_mirror"
    LABEL = "Parabolic Mirror"
    COLOR = "#e8814d"
    ICON = "⌒"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            # ── mirror centre position ────────────────────────────────────
            ParamSpec("r_x", "R_x", "float", 1, "m",
                      min=-100.0, max=100.0,
                      tooltip="Mirror aperture centre position X"),
            ParamSpec("r_y", "R_y", "float", 0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Mirror aperture centre position Y"),
            ParamSpec("r_z", "R_z", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Mirror aperture centre position Z"),

            # ── beam directions ───────────────────────────────────────────
            ParamSpec("k_in_x",  "k_in X",  "float",  1.0, "",
                      min=-1.0, max=1.0,
                      tooltip="Incoming beam direction X (will be normalised)"),
            ParamSpec("k_in_y",  "k_in Y",  "float",  0.0, "",
                      min=-1.0, max=1.0),
            ParamSpec("k_in_z",  "k_in Z",  "float",  0.0, "",
                      min=-1.0, max=1.0),
            ParamSpec("k_out_x", "k_out X", "float",  0.0, "",
                      min=-1.0, max=1.0,
                      tooltip="Desired output beam direction X (will be normalised)"),
            ParamSpec("k_out_y", "k_out Y", "float",  0.0, "",
                      min=-1.0, max=1.0),
            ParamSpec("k_out_z", "k_out Z", "float",  1.0, "",
                      min=-1.0, max=1.0),

            # ── optics ────────────────────────────────────────────────────
            ParamSpec("focal_length", "Focal Length (PFL)", "float", 0.1, "m",
                      min=1e-4, max=100.0,
                      tooltip="Paraboloid focal length f"),
            ParamSpec("n",     "n (real)",       "float", 0.036759, "",
                      min=0.0, max=10.0,
                      tooltip="Real part of mirror coating refractive index (manual mode)"),
            ParamSpec("k_ext", "k (extinction)", "float", 5.5698,   "",
                      min=0.0, max=30.0,
                      tooltip="Extinction coefficient of mirror coating (manual mode)"),
            ParamSpec("wavelength", "Wavelength", "float", 800e-9, "m",
                      min=200e-9, max=10_000e-9,
                      tooltip="Reference wavelength (manual mode only)"),
            ParamSpec("beam_shifts", "GH/IF Beam Shifts", "bool", False,
                      tooltip="Enable Goos-Hänchen and Imbert-Fedorov shifts"),

            # Aperture / visualisation
            ParamSpec("radius", "Aperture Radius", "float", 0.05, "m",
                      min=0.0, max=100.0,
                      tooltip="Mirror aperture radius, measured from the mirror "
                              "centre (also used for the 3-D viewer disc). Rays "
                              "landing outside this radius are blocked. "
                              "0 = unbounded (no clipping)."),

            # ── wavelength-dependent n/k (LnkMixin) ──────────────────────
            *cls.lnk_param_specs(),
        ]

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        f   = float(self.params["focal_length"])

        mirror_center = np.array([
            float(self.params["r_x"]),
            float(self.params["r_y"]),
            float(self.params["r_z"]),
        ], dtype=float)

        k_in_dir = np.array([
            float(self.params["k_in_x"]),
            float(self.params["k_in_y"]),
            float(self.params["k_in_z"]),
        ], dtype=float)

        k_out_dir = np.array([
            float(self.params["k_out_x"]),
            float(self.params["k_out_y"]),
            float(self.params["k_out_z"]),
        ], dtype=float)

        if np.linalg.norm(k_in_dir) < 1e-10:
            raise ValueError("k_in direction vector is zero.")
        if np.linalg.norm(k_out_dir) < 1e-10:
            raise ValueError("k_out direction vector is zero.")

        beam_shifts = bool(self.params.get("beam_shifts", False))

        radius = float(self.params.get("radius", 0.0))

        out_photons = []
        info_list   = []
        n_blocked   = 0

        # Build mirror once per unique (n, k, lam) combination.
        # In practice most scenes use a single wavelength, so a simple
        # per-photon build is clean and fast enough.  The geometry
        # (mirror_center, k_in, k_out, f) is always the same.
        _mirror_cache: dict[tuple, object] = {}

        for ph in photons:
            lam = float(ph.wavelength)
            n, k_ext = self._resolve_nk(lam, n_key="n", k_key="k_ext")

            cache_key = (round(n, 10), round(k_ext, 10), round(lam, 18))
            if cache_key not in _mirror_cache:
                try:
                    _mirror_cache[cache_key] = OffAxisParaboloid(
                        n, k_ext, lam, f,
                        mirror_center, k_in_dir, k_out_dir,
                        beam_shifts=beam_shifts,
                    )
                except Exception as e:
                    out_photons.append(ph)
                    info_list.append({"ok": False, "error": str(e)})
                    continue

            mirror = _mirror_cache[cache_key]
            vacuum = Material.vacuum(lam)

            try:
                ph_r, ph_at = mirror.reflect(ph, vacuum)

                # ── aperture clipping (projected onto the plane ⊥ k_in) ──
                # The mirror's clear aperture is a circle of *projected* radius
                # `radius`, measured in the plane perpendicular to the design
                # incoming beam direction k_in_hat (exactly how a real OAP's
                # clear aperture is specified) and centred on mirror_center.
                # Using the raw 3-D distance to the (curved) hit point instead
                # would over- or under-count the aperture depending on how
                # strongly the patch is curved / tilted, so the offset is
                # projected transverse to k_in_hat first, matching the curved
                # surface patch drawn in the 3-D viewer.
                if radius > 0:
                    k_in_hat_ap = k_in_dir / np.linalg.norm(k_in_dir)
                    delta = np.real(ph_at.r).astype(float) - mirror_center
                    transverse = delta - np.dot(delta, k_in_hat_ap) * k_in_hat_ap
                    d_to_axis = float(np.linalg.norm(transverse))
                    if d_to_axis > radius:
                        n_blocked += 1
                        info_list.append({
                            "ok": False,
                            "blocked": True,
                            "distance_from_center": round(d_to_axis, 6),
                            "radius": radius,
                        })
                        continue

                out_photons.append(ph_r)

                k_hat_in = np.real(ph.k) / np.linalg.norm(np.real(ph.k))
                pt_c  = mirror._to_canonical_pos(np.real(ph_at.r).astype(float))
                khat_c = mirror._to_canonical_dir(k_hat_in)
                n_c   = mirror._normal_canonical(pt_c, khat_c)
                n_g   = mirror._to_global_dir(n_c)
                theta_i = float(np.degrees(np.arccos(
                    np.clip(abs(np.dot(k_hat_in, n_g)), 0.0, 1.0)
                )))

                info_list.append({
                    "ok":               True,
                    "theta_i_deg":      round(theta_i, 3),
                    "reflection_point": np.real(ph_at.r).tolist(),
                    "I_in":             float(ph.I),
                    "I_out":            float(ph_r.I),
                    "n_used":           round(n, 6),
                    "k_used":           round(k_ext, 6),
                    "wavelength_nm":    round(lam * 1e9, 3),
                })
            except Exception as e:
                out_photons.append(ph)
                info_list.append({"ok": False, "error": str(e)})

        # Use the last successfully built mirror for reporting geometry
        last_mirror = next(reversed(_mirror_cache.values()), None)

        return out_photons, {
            "nk_mode":      self.params.get("nk_mode", "manual"),
            "focal_length": f,
            "off_axis_deg": round(float(np.degrees(last_mirror.off_axis_angle)), 2) if last_mirror else None,
            "focus_position": last_mirror.F.tolist() if last_mirror else None,
            "mirror_center":  mirror_center.tolist(),
            "radius":         radius,
            "n_blocked":      n_blocked,
            "reflections":    info_list,
        }