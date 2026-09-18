"""
modules/source.py  – Photon source (generates photons)
modules/propagate.py – Free-space propagation
modules/beamsplitter.py – 50/50 or arbitrary beam splitter
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon, Material, Surface
from .base import OpticalModule, ParamSpec


# ══════════════════════════════════════════════════════════════════════════════
# PhotonSource
# ══════════════════════════════════════════════════════════════════════════════

class PhotonSource(OpticalModule):
    MODULE_TYPE = "photon_source"
    LABEL = "Photon Source"
    COLOR = "#f5c518"
    ICON = "☀"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec("r_x", "R_x", "float", -0.1, "m"),
            ParamSpec("r_y", "R_y", "float",  0.0, "m"),
            ParamSpec("r_z", "R_z", "float",  0.0, "m"),
            ParamSpec("k_x", "k_x", "float",  1.0, ""),
            ParamSpec("k_y", "k_y", "float",  0.0, ""),
            ParamSpec("k_z", "k_z", "float",  0.0, ""),
            ParamSpec("E_x", "E_x", "float", 0.0, ""),
            ParamSpec("E_y", "E_y", "float", 0.0, ""),
            ParamSpec("E_z", "E_z", "float", 1.0, ""),
            ParamSpec("Phase_x", "Phase_x", "float", 0.0, "deg"),
            ParamSpec("Phase_y", "Phase_y", "float", 0.0, "deg"),
            ParamSpec("Phase_z", "Phase_z", "float", 0.0, "deg"),
            ParamSpec("wavelength", "Wavelength", "float", 800e-9, "m",
                      min=200e-9, max=10_000e-9),
            ParamSpec("intensity", "Intensity", "float", 1.0, "a.u.",
                      min=0.0, max=1e6),

            # pass-through
            ParamSpec("pass_through", "Pass-through", "bool", True,
                      tooltip="If enabled, incoming photons are forwarded unchanged "
                              "and new photons are added on top of them."),

            # circle beam pattern
            ParamSpec("n_circles", "N Rings", "int", 0, "",
                      min=0, max=20,
                      tooltip="Number of concentric rings of photons (0 = single central ray only)"),
            ParamSpec("circle_radius", "Ring Radius", "float", 5e-3, "m",
                      min=0.0, max=10.0,
                      tooltip="Transverse radius of the first ring; ring n uses n × this value"),
            ParamSpec("photons_per_circle", "Photons / Ring", "int", 8, "",
                      min=1, max=360,
                      tooltip="Number of photons placed uniformly on each ring"),
            ParamSpec("opening_angle", "Ring Half-angle", "float", 0.0, "°",
                      min=0, max=90,
                      tooltip="half angle of the first Ring, every ring opens with half-angle x N"
                            "using this, overwrites Focus Distance"),
            ParamSpec("focus_distance", "Focus Distance", "float", 0.0, "m",
                      min=-1e9, max=1e9,
                      tooltip="Virtual focus/source distance for the ring photons. 0 means infinity -> collimated (all k-vectors parallel).  "
                              "Positive: rays converge to a focus at that distance ahead.  "
                              "Negative: rays diverge as if emitted from a virtual point "
                              "source that many metres behind the source plane."),

            # fan (needed for backward compatibility. I think. not sure why, but when I delete it i get problems with load/save) 
            ParamSpec("n_rays", "Fan Rays", "int", 0, "",
                      min=1, max=100,
                      tooltip="Number of rays in a 1-D fan spread (x-z-plane)"),
            ParamSpec("fan_angle_deg", "Fan Half-angle", "float", 0.0, "°",
                      min=0.0, max=90.0,
                      tooltip="Half-angle of the ray fan (0 = single ray)"),
        ]

    # helpers

    @staticmethod
    def _build_transverse_basis(k_hat: np.ndarray):
        """Return two unit vectors (u, v) perpendicular to k_hat."""
        # Pick a reference that is not collinear with k_hat
        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(k_hat, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(k_hat, ref)
        u /= np.linalg.norm(u)
        v = np.cross(k_hat, u)
        v /= np.linalg.norm(v)
        return u, v

    def _make_ring_photons(
        self,
        r0: np.ndarray,          # source position (real float 3-vector)
        k0: np.ndarray,          # central k-vector (real float 3-vector, already scaled)
        E:  np.ndarray,          # polarisation (complex 3-vector, normalised)
        lam: float,
        I:   float,
    ) -> list[Photon]:
        """Generate central ray plus N_circles rings of photons."""
        n_circles = int(self.params["n_circles"])
        r_step    = float(self.params["circle_radius"])
        n_per     = int(self.params["photons_per_circle"])
        f_angle    = float(self.params["opening_angle"])
        f_dist    = float(self.params["focus_distance"])

        k_hat  = k0 / np.linalg.norm(k0)
        k_mag  = np.linalg.norm(k0)
        u, v   = self._build_transverse_basis(k_hat)

        out: list[Photon] = []

        # Central ray always present
        out.append(Photon(
            r0.astype(complex), k0.astype(complex),
            E.copy(), np.zeros(3, dtype=complex), lam, I,
        ))

        for ci in range(1, n_circles + 1):
            ring_r = ci * r_step          # transverse radius of this ring
            angles = np.linspace(0.0, 2 * np.pi, n_per, endpoint=False)
            for phi in angles:
                # Transverse offset in the source plane
                offset = ring_r * (np.cos(phi) * u + np.sin(phi) * v)
                r_ph   = r0 + offset

                if f_angle > 0:
                    c_angle = np.radians(ci * f_angle)

                    r_hat = np.cos(phi) * u + np.sin(phi) * v

                    kv_hat = (
                        np.cos(c_angle) * k_hat
                        + np.sin(c_angle) * r_hat
                    )

                    kv = kv_hat * k_mag
                elif f_dist != 0.0:
                    # virtual_point is the convergence/divergence origin.
                    # Positive f_dist: focus ahead -> k points FROM r_ph TOWARD focus.
                    # Negative f_dist: virtual source behind -> k points FROM that
                    #   source THROUGH r_ph and onward (diverging beam).
                    # Both cases: k direction = r_ph - virtual_point, then flip if
                    # f_dist > 0 so that k aims toward the focus, not away from it.
                    virtual_point = r0 + f_dist * k_hat
                    if f_dist > 0.0:
                        # converging: k points from ring photon toward focus
                        kv = virtual_point - r_ph
                    else:
                        # diverging: k points away from the virtual source
                        kv = r_ph - virtual_point
                    kv = kv / np.linalg.norm(kv) * k_mag
                else:
                    # Parallel beam - same k as central ray
                    kv = k0.copy()

                out.append(Photon(
                    r_ph.astype(complex), kv.astype(complex),
                    E.copy(), np.zeros(3, dtype=complex), lam, I,
                ))

        return out

    # ── process ───────────────────────────────────────────────────────────────

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        lam = float(self.params["wavelength"])
        r0 = np.array([float(self.params["r_x"]),
                       float(self.params["r_y"]),
                       float(self.params["r_z"])], dtype=float)
        k0 = np.array([float(self.params["k_x"]),
                       float(self.params["k_y"]),
                       float(self.params["k_z"])], dtype=float)
        k0 = k0 / np.linalg.norm(k0) * (2 * np.pi / lam)

        E = np.array([
            float(self.params["E_x"]) * np.exp(1j * np.deg2rad(float(self.params["Phase_x"]))),
            float(self.params["E_y"]) * np.exp(1j * np.deg2rad(float(self.params["Phase_y"]))),
            float(self.params["E_z"]) * np.exp(1j * np.deg2rad(float(self.params["Phase_z"]))),
        ], dtype=complex)
        E_norm = np.linalg.norm(np.abs(E))
        if E_norm > 0:
            E = E / E_norm

        I        = float(self.params["intensity"])
        n_r      = int(self.params.get("n_rays", 1))
        fan      = float(self.params.get("fan_angle_deg", 0.0))
        n_circles = int(self.params.get("n_circles", 0))

        new_photons: list[Photon] = []

        # ── ring / circle pattern (takes precedence over legacy fan) ──────────
        if n_circles > 0:
            new_photons = self._make_ring_photons(r0, k0, E, lam, I)

        # ── legacy fan ────────────────────────────────────────────────────────
        elif n_r == 1 or fan == 0.0:
            new_photons.append(
                Photon(r0.astype(complex), k0.astype(complex),
                       E, np.zeros(3, dtype=complex), lam, I)
            )
        else:
            k_hat = k0 / np.linalg.norm(k0)
            perp  = np.cross(k_hat, np.array([0.0, 1.0, 0.0]))
            if np.linalg.norm(perp) < 1e-10:
                perp = np.cross(k_hat, np.array([1.0, 0.0, 0.0]))
            perp = perp / np.linalg.norm(perp)
            for a in np.linspace(-np.radians(fan), np.radians(fan), n_r):
                kv = np.cos(a) * k_hat + np.sin(a) * perp
                kv = kv / np.linalg.norm(kv) * np.linalg.norm(k0)
                new_photons.append(
                    Photon(r0.astype(complex), kv.astype(complex),
                           E.copy(), np.zeros(3, dtype=complex), lam, I)
                )

        # ── pass-through: merge incoming + new ───────────────────────────────
        if bool(self.params.get("pass_through", False)):
            out = list(photons) + new_photons
        else:
            out = new_photons

        return out, {
            "n_generated":   len(new_photons),
            "n_passed":      len(photons) if bool(self.params.get("pass_through", False)) else 0,
            "n_out":         len(out),
            "n_circles":     n_circles,
            "focus_distance": float(self.params.get("focus_distance", 0.0)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FreePropagate
# ══════════════════════════════════════════════════════════════════════════════

class FreePropagate(OpticalModule):
    MODULE_TYPE = "free_propagate"
    LABEL = "Free Propagation"
    COLOR = "#34d399"
    ICON = "->"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec("distance", "Distance", "float", 0.10, "m",
                      min=0.0, max=1000.0,
                      tooltip="Free-space travel distance"),
        ]

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        d = float(self.params["distance"])
        out = [ph.travel_distance(d) for ph in photons]
        return out, {"distance": d}


# ══════════════════════════════════════════════════════════════════════════════
# Detector / absorber — useful as pipeline terminus
# ══════════════════════════════════════════════════════════════════════════════

class Detector(OpticalModule):
    MODULE_TYPE = "detector"
    LABEL = "Detector"
    COLOR = "#6b7280"
    ICON = "⦿"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec("label", "Label", "str", "Det", tooltip="Detector label (cosmetic)"),

            # Position — same convention as FlatMirror
            ParamSpec("r_x", "Position X", "float", 0.5, "m",
                      min=-100.0, max=100.0, tooltip="Detector plane center X"),
            ParamSpec("r_y", "Position Y", "float", 0.0, "m",
                      min=-100.0, max=100.0, tooltip="Detector plane center Y"),
            ParamSpec("r_z", "Position Z", "float", 0.0, "m",
                      min=-100.0, max=100.0, tooltip="Detector plane center Z"),

            # Surface normal — defines the orientation of the detector plane
            ParamSpec("normal_x", "Normal X", "float", -1.0, "",
                      min=-1.0, max=1.0, tooltip="Detector surface normal X (pointing toward incoming beam)"),
            ParamSpec("normal_y", "Normal Y", "float",  0.0, "",
                      min=-1.0, max=1.0, tooltip="Detector surface normal Y"),
            ParamSpec("normal_z", "Normal Z", "float",  0.0, "",
                      min=-1.0, max=1.0, tooltip="Detector surface normal Z"),

            # Aperture / visualisation
            ParamSpec("radius", "Detector Radius", "float", 0.05, "m",
                      min=0.0, max=100.0,
                      tooltip="Detector disc radius (also used for the 3-D "
                              "viewer disc). Rays landing outside this radius "
                              "are not detected. 0 = unbounded (no clipping)."),
        ]

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        r_det = np.array([
            float(self.params["r_x"]),
            float(self.params["r_y"]),
            float(self.params["r_z"]),
        ], dtype=float)

        normal = np.array([
            float(self.params["normal_x"]),
            float(self.params["normal_y"]),
            float(self.params["normal_z"]),
        ], dtype=float)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-12:
            normal = normal / norm_len
        else:
            normal = np.array([0.0, 0.0, 1.0])

        radius = float(self.params.get("radius", 0.0))

        detected_info = []
        photon_status = []   # 1:1 with input photons; used by scene.py to
                              # correctly re-associate output photons with
                              # their input when some rays are blocked.
        out_photons: list[Photon] = []
        n_blocked = 0

        for ph in photons:
            r_ph = np.real(ph.r).astype(float)
            k_ph = np.real(ph.k).astype(float)

            k_norm = np.linalg.norm(k_ph)
            k_hat = k_ph / (k_norm + 1e-30) if k_norm > 1e-30 else np.array([0.0, 0.0, 0.0])
            k_dot_n = float(np.dot(k_hat, normal))

            if abs(k_dot_n) < 1e-20:
                hit = r_ph.copy()
                t_dist = 0.0
            else:
                t_dist = float(np.dot(r_det - r_ph, normal)) / k_dot_n
                hit = r_ph + k_hat * t_dist

            if abs(t_dist) > 1e-15:
                ph_hit = ph.travel_distance(t_dist)
            else:
                ph_hit = ph

            # ── aperture clipping ────────────────────────────────────────
            if radius > 0:
                d_to_axis = float(np.linalg.norm(np.real(ph_hit.r).astype(float) - r_det))
                if d_to_axis > radius:
                    n_blocked += 1
                    photon_status.append({"blocked": True})
                    continue

            out_photons.append(ph_hit)
            photon_status.append({"blocked": False})

            # Full complex E-field info
            E = ph.E  # complex (3,)
            detected_info.append({
                "label":         self.params.get("label", ""),
                "wavelength_nm": round(float(ph.wavelength) * 1e9, 6),
                "I":             float(ph.I),
                "hit_position":  [round(float(v), 9) for v in hit],
                # Position at detection (after propagation)
                "r": [round(float(v), 9) for v in np.real(ph_hit.r)],
                # Wave-vector at detection
                "k": [round(float(v), 9) for v in np.real(ph_hit.k)],
                # Complex E-field components: real, imag, amplitude, phase
                "E": {
                    "x_real":  round(float(E[0].real), 9),
                    "x_imag":  round(float(E[0].imag), 9),
                    "x_amp":   round(float(abs(E[0])), 9),
                    "x_phase": round(float(np.angle(E[0])), 9),
                    "y_real":  round(float(E[1].real), 9),
                    "y_imag":  round(float(E[1].imag), 9),
                    "y_amp":   round(float(abs(E[1])), 9),
                    "y_phase": round(float(np.angle(E[1])), 9),
                    "z_real":  round(float(E[2].real), 9),
                    "z_imag":  round(float(E[2].imag), 9),
                    "z_amp":   round(float(abs(E[2])), 9),
                    "z_phase": round(float(np.angle(E[2])), 9),
                },
                "time_traveled": round(float(ph_hit.t), 18), #make sure this is short enough. we have to round for the frontend, but still need attosecond resolution
            })

        label = self.params.get("label", "")
        return out_photons, {
            "detector_label": label,
            "n_detected":     len(detected_info),
            "detected":       detected_info,
            "position":       r_det.tolist(),
            "normal":         normal.tolist(),
            "radius":         radius,
            "n_blocked":      n_blocked,
            "photon_status":  photon_status,
        }

# ══════════════════════════════════════════════════════════════════════════════
# UploadPhotons  –  replay photons captured by a Detector export
# ══════════════════════════════════════════════════════════════════════════════

class UploadPhotons(OpticalModule):
    """
    Replays a photon bundle that was previously exported from a Detector node.

    The ``photon_data`` param holds the raw JSON string produced by
    ``downloadDetectorJson()`` in the frontend.  On ``process()`` the module
    reconstructs Photon objects from that data and (optionally) re-centres them
    by subtracting the detector position so the bundle origin becomes (0,0,0).

    Incoming photons from the pipeline are forwarded unchanged when
    ``pass_through`` is True (mirrors the behaviour of PhotonSource).
    """

    MODULE_TYPE = "upload_photons"
    LABEL       = "Upload Photons"
    COLOR       = "#a78bfa"   # violet – visually distinct from PhotonSource amber
    ICON        = "⬆"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "photon_data", "Photon Data (JSON)", "json_blob", "",
                tooltip="Paste or upload the JSON exported from a Detector node.",
            ),
            ParamSpec(
                "subtract_detector_pos", "Re-centre at origin", "bool", True,
                tooltip=(
                    "Subtract the detector position from every photon r-vector "
                    "so the bundle starts at (0, 0, 0).  The k-vectors are "
                    "unchanged."
                ),
            ),
            ParamSpec(
                "pass_through", "Pass-through", "bool", False,
                tooltip=(
                    "Forward any photons arriving from upstream unchanged and "
                    "add the uploaded bundle on top of them."
                ),
            ),
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_photons(self) -> tuple[list[Photon], dict]:
        """
        Parse ``self.params['photon_data']`` and return a list of Photons plus
        a small diagnostic dict.
        """
        raw = self.params.get("photon_data", "")
        if not raw:
            return [], {"error": "no data loaded"}

        try:
            payload = raw if isinstance(raw, dict) else __import__("json").loads(raw)
        except Exception as exc:
            return [], {"error": f"JSON parse error: {exc}"}

        # Support both the bare list format [photon, ...] and the wrapped
        # detector export format {"info": {"detected": [...], "position": [...]}}
        detected: list[dict] = []
        det_pos = np.zeros(3, dtype=float)

        if isinstance(payload, list):
            detected = payload
        elif isinstance(payload, dict):
            info = payload.get("info", payload)
            detected = info.get("detected", [])
            raw_pos  = info.get("position", [0.0, 0.0, 0.0])
            det_pos  = np.array(raw_pos, dtype=float)

        subtract = bool(self.params.get("subtract_detector_pos", True))
        offset   = det_pos if subtract else np.zeros(3, dtype=float)

        photons: list[Photon] = []
        for d in detected:
            try:
                r_vec   = np.array(d.get("r") or d.get("hit_position", [0, 0, 0]),
                                 dtype=float) - offset
                k_vec   = np.array(d.get("k", [1.0, 0.0, 0.0]), dtype=float)
                lam     = float(d.get("wavelength_nm", 800.0)) * 1e-9
                I       = float(d.get("I", 1.0))
                t       = float(d.get("time_traveled", 0.0))

                E_data = d.get("E", {})
                E = np.array([
                    complex(E_data.get("x_real", 0.0), E_data.get("x_imag", 0.0)),
                    complex(E_data.get("y_real", 0.0), E_data.get("y_imag", 0.0)),
                    complex(E_data.get("z_real", 0.0), E_data.get("z_imag", 0.0)),
                ], dtype=complex)
                E_norm = np.linalg.norm(np.abs(E))
                if E_norm < 1e-30:
                    E = np.array([0.0, 0.0, 1.0], dtype=complex)
                else:
                    E = E / E_norm

                photons.append(Photon(
                    r_vec.astype(complex),
                    k_vec.astype(complex),
                    E,
                    np.zeros(3, dtype=complex),
                    lam,
                    I,
                    t,
                ))
            except Exception:
                continue   # skip malformed entries

        info_out = {
            "n_loaded":       len(photons),
            "detector_pos":   det_pos.tolist(),
            "subtract_offset": subtract,
        }
        return photons, info_out

    # ── process ───────────────────────────────────────────────────────────────

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        new_photons, load_info = self._load_photons()

        if bool(self.params.get("pass_through", False)):
            out = list(photons) + new_photons
        else:
            out = new_photons

        return out, {
            **load_info,
            "n_passed":   len(photons) if bool(self.params.get("pass_through", False)) else 0,
            "n_out":      len(out),
        }