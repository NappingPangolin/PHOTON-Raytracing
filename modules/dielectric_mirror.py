"""
modules/dielectric_mirror.py
-----------------------------
DielectricMirror — a multilayer (DBR-style) dielectric stack that reports
only the two "headline" output beams: the fully reflected beam and the
fully transmitted beam, computed from standard thin-film transfer-matrix
(Rouard) theory — the same physics as ``dielectric_mirror_model.py``.

Geometry
~~~~~~~~
Like Substrate, this is an N-layer stack (1..MAX_LAYERS) measured along a
single shared axis. Unlike Substrate:

  - There is only **one** surface normal (`normal_x/y/z`), used for every
    interface in the stack — no per-surface wedging.
  - There is no internal-reflection bookkeeping: instead of following every
    individual bounce between layers, the whole stack's amplitude
    reflectivity/transmissivity is computed in one shot via the recursive
    Fresnel (Rouard) method (see `dielectric_mirror_model.py`), which
    already sums the full multiple-beam interference for you. This is
    standard thin-film / DBR theory — physically equivalent to tracing
    every internal bounce to infinity, just solved in closed form.

Every input photon therefore produces **exactly two** output photons:
a reflected beam (law-of-reflection direction off the front surface) and a
transmitted beam (continues in the incident direction, exits the back of
the stack). This mirrors BeamSplitter's interleaved-pairs convention
(`out[2*i]` = reflected, `out[2*i+1]` = transmitted) rather than
Substrate's variable-count one.

Polarisation
~~~~~~~~~~~~
The incident field is decomposed into s/p components exactly as
`Surface.reflect()` does. Each component picks up the stack's own complex
amplitude reflectivity (`sqrt(R) * exp(i*phase)`, from the Rouard
recursion) instead of a single-interface Fresnel coefficient. The
transmitted beam keeps the incident polarisation state and direction
(no refraction is modelled — only overall power splitting), with
`I_t = I_i - I_r` enforcing energy conservation.
Front/back sidedness
~~~~~~~~~~~~~~~~~~~~
DielectricMirror is a *sided* module (``SIDED = True``). A front-input
photon sees the stack exactly as described above: vacuum -> layer 1..N ->
backing medium. A back-input photon physically approaches from beyond the
backing medium, so it is traced through the equivalent *flipped* stack:
incidence medium = the backing medium, layers in reverse order (N..1), and
the far medium = vacuum (what was originally in front) — i.e. the whole
transfer-matrix calculation is re-run with the incidence and substrate
media swapped and the layer list reversed, which is the physically correct
description of illuminating the same physical stack from the far side.
Reflected beams always stay on the side they entered (front-in ->
front-out, back-in -> back-out); transmitted beams always cross to the
opposite side.
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon
from .base import OpticalModule, LnkMixin, ParamSpec

MAX_LAYERS = 16   # maximum number of stacked dielectric layers

def _compute_eta(n, kz, pol):
    return kz if pol == 's' else n**2 / kz

def _reflectivity_phase_oblique(lambda_vac, n_list, d_list, n0=1.0, ns=1.5,
                                theta0=0.0, pol='s'):
    """
    Computes reflection using the recursive Fresnel (Rouard) method,
    which is numerically stable for absorbing layers of any thickness.
    """
    k0 = 2 * np.pi / lambda_vac
    kx = n0 * np.sin(theta0)

    def kz_of(n):
        kz = np.sqrt(n**2 - kx**2 + 0j)
        if np.imag(kz) < 0:
            kz = -kz
        return kz

    # Build full list: [incident, layer1, ..., layerN, substrate]
    n_all = [n0] + list(n_list) + [ns]
    kz_all = [kz_of(n) for n in n_all]
    eta_all = [_compute_eta(n, kz, pol) for n, kz in zip(n_all, kz_all)]

    # Start from the substrate and recurse backwards (Rouard method)
    # r_eff starts as the reflection at the last interface (layer N -> substrate)
    r_eff = (eta_all[-2] - eta_all[-1]) / (eta_all[-2] + eta_all[-1])

    # Walk backwards through layers (excluding incident medium and substrate)
    for i in range(len(n_list) - 1, -1, -1):
        eta_i  = eta_all[i + 1]   # current layer admittance
        eta_next = eta_all[i + 2] # next layer/substrate admittance (already in r_eff)
        d_i    = d_list[i]
        kz_i   = kz_all[i + 1]

        # Interface from layer above (i) into current layer (i+1)
        r_interface = (eta_all[i] - eta_i) / (eta_all[i] + eta_i)

        # Phase factor for round trip through current layer
        phi = k0 * kz_i * d_i
        P = np.exp(2j * phi)   # round-trip phase (exp decays for Im(phi)>0: stable!)

        # Rouard combination formula
        r_eff = (r_interface + r_eff * P) / (1 + r_interface * r_eff * P)

    R = np.abs(r_eff)**2
    phase = np.angle(r_eff)
    return R, phase

def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


class DielectricMirror(LnkMixin, OpticalModule):
    MODULE_TYPE = "dielectric_mirror"
    LABEL = "Dielectric Mirror"
    COLOR = "#22c1d6"
    ICON = "▥"
    SIDED = True

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        specs = [
            # ── front-surface position ────────────────────────────────────
            ParamSpec("r_x", "R_x", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Front surface centre position X"),
            ParamSpec("r_y", "R_y", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Front surface centre position Y"),
            ParamSpec("r_z", "R_z", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Front surface centre position Z"),

            # ── single shared normal (stack axis) ──────────────────────────
            ParamSpec("normal_x", "N_x", "float", -1.0, "",
                      min=-1.0, max=1.0,
                      tooltip="Surface normal / stacking axis, shared by "
                              "every layer interface (no wedging)"),
            ParamSpec("normal_y", "N_y", "float", 0.0, "", min=-1.0, max=1.0),
            ParamSpec("normal_z", "N_z", "float", 0.0, "", min=-1.0, max=1.0),

            # ── stack size / aperture ──────────────────────────────────────
            ParamSpec("n_layers", "Number of Layers", "int", 2, "",
                      min=1, max=MAX_LAYERS,
                      tooltip="Number of dielectric layers in the stack "
                              "(e.g. alternating high/low index pairs for "
                              "a DBR)"),
            ParamSpec("diameter", "Clear Aperture ⌀", "float", 0.05, "m",
                      min=0.0, max=100.0,
                      tooltip="Clear-aperture diameter. 0 = unbounded "
                              "(no clipping)."),

            # ── backing medium (semi-infinite, behind the last layer) ─────
            ParamSpec("n_substrate", "Substrate n", "float", 1.0, "",
                      min=0.0, max=10.0,
                      tooltip="Real index of the semi-infinite medium "
                              "behind the last layer (1.0 = free-standing "
                              "stack in air/vacuum; set e.g. 1.52 for a "
                              "glass-backed coating)"),
            ParamSpec("k_substrate", "Substrate k", "float", 0.0, "",
                      min=0.0, max=30.0,
                      tooltip="Extinction coefficient of the backing medium"),
            ParamSpec("wavelength", "Wavelength", "float", 800e-9, "m",
                      min=1e-12, max=10_000e-9,
                      tooltip="Reference wavelength (manual mode only)"),
        ]

        # Per-layer thickness / n / k / lnk table (layers 1..MAX)
        for i in range(1, MAX_LAYERS + 1):
            specs += [
                ParamSpec(f"t_{i}", f"Thickness t{i}", "float", 100e-9, "m",
                          min=1e-9, max=10.0,
                          tooltip=f"Physical thickness of layer {i}"),
                ParamSpec(f"n_{i}", f"n{i} (real)", "float", 1.45, "",
                          min=1.0, max=5.0,
                          tooltip=f"Real refractive index of layer {i} "
                                  f"(manual mode)"),
                ParamSpec(f"k_{i}", f"k{i} (extinction)", "float", 0.0, "",
                          min=0.0, max=10.0,
                          tooltip=f"Extinction coefficient of layer {i} "
                                  f"(manual mode)"),
                ParamSpec(f"nk_mode_{i}", f"n/k Source {i}", "select", "manual",
                          options=["manual", "lnk_table"]),
                ParamSpec(f"lnk_data_{i}", f"n/k Table {i} (lnk)", "lnk_table", []),
            ]

        return specs

    # ------------------------------------------------------------------
    # Geometry / material resolution
    # ------------------------------------------------------------------

    def _layer_nk(self, i: int, lam: float) -> tuple[float, float, str]:
        p = self.params
        mode = p.get(f"nk_mode_{i}", "manual")
        rows = p.get(f"lnk_data_{i}", [])
        if mode == "lnk_table" and rows:
            try:
                from .lnk_table import LnkTable
                tbl = LnkTable.from_rows(rows)
                n_i, k_i = tbl.get_nk(lam)
                return n_i, k_i, "lnk_table"
            except Exception:
                pass
        return float(p[f"n_{i}"]), float(p[f"k_{i}"]), "manual"

    def _geometry(self, wavelength_m: float, side: str = "front") -> dict:
        """
        Resolve stack geometry for illumination from `side`.

        front: entry point = r_front, entry normal = normal (as configured),
               layers in configured order 1..N, incidence medium = vacuum
               (n0=1), far medium = the configured backing substrate.

        back:  entry point = the far plane (r_front + total_thickness * normal),
               entry normal = -normal (facing the oncoming back-side beam),
               layers in *reverse* order N..1, incidence medium = the
               configured backing substrate, far medium = vacuum — i.e. the
               same physical stack, correctly re-described for illumination
               from the opposite face (see module docstring).
        """
        p = self.params
        N = max(1, min(MAX_LAYERS, int(p.get("n_layers", 2))))

        r_front = np.array([float(p["r_x"]), float(p["r_y"]), float(p["r_z"])])
        normal = _normalise(np.array([
            float(p["normal_x"]), float(p["normal_y"]), float(p["normal_z"]),
        ]))
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([1.0, 0.0, 0.0])

        d_list, n_layers, k_layers, sources = [], [], [], []
        for i in range(1, N + 1):
            n_i, k_i, src = self._layer_nk(i, wavelength_m)
            d_list.append(float(p[f"t_{i}"]))
            n_layers.append(n_i)
            k_layers.append(k_i)
            sources.append(src)

        n_sub = float(p.get("n_substrate", 1.0))
        k_sub = float(p.get("k_substrate", 0.0))
        total_thickness = float(sum(d_list))

        if side == "back":
            entry_point = r_front + total_thickness * normal
            entry_normal = -normal
            exit_point = r_front
            d_list = list(reversed(d_list))
            n_layers = list(reversed(n_layers))
            k_layers = list(reversed(k_layers))
            sources = list(reversed(sources))
            n_incidence, k_incidence = n_sub, k_sub
            n_far, k_far = 1.0, 0.0
        else:
            entry_point = r_front
            entry_normal = normal
            exit_point = r_front + total_thickness * normal
            n_incidence, k_incidence = 1.0, 0.0
            n_far, k_far = n_sub, k_sub

        return dict(
            N=N, r_front=r_front, normal=normal,
            entry_point=entry_point, entry_normal=entry_normal, exit_point=exit_point,
            d_list=d_list, n_layers=n_layers, k_layers=k_layers,
            nk_sources=sources,
            n_incidence=n_incidence, k_incidence=k_incidence,
            n_far=n_far, k_far=k_far,
            n_sub=n_sub, k_sub=k_sub,
            total_thickness=total_thickness,
        )

    # ------------------------------------------------------------------
    # Per-photon reflect/transmit
    # ------------------------------------------------------------------

    def _process_one(self, ph: Photon, g: dict, diameter: float):
        """
        Returns (photon_r, photon_t, diag) or (None, None, diag) if the ray
        misses the clear aperture / runs parallel to the stack.
        """
        r_entry, normal0 = g["entry_point"], g["entry_normal"]
        lam = float(ph.wavelength)

        r0 = np.real(ph.r).astype(float)
        k_hat = _normalise(np.real(ph.k).astype(float))

        denom = float(np.dot(k_hat, normal0))
        if abs(denom) < 1e-12:
            return None, None, {"blocked": True, "reason": "parallel to stack"}
        t_hit = float(np.dot(r_entry - r0, normal0) / denom)
        r_hit = r0 + t_hit * k_hat

        if diameter > 0:
            h = diameter / 2.0
            transverse = r_hit - r_entry
            transverse = transverse - np.dot(transverse, normal0) * normal0
            if float(np.linalg.norm(transverse)) > h:
                return None, None, {"blocked": True,
                                     "distance_from_center": round(float(np.linalg.norm(transverse)), 6),
                                     "radius": h}

        ph_at = ph.travel_distance(t_hit)
        ph_at.r = r_hit.astype(complex)

        # Orient the working normal to face the incoming beam (same
        # convention as Surface.reflect()).
        normal = normal0.copy()
        theta_i = np.arccos(np.clip(-np.dot(k_hat, normal), -1.0, 1.0))
        if theta_i > np.pi / 2:
            normal = -normal
            theta_i = np.arccos(np.clip(-np.dot(k_hat, normal), -1.0, 1.0))

        # --- s/p decomposition of the (unit-normalised) incident field ---
        e_norm = np.linalg.norm(ph_at.E)
        E_in = ph_at.E / e_norm if e_norm > 0 else ph_at.E.copy()

        Sn = np.cross(normal, k_hat)
        sn_norm = np.linalg.norm(Sn)
        if sn_norm < 1e-12:
            Eip, Eis = E_in.copy(), np.zeros(3, dtype=complex)
        else:
            Sn = Sn / sn_norm
            Pn_i = -np.cross(Sn, k_hat)
            Eis = np.dot(Sn, E_in) * Sn
            Eip = np.dot(Pn_i, E_in) * Pn_i

        # --- stack amplitude reflectivity (Rouard / transfer-matrix) ------
        # n0/ns already carry the correct incidence/far media for whichever
        # side this photon entered from (see _geometry: front uses vacuum
        # incidence + the configured backing substrate as the far medium;
        # back swaps the two and reverses the layer order).
        n_list = [complex(n, -k) for n, k in zip(g["n_layers"], g["k_layers"])]
        n0 = complex(g["n_incidence"], -g["k_incidence"])
        ns = complex(g["n_far"], -g["k_far"])

        R_s, phase_s = reflectivity_phase_oblique(
            lam, n_list, g["d_list"], n0=n0, ns=ns, theta0=theta_i, pol='s')
        R_p, phase_p = reflectivity_phase_oblique(
            lam, n_list, g["d_list"], n0=n0, ns=ns, theta0=theta_i, pol='p')
        R_s = float(np.clip(R_s, 0.0, 1.0))
        R_p = float(np.clip(R_p, 0.0, 1.0))

        reff_s = np.sqrt(R_s) * np.exp(1j * phase_s)
        reff_p = np.sqrt(R_p) * np.exp(1j * phase_p)

        Ers = Eis * reff_s
        if sn_norm < 1e-12:
            Erp = -Eip
        else:
            Eip_scalar = np.dot(Pn_i, E_in)
            Pn_r = Pn_i - 2.0 * np.dot(normal, Pn_i) * normal
            pn_r_norm = np.linalg.norm(Pn_r)
            if pn_r_norm > 1e-12:
                Pn_r = Pn_r / pn_r_norm
            Erp = -(Pn_r * Eip_scalar)
        Erp = Erp * reff_p

        Rs_used = float(np.abs(np.linalg.norm(Ers)) ** 2)
        Rp_used = float(np.abs(np.linalg.norm(Erp)) ** 2)
        R_total = min(1.0, Rs_used + Rp_used)

        # --- reflected photon: standard law of reflection ------------------
        photon_r = ph_at._copy()
        k_mag = np.dot(np.real(ph_at.k), normal)
        photon_r.k = ph_at.k - 2 * k_mag * normal
        k0 = 2.0 * np.pi / photon_r.wavelength
        kr_norm = np.linalg.norm(np.real(photon_r.k))
        if kr_norm > 0:
            photon_r.k = (np.real(photon_r.k) / kr_norm) * k0
        photon_r.E = Ers + Erp
        er_norm = np.linalg.norm(photon_r.E)
        if er_norm > 0:
            photon_r.E = photon_r.E / er_norm
        photon_r.I = ph_at.I * R_total
        photon_r.B = np.cross(np.real(photon_r.k), np.real(photon_r.E))
        b_norm = np.linalg.norm(photon_r.B)
        if b_norm > 0:
            photon_r.B = photon_r.B / b_norm

        # --- transmitted photon: straight through, exits the far plane ---
        # (front face if this photon entered from the back, back face if
        # it entered from the front — see _geometry's exit_point).
        exit_point = g["exit_point"]
        denom2 = float(np.dot(k_hat, normal0))
        t_exit = float(np.dot(exit_point - r_hit, normal0) / denom2) if abs(denom2) > 1e-12 else 0.0
        photon_t = ph_at.travel_distance(t_exit)
        photon_t.I = max(0.0, ph_at.I - photon_r.I)

        diag = {
            "R": round(R_total, 6),
            "T": round(1.0 - R_total, 6),
            "theta_i_deg": round(float(np.degrees(theta_i)), 3),
            "n_layers_used": [round(n, 6) for n in g["n_layers"]],
            "k_layers_used": [round(k, 6) for k in g["k_layers"]],
            "nk_sources": g["nk_sources"],
            "distance": round(t_hit, 6),
        }
        return photon_r, photon_t, diag

    # ------------------------------------------------------------------

    def process(self, photons: list[Photon], side: str = "front") -> tuple[list[Photon], dict]:
        out_photons: list[Photon] = []
        rays_log = []
        n_blocked = 0
        g_ref = None
        diameter = float(self.params.get("diameter", 0.0))

        for ph in photons:
            g = self._geometry(wavelength_m=float(ph.wavelength), side=side)
            g_ref = g

            ph_r, ph_t, diag = self._process_one(ph, g, diameter)
            if ph_r is None:
                n_blocked += 1
                rays_log.append({"blocked": True})
                continue

            out_photons.append(ph_r)
            out_photons.append(ph_t)
            rays_log.append(diag)

        return out_photons, {
            "n_in":        len(photons),
            "n_out":       len(out_photons),
            "n_blocked":   n_blocked,
            "n_layers":    g_ref["N"] if g_ref else int(self.params.get("n_layers", 2)),
            "total_thickness": round(g_ref["total_thickness"], 9) if g_ref else 0.0,
            "n_substrate": g_ref["n_sub"] if g_ref else float(self.params.get("n_substrate", 1.0)),
            "k_substrate": g_ref["k_sub"] if g_ref else float(self.params.get("k_substrate", 0.0)),
            "reflections": rays_log,
        }