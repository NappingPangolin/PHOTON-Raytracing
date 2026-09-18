"""
modules/substrate.py
----------------------
Substrate — a flat (optionally wedged) plate, or an N-element stack of
cemented/air-gapped flat plates, that tracks Fresnel-correct multi-order
internal reflections.

Geometry
~~~~~~~~
Like CompoundLens, a substrate is N elements (1..MAX_ELEMENTS) stacked along
a main axis, each with its own thickness and (optionally lnk-table-driven)
n/k. Unlike CompoundLens:

  - `r_x/r_y/r_z` gives the **front** surface's centre directly (not the
    stack midpoint), matching a flat mirror's convention.
  - Every one of the N+1 surfaces has its **own** normal vector
    (`normal_{i}_x/y/z`, i = 0..N), independent of the main stacking axis
    and of each other, so any surface can be wedged relative to the rest
    of the stack. Thickness is still measured along the shared axis
    (`n_x/n_y/n_z`) — the usual "centre thickness" convention for a wedge.

Multi-order internal reflections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
At `n_internal = 0` a substrate behaves like a simple partially-reflective
plate: for every incoming photon we get exactly one *reflected* beam (off
the front surface) and one *transmitted* beam (refracted through every
element in turn and out the back) — 2 output beams per input.

Raising `n_internal` to N asks for N additional internal-reflection
generations: the light that would normally be lost inside the stack is
followed as it keeps bouncing between surfaces, each bounce leaking one
more beam out of whichever surface it currently reaches (alternating
faces for a single element; more varied for a multi-element stack, since
internal glass/glass boundaries can bounce light back into the stack
instead of only towards the two outer faces). Every followed bounce adds
exactly one more output beam, so a single element produces
    N_out = 2 + n_internal
output beams per input photon, and — since every surface can also throw a
beam back into the stack instead of out of it — an N-element stack can, in
the worst case, branch a great deal faster than that. To keep this
tractable, every sub-beam is independently pruned as soon as its intensity
falls below `intensity_threshold` (a fraction of the *original* incident
intensity, default 1e-4 = 0.01%) *and* the bounce budget `n_internal` is a
hard ceiling on how many internal-reflection generations are ever explored
in the first place. For an uncoated single element the natural Fresnel
reflectance is only a few percent, so in practice the threshold alone
already prunes third- and higher-order beams long before `n_internal`
becomes relevant; the budget mainly matters for high-reflectivity coatings
or when deliberately modelling something like an etalon.

Every emitted beam is tagged with which face it left through ("front" or
"back") and its bounce order, so the two physical output populations
(reflected-side vs. transmitted-side) can be told apart even though their
counts are not fixed — see `info["reflections"]`.

Front/back sidedness
~~~~~~~~~~~~~~~~~~~~
Substrate is a *sided* module (``SIDED = True``): light can enter through
either the front face (surface 0) or the back face (surface N), and
``process(photons, side=...)`` starts the trace at the corresponding
surface, walking inward in the appropriate direction (see
``_trace_photon``'s ``start_s``/``start_dir``). The "front"/"back" tag on
every emitted beam already describes which face it *exits* through and
needs no further adjustment for the entry side — a beam entering the back
face and reflecting straight back out is correctly tagged "back" (stays on
the entry side), while one that fully transmits through to surface 0 is
tagged "front" (crosses to the opposite side), exactly mirroring the
front-input case.
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon, Material, Surface
from .base import OpticalModule, LnkMixin, ParamSpec

MAX_ELEMENTS = 8          # maximum number of stacked elements
HARD_BOUNCE_CAP = 24      # absolute safety ceiling on n_internal, regardless
                          # of what a saved scene / the UI asks for


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def _ray_plane_intersect(origin: np.ndarray, direction: np.ndarray,
                          point: np.ndarray, normal: np.ndarray):
    """
    Signed parametric distance from *origin* to the plane (point, normal)
    along unit *direction*. Returns None only if the ray is (numerically)
    parallel to the plane — the sign is deliberately *not* restricted to
    positive values, so a beam that ends up behind a surface (e.g. after a
    `free_propagate` overshoot) still gets a well-defined distance and can
    be walked back to the surface by `Photon.travel_distance`, exactly like
    `Surface.calc_distance_to_surface` already does for flat mirrors.
    """
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-12:
        return None
    return float(np.dot(point - origin, normal) / denom)


def _wedge_shift(normal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Per-unit-aperture-radius axial shift vector of a (possibly wedged) flat
    surface, confined to the plane transverse to `axis`.

    For a plane through its vertex with unit normal `normal`, the point on
    that plane at transverse offset `h * u_hat` (any unit vector u_hat ⟂
    axis) sits at local axial position (relative to the vertex):

        s(u_hat) = -h * (u_hat · normal) / (axis · normal)

    Writing this as s(u_hat) = -h * (u_hat · D) isolates a single vector

        D = (normal - (normal · axis) * axis) / (axis · normal)

    i.e. the surface's transverse tilt, rescaled so its axis-component is
    implicitly 1. D alone is enough to find the worst-case axial excursion
    of the surface anywhere on the aperture disk, in any direction, without
    stepping through azimuths one at a time.
    """
    denom = float(np.dot(axis, normal))
    if abs(denom) < 1e-9:
        return np.zeros(3)   # surface edge-on to the axis; degenerate, treat as untilted
    perp = normal - denom * axis
    return perp / denom


def _t_min_wedge(h: float, D_front: np.ndarray, D_back: np.ndarray) -> float:
    """
    Minimum centre thickness so that two independently-wedged flat surfaces
    — sharing a clear-aperture radius `h` — never cut through each other.

    Every point on the front surface at radius <= h sits at local axial
    position s_front(u_hat) = -h*(u_hat . D_front) relative to *its* vertex,
    and likewise s_back(u_hat) = -h*(u_hat . D_back) relative to the back
    surface's vertex, which sits a distance t further along the axis.
    Requiring the front surface to stay at-or-before the back surface for
    every direction u_hat on the aperture disk:

        s_front(u_hat) <= t + s_back(u_hat)      for every unit vector u_hat
        t >= max_u_hat [ s_front(u_hat) - s_back(u_hat) ]
           = max_u_hat [ u_hat . (D_back - D_front) ]
           = |D_back - D_front|

    (the maximum of u_hat·V over all unit vectors u_hat is just |V|, attained
    when u_hat is aligned with V — i.e. the worst case blends both surfaces'
    tilt directions, not just one of them in isolation).
    """
    return float(h * np.linalg.norm(D_back - D_front))


class Substrate(LnkMixin, OpticalModule):
    MODULE_TYPE = "substrate"
    LABEL = "Substrate"
    COLOR = "#efece2"   # white-ish
    ICON = "▭"
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

            # ── main stacking axis (thickness direction) ──────────────────
            ParamSpec("n_x", "Axis X", "float", 1.0, "",
                      min=-1.0, max=1.0,
                      tooltip="Main stacking axis (thickness direction). "
                              "Individual surfaces may still be wedged "
                              "relative to this via their own normals below."),
            ParamSpec("n_y", "Axis Y", "float", 0.0, "", min=-1.0, max=1.0),
            ParamSpec("n_z", "Axis Z", "float", 0.0, "", min=-1.0, max=1.0),

            # ── stack size / aperture ──────────────────────────────────────
            ParamSpec("n_elements", "Number of Elements", "int", 1, "",
                      min=1, max=MAX_ELEMENTS,
                      tooltip="Number of cemented/stacked elements (1 = a "
                              "simple plate with just a front and back surface)"),
            ParamSpec("diameter", "Clear Aperture ⌀", "float", 0.05, "m",
                      min=0.0, max=100.0,
                      tooltip="Shared clear-aperture diameter for every "
                              "surface. 0 = unbounded (no clipping)."),

            # ── internal reflections ──────────────────────────────────────
            ParamSpec("n_internal", "Internal Reflections N", "int", 0, "",
                      min=0, max=12,
                      tooltip="How many additional internal-reflection "
                              "generations to trace. 0: only the direct "
                              "front reflection + the straight-through "
                              "transmission (2 beams/input). Each unit "
                              "beyond that follows one more bounce, adding "
                              "one more output beam (alternating exit face "
                              "for a single element)."),
            ParamSpec("intensity_threshold", "Intensity Cutoff", "float", 1e-4, "",
                      min=1e-8, max=1.0,
                      tooltip="Sub-beams whose intensity falls below this "
                              "fraction of the incident intensity are "
                              "dropped instead of traced further (e.g. "
                              "1e-4 = 0.01%). Keeps multi-element stacks "
                              "with many internal surfaces tractable."),
            ParamSpec("beam_shifts", "GH/IF Beam Shifts", "bool", False,
                      tooltip="Enable Goos-Hänchen and Imbert-Fedorov shifts "
                              "at every surface"),
        ]

        # Per-element thickness / n / k / lnk table (elements 1..MAX)
        for i in range(1, MAX_ELEMENTS + 1):
            specs += [
                ParamSpec(f"t_{i}", f"Thickness t{i}", "float", 0.005, "m",
                          min=1e-6, max=10.0,
                          tooltip=f"Centre thickness of element {i} "
                                  f"(measured along the main axis)"),
                ParamSpec(f"n_{i}", f"n{i} (real)", "float", 1.5168, "",
                          min=1.0, max=5.0,
                          tooltip=f"Real refractive index of element {i} "
                                  f"(manual mode)"),
                ParamSpec(f"k_{i}", f"k{i} (extinction)", "float", 0.0, "",
                          min=0.0, max=10.0,
                          tooltip=f"Extinction coefficient of element {i} "
                                  f"(manual mode)"),
            ]
            specs += [
                ParamSpec(f"nk_mode_{i}", f"n/k Source {i}", "select", "manual",
                          options=["manual", "lnk_table"]),
                ParamSpec(f"lnk_data_{i}", f"n/k Table {i} (lnk)", "lnk_table", []),
            ]

        # Per-surface normal (surfaces 0..MAX, i.e. one more than elements)
        for i in range(0, MAX_ELEMENTS + 1):
            specs += [
                ParamSpec(f"normal_{i}_x", f"N{i} X", "float", 1.0, "",
                          min=-1.0, max=1.0,
                          tooltip=f"Surface {i} normal — defaults to the main "
                                  f"axis; change to wedge this surface"),
                ParamSpec(f"normal_{i}_y", f"N{i} Y", "float", 0.0, "",
                          min=-1.0, max=1.0),
                ParamSpec(f"normal_{i}_z", f"N{i} Z", "float", 0.0, "",
                          min=-1.0, max=1.0),
            ]

        return specs

    # ------------------------------------------------------------------
    # Geometry / material resolution (mirrors CompoundLens._geometry)
    # ------------------------------------------------------------------

    def _geometry(self, wavelength_m: float | None = None) -> dict:
        p = self.params
        N = max(1, min(MAX_ELEMENTS, int(p.get("n_elements", 1))))

        r_ctr = np.array([float(p["r_x"]), float(p["r_y"]), float(p["r_z"])])
        axis = _normalise(np.array([float(p["n_x"]), float(p["n_y"]), float(p["n_z"])]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])

        diameter = float(p.get("diameter", 0.0))
        h = diameter / 2.0

        t_raw = [float(p[f"t_{i}"]) for i in range(1, N + 1)]

        # Per-surface normals (N+1 of them)
        normals = []
        for i in range(N + 1):
            nv = np.array([
                float(p.get(f"normal_{i}_x", 1.0)),
                float(p.get(f"normal_{i}_y", 0.0)),
                float(p.get(f"normal_{i}_z", 0.0)),
            ])
            normals.append(_normalise(nv) if np.linalg.norm(nv) > 1e-9 else axis)

        # Minimum centre thickness per element so that its front/back
        # surfaces — independently wedged, sharing the clear aperture —
        # never cut through each other (see _wedge_shift / _t_min_wedge).
        # diameter == 0 ("unbounded") means no clipping is enforced anywhere
        # else in this module either, so skip the check the same way.
        if h > 0:
            D = [_wedge_shift(n, axis) for n in normals]
            t_mins = [_t_min_wedge(h, D[i], D[i + 1]) for i in range(N)]
        else:
            t_mins = [0.0] * N
        thicknesses = [max(t_raw[i], t_mins[i]) for i in range(N)]

        # Vertices along the main axis, front surface = r_ctr exactly
        vertices = [r_ctr.copy()]
        for t in thicknesses:
            vertices.append(vertices[-1] + t * axis)

        # Resolve n/k per element — lnk table takes over when active
        lam = wavelength_m if wavelength_m is not None else 800e-9
        n_glass, k_glass, nk_sources = [], [], []
        for i in range(1, N + 1):
            mode = p.get(f"nk_mode_{i}", "manual")
            rows = p.get(f"lnk_data_{i}", [])
            source = "manual"
            if mode == "lnk_table" and rows:
                try:
                    from .lnk_table import LnkTable
                    tbl = LnkTable.from_rows(rows)
                    n_i, k_i = tbl.get_nk(lam)
                    source = "lnk_table"
                except Exception:
                    n_i, k_i = float(p[f"n_{i}"]), float(p[f"k_{i}"])
            else:
                n_i, k_i = float(p[f"n_{i}"]), float(p[f"k_{i}"])
            n_glass.append(n_i)
            k_glass.append(k_i)
            nk_sources.append(source)

        return dict(
            N=N, axis=axis, h=h,
            vertices=vertices, normals=normals,
            thicknesses=thicknesses, t_raw=t_raw, t_mins=t_mins,
            n_glass=n_glass, k_glass=k_glass, nk_sources=nk_sources,
        )

    # ------------------------------------------------------------------
    # Per-photon multi-order trace
    # ------------------------------------------------------------------

    def _trace_photon(self, ph: Photon, g: dict, lam: float,
                       bounce_cap: int, threshold: float,
                       beam_shifts: bool,
                       start_s: int = 0, start_dir: int = 1) -> list[dict]:
        """
        Returns a list of {"photon", "side", "order", "waypoints"} dicts —
        one entry per surviving output beam for this single input photon.
        "side" is "front" (the beam ultimately exits surface 0, the
        geometric front face) or "back" (exits surface N, the geometric
        back face) — this is which *face* the beam leaves through, and is
        correct regardless of which face the input photon entered through.

        ``start_s`` / ``start_dir`` select where the incoming photon first
        hits the stack: (0, +1) for a front-input photon (walks forward
        through increasing surface indices, as for the plate's usual
        front-to-back illumination), or (N, -1) for a back-input photon
        (walks backward from the last surface) — see Substrate.process().
        """
        N, h = g["N"], g["h"]
        vertices, normals = g["vertices"], g["normals"]
        # media[m]: (n, k) for m = 0 (front vacuum) .. N+1 (back vacuum)
        media = [(1.0, 0.0)] + list(zip(g["n_glass"], g["k_glass"])) + [(1.0, 0.0)]

        I0 = ph.I
        emitted: list[dict] = []

        def emit(beam: Photon, side: str, order: int, waypoints: list):
            # A beam can reach here two ways: (a) genuinely above threshold,
            # in which case this is just bookkeeping, or (b) a *degenerate*
            # Fresnel branch — most importantly Total Internal Reflection,
            # where Surface.reflect() correctly hands back a transmitted
            # photon with k = (0, 0, 0) (there is no real propagating
            # transmitted wave under TIR) but — by this project's "leave I
            # untouched" convention — an intensity that trace() computes as
            # I_i - I_r itself. That difference should be exactly 0 under
            # TIR, but complex-Fresnel round-off typically leaves a ~1e-19
            # residual, which is *not* caught by the threshold check at the
            # top of trace() because a beam exiting on the very bounce it
            # was produced skips straight to emit() without ever re-entering
            # trace(). A zero-k, "nonzero" intensity beam that slips through
            # here works fine right up until the next time *anything* calls
            # travel_distance() on it (a downstream free_propagate, a
            # cascaded second substrate, even just building 3-D preview ray
            # segments), at which point k_hat = k / |k| divides by zero and
            # produces NaN, silently breaking whatever consumes it. This
            # tends to surface only at lower intensity_threshold settings,
            # simply because a lower threshold lets tracing continue deep
            # enough into a wedge's oblique internal bounces to actually
            # reach a TIR interface in the first place — shallower traces
            # get pruned by trace()'s threshold check before ever getting
            # there, which is why raising the cutoff "fixes" it by accident.
            if float(np.linalg.norm(np.real(beam.k))) < 1e-20:
                return   # no real propagating direction (TIR/evanescent) - drop it
            if beam.I < threshold * I0:
                return
            emitted.append({"photon": beam, "side": side,
                             "order": order, "waypoints": waypoints})

        def trace(beam: Photon, s: int, direction: int, budget: int,
                  order: int, waypoints: list):
            if beam.I < threshold * I0:
                return
            if direction == 1:
                m_in, m_out = s, s + 1
            else:
                m_in, m_out = s + 1, s

            vertex, normal = vertices[s], normals[s]
            r0 = np.real(beam.r).astype(float)
            k_hat = _normalise(np.real(beam.k).astype(float))

            t = _ray_plane_intersect(r0, k_hat, vertex, normal)
            if t is None:
                return  # beam runs parallel to this (possibly wedged) surface

            r_hit = r0 + t * k_hat
            if h > 0:
                transverse = r_hit - vertex
                transverse = transverse - np.dot(transverse, normal) * normal
                if float(np.linalg.norm(transverse)) > h:
                    return  # clipped by the clear aperture

            n_in, k_in = media[m_in]
            n_out, k_out = media[m_out]

            # Walk the beam to the surface through whichever medium it is
            # currently in — n/k here matter for the phase advance and
            # (for k>0) the Beer-Lambert attenuation over that path length.
            # A negative t (surface behind the beam) is handled the same
            # way: travel_distance walks it back in time to the surface.
            beam_at = beam.travel_distance(t, n=n_in, k=k_in)
            beam_at.r = r_hit.astype(complex)
            wp = waypoints + [r_hit.tolist()]

            surf = Surface(n_out, k_out, 1.0, 1.0, lam, normal,
                            r=r_hit, beam_shifts=beam_shifts)
            incoming = Material.custom(lam, n_in, k_in)
            ph_r, ph_t = surf.reflect(beam_at, incoming)

            # Surface.reflect() sets the Fresnel-correct reflected intensity
            # but (by this project's convention, see flat_mirror/compound_lens)
            # leaves the transmitted photon's intensity untouched — enforce
            # R + T = 1 (relative to the interface itself; bulk absorption
            # between surfaces is handled separately via travel_distance's k).
            ph_r.I = float(min(beam_at.I, ph_r.I))
            if float(np.linalg.norm(np.real(ph_t.k))) < 1e-20:
                # Total internal reflection (or evanescent): no real
                # transmitted wave, so its intensity is exactly zero rather
                # than whatever small residual beam_at.I - ph_r.I leaves
                # behind from complex-Fresnel round-off.
                ph_t.I = 0.0
            else:
                ph_t.I = float(max(0.0, beam_at.I - ph_r.I))

            new_order = order + 1

            # --- reflected component: stays in m_in, reverses direction ---
            if m_in == 0 or m_in == N + 1:
                emit(ph_r, "front" if m_in == 0 else "back", new_order, wp)
            elif budget > 0 and ph_r.I >= threshold * I0:
                other_s = s - 1 if direction == 1 else s + 1
                trace(ph_r, other_s, -direction, budget - 1, new_order, wp)
            # else: internal bounce budget exhausted, or already below
            # threshold - dropped silently either way

            # --- transmitted component: enters m_out, keeps direction ---
            if m_out == 0 or m_out == N + 1:
                emit(ph_t, "front" if m_out == 0 else "back", new_order, wp)
            elif ph_t.I >= threshold * I0:
                next_s = s + 1 if direction == 1 else s - 1
                trace(ph_t, next_s, direction, budget, new_order, wp)
            # else: TIR (zero-k) or below threshold - dropped silently

        r_start = np.real(ph.r).astype(float).tolist()
        trace(ph, start_s, start_dir, bounce_cap, 0, [r_start])
        return emitted

    # ------------------------------------------------------------------

    def process(self, photons: list[Photon], side: str = "front") -> tuple[list[Photon], dict]:
        p = self.params
        n_internal = max(0, int(p.get("n_internal", 0)))
        bounce_cap = min(n_internal, HARD_BOUNCE_CAP)
        threshold = max(0.0, float(p.get("intensity_threshold", 1e-4)))
        beam_shifts = bool(p.get("beam_shifts", False))

        out_photons: list[Photon] = []
        rays_log = []
        n_blocked = 0
        g_ref = None

        for ph in photons:
            lam = float(ph.wavelength)
            g = self._geometry(wavelength_m=lam)
            g_ref = g

            # A front-input photon physically reaches surface 0 first and
            # walks forward through increasing surface indices; a back-input
            # photon reaches surface N first and walks backward. The
            # emitted beams' "front"/"back" side tags (which face they
            # ultimately leave through) come out correct either way -
            # see _trace_photon's direction-agnostic trace() closure.
            if side == "back":
                emitted = self._trace_photon(ph, g, lam, bounce_cap, threshold,
                                              beam_shifts, start_s=g["N"], start_dir=-1)
            else:
                emitted = self._trace_photon(ph, g, lam, bounce_cap, threshold,
                                              beam_shifts, start_s=0, start_dir=1)

            if not emitted:
                n_blocked += 1
                rays_log.append({"blocked": True})
                continue

            fronts = sorted((e for e in emitted if e["side"] == "front"), key=lambda e: e["order"])
            backs  = sorted((e for e in emitted if e["side"] == "back"),  key=lambda e: e["order"])

            for e in fronts + backs:
                out_photons.append(e["photon"])

            rays_log.append({
                "n_reflected":   len(fronts),
                "n_transmitted": len(backs),
                "reflected": [
                    {"order": e["order"], "I": round(float(e["photon"].I), 6),
                     "waypoints": e["waypoints"]}
                    for e in fronts
                ],
                "transmitted": [
                    {"order": e["order"], "I": round(float(e["photon"].I), 6),
                     "waypoints": e["waypoints"]}
                    for e in backs
                ],
            })

        return out_photons, {
            "n_in":        len(photons),
            "n_out":       len(out_photons),
            "n_blocked":   n_blocked,
            "n_elements":  g_ref["N"] if g_ref else int(p.get("n_elements", 1)),
            "n_internal_used":     bounce_cap,
            "intensity_threshold": threshold,
            "n_glass_used": [round(n, 6) for n in g_ref["n_glass"]] if g_ref else [],
            "k_glass_used": [round(k, 6) for k in g_ref["k_glass"]] if g_ref else [],
            "nk_sources":   g_ref["nk_sources"] if g_ref else [],
            "thicknesses_used": [round(t, 9) for t in g_ref["thicknesses"]] if g_ref else [],
            "t_mins":            [round(t, 9) for t in g_ref["t_mins"]] if g_ref else [],
            "rays": rays_log,
        }