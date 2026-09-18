"""
modules/compound_lens.py
------------------------
CompoundLens  -  a general N-element cemented lens stack.

Each element is a single piece of glass bounded by two spherical (or flat)
surfaces.  Adjacent elements share a surface (cemented, no air gap, no glue
modelled).  The stack contains N elements, N+1 surfaces, and is described by:

    n_elements  : integer 1 … 8
    R_0 … R_N   : N+1 radii of curvature  (one per surface, m)
    t_1 … t_N   : N centre thicknesses     (one per element, m)
    n_1 … n_N   : N real refractive indices
    k_1 … k_N   : N extinction coefficients

A 1-element stack is equivalent to a thick lens; a 2-element stack is the
classical cemented doublet (previously the only mode).

Sign convention (uniform across all surfaces)
---------------------------------------------
    C_i = v_i + R_i · n̂

    R > 0  →  centre of curvature on the *outgoing* (exit) side
    R < 0  →  centre of curvature on the *incoming* side
    R = 0  →  flat surface

Minimum thickness enforcement
------------------------------
Each element is checked independently.  The check compares the front radius
R_{i-1} and back radius R_i (both in the uniform convention) and enforces a
positive centre thickness.

ABCD matrix
-----------
The system matrix is built by chaining refraction and propagation matrices:

    M = R(n_{N}, 1, R_N) · T(t_N, n_N) · R(n_{N-1}, n_N, R_{N-1}) · …
        · T(t_1, n_1) · R(1, n_1, R_0)

where R(n_in, n_out, R) is the refraction matrix and T(t, n) the propagation
matrix, both in the [y, n·θ] convention.

    EFL = −1 / C         (C = M[1,0])
    FFL =  D / C         (from first vertex)
    BFL = −A / C         (from last vertex)
    H1  = (D−1) / C      (from first vertex to front principal plane)
    H2  = (1−A) / C      (from last vertex  to rear  principal plane)

Output / waypoints
------------------
Waypoints list per ray: [r_before, r_hit_0, r_hit_1, …, r_hit_N]
  r_before  : photon position before surface 0
  r_hit_i   : hit position at surface i
scene.py draws r_hit_{i} → r_hit_{i+1} as intra_module segments.

Front/back sidedness
---------------------
CompoundLens is a *sided* module (``SIDED = True``). It only ever models
transmission (no surface reflections), so a photon entering either face
always exits the opposite one - there's no reflected branch to route.
Illuminating the back face is handled by tracing through the equivalent
*flipped* prescription: element/thickness/glass order reversed and every
radius negated+reordered (``R_new_i = -R_old_{N-i}``), which is the
standard "flip the lens" trick for reverse ray tracing - physically the
same stack of glass, just walked from the other end, reusing the exact
same forward-marching surface loop. See ``_geometry(..., flip=True)``.
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon, Material, Surface
from .base import OpticalModule, LnkMixin, ParamSpec

MAX_ELEMENTS = 8   # maximum number of elements in a stack

# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def _sagitta(R: float, h: float) -> float:
    if R == 0.0:
        return 0.0
    R2v, h2 = R * R, h * h
    if h2 >= R2v:
        return abs(R)
    return abs(R) - np.sqrt(R2v - h2)


def _ray_sphere_intersect(origin, direction, centre, radius, prefer_entry):
    oc = origin - centre
    b  = 2.0 * np.dot(direction, oc)
    c  = np.dot(oc, oc) - radius ** 2
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    sd = np.sqrt(disc)
    t1, t2 = (-b - sd) / 2.0, (-b + sd) / 2.0
    valid = [t for t in (t1, t2) if t > 1e-9]
    if not valid:
        return None
    return min(valid) if prefer_entry else max(valid)


def _ray_plane_intersect(origin, direction, point, normal):
    denom = np.dot(direction, normal)
    if abs(denom) < 1e-12:
        return None
    t = np.dot(point - origin, normal) / denom
    return t if t > 1e-9 else None


def _t_min(R_front: float, R_back: float, sag_f: float, sag_b: float) -> float:
    """
    Minimum centre thickness for one element with uniform-sign radii.

    Convert to ThickLens convention (R_back_TL = -R_back) so the same
    case logic applies.
    """
    Ra, Rb = R_front, -R_back   # TL-equivalent radii
    if Ra > 0 and Rb < 0:
        return 1e-10 if abs(Ra) >= abs(Rb) else abs(sag_f - sag_b)
    elif Rb > 0 and Ra < 0:
        return 1e-10 if abs(Ra) <= abs(Rb) else abs(sag_f - sag_b)
    elif Ra < 0 and Rb < 0:
        return 1e-10
    else:
        return sag_f + sag_b


def _stack_abcd(radii: list[float],
                thicknesses: list[float],
                n_media: list[float]) -> tuple[float, float, float, float]:
    """
    Build the ABCD system matrix for a stack of N cemented elements.

    Parameters
    ----------
    radii       : [R_0, R_1, …, R_N]    length N+1
    thicknesses : [t_1, …, t_N]         length N
    n_media     : [1.0, n_1, …, n_N, 1.0]  length N+2
                  n_media[0] = outside (air), n_media[i] = glass_i,
                  n_media[-1] = outside (air)

    Returns (A, B, C, D) of the composite 2×2 matrix.
    """
    def refract_m(n_in, n_out, R):
        phi = 0.0 if R == 0.0 else (n_out - n_in) / R
        return np.array([[1.0, 0.0], [-phi, 1.0]])

    def propagate_m(t, n):
        return np.array([[1.0, t / n], [0.0, 1.0]])

    N = len(thicknesses)
    # Build right-to-left: last refraction first in the chain
    M = refract_m(n_media[N], n_media[N + 1], radii[N])   # exit surface
    for i in range(N - 1, -1, -1):
        M = M @ propagate_m(thicknesses[i], n_media[i + 1])
        M = M @ refract_m(n_media[i], n_media[i + 1], radii[i])

    return float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1])


def _abcd_to_focal(A, B, C, D):
    """Convert ABCD to {efl, ffl, bfl, h1, h2} or all-inf if afocal."""
    if abs(C) < 1e-30:
        inf = float('inf')
        return dict(efl=inf, ffl=inf, bfl=inf, h1=inf, h2=inf)
    return dict(
        efl = -1.0 / C,
        ffl =  D   / C,
        bfl = -A   / C,
        h1  = (D - 1.0) / C,
        h2  = (1.0 - A) / C,
    )


# ══════════════════════════════════════════════════════════════════════════════

class CompoundLens(LnkMixin, OpticalModule):
    """
    General N-element cemented lens stack (N = 1 … 8).

    A 1-element stack is a standard thick lens; 2 elements form a cemented
    doublet; higher counts model more complex designs.

    Parameters are a fixed set for up to MAX_ELEMENTS elements; only the
    first n_elements entries are used.  The UI hides irrelevant rows.
    """

    MODULE_TYPE = "compound_lens"
    LABEL       = "Compound Lens"
    COLOR       = "#a78bfa"
    ICON        = "⌖⌖"
    SIDED       = True

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        specs = [
            # ── Position ────────────────────────────────────────────────
            ParamSpec("r_x", "R_x", "float", 0.0, "m",
                      min=-100.0, max=100.0, tooltip="Lens stack centre X"),
            ParamSpec("r_y", "R_y", "float", 0.0, "m",
                      min=-100.0, max=100.0, tooltip="Lens stack centre Y"),
            ParamSpec("r_z", "R_z", "float", 0.0, "m",
                      min=-100.0, max=100.0, tooltip="Lens stack centre Z"),
            # ── Optical axis ────────────────────────────────────────────
            ParamSpec("n_x", "N_x", "float",  1.0, "",
                      min=-1.0, max=1.0, tooltip="Optical axis X"),
            ParamSpec("n_y", "N_y", "float",  0.0, "",
                      min=-1.0, max=1.0, tooltip="Optical axis Y"),
            ParamSpec("n_z", "N_z", "float",  0.0, "",
                      min=-1.0, max=1.0, tooltip="Optical axis Z"),
            # ── Number of elements ───────────────────────────────────────
            ParamSpec("n_elements", "Number of elements", "int", 2, "",
                      min=1, max=MAX_ELEMENTS,
                      tooltip=(
                          f"Number of cemented glass elements (1-{MAX_ELEMENTS}). "
                          "1 = single thick lens; 2 = doublet; etc."
                      )),
            # ── Common aperture ──────────────────────────────────────────
            ParamSpec("diameter", "Clear aperture diameter", "float", 0.05, "m",
                      min=1e-6, max=1.0, tooltip="Clear aperture shared by all elements"),
        ]

        # Surface radii R_0 … R_{MAX_ELEMENTS}
        for i in range(MAX_ELEMENTS + 1):
            if i == 0:
                label   = "R₀ (front)"
                default = 0.1
                tip     = ("Front surface of element 1. "
                           "R > 0: convex toward incoming light.")
            elif i == MAX_ELEMENTS:
                label   = f"R{i} (rear)"
                default = -0.1
                tip     = f"Rear surface of element {MAX_ELEMENTS}."
            else:
                label   = f"R{i} (interface {i})"
                default = 0.08 if i % 2 == 0 else -0.08
                tip     = (f"Shared interface between element {i} and element {i+1}. "
                           "R > 0: centre on outgoing side.")
            specs.append(ParamSpec(
                f"R_{i}", label, "float", default, "m",
                min=-10.0, max=10.0, tooltip=tip,
            ))

        # Element thicknesses t_1 … t_{MAX_ELEMENTS}
        for i in range(1, MAX_ELEMENTS + 1):
            specs.append(ParamSpec(
                f"t_{i}", f"t{i} (thickness)", "float", 0.010, "m",
                min=1e-6, max=1.0,
                tooltip=f"Centre thickness of element {i}. Clamped to geometric minimum.",
            ))

        # Refractive indices n_1 … n_{MAX_ELEMENTS}
        for i in range(1, MAX_ELEMENTS + 1):
            n_default = 1.5168 if i % 2 == 1 else 1.6727
            specs.append(ParamSpec(
                f"n_{i}", f"n{i} (glass {i})", "float", n_default, "",
                min=1.0, max=5.0,
                tooltip=f"Refractive index of element {i}.",
            ))

        # Extinction coefficients k_1 … k_{MAX_ELEMENTS}
        for i in range(1, MAX_ELEMENTS + 1):
            specs.append(ParamSpec(
                f"k_{i}", f"k{i} (extinction)", "float", 0.0, "",
                min=0.0, max=10.0,
                tooltip=f"Extinction coefficient of element {i} (0 = lossless).",
            ))

        # Per-element n/k source selector and lnk table
        for i in range(1, MAX_ELEMENTS + 1):
            specs.append(ParamSpec(
                f"nk_mode_{i}", f"n/k Source {i}", "select", "manual",
                options=["manual", "lnk_table"],
                tooltip=(
                    f"Element {i}: manual uses n{i}/k{i} directly; "
                    f"lnk_table interpolates from the uploaded wavelength table."
                ),
            ))
            specs.append(ParamSpec(
                f"lnk_data_{i}", f"lnk table {i}", "lnk_table", [],
                tooltip=(
                    f"Element {i}: upload a two-block lnk file (wl[µm] | n, then wl[µm] | k). "
                    "The table is embedded in the scene on save/load."
                ),
            ))

        specs.append(ParamSpec(
            "display_wavelength", "Display λ", "float", 800e-9, "m",
            min=200e-9, max=10_000e-9,
            tooltip="Wavelength for the focal-length display only.",
        ))

        return specs

    # ── internal geometry ─────────────────────────────────────────────────────

    def _geometry(self, wavelength_m: float | None = None, flip: bool = False) -> dict:
        """
        Resolve the full stack geometry.

        Parameters
        ----------
        wavelength_m : photon wavelength in metres, used for lnk table
                       interpolation.  Falls back to display_wavelength when
                       None (e.g. for the ABCD info block).
        flip : if True, build the geometrically flipped lens instead - the
               element order, thicknesses, and glass indices reversed, and
               every radius negated and reordered
               (``R_new_i = -R_old[N-i]``) so it can be walked by the exact
               same forward surface-marching loop in process() to correctly
               trace a photon entering from the back face. See the module
               docstring's "Front/back sidedness" section.

        Returns
        -------
        n_hat     : unit optical-axis vector
        N         : number of elements
        radii     : list of N+1 radii  [R_0 … R_N]
        thicknesses : list of N clamped centre thicknesses
        n_glass   : list of N refractive indices
        k_glass   : list of N extinction coefficients
        h         : semi-aperture
        vertices  : list of N+1 vertex positions (3-vectors)
        centres   : list of N+1 CoC positions (3-vector or None for flat)
        sags      : list of N+1 sagitta values
        t_mins    : list of N minimum thicknesses
        r_centre  : compound-lens centre position
        """
        p     = self.params
        r_ctr = np.array([float(p["r_x"]), float(p["r_y"]), float(p["r_z"])])
        n_hat = _normalise(np.array([float(p["n_x"]), float(p["n_y"]), float(p["n_z"])]))

        N  = max(1, min(MAX_ELEMENTS, int(float(p.get("n_elements", 2)))))
        d  = float(p["diameter"])
        h  = d / 2.0

        radii   = [float(p[f"R_{i}"]) for i in range(N + 1)]
        t_raw   = [float(p[f"t_{i}"]) for i in range(1, N + 1)]

        # Resolve n/k per element — use lnk table when active
        lam = wavelength_m if wavelength_m is not None else float(p.get("display_wavelength", 800e-9))
        n_glass = []
        k_glass = []
        nk_sources = []
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
                    n_i = float(p[f"n_{i}"])
                    k_i = float(p[f"k_{i}"])
            else:
                n_i = float(p[f"n_{i}"])
                k_i = float(p[f"k_{i}"])
            n_glass.append(n_i)
            k_glass.append(k_i)
            nk_sources.append(source)

        if flip:
            radii      = [-r for r in reversed(radii)]
            t_raw      = list(reversed(t_raw))
            n_glass    = list(reversed(n_glass))
            k_glass    = list(reversed(k_glass))
            nk_sources = list(reversed(nk_sources))
            # Walk from the ORIGINAL rear position backward (i.e. along
            # -n_hat) so the flipped stack's surfaces land on the same
            # physical span as the original lens, just re-numbered from the
            # other end - not a fresh re-centred stack starting at the old
            # front. Every direction-dependent quantity below (vertices,
            # curvature centres, and therefore the surface normals
            # _hit_surface derives from them) uses this flipped axis
            # instead of n_hat for the rest of this geometry build.
            axis = -n_hat
        else:
            axis = n_hat

        sags   = [_sagitta(R, h) for R in radii]
        t_mins = [_t_min(radii[i], radii[i + 1], sags[i], sags[i + 1])
                  for i in range(N)]
        thicknesses = [max(t_raw[i], t_mins[i]) for i in range(N)]

        t_total = sum(thicknesses)

        # Vertex positions: v[0] is the front vertex (the face this geometry
        # is illuminated through), v[N] is the rear vertex. The compound-lens
        # *centre* is the midpoint of the total axial span, so v[0] sits at
        # the physically-correct end of the (possibly flipped) span either way.
        v0 = r_ctr - (t_total / 2.0) * axis
        vertices = [v0]
        for t in thicknesses:
            vertices.append(vertices[-1] + t * axis)

        n_hat = axis   # everything past this point (centres, process()'s
                        # surface loop) should use the flipped axis

        # Centres of curvature: C_i = v_i + R_i · n̂
        centres = [(vertices[i] + radii[i] * n_hat) if radii[i] != 0.0 else None
                   for i in range(N + 1)]

        return dict(
            r_centre=r_ctr, n_hat=n_hat,
            N=N, radii=radii, thicknesses=thicknesses,
            n_glass=n_glass, k_glass=k_glass, nk_sources=nk_sources,
            h=h, vertices=vertices, centres=centres,
            sags=sags, t_mins=t_mins,
        )

    # ── surface hit helper ────────────────────────────────────────────────────

    @staticmethod
    def _hit_surface(ph: Photon,
                     vertex: np.ndarray,
                     n_hat: np.ndarray,
                     R: float,
                     C,
                     h: float,
                     prefer_entry: bool = True):
        r0    = np.real(ph.r).copy()
        k_hat = _normalise(np.real(ph.k))

        if R == 0.0 or C is None:
            t = _ray_plane_intersect(r0, k_hat, vertex, n_hat)
            if t is None:
                return None, None
            r_hit     = r0 + t * k_hat
            transverse = r_hit - vertex
            transverse -= np.dot(transverse, n_hat) * n_hat
            if np.linalg.norm(transverse) > h:
                return None, None
            normal = -n_hat if np.dot(k_hat, n_hat) > 0 else n_hat
        else:
            t = _ray_sphere_intersect(r0, k_hat, C, abs(R), prefer_entry)
            if t is None:
                return None, None
            r_hit     = r0 + t * k_hat
            d_to_axis = r_hit - vertex
            d_to_axis -= np.dot(d_to_axis, n_hat) * n_hat
            if np.linalg.norm(d_to_axis) > h:
                return None, None
            raw = _normalise(r_hit - C)
            normal = -raw if np.dot(k_hat, raw) > 0 else raw

        return r_hit, normal

    # ── process ───────────────────────────────────────────────────────────────

    def process(self, photons: list[Photon], side: str = "front") -> tuple[list[Photon], dict]:
        """
        Trace each photon through all N+1 surfaces of the lens stack.

        Geometry (n/k per element) is resolved per photon so that lnk tables
        are interpolated at each photon's own wavelength.

        `side` selects which face the given photons are arriving at:
        "front" walks the stack as configured; "back" walks the
        geometrically flipped stack (see _geometry(flip=True) / the module
        docstring's "Front/back sidedness" section) so a back-illuminated
        lens refracts correctly rather than just running the front
        prescription with the ray direction reversed.

        Waypoints per ray: [r_before, r_hit_0, r_hit_1, …, r_hit_N]
        """
        out_photons: list[Photon] = []
        log: list[dict] = []
        flip = (side == "back")

        # Keep a reference geometry for the ABCD block at the end.
        # Always use display_wavelength here (not a traced photon's own
        # wavelength) so the reported EFL/FFL/BFL match what the "Display λ"
        # label promises and stay consistent with the live editor preview,
        # which also evaluates the lnk table (when active) at this same
        # wavelength. Individual photons are still traced at their own
        # wavelength further down, lnk table included.
        ref_lam = float(self.params.get("display_wavelength", 800e-9))
        g_ref = self._geometry(wavelength_m=ref_lam, flip=flip)

        for ph in photons:
            lam = float(ph.wavelength)
            g   = self._geometry(wavelength_m=lam, flip=flip)

            N           = g["N"]
            n_hat       = g["n_hat"]
            vertices    = g["vertices"]
            centres     = g["centres"]
            radii       = g["radii"]
            thicknesses = g["thicknesses"]
            n_glass     = g["n_glass"]
            k_glass     = g["k_glass"]
            h           = g["h"]
            n_media     = [1.0] + n_glass + [1.0]

            r_hits  = []
            blocked = None
            ph_cur  = ph
            r_before = np.real(ph.r).copy()

            for i in range(N + 1):
                r_hit, n_hit = self._hit_surface(
                    ph_cur, vertices[i], n_hat,
                    radii[i], centres[i], h,
                )
                if r_hit is None:
                    blocked = f"miss_surface_{i}"
                    break

                if i == 0:
                    r_before = np.real(ph_cur.r).copy()

                # Medium the photon is travelling through *right now*, i.e.
                # over the segment that ends at this hit (vacuum before the
                # first surface, otherwise the glass of the element it just
                # entered at the previous surface).
                n_in  = n_media[i]
                n_out = n_media[i + 1]
                k_out = k_glass[i] if i < N else 0.0
                k_in  = 0.0 if i == 0 else k_glass[i - 1]

                # Propagate through that medium with its own n/k so time,
                # phase, and (via k) Lambert-Beer amplitude attenuation are
                # all correct — previously this always defaulted to n=1
                # (vacuum), so a photon's travel time through a lens with
                # n_glass != 1 was silently computed as if it had crossed
                # the same distance in air instead.
                dist   = float(np.linalg.norm(r_hit - np.real(ph_cur.r)))
                ph_cur = ph_cur.travel_distance(dist, n=n_in, k=k_in)
                ph_cur.r = r_hit.astype(complex)

                surf = Surface(
                    n_out, k_out, 1.0, 1.0, lam,
                    normal=n_hit, r=r_hit, beam_shifts=False,
                )
                incoming_mat = Material.vacuum(lam) if i == 0 \
                    else Material.custom(lam, n_in, k_in)
                _refl, ph_t = surf.reflect(ph_cur, incoming_mat)

                if ph_t is None:
                    blocked = f"TIR_surface_{i}"
                    break

                r_hits.append(r_hit)

                ph_cur = ph_t

            if blocked:
                log.append({"blocked": blocked})
                continue

            out_photons.append(ph_cur)
            log.append({
                "hits":      [r.tolist() for r in r_hits],
                "I_out":     round(ph_cur.I, 6),
                "waypoints": [r_before.tolist()] + [r.tolist() for r in r_hits],
            })

        n_blocked = len(photons) - len(out_photons)

        # ── ABCD paraxial properties (at the reference wavelength) ───────
        n_media_stack = [1.0] + g_ref["n_glass"] + [1.0]
        A, B, C, D = _stack_abcd(g_ref["radii"], g_ref["thicknesses"], n_media_stack)
        focal       = _abcd_to_focal(A, B, C, D)

        def _r(v):
            return round(v, 9) if np.isfinite(v) else None

        info = {
            "n_in":       len(photons),
            "n_out":      len(out_photons),
            "n_blocked":  n_blocked,
            "efl":        _r(focal["efl"]),
            "ffl":        _r(focal["ffl"]),
            "bfl":        _r(focal["bfl"]),
            "h1":         _r(focal["h1"]),
            "h2":         _r(focal["h2"]),
            "A":          round(A, 12),
            "B":          round(B, 12),
            "C":          round(C, 12),
            "D":          round(D, 12),
            "ref_wavelength_nm": round(ref_lam * 1e9, 3),
            "n_elements": g_ref["N"],
            "radii":      g_ref["radii"],
            "thicknesses_used": g_ref["thicknesses"],
            "t_mins":     [round(t, 9) for t in g_ref["t_mins"]],
            "n_glass_used": [round(n, 6) for n in g_ref["n_glass"]],
            "k_glass_used": [round(k, 6) for k in g_ref["k_glass"]],
            "nk_sources":   g_ref["nk_sources"],
            "rays":       log,
        }
        return out_photons, info