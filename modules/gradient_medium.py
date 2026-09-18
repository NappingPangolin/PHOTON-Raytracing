"""
modules/gradient_medium.py
"""

from __future__ import annotations

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon
from .base import OpticalModule, LnkMixin, ParamSpec

C_LIGHT = 299792458.0


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers (module-level, no dependency on instance state)
# ══════════════════════════════════════════════════════════════════════════════

def _ray_box_intersect(origin: np.ndarray, khat: np.ndarray,
                        box_min: np.ndarray, box_max: np.ndarray):
    """
    Slab-method ray/AABB intersection.

    Returns (t_enter, t_exit) in units of |khat| (khat need not be unit
    length; distances are along khat as given), or None if the ray misses
    the box or the box is entirely behind the ray origin.

    t_enter is clamped to 0 if the origin already lies inside the box.
    """
    tmin, tmax = -np.inf, np.inf
    for i in range(3):
        if abs(khat[i]) < 1e-15:
            if origin[i] < box_min[i] or origin[i] > box_max[i]:
                return None
            continue
        t1 = (box_min[i] - origin[i]) / khat[i]
        t2 = (box_max[i] - origin[i]) / khat[i]
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None
    if tmax < 0:
        return None
    return max(tmin, 0.0), tmax


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for rotating by *angle* radians around unit *axis*."""
    K = np.array([
        [0.0,        -axis[2],  axis[1]],
        [axis[2],     0.0,     -axis[0]],
        [-axis[1],    axis[0],  0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_from_z(nhat: np.ndarray) -> np.ndarray:
    """
    Rotation matrix R such that  R @ [0,0,1] = nhat  (minimal rotation),
    replicating THREE.Quaternion().setFromUnitVectors(zAxis, nhat), which is
    what the frontend uses to orient the (1D exp) gradient box.  Returns the
    identity if nhat is already +z.
    """
    z = np.array([0.0, 0.0, 1.0])
    nhat = np.asarray(nhat, dtype=float)
    nrm = np.linalg.norm(nhat)
    nhat = nhat / nrm if nrm > 1e-12 else z.copy()

    v = np.cross(z, nhat)
    c = float(np.dot(z, nhat))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate 180 deg around any axis perpendicular to z
        axis = np.array([1.0, 0.0, 0.0])
        if abs(nhat[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, z) * z
        axis = axis / np.linalg.norm(axis)
        return _rodrigues(axis, np.pi)

    axis = v / s
    angle = float(np.arctan2(s, c))
    return _rodrigues(axis, angle)


def _segment_box_exit_fraction(r_prev: np.ndarray, r_new: np.ndarray,
                                box_min: np.ndarray, box_max: np.ndarray) -> float:
    """
    Given a short straight sub-segment r_prev -> r_new where r_prev is inside
    the box and r_new is outside (or on the boundary), return the fraction
    f in [0, 1] along the segment where it first crosses the box boundary.

    Falls back to 1.0 (i.e. use r_new as-is) if no crossing is found, which
    can only happen from numerical edge cases.
    """
    d = r_new - r_prev
    f_exit = 1.0
    for i in range(3):
        if abs(d[i]) < 1e-15:
            continue
        for bound in (box_min[i], box_max[i]):
            f = (bound - r_prev[i]) / d[i]
            if 0.0 <= f <= 1.0:
                # Check this crossing point actually lies on the box face
                # (within bounds on the other two axes).
                p = r_prev + f * d
                ok = True
                for j in range(3):
                    if j == i:
                        continue
                    if p[j] < box_min[j] - 1e-12 or p[j] > box_max[j] + 1e-12:
                        ok = False
                        break
                if ok and f < f_exit:
                    f_exit = f
    return f_exit


# ══════════════════════════════════════════════════════════════════════════════
# Module
# ══════════════════════════════════════════════════════════════════════════════

class GradientMedium(LnkMixin, OpticalModule):

    MODULE_TYPE = "gradient_medium"
    LABEL       = "Gradient Medium"
    COLOR       = "#22d3ee"   # cyan
    ICON        = "≋"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            # ── gradient source ───────────────────────────────────────────────
            ParamSpec(
                "gradient_mode", "Gradient Source", "select", "exponential",
                options=["1D exp","3D exp", "grid_file"],
                tooltip=(
                    "1D exp: analytic n₀·exp(-x̃/L) model.  "
                    "3D exp: analytic n₀·exp(-|r|/L) model.  "
                    "grid_file: load a density map from a file (VTK/CSV/NPY/NPZ)."
                ),
            ),

            # ── exponential model ─────────────────────────────────────────────
            ParamSpec("r_x", "Origin X", "float", 0.0, "m",
                      min=-100.0, max=100.0,
                      tooltip="Position r₀ where peak density ρ₀ is defined"),
            ParamSpec("r_y", "Origin Y", "float", 0.0, "m",
                      min=-100.0, max=100.0),
            ParamSpec("r_z", "Origin Z", "float", 0.0, "m",
                      min=-100.0, max=100.0),
            ParamSpec("l_x", "size X", "float", 0.1, "m",
                      min=1e-9, max=100.0,
                      tooltip="box size with center at r₀, in which the gradient exists"),
            ParamSpec("l_y", "size Y", "float", 0.1, "m",
                      min=1e-9, max=100.0),
            ParamSpec("l_z", "size Z", "float", 0.1, "m",
                      min=1e-9, max=100.0),

#show grad_n_i parameters, only if 1D exp is selected.
            ParamSpec("grad_n_x", "Gradient Normal X", "float", 1.0, "",
                      min=-1.0, max=1.0,
                      tooltip="Unit vector n̂ along which 1D exp density decreases. "
                              "Density increases in the -n̂ direction."),
            ParamSpec("grad_n_y", "Gradient Normal Y", "float", 0.0, "",
                      min=-1.0, max=1.0),
            ParamSpec("grad_n_z", "Gradient Normal Z", "float", 0.0, "",
                      min=-1.0, max=1.0),

            ParamSpec("rho0", "Peak Density ρ₀", "float", 1e25, "m⁻³",
                      min=0.0, max=1e32,
                      tooltip="Electron/neutral density at r₀ (the dense side)"),
            ParamSpec("scale_length", "Scale Length L", "float", 0.01, "m",
                      min=1e-9, max=100.0,
                      tooltip="Exponential scale length: ρ(r) = ρ₀·exp(-x̃/L)"),

            # ── grid file ─────────────────────────────────────────────────────
            ParamSpec(
                "grid_file_rows", "Density Map Data", "lnk_table", [],
                tooltip=(
                    "Serialised density-map rows [x, y, z, density] embedded "
                    "in the scene file.  Populated automatically when a file "
                    "is uploaded via the frontend; do not edit by hand."
                ),
            ),
            ParamSpec(
                "density_scale", "Density Scale Factor", "float", 1.0, "",
                min=0.0, max=1e20,
                tooltip="Multiply all density values by this factor (unit conversion).",
            ),

            # ── index mapping (Lorentz-Lorenz) ─────────────────────────────────
            ParamSpec("n", "n (reference, real)", "float", 1.5, "",
                      min=0.0, max=10.0,
                      tooltip="Refractive index of the medium at the reference "
                              "density ρ_ref (manual mode). Used with ρ_ref to "
                              "derive the Lorentz-Lorenz polarisability."),
            ParamSpec("k", "k (reference, extinction)", "float", 0.0, "",
                      min=0.0, max=30.0,
                      tooltip="Extinction coefficient at ρ_ref (manual mode); "
                              "scaled linearly with local density for Beer-Lambert "
                              "absorption."),
            ParamSpec(
                "rho_max", "Reference Density ρ_ref", "float", 1e25, "m⁻³",
                min=0.0, max=1e32,
                tooltip=(
                    "Density at which the n/k values above (or the lnk table) "
                    "are defined.  0 = auto (uses ρ₀ for the exponential model, "
                    "or the grid maximum)."
                ),
            ),
            # ── wavelength-dependent n/k (LnkMixin) ──────────────────────
            *cls.lnk_param_specs(),

            # ── propagation length ────────────────────────────────────────────
            ParamSpec(
                "total_length", "Total Path Length", "float", 5e-3, "m",
                min=1e-9, max=1000.0,
                tooltip="Fallback path length used only when no box (l_x/y/z) "
                        "is defined [m]. When a box is set, the actual "
                        "entry/exit crossing distance is used instead.",
            ),

            # ── integrator ────────────────────────────────────────────────────
            ParamSpec(
                "n_steps", "Number of Steps", "int", 0,
                min=0, max=100_000,
                tooltip=(
                    "Number of RK integration steps across the box.  "
                    "0 = auto-select based on the crossing distance and scale length."
                ),
            ),
            ParamSpec(
                "dt", "Step Size dt", "float", 0.0, "m",
                min=0.0, max=1.0,
                tooltip=(
                    "Arc-length integration step [m].  "
                    "0 = auto (crossing_distance / n_steps)."
                ),
            ),
            ParamSpec(
                "dx", "Gradient dx", "float", 0.0, "m",
                min=0.0, max=1.0,
                tooltip=(
                    "Finite-difference step for ∇(n²) [m].  "
                    "0 = auto (crossing_distance / 1000, clamped to [50nm, 1mm])."
                ),
            ),
            ParamSpec(
                "max_waypoints", "Max Waypoints / Ray", "int", 20,
                min=2, max=2000,
                tooltip=(
                    "Number of path points reported per ray for the 3-D viewer "
                    "(the physics itself always runs at full resolution; this "
                    "only controls how many points are sent to the frontend)."
                ),
            ),
        ]

    # ══════════════════════════════════════════════════════════════════════
    # Density-field helpers
    # ══════════════════════════════════════════════════════════════════════
    #
    # These describe the *geometry* of the medium (where the density is high
    # or low) so the frontend preview and the physics below agree on what
    # "the gradient" means.

    def _origin(self) -> np.ndarray:
        return np.array([
            float(self.params.get("r_x", 0.0)),
            float(self.params.get("r_y", 0.0)),
            float(self.params.get("r_z", 0.0)),
        ], dtype=float)

    def _box_size(self) -> np.ndarray:
        return np.array([
            float(self.params.get("l_x", 0.0)),
            float(self.params.get("l_y", 0.0)),
            float(self.params.get("l_z", 0.0)),
        ], dtype=float)

    def _box_rotation(self) -> np.ndarray:
        """
        Rotation matrix (local -> global) for the box frame, matching the
        frontend exactly: for "1D exp" the box is rotated so its local Z
        axis points along the gradient normal n̂; every other mode keeps
        the box axis-aligned (identity).
        """
        mode = self.params.get("gradient_mode", "1D exp")
        if mode != "1D exp":
            return np.eye(3)
        return _rotation_from_z(self._grad_normal())

    def _grad_normal(self) -> np.ndarray:
        v = np.array([
            float(self.params.get("grad_n_x", 1.0)),
            float(self.params.get("grad_n_y", 0.0)),
            float(self.params.get("grad_n_z", 0.0)),
        ], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])

    def density_at(self, r: np.ndarray) -> float:
        """
        Return the number density ρ(r) [m⁻³] predicted by the currently
        selected analytic model, evaluated at global position *r*.

          1D exp: ρ(r) = ρ₀ · exp(-s / L),   s = (r - r₀) · n̂
          3D exp: ρ(r) = ρ₀ · exp(-|r - r₀| / L)
          grid_file: not implemented yet -> always 0.
        """
        mode = self.params.get("gradient_mode", "1D exp")
        rho0 = float(self.params.get("rho0", 0.0))
        L    = max(float(self.params.get("scale_length", 0.0)), 1e-300)
        r0   = self._origin()

        if mode == "1D exp":
            s = float(np.dot(np.asarray(r, dtype=float) - r0, self._grad_normal()))
            return rho0 * np.exp(-s / L)

        if mode == "3D exp":
            dist = float(np.linalg.norm(np.asarray(r, dtype=float) - r0))
            return rho0 * np.exp(-dist / L)

        return 0.0   # grid_file: density map not implemented yet

    # ── Lorentz-Lorenz density -> refractive index mapping ─────────────────

    @staticmethod
    def _lorentz_lorenz_n(n_ref: float, rho: float, rho_ref: float) -> float:
        """
        n(ρ) via the Lorentz-Lorenz relation, using a reference (n_ref, ρ_ref)
        point to fix the molar polarisability:

            R      = (n_ref² - 1) / (n_ref² + 2)
            n(ρ)   = sqrt( (1 + 2R·ρ/ρ_ref) / (1 - R·ρ/ρ_ref) )

        Degenerates to n_ref (density-independent) if ρ_ref <= 0.

        The relation has a genuine physical singularity as R·ρ/ρ_ref -> 1
        (the "close-packing" limit) - local densities that push past it are
        clamped rather than allowed to diverge, so a stray high-density
        region can't blow up the integrator's step size.
        """
        if rho_ref <= 0:
            return n_ref
        R = (n_ref**2 - 1.0) / (n_ref**2 + 2.0)
        x = R * (rho / rho_ref)
        x = min(x, 0.98)   # stay well clear of the 1/R close-packing singularity
        num = max(1.0 + 2.0*x, 0.0)
        den = max(1.0 - x, 1e-6)
        n_val = float(np.sqrt(num / den))
        return min(n_val, 15.0)   # hard safety cap (well above any realistic optical n)

    def _reference_density(self) -> float:
        rho_max = float(self.params.get("rho_max", 0.0))
        if rho_max > 0:
            return rho_max
        rho0 = float(self.params.get("rho0", 0.0))
        return rho0 if rho0 > 0 else 1.0

    def _local_nk(self, r: np.ndarray, wavelength_m: float) -> tuple[float, float]:
        """
        Local (n, k) at global position r, for a photon of the given
        wavelength: look up the reference (n_ref, k_ref) at that wavelength
        (manual or lnk table, via LnkMixin), then map to the local density
        via Lorentz-Lorenz (n) and linear scaling (k, simple Beer-Lambert
        concentration assumption absent better data).
        """
        n_ref, k_ref = self._resolve_nk(wavelength_m, n_key="n", k_key="k")
        rho     = self.density_at(r)
        rho_ref = self._reference_density()
        n_local = self._lorentz_lorenz_n(n_ref, rho, rho_ref)
        k_local = max(k_ref * (rho / rho_ref), 0.0) if rho_ref > 0 else k_ref
        return n_local, k_local

    def _index_gradient(self, r: np.ndarray, wavelength_m: float, dx: float) -> np.ndarray:
        """
        D(r) = n(r)*grad(n(r)), using ONLY the real part of n (curvature is
        driven by the real index; absorption is handled separately via the
        imaginary part / Beer-Lambert law).  Central differences of n², per:

            D_i = (n(r+dx·ê_i)² - n(r-dx·ê_i)²) / (4·dx)
        """
        d = np.zeros(3)
        for i in range(3):
            step = np.zeros(3); step[i] = dx
            n_plus,  _ = self._local_nk(r + step, wavelength_m)
            n_minus, _ = self._local_nk(r - step, wavelength_m)
            d[i] = (n_plus**2 - n_minus**2) / (4.0 * dx)
        return d

    # ══════════════════════════════════════════════════════════════════════
    # process()
    # ══════════════════════════════════════════════════════════════════════

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        """
        Trace each photon through the medium's box.

        1. Find where the (straight) incoming ray enters/exits the box
           (AABB), like the surface-hit tests in compound_lens.
        2. Travel the photon (in vacuum) up to the entry point.
        3. Integrate the curved path inside the box with the modified RK4
           scheme (matches compute_delta_d_V9.m), using only the real part
           of n for curvature and the imaginary part for Beer-Lambert
           absorption along each sub-step.
        4. Stop at the box boundary (refined to the exact crossing point),
           take the exit direction from the integrator, and report a
           down-sampled (<= max_waypoints) path for the 3-D viewer.

        grid_file mode and photons that miss the box pass straight through
        unaffected.
        """
        mode = self.params.get("gradient_mode", "1D exp")

        box_size = self._box_size()
        has_box  = bool(np.all(box_size > 0))
        r0       = self._origin()

        n_viz = max(2, min(int(self.params.get("max_waypoints", 20)), 20))

        out_photons: list[Photon] = []
        ray_info: list[dict] = []

        if mode == "grid_file" or not has_box:
            # Not implemented / no box defined -> pass straight through.
            for ph in photons:
                out_photons.append(ph)
                ray_info.append({
                    "hit_box": False,
                    "reason": "grid_file not implemented yet" if mode == "grid_file"
                              else "no box defined (l_x/l_y/l_z must all be > 0)",
                })
            return out_photons, {
                "physics_implemented": mode in ("1D exp", "3D exp") and has_box,
                "gradient_mode": mode,
                "origin": r0.tolist(),
                "box_size": box_size.tolist(),
                "n_photons": len(photons),
                "rays": ray_info,
            }

        # Box is defined in a LOCAL frame centred on r0; for "1D exp" that
        # local frame is rotated so its Z axis follows the gradient normal
        # n̂ — exactly matching the frontend's Three.js box. box_min/box_max
        # below are therefore in local (unrotated) coordinates.
        Rbox = self._box_rotation()

        def to_local(p: np.ndarray) -> np.ndarray:
            return Rbox.T @ (p - r0)

        def to_global(p_local: np.ndarray) -> np.ndarray:
            return Rbox @ p_local + r0

        box_min = -box_size / 2.0
        box_max = box_size / 2.0
        L_scale = max(float(self.params.get("scale_length", 0.0)), 1e-300)

        dt_param = float(self.params.get("dt", 0.0))
        dx_param = float(self.params.get("dx", 0.0))
        n_steps_param = int(self.params.get("n_steps", 0))
        MAX_PHYSICS_STEPS = 20000

        for ph in photons:
            lam   = float(ph.wavelength)
            r_cur = np.real(ph.r).astype(float)
            r_before = r_cur.copy()   # true incoming position, before any travel
            k_vec = np.real(ph.k).astype(float)
            k_mag = np.linalg.norm(k_vec)
            if k_mag < 1e-300:
                out_photons.append(ph)
                ray_info.append({"hit_box": False, "reason": "zero wavevector"})
                continue
            khat = k_vec / k_mag

            r_cur_local  = to_local(r_cur)
            khat_local   = Rbox.T @ khat   # rotation only, no translation
            hit = _ray_box_intersect(r_cur_local, khat_local, box_min, box_max)
            if hit is None:
                out_photons.append(ph)
                ray_info.append({"hit_box": False, "reason": "ray misses box"})
                continue
            t_enter, t_exit = hit
            L_box = t_exit - t_enter
            if L_box <= 1e-15:
                out_photons.append(ph)
                ray_info.append({"hit_box": False, "reason": "zero-length crossing"})
                continue

            # ── travel (vacuum) up to the box entry point ──────────────────
            ph_entry = ph.travel_distance(t_enter) if t_enter > 0 else ph._copy()
            r_start  = np.real(ph_entry.r).astype(float)

            # ── resolve integrator step sizes ───────────────────────────────
            if dt_param > 0:
                dt = dt_param
            elif n_steps_param > 0:
                dt = L_box / n_steps_param
            else:
                n_auto = int(np.clip(np.ceil(L_box / max(L_scale/20.0, 1e-300)), 50, 5000))
                dt = L_box / n_auto

            dx = dx_param if dx_param > 0 else float(np.clip(L_box/1000.0, 50e-9, 1e-3))

            max_steps = min(MAX_PHYSICS_STEPS, int(np.ceil(L_box/dt)) * 3 + 50)

            # ── RK4 (compute_delta_d_V9.m scheme) ───────────────────────────
            n_start, k_start = self._local_nk(r_start, lam)
            r_i = r_start.copy()
            T_i = khat * n_start   # ||T(0)|| = n(r(0))

            waypoints  = [r_before.tolist(), r_start.tolist()]
            optical_path  = 0.0
            atten_exponent = 0.0
            elapsed_time   = 0.0
            steps_taken    = 0
            exited         = False
            r_exit         = r_start.copy()

            for _step in range(max_steps):
                D_A = self._index_gradient(r_i, lam, dx)
                A = dt * D_A
                D_B = self._index_gradient(r_i + dt*T_i/2 + dt*A/8, lam, dx)
                B = dt * D_B
                D_C = self._index_gradient(r_i + dt*T_i + dt*B/2, lam, dx)
                C = dt * D_C

                r_next = r_i + dt * (T_i + (A + 2*B)/6.0)
                T_next = T_i + (A + 4*B + C)/6.0

                ds = float(np.linalg.norm(r_next - r_i))
                r_next_local = to_local(r_next)
                inside = np.all(r_next_local >= box_min - 1e-12) and np.all(r_next_local <= box_max + 1e-12)

                if not inside:
                    f = _segment_box_exit_fraction(to_local(r_i), r_next_local, box_min, box_max)
                    r_boundary = r_i + f * (r_next - r_i)
                    ds_partial = float(np.linalg.norm(r_boundary - r_i))
                    n_local, k_local = self._local_nk(r_i, lam)
                    optical_path   += ds_partial * n_local
                    alpha_field     = (2.0*np.pi/lam) * k_local * ds_partial
                    atten_exponent += alpha_field
                    elapsed_time   += ds_partial * n_local / C_LIGHT
                    r_exit = r_boundary
                    waypoints.append(r_exit.tolist())
                    exited = True
                    steps_taken += 1
                    break

                n_local, k_local = self._local_nk(r_i, lam)
                optical_path   += ds * n_local
                alpha_field     = (2.0*np.pi/lam) * k_local * ds
                atten_exponent += alpha_field
                elapsed_time   += ds * n_local / C_LIGHT

                r_i, T_i = r_next, T_next
                steps_taken += 1
                if steps_taken % max(1, max_steps // n_viz) == 0:
                    waypoints.append(r_i.tolist())

            if not exited:
                # Ran out of steps without leaving the box (shouldn't normally
                # happen given the margins above) - use the last position.
                r_exit = r_i
                waypoints.append(r_exit.tolist())

            # down-sample to <= n_viz waypoints for the 3-D viewer
            if len(waypoints) > n_viz:
                idx = np.linspace(0, len(waypoints) - 1, n_viz).round().astype(int)
                waypoints = [waypoints[i] for i in sorted(set(idx.tolist()))]

            t_dir_norm = np.linalg.norm(T_i)
            k_hat_out = (T_i / t_dir_norm) if t_dir_norm > 1e-300 else khat

            # ── build the outgoing photon ────────────────────────────────
            ph_out = ph_entry._copy()
            ph_out.r = r_exit.astype(complex)
            ph_out.k = (k_hat_out * k_mag).astype(complex)
            phase = (2.0*np.pi/lam) * optical_path
            ph_out.E = ph_entry.E * np.exp(1j*phase) * np.exp(-atten_exponent)
            ph_out.I = float(ph_entry.I) * float(np.exp(-2.0*atten_exponent))
            ph_out.t = float(ph_entry.t) + elapsed_time

            out_photons.append(ph_out)
            ray_info.append({
                "hit_box":       True,
                "r_enter":       r_start.tolist(),
                "r_exit":        r_exit.tolist(),
                "steps":         steps_taken,
                "dt_used":       dt,
                "dx_used":       dx,
                "crossing_length": L_box,
                "optical_path":  optical_path,
                "I_in":          float(ph.I),
                "I_out":         float(ph_out.I),
                "n_at_entry":    round(n_start, 6),
                "k_at_entry":    round(k_start, 6),
                "wavelength_nm": round(lam * 1e9, 3),
                "waypoints":     waypoints,
            })

        return out_photons, {
            "physics_implemented": True,
            "gradient_mode":  mode,
            "origin":         r0.tolist(),
            "box_size":       box_size.tolist(),
            "rho0":           float(self.params.get("rho0", 0.0)),
            "scale_length":   L_scale,
            "rho_ref":        self._reference_density(),
            "n_photons":      len(photons),
            "rays":           ray_info,
        }