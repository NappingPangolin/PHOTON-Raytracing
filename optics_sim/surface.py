"""
Surface class: flat mirror / interface with Fresnel reflection and optional
Goos-Hänchen (GH) and Imbert-Fedorov (IF) beam shifts.

Goos-Hänchen shift  - lateral displacement of the reflected beam *in the
    plane of incidence*, along the surface.
Imbert-Fedorov shift - transverse (out-of-plane) displacement, sensitive to
    the helicity (circular polarisation) of the incoming beam.

References
----------
Aiello & Woerdman, Opt. Lett. 33, 1437 (2008)  - unified GH/IF formulas.
Bliokh & Aiello, J. Opt. 15, 014001 (2013)      - review.
"""

from __future__ import annotations

import numpy as np
from .photon import Photon
from .material import Material, MU0


class Surface:
    """
    A planar reflective / refractive interface.

    Parameters
    ----------
    n, k      : optical constants of the mirror material
    mu_r      : relative magnetic permeability
    density   : (reserved, not used in Fresnel)
    wavelength: free-space wavelength (m)
    normal    : surface normal vector *or* pair of directions.

                Accepted shapes
                ~~~~~~~~~~~~~~~
                (3,)     - normal vector directly
                (2, 3)   - [k_in, k_out]: normal is bisector
                (6,)     - [k_in(3), k_out(3)] concatenated
    r         : position of the surface (set after first reflection if needed)
    beam_shifts: enable Goos-Hänchen + Imbert-Fedorov shifts (default False)
    """

    def __init__(
        self,
        n: float,
        k: float,
        mu_r: float,
        density: float,
        wavelength: float,
        normal,
        r: np.ndarray | None = None,
        *,
        beam_shifts: bool = False,
    ):
        self.n = float(n)
        self.k = float(k)
        self.mu_r = float(mu_r)
        self.density = float(density)
        self.wavelength = float(wavelength)
        self.nk: complex = complex(n, -k)
        self.mu = self.mu_r * MU0
        self.r = np.zeros(3) if r is None else np.asarray(r, dtype=float)
        self.beam_shifts = bool(beam_shifts)

        self.normal = self._parse_normal(normal)

    # ------------------------------------------------------------------
    # Normal parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_normal(nor_in) -> np.ndarray:
        nor_in = np.asarray(nor_in, dtype=float)
        if nor_in.shape == (3,):
            return nor_in / np.linalg.norm(nor_in)
        if nor_in.shape == (6,):
            d_in, d_out = nor_in[:3], nor_in[3:]
            nor = np.linalg.norm(d_in) * d_out - np.linalg.norm(d_out) * d_in
            return nor / np.linalg.norm(nor)
        if nor_in.shape == (2, 3):
            d_in, d_out = nor_in[0], nor_in[1]
            nor = np.linalg.norm(d_in) * d_out - np.linalg.norm(d_out) * d_in
            return nor / np.linalg.norm(nor)
        raise ValueError(
            "normal must be shape (3,), (6,), or (2,3). Got: " + str(nor_in.shape)
        )

    # ------------------------------------------------------------------
    # Distance helper
    # ------------------------------------------------------------------

    def calc_distance_to_surface(self, beams) -> np.ndarray:
        """
        Return the propagation distance from each photon to this surface plane.

        Parameters
        ----------
        beams : Photon or list of Photon
        """
        if isinstance(beams, Photon):
            beams = [beams]
        dists = np.zeros(len(beams))
        for i, p in enumerate(beams):
            k_hat = np.real(p.k) / np.linalg.norm(np.real(p.k))
            dists[i] = np.dot(self.r - np.real(p.r), self.normal) / np.dot(k_hat, self.normal)
        return dists

    # ------------------------------------------------------------------
    # Main reflection
    # ------------------------------------------------------------------

    def reflect(
        self,
        photon_i: Photon,
        medium1: Material,
        *,
        show_polarization: bool = False,
        beam_shifts: bool | None = None,
    ) -> tuple[Photon, Photon]:
        """
        Compute Fresnel reflection (and transmission) at this surface.

        Parameters
        ----------
        photon_i         : incident Photon
        medium1          : Material of the medium the photon travels in
        show_polarization: print s/p diagnostics
        beam_shifts      : override the instance-level ``beam_shifts`` flag

        Returns
        -------
        photon_r : reflected Photon
        photon_t : transmitted Photon
        """
        use_shifts = self.beam_shifts if beam_shifts is None else beam_shifts

        npk = np.linalg.norm(np.real(photon_i.k))
        photon_r = photon_i._copy()
        photon_t = photon_i._copy()

        # Normalise E
        e_norm = np.linalg.norm(photon_r.E)
        if e_norm > 0:
            photon_r.E = photon_r.E / e_norm
            photon_t.E = photon_t.E / e_norm

        normal = self.normal.copy()

        # Angle of incidence
        theta_i = np.arccos(
            np.clip(-np.dot(np.real(photon_i.k), normal) / npk, -1.0, 1.0)
        )
        if theta_i > np.pi / 2:
            normal = -normal
            theta_i = np.arccos(
                np.clip(-np.dot(np.real(photon_i.k), normal) / npk, -1.0, 1.0)
            )

        # Reflected k
        mag = np.dot(np.real(photon_i.k), normal)
        photon_r.k = photon_i.k - 2 * mag * normal

        # S/P decomposition
        Sn = np.cross(normal, np.real(photon_i.k) / npk)
        if np.linalg.norm(Sn) < 1e-12:
            # Normal incidence
            Eip = photon_i.E.copy()
            Eis = np.zeros(3, dtype=complex)
            if show_polarization:
                print("Normal incidence")
        else:
            Sn = Sn / np.linalg.norm(Sn)
            Pn_i = -np.cross(Sn, np.real(photon_i.k) / npk)
            Eis = np.dot(Sn, photon_i.E) * Sn
            Eip = np.dot(Pn_i, photon_i.E) * Pn_i
            if show_polarization:
                if np.linalg.norm(Eis) < 1e-9:
                    print("p-polarised")
                if np.linalg.norm(Eip) < 1e-9:
                    print("s-polarised")

        # Fresnel coefficients
        # cos_i from geometry; cos_t computed directly via the principal
        # branch of sqrt(1 - sin_t^2), matching the convention used by
        # the rs/rp formulas below (cos_i, cos_t both "point" the same way,
        # analogous to the old arcsin-based theta_t).
        #
        # NOTE: do NOT derive cos_t from k_tn (the transmitted-k branch
        # variable computed further down). k_tn's sign is chosen for the
        # *ray propagation direction* convention (Re(k_tn) > 0 so that
        # k_t = k_par - k_tn*normal points into the transmitted medium).
        # Dividing that k_tn by the complex n_t does NOT yield cos(theta_t)
        # in the same sign convention the Fresnel rs/rp formulas expect -
        # doing so silently flips the sign of Re(cos_t) for absorbing
        # media, which corrupts Rs/Rp (verified: Rs is accidentally
        # insensitive to this sign flip due to symmetric placement of
        # cos_t in its formula, but Rp is not, producing wrong reflected
        # intensities while leaving Rs looking deceptively "fine").
        cos_i  = np.cos(theta_i)
        _n_i    = medium1.nk
        _n_t    = self.nk
        sin_t   = (_n_i / _n_t) * np.sin(theta_i)
        cos_t   = np.sqrt(1.0 - sin_t ** 2 + 0j)
        if np.real(cos_t) < 0.0:
            cos_t = -cos_t        # principal branch: Re(cos_t) >= 0
        ni_mu   = medium1.nk / medium1.mu
        nt_mu   = self.nk / self.mu

        rs = (ni_mu * cos_i - nt_mu * cos_t) / (ni_mu * cos_i + nt_mu * cos_t)
        rp = (nt_mu * cos_i - ni_mu * cos_t) / (nt_mu * cos_i + ni_mu * cos_t)

        # Reflected E
        Ers = Eis * rs

        if np.linalg.norm(Sn) < 1e-12:
            # Normal incidence: Eip already holds the *entire* (purely
            # transverse) field and has no component along the normal, so
            # "v - 2(v.n)n" is the identity here - no direction transform
            # needed, complex phase is untouched either way.
            Erp = -Eip
        else:
            # Oblique incidence: Eip = Eip_scalar * Pn_i, where Eip_scalar is
            # a complex amplitude that may carry an arbitrary phase (e.g. a
            # circularly/elliptically polarised source, or upstream phase
            # shifts). Reflecting only np.real(Eip) - as the previous
            # implementation did - discards that phase whenever Pn_i has a
            # nonzero component along the normal (the general oblique case),
            # since the imaginary part is silently dropped from the
            # projection instead of following the amplitude through.
            #
            # Fix: reflect the real, purely geometric p-direction unit
            # vector about the normal to get the new p-direction, then
            # reapply the full complex amplitude (phase and all) onto it.
            Eip_scalar = np.dot(Pn_i, photon_i.E)
            Pn_r = Pn_i - 2.0 * np.dot(normal, Pn_i) * normal
            pn_r_norm = np.linalg.norm(Pn_r)
            if pn_r_norm > 1e-12:
                Pn_r = Pn_r / pn_r_norm
            Erp = -(Pn_r * Eip_scalar)
        Erp = Erp * rp

        # Pn_r = -np.cross(Sn, np.real(photon_r.k) / np.linalg.norm(np.real(photon_r.k))) # this regularly breaks with normal incidence. using my original code instead, much more resilient.
        # Eip_scalar = np.dot(Pn_i, photon_i.E)   # scalar amplitude
        # Erp = rp * Eip_scalar * Pn_r

        Rs = np.abs(np.linalg.norm(Ers)) ** 2
        Rp = np.abs(np.linalg.norm(Erp)) ** 2

        photon_r.E = Ers + Erp
        photon_r.I = photon_i.I * (Rs + Rp)
        e_norm_r = np.linalg.norm(photon_r.E)
        if e_norm_r > 0:
            photon_r.E = photon_r.E / e_norm_r

        # Clean up small values
        photon_r.E[np.abs(photon_r.E) < 1e-9] = 0
        e_norm_r2 = np.linalg.norm(photon_r.E)
        if e_norm_r2 > 0:
            photon_r.E = photon_r.E / e_norm_r2
        photon_r.k[np.abs(photon_r.k) < 1e-9] = 0
        k_norm_r = np.linalg.norm(photon_r.k)
        if k_norm_r > 0:
            photon_r.k = photon_r.k / k_norm_r * 2 * np.pi / photon_r.wavelength
        photon_r.B = np.cross(np.real(photon_r.k), np.real(photon_r.E))
        b_norm = np.linalg.norm(photon_r.B)
        if b_norm > 0:
            photon_r.B = photon_r.B / b_norm

        # ------------------------------------------------------------------
        # Transmitted k — unified vector Snell's law for real and complex n
        # ------------------------------------------------------------------
        # Strategy (matches AbsorbingMedia_working.py):
        #   1. Conserve the tangential component of k across the interface.
        #   2. Solve the dispersion relation k_t² = (k0 · n_t)² for the
        #      normal component k_tn (complex for absorbing media).
        #   3. Choose the branch: Re(k_tn) > 0 (forward), Im(k_tn) ≥ 0 (decay).
        #   4. The ray-direction used for subsequent propagation is Re(k_t),
        #      i.e. the phase-front propagation direction — this equals the
        #      Poynting-vector direction for non-magnetic, non-absorbing media,
        #      and is the physically correct wavefront normal in absorbing media.
        #
        # The normal here must point INTO the incident medium (away from
        # the transmitted side), which is already guaranteed by the flip
        # above (theta_i ≤ π/2, so dot(k_hat_i, normal) < 0).
        # ------------------------------------------------------------------
        k0      = 2.0 * np.pi / photon_t.wavelength   # free-space wavenumber
        n_i     = medium1.nk                           # complex n of incident medium
        n_t_cplx = self.nk                             # complex n of transmitted medium
        k_hat_i = np.real(photon_i.k).astype(float) / npk

        # Tangential (parallel) component — conserved across the interface.
        # k_i_normal projects onto the surface normal that points into medium 1
        # (i.e. against the ray), so k_i · n̂ < 0.
        k_i_vec  = k0 * n_i * k_hat_i                  # complex wave vector of incident ray
        k_par    = k_i_vec - np.dot(k_i_vec, normal) * normal  # tangential part (lies in surface)

        # Normal component of transmitted wave from dispersion relation
        k_par_sq = np.dot(k_par, k_par)                # may be complex if n_i is complex
        k_tn_sq  = (k0 * n_t_cplx) ** 2 - k_par_sq
        k_tn     = np.sqrt(complex(k_tn_sq))

        # Branch selection: transmitted wave must travel away from the surface
        # into the transmitted medium, i.e. in the direction of –normal
        # (since normal points into the incident medium).
        # We need Re(k_tn) > 0 so that k_t = k_par – k_tn·normal points
        # away from the surface into the transmitted half-space.
        # Then also ensure Im(k_tn) ≥ 0 (amplitude decays into the medium,
        # not grows), but only flip the sign of Im when Re is already correct.
        if np.real(k_tn) < 0.0:
            k_tn = -k_tn          # flip both Re and Im together
        # After guaranteeing Re > 0, also check Im ≥ 0 independently.
        # For most physical media Im(k_tn) ≥ 0 automatically once Re > 0.
        if np.imag(k_tn) < 0.0:
            k_tn = k_tn.real - 1j * k_tn.imag  # only flip Im, keep Re positive

        # Full complex transmitted wave vector
        # The transmitted ray travels in the –normal half-space
        k_t_cplx = k_par - k_tn * normal               # complex k vector in global frame

        # Ray direction for propagation = direction of Re(k_t)
        # (phase-front normal; equals Poynting direction for non-magnetic media).
        k_t_real = np.real(k_t_cplx)
        k_t_norm = np.linalg.norm(k_t_real)

        if k_t_norm > 1e-30 and np.real(k_tn) > 1e-10:
            # Forward-propagating transmitted ray: set k to the correct magnitude
            # |k| = k0 = 2π/λ₀ for the free-space reference, but we store the
            # real-space propagation direction scaled to k0 as convention.
            photon_t.k = (k_t_real / k_t_norm) * k0
        else:
            # Total internal reflection or evanescent wave: zero transmitted k
            photon_t.k = np.zeros(3, dtype=complex)

        # ------------------------------------------------------------------
        # Beam shifts (Goos-Hänchen + Imbert-Fedorov)
        # ------------------------------------------------------------------
        if use_shifts:
            photon_r = self._apply_beam_shifts(
                photon_r, photon_i, normal, theta_i, rs, rp, medium1
            )

        return photon_r, photon_t

    # ------------------------------------------------------------------
    # Goos-Hänchen and Imbert-Fedorov shifts
    # ------------------------------------------------------------------

    def _apply_beam_shifts(
        self,
        photon_r: Photon,
        photon_i: Photon,
        normal: np.ndarray,
        theta_i: float,
        rs: complex,
        rp: complex,
        medium1: Material,
    ) -> Photon:
        """
        Apply Goos-Hänchen (GH) and Imbert-Fedorov (IF) beam-position shifts
        to the reflected photon.

        The GH shift is a longitudinal (in-plane) displacement of the beam
        centroid along the surface in the plane of incidence.
        The IF shift is a transverse displacement perpendicular to the plane
        of incidence, proportional to the photon spin (helicity).

        Formulas follow Aiello & Woerdman, Opt. Lett. 33, 1437 (2008), which
        express the shifts in terms of d(arg r)/d(theta) evaluated at theta_i.

        The shifts are applied as a displacement of photon_r.r along the
        surface.
        """
        k0 = 2 * np.pi / photon_i.wavelength
        k_hat_i = np.real(photon_i.k) / np.linalg.norm(np.real(photon_i.k))

        # In-plane unit vector (lies in surface, in plane of incidence)
        # p_hat: component of k_in projected onto the surface
        k_proj = k_hat_i - np.dot(k_hat_i, normal) * normal
        k_proj_norm = np.linalg.norm(k_proj)
        if k_proj_norm < 1e-12:
            return photon_r  # normal incidence - no well-defined in-plane dir
        p_hat = k_proj / k_proj_norm  # in-plane unit vector along propagation direction

        # Out-of-plane unit vector on the surface
        q_hat = np.cross(normal, p_hat)
        q_hat = q_hat / np.linalg.norm(q_hat)

        # Numerical derivative of arg(r_s), arg(r_p) w.r.t. theta
        dtheta = 1e-6
        theta_plus = theta_i + dtheta
        theta_minus = theta_i - dtheta

        def _fresnel(theta):
            sin_t = (medium1.nk / self.nk) * np.sin(theta)
            cos_t = np.sqrt(1 - sin_t ** 2 + 0j)
            cos_i = np.cos(theta)
            ni_mu = medium1.nk / medium1.mu
            nt_mu = self.nk / self.mu
            rs_ = (ni_mu * cos_i - nt_mu * cos_t) / (ni_mu * cos_i + nt_mu * cos_t)
            rp_ = (nt_mu * cos_i - ni_mu * cos_t) / (nt_mu * cos_i + ni_mu * cos_t)
            return rs_, rp_

        rs_p, rp_p = _fresnel(theta_plus)
        rs_m, rp_m = _fresnel(theta_minus)

        d_phi_s = (np.angle(rs_p) - np.angle(rs_m)) / (2 * dtheta)
        d_phi_p = (np.angle(rp_p) - np.angle(rp_m)) / (2 * dtheta)

        # Decompose reflected E into s and p components
        Sn = np.cross(normal, k_hat_i)
        Sn_norm = np.linalg.norm(Sn)
        if Sn_norm < 1e-12:
            return photon_r
        Sn = Sn / Sn_norm
        Pn_i = -np.cross(Sn, k_hat_i)

        E_r = photon_r.E
        Es_r = np.dot(Sn, E_r)
        Ep_r = np.dot(Pn_i, E_r)

        # Intensities
        Is = np.abs(Es_r) ** 2
        Ip = np.abs(Ep_r) ** 2
        I_tot = Is + Ip + 1e-300

        # Helicity (Stokes V-like parameter normalised)
        # σ = Im(Es* Ep) / (|Es|² + |Ep|²) in the range [-1, 1]
        sigma = np.imag(np.conj(Es_r) * Ep_r) / I_tot

        # --- Goos-Hänchen shift (in-plane, along p_hat) ---
        # D_GH = (Is * d_phi_s/dθ + Ip * d_phi_p/dθ) / (k0 * cos θ * I_tot)
        # factor 1/cos(theta_i) converts from angular to spatial shift
        gh_shift = (Is * d_phi_s + Ip * d_phi_p) / (k0 * np.cos(theta_i) * I_tot)

        # --- Imbert-Fedorov shift (out-of-plane, along q_hat) ---
        # D_IF = σ * (|rs|² + |rp|²) / (k0 * I_tot)
        # This is the transverse spin Hall of light shift.
        # Sign follows Bliokh & Aiello (2013) convention.
        Rsp = (np.abs(rs) ** 2 + np.abs(rp) ** 2)
        if_shift = sigma * Rsp / (k0 * I_tot)

        # Apply shifts to position (spatial displacement on the surface)
        photon_r.r = photon_r.r + gh_shift * p_hat + if_shift * q_hat

        return photon_r

    # ------------------------------------------------------------------
    # Bunch helpers
    # ------------------------------------------------------------------

    def reflect_bunch(
        self, photon_bunch: list, medium1: Material
    ) -> tuple[list, list, list]:
        """Reflect a list of photons off this surface."""
        dists = self.calc_distance_to_surface(photon_bunch)
        photon_bunch_r = []
        photon_bunch_t = []
        photon_bunch_i_out = []
        for p, d in zip(photon_bunch, dists):
            p_at = p.travel_distance(d)
            pr, pt = self.reflect(p_at, medium1)
            photon_bunch_r.append(pr)
            photon_bunch_t.append(pt)
            photon_bunch_i_out.append(p_at)
        return photon_bunch_r, photon_bunch_t, photon_bunch_i_out

    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Surface(n={self.n}, k={self.k}, normal={self.normal}, "
            f"beam_shifts={self.beam_shifts})"
        )