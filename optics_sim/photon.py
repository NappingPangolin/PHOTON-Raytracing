"""
Photon class: represents a single photon (or plane-wave ray) with position,
wave-vector, polarisation (E-field), and intensity.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import count

_PHOTON_ID_COUNTER = count()

class Photon:
    """
    A ray/photon with:
        r           - position vector  [3]  (metres)
        k           - wave-vector       [3]  (rad m⁻¹), |k| = 2π/λ
        E           - complex electric-field Jones-like vector [3]
        B           - magnetic field direction [3]
        wavelength  - free-space (i.e., vacuum) wavelength (metres)
        I           - intensity (arbitrary units)
        t           - time (how much time passed for the photon to get to current position from its initial position)
        c           - speed of light [m/s]
    """

    def __init__(
        self,
        r:          np.ndarray | list,
        k:          np.ndarray | list,
        E:          np.ndarray | list,
        B:          np.ndarray | list,
        wavelength: float,
        intensity:  float = 1.0,
        time:       float = 0.0,
        photon_id: int | None = None,
    ):
        self.id = (
            next(_PHOTON_ID_COUNTER)
            if photon_id is None
            else photon_id
        )

        self.r          = np.asarray(r, dtype=complex)
        self.k          = np.asarray(k, dtype=complex)
        self.E          = np.asarray(E, dtype=complex)
        self.B          = np.asarray(B, dtype=complex)
        self.wavelength = float(wavelength)
        self.I          = float(intensity)
        self.t          = float(time)
        self.c          = float(299792458.0)

    # ------------------------------------------------------------------
    # Travel helpers
    # ------------------------------------------------------------------

    def travel_distance(self, distance: float, n: float = 1.0, k: float = 0.0) -> "Photon":
        """Return a new photon propagated by *distance* along k.

        This allows propagation "back in time", if the raytracing needs virtual
        photons — these can thus be created easily without breaking anything.

        Parameters
        ----------
        distance : float
            Propagation distance [m].  May be negative (virtual back-propagation).
        n : float
            Real part of the refractive index of the medium being traversed.
            Used to accumulate travel time and optical phase.
        k : float
            Extinction coefficient (imaginary part of the refractive index, ≥ 0).
            Governs Lambert-Beer amplitude attenuation:
                |E(d)| = |E(0)| · exp(-2π k d / λ₀)
            which gives an intensity decay
                I(d)  = I(0)  · exp(-4π k d / λ₀)
            Pass k=0 (default) for a lossless medium.
        """
        p = self._copy()
        k_hat = p.k / np.linalg.norm(p.k)
        p.r = p.r + distance * k_hat
        p.t = p.t + distance * n / p.c

        k_mag = 2.0 * np.pi / p.wavelength

        # --- Propagating phase (real part of index) ---
        phase = k_mag * n * distance          # radians

        # --- Lambert-Beer amplitude attenuation (extinction coefficient) ---
        # The complex wave decays as exp(i k_complex · x) where
        # k_complex = k_mag * (n + ik), so the amplitude envelope decays as
        # exp(-k_mag * k * d) for forward propagation.
        # We use abs(distance) so back-propagated virtual photons don't gain energy.
        alpha_field = k_mag * float(k) * abs(distance)   # field attenuation exponent

        p.E = p.E * np.exp(1j * phase) * np.exp(-alpha_field)
        p.I = float(p.I) * np.exp(-2.0 * alpha_field)    # intensity ∝ |E|²

        return p

    def travel_to_point(self, point: np.ndarray) -> "Photon":
        """Return a new photon whose position is *point*."""
        p = self._copy()
        p.r = np.asarray(point, dtype=complex)
        return p

    def transverse_distance(self, distance: float) -> "Photon":
        """Alias for travel_distance (kept for compatibility)."""
        return self.travel_distance(distance)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_Efield(self):
        """Visualise the complex E-field components."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        comps = ["x", "y", "z"]
        vals  = self.E
        axes[0].bar(comps, np.real(vals))
        axes[0].set_title("Re(E)")
        axes[1].bar(comps, np.imag(vals))
        axes[1].set_title("Im(E)")
        plt.suptitle("Electric field")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _copy(self) -> "Photon":
        """Return a copy that preserves this photon's ID (same physical ray)."""
        return Photon(
            self.r.copy(),
            self.k.copy(),
            self.E.copy(),
            self.B.copy(),
            self.wavelength,
            self.I,
            self.t,
            photon_id=self.id,
        )

    def __repr__(self):
        return (
            f"Photon(r={np.real(self.r)}, |k|={np.abs(np.linalg.norm(self.k)):.3g}, "
            f"|E|={np.abs(np.linalg.norm(self.E)):.3g}, I={self.I:.3g}, "
            f"λ={self.wavelength*1e9:.1f} nm)"
        )


# ---------------------------------------------------------------------------
# Mirror-patch geometry  (ported from MATLAB create_mirror_patch)
# ---------------------------------------------------------------------------

def _create_mirror_patch(
    k_in:   np.ndarray,
    k_out:  np.ndarray,
    center: np.ndarray,
    size:   float,
):
    """
    Return the four corners of a square mirror patch.

    The mirror normal bisects k_in and k_out - exactly the MATLAB logic::

        nor = |k_in| * k_out - |k_out| * k_in

    The patch lies in the plane perpendicular to *nor* and is centred on
    *center*.

    Returns
    -------
    corners : (4, 3) float array, or None for degenerate cases.
    """
    k_in  = np.asarray(k_in,  dtype=float)
    k_out = np.asarray(k_out, dtype=float)

    nor = np.linalg.norm(k_in) * k_out - np.linalg.norm(k_out) * k_in
    nor_norm = np.linalg.norm(nor)
    if nor_norm < 1e-12:
        return None          # beam doesn't change direction
    nor = nor / nor_norm

    # First in-plane vector (perpendicular to normal)
    v1 = np.array([1.0, 0.0, 0.0])
    if np.linalg.norm(v1 - nor) < 1e-6:
        v1 = np.array([0.0, 1.0, 0.0])
    v1 = np.cross(v1, nor)
    v1_n = np.linalg.norm(v1)
    if v1_n < 1e-12:
        return None
    v1 = v1 / v1_n             # unit vector in mirror plane

    # Second in-plane vector (perpendicular to both nor and v1)
    v2 = np.cross(nor, v1)     # already unit length

    half = size / 2.0
    c = np.asarray(center, dtype=float)
    corners = np.array([
        c - v1 * half - v2 * half,
        c + v1 * half - v2 * half,
        c + v1 * half + v2 * half,
        c - v1 * half + v2 * half,
    ])
    return corners


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------

def plot_photon_bunch(
    bunch:          list,
    scale:          float | None = None,
    title:          str   = "Photon path",
    mirror_size:    float = 0.05,
    arrow_every:    int   = 5,
    path_color:     str   = "red",
    mirror_color:   str   = "royalblue",
    mirror_alpha:   float = 0.55,
    path_linewidth: float = 1.5,
    show_arrows:    bool  = True,
    view_angles:    tuple = (10, 30),
):
    """
    Plot a 3-D ray path from a list of :class:`Photon` objects, with a
    square mirror patch drawn at every reflection point (wherever the
    propagation direction changes noticeably).

    The mirror geometry replicates the MATLAB ``create_mirror_patch``
    function: the patch normal bisects the incoming and outgoing wave-vectors
    and the patch is a square of side *mirror_size* centred on the knot.

    Parameters
    ----------
    bunch         : list of Photon  (e.g. the second return value of
                    ``Chamber.reflect``)
    scale         : arrow length (m); None → 2.5 % of path diagonal
    title         : figure title
    mirror_size   : side length of each mirror square patch (metres)
    arrow_every   : draw a k-direction arrow every N photons
    path_color    : colour of the ray polyline
    mirror_color  : face colour of the mirror patches
    mirror_alpha  : transparency of mirror patches (0=invisible, 1=opaque)
    path_linewidth: width of the ray line
    show_arrows   : draw small direction arrows along the path
    view_angles   : (azimuth, elevation) for the initial 3-D view
    """
    if not bunch:
        print("plot_photon_bunch: empty bunch - nothing to plot.")
        return

    # ---- Extract positions and unit k-vectors -------------------------
    xs = [float(np.real(p.r[0])) for p in bunch]
    ys = [float(np.real(p.r[1])) for p in bunch]
    zs = [float(np.real(p.r[2])) for p in bunch]

    ks = []
    for p in bunch:
        kv = np.real(p.k).copy()
        n  = np.linalg.norm(kv)
        ks.append(kv / n if n > 1e-30 else kv)

    # ---- Figure --------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=13, pad=12)

    # ---- Ray polyline --------------------------------------------------
    ax.plot(xs, ys, zs,
            color=path_color, linewidth=path_linewidth,
            solid_capstyle="round", zorder=2, label="ray path")

    # ---- Auto-scale arrows --------------------------------------------
    if scale is None:
        x_r = max(xs) - min(xs) if len(xs) > 1 else 1.0
        y_r = max(ys) - min(ys) if len(ys) > 1 else 1.0
        z_r = max(zs) - min(zs) if len(zs) > 1 else 1.0
        diag  = np.sqrt(x_r**2 + y_r**2 + z_r**2)
        scale = max(diag * 0.025, mirror_size * 0.4)

    # ---- Direction arrows ---------------------------------------------
    if show_arrows:
        for i, (p, khat) in enumerate(zip(bunch, ks)):
            if i % arrow_every != 0:
                continue
            r = np.real(p.r)
            ax.quiver(
                r[0], r[1], r[2],
                khat[0] * scale, khat[1] * scale, khat[2] * scale,
                color="darkred", linewidth=0.7,
                arrow_length_ratio=0.3, alpha=0.75,
            )

    # ---- Mirror patches at reflection knots ---------------------------
    # A reflection is where the propagation direction changes noticeably.
    # We look at consecutive triplets (before, at, after) in the bunch.
    patches = []
    for i in range(1, len(bunch) - 1):
        k_in  = ks[i - 1]
        k_out = ks[i + 1]
        # Skip if beam direction hasn't changed (straight segment)
        if np.dot(k_in, k_out) > 0.9999:
            continue
        center  = np.array([xs[i], ys[i], zs[i]])
        corners = _create_mirror_patch(k_in, k_out, center, mirror_size)
        if corners is not None:
            patches.append(corners)

    if patches:
        poly = Poly3DCollection(
            patches,
            facecolor=mirror_color,
            edgecolor="navy",
            linewidth=0.7,
            alpha=mirror_alpha,
            zorder=3,
        )
        ax.add_collection3d(poly)

    # ---- Start / end markers ------------------------------------------
    ax.scatter([xs[0]],  [ys[0]],  [zs[0]],
               color="limegreen", s=70, zorder=5, depthshade=False)
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
               color="darkorange", s=70, zorder=5, depthshade=False)

    # ---- Labels & view ------------------------------------------------
    ax.set_xlabel("x (m)", labelpad=8)
    ax.set_ylabel("y (m)", labelpad=8)
    ax.set_zlabel("z (m)", labelpad=8)
    ax.view_init(elev=view_angles[1], azim=view_angles[0])

    # Equal-ish aspect ratio in 3-D (matplotlib workaround)
    all_pts = np.array([xs, ys, zs])
    mins     = all_pts.min(axis=1)
    maxs     = all_pts.max(axis=1)
    ranges   = maxs - mins
    max_r    = max(ranges.max() / 2.0, 1e-6)
    mid      = (mins + maxs) / 2.0
    ax.set_xlim(mid[0] - max_r, mid[0] + max_r)
    ax.set_ylim(mid[1] - max_r, mid[1] + max_r)
    ax.set_zlim(mid[2] - max_r, mid[2] + max_r)

    # ---- Legend -------------------------------------------------------
    mirror_proxy = mpatches.Patch(
        facecolor=mirror_color, edgecolor="navy",
        alpha=mirror_alpha, label=f"mirrors  ({len(patches)})",
    )
    ax.legend(handles=[
        plt.Line2D([0], [0], color=path_color, lw=2, label="ray path"),
        mirror_proxy,
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="limegreen",  markersize=9, label="start"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=9, label="end"),
    ], loc="upper left", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.show()