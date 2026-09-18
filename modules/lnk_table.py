"""
modules/lnk_table.py
---------------------
Utility for wavelength-dependent refractive index tables.

File format (space/tab separated, two-column blocks):
  wl   n          <- header (ignored)
  0.38  1.45       <- wavelength in µm, then n values
  ...

  wl   k          <- second block header
  0.38  0.0        <- same wavelengths (or different), then k values
  ...

The two blocks can appear in any order and may have completely different
wavelength grids — each is interpolated on its own axis and then evaluated
at the union of all wavelengths.  A file with only an n block (no k) is
accepted; k defaults to 0 for all wavelengths.  A file with only a k block
has n defaulting to 1.

Public API
----------
LnkTable.from_text(text: str) -> LnkTable
    Parse raw text (the content of an lnk file).

LnkTable.from_rows(rows: list[list[float]]) -> LnkTable
    Reconstruct from the serialised form stored in module params
    (a list of [wl_um, n, k] triples).

lnk.get_nk(wavelength_m: float) -> tuple[float, float]
    Return (n, k) at the given wavelength via linear interpolation.
    Clamps to the table's range if the photon wavelength is outside it.

lnk.to_rows() -> list[list[float]]
    Serialise to a JSON-safe list of [wl_um, n, k] triples for storage
    in module params (and therefore in saved scenes).
"""

from __future__ import annotations
import re
import numpy as np


class LnkTable:
    """Wavelength-interpolated n/k table."""

    def __init__(self, wl_um: np.ndarray, n: np.ndarray, k: np.ndarray):
        """
        Parameters
        ----------
        wl_um : wavelengths in micrometres, strictly increasing
        n     : real part of refractive index at each wavelength
        k     : extinction coefficient at each wavelength
        """
        order = np.argsort(wl_um)
        self._wl = wl_um[order].astype(float)
        self._n  = n[order].astype(float)
        self._k  = k[order].astype(float)

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_text(cls, text: str) -> "LnkTable":
        """
        Parse an lnk file (two-block format as used in the project).

        Handles both tab- and space-delimited files.  Lines beginning with
        '#' or that contain non-numeric text (headers) are skipped.
        """
        # Split file into the two blocks by detecting the header lines
        # ("wl  n"  and  "wl  k").  Everything else is data.
        n_block: dict[float, float] = {}
        k_block: dict[float, float] = {}
        current: dict[float, float] | None = None

        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # Detect header rows
            low = line.lower()
            if re.match(r'^wl\s+n\b', low):
                current = n_block
                continue
            if re.match(r'^wl\s+k\b', low):
                current = k_block
                continue
            # Try parsing as two numbers
            parts = re.split(r'[\s,;]+', line)
            if len(parts) >= 2:
                try:
                    wl = float(parts[0])
                    v  = float(parts[1])
                    if current is not None:
                        current[wl] = v
                except ValueError:
                    pass  # skip non-numeric lines

        if not n_block and not k_block:
            raise ValueError("No valid data found in lnk file.")

        # Build each axis independently so misaligned grids are handled correctly.
        # n and k are interpolated on their own wavelength axes, then evaluated
        # at the union of all wavelengths.  Missing blocks default to n=1, k=0.
        n_wl = np.array(sorted(n_block), dtype=float)
        n_v  = np.array([n_block[w] for w in n_wl], dtype=float)
        k_wl = np.array(sorted(k_block), dtype=float)
        k_v  = np.array([k_block[w] for w in k_wl], dtype=float)

        all_wl = np.array(sorted(set(list(n_wl) + list(k_wl))), dtype=float)

        if len(n_wl) >= 2:
            n_arr = np.interp(all_wl, n_wl, n_v)
        elif len(n_wl) == 1:
            n_arr = np.full(len(all_wl), n_v[0])
        else:
            n_arr = np.ones(len(all_wl))   # no n block → assume n=1

        if len(k_wl) >= 2:
            k_arr = np.interp(all_wl, k_wl, k_v)
        elif len(k_wl) == 1:
            k_arr = np.full(len(all_wl), k_v[0])
        else:
            k_arr = np.zeros(len(all_wl))  # no k block → assume k=0

        return cls(all_wl, n_arr, k_arr)

    @classmethod
    def from_rows(cls, rows: list[list[float]]) -> "LnkTable":
        """
        Reconstruct from the serialised [wl_um, n, k] row format stored in
        module params / saved scenes.
        """
        if not rows:
            raise ValueError("Empty lnk_data rows.")
        arr = np.array(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"lnk_data must be Nx3, got shape {arr.shape}")
        return cls(arr[:, 0], arr[:, 1], arr[:, 2])

    # ── interpolation ─────────────────────────────────────────────────────────

    def get_nk(self, wavelength_m: float) -> tuple[float, float]:
        """
        Return (n, k) at *wavelength_m* (SI, metres) by linear interpolation.

        Clamps to the table's wavelength range so edge photons don't cause
        index-out-of-bounds errors — a warning could be logged, but keeping
        this dependency-free for now.
        """
        wl_um = wavelength_m * 1e6   # convert m → µm
        n = float(np.interp(wl_um, self._wl, self._n))
        k = float(np.interp(wl_um, self._wl, self._k))
        return n, k

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_rows(self) -> list[list[float]]:
        """
        Serialise to a JSON-safe list of [wl_um, n, k] triples.

        This is the format stored as ``lnk_data`` in module params so that
        scenes can be saved and reloaded without re-uploading the file.
        """
        return [
            [round(float(w), 8), round(float(n), 8), round(float(k), 8)]
            for w, n, k in zip(self._wl, self._n, self._k)
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def wl_range_um(self) -> tuple[float, float]:
        return float(self._wl[0]), float(self._wl[-1])

    def __len__(self) -> int:
        return len(self._wl)

    def __repr__(self) -> str:
        lo, hi = self.wl_range_um
        return f"LnkTable({len(self)} points, {lo:.3f}–{hi:.3f} µm)"