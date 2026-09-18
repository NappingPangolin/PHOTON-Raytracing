"""
modules/plasma_gradient.py
---------------------------
Plasma density gradient — integrates the ray equation through a user-defined
refractive-index profile.

Currently ships with:
  - "exponential"  : n_e(x) = n0 * exp(-x/L0)   (laser-plasma canonical case)
  - "linear"       : n_e(x) = n0 * (1 - x/L0)
  - "gaussian"     : n_e(x) = n0 * exp(-x²/(2*L0²))

The module is intentionally general: adding a new profile is a one-liner
inside _density().
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optics_sim import Photon
from .base import OpticalModule, ParamSpec

# Physical constants
C_0       = 2.998e8
EPSILON_0 = 8.854187817e-12
M_E       = 9.10938356e-31
E_CHARGE  = 1.602176634e-19
K_B       = 1.380649e-23
EV_TO_K   = 11604.525

DENSITY_PROFILES = ["exponential", "linear", "gaussian"]


class PlasmaGradient(OpticalModule):
    MODULE_TYPE = "plasma_gradient"
    LABEL = "Plasma Gradient"
    COLOR = "#a855f7"
    ICON = "≋"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec("profile", "Density Profile", "select", "exponential",
                      options=DENSITY_PROFILES,
                      tooltip="Shape of the electron density gradient"),
            ParamSpec("L0", "Scale Length L₀", "float", 1e-8, "m",
                      min=1e-12, max=1e-3,
                      tooltip="Gradient scale length"),
            ParamSpec("T_eV", "Temperature", "float", 20000.0, "eV",
                      min=1.0, max=1e7),
            ParamSpec("Z", "Mean Ionisation Z̄", "float", 50.0, "",
                      min=1.0, max=100.0,
                      tooltip="Mean charge state of the plasma ions"),
            ParamSpec("wavelength", "Laser Wavelength", "float", 800e-9, "m",
                      min=200e-9, max=10_000e-9),
            ParamSpec("dt", "Integration Step dt", "float", 1e-11, "s",
                      min=1e-15, max=1e-9,
                      tooltip="Adams-Bashforth time step"),
            ParamSpec("n_steps", "Max Steps", "int", 5000, "",
                      min=100, max=100_000),
        ]

    # ── physics ──────────────────────────────────────────────────────────────

    def _nc(self) -> float:
        lam   = float(self.params["wavelength"])
        omega = 2 * np.pi * C_0 / lam
        return EPSILON_0 * M_E * omega ** 2 / E_CHARGE ** 2

    def _density(self, r: np.ndarray) -> float:
        L0      = float(self.params["L0"])
        nc      = self._nc()
        profile = self.params.get("profile", "exponential")
        x       = r[0]
        if profile == "exponential":
            return nc * np.exp(-x / L0)
        elif profile == "linear":
            return max(0.0, nc * (1 - x / L0))
        elif profile == "gaussian":
            return nc * np.exp(-x ** 2 / (2 * L0 ** 2))
        return nc * np.exp(-x / L0)

    def _ref_index(self, r: np.ndarray) -> complex:
        ne      = self._density(r)
        nc      = self._nc()
        T_eV    = float(self.params["T_eV"])
        Z       = float(self.params["Z"])
        T_K     = T_eV * EV_TO_K
        lam     = float(self.params["wavelength"])
        omega   = 2 * np.pi * C_0 / lam

        omega_p2 = ne * E_CHARGE ** 2 / (M_E * EPSILON_0)
        ni       = ne / max(Z, 1)

        ln_Lambda = np.log(max(1.5e13 * np.sqrt(max(T_eV, 1) ** 3 / max(ne, 1)), 1.1))
        prefactor = np.sqrt(2 / (M_E * (np.pi * K_B * T_K) ** 3))
        coeff     = E_CHARGE ** 4 / (12 * EPSILON_0 ** 2) * ln_Lambda

        nu_ee = prefactor * ne * coeff
        nu_ei = prefactor * ni * Z ** 2 * coeff
        nu    = np.sqrt(nu_ei ** 2 + nu_ee ** 2)

        return np.sqrt(1 - omega_p2 / (omega * (omega - 1j * nu)))

    def _grad_n2_over2(self, r: np.ndarray, dr: float = 1e-11) -> np.ndarray:
        result = np.zeros(3)
        for axis in range(3):
            rp = r.copy(); rp[axis] += dr
            rm = r.copy(); rm[axis] -= dr
            n2p = np.real(self._ref_index(rp) ** 2)
            n2m = np.real(self._ref_index(rm) ** 2)
            result[axis] = (n2p - n2m) / (2 * dr)
        return result / 2

    def _trace(self, ph: Photon) -> tuple[np.ndarray, np.ndarray, float]:
        dt      = float(self.params["dt"])
        n_steps = int(self.params["n_steps"])

        r0 = np.real(ph.r).copy().astype(float)
        k0 = np.real(ph.k).copy().astype(float)
        k0 = k0 / np.linalg.norm(k0)

        r = np.zeros((n_steps, 3))
        v = np.zeros((n_steps, 3))
        r[0] = r0
        v[0] = k0
        optical_path = 0.0

        for i in range(n_steps - 1):
            n_local = float(np.real(self._ref_index(r[i])))
            A = dt * self._grad_n2_over2(r[i])
            B = dt * self._grad_n2_over2(r[i] + dt * v[i] / 2 + dt * A / 8)
            C = dt * self._grad_n2_over2(r[i] + dt * v[i] + dt * B / 2)

            r[i + 1] = r[i] + dt * (v[i] + (A + 2 * B) / 6)
            v[i + 1] = v[i] + (A + 4 * B + C) / 6
            optical_path += np.linalg.norm(r[i + 1] - r[i]) * n_local

        v_end = v[-1] / max(np.linalg.norm(v[-1]), 1e-30)
        return r, v_end, optical_path

    # ── module interface ──────────────────────────────────────────────────────

    def process(self, photons: list[Photon]) -> tuple[list[Photon], dict]:
        out_photons = []
        info_list   = []

        for ph in photons:
            try:
                r_traj, v_end, opd = self._trace(ph)
                # Build output photon at exit position with new direction
                ph_out = ph._copy()
                ph_out.r = r_traj[-1].astype(complex)
                ph_out.k = (v_end * np.linalg.norm(np.real(ph.k))).astype(complex)
                out_photons.append(ph_out)
                info_list.append({
                    "optical_path": float(opd),
                    "exit_r": r_traj[-1].tolist(),
                    "exit_dir": v_end.tolist(),
                    "trajectory": r_traj[::max(1, len(r_traj)//200)].tolist(),  # downsampled
                    "ok": True,
                })
            except Exception as e:
                out_photons.append(ph)
                info_list.append({"ok": False, "error": str(e)})

        return out_photons, {"reflections": info_list}
