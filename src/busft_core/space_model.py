"""
Physics helpers: mass profiles and potentials used by AM2 kernel
- Exponential disk
- Hernquist bulge
- NFW halo
- Mass-enclosed, potential, circular velocity
- Einstein radius helper

This module is internal and intended for use by `_kernel.py` and calibration scripts.
"""
from __future__ import annotations
import math
from typing import Tuple

# Physical constants
G = 6.67430e-11
M_SUN = 1.989e30
PC = 3.085677581e16
KPC = 3.085677581e19


def nfw_mass_enclosed(r_m: float, rho0: float, rs_m: float) -> float:
    """Mass enclosed for NFW profile up to radius r (m)."""
    x = r_m / rs_m
    if x <= 0:
        return 0.0
    # M(r) = 4πρ0 rs^3 [ ln(1+x) - x/(1+x) ]
    return 4.0 * math.pi * rho0 * (rs_m**3) * (math.log(1.0 + x) - x / (1.0 + x))


def nfw_potential(r_m: float, rho0: float, rs_m: float) -> float:
    """Approximate potential for NFW by integrating mass enclosed / r."""
    if r_m <= 0:
        return 0.0
    m_enc = nfw_mass_enclosed(r_m, rho0, rs_m)
    return -G * m_enc / r_m


def exponential_disk_mass_enclosed(r_kpc: float, M_disk_kg: float, R_d_kpc: float) -> float:
    """Approximate enclosed mass for exponential disk.
    Uses M_enc(R) = M_disk * (1 - e^{-R/Rd} (1 + R/Rd)).
    r_kpc: radius in kpc
    M_disk_kg: total disk mass in kg
    R_d_kpc: disk scale length in kpc
    """
    if R_d_kpc <= 0 or r_kpc <= 0:
        return 0.0
    val = 1.0 - math.exp(-r_kpc / R_d_kpc) * (1.0 + r_kpc / R_d_kpc)
    return M_disk_kg * val


def hernquist_mass_enclosed(r_kpc: float, M_bulge_kg: float, a_kpc: float) -> float:
    """Hernquist enclosed mass: M(r) = M_bulge * r^2 / (r + a)^2 (r and a in kpc)"""
    if r_kpc <= 0:
        return 0.0
    return M_bulge_kg * (r_kpc * r_kpc) / ((r_kpc + a_kpc) * (r_kpc + a_kpc))


def total_mass_enclosed_m33(r_kpc: float, M_visible_kg: float, R_d_kpc: float = 3.0, bulge_frac: float = 0.2, a_kpc: float = 0.5) -> float:
    """Return total enclosed mass (kg) for M33-like model.
    M_visible_kg: total visible mass allocated to disk+bulge.
    """
    M_vis_solar = M_visible_kg / M_SUN
    M_bulge_kg = M_visible_kg * bulge_frac
    M_disk_kg = max(M_visible_kg - M_bulge_kg, 0.0)
    M_enc_disk = exponential_disk_mass_enclosed(r_kpc, M_disk_kg, R_d_kpc)
    M_enc_bulge = hernquist_mass_enclosed(r_kpc, M_bulge_kg, a_kpc)
    return M_enc_disk + M_enc_bulge


def circular_velocity_from_mass(r_m: float, M_enc_kg: float) -> float:
    """Return circular velocity (m/s) from enclosed mass M_enc at radius r."""
    if r_m <= 0:
        return 0.0
    return math.sqrt(G * M_enc_kg / r_m)


def einstein_angle(mass_kg: float, D_d_m: float, D_s_m: float, D_ds_m: float) -> float:
    """Classic Einstein radius (radians) for a point mass lens approximation."""
    return math.sqrt(4.0 * G * mass_kg / (299792458.0**2) * (D_ds_m / (D_d_m * D_s_m)))


# Quick import test function
def _self_test():
    r_kpc = 8.0
    M_vis = 5e10 * M_SUN
    M_enc = total_mass_enclosed_m33(r_kpc, M_vis)
    v = circular_velocity_from_mass(r_kpc * KPC, M_enc)
    print('M_enc (kg)=', M_enc, 'v (m/s)=', v)

if __name__ == '__main__':
    _self_test()
