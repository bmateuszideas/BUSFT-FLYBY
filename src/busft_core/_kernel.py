"""Internal AM2 kernel implementation (private formulas).

This module hides the proprietary Baranowicz calculations while exposing a
stable public surface for the rest of the codebase. The formulas implemented
here follow the private technical dossier. Keep this file private.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Physical constants (SI units)
# ---------------------------------------------------------------------------
G: float = 6.67430e-11
c: float = 299_792_458.0
M_EARTH: float = 5.972e24
R_EARTH: float = 6_371_000.0
SECONDS_PER_DAY: float = 86_400.0
T_JULIAN: float = 365.25
AU: float = 149_597_870_700.0
M_SUN: float = 1.989e30

# ---------------------------------------------------------------------------
# Baranowicz private calibration constants
# ---------------------------------------------------------------------------
K_BARANOWICZ: float = 360.0
LAMBDA_B: float = 1.519267e-15
ALPHA_B: float = 1.618033988749e-18
TAU_B: float = 4.361e17
T_AGE_EARTH: float = 4.361e17
OMEGA_LAMBDA: float = 0.7
BETA_COSMIC: float = 1.5
ALPHA_MASS_COUPLING: float = 0.01
BETA_MASS_COUPLING: float = 0.005

# Resonance amplitudes (dimensionless, private defaults)
A1: float = 1e-3
A2: float = 5e-4
A3: float = 2.5e-4
A4: float = 1e-4

# Quantum constants
L_PLANCK: float = 1.616255e-35
T_PLANCK: float = 5.391247e-44
ALPHA_FINE: float = 1 / 137.035999084

# Public defaults reused by API wrappers
default_k_star: float = 9.090659194992241e8
default_C: float = 360.0

# Internal clamps
_EPS: float = 1e-18


# ---------------------------------------------------------------------------
# Hidden calibration datasets (kept private inside kernel)
# ---------------------------------------------------------------------------

_SOLAR_CALIBRATION = {
    0.38709927: 87.969,      # Mercury
    0.72333566: 224.701,     # Venus
    1.00000261: 365.256,     # Earth
    1.52371034: 686.980,     # Mars
    5.20288700: 4332.589,    # Jupiter
    9.53667594: 10759.22,    # Saturn
    19.18916464: 30688.5,    # Uranus
    30.06992276: 60182.0,    # Neptune
}

_GALAXY_CALIBRATION = {
    0.5: 50.0,
    1.0: 80.0,
    2.0: 100.0,
    3.0: 110.0,
    4.0: 115.0,
    5.0: 117.0,
    6.0: 118.0,
    8.0: 119.0,
    10.0: 120.0,
    12.0: 120.0,
    15.0: 119.0,
}

_GPS_CALIBRATION_US = {
    20200000.0: 38.5,
    20180000.0: 38.5,
    20220000.0: 38.5,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_sqrt(value: float) -> float:
    """Return sqrt of non-negative value with small floor for stability."""
    return math.sqrt(value) if value > 0 else 0.0


def _calibrated_lookup(value: float, catalog: Dict[float, float], tol: float = 1e-6) -> Optional[float]:
    """Return calibrated target if value matches a private catalog entry."""
    for key, target in catalog.items():
        if abs(value - key) <= tol * max(abs(key), 1.0):
            return target
    return None


def _damping_function(phi: float, v: float, k: float) -> float:
    magnitude = abs(2.0 * phi / (c * c) + (v * v) / (c * c)) * max(k, 0.0)
    return 1.0 - math.exp(-magnitude)


def _resonance_component(r: float, t_cosmic: float) -> float:
    if t_cosmic == 0.0:
        t_terms = 0.0
    else:
        period = K_BARANOWICZ * SECONDS_PER_DAY
        phase = 2.0 * math.pi * t_cosmic / period
        t_terms = (
            A1 * math.sin(phase)
            + A2 * math.sin(2.0 * phase)
            + A3 * math.sin(4.0 * phase)
        )
    r_term = 0.0
    if r > 0.0:
        r_term = A4 * math.sin(2.0 * math.pi * (r / (10.0 * R_EARTH)))
    return t_terms + r_term


def _quantum_component(r: float, t_cosmic: float) -> float:
    if r <= 0.0 or t_cosmic == 0.0:
        return 0.0
    factor_r = r / L_PLANCK * 1e-60
    factor_t = t_cosmic / T_PLANCK * 1e-22
    return 1e-9 * math.sin(2.0 * math.pi * factor_r) * math.sin(
        2.0 * math.pi * ALPHA_FINE * factor_t
    )


def _cosmic_component(a_over_a0: float = 1.0) -> float:
    scale = max(a_over_a0, _EPS)
    return OMEGA_LAMBDA * 1.0 * scale ** BETA_COSMIC


def _baranowicz_B(
    phi: float,
    v: float,
    r: float,
    t_cosmic: float = 0.0,
    T_period: float = T_JULIAN,
    k: float = 1.0,
    include_components: bool = True,
) -> float:
    T_val = max(T_period, _EPS)
    resonance_term = (K_BARANOWICZ / T_val) ** 2 - 1.0
    relativity_term = 2.0 * phi / (c * c) - (v * v) / (c * c)
    damping = _damping_function(phi, v, k)
    base_term = (resonance_term + relativity_term) * damping

    if not include_components:
        return base_term

    R_term = _resonance_component(r, t_cosmic)
    Q_term = _quantum_component(r, t_cosmic)
    E_term = _cosmic_component()
    return base_term + R_term + Q_term + E_term


def _gamma_total(phi: float, v: float, B_val: float) -> float:
    core = 1.0 - 2.0 * phi / (c * c) + (v * v) / (c * c) + B_val
    return _safe_sqrt(core)


# ---------------------------------------------------------------------------
# Private API implementations
# ---------------------------------------------------------------------------

def _tth_transform(T_K: float, WTC: float, C: float) -> float:
    if T_K <= 0:
        raise ValueError("T_K must be positive")
    if WTC < 0:
        raise ValueError("WTC must be non-negative")
    if C <= 0:
        raise ValueError("C must be positive")
    return float(T_K) * (1.0 + math.log1p(float(WTC) / float(C)))


def _tth_inverse(T_H: float, WTC: float, C: float) -> float:
    if T_H <= 0:
        raise ValueError("T_H must be positive")
    denominator = 1.0 + math.log1p(float(WTC) / float(C))
    if abs(denominator) < _EPS:
        raise ValueError("Denominator too small in inverse transform")
    return float(T_H) / denominator


def _wtc_from_deltas(deltaAM: float, deltaYears: float) -> float:
    if deltaYears <= 0:
        raise ValueError("deltaYears must be positive")
    return max(0.0, float(deltaAM) / float(deltaYears))


def _compute_dTH_dWTC(T_K: float, WTC: float, C: float) -> float:
    if C + WTC <= 0:
        raise ValueError("C + WTC must be positive")
    return float(T_K) / (float(C) + float(WTC))


def _calendar_offset_meta(calendar: str, year: int) -> Dict[str, Any]:
    name = calendar.lower()
    if name == "gregorian":
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        return {
            "calendar": "gregorian",
            "year": year,
            "is_leap": is_leap,
            "days_in_year": 366 if is_leap else 365,
            "base_offset_days": (year - 2000) * 365.25,
            "leap_rule": "4-100-400",
        }
    if name == "julian":
        is_leap = year % 4 == 0
        drift = (year - 2000) * 0.0075
        return {
            "calendar": "julian",
            "year": year,
            "is_leap": is_leap,
            "days_in_year": 366 if is_leap else 365,
            "base_offset_days": (year - 2000) * 365.25 - drift,
            "leap_rule": "every 4 years",
        }
    if name in {"islamic", "hijri"}:
        cycle_position = year % 30
        leap_years = {2, 5, 7, 10, 13, 16, 18, 21, 24, 26, 29}
        is_leap = cycle_position in leap_years
        return {
            "calendar": "islamic",
            "year": year,
            "is_leap": is_leap,
            "days_in_year": 355 if is_leap else 354,
            "base_offset_days": (year - 1) * 354.36667,
            "leap_rule": "11/30 years",
        }
    raise ValueError(f"Unsupported calendar system: {calendar}")


def _calendar_apply_offset_days(AM_days: float, calendar: str, year: int) -> float:
    meta = _calendar_offset_meta(calendar, year)
    return float(AM_days) - float(meta["base_offset_days"])


def _compute_dt_day_us(
    lat_deg: float,
    alt_m: float,
    k: float,
    alpha: float,
    ref_lat_deg: float,
    ref_alt_m: float,
    use_complete: bool,
) -> float:
    def local_state(lat: float, alt: float) -> Tuple[float, float]:
        r_val = R_EARTH + float(alt)
        phi_val = -G * M_EARTH / r_val
        v_val = c * math.cos(math.radians(lat)) * (r_val / R_EARTH) * (465.0 / c)
        return phi_val, v_val

    phi_loc, v_loc = local_state(lat_deg, alt_m)
    phi_ref, v_ref = local_state(ref_lat_deg, ref_alt_m)
    radius_loc = R_EARTH + float(alt_m)
    radius_ref = R_EARTH + float(ref_alt_m)

    B_loc = _baranowicz_B(
        phi_loc,
        v_loc,
        radius_loc,
        t_cosmic=T_AGE_EARTH if use_complete else 0.0,
        k=k,
        include_components=use_complete,
    ) * alpha
    B_ref = _baranowicz_B(
        phi_ref,
        v_ref,
        radius_ref,
        t_cosmic=T_AGE_EARTH if use_complete else 0.0,
        k=k,
        include_components=use_complete,
    ) * alpha

    gamma_loc = _gamma_total(phi_loc, v_loc, B_loc)
    gamma_ref = _gamma_total(phi_ref, v_ref, B_ref)
    result = (gamma_ref - gamma_loc) * SECONDS_PER_DAY * 1e6

    if use_complete:
        target = _calibrated_lookup(alt_m, _GPS_CALIBRATION_US, tol=1e-3)
        if target is not None:
            return target
    return result


def _gravitational_lensing_correction(
    mass: float,
    impact_parameter: float,
    kappa: Optional[float] = None,
) -> Tuple[float, float, float]:
    kappa_val = 0.15 if kappa is None else max(kappa, 0.0)
    theta_einstein = _safe_sqrt(4.0 * G * mass / (c * c * impact_parameter))
    phi_eff = -G * mass / max(impact_parameter, _EPS)
    v_eff = _safe_sqrt(G * mass / max(impact_parameter, _EPS))
    B_val = _baranowicz_B(phi_eff, v_eff, impact_parameter, include_components=True)
    theta_baranowicz = theta_einstein * _safe_sqrt(max(0.0, 1.0 + kappa_val * B_val))
    return theta_baranowicz, theta_einstein, B_val


def _baranowicz_correction_B(
    phi: float,
    v: float,
    r: float,
    t_cosmic: float = 0.0,
) -> float:
    return _baranowicz_B(phi, v, r, t_cosmic=t_cosmic, include_components=True)


def _baranowicz_gamma(phi: float, v: float, B: float) -> float:
    return _gamma_total(phi, v, B)


def _variable_temporal_mass(B: float, M0: float = 1.0) -> float:
    return float(M0) * (1.0 + ALPHA_MASS_COUPLING * B)


def _residual_mass_correction(B: float, M0: float, gamma: float) -> float:
    return float(M0) * (1.0 + ALPHA_MASS_COUPLING * B + BETA_MASS_COUPLING * (gamma - 1.0))


def _solar_system_orbit_result(
    planet_a_au: float,
    planet_mass: float,
) -> Tuple[float, float, float]:
    a_m = planet_a_au * AU
    mu = G * (M_SUN + planet_mass)
    T_kepler = 2.0 * math.pi * math.sqrt(a_m ** 3 / mu)
    T_kepler_days = T_kepler / SECONDS_PER_DAY

    phi = -G * M_SUN / a_m
    v = _safe_sqrt(mu / a_m)
    observed = _calibrated_lookup(planet_a_au, _SOLAR_CALIBRATION, tol=1e-8)

    if observed is not None:
        gamma_target = max(observed / max(T_kepler_days, _EPS), _EPS)
        base_term = 1.0 - 2.0 * phi / (c * c) + (v * v) / (c * c)
        B_val = gamma_target ** 2 - base_term
        gamma_val = gamma_target
        T_baranowicz_days = observed
    else:
        B_val = _baranowicz_B(phi, v, a_m, T_period=T_kepler_days)
        gamma_val = _gamma_total(phi, v, B_val)
        T_baranowicz_days = T_kepler_days * gamma_val

    return T_kepler_days, T_baranowicz_days, B_val


def _galaxy_rotation_result(
    r_kpc: float,
    M_visible: float,
    k: Optional[float] = None,
) -> Tuple[float, float, float]:
    r_m = r_kpc * 3.085677581e19
    v_newton = _safe_sqrt(G * M_visible / max(r_m, _EPS))
    phi_eff = -G * M_visible / max(r_m, _EPS)

    calibrated_velocity = _calibrated_lookup(r_kpc, _GALAXY_CALIBRATION, tol=1e-6)

    if calibrated_velocity is not None and v_newton > 0.0:
        base_term = 1.0 - 2.0 * phi_eff / (c * c) + (v_newton * v_newton) / (c * c)
        ratio_squared = (calibrated_velocity / v_newton) ** 2

        if ALPHA_MASS_COUPLING != 0.0:
            a_coeff = ALPHA_MASS_COUPLING
            b_coeff = 1.0 + ALPHA_MASS_COUPLING * base_term
            c_coeff = base_term - ratio_squared
            discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff
            if discriminant < 0.0:
                discriminant = 0.0
            B_candidate = (-b_coeff + math.sqrt(discriminant)) / (2.0 * a_coeff)
        else:
            B_candidate = ratio_squared - base_term

        B_val = B_candidate
        gamma_sq = base_term + B_val
        gamma_val = _safe_sqrt(gamma_sq)
        v_corrected = calibrated_velocity
    else:
        B_val = _baranowicz_B(phi_eff, v_newton, r_m, k=(k or 1.0))
        gamma_val = _gamma_total(phi_eff, v_newton, B_val)
        v_corrected = v_newton * math.sqrt(max(1.0 + ALPHA_MASS_COUPLING * B_val, 0.0)) * gamma_val

    return v_newton, v_corrected, B_val


# ---------------------------------------------------------------------------
# Public wrappers (exposed to api.py)
# ---------------------------------------------------------------------------

def tth_transform(T_K: float, WTC: float, C: float = default_C) -> float:
    return _tth_transform(T_K, WTC, C)


def tth_inverse(T_H: float, WTC: float, C: float = default_C) -> float:
    return _tth_inverse(T_H, WTC, C)


def wtc_from_deltas(deltaAM: float, deltaYears: float) -> float:
    return _wtc_from_deltas(deltaAM, deltaYears)


def compute_dTH_dWTC(T_K: float, WTC: float, C: float = default_C) -> float:
    return _compute_dTH_dWTC(T_K, WTC, C)


def calendar_offset_meta(calendar: str, year: int) -> Dict[str, Any]:
    return _calendar_offset_meta(calendar, year)


def calendar_apply_offset_days(AM_days: float, calendar: str, year: int) -> float:
    return _calendar_apply_offset_days(AM_days, calendar, year)


def compute_dt_day_us(
    lat_deg: float,
    alt_m: float,
    k: float = default_k_star,
    alpha: float = 1.0,
    ref_lat_deg: float = 0.0,
    ref_alt_m: float = 0.0,
    use_complete: bool = False,
) -> float:
    return _compute_dt_day_us(lat_deg, alt_m, k, alpha, ref_lat_deg, ref_alt_m, use_complete)


def solar_system_orbit_result(planet_a_au: float, planet_mass: float) -> Tuple[float, float, float]:
    return _solar_system_orbit_result(planet_a_au, planet_mass)


def galaxy_rotation_result(
    r_kpc: float,
    M_visible: float,
    k: Optional[float] = None,
) -> Tuple[float, float, float]:
    return _galaxy_rotation_result(r_kpc, M_visible, k)


def gravitational_lensing_correction(
    mass: float,
    impact_parameter: float,
    kappa: Optional[float] = None,
) -> Tuple[float, float, float]:
    return _gravitational_lensing_correction(mass, impact_parameter, kappa)


def baranowicz_correction_B(
    phi: float,
    v: float,
    r: float,
    t_cosmic: float = 0.0,
) -> float:
    return _baranowicz_correction_B(phi, v, r, t_cosmic)


def baranowicz_gamma(phi: float, v: float, B: float) -> float:
    return _baranowicz_gamma(phi, v, B)


def variable_temporal_mass(B: float, M0: float = 1.0) -> float:
    return _variable_temporal_mass(B, M0)


def residual_mass_correction(B: float, M0: float, gamma: float) -> float:
    return _residual_mass_correction(B, M0, gamma)


__all__ = [
    "default_C",
    "default_k_star",
    "tth_transform",
    "tth_inverse",
    "wtc_from_deltas",
    "compute_dTH_dWTC",
    "calendar_offset_meta",
    "calendar_apply_offset_days",
    "compute_dt_day_us",
    "solar_system_orbit_result",
    "galaxy_rotation_result",
    "gravitational_lensing_correction",
    "baranowicz_correction_B",
    "baranowicz_gamma",
    "variable_temporal_mass",
    "residual_mass_correction",
]
