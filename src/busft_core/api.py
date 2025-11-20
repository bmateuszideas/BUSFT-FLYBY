"""
AM2 Core Public API

This module provides the public interface for AM2 core functionality.
All formulas are implemented in the internal _kernel module.

This is a thin wrapper layer that delegates to internal implementations
without exposing the proprietary formulas directly.

Author: Mateusz Baranowicz
License: Proprietary - See LICENSE, NO_COMMERCIAL.md
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

from ._kernel import (
    tth_transform as _tth_transform,
    tth_inverse as _tth_inverse,
    wtc_from_deltas as _wtc_from_deltas,
    compute_dTH_dWTC as _compute_dTH_dWTC,
    calendar_offset_meta as _calendar_offset_meta,
    calendar_apply_offset_days as _calendar_apply_offset_days,
    compute_dt_day_us as _compute_dt_day_us,
    solar_system_orbit_result as _solar_system_orbit_result,
    galaxy_rotation_result as _galaxy_rotation_result,
    gravitational_lensing_correction as _gravitational_lensing_correction,
    baranowicz_correction_B as _baranowicz_correction_B,
    baranowicz_gamma as _baranowicz_gamma,
    variable_temporal_mass as _variable_temporal_mass,
    residual_mass_correction as _residual_mass_correction,
    default_C,
    default_k_star,
)


__test__ = False  # Prevent pytest from collecting this module


def tth_transform(T_K: float, WTC: float, C: float = default_C) -> float:
    """
    Transform kinematic time scale to historical time scale.
    
    Args:
        T_K: Kinematic time scale (must be > 0)
        WTC: Workload/Throughput Coefficient (dimensionless, >= 0)
        C: Conventional constant (default 360)
    
    Returns:
        T_H: Historical time scale
    
    Example:
        >>> tth_transform(1.0, 360.0, 360.0)
        1.693147...
    """
    return _tth_transform(T_K, WTC, C)


def tth_inverse(T_H: float, WTC: float, C: float = default_C) -> float:
    """
    Inverse transform: recover kinematic time from historical time.
    
    Args:
        T_H: Historical time scale (must be > 0)
        WTC: Workload/Throughput Coefficient (dimensionless, >= 0)
        C: Conventional constant (default 360)
    
    Returns:
        T_K: Kinematic time scale
    
    Example:
        >>> tth_inverse(1.693147, 360.0, 360.0)
        1.0...
    """
    return _tth_inverse(T_H, WTC, C)


def wtc_from_deltas(deltaAM: float, deltaYears: float) -> float:
    """
    Reconstruct WTC from changes in AM units and years.
    
    Args:
        deltaAM: Change in AM units (dimensionless)
        deltaYears: Change in years (must be positive)
    
    Returns:
        WTC: Estimated Workload/Throughput Coefficient
    
    Example:
        >>> wtc_from_deltas(100.0, 1.0)
        100.0
    """
    return _wtc_from_deltas(deltaAM, deltaYears)


def compute_dTH_dWTC(T_K: float, WTC: float, C: float = default_C) -> float:
    """
    Compute derivative of T_H with respect to WTC.
    
    Args:
        T_K: Kinematic time scale
        WTC: Workload/Throughput Coefficient
        C: Conventional constant (default 360)
    
    Returns:
        Derivative dT_H/dWTC
    
    Example:
        >>> compute_dTH_dWTC(1.0, 0.0, 360.0)
        0.00277...
    """
    return _compute_dTH_dWTC(T_K, WTC, C)


def calendar_offset_meta(calendar: str, year: int) -> Dict[str, Any]:
    """
    Get calendar offset metadata for a given calendar system and year.
    
    Supported calendars: 'gregorian', 'julian', 'islamic'/'hijri'
    
    Args:
        calendar: Calendar system name
        year: Year in the specified calendar
    
    Returns:
        Dictionary with offset metadata including:
        - calendar: Calendar system name
        - year: Year value
        - is_leap: Whether it's a leap year
        - days_in_year: Number of days in the year
        - base_offset_days: Days offset from AM2 epoch
        - leap_rule: Description of leap year rule
    
    Example:
        >>> meta = calendar_offset_meta('gregorian', 2024)
        >>> meta['is_leap']
        True
        >>> meta['days_in_year']
        366
    """
    return _calendar_offset_meta(calendar, year)


def calendar_apply_offset_days(AM_days: float, calendar: str, year: int) -> float:
    """
    Apply calendar-specific offset to AM days.
    
    Args:
        AM_days: Time in AM days
        calendar: Calendar system name
        year: Year in the specified calendar
    
    Returns:
        Corrected AM days adjusted for calendar offset
    
    Example:
        >>> calendar_apply_offset_days(1000.0, 'gregorian', 2024)
        1000.0...
    """
    return _calendar_apply_offset_days(AM_days, calendar, year)


def compute_dt_day_us(
    lat_deg: float,
    alt_m: float,
    k: float = default_k_star,
    alpha: float = 1.0,
    ref_lat_deg: float = 0.0,
    ref_alt_m: float = 0.0,
    use_complete: bool = False
) -> float:
    """
    Compute local time dilation effect in microseconds per day.
    
    Args:
        lat_deg: Latitude in degrees
        alt_m: Altitude in meters
        k: Calibration parameter (default from model)
        alpha: Scaling parameter (default 1.0)
        ref_lat_deg: Reference latitude (default 0.0)
        ref_alt_m: Reference altitude (default 0.0)
        use_complete: Use complete Baranowicz theory with all components
    
    Returns:
        Time difference in microseconds per day vs reference
    
    Example:
        >>> dt = compute_dt_day_us(45.0, 1000.0)
        >>> isinstance(dt, float)
        True
    """
    return _compute_dt_day_us(lat_deg, alt_m, k, alpha, ref_lat_deg, ref_alt_m, use_complete)




# Public test helpers (not named test_*)
def solar_system_orbit_result(planet_a_au: float, planet_mass: float) -> Tuple[float, float, float]:
    """
    Zwraca (T_kepler, T_baranowicz, B_correction) dla danej planety.
    """
    return _solar_system_orbit_result(planet_a_au, planet_mass)

def galaxy_rotation_result(r_kpc: float, M_visible: float, k: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Zwraca (v_newton, v_baranowicz, B_correction) dla danej galaktyki.
    """
    if k is None:
        return _galaxy_rotation_result(r_kpc, M_visible)
    return _galaxy_rotation_result(r_kpc, M_visible, k)


def gravitational_lensing_correction(mass: float, impact_parameter: float, kappa: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Calculate gravitational lensing without dark matter.
    
    Args:
        mass: Lens mass in kg
        impact_parameter: Impact parameter in meters
    
    Returns:
        Tuple of (theta_Baranowicz, theta_Einstein, B_lens)
    
    Note:
        Shows how Baranowicz theory explains strong lensing
        without requiring dark matter.
    """
    if kappa is None:
        return _gravitational_lensing_correction(mass, impact_parameter)
    return _gravitational_lensing_correction(mass, impact_parameter, kappa=kappa)


def baranowicz_correction_B(phi: float, v: float, r: float, t_cosmic: float = 0) -> float:
    """
    Calculate the complete Baranowicz correction function B(x,t).
    
    Args:
        phi: Gravitational potential (J/kg)
        v: Velocity (m/s)
        r: Spatial coordinate (m)
        t_cosmic: Cosmic time (s)
    
    Returns:
        B(x,t) correction value
    
    Note:
        This is the core correction function that replaces dark matter/energy.
        Implementation details are proprietary.
    """
    return _baranowicz_correction_B(phi, v, r, t_cosmic)


def baranowicz_gamma(phi: float, v: float, B: float) -> float:
    """
    Calculate the Baranowicz gamma factor.
    
    Args:
        phi: Gravitational potential (J/kg)
        v: Velocity (m/s)
        B: Baranowicz correction B(x,t)
    
    Returns:
        γ(x,t) = √[1 - 2Φ/c² + v²/c² + B(x,t)]
    """
    return _baranowicz_gamma(phi, v, B)


def variable_temporal_mass(B: float, M0: float = 1.0) -> float:
    """
    Calculate variable temporal mass M(t).
    
    Args:
        B: Baranowicz correction B(x,t)
        M0: Rest mass (kg)
    
    Returns:
        M(t) = M₀·[1 + α·B(x,t)]
    
    Note:
        This is a key discovery of the Baranowicz theory - mass varies with time.
    """
    return _variable_temporal_mass(B, M0)


def residual_mass_correction(B: float, M0: float, gamma: float) -> float:
    """
    Calculate residual mass correction.
    
    Args:
        B: Baranowicz correction B(x,t)
        M0: Rest mass (kg)
        gamma: Gamma factor
    
    Returns:
        M_residual = M₀ · [1 + α·B(x,t) + β·(γ-1)]
    
    Note:
        This explains additional mass effects observed in gravitational lensing.
    """
    return _residual_mass_correction(B, M0, gamma)


# Export public API
__all__ = [
    'tth_transform',
    'tth_inverse',
    'wtc_from_deltas',
    'compute_dTH_dWTC',
    'calendar_offset_meta',
    'calendar_apply_offset_days',
    'compute_dt_day_us',
    'solar_system_orbit_result',
    'galaxy_rotation_result',
    'gravitational_lensing_correction',
    'baranowicz_correction_B',
    'baranowicz_gamma',
    'variable_temporal_mass',
    'residual_mass_correction',
]
