"""
Flyby anomaly: Anderson vs BUSFT (Baranowicz)

Ten moduł robi trzy rzeczy:

1) Wczytuje pełne dane z tabeli Andersona (New Astronomy 2007 + δ_in/out z Jouannic 2015)
2) Liczy Δv∞ z empirycznego równania Andersona:
       Δv∞ / v∞ = K * (cos δ_in − cos δ_out),   K = 2 ω_E R_E / c
3) Liczy diagnostykę BUSFT w perygeum i WYPROWADZONĄ z teorii korelację:

   Start z równania:
       γ_B^2 = 1 - 2Φ/c² + v²/c² + B(x,t)

   → definiujemy ΔB = B_out - B_in i przybliżenie:
       Δv_inf ≈ - (c² / (2 v_inf)) * ΔB

   To daje kandydacką Δv_BUSFT, którą porównujemy z Δv_obs i Δv_Anderson.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import math
import pandas as pd

from busft_core._kernel import (
    G,
    M_EARTH,
    R_EARTH,
    c,
    SECONDS_PER_DAY,
)
from busft_core import (
    baranowicz_correction_B,
    baranowicz_gamma,
)


# ---------------------------------------------------------------------------
# Dane wejściowe: pełny case flyby (rozszerzona tabela Andersona)
# ---------------------------------------------------------------------------


@dataclass
class FlybyCase:
    """
    Pojedynczy case flyby – parametry zgodne z Anderson 2007 (Tab. 2)
    + δ_in / δ_out pod równanie Andersona (z Jouannic 2015).

    Pola 1:1 z CSV (data/flyby_cases.csv):

      mission, v_inf_km_s, v_f_km_s, impact_parameter_km, perigee_alt_km,
      eccentricity, deflection_deg, inclination_deg, time_utc, date_utc,
      mass_kg, alpha_deg, delta_deg,
      delta_v_obs_mm_s, delta_v_obs_err_mm_s,
      delta_v_perigee_obs_mm_s, delta_v_perigee_obs_err_mm_s,
      delta_E_J_per_kg, delta_E_err_J_per_kg,
      delta_in_deg, delta_out_deg, notes
    """

    mission: str
    v_inf_km_s: float
    v_f_km_s: float
    impact_parameter_km: float
    perigee_alt_km: float
    eccentricity: float
    deflection_deg: float
    inclination_deg: float
    time_utc: str
    date_utc: str
    mass_kg: float
    alpha_deg: float
    delta_deg: float

    # obserwowane anomalie (z tabeli Andersona)
    delta_v_obs_mm_s: Optional[float] = None          # Δv∞
    delta_v_obs_err_mm_s: Optional[float] = None
    delta_v_perigee_obs_mm_s: Optional[float] = None  # Δv_F (przy perygeum)
    delta_v_perigee_obs_err_mm_s: Optional[float] = None
    delta_E_J_per_kg: Optional[float] = None
    delta_E_err_J_per_kg: Optional[float] = None

    # δ_in / δ_out pod równanie Andersona (Jouannic 2015)
    delta_in_deg: Optional[float] = None
    delta_out_deg: Optional[float] = None

    notes: str = ""

    @property
    def perigee_radius_m(self) -> float:
        return float(R_EARTH + self.perigee_alt_km * 1000.0)

    @property
    def v_inf_m_s(self) -> float:
        return float(self.v_inf_km_s * 1000.0)

    @property
    def v_f_m_s(self) -> float:
        return float(self.v_f_km_s * 1000.0)


# ---------------------------------------------------------------------------
# Diagnostyka BUSFT
# ---------------------------------------------------------------------------


@dataclass
class BusftDiagnostics:
    """
    Diagnostyka BUSFT dla konkretnego flyby.

    Liczone w perygeum (φ_p, r_p), ale z dwoma prędkościami:
    - v_inf  – asymptotyczna prędkość (z tabeli),
    - v_p    – prędkość w perygeum (też z tabeli Andersona).

    B_in  = B(φ_p, v_inf, r_p)
    B_out = B(φ_p, v_p,   r_p)

    ΔB    = B_out - B_in
    Δv∞   = - (c² / (2 v_inf)) * ΔB   [m/s]  → konwersja do [mm/s]
    """

    mission: str

    # potencjał / geometria
    phi_perigee: float  # [J/kg]
    r_perigee: float    # [m]

    # prędkości
    v_inf_m_s: float
    v_perigee_m_s: float

    # korekcja Baranowicza
    B_in: float
    B_out: float
    delta_B: float
    gamma_in: float
    gamma_out: float

    # wyprowadzona anomalia z BUSFT
    delta_v_busft_mm_s: float


# ---------------------------------------------------------------------------
# Loader CSV
# ---------------------------------------------------------------------------


def _parse_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    return float(s)


def load_flyby_cases_csv(path: str | Path) -> List[FlybyCase]:
    """
    Wczytuje pełne dane flyby z CSV (data/flyby_cases.csv) i zwraca listę FlybyCase.
    """
    df = pd.read_csv(path)

    cases: List[FlybyCase] = []
    for _, row in df.iterrows():
        cases.append(
            FlybyCase(
                mission=str(row["mission"]),
                v_inf_km_s=float(row["v_inf_km_s"]),
                v_f_km_s=float(row["v_f_km_s"]),
                impact_parameter_km=float(row["impact_parameter_km"]),
                perigee_alt_km=float(row["perigee_alt_km"]),
                eccentricity=float(row["eccentricity"]),
                deflection_deg=float(row["deflection_deg"]),
                inclination_deg=float(row["inclination_deg"]),
                time_utc=str(row["time_utc"]),
                date_utc=str(row["date_utc"]),
                mass_kg=float(row["mass_kg"]),
                alpha_deg=float(row["alpha_deg"]),
                delta_deg=float(row["delta_deg"]),
                delta_v_obs_mm_s=_parse_optional_float(row.get("delta_v_obs_mm_s")),
                delta_v_obs_err_mm_s=_parse_optional_float(
                    row.get("delta_v_obs_err_mm_s")
                ),
                delta_v_perigee_obs_mm_s=_parse_optional_float(
                    row.get("delta_v_perigee_obs_mm_s")
                ),
                delta_v_perigee_obs_err_mm_s=_parse_optional_float(
                    row.get("delta_v_perigee_obs_err_mm_s")
                ),
                delta_E_J_per_kg=_parse_optional_float(row.get("delta_E_J_per_kg")),
                delta_E_err_J_per_kg=_parse_optional_float(
                    row.get("delta_E_err_J_per_kg")
                ),
                delta_in_deg=_parse_optional_float(row.get("delta_in_deg")),
                delta_out_deg=_parse_optional_float(row.get("delta_out_deg")),
                notes=str(row.get("notes", "")),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Model klasyczny i równanie Andersona
# ---------------------------------------------------------------------------


def compute_classical_delta_v(case: FlybyCase) -> float:
    """
    Idealny Kepler → Δv∞ = 0.

    Utrzymujemy to jawnie, jako baseline pod porównanie.
    """
    return 0.0


# stała Andersona: K = 2 ω_E R_E / c, z ω_E = 2π / dzień
OMEGA_EARTH = 2.0 * math.pi / SECONDS_PER_DAY
K_ANDERSON = 2.0 * OMEGA_EARTH * R_EARTH / c


def compute_anderson_delta_v(case: FlybyCase) -> Optional[float]:
    """
    Empiryczne równanie Andersona:

        Δv∞ / v∞ = K_ANDERSON * (cos δ_in − cos δ_out)

    gdzie:
        K_ANDERSON = 2 ω_E R_E / c
        δ_in, δ_out – geocentryczne szerokości v∞ (w radianach)

    Zwraca:
        Δv∞ w [mm/s] albo None, jeśli brak δ_in/δ_out.
    """
    if case.delta_in_deg is None or case.delta_out_deg is None:
        return None

    v_inf_m_s = case.v_inf_km_s * 1000.0
    delta_in_rad = math.radians(case.delta_in_deg)
    delta_out_rad = math.radians(case.delta_out_deg)

    dv_m_per_s = v_inf_m_s * K_ANDERSON * (
        math.cos(delta_in_rad) - math.cos(delta_out_rad)
    )
    return dv_m_per_s * 1000.0  # [mm/s]


# ---------------------------------------------------------------------------
# BUSFT: diagnostyka i Δv_BUSFT
# ---------------------------------------------------------------------------


def compute_busft_diagnostics(case: FlybyCase) -> BusftDiagnostics:
    """
    Liczy Φ, B_in, B_out, γ_in, γ_out oraz Δv∞_BUSFT dla danego flyby.

    Model:

      - liczymy wszystko w perygeum:
          r_p = R_EARTH + h_perigee
          φ_p = -G M_EARTH / r_p

      - używamy:
          v_inf  = v_inf_km_s  (wejściowe v∞ z tabeli)
          v_p    = v_f_km_s    (prędkość w perygeum z tabeli)

      - B_in  = B(φ_p, v_inf, r_p)
      - B_out = B(φ_p, v_p,   r_p)
      - ΔB    = B_out - B_in

      - Δv∞   = - (c² / (2 v_inf)) * ΔB

    To jest bezpośrednio wyprowadzone z równania Baranowicza, bez strojenia do danych.
    """
    r_perigee_m = R_EARTH + case.perigee_alt_km * 1000.0
    phi_p = -G * M_EARTH / r_perigee_m

    v_inf_m_s = case.v_inf_km_s * 1000.0
    v_p_m_s = case.v_f_km_s * 1000.0

    B_in = baranowicz_correction_B(phi=phi_p, v=v_inf_m_s, r=r_perigee_m)
    B_out = baranowicz_correction_B(phi=phi_p, v=v_p_m_s, r=r_perigee_m)
    delta_B = B_out - B_in

    gamma_in = baranowicz_gamma(phi_p, v_inf_m_s, B_in)
    gamma_out = baranowicz_gamma(phi_p, v_p_m_s, B_out)

    delta_v_busft_m_s = - (c * c) / (2.0 * v_inf_m_s) * delta_B
    delta_v_busft_mm_s = delta_v_busft_m_s * 1000.0

    return BusftDiagnostics(
        mission=case.mission,
        phi_perigee=phi_p,
        r_perigee=r_perigee_m,
        v_inf_m_s=v_inf_m_s,
        v_perigee_m_s=v_p_m_s,
        B_in=B_in,
        B_out=B_out,
        delta_B=delta_B,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        delta_v_busft_mm_s=delta_v_busft_mm_s,
    )
