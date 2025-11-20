from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# --- ścieżki i importy projektu ---

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from flyby_anomaly import (
    load_flyby_cases_csv,
    compute_classical_delta_v,
    compute_anderson_delta_v,
    compute_busft_diagnostics,
)


def build_flyby_dataframe() -> pd.DataFrame:
    """
    Buduje kompletną tabelę:
    - obserwacje (Anderson 2007)
    - Δv_classical (Kepler)
    - Δv_Anderson (empiryczne równanie)
    - Δv_BUSFT (wyprowadzone z równania Baranowicza)
    - diagnostyka BUSFT w perygeum
    """
    cases = load_flyby_cases_csv(DATA / "flyby_cases.csv")

    rows = []
    for case in cases:
        dv_classical = compute_classical_delta_v(case)
        dv_anderson = compute_anderson_delta_v(case)
        diag = compute_busft_diagnostics(case)

        rows.append(
            {
                "mission": case.mission,
                "perigee_alt_km": case.perigee_alt_km,
                "v_inf_km_s": case.v_inf_km_s,
                "delta_v_obs_mm_s": case.delta_v_obs_mm_s,
                "delta_v_classical_mm_s": dv_classical,
                "delta_v_anderson_mm_s": dv_anderson,
                "delta_v_busft_mm_s": diag.delta_v_busft_mm_s,
                "phi_perigee_J_per_kg": diag.phi_perigee,
                "r_perigee_m": diag.r_perigee,
                "v_inf_m_s": diag.v_inf_m_s,
                "v_perigee_m_s": diag.v_perigee_m_s,
                "B_in": diag.B_in,
                "B_out": diag.B_out,
                "delta_B": diag.delta_B,
                "gamma_in": diag.gamma_in,
                "gamma_out": diag.gamma_out,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    df = build_flyby_dataframe()

    out_csv = ROOT / "flyby_summary.csv"
    df.to_csv(out_csv, index=False)

    print("=== FLYBY SUMMARY (BUSFT vs ANDERSON) ===")
    print()
    print(df.to_string(index=False))
    print()
    print(f"Zapisano tabelę do: {out_csv}")


if __name__ == "__main__":
    main()
