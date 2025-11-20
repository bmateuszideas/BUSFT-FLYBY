from __future__ import annotations

"""
Szybki smoke-test flyby anomaly:

- wczytanie cases z CSV
- Anderson vs BUSFT vs obserwacje
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from flyby_anomaly import (
    load_flyby_cases_csv,
    compute_classical_delta_v,
    compute_anderson_delta_v,
    compute_busft_diagnostics,
)


CASES_PATH = ROOT / "data" / "flyby_cases.csv"


def main() -> None:
    cases = load_flyby_cases_csv(CASES_PATH)

    for case in cases:
        dv_classical = compute_classical_delta_v(case)
        dv_anderson = compute_anderson_delta_v(case)
        diag = compute_busft_diagnostics(case)

        print(f"=== {case.mission} ===")
        print("perigee_alt_km:             ", case.perigee_alt_km)
        print("v_inf_km_s:                 ", case.v_inf_km_s)
        print("delta_v_obs_mm_s:           ", case.delta_v_obs_mm_s)
        print("delta_v_classical_mm_s:     ", dv_classical)
        print("delta_v_anderson_mm_s:      ", dv_anderson)
        print("delta_v_busft_mm_s:         ", diag.delta_v_busft_mm_s)
        print("phi_perigee [J/kg]:         ", diag.phi_perigee)
        print("r_perigee [m]:              ", diag.r_perigee)
        print("v_inf [m/s]:                ", diag.v_inf_m_s)
        print("v_perigee [m/s]:            ", diag.v_perigee_m_s)
        print("B_in:                       ", diag.B_in)
        print("B_out:                      ", diag.B_out)
        print("delta_B:                    ", diag.delta_B)
        print("gamma_in:                   ", diag.gamma_in)
        print("gamma_out:                  ", diag.gamma_out)
        print()


if __name__ == "__main__":
    main()
