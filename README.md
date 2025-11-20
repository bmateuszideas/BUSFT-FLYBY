# BUSFT-FLYBY – Baranowicz Unified Spacetime Field Theory & Earth Flyby Anomaly

This repository packages the **Baranowicz Unified Spacetime Field Theory (BUSFT / AM2 core)** and applies it to the **Earth flyby anomaly** as a numerical experiment.

The goal is not to *prove* the theory right, but to:

1. Implement the BUSFT / AM2 equations in clean, testable Python code.  
2. Use real flyby data (Anderson et al. 2007 + Jouannic 2015) as a falsification test.  
3. Compare three levels of prediction for Earth flybys:  
   - classical two-body gravity (Kepler) → `Δv_classical`,  
   - Anderson’s empirical formula → `Δv_Anderson`,  
   - BUSFT-based prediction derived directly from the Baranowicz time-dilation equation → `Δv_BUSFT`.

The repository is structured as a **portfolio-ready research project**: clean separation between theory and application, real spacecraft data, and an explicit, quantitative comparison of competing models.

---

## Project structure

```text
BUSFT-FLYBY/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ data/
│  └─ flyby_cases.csv          # Earth flyby cases (Galileo, NEAR, Rosetta, Cassini, Messenger...)
├─ notebooks/
│  ├─ 01_busft_theory_overview.ipynb
│  └─ 02_flyby_anomaly_with_busft.ipynb
├─ src/
│  ├─ busft_core/
│  │  ├─ __init__.py
│  │  ├─ _kernel.py            # internal AM2 kernel (private formulas)
│  │  ├─ api.py                # public API wrapper around _kernel
│  │  └─ space_model.py        # mass profiles / potentials helpers
│  └─ flyby_anomaly.py         # Earth flyby anomaly application
├─ test_flyby.py               # quick smoke-test script
├─ make_flyby_summary.py       # builds flyby_summary.csv table
└─ flyby_summary.csv           # generated comparison table (created by make_flyby_summary.py)

# web/

Optional folder for future HTML/JS demos (e.g. BUSFT flyby visualizer).

```

---

## 1. BUSFT / AM2 core

The core physics lives in `src/busft_core/_kernel.py`. This module implements:

- physical constants (`G`, `c`, `M_EARTH`, `R_EARTH`, etc.),  
- Baranowicz-specific parameters (`K_BARANOWICZ`, `LAMBDA_B`, etc.),  
- the correction field \( B(x,t) \),  
- the Baranowicz gamma factor \( \gamma_B \),  
- time-dilation helper functions,  
- orbit / rotation / lensing style helpers.

Conceptually, the Baranowicz gamma factor has the form:

\[
\gamma_B^2 = 1 - \frac{2\Phi}{c^2} + \frac{v^2}{c^2} + B(x,t),
\]

where \( \Phi \) is the gravitational potential, \( v \) is the local velocity, and \( B(x,t) \) is the BUSFT correction term.

From this, the kernel builds objects such as:

- “360-day resonance” corrections for the Earth’s orbit,  
- solar-system orbital corrections,  
- galaxy rotation curve helpers,  
- gravitational lensing corrections,  
- GPS-style local time-dilation estimates,  
- variable temporal mass and residual mass corrections.

All formulas in `_kernel.py` follow a private technical dossier and are treated here as a **black box implementation** of BUSFT / AM2. The rest of the repository interacts with it only through the public API.

The public interface is exposed by `src/busft_core/api.py`, which wraps the kernel functions:

- `tth_transform`, `tth_inverse`, `wtc_from_deltas`,  
- `calendar_offset_meta`, `calendar_apply_offset_days`,  
- `compute_dt_day_us`,  
- `solar_system_orbit_result`, `galaxy_rotation_result`,  
- `gravitational_lensing_correction`,  
- `baranowicz_correction_B`, `baranowicz_gamma`,  
- `variable_temporal_mass`, `residual_mass_correction`.

`space_model.py` contains helper routines for mass profiles and potentials (NFW halo, exponential disk, Hernquist bulge, etc.), primarily used for galaxy-scale tests.

---

## 2. Earth flyby anomaly layer

The flyby anomaly application lives in `src/flyby_anomaly.py`.

### 2.1 Data model

Flyby cases are stored in `data/flyby_cases.csv`, with columns closely matching Anderson (2007) and \( \delta_\text{in} / \delta_\text{out} \) angles from Jouannic (2015). In Python they are represented as:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FlybyCase:
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
    delta_v_obs_mm_s: Optional[float]
    delta_v_obs_err_mm_s: Optional[float]
    delta_v_perigee_obs_mm_s: Optional[float]
    delta_v_perigee_obs_err_mm_s: Optional[float]
    delta_E_J_per_kg: Optional[float]
    delta_E_err_J_per_kg: Optional[float]
    delta_in_deg: Optional[float]
    delta_out_deg: Optional[float]
    notes: str
```

The helper `load_flyby_cases_csv(...)` reads the CSV file into a list of `FlybyCase` objects, handling missing values and NaNs.

---

### 2.2 Classical model

Under ideal Keplerian two-body dynamics, the asymptotic velocity is conserved:

\[
\Delta v_\infty^\text{classical} = 0.
\]

In code:

```python
def compute_classical_delta_v(case: FlybyCase) -> float:
    return 0.0
```

This serves as a baseline.

---

### 2.3 Anderson’s empirical formula

We implement the Anderson et al. empirical relation for the anomalous velocity shift:

\[
\frac{\Delta v_\infty}{v_\infty}
= K_\text{Anderson}\, \big(\cos \delta_\text{in} - \cos \delta_\text{out}\big),
\]

with:

\[
K_\text{Anderson} = \frac{2 \, \Omega_\oplus \, R_\oplus}{c},
\]

where \( \Omega_\oplus \) is the Earth’s rotation rate and \( R_\oplus \) is the Earth’s radius.

The implementation returns \( \Delta v_\infty^\text{Anderson} \) in mm/s when both \( \delta_\text{in} \) and \( \delta_\text{out} \) are available.

---

### 2.4 BUSFT-based prediction

For each flyby we compute a BUSFT-based diagnostic at perigee:

- perigee radius \( r_p = R_\oplus + h_\text{perigee} \),  
- gravitational potential at perigee \( \phi_p = - G M_\oplus / r_p \),  
- inbound/outbound velocities \( v_\infty \), \( v_p \) from the dataset,  
- BUSFT correction field:
  - \( B_\text{in} = B(\phi_p, v_\infty, r_p) \),  
  - \( B_\text{out} = B(\phi_p, v_p, r_p) \),  
- difference \( \Delta B = B_\text{out} - B_\text{in} \).

Using the BUSFT prescription implemented here, the induced asymptotic velocity shift is:

\[
\Delta v_\infty^\text{BUSFT} \approx - \frac{c^2}{2 v_\infty} \, \Delta B.
\]

The resulting diagnostics are wrapped in a `BusftDiagnostics` dataclass, containing:

- `phi_perigee`, `r_perigee`,  
- `v_inf_m_s`, `v_perigee_m_s`,  
- `B_in`, `B_out`, `delta_B`,  
- `gamma_in`, `gamma_out`,  
- `delta_v_busft_mm_s`.

---

## 3. Results (short version)

- **Classical Kepler**: predicts `Δv_classical = 0` for all cases, and therefore does not account for the observed mm/s-level anomalies.  
- **Anderson’s formula**: produces anomalous `Δv_Anderson` values in the **mm/s range**, reproducing both **magnitude and sign** of the observed anomalies reasonably well.  
- **BUSFT implementation (this repository)**: with the specific BUSFT coupling used here, the model typically predicts:
  - \( \Delta v_\infty^\text{BUSFT} \sim 10^4\text{–}10^5 \,\text{mm/s} \) (tens to hundreds of m/s),  
  - often with the **opposite sign** relative to the observed anomalies.

In other words, **this particular BUSFT/AM2 coupling is strongly falsified by Earth flyby data**. From a scientific perspective, this is still a positive outcome: the model is implemented, confronted with data, and rejected quantitatively.

The notebook `notebooks/02_flyby_anomaly_with_busft.ipynb` contains:

- the full diagnostic DataFrame,  
- plots comparing observed vs predicted `Δv`,  
- BUSFT diagnostics (`B`, `gamma_B`) per mission.

---

## 4. How to run

### 4.1 Setup

Create and activate a virtual environment (optional but recommended), then:

```bash
pip install -r requirements.txt
```

Dependencies are minimal: `numpy`, `pandas`, `matplotlib`, `jupyter`.

---

### 4.2 Quick smoke test

To run a quick per-mission diagnostic in the terminal:

```bash
python test_flyby.py
```

This will:

- load all flyby cases from `data/flyby_cases.csv`,  
- compute classical, Anderson, and BUSFT anomalies,  
- print per-case diagnostics (including `B_in`, `B_out`, `delta_B`, `gamma`).

---

### 4.3 Full summary table

To build the comparison table and save it to `flyby_summary.csv`:

```bash
python make_flyby_summary.py
```

The script:

- loads the CSV,  
- computes `Δv_classical`, `Δv_Anderson`, `Δv_BUSFT`,  
- assembles a diagnostic DataFrame,  
- saves the result to `flyby_summary.csv`,  
- prints a human-readable table to stdout.

---

### 4.4 Interactive exploration

To explore the analysis in a notebook:

```bash
jupyter notebook notebooks/02_flyby_anomaly_with_busft.ipynb
```

---

## 5. License / usage

The internal AM2 kernel (`src/busft_core/_kernel.py`) follows a private technical dossier and should be treated as **proprietary research code**.

For portfolio / academic review you are welcome to:

- inspect the public API (`src/busft_core/api.py`),  
- run the flyby analysis,  
- reuse the structure of the project as a research template.

For any commercial or derivative use of the AM2 kernel itself, please contact the original author.