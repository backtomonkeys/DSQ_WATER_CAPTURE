# Part 1 - Zeolite 3A + core inlet gas

DSQ water-capture code, CPE 100 project. Written with the tools taught in CPE 100
(functions, if / match-case, for / while, lists, dictionaries, files with `with open`,
classes, custom exceptions, assert tests, NumPy, pandas, matplotlib).
Python >= 3.10 (uses `match`). Needs numpy, pandas, matplotlib, openpyxl.

| File | Purpose |
|---|---|
| `parameters.csv` | All paper values, one row each, with unit and source (read with `pd.read_csv` + for loop into a dict) |
| `zeolite.py` | `Zeolite3A` class, isotherm functions, `OutOfRangeError` |
| `feed_gas.py` | `inlet_gas()` - gas entering the core from X, S (1 t CO2/day, full recycle) |
| `test_zeolite.py` | 35 tests (unit, integration, regression, step halving, contract): `python test_zeolite.py` or `pytest test_zeolite.py` |
| `analysis.ipynb` | Report figures and tables (already executed) |
| `fig1_isotherms.png` | Sips vs GSTA + capillary at 200 / 225 / 250 degC |
| `fig2_kLDF.png` | LDF rate constant and time constant vs pH2O |
| `fig3_sensitivity_X_S.png` | H2O mole fraction and gas flow into the core vs X and S |
| `fig4_min_circulation.png` | Minimum zeolite circulation vs zeolite feed temperature 200-250 degC |
| `part1_*.xlsx` | Data behind the figures and the key-number table |

Run everything from inside the `part1_zeolite` folder, because the code opens `parameters.csv`
by its file name. From another folder, give the path: `Zeolite3A("../part1_zeolite/parameters.csv")`.

## Things used that CPE 100 did not teach (and why)

Following `claude/cpe100-code-style.md`: everything else in Part 1 uses only course material.

| What | Where | Why |
|---|---|---|
| `class OutOfRangeError(ValueError)` instead of `(Exception)` | zeolite.py | Part 2 can catch it with `except ValueError` as well (agreed in the interface) |
| IAPWS-IF97 saturation-pressure formula | zeolite.py `water_psat_bar` | the capillary term needs ps(T); this is the standard steam-table equation |
| Central finite difference for d ln q / d ln p | zeolite.py `darken_factor`, `isosteric_heat_J_mol` | Darken factor and isosteric heat are derivatives of the isotherm; checked by step halving |
| `figsize=`, `markersize=`, `ncol=` in `legend` | analysis.ipynb | readable figures for the report |
| `plt.savefig("fig1_isotherms.png", dpi=200)` | analysis.ipynb | save each figure as a PNG for the report |

## Interface for Part 2

Same content as `claude/interface-contract.md` in the project. All functions take **one number**
(float) per argument - call them inside a for loop.

```python
from zeolite import Zeolite3A, OutOfRangeError
from feed_gas import inlet_gas

zeo = Zeolite3A("parameters.csv")              # Zeolite3A("parameters.csv", allow_extrapolation=True) skips range checks
zeo.loading(T_K, p_bar)                        # kg H2O / kg pellet, Sips (model="gsta_cap" to compare)
zeo.k_ldf(T_K, p_bar)                          # 1/s, micropore LDF with Darken factor
zeo.heat_of_adsorption()                       # J/mol, -60100 (method="constant" -> -45950)
zeo.equilibrium_pressure(q_kgkg, T_K)          # bar, inverse isotherm (bisection)
zeo.rho_bed_kg_m3, zeo.rho_pellet_kg_m3, zeo.eps_bed, zeo.d_pellet_m, zeo.cp_J_kgK

gas = inlet_gas()                              # defaults: X=0.30, S=0.60, P_bar=30.0, T_C=250.0
gas["F_total_mol_s"], gas["y"]["H2O"], gas["p_H2O_bar"], gas["water_kg_h"],
gas["M_mix_kg_mol"], gas["m_gas_kg_s"], gas["F_mol_s"]   # dict per species
```

`p_bar` is always the H2O **partial** pressure. `OutOfRangeError` is a `ValueError` and is raised
outside the data range of the model in use: Sips 200-250 degC, GSTA / LDF 200-350 degC, pH2O 0.05-4.5 bar.

Sips is a working-capacity isotherm, so zeolite fed to the core has q = 0 on the Sips scale.
GSTA is a true equilibrium isotherm - comparison only (with q_feed = 0 it overestimates capacity).

## Team decisions (21 Nov 2026)

* Core pressure 30 bar (40 bar = comparison case); base case X = 0.30, S = 0.60 (spec);
  heat of adsorption -60.1 kJ/mol (Sips).
* All parameters checked against the papers. cp = 960 J/(kg K) is a typical porous-ceramic value
  (Guffanti et al. 2021), not measured for zeolite 3A - call it an estimate in the report.

## Notes for the team

* Full recycle (spec): all carbon ends in DME, so water = 1.5 x fresh CO2 = 25.6 kg/h for every X and S.
  X and S set the per-pass gas composition; the flow is scaled to that water flow.
  Base case X = 0.30, S = 0.60: 3.68 mol/s, 10.7 mol% H2O (matches the spec hand calculation).
* At 30 bar the base-case inlet pH2O is 3.21 bar. It exceeds the 4.5 bar data limit when X > 0.41 (S = 0.6).
* Zeolite feed temperature: 200-250 degC (default 200 degC), the Sips data range (team decision 21 Nov 2026).
* Sips vs GSTA agree within 10 % at 250 degC for pH2O >= 0.2 bar only (23 % apart at 0.05 bar).
* Sips was fitted on 200 and 250 degC data only. If the core heats above 250 degC, Sips raises
  OutOfRangeError. Do NOT switch to GSTA in the middle of the core (Sips is a working capacity, GSTA a
  true equilibrium: the loading would jump and the mass balance would not close). Keep Sips, design the
  core to stay at or below 250 degC, and where that fails use allow_extrapolation=True and label it.
* Isosteric heat (GSTA + capillary) at 250 degC runs from about -69 kJ/mol (0.1 bar) to -40 kJ/mol (4.5 bar).
  The value -45.95 kJ/mol (Guffanti et al., Ind. Eng. Chem. Res. 2021, Table 4, citing Gabrus et al.)
  is kept for comparison only.
* GSTA P0 = 1 bar: the paper gives no number, only "standard pressure" (1.01325 bar changes results < 2 %).
* Equation forms confirmed against the PDFs: Sips exponent 1/n, capillary q = (mu/RT) / ln(ps/p).
