"""
base_case_check.py - check mode at the spec base case, with the stub and with the
real Part 1, side by side.

Run (inside part2_core):   python base_case_check.py
Writes base_case_check.csv next to this file.

Base case (claude/water-capture-code-spec.md): X 0.30, S 0.60, 30 bar, gas 250 C,
zeolite fed at 200 C, q_in 0, 3 mm pellets, screw efficiency 0.9, dH -60.1 kJ/mol,
Sips isotherm. Core size and zeolite rate are the check-mode defaults in
core_parameters.csv (D 0.30 m, H 2.0 m, 3000 kg/h).
"""

import os

import pandas as pd

import design as dz

ROW_NAMES = [
    ["capture_pct", "Water captured (%)"],
    ["T_bed_max_C", "Peak bed temperature (C)"],
    ["z_T_max_m", "Height of the peak temperature above the gas inlet (m)"],
    ["dp_bar", "Pressure drop in the core (bar)"],
    ["lift_ratio_max", "Gas drag / bed weight, largest value (-)"],
    ["m_zeo_kgh", "Zeolite circulation rate used (kg/h)"],
    ["m_min_eq_kgh", "Lowest possible rate, equilibrium with inlet gas (kg/h)"],
    ["q_out_kgkg", "Zeolite loading leaving the core (kg/kg)"],
    ["henry_pct", "Height with pH2O < 0.05 bar, Henry's law (%)"],
    ["T_gas_out_C", "Gas temperature at the top (C)"],
    ["T_zeo_out_C", "Zeolite temperature at the bottom (C)"],
    ["water_in_kgh", "Water entering with the gas (kg/h)"],
    ["water_captured_kgh", "Water captured (kg/h)"],
    ["screw_rpm", "Screw speed (rpm)"],
    ["water_closure_rel", "Water balance error (relative)"],
    ["energy_closure_rel", "Energy balance error (relative)"],
]


def base_case(params):
    """The spec base case; every value is set here so a changed CSV cannot move it."""
    changes = {"X": 0.30, "S": 0.60, "p_bar": 30.0, "T_gas_in_C": 250.0,
               "T_zeo_in_C": 200.0, "q_in_kgkg": 0.0, "d_pellet_mm": 3.0,
               "eta_screw": 0.9, "dH_method": "sips", "isotherm_model": "sips"}
    return dz.updated_case(dz.default_case(params), changes)


def run_one(case, source, params, criteria):
    """Check mode with zeolite and inlet gas from one source ("stub" or "part1")."""
    zeolite, used = dz.load_zeolite(source)
    res = dz.run_check(case, zeolite, params, criteria, gas_source=source)
    flows, gas_used = dz.get_inlet_gas(case, source)
    profile = res["profile"]
    i_max = profile["T_C"].idxmax()
    numbers = {
        "capture_pct": 100.0 * res["capture_frac"],
        "T_bed_max_C": res["T_bed_max_C"],
        "z_T_max_m": float(profile.loc[i_max, "z_m"]),
        "dp_bar": res["dp_bar"],
        "lift_ratio_max": res["lift_ratio_max"],
        "m_zeo_kgh": case["m_zeo_kgh"],
        "m_min_eq_kgh": dz.equilibrium_min_rate(case, zeolite, flows),
        "q_out_kgkg": res["q_out_kgkg"],
        "henry_pct": 100.0 * res["frac_henry"],
        "T_gas_out_C": res["T_gas_out_C"],
        "T_zeo_out_C": res["T_zeo_out_C"],
        "water_in_kgh": res["water_in_kgh"],
        "water_captured_kgh": res["water_captured_kgh"],
        "screw_rpm": res["screw_rpm"],
        "water_closure_rel": res["water_closure_rel"],
        "energy_closure_rel": res["energy_closure_rel"],
    }
    return numbers, res


def comparison_table(stub, real):
    """DataFrame: one row per quantity, columns stub / Part 1 / difference."""
    rows = []
    for key, label in ROW_NAMES:
        row = {"quantity": label, "stub": stub[key]}
        if real is not None:
            row["part1"] = real[key]
            row["part1_minus_stub"] = real[key] - stub[key]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    params = dz.load_params()
    criteria = dz.default_criteria(params)
    case = base_case(params)
    print("Base case: X", case["X"], "S", case["S"], "|", case["p_bar"], "bar | gas",
          case["T_gas_in_C"], "C | zeolite", case["T_zeo_in_C"], "C,", case["m_zeo_kgh"],
          "kg/h | D", case["D_m"], "m x H", case["H_m"], "m")

    stub, res_stub = run_one(case, "stub", params, criteria)
    real = None
    res_real = None
    try:
        real, res_real = run_one(case, "part1", params, criteria)
    except dz.Part1NotFoundError as error:
        print("PART 1 NOT FOUND:", error)
        print("Put zeolite.py, feed_gas.py, parameters.csv in",
              os.path.abspath(dz.PART1_DIR))

    table = comparison_table(stub, real)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print(table.to_string(index=False, float_format="%.4g"))
    table.to_csv(os.path.join(dz.HERE, "base_case_check.csv"), index=False,
                 encoding="utf-8")

    if res_real is not None:
        print()
        print("Part 1 warnings:")
        for text in res_real["warnings"]:
            print("  -", text)
        if real["m_zeo_kgh"] <= real["m_min_eq_kgh"]:
            print("ERROR: zeolite rate below the equilibrium minimum - mass balance wrong")


if __name__ == "__main__":
    main()
