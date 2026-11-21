"""
design.py - Check mode and design mode for the DSQ water-capture core.

check mode : the user gives the core / screw size  -> run_check()
design mode: (1) lowest zeolite rate with peak bed T <= 250 C (bisection)
             (2) at that rate sweep D and H, pick the smallest core passing all five
                 criteria, refine its height       -> design_with_rate()
             (3) report that rate against the capacity minimum (equilibrium)
             (4) lowest rate vs zeolite feed temperature -> min_rate_vs_feed_temperature()
             (5) Cp +-20 % sensitivity (not on the web page) -> cp_sensitivity()
extras     : feed_temperature_sweep(), zeolite_rate_sweep(), min_zeolite_rate(),
             max_feed_temperature(), results_to_excel()

Design criteria (core_parameters.csv, changeable on the web page):
  capture >= 95 %, gas drag <= 70 % of bed weight, dP <= 1 bar, peak bed T <= 250 C,
  H/D >= 2. H/D >= 2 is an assumed rule of thumb (no specific source): it keeps the
  design from picking a short wide core, where a moving bed flows unevenly.

A "case" is a dictionary with every input a user can change (see default_case).
"""

import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

import core_column as cc
import property_tables
import screw
import zeolite_stub

# Paths from the location of THIS file, not from the folder the program is started in,
# so "streamlit run app.py" works from any folder (pathlib: outside the course, see README)
HERE = Path(__file__).resolve().parent                     # .../repo/part2_core
PARAMS_FILE = HERE / "core_parameters.csv"
PART1_DIR = HERE.parent / "part1_zeolite"                  # .../repo/part1_zeolite


class Part1NotFoundError(Exception):
    """The Part 1 code (part1_zeolite/zeolite.py, feed_gas.py) was not found."""
    pass


# ---------------------------------------------------------------------------
# Part 1 hook-up: use the real code if the folder exists, the stubs otherwise
# ---------------------------------------------------------------------------
if str(PART1_DIR) not in sys.path:
    sys.path.append(str(PART1_DIR))     # lets Python find zeolite.py and feed_gas.py
try:
    import zeolite as part1_zeolite
    PART1_ZEOLITE_FOUND = True
except ImportError:
    PART1_ZEOLITE_FOUND = False
try:
    import feed_gas as part1_feed_gas
    PART1_FEED_FOUND = True
except ImportError:
    PART1_FEED_FOUND = False


def new_part1_zeolite():
    """Create the real Part 1 Zeolite3A with the full path of its parameters.csv.

    zeolite.py reads only that one file, so the working folder is never changed.
    """
    return part1_zeolite.Zeolite3A(str(PART1_DIR / "parameters.csv"))


def load_zeolite(source="auto"):
    """Return (zeolite object, name of the source actually used).

    source: "part1" (real Zeolite3A), "stub" (constant 0.1 kg/kg),
            "preview" (Sips preview), "auto" (part1 if found, else preview)
    """
    match source:
        case "stub":
            return zeolite_stub.Zeolite3A(), "stub"
        case "preview":
            return zeolite_stub.Zeolite3ASipsPreview(), "preview"
        case "part1":
            if not PART1_ZEOLITE_FOUND:
                raise Part1NotFoundError("part1_zeolite/zeolite.py not found")
            return new_part1_zeolite(), "part1"
        case "auto":
            if PART1_ZEOLITE_FOUND:
                return new_part1_zeolite(), "part1"
            return zeolite_stub.Zeolite3ASipsPreview(), "preview"
        case _:
            raise ValueError("Unknown zeolite source: " + str(source))


def call_inlet_gas(module, case, used):
    """Call module.inlet_gas() for a case; returns the full dictionary."""
    try:
        return module.inlet_gas(X=case["X"], S=case["S"], P_bar=case["p_bar"],
                                T_C=case["T_gas_in_C"],
                                CO2_fresh_t_day=case["basis_tCO2_day"])
    except FileNotFoundError as error:
        raise FileNotFoundError(used + " inlet_gas() opened a file relative to the folder "
                                "the program was started in (" + os.getcwd() + "): "
                                + str(error) + ". Part 2 no longer changes folder - "
                                "tell the main chat.")


def part1_inlet_gas_dict(case):
    """The full dictionary from Part 1 feed_gas.inlet_gas() (raises if Part 1 is missing)."""
    if not PART1_FEED_FOUND:
        raise Part1NotFoundError("part1_zeolite/feed_gas.py not found")
    return call_inlet_gas(part1_feed_gas, case, "part1")


def get_inlet_gas(case, source="auto"):
    """Gas entering the core {species: mol/s} and the source used.

    Calls inlet_gas(X, S, P_bar, T_C, CO2_fresh_t_day) as in the interface contract
    and takes the flows from the key "F_mol_s".
    """
    match source:
        case "stub":
            module = zeolite_stub
            used = "stub"
        case "part1" | "auto":
            if PART1_FEED_FOUND:
                module = part1_feed_gas
                used = "part1"
            elif source == "part1":
                raise Part1NotFoundError("part1_zeolite/feed_gas.py not found")
            else:
                module = zeolite_stub
                used = "stub"
        case _:
            raise ValueError("Unknown gas source: " + str(source))

    result = call_inlet_gas(module, case, used)
    try:
        flows = result["F_mol_s"]
    except KeyError:
        raise KeyError("inlet_gas() must return the key 'F_mol_s' "
                       "(claude/interface-contract.md)")
    return flows, used


# ---------------------------------------------------------------------------
# Data range of the zeolite functions, found by bisection
# ---------------------------------------------------------------------------
def point_works(zeolite, what, model, T_K, p_bar):
    """True if zeolite.loading / zeolite.k_ldf accepts (T_K, p_bar)."""
    try:
        match what:
            case "loading":
                zeolite.loading(T_K, p_bar, model)
            case "k_ldf":
                zeolite.k_ldf(T_K, p_bar)
        return True
    except ValueError:                   # OutOfRangeError is a kind of ValueError
        return False


def works_at(zeolite, what, model, vary, value, fixed):
    """point_works() with T or p as the changing variable."""
    if vary == "T":
        return point_works(zeolite, what, model, value, fixed)
    return point_works(zeolite, what, model, fixed, value)


def find_edge(zeolite, what, model, vary, inside, outside, fixed, tol):
    """Bisection for the edge of the data range along T or p.

    inside  : a value the function accepts
    outside : the furthest value we want (from core_parameters.csv)
    Returns outside if the function accepts it, otherwise the last accepted value.
    """
    if works_at(zeolite, what, model, vary, outside, fixed):
        return outside
    while abs(outside - inside) > tol:
        middle = 0.5 * (inside + outside)
        if works_at(zeolite, what, model, vary, middle, fixed):
            inside = middle
        else:
            outside = middle
    return inside


def data_range(zeolite, what, model, params):
    """Range of T and pH2O that zeolite.loading (or k_ldf) accepts.

    Never wider than the range in core_parameters.csv (the contract range).
    Tight tolerances (1e-6 K, 1e-9 bar; ~30 calls each) so that a point exactly on
    the edge, e.g. 250.00 C, is treated as inside the data.
    """
    T_inside_K = 503.15                  # 230 C and 1 bar: inside every expected range
    p_inside_bar = 1.0
    if not point_works(zeolite, what, model, T_inside_K, p_inside_bar):
        raise ValueError(what + " (" + model + ") does not accept 230 C and 1 bar")
    return {
        "T_min_K": find_edge(zeolite, what, model, "T", T_inside_K,
                             params["T_min_C"] + 273.15, p_inside_bar, 1e-6),
        "T_max_K": find_edge(zeolite, what, model, "T", T_inside_K,
                             params["T_max_C"] + 273.15, p_inside_bar, 1e-6),
        "p_min_bar": find_edge(zeolite, what, model, "p", p_inside_bar,
                               params["pH2O_min_bar"], T_inside_K, 1e-9),
        "p_max_bar": find_edge(zeolite, what, model, "p", p_inside_bar,
                               params["pH2O_max_bar"], T_inside_K, 1e-8),
    }


TABLE_CACHE = {}    # {(id of zeolite, isotherm model): [zeolite, limits, q table, k table]}


def fast_functions(zeolite, model, params):
    """Data ranges + lookup tables of q* and k_LDF for one zeolite object (built once).

    Returns (limits, loading function, k_ldf function). Tables are built only once
    per zeolite object and isotherm model and kept in TABLE_CACHE.
    """
    key = (id(zeolite), model)
    if key in TABLE_CACHE and TABLE_CACHE[key][0] is zeolite:
        entry = TABLE_CACHE[key]
    else:
        limits = zeolite_limits(zeolite, model, params)
        q_table = property_tables.PropertyTable(zeolite.loading, model, limits["loading"],
                                                params["table_dT_K"],
                                                params["table_dlnp"])
        k_table = property_tables.PropertyTable(zeolite.k_ldf, None, limits["k_ldf"],
                                                params["table_dT_K"],
                                                params["table_dlnp"])
        entry = [zeolite, limits, q_table, k_table]
        TABLE_CACHE[key] = entry
    return entry[1], entry[2].loading, entry[3].k_ldf


def zeolite_limits(zeolite, model, params):
    """Data ranges of the isotherm and of k_LDF, plus the extrapolation constants."""
    return {
        "loading": data_range(zeolite, "loading", model, params),
        "k_ldf": data_range(zeolite, "k_ldf", model, params),
        "E_iso_shift_Jmol": params["E_iso_shift_Jmol"],
        "Ea_kldf_Jmol": params["Ea_kldf_Jmol"],
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_params(csv_path=PARAMS_FILE):
    """Read core_parameters.csv into a dictionary {name: value}."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    params = {}
    for i in range(len(df)):
        params[df.loc[i, "name"]] = float(df.loc[i, "value"])
    return params


def load_param_table(csv_path=PARAMS_FILE):
    """The full parameter table (with units and sources) for display or export."""
    return pd.read_csv(csv_path, encoding="utf-8")


def default_case(params):
    """A case dictionary filled from core_parameters.csv."""
    D_m = params["D_m"]
    return {
        "X": params["X_conv"],
        "S": params["S_dme"],
        "basis_tCO2_day": params["basis_tCO2_day"],
        "p_bar": params["p_bar"],
        "T_gas_in_C": params["T_gas_in_C"],
        "T_zeo_in_C": params["T_zeolite_in_C"],
        "m_zeo_kgh": params["m_zeolite_kgh"],
        "q_in_kgkg": params["q_in_kgkg"],
        "d_pellet_mm": params["d_pellet_mm"],
        "eta_screw": params["eta_screw"],
        "D_m": D_m,
        "H_m": params["H_m"],
        "d_shaft_m": params["d_shaft_ratio"] * D_m,
        "pitch_m": params["pitch_ratio"] * D_m,
        "U_wall_Wm2K": params["U_wall_Wm2K"],
        "T_wall_C": params["T_wall_C"],
        "isotherm_model": "sips",
        "dH_method": "sips",            # "sips": -60.1 kJ/mol, "constant": -45.95 kJ/mol
        "cp_factor": 1.0,               # x zeolite.cp_J_kgK, only for the Cp sensitivity
        "k_ldf_factor": 1.0,            # x k_LDF from Part 1 (0.5 for the recommended size)
    }


def default_criteria(params):
    """Design criteria from core_parameters.csv.

    HD_min (H/D >= 2) is an assumed rule of thumb with no specific source (team
    decision 21 Nov 2026); the other four come from the spec.
    """
    return {
        "capture_min": params["capture_min"],
        "lift_max": params["lift_max_ratio"],
        "dp_max_bar": params["dp_max_bar"],
        "HD_min": params["HD_min"],
        "T_bed_max_C": params["T_bed_max_C"],
    }


def updated_case(case, changes):
    """A copy of case with some values changed; the original case is not touched."""
    new_case = dict(case)
    for key in changes:
        if key not in case:
            raise KeyError("Unknown case input: " + str(key))
        new_case[key] = changes[key]
    return new_case


def bed_data(zeolite, params, dH_method="sips", cp_factor=1.0):
    """Pellet density, bed voidage, Cp and heat of adsorption from the zeolite object.

    Always read from the zeolite object (Part 1 Zeolite3A or the stub); no densities
    are typed in Part 2. cp_factor scales Cp for the sensitivity study only.
    """
    try:
        rho_pellet = zeolite.rho_pellet_kg_m3
        eps_bed = zeolite.eps_bed
        cp_JkgK = zeolite.cp_J_kgK * cp_factor
        dH_Jmol = zeolite.heat_of_adsorption(method=dH_method)
    except AttributeError as error:
        raise AttributeError("The zeolite object lacks an attribute required by "
                             "claude/interface-contract.md: " + str(error))
    if dH_Jmol > -1.0e3 or dH_Jmol < -2.0e5:
        raise ValueError("heat_of_adsorption() = " + str(dH_Jmol)
                         + ": expected J/mol, e.g. -60100")
    return {"rho_pellet_kgm3": float(rho_pellet), "eps_bed": float(eps_bed),
            "dH_ads_Jmol": float(dH_Jmol), "cp_zeolite_JkgK": float(cp_JkgK),
            "source": "zeolite model"}


def build_model_inputs(case, params, gas_mols, props):
    """Turn a case into the dictionaries that core_column.solve_column() needs."""
    geom = {"D_m": case["D_m"], "H_m": case["H_m"], "d_shaft_m": case["d_shaft_m"],
            "pitch_m": case["pitch_m"]}
    bed = {"rho_pellet_kgm3": props["rho_pellet_kgm3"], "eps_bed": props["eps_bed"],
           "d_pellet_m": case["d_pellet_mm"] / 1000.0,
           "cp_zeolite_JkgK": props["cp_zeolite_JkgK"],
           "dH_ads_Jmol": props["dH_ads_Jmol"], "U_wall_Wm2K": case["U_wall_Wm2K"],
           "T_wall_K": case["T_wall_C"] + 273.15, "k_ldf_factor": case["k_ldf_factor"]}
    op = {"gas_in_mols": gas_mols, "T_gas_in_K": case["T_gas_in_C"] + 273.15,
          "p_in_bar": case["p_bar"], "m_zeo_kgs": case["m_zeo_kgh"] / 3600.0,
          "T_zeo_in_K": case["T_zeo_in_C"] + 273.15, "q_in_kgkg": case["q_in_kgkg"]}
    return geom, bed, op


# ---------------------------------------------------------------------------
# Check mode
# ---------------------------------------------------------------------------
def evaluate_criteria(res, criteria, D_m, H_m):
    """Pass/fail for the design criteria (4 from the spec + H/D added by Part 2)."""
    flags = {
        "pass_capture": res["capture_frac"] >= criteria["capture_min"],
        "pass_lift": res["lift_ratio_max"] <= criteria["lift_max"],
        "pass_dp": res["dp_bar"] <= criteria["dp_max_bar"],
        "pass_temperature": res["T_bed_max_C"] <= criteria["T_bed_max_C"],
        "pass_aspect": H_m / D_m >= criteria["HD_min"],
    }
    flags["feasible"] = (flags["pass_capture"] and flags["pass_lift"]
                         and flags["pass_dp"] and flags["pass_temperature"]
                         and flags["pass_aspect"])
    return flags


def run_check(case, zeolite, params, criteria, gas_source="auto", n_steps=None,
              policy="clip", use_tables=True):
    """Check mode: simulate one core size.

    use_tables : read q* and k_LDF from lookup tables (property_tables.py, ~20x
                 faster with the real Part 1; relative error < 1e-4). Only with the
                 "clip" policy; use_tables=False calls the zeolite object directly.
    Returns the core_column result dictionary plus the screw numbers, the criteria
    flags and the inputs.
    """
    gas_mols, gas_used = get_inlet_gas(case, gas_source)
    props = bed_data(zeolite, params, case["dH_method"], case["cp_factor"])
    geom, bed, op = build_model_inputs(case, params, gas_mols, props)
    if use_tables and policy == "clip":
        limits, loading_fn, kldf_fn = fast_functions(zeolite, case["isotherm_model"],
                                                     params)
    else:
        limits = zeolite_limits(zeolite, case["isotherm_model"], params)
        loading_fn = zeolite.loading
        kldf_fn = zeolite.k_ldf
    if n_steps is None:
        n_steps = int(params["n_steps"])

    # the isotherm and LDF functions are passed as arguments
    res = cc.solve_column(geom, op, bed, loading_fn, kldf_fn, limits=limits,
                          isotherm_model=case["isotherm_model"], policy=policy,
                          n_steps=n_steps, pinch_tol=params["pinch_tol"])

    rho_bulk = props["rho_pellet_kgm3"] * (1.0 - props["eps_bed"])
    screw_numbers = screw.screw_summary(op["m_zeo_kgs"], rho_bulk, case["D_m"],
                                        case["d_shaft_m"], case["pitch_m"],
                                        case["eta_screw"])
    for key in screw_numbers:
        res[key] = screw_numbers[key]
    flags = evaluate_criteria(res, criteria, case["D_m"], case["H_m"])
    for key in flags:
        res[key] = flags[key]
    res["volume_m3"] = cc.outer_volume_m3(geom)
    res["gas_source"] = gas_used
    res["gas_in_mols"] = gas_mols
    res["dH_ads_kJmol"] = props["dH_ads_Jmol"] / 1000.0
    res["rho_bulk_kgm3"] = rho_bulk
    res["eps_bed"] = props["eps_bed"]
    res["bed_props_source"] = props["source"]
    res["cp_zeolite_JkgK"] = props["cp_zeolite_JkgK"]
    if case["p_bar"] > params["p_tested_max_bar"]:
        res["warnings"].append("Pressure above " + str(round(params["p_tested_max_bar"]))
                               + " bar: the sorbent data (5-30 bar) are extrapolated")
    # contract: Sips is a working capacity after regeneration, so q_in = 0 fits it;
    # GSTA is absolute, so it needs the real loading left after regeneration
    if case["isotherm_model"] != "sips" and case["q_in_kgkg"] == 0:
        res["warnings"].append("GSTA with q_in = 0 overestimates the capacity: set q_in to "
                               "the equilibrium loading after regeneration (not given in "
                               "the paper). Use GSTA only for comparison.")
    res["case"] = dict(case)
    return res


def short_number(x):
    """Number as text with 3 significant digits, e.g. 1.23e-06 (no format strings)."""
    if x == 0:
        return "0"
    exponent = math.floor(math.log10(abs(x)))
    if -3 <= exponent <= 3:
        return str(round(x, 3 - exponent))
    return str(round(x / 10.0 ** exponent, 2)) + "e" + str(exponent)


def summary_table(res):
    """Two-column table (quantity, value) of the main results."""
    case = res["case"]
    rows = [
        ["Thermal regime", res["regime"]],
        ["Water captured", format_capture(res["capture_frac"])],
        ["Water in / captured [kg/h]", str(round(res["water_in_kgh"], 2)) + " / "
         + str(round(res["water_captured_kgh"], 2))],
        ["y_H2O in / out [-]", str(round(res["y_H2O_in"], 4)) + " / "
         + short_number(res["y_H2O_out"])],
        ["Zeolite loading in / out [kg/kg]", str(round(case["q_in_kgkg"], 4)) + " / "
         + str(round(res["q_out_kgkg"], 4))],
        ["Bed T max [C]", round(res["T_bed_max_C"], 1)],
        ["Gas out T / zeolite out T [C]", str(round(res["T_gas_out_C"], 1)) + " / "
         + str(round(res["T_zeo_out_C"], 1))],
        ["W_solid / W_gas [-]", round(res["W_ratio_solid_gas"], 3)],
        ["Heat of adsorption used [kJ/mol]", round(res["dH_ads_kJmol"], 2)],
        ["Heat of adsorption released [kW]", round(res["Q_ads_kW"], 2)],
        ["Bed bulk density [kg/m3] / voidage [-]", str(round(res["rho_bulk_kgm3"]))
         + " / " + str(round(res["eps_bed"], 3))],
        ["Pressure drop [bar]", round(res["dp_bar"], 5)],
        ["Gas drag / bed weight, largest local value (at z = "
         + str(round(res["z_lift_max_m"], 2)) + " m) [-]", round(res["lift_ratio_max"], 3)],
        ["Gas drag / bed weight, height average [-]", round(res["lift_ratio_avg"], 3)],
        ["Zeolite residence time [min]", round(res["tau_zeo_s"] / 60, 2)],
        ["Zeolite hold-up [kg]", round(res["holdup_kg"], 1)],
        ["Screw speed [rpm]", round(res["screw_rpm"], 2)],
        ["Outer volume [m3]", round(res["volume_m3"], 4)],
        ["Water balance closure [-]", res["water_closure_rel"]],
        ["Energy balance closure [-]", res["energy_closure_rel"]],
    ]
    for row in rows:
        row[1] = str(row[1])            # one text column for the web table
    return pd.DataFrame(rows, columns=["quantity", "value"])


# ---------------------------------------------------------------------------
# Design mode
# ---------------------------------------------------------------------------
def sized_case(case, D_m, H_m, d_ratio, pitch_ratio):
    """Case with a new size; shaft and pitch scale with D."""
    return updated_case(case, {"D_m": D_m, "H_m": H_m, "d_shaft_m": d_ratio * D_m,
                               "pitch_m": pitch_ratio * D_m})


def row_from_result(case, res, error):
    """One row of the design table."""
    row = {"D_m": case["D_m"], "H_m": case["H_m"], "d_shaft_m": case["d_shaft_m"],
           "pitch_m": case["pitch_m"],
           "volume_m3": 3.141592653589793 / 4 * case["D_m"] ** 2 * case["H_m"],
           "error": error}
    if res is None:
        row["feasible"] = False
        return row
    keys = ["regime", "capture_frac", "T_bed_max_C", "T_gas_out_C", "dp_bar",
            "lift_ratio_max", "screw_rpm", "tau_zeo_s", "holdup_kg", "pass_capture",
            "pass_lift", "pass_dp", "pass_temperature", "pass_aspect", "feasible",
            "frac_p_extrapolated",
            "frac_T_extrapolated"]
    for key in keys:
        row[key] = res[key]
    # safety factor against lifting the bed = bed weight / gas drag (1 / lift ratio)
    row["lift_safety_factor"] = 1.0 / max(res["lift_ratio_max"], 1e-12)
    return row


def design_sweep(case, zeolite, params, criteria, D_list_m, H_list_m, d_ratio=None,
                 pitch_ratio=None, gas_source="auto", n_steps=None, progress=None):
    """Grid sweep over D and H (two for-loops). Returns one DataFrame row per size.

    progress : optional function progress(fraction_done) for the web page
    """
    if d_ratio is None:
        d_ratio = params["d_shaft_ratio"]
    if pitch_ratio is None:
        pitch_ratio = params["pitch_ratio"]
    rows = []
    n_total = len(D_list_m) * len(H_list_m)
    n_done = 0
    for D_m in D_list_m:
        for H_m in H_list_m:
            c = sized_case(case, float(D_m), float(H_m), d_ratio, pitch_ratio)
            try:
                res = run_check(c, zeolite, params, criteria, gas_source, n_steps)
                rows.append(row_from_result(c, res, ""))
            except cc.CoreModelError as error:
                rows.append(row_from_result(c, None, str(error)))
            except ValueError as error:
                rows.append(row_from_result(c, None, str(error)))
            n_done = n_done + 1
            if progress is not None:
                progress(n_done / n_total)
    return pd.DataFrame(rows)


def pick_smallest(df):
    """Row with the smallest outer volume among feasible sizes (None if none)."""
    feasible = df[df["feasible"] == True]  # noqa: E712 (column may hold numpy bools)
    if len(feasible) == 0:
        return None
    i = feasible["volume_m3"].idxmin()
    return feasible.loc[i]


def capture_pivot(df):
    """Capture [%] as a table: rows = H, columns = D (pandas pivot)."""
    data = df.copy()
    data["capture_pct"] = 100.0 * data["capture_frac"]
    return data.pivot(index="H_m", columns="D_m", values="capture_pct")


def refine_height(case, zeolite, params, criteria, D_m, H_low_m, H_high_m,
                  d_ratio=None, pitch_ratio=None, gas_source="auto", tol_m=0.02):
    """Bisection on H (fixed D) for the smallest height that reaches the target.

    H_low_m must miss the capture target and H_high_m must reach it.
    Returns (H_m, result at that H).
    """
    if d_ratio is None:
        d_ratio = params["d_shaft_ratio"]
    if pitch_ratio is None:
        pitch_ratio = params["pitch_ratio"]
    target = criteria["capture_min"]

    res_high = run_check(sized_case(case, D_m, H_high_m, d_ratio, pitch_ratio), zeolite,
                         params, criteria, gas_source)
    if res_high["capture_frac"] < target:
        raise ValueError("H_high_m does not reach the capture target")

    # the H/D criterion sets the lowest allowed height
    H_floor_m = criteria["HD_min"] * D_m
    if H_low_m < H_floor_m <= H_high_m:
        res_floor = run_check(sized_case(case, D_m, H_floor_m, d_ratio, pitch_ratio),
                              zeolite, params, criteria, gas_source)
        if res_floor["capture_frac"] >= target:
            return H_floor_m, res_floor
        H_low_m = H_floor_m

    while H_high_m - H_low_m > tol_m:
        H_mid = 0.5 * (H_low_m + H_high_m)
        res_mid = run_check(sized_case(case, D_m, H_mid, d_ratio, pitch_ratio), zeolite,
                            params, criteria, gas_source)
        if res_mid["capture_frac"] >= target:
            H_high_m = H_mid
            res_high = res_mid
        else:
            H_low_m = H_mid
    return H_high_m, res_high


def design_core(case, zeolite, params, criteria, D_list_m, H_list_m, gas_source="auto",
                refine=True, progress=None):
    """Design mode in one call: grid sweep -> smallest feasible -> refine its height.

    Returns {"sweep": DataFrame, "best": row or None, "refined": dict or None}.
    """
    df = design_sweep(case, zeolite, params, criteria, D_list_m, H_list_m,
                      gas_source=gas_source, progress=progress)
    best = pick_smallest(df)
    refined = None
    if best is not None and refine:
        D_m = float(best["D_m"])
        H_high = float(best["H_m"])
        lower = df[(df["D_m"] == D_m) & (df["H_m"] < H_high)]
        if len(lower) > 0:
            H_low = float(lower["H_m"].max())
        else:
            H_low = 0.5 * H_high
        try:
            H_m, res = refine_height(case, zeolite, params, criteria, D_m, H_low, H_high,
                                     gas_source=gas_source)
            if res["feasible"]:
                refined = {"D_m": D_m, "H_m": H_m, "result": res}
        except cc.CoreModelError:
            refined = None
        except ValueError:
            refined = None
    return {"sweep": df, "best": best, "refined": refined}


# ---------------------------------------------------------------------------
# Sweeps at a fixed size
# ---------------------------------------------------------------------------
def feed_temperature_sweep(case, zeolite, params, criteria, T_list_C, gas_source="auto"):
    """Effect of the zeolite feed temperature (spec: 150-250 C)."""
    rows = []
    for T_C in T_list_C:
        c = updated_case(case, {"T_zeo_in_C": float(T_C)})
        try:
            res = run_check(c, zeolite, params, criteria, gas_source)
            rows.append({"T_zeo_in_C": T_C, "capture_pct": 100 * res["capture_frac"],
                         "T_bed_max_C": res["T_bed_max_C"],
                         "T_gas_out_C": res["T_gas_out_C"],
                         "frac_T_extrapolated": res["frac_T_extrapolated"],
                         "regime": res["regime"], "error": ""})
        except cc.CoreModelError as error:
            rows.append({"T_zeo_in_C": T_C, "error": str(error)})
        except ValueError as error:
            rows.append({"T_zeo_in_C": T_C, "error": str(error)})
    return pd.DataFrame(rows)


def max_feed_temperature(case, zeolite, params, criteria, gas_source="auto", tol_C=0.5):
    """Highest zeolite feed temperature that keeps the bed at or below the T criterion.

    The bed is never colder than the zeolite feed, and it gets hotter as the feed gets
    hotter, so bisection on the feed temperature works. Searched between the lower
    edge of the data (T_min_C) and 250 C.
    Returns (T_feed_C, result) or (None, None) if even the coldest feed is too hot.
    """
    T_limit = criteria["T_bed_max_C"]
    T_low = params["T_min_C"]
    T_high = 250.0

    res_high = run_check(updated_case(case, {"T_zeo_in_C": T_high}), zeolite, params,
                         criteria, gas_source)
    if res_high["T_bed_max_C"] <= T_limit:
        return T_high, res_high
    res_low = run_check(updated_case(case, {"T_zeo_in_C": T_low}), zeolite, params,
                        criteria, gas_source)
    if res_low["T_bed_max_C"] > T_limit:
        return None, None

    while T_high - T_low > tol_C:
        T_mid = 0.5 * (T_low + T_high)
        res_mid = run_check(updated_case(case, {"T_zeo_in_C": T_mid}), zeolite, params,
                            criteria, gas_source)
        if res_mid["T_bed_max_C"] <= T_limit:
            T_low = T_mid
            res_low = res_mid
        else:
            T_high = T_mid
    return T_low, res_low


def zeolite_rate_sweep(case, zeolite, params, criteria, m_list_kgh, gas_source="auto"):
    """Capture vs zeolite circulation (shows the gap where W_solid ~ W_gas)."""
    rows = []
    for m_kgh in m_list_kgh:
        c = updated_case(case, {"m_zeo_kgh": float(m_kgh)})
        try:
            res = run_check(c, zeolite, params, criteria, gas_source)
            rows.append({"m_zeo_kgh": m_kgh, "capture_pct": 100 * res["capture_frac"],
                         "T_bed_max_C": res["T_bed_max_C"], "regime": res["regime"],
                         "W_ratio_solid_gas": res["W_ratio_solid_gas"],
                         "screw_rpm": res["screw_rpm"], "error": ""})
        except cc.CoreModelError as error:
            rows.append({"m_zeo_kgh": m_kgh, "error": str(error)})
        except ValueError as error:
            rows.append({"m_zeo_kgh": m_kgh, "error": str(error)})
    return pd.DataFrame(rows)


def equilibrium_min_rate(case, zeolite, gas_mols, model="sips"):
    """Lowest possible zeolite rate [kg/h]: all water leaves on zeolite at equilibrium
    with the INLET gas at the zeolite feed temperature (mass-balance lower bound).

    m_min = water_in / (q*(T_zeo_in, pH2O_in) - q_in). Any working design must use more.
    """
    F_total = 0.0
    for species in gas_mols:
        F_total = F_total + gas_mols[species]
    pH2O_in_bar = gas_mols["H2O"] / F_total * case["p_bar"]
    q_star = zeolite.loading(case["T_zeo_in_C"] + 273.15, pH2O_in_bar, model)
    water_kgh = gas_mols["H2O"] * cc.M_W_KGMOL * 3600.0
    return water_kgh / (q_star - case["q_in_kgkg"])


def lowest_stable_rate(case, zeolite, params, gas_source="auto"):
    """Zeolite rate [kg/h] just above the 'balanced' rate (W_solid = W_gas).

    Near the balanced rate the temperature front has no fixed place (pinch), so the
    rate searches start a little above it (pinch_tol in core_parameters.csv).
    """
    gas_mols, used = get_inlet_gas(case, gas_source)
    props = bed_data(zeolite, params, case["dH_method"], case["cp_factor"])
    m_bal_kgh = 3600.0 * cc.balanced_zeolite_rate_kgs(
        gas_mols, case["T_gas_in_C"] + 273.15, props["cp_zeolite_JkgK"],
        case["q_in_kgkg"])
    return m_bal_kgh * (1.0 + 3.0 * params["pinch_tol"]) + 1.0


def min_zeolite_rate(case, zeolite, params, criteria, m_max_kgh=6000.0,
                     gas_source="auto", tol_kgh=5.0):
    """Smallest zeolite rate [kg/h] that reaches the capture target at a fixed size.

    Searched only above the 'balanced' rate (W_solid > W_gas), where capture rises
    with the zeolite rate. Returns (m_kgh, result) or (None, None).
    """
    m_low = lowest_stable_rate(case, zeolite, params, gas_source)
    m_high = m_max_kgh
    target = criteria["capture_min"]

    res_high = run_check(updated_case(case, {"m_zeo_kgh": m_high}), zeolite, params,
                         criteria, gas_source)
    if res_high["capture_frac"] < target:
        return None, None
    try:
        res_low = run_check(updated_case(case, {"m_zeo_kgh": m_low}), zeolite, params,
                            criteria, gas_source)
        if res_low["capture_frac"] >= target:
            return m_low, res_low
    except cc.CoreModelError:
        pass

    while m_high - m_low > tol_kgh:
        m_mid = 0.5 * (m_low + m_high)
        try:
            res_mid = run_check(updated_case(case, {"m_zeo_kgh": m_mid}), zeolite, params,
                                criteria, gas_source)
            ok = res_mid["capture_frac"] >= target
        except cc.CoreModelError:
            ok = False
        if ok:
            m_high = m_mid
            res_high = res_mid
        else:
            m_low = m_mid
    return m_high, res_high


# ---------------------------------------------------------------------------
# Design mode with the zeolite rate as a design variable (team decision 21 Nov 2026)
# ---------------------------------------------------------------------------
def format_capture(capture_frac):
    """Capture as text; above 99.9 % it is shown as "> 99.9%" because the top of the
    core is on Henry's law and the outlet water cannot be quoted precisely."""
    if capture_frac > 0.999:
        return "> 99.9%"
    return str(round(100.0 * capture_frac, 2)) + "%"


def check_T_falls_with_rate(points):
    """assert: peak bed T goes down when the zeolite rate goes up.

    points : list of [m_kgh, T_bed_max_C] evaluated during the bisection
    """
    points.sort()                       # a list of [m, T] sorts by m
    for i in range(len(points) - 1):
        m_a, T_a = points[i]
        m_b, T_b = points[i + 1]
        assert T_b <= T_a + 1e-6, ("peak bed T rose from " + str(round(T_a, 3)) + " to "
                                   + str(round(T_b, 3)) + " C when the zeolite rate rose "
                                   "from " + str(round(m_a)) + " to " + str(round(m_b))
                                   + " kg/h")


def min_rate_for_temperature(case, zeolite, params, criteria, gas_source="auto",
                             m_max_kgh=50000.0, tol_rel=1e-3):
    """Lowest zeolite rate [kg/h] with peak bed T <= criteria["T_bed_max_C"] (bisection).

    The zeolite carries the heat of adsorption away, so more zeolite = cooler bed
    (checked with an assert on every evaluated point). The core size of case is used.
    Returns (m_kgh, result), or (None, result at m_max_kgh) if no rate up to
    m_max_kgh keeps the bed cool enough.
    """
    T_limit = criteria["T_bed_max_C"]
    points = []
    res_high = run_check(updated_case(case, {"m_zeo_kgh": m_max_kgh}), zeolite, params,
                         criteria, gas_source)
    points.append([m_max_kgh, res_high["T_bed_max_C"]])
    if res_high["T_bed_max_C"] > T_limit:
        return None, res_high

    m_low = lowest_stable_rate(case, zeolite, params, gas_source)
    m_high = m_max_kgh
    try:
        res_low = run_check(updated_case(case, {"m_zeo_kgh": m_low}), zeolite, params,
                            criteria, gas_source)
        points.append([m_low, res_low["T_bed_max_C"]])
        if res_low["T_bed_max_C"] <= T_limit:
            check_T_falls_with_rate(points)
            return m_low, res_low
    except cc.CoreModelError:
        pass                            # too close to the pinch: treat as "too hot"

    while m_high - m_low > tol_rel * m_high:
        m_mid = 0.5 * (m_low + m_high)
        try:
            res_mid = run_check(updated_case(case, {"m_zeo_kgh": m_mid}), zeolite,
                                params, criteria, gas_source)
            points.append([m_mid, res_mid["T_bed_max_C"]])
            cool_enough = res_mid["T_bed_max_C"] <= T_limit
        except cc.CoreModelError:
            cool_enough = False
        if cool_enough:
            m_high = m_mid
            res_high = res_mid
        else:
            m_low = m_mid
    check_T_falls_with_rate(points)
    return m_high, res_high


def reference_case(case, params):
    """Case at the check-mode size (D 0.30 m x H 2.0 m): all water is captured there,
    so the heat released is the largest -> the rate found is on the safe side."""
    return sized_case(case, params["D_m"], params["H_m"], params["d_shaft_ratio"],
                      params["pitch_ratio"])


def design_with_rate(case, zeolite, params, criteria, D_list_m, H_list_m,
                     gas_source="auto", progress=None):
    """Design mode steps (1)-(3).

    (1) lowest zeolite rate with peak bed T <= limit, at the reference size
    (2) at that rate: sweep D x H, smallest core passing all five criteria, refine H;
        plus the recommended size (all criteria still met with k_LDF x 0.5)
    (3) that rate vs the capacity minimum (equilibrium with the inlet gas)
    Returns a dictionary; "design" is None (with "reason") when nothing passes.
    """
    ref = reference_case(case, params)
    m_T, res_T = min_rate_for_temperature(ref, zeolite, params, criteria, gas_source)
    gas_mols, used = get_inlet_gas(case, gas_source)
    m_cap = equilibrium_min_rate(case, zeolite, gas_mols, case["isotherm_model"])
    out = {"m_T_kgh": m_T, "res_T": res_T, "m_capacity_kgh": m_cap,
           "ratio_T_to_capacity": None, "ref_D_m": ref["D_m"], "ref_H_m": ref["H_m"],
           "design": None, "recommended": None, "reason": ""}
    if m_T is None:
        out["reason"] = ("No zeolite rate up to " + str(round(res_T["case"]["m_zeo_kgh"]))
                         + " kg/h keeps the bed at or below "
                         + str(round(criteria["T_bed_max_C"])) + " C (peak "
                         + str(round(res_T["T_bed_max_C"], 1)) + " C)")
        return out
    out["ratio_T_to_capacity"] = m_T / m_cap
    rate_case = updated_case(case, {"m_zeo_kgh": m_T})
    out["design"] = design_core(rate_case, zeolite, params, criteria, D_list_m, H_list_m,
                                gas_source=gas_source, progress=progress)
    if out["design"]["best"] is None:
        out["reason"] = why_nothing_passes(out["design"]["sweep"])
        out["recommended"] = None
        return out
    out["recommended"] = recommended_size(rate_case, zeolite, params, criteria,
                                          out["design"]["sweep"], gas_source)
    return out


K_LDF_MARGIN_TEXT = ("k_LDF halved: the model matches van Kampen 2021 Fig. 9 only within "
                     "a factor of ~2 and ignores macropore resistance of 3 mm pellets "
                     "(assumption)")


def recommended_size(rate_case, zeolite, params, criteria, df_sweep, gas_source="auto",
                     k_factor=None):
    """Recommended size: the smallest size that still passes all five criteria with
    k_LDF x 0.5 (k_ldf_factor_recommended in core_parameters.csv).

    k_LDF halved: the model matches van Kampen 2021 Fig. 9 only within a factor of ~2
    and ignores macropore resistance of 3 mm pellets (assumption).

    Sizes are tried from the smallest volume up, among the sizes that pass at the
    nominal k_LDF (df_sweep), so the recommended size also passes at k_LDF x 1.
    Its height is then refined by bisection. Returns a dictionary or None.
    """
    if k_factor is None:
        k_factor = params["k_ldf_factor_recommended"]
    d_ratio = params["d_shaft_ratio"]
    pitch_ratio = params["pitch_ratio"]
    slow_case = updated_case(rate_case, {"k_ldf_factor": k_factor})
    feasible = df_sweep[df_sweep["feasible"] == True]  # noqa: E712 (numpy bools)
    feasible = feasible.sort_values("volume_m3")
    for i in feasible.index:
        D_m = float(feasible.loc[i, "D_m"])
        H_grid = float(feasible.loc[i, "H_m"])
        try:
            res = run_check(sized_case(slow_case, D_m, H_grid, d_ratio, pitch_ratio),
                            zeolite, params, criteria, gas_source)
        except cc.CoreModelError:
            continue
        if not res["feasible"]:
            continue
        H_m = H_grid
        res_slow = res
        lower = df_sweep[(df_sweep["D_m"] == D_m) & (df_sweep["H_m"] < H_grid)]
        if len(lower) > 0:
            try:
                H_try, res_try = refine_height(slow_case, zeolite, params, criteria, D_m,
                                               float(lower["H_m"].max()), H_grid,
                                               d_ratio, pitch_ratio, gas_source)
                nominal_try = run_check(sized_case(rate_case, D_m, H_try, d_ratio,
                                                   pitch_ratio),
                                        zeolite, params, criteria, gas_source)
                if res_try["feasible"] and nominal_try["feasible"]:
                    H_m = H_try
                    res_slow = res_try
            except cc.CoreModelError:
                pass
            except ValueError:
                pass
        nominal = run_check(sized_case(rate_case, D_m, H_m, d_ratio, pitch_ratio),
                            zeolite, params, criteria, gas_source)
        return {"D_m": D_m, "H_m": H_m, "k_ldf_factor": k_factor,
                "result_slow": res_slow, "result_nominal": nominal}
    return None


def why_nothing_passes(df):
    """Text listing how many grid sizes fail each criterion."""
    ok = df[df["error"] == ""]
    text = "No size in the grid passes all criteria (" + str(len(df)) + " sizes"
    if len(ok) < len(df):
        text = text + ", " + str(len(df) - len(ok)) + " without a steady solution"
    text = text + "). Sizes failing each criterion:"
    names = [["pass_capture", "capture"], ["pass_lift", "lift"], ["pass_dp", "dP"],
             ["pass_temperature", "T"], ["pass_aspect", "H/D"]]
    for column, label in names:
        n_fail = int((ok[column] == False).sum())  # noqa: E712 (numpy bools)
        text = text + " " + label + " " + str(n_fail) + ";"
    return text


def min_rate_vs_feed_temperature(case, zeolite, params, criteria, T_list_C,
                                 gas_source="auto"):
    """Design step (4): lowest zeolite rate (T criterion) for each feed temperature."""
    rows = []
    ref = reference_case(case, params)
    gas_mols, used = get_inlet_gas(case, gas_source)
    for T_C in T_list_C:
        c = updated_case(ref, {"T_zeo_in_C": float(T_C)})
        m_cap = equilibrium_min_rate(c, zeolite, gas_mols, case["isotherm_model"])
        try:
            m_T, res = min_rate_for_temperature(c, zeolite, params, criteria, gas_source)
            note = ""
            if m_T is None:
                note = ("no rate up to " + str(round(res["case"]["m_zeo_kgh"]))
                        + " kg/h: peak " + str(round(res["T_bed_max_C"], 1)) + " C")
        except cc.CoreModelError as error:
            m_T = None
            note = "model error: " + str(error)
        row = {"T_zeo_in_C": T_C, "m_min_T_kgh": m_T, "m_min_capacity_kgh": m_cap,
               "ratio": float("nan"), "note": note}
        if m_T is not None:
            row["ratio"] = m_T / m_cap
        rows.append(row)
    return pd.DataFrame(rows)


def cp_sensitivity(case, zeolite, params, criteria, factors=(0.8, 1.0, 1.2),
                   gas_source="auto"):
    """Design step (5): lowest zeolite rate for zeolite Cp x 0.8 / 1.0 / 1.2.

    Cp comes from the zeolite object (960 J/(kg K), a generic porous-ceramic value),
    so only the factor is set here. Not shown on the web page.
    """
    rows = []
    ref = reference_case(case, params)
    base_m = None
    for factor in factors:
        c = updated_case(ref, {"cp_factor": float(factor)})
        m_T, res = min_rate_for_temperature(c, zeolite, params, criteria, gas_source)
        cp = zeolite.cp_J_kgK * factor
        row = {"cp_factor": factor, "cp_J_kgK": cp, "m_min_T_kgh": m_T,
               "change_vs_base_pct": float("nan")}
        if factor == 1.0:
            base_m = m_T
        rows.append(row)
    df = pd.DataFrame(rows)
    if base_m is not None:
        for i in range(len(df)):
            if pd.notna(df.loc[i, "m_min_T_kgh"]):
                df.loc[i, "change_vs_base_pct"] = 100.0 * (df.loc[i, "m_min_T_kgh"]
                                                           / base_m - 1.0)
    return df


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def results_to_excel(path, sheets):
    """Write {sheet_name: DataFrame} to one Excel file (pandas ExcelWriter)."""
    with pd.ExcelWriter(path) as writer:
        for name in sheets:
            sheets[name].to_excel(writer, sheet_name=str(name)[:31], index=False)
    return path


# ---------------------------------------------------------------------------
# Run design mode from the terminal:  python design.py
# ---------------------------------------------------------------------------
def number_list(start, stop, step):
    """[start, start + step, ..., stop] (for-loop instead of numpy.arange)."""
    values = []
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        values.append(round(start + i * step, 6))
    return values


DESIGN_D_LIST_M = number_list(0.10, 0.50, 0.05)
DESIGN_H_LIST_M = number_list(0.25, 3.00, 0.25)
FEED_T_LIST_C = number_list(200.0, 250.0, 5.0)


def print_progress(fraction_done):
    """Progress of the design sweep on one terminal line (\r goes back to its start)."""
    print("\r  design sweep: " + str(round(100 * fraction_done)) + " %   ", end="",
          flush=True)


def criteria_lines(res, criteria):
    """The five criteria of one result as printable lines."""
    case = res["case"]
    marks = {True: "PASS", False: "FAIL"}
    return [
        "  capture  " + format_capture(res["capture_frac"]) + "  (>= "
        + str(round(100 * criteria["capture_min"], 1)) + "%)  " + marks[res["pass_capture"]],
        "  lift     drag/weight " + str(round(res["lift_ratio_max"], 3)) + "  (<= "
        + str(criteria["lift_max"]) + ")  " + marks[res["pass_lift"]],
        "  dP       " + str(round(res["dp_bar"], 4)) + " bar  (<= "
        + str(criteria["dp_max_bar"]) + " bar)  " + marks[res["pass_dp"]],
        "  T max    " + str(round(res["T_bed_max_C"], 1)) + " C  (<= "
        + str(criteria["T_bed_max_C"]) + " C)  " + marks[res["pass_temperature"]],
        "  H/D      " + str(round(case["H_m"] / case["D_m"], 2)) + "  (>= "
        + str(criteria["HD_min"]) + ", assumed rule of thumb)  " + marks[res["pass_aspect"]],
    ]


def main():
    """Design mode at the spec base case; prints the results and writes an Excel file."""
    params = load_params()
    criteria = default_criteria(params)
    zeolite, used = load_zeolite("auto")
    case = default_case(params)
    print("Zeolite model:", used, "| zeolite feed", case["T_zeo_in_C"], "C | Cp",
          zeolite.cp_J_kgK, "J/(kg K)")
    t_start = time.time()
    print("Building lookup tables of q* and k_LDF ...", flush=True)
    fast_functions(zeolite, case["isotherm_model"], params)
    print("  done in " + str(round(time.time() - t_start, 1)) + " s")
    print("(1) Bisection on the zeolite rate, then (2) sweep of "
          + str(len(DESIGN_D_LIST_M) * len(DESIGN_H_LIST_M)) + " sizes ...", flush=True)

    out = design_with_rate(case, zeolite, params, criteria, DESIGN_D_LIST_M,
                           DESIGN_H_LIST_M, progress=print_progress)
    print()
    print("  time so far " + str(round(time.time() - t_start)) + " s")
    print("(1)-(3) Lowest zeolite rate with peak bed T <= " + str(criteria["T_bed_max_C"])
          + " C (at D " + str(out["ref_D_m"]) + " m x H " + str(out["ref_H_m"]) + " m):")
    if out["m_T_kgh"] is None:
        print("  none -", out["reason"])
    else:
        print("  " + str(round(out["m_T_kgh"], 1)) + " kg/h  (capacity minimum "
              + str(round(out["m_capacity_kgh"], 1)) + " kg/h, ratio "
              + str(round(out["ratio_T_to_capacity"], 2)) + ")")
        design = out["design"]
        if design["best"] is None:
            print("  No size passes:", out["reason"])
        else:
            if design["refined"] is not None:
                res = design["refined"]["result"]
                label = "refined height"
            else:
                rate_case = updated_case(case, {"m_zeo_kgh": out["m_T_kgh"]})
                res = run_check(sized_case(rate_case, float(design["best"]["D_m"]),
                                           float(design["best"]["H_m"]),
                                           params["d_shaft_ratio"], params["pitch_ratio"]),
                                zeolite, params, criteria)
                label = "grid point"
            print("  Smallest size (" + label + "): D " + str(round(res["case"]["D_m"], 3))
                  + " m x H " + str(round(res["case"]["H_m"], 3)) + " m, volume "
                  + str(round(res["volume_m3"], 4)) + " m3, screw "
                  + str(round(res["screw_rpm"], 1)) + " rpm")
            for line in criteria_lines(res, criteria):
                print(line)
            for text in res["warnings"]:
                print("  warning:", text)
            for text in res["notes"]:
                print("  note:", text)
        rec = out["recommended"]
        if rec is None:
            print("  Recommended size: none in the grid passes with k_LDF x "
                  + str(params["k_ldf_factor_recommended"]))
        else:
            res = rec["result_slow"]
            print("  Recommended size (" + K_LDF_MARGIN_TEXT + "): D "
                  + str(round(rec["D_m"], 3)) + " m x H " + str(round(rec["H_m"], 3))
                  + " m, volume " + str(round(res["volume_m3"], 4)) + " m3")
            print("  criteria with k_LDF x " + str(rec["k_ldf_factor"]) + ":")
            for line in criteria_lines(res, criteria):
                print(line)
            print("  capture at the nominal k_LDF: "
                  + format_capture(rec["result_nominal"]["capture_frac"]))

    print()
    print("(4) Lowest zeolite rate vs zeolite feed temperature (11 bisections) ...",
          flush=True)
    df_T = min_rate_vs_feed_temperature(case, zeolite, params, criteria, FEED_T_LIST_C)
    print(df_T.round(2).to_string(index=False))

    print()
    print("(5) Cp sensitivity (zeolite feed " + str(case["T_zeo_in_C"]) + " C) ...",
          flush=True)
    df_cp = cp_sensitivity(case, zeolite, params, criteria)
    print(df_cp.round(2).to_string(index=False))

    sheets = {"min_rate_vs_feed_T": df_T, "cp_sensitivity": df_cp}
    if out["design"] is not None:
        sheets["design_sweep"] = out["design"]["sweep"]
    path = results_to_excel(HERE / "design_results.xlsx", sheets)
    print()
    print("Saved", path, "| total time " + str(round(time.time() - t_start)) + " s")


if __name__ == "__main__":
    main()
