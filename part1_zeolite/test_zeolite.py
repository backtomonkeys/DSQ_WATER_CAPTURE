"""
test_zeolite.py - Tests for Part 1 (zeolite.py and feed_gas.py).

Run from the part1_zeolite folder:
    python test_zeolite.py

Unit tests       does one function work?
Integration      do zeolite.py and feed_gas.py work together?
Regression       did a change break old results?
Step halving     does the answer stay the same with half the numerical step?
Contract         do names, defaults and base-case values match claude/interface-contract.md?
"""

import numpy as np

from feed_gas import M_kg_mol, atom_flows_mol_s, fresh_co2_mol_s, inlet_gas
from zeolite import (OutOfRangeError, Zeolite3A, capillary_kgkg, capillary_mol_kg,
                     capillary_table, darken_factor, equilibrium_pressure_bar,
                     gsta_capillary_kgkg, isosteric_heat_J_mol, sips_kgkg, water_psat_bar)

T200_K = 473.15
T250_K = 523.15
zeo = Zeolite3A("parameters.csv")
zeo_ext = Zeolite3A("parameters.csv", allow_extrapolation=True)
params = zeo.params


# ================================================================== unit tests
def test_psat_iapws():
    # IAPWS-IF97 verification values (Table 35), MPa
    assert abs(water_psat_bar(300) / 10 - 0.353658941e-2) < 1e-11
    assert abs(water_psat_bar(500) / 10 - 0.263889776e1) < 1e-8
    assert abs(water_psat_bar(600) / 10 - 0.123443146e2) < 1e-7


def test_parameters_from_csv():
    assert params["sips_qs_kgkg"] == 0.21
    assert params["sips_n"] == 1.53
    assert params["gsta_qs_kgkg"] == 0.255
    assert params["cp_pellet_J_kgK"] == 960
    T_list_K, muRT_list = capillary_table(params)
    assert T_list_K == [473.15, 523.15, 573.15, 623.15]
    assert muRT_list == [2.63, 1.67, 0.849, 0.196]


def test_capillary_unit_mol_kg_to_kg_kg():
    # Table 4 is in mol/kg; it must be multiplied by M_H2O before adding to kg/kg
    q_mol_kg = capillary_mol_kg(T250_K, 1.0, params)
    q_kgkg = capillary_kgkg(T250_K, 1.0, params)
    assert abs(q_kgkg - q_mol_kg * 0.01801528) < 1e-15
    assert 0.3 < q_mol_kg < 0.6
    assert q_kgkg < 0.02
    # adding the mol/kg number by mistake would give about 0.52 kg/kg instead of 0.079
    assert zeo.loading(T250_K, 1.0, "gsta_cap") < 0.1


def test_capillary_table_points():
    T_list_K, muRT_list = capillary_table(params)
    for i in range(len(T_list_K)):
        ln_ratio = np.log(water_psat_bar(T_list_K[i]) / 1.0)
        assert abs(capillary_mol_kg(T_list_K[i], 1.0, params) * ln_ratio - muRT_list[i]) < 1e-12


def test_isotherms_rise_with_p_and_fall_with_T():
    for model in ["sips", "gsta", "gsta_cap"]:
        q_previous = 0.0
        for p_bar in np.arange(0.05, 4.501, 0.05):
            q = zeo.loading(T250_K, p_bar, model)
            assert q > q_previous, model + ": q must rise with p"
            q_previous = q
        T_max_K = 523.15 if model == "sips" else 623.15
        q_previous = 1.0
        for T_K in np.arange(473.15, T_max_K + 0.01, 5.0):
            q = zeo.loading(T_K, 1.0, model)
            assert q < q_previous, model + ": q must fall with T"
            q_previous = q


def test_sips_darken_matches_formula():
    # Sips: Gamma = d ln p / d ln q = n (1 + x)
    n = params["sips_n"]
    for p_bar in [0.1, 0.5, 1.0, 3.0]:
        b = params["sips_b0_1_bar"] * np.exp(-params["sips_Ea_J_mol"] / (params["R_J_molK"] * T250_K))
        x = (b * p_bar) ** (1 / n)
        gamma = darken_factor(sips_kgkg, T250_K, p_bar, params)
        assert abs(gamma / (n * (1 + x)) - 1) < 1e-7


def test_sips_isosteric_heat_equals_Ea():
    for p_bar in [0.2, 1.0, 4.0]:
        dH = zeo.heat_of_adsorption(T250_K, p_bar, "isosteric_sips")
        assert abs(dH / params["sips_Ea_J_mol"] - 1) < 1e-5


def test_heat_of_adsorption_methods():
    assert zeo.heat_of_adsorption() == -60100         # default = Sips (team decision)
    assert zeo.heat_of_adsorption(method="constant") == -45950
    dH = zeo.heat_of_adsorption(T250_K, 1.0, "isosteric_gsta_cap")
    assert -80e3 < dH < -30e3


def test_out_of_range():
    for T_K, p_bar in [(150 + 273.15, 1.0), (360 + 273.15, 1.0), (T250_K, 0.01), (T250_K, 5.0)]:
        try:
            zeo.loading(T_K, p_bar)
        except OutOfRangeError:
            pass
        else:
            assert False, "no OutOfRangeError at T = " + str(T_K) + " K, p = " + str(p_bar) + " bar"
    zeo.loading(473.15, 0.05)                          # range edges are allowed
    zeo.loading(523.15, 4.5)
    zeo.loading(623.15, 4.5, "gsta_cap")
    zeo.k_ldf(623.15, 4.5)
    try:                                               # OutOfRangeError is also a ValueError
        zeo.loading(T250_K, 5.0)
    except ValueError:
        pass


def test_sips_range_200_to_250C():
    # Sips was fitted on 200 and 250 degC data only (RCE 2021, Fig. 3)
    try:
        zeo.loading(260 + 273.15, 1.0, "sips")
    except OutOfRangeError:
        pass
    else:
        assert False, "Sips above 250 degC must raise OutOfRangeError"
    zeo.loading(300 + 273.15, 1.0, "gsta_cap")       # GSTA is valid up to 350 degC
    zeo.k_ldf(300 + 273.15, 1.0)
    assert zeo_ext.loading(300 + 273.15, 1.0, "sips") > 0


def test_allow_extrapolation():
    assert zeo_ext.loading(150 + 273.15, 1.0) > zeo.loading(T200_K, 1.0)
    assert zeo_ext.loading(T250_K, 0.0) == 0.0
    assert zeo_ext.k_ldf(T250_K, 0.0) > 0


def test_unknown_model():
    try:
        zeo.loading(T250_K, 1.0, "langmuir")
    except ValueError:
        pass
    else:
        assert False, "unknown model must raise ValueError"
    try:
        zeo.heat_of_adsorption(method="guess")
    except ValueError:
        pass
    else:
        assert False, "unknown method must raise ValueError"


def test_missing_parameter_file():
    try:
        Zeolite3A("no_such_file.csv")
    except FileNotFoundError:
        pass
    else:
        assert False, "a missing file must raise FileNotFoundError"


def test_equilibrium_pressure_roundtrip():
    for model in ["sips", "gsta_cap"]:
        for p_bar in [0.07, 0.5, 2.0, 4.4]:
            q = zeo.loading(T250_K, p_bar, model)
            assert abs(zeo.equilibrium_pressure(q, T250_K, model) / p_bar - 1) < 1e-8


def test_equilibrium_pressure_vs_sips_formula():
    # Sips can be inverted by hand: x = q/(qs - q), p = x^n / b
    q = 0.1
    b = params["sips_b0_1_bar"] * np.exp(-params["sips_Ea_J_mol"] / (params["R_J_molK"] * T250_K))
    x = q / (params["sips_qs_kgkg"] - q)
    p_formula = x ** params["sips_n"] / b
    assert abs(zeo.equilibrium_pressure(q, T250_K) / p_formula - 1) < 1e-8


# ================================== check values (papers + team hand calculation)
def test_sips_check_value():
    assert abs(zeo.loading(T250_K, 1.0, "sips") - 0.075) < 0.001


def test_gsta_check_values():
    expected = {0.5: 0.056, 1.0: 0.079, 2.0: 0.103, 3.0: 0.117}
    for p_bar, q_expected in expected.items():
        assert abs(zeo.loading(T250_K, p_bar, "gsta_cap") - q_expected) < 0.001, p_bar


def test_sips_vs_gsta_within_10_percent():
    # holds at 250 degC for pH2O >= 0.2 bar (about 23 % apart at 0.05 bar)
    for p_bar in np.arange(0.2, 4.501, 0.1):
        q_sips = zeo.loading(T250_K, p_bar, "sips")
        q_gsta = zeo.loading(T250_K, p_bar, "gsta_cap")
        assert abs(q_sips / q_gsta - 1) < 0.10


def test_k_ldf_check_values():
    assert 4.0e-3 < zeo.k_ldf(T250_K, 0.5) < 6.0e-3
    assert 8.0e-3 < zeo.k_ldf(T250_K, 3.0) < 1.0e-2
    assert abs(zeo.k_ldf(T200_K, 0.5) / 2.7e-3 - 1) < 0.10    # Fig. 9 of the paper


def test_feed_hand_calculation():
    # spec: water ~25.6 kg/h, ~3.7 mol/s into the core, ~10.7 mol% H2O at X = 0.30, S = 0.60
    gas = inlet_gas(X=0.30, S=0.60)
    assert abs(gas["water_kg_h"] / 25.6 - 1) < 0.01
    assert abs(gas["F_total_mol_s"] / 3.7 - 1) < 0.02
    assert 0.10 <= gas["y"]["H2O"] <= 0.11


def test_feed_default_base_case():
    gas = inlet_gas()
    assert gas["X"] == 0.30 and gas["S"] == 0.60 and gas["P_bar"] == 30.0
    assert abs(gas["p_H2O_bar"] - 3.214) < 0.001


def test_feed_atom_balance():
    for X in [0.1, 0.25, 0.5, 1.0]:
        for S in [0.0, 0.5, 0.9, 1.0]:
            for basis in ["recycle", "once_through"]:
                gas = inlet_gas(X=X, S=S, basis=basis)
                atoms_in = atom_flows_mol_s(gas["feed_mol_s"])
                atoms_out = atom_flows_mol_s(gas["F_mol_s"])
                for i in range(3):                     # C, H, O
                    assert abs(atoms_out[i] / atoms_in[i] - 1) < 1e-12


def test_plant_balance_with_recycle():
    # full recycle: fresh CO2 + fresh H2 -> DME + H2O over the plant
    F_fresh = fresh_co2_mol_s()
    water_kg_h = 1.5 * F_fresh * M_kg_mol["H2O"] * 3600
    for X in [0.1, 0.3, 0.5]:
        for S in [0.0, 0.6, 1.0]:
            gas = inlet_gas(X=X, S=S)
            fresh = {"CO2": gas["fresh_CO2_mol_s"], "H2": gas["fresh_H2_mol_s"]}
            products = {"DME": gas["DME_kg_day"] / M_kg_mol["DME"] / 86400, "H2O": gas["water_mol_s"]}
            atoms_fresh = atom_flows_mol_s(fresh)
            atoms_products = atom_flows_mol_s(products)
            for i in range(3):
                assert abs(atoms_products[i] / atoms_fresh[i] - 1) < 1e-12
            # water load is fixed by fresh CO2 (1.5 mol H2O per mol CO2), not by X or S
            assert abs(gas["water_kg_h"] / water_kg_h - 1) < 1e-12


def test_mole_fractions_sum_to_one():
    gas = inlet_gas(X=0.35, S=0.7, P_bar=30.0)
    total = 0.0
    for value in gas["y"].values():
        total = total + value
    assert abs(total - 1) < 1e-12
    assert abs(gas["p_H2O_bar"] - gas["y"]["H2O"] * 30.0) < 1e-12


def test_feed_bad_inputs():
    # (X, S, P_bar, basis, H2_CO2_ratio)
    bad_inputs = [(0.0, 1.0, 30.0, "recycle", 3.0),
                  (1.2, 1.0, 30.0, "recycle", 3.0),
                  (0.25, -0.1, 30.0, "recycle", 3.0),
                  (0.25, 1.1, 30.0, "recycle", 3.0),
                  (0.25, 1.0, 0.0, "recycle", 3.0),
                  (0.25, 1.0, 30.0, "partial", 3.0),
                  (1.0, 1.0, 30.0, "recycle", 2.0)]     # not enough H2
    for X, S, P_bar, basis, ratio in bad_inputs:
        try:
            inlet_gas(X=X, S=S, P_bar=P_bar, basis=basis, H2_CO2_ratio=ratio)
        except ValueError:
            pass
        else:
            assert False, "inlet_gas must raise ValueError for " + str((X, S, P_bar, basis, ratio))


# ======================================================== step-halving tests
def test_darken_step_halving():
    for model in ["sips", "gsta_cap"]:
        f = zeo.isotherm(model)
        g1 = darken_factor(f, T250_K, 1.0, params, step=1e-4)
        g2 = darken_factor(f, T250_K, 1.0, params, step=0.5e-4)
        assert abs(g2 / g1 - 1) < 1e-6


def test_isosteric_heat_step_halving():
    dH1 = isosteric_heat_J_mol(gsta_capillary_kgkg, 530.0, 1.0, params, step=1e-4)
    dH2 = isosteric_heat_J_mol(gsta_capillary_kgkg, 530.0, 1.0, params, step=0.5e-4)
    assert abs(dH2 / dH1 - 1) < 1e-6


def test_bisection_tolerance_halving():
    p1 = equilibrium_pressure_bar(sips_kgkg, 0.09, T250_K, params, 0.05, 4.5, tolerance=1e-8)
    p2 = equilibrium_pressure_bar(sips_kgkg, 0.09, T250_K, params, 0.05, 4.5, tolerance=0.5e-8)
    assert abs(p2 / p1 - 1) < 1e-8


# ========================================================== regression tests
def test_regression_values():
    # results of the first version (21 Nov 2026); a change here means the model changed
    # (feed values updated 21 Nov 2026 when the recycle basis was aligned with the spec)
    assert abs(zeo_ext.loading(573.15, 2.0, "sips") / 0.059989234816492085 - 1) < 1e-9
    assert abs(zeo.loading(573.15, 2.0, "gsta_cap") / 0.06304228109211267 - 1) < 1e-9
    assert abs(zeo.loading(473.15, 0.3, "gsta") / 0.07400642821092536 - 1) < 1e-9
    assert abs(zeo.k_ldf(523.15, 1.0) / 0.006319745713823624 - 1) < 1e-6
    assert abs(zeo.k_ldf(623.15, 4.0) / 0.026399899920986028 - 1) < 1e-6
    gas = inlet_gas(X=0.3, S=0.8, P_bar=30.0)
    assert abs(gas["F_total_mol_s"] / 3.3061644759119164 - 1) < 1e-12
    assert abs(gas["water_kg_h"] / 25.58436246719458 - 1) < 1e-12


# ========================================================== integration test
def test_integration_feed_into_zeolite():
    gas = inlet_gas(X=0.30, S=0.60, P_bar=30.0)       # spec base case
    q_in = zeo.loading(T250_K, gas["p_H2O_bar"])
    k_in = zeo.k_ldf(T250_K, gas["p_H2O_bar"])
    m_zeolite_min_kg_h = gas["water_kg_h"] / q_in     # equilibrium lower bound
    assert 150 < m_zeolite_min_kg_h < 300
    assert 5e-3 < k_in < 2e-2
    # at 30 bar and S = 0.6, X > 0.405 pushes pH2O above 4.5 bar -> the model must refuse
    gas_high = inlet_gas(X=0.50, S=0.60, P_bar=30.0)
    assert gas_high["p_H2O_bar"] > params["p_max_bar"]
    try:
        zeo.loading(T250_K, gas_high["p_H2O_bar"])
    except OutOfRangeError:
        pass
    else:
        assert False, "expected OutOfRangeError for pH2O > 4.5 bar"


# ============================= contract tests (claude/interface-contract.md)
def test_contract_attributes():
    # densities agreed 21 Nov 2026: van Kampen 2021 Table 1, not 1200 kg/m3
    assert zeo.rho_bed_kg_m3 == 640
    assert abs(zeo.rho_pellet_kg_m3 - 1019) < 1
    assert zeo.eps_bed == 0.372
    assert zeo.d_pellet_m == 1.6e-3
    assert zeo.cp_J_kgK == 960
    assert zeo.M_H2O_kg_mol == 0.01801528


def test_contract_base_case_table():
    gas = inlet_gas()                                  # defaults must be the base case
    assert gas["X"] == 0.30 and gas["S"] == 0.60 and gas["basis"] == "recycle"
    assert gas["P_bar"] == 30.0 and abs(gas["T_K"] - 523.15) < 1e-9
    assert abs(gas["fresh_CO2_mol_s"] * 44.0095e-3 * 86400 - 1000) < 1e-6        # 1 t CO2/day
    assert abs(gas["feed_mol_s"]["H2"] / gas["feed_mol_s"]["CO2"] - 3.0) < 1e-12  # H2/CO2 = 3
    assert round(gas["F_total_mol_s"], 3) == 3.682
    assert round(gas["F_mol_s"]["H2O"], 4) == 0.3945
    assert round(gas["F_mol_s"]["CO"], 4) == 0.1214
    assert round(gas["y"]["H2O"], 4) == 0.1071
    assert round(gas["p_H2O_bar"], 2) == 3.21
    assert round(gas["water_kg_h"], 2) == 25.58
    assert round(gas["M_mix_kg_mol"], 5) == 0.01375
    assert round(gas["m_gas_kg_s"], 4) == 0.0506
    assert round(gas["fresh_H2_kg_day"], 1) == 137.4
    assert round(gas["DME_kg_day"], 1) == 523.4


def test_contract_pH2O_limits():
    # pH2O at the core inlet passes 4.5 bar when X > 0.41 (30 bar) or X > 0.31 (40 bar), S = 0.6
    assert inlet_gas(X=0.40, S=0.60, P_bar=30.0)["p_H2O_bar"] <= 4.5
    assert inlet_gas(X=0.41, S=0.60, P_bar=30.0)["p_H2O_bar"] > 4.5
    assert inlet_gas(X=0.31, S=0.60, P_bar=40.0)["p_H2O_bar"] <= 4.5
    assert inlet_gas(X=0.32, S=0.60, P_bar=40.0)["p_H2O_bar"] > 4.5


def test_contract_errors():
    # k_ldf and GSTA: 200-350 degC; outside -> OutOfRangeError
    for T_K in [473.15 - 1, 623.15 + 1]:
        try:
            zeo.k_ldf(T_K, 1.0)
        except OutOfRangeError:
            pass
        else:
            assert False, "k_ldf must raise OutOfRangeError at T = " + str(T_K) + " K"
    # pH2O below 0.05 bar -> OutOfRangeError (Part 2 then uses Henry's law)
    try:
        zeo.loading(T250_K, 0.049)
    except OutOfRangeError:
        pass
    else:
        assert False, "pH2O < 0.05 bar must raise OutOfRangeError"
    # p < 0 and p >= saturation pressure -> ValueError, even with allow_extrapolation
    for p_bar in [-0.1, water_psat_bar(T250_K)]:
        try:
            zeo_ext.loading(T250_K, p_bar)
        except OutOfRangeError:
            assert False, "this must be a plain ValueError, not OutOfRangeError"
        except ValueError:
            pass
        else:
            assert False, "p = " + str(p_bar) + " bar must raise ValueError"


def test_contract_min_circulation():
    # equilibrium lower bound with q_in = 0: about 161 kg/h (200 degC) and 223 kg/h (250 degC)
    gas = inlet_gas()
    assert round(gas["water_kg_h"] / zeo.loading(473.15, gas["p_H2O_bar"])) == 161
    assert round(gas["water_kg_h"] / zeo.loading(523.15, gas["p_H2O_bar"])) == 223


# ===================================================================== runner
tests = {
    "psat_iapws": test_psat_iapws,
    "parameters_from_csv": test_parameters_from_csv,
    "capillary_unit_mol_kg_to_kg_kg": test_capillary_unit_mol_kg_to_kg_kg,
    "capillary_table_points": test_capillary_table_points,
    "isotherms_rise_with_p_and_fall_with_T": test_isotherms_rise_with_p_and_fall_with_T,
    "sips_darken_matches_formula": test_sips_darken_matches_formula,
    "sips_isosteric_heat_equals_Ea": test_sips_isosteric_heat_equals_Ea,
    "heat_of_adsorption_methods": test_heat_of_adsorption_methods,
    "out_of_range": test_out_of_range,
    "sips_range_200_to_250C": test_sips_range_200_to_250C,
    "allow_extrapolation": test_allow_extrapolation,
    "unknown_model": test_unknown_model,
    "missing_parameter_file": test_missing_parameter_file,
    "equilibrium_pressure_roundtrip": test_equilibrium_pressure_roundtrip,
    "equilibrium_pressure_vs_sips_formula": test_equilibrium_pressure_vs_sips_formula,
    "sips_check_value": test_sips_check_value,
    "gsta_check_values": test_gsta_check_values,
    "sips_vs_gsta_within_10_percent": test_sips_vs_gsta_within_10_percent,
    "k_ldf_check_values": test_k_ldf_check_values,
    "feed_hand_calculation": test_feed_hand_calculation,
    "feed_default_base_case": test_feed_default_base_case,
    "feed_atom_balance": test_feed_atom_balance,
    "plant_balance_with_recycle": test_plant_balance_with_recycle,
    "mole_fractions_sum_to_one": test_mole_fractions_sum_to_one,
    "feed_bad_inputs": test_feed_bad_inputs,
    "darken_step_halving": test_darken_step_halving,
    "isosteric_heat_step_halving": test_isosteric_heat_step_halving,
    "bisection_tolerance_halving": test_bisection_tolerance_halving,
    "regression_values": test_regression_values,
    "integration_feed_into_zeolite": test_integration_feed_into_zeolite,
    "contract_attributes": test_contract_attributes,
    "contract_base_case_table": test_contract_base_case_table,
    "contract_pH2O_limits": test_contract_pH2O_limits,
    "contract_errors": test_contract_errors,
    "contract_min_circulation": test_contract_min_circulation,
}

passed = 0
for name, run_test in tests.items():
    try:
        run_test()
        print("PASS", name)
        passed = passed + 1
    except AssertionError:
        print("FAIL", name)
print(passed, "/", len(tests), "tests passed")
