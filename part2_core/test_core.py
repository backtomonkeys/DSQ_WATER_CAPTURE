"""
test_core.py - assert tests for Part 2.

Run:  python test_core.py      (or: pytest)

unit        : screw rpm, Ergun, stub feed gas, out-of-range handling, regimes
balances    : the water and energy balances close
numerics    : halving the height step changes results < 1%; analytic solutions;
              cross-check with scipy solve_bvp (needs scipy)
physics     : feed temperature trend, outlet limited by equilibrium at the top
regression  : stored results of the base cases (update on purpose if the model changes)
integration : the real Part 1 (zeolite.py, feed_gas.py, parameters.csv) - FAILS if missing
"""

import math
import os
import sys
import tempfile

import numpy as np

import core_column as cc
import design as dz
import gas_props
import property_tables
import screw
import zeolite_stub as zs

PARAMS = dz.load_params()
CRIT = dz.default_criteria(PARAMS)
BASE = dz.updated_case(dz.default_case(PARAMS), {"H_m": 1.0})   # D 0.3, H 1, 1500 kg/h
PREVIEW = zs.Zeolite3ASipsPreview()
STUB = zs.Zeolite3A()
# No test is ever skipped: a missing Part 1 folder or scipy is a FAILED test with a
# message that says what to install or where to put the files.
PART1_MISSING = ("Part 1 not found: put zeolite.py, feed_gas.py, parameters.csv in "
                 + os.path.abspath(dz.PART1_DIR) + " (the folder next to part2_core)")


def need_part1():
    """Stop the test with a clear message when the Part 1 files are not there."""
    assert dz.PART1_ZEOLITE_FOUND, PART1_MISSING + " - zeolite.py missing"
    assert dz.PART1_FEED_FOUND, PART1_MISSING + " - feed_gas.py missing"
    assert os.path.isfile(os.path.join(dz.PART1_DIR, "parameters.csv")), \
        PART1_MISSING + " - parameters.csv missing"


def run(changes, zeolite=PREVIEW, n_steps=None):
    """Run check mode on BASE with some inputs changed."""
    case = dz.updated_case(BASE, changes)
    return dz.run_check(case, zeolite, PARAMS, CRIT, gas_source="stub", n_steps=n_steps)


def guarded_col(zeolite):
    """A small 'col' dictionary so the out-of-range functions can be tested alone."""
    col = {}
    col["limits_q"] = dict(cc.DEFAULT_LIMITS["loading"])
    col["limits_k"] = dict(cc.DEFAULT_LIMITS["k_ldf"])
    col["E_iso_shift_Jmol"] = cc.DEFAULT_LIMITS["E_iso_shift_Jmol"]
    col["Ea_kldf_Jmol"] = cc.DEFAULT_LIMITS["Ea_kldf_Jmol"]
    col["loading_fn"] = zeolite.loading
    col["kldf_fn"] = zeolite.k_ldf
    col["isotherm_model"] = "sips"
    col["policy"] = "clip"
    return col


# --------------------------------------------------------------------------- unit
def test_screw_rpm_hand_value():
    # hand calc: Q = 1500/3600/713 = 5.844e-4 m3/s; swept = pi/4*(0.3^2-0.075^2)*0.3
    # = 0.019880 m3/rev; rpm = 60*Q/(swept*0.9) = 1.960
    rpm = screw.screw_rpm(1500 / 3600 / 713.0, 0.3, 0.075, 0.3, 0.9)
    assert abs(rpm - 1.960) < 0.002


def test_screw_rejects_bad_eta():
    for eta in [0.0, 1.2]:
        try:
            screw.screw_rpm(1e-3, 0.3, 0.075, 0.3, eta)
            raised = False
        except ValueError:
            raised = True
        assert raised, "eta = " + str(eta) + " should raise ValueError"


def test_ergun_hand_value():
    # hand calc: viscous 163.9 + inertial 395.8 = 559.7 Pa/m
    dpdz = cc.ergun_dpdz_Pa_m(0.078, 1.8e-5, 9.87, 0.38, 0.003)
    assert abs(dpdz - 559.7) < 0.5


def test_cp_interpolation():
    gas_data = gas_props.load_gas_data()
    assert gas_props.cp_JmolK("H2O", 500.0, gas_data) == 35.22
    assert abs(gas_props.cp_JmolK("H2O", 550.0, gas_data) - 35.77) < 1e-9
    assert gas_props.cp_JmolK("H2O", 900.0, gas_data) == 37.50


def test_stub_feed_gas_matches_spec_and_atoms():
    # base case of claude/water-capture-code-spec.md: X 30 %, S 60 %
    gas = zs.inlet_gas(X=0.30, S=0.60, P_bar=30.0, T_C=250.0)
    assert abs(gas["F_total_mol_s"] - 3.682) < 1e-3
    assert abs(gas["y"]["H2O"] - 0.1071) < 1e-4
    assert abs(gas["p_H2O_bar"] - 3.21) < 0.01
    assert abs(gas["water_kg_h"] - 25.58) < 0.01
    assert abs(gas["M_mix_kg_mol"] - 0.01375) < 5e-5
    assert abs(gas["m_gas_kg_s"] - 0.0506) < 2e-4
    assert abs(gas["fresh_H2_kg_day"] - 137.4) < 0.1
    assert abs(gas["DME_kg_day"] - 523.4) < 0.1
    assert abs(gas["F_mol_s"]["CO"] - 0.1214) < 1e-4
    # total water is fixed by the fresh CO2 (1.5 mol H2O per mol CO2), whatever X and S
    for X, S in [(0.30, 0.60), (0.15, 1.0), (0.45, 0.30)]:
        assert abs(zs.inlet_gas(X=X, S=S)["water_kg_h"] - 25.58) < 0.01
    # atom balance of one pass: feed (k CO2 + 3k H2) -> outlet
    for X, S in [(0.30, 0.60), (0.15, 1.0), (0.45, 0.0)]:
        F = zs.inlet_gas(X=X, S=S)["F_mol_s"]
        k = F["CO2"] / (1.0 - X)                     # mol/s CO2 fed in this pass
        C_out = F["CO2"] + F["CO"] + 2 * F["DME"]
        H_out = 2 * F["H2"] + 6 * F["DME"] + 2 * F["H2O"]
        O_out = 2 * F["CO2"] + F["CO"] + F["DME"] + F["H2O"]
        assert abs(k - C_out) < 1e-12
        assert abs(6 * k - H_out) < 1e-12
        assert abs(2 * k - O_out) < 1e-12


def test_stub_follows_contract():
    for zeolite in [STUB, PREVIEW]:
        assert zeolite.heat_of_adsorption() == -60100.0
        assert zeolite.heat_of_adsorption(method="constant") == -45950.0
        assert zeolite.rho_bed_kg_m3 == 640.0
        assert zeolite.eps_bed == 0.372
        assert zeolite.cp_J_kgK == 960.0
        assert zeolite.loading(523.15, 1.0) > 0 and zeolite.k_ldf(523.15, 1.0) > 0
        assert abs(zeolite.rho_pellet_kg_m3 - 1019.1) < 0.1
        assert zeolite.M_H2O_kg_mol == 0.01801528
    assert issubclass(zs.OutOfRangeError, ValueError)
    assert STUB.loading(523.15, 1.0, model="gsta") == 0.1
    assert zs.Zeolite3A(allow_extrapolation=True).loading(400.0, 0.01) == 0.1
    q = PREVIEW.loading(523.15, 1.3)
    assert abs(PREVIEW.equilibrium_pressure(q, 523.15) - 1.3) < 1e-9


def test_preview_isotherm_check_value():
    # part1 brief: Sips at 250 C and 1 bar = 0.075 kg/kg
    assert abs(PREVIEW.loading(523.15, 1.0) - 0.0753) < 5e-4


def test_out_of_range_policies():
    col = guarded_col(STUB)
    col["policy"] = "raise"
    try:
        cc.qstar_safe(423.15, 1.0, col)
        raised = False
    except zs.OutOfRangeError:
        raised = True
    assert raised, "the raise policy must pass OutOfRangeError on"

    col = guarded_col(PREVIEW)
    # Henry's law below 0.05 bar: half the pressure gives half the loading
    q_half = cc.qstar_safe(523.15, 0.025, col)
    q_edge = cc.qstar_safe(523.15, 0.05, col)
    assert abs(q_half - 0.5 * q_edge) < 1e-12
    # the temperature shift gives the exact Sips value at 650 K, 1 bar
    assert abs(cc.qstar_safe(650.0, 1.0, col) - 0.0183818) < 1e-6
    assert cc.qstar_safe(423.15, 1.0, col) > cc.qstar_safe(473.15, 1.0, col)
    assert cc.kldf_safe(423.15, 1.0, col) < cc.kldf_safe(473.15, 1.0, col)


def test_henry_branch_is_continuous_at_p_min():
    # K_H = q_Sips(T, 0.05 bar) / 0.05 at the local T: no jump at 0.05 bar
    col = guarded_col(PREVIEW)
    for T_C in [200.0, 225.0, 250.0, 300.0]:
        T_K = T_C + 273.15
        q_below = cc.qstar_safe(T_K, 0.05 * (1 - 1e-9), col)
        q_at = cc.qstar_safe(T_K, 0.05, col)
        q_above = cc.qstar_safe(T_K, 0.05 * (1 + 1e-9), col)
        assert abs(q_below - q_at) / q_at < 1e-8, T_C
        assert abs(q_above - q_at) / q_at < 1e-8, T_C
        # the slope in the Henry region is K_H of THIS temperature
        K_H = PREVIEW.loading(min(T_K, 623.15), 0.05) / 0.05
        if T_C <= 250.0:
            assert abs(cc.qstar_safe(T_K, 0.02, col) - K_H * 0.02) < 1e-12


def test_henry_warning_on_web_page():
    res = run({"H_m": 2.0})                     # dry gas in the upper part of the core
    assert res["frac_p_extrapolated"] > 0
    found = False
    for text in res["warnings"]:
        if "Henry's-law" in text:
            found = True
    assert found


def test_regime_selection():
    assert run({"m_zeo_kgh": 200})["regime"] == "gas_carries_heat"
    assert run({})["regime"] == "solid_carries_heat"
    try:
        run({"m_zeo_kgh": 480})
        raised = False
    except cc.ThermalPinchError:
        raised = True
    assert raised, "W_solid ~ W_gas must raise ThermalPinchError"


def test_bed_data_taken_from_zeolite_model():
    res = run({})
    assert res["bed_props_source"] == "zeolite model"
    assert res["dH_ads_kJmol"] == -60.1
    assert abs(res["rho_bulk_kgm3"] - 640.0) < 1e-6


def test_isotherm_model_is_passed_on():
    class SpyZeolite:
        """Records which isotherm model the core asks for."""
        def __init__(self):
            self.preview = zs.Zeolite3ASipsPreview()
            self.models_seen = []
            self.rho_pellet_kg_m3 = self.preview.rho_pellet_kg_m3
            self.eps_bed = self.preview.eps_bed
            self.cp_J_kgK = self.preview.cp_J_kgK

        def loading(self, T_K, p_bar, model="sips"):
            if model not in self.models_seen:
                self.models_seen.append(model)
            return self.preview.loading(T_K, p_bar)

        def k_ldf(self, T_K, p_bar):
            return self.preview.k_ldf(T_K, p_bar)

        def heat_of_adsorption(self, method="sips"):
            return -60100.0

    spy = SpyZeolite()
    run({"isotherm_model": "gsta"}, zeolite=spy)
    assert spy.models_seen == ["gsta"]


class NarrowSipsZeolite:
    """Like Part 1: the Sips isotherm is only accepted at 200-250 C."""

    def __init__(self):
        self.preview = zs.Zeolite3ASipsPreview()
        self.rho_pellet_kg_m3 = self.preview.rho_pellet_kg_m3
        self.eps_bed = self.preview.eps_bed
        self.cp_J_kgK = self.preview.cp_J_kgK

    def loading(self, T_K, p_bar, model="sips"):
        if T_K > 523.15:
            raise zs.OutOfRangeError("T is outside the sips data range 200-250 degC")
        return self.preview.loading(T_K, p_bar)        # same curve for every model here

    def k_ldf(self, T_K, p_bar):
        return self.preview.k_ldf(T_K, p_bar)

    def heat_of_adsorption(self, method="sips"):
        return -60100.0


def test_data_range_is_detected():
    narrow = NarrowSipsZeolite()
    limits = dz.zeolite_limits(narrow, "sips", PARAMS)
    assert abs(limits["loading"]["T_max_K"] - 523.15) < 0.02
    assert limits["loading"]["T_min_K"] == 473.15
    assert limits["k_ldf"]["T_max_K"] == 623.15
    assert limits["loading"]["p_min_bar"] == 0.05
    assert limits["loading"]["p_max_bar"] == 4.5
    # the core runs although the bed is hotter than 250 C; the Sips shift is exact,
    # so the answer equals the preview with the full range
    hot = {"T_zeo_in_C": 250, "m_zeo_kgh": 1500}          # bed above 250 C
    res = run(hot, zeolite=narrow)
    assert abs(res["capture_frac"] - run(hot)["capture_frac"]) < 1e-4
    assert res["frac_T_extrapolated"] > 0
    assert res["T_data_range_C"] == "200.0-250.0"


def test_pick_smallest():
    import pandas as pd
    df = pd.DataFrame({"D_m": [0.2, 0.3, 0.2], "H_m": [1.0, 0.6, 0.5],
                       "volume_m3": [0.0314, 0.0424, 0.0157],
                       "feasible": [True, True, False]})
    best = dz.pick_smallest(df)
    assert best["D_m"] == 0.2 and best["H_m"] == 1.0


# --------------------------------------------------------------------------- balances
def test_water_balance_closes():
    for m_kgh in [200, 1500]:
        res = run({"m_zeo_kgh": m_kgh})
        assert abs(res["water_closure_rel"]) < 1e-6
        # independent check: integral of the adsorption rate over the height
        prof = res["profile"]
        N_integral = cc.trapezoid_integral(list(prof["N_mol_s_m"]), list(prof["z_m"]))
        captured = res["water_captured_kgh"] / 3600 / cc.M_W_KGMOL
        assert abs(N_integral - captured) / captured < 0.01


def test_energy_balance_closes():
    for changes in [{"m_zeo_kgh": 200}, {}, {"T_zeo_in_C": 180}, {"U_wall_Wm2K": 50.0}]:
        res = run(changes)
        assert abs(res["energy_closure_rel"]) < 1e-5, res["energy_closure_rel"]


# --------------------------------------------------------------------------- numerics
def test_step_halving_changes_less_than_1pct():
    for changes in [{}, {"m_zeo_kgh": 200}, {"m_zeo_kgh": 900}]:
        a = run(changes, n_steps=100)
        b = run(changes, n_steps=200)
        assert abs(a["capture_frac"] - b["capture_frac"]) / b["capture_frac"] < 0.01
        assert abs(a["T_bed_max_C"] - b["T_bed_max_C"]) / (b["T_bed_max_C"] + 273.15) < 0.01
        assert abs(a["dp_bar"] - b["dp_bar"]) / b["dp_bar"] < 0.01


def test_analytic_constant_capacity():
    # stub: q* and k are constant, so every pellet follows
    # q = q* - (q* - q_in) * exp(-k t)   (t = time in the core)
    res = run({"H_m": 0.1}, zeolite=STUB)
    assert res["profile"]["pH2O_bar"].min() > 0.05      # stays in the constant region
    q_out = 0.10 - (0.10 - BASE["q_in_kgkg"]) * math.exp(-6e-3 * res["tau_zeo_s"])
    assert abs(res["q_out_kgkg"] - q_out) / q_out < 1e-6


def test_analytic_temperature_rise():
    # no wall loss: |W_gas - W_solid| * temperature rise = heat of adsorption
    for m_kgh in [200, 1500]:
        res = run({"m_zeo_kgh": m_kgh})
        rise = res["Q_ads_kW"] * 1e3 / abs(res["dW_WK"])
        if res["regime"] == "gas_carries_heat":
            assert abs(res["T_bed_top_C"] - BASE["T_gas_in_C"] - rise) < 1e-6
        else:
            assert abs(res["T_bed_bottom_C"] - BASE["T_zeo_in_C"] - rise) < 1e-6


def test_cross_check_with_scipy_solve_bvp():
    try:
        from scipy.integrate import solve_bvp
    except ImportError:
        raise AssertionError("scipy is not installed: python -m pip install scipy")
    gas = zs.inlet_gas(0.25, 1.0)["F_mol_s"]
    gas_data = gas_props.load_gas_data()
    W_dry, cp_v = cc.gas_heat_capacity(gas, 523.15, gas_data)
    F_dry = 0.0
    for species in gas:
        if species != "H2O":
            F_dry = F_dry + gas[species]
    F_in = gas["H2O"]
    col = guarded_col(PREVIEW)
    M = cc.M_W_KGMOL

    for m_kgh in [1500, 200]:
        geom = {"D_m": 0.30, "H_m": 1.0, "d_shaft_m": 0.075, "pitch_m": 0.30}
        bed = {"rho_pellet_kgm3": 640 / (1 - 0.372), "eps_bed": 0.372, "d_pellet_m": 0.003,
               "cp_zeolite_JkgK": 960, "dH_ads_Jmol": -60100, "U_wall_Wm2K": 0.0,
               "T_wall_K": 523.15}
        op = {"gas_in_mols": gas, "T_gas_in_K": 523.15, "p_in_bar": 30.0,
              "m_zeo_kgs": m_kgh / 3600, "T_zeo_in_K": 523.15, "q_in_kgkg": 0.01}
        res = cc.solve_column(geom, op, bed, PREVIEW.loading, PREVIEW.k_ldf,
                              include_dp=False)
        m_s = op["m_zeo_kgs"]
        rhoA = 640 * cc.flow_area_m2(geom)

        def rhs(z, y):
            out = np.zeros(y.shape)
            for i in range(y.shape[1]):
                Fw, q, T = y[0, i], y[1, i], y[2, i]
                dW = W_dry - m_s * 960 + cp_v * (Fw - m_s * q / M)
                pw = max(Fw, 0.0) * 30.0 / (F_dry + max(Fw, 0.0))
                N = rhoA * cc.kldf_safe(T, pw, col) * (cc.qstar_safe(T, pw, col) - q) / M
                out[0, i] = -N
                out[1, i] = -N * M / m_s
                out[2, i] = N * 60100 / dW
            return out

        top_fixed = res["regime"] == "solid_carries_heat"

        def boundary(ya, yb):
            if top_fixed:
                T_condition = yb[2] - 523.15
            else:
                T_condition = ya[2] - 523.15
            return np.array([ya[0] - F_in, yb[1] - 0.01, T_condition])

        z = np.linspace(0, 1.0, 21)
        y_guess = np.array([np.linspace(F_in, 0.1 * F_in, 21), np.linspace(0.03, 0.01, 21),
                            np.linspace(560.0, 560.0, 21)])
        sol = solve_bvp(rhs, boundary, z, y_guess, tol=1e-5, max_nodes=20000)
        assert sol.status == 0
        capture_bvp = 1 - sol.sol(1.0)[0] / F_in
        assert abs(capture_bvp - res["capture_frac"]) < 1e-3


# --------------------------------------------------------------------------- physics
def test_colder_zeolite_feed_helps():
    hot = run({"T_zeo_in_C": 250})
    cold = run({"T_zeo_in_C": 200})
    assert cold["capture_frac"] > hot["capture_frac"]
    assert cold["T_bed_max_C"] < hot["T_bed_max_C"]


def test_outlet_not_drier_than_top_equilibrium():
    # the gas leaving the top cannot be drier than equilibrium with the feed zeolite
    # (needs q_in > 0; with q_in = 0 there is no lower limit)
    q_in = 0.02
    res = run({"q_in_kgkg": q_in})
    col = guarded_col(PREVIEW)
    T_feed_K = BASE["T_zeo_in_C"] + 273.15
    p_low, p_high = 1e-6, 4.5
    while p_high - p_low > 1e-9:                    # bisection, as in the lecture
        p_mid = 0.5 * (p_low + p_high)
        if cc.qstar_safe(T_feed_K, p_mid, col) < q_in:
            p_low = p_mid
        else:
            p_high = p_mid
    assert res["pH2O_out_bar"] >= 0.99 * p_low


def test_q_in_lowers_capture():
    # q_in > 0 means regeneration less complete than in the paper
    capture = []
    for q_in in [0.0, 0.01, 0.03, 0.05]:
        capture.append(run({"H_m": 2.0, "q_in_kgkg": q_in})["capture_frac"])
    for i in range(1, len(capture)):
        assert capture[i] < capture[i - 1]
    assert abs(capture[2] - 0.98831) < 2e-3         # q_in 0.03 -> about 98.8 %


def test_gsta_with_zero_q_in_is_flagged():
    spy = NarrowSipsZeolite()
    res = run({"isotherm_model": "gsta"}, zeolite=spy)
    found = False
    for text in res["warnings"]:
        if "GSTA with q_in = 0" in text:
            found = True
    assert found


def test_complete_capture_case_solves():
    # cold zeolite feed with q_in = 0 captures (almost) all water: the answer sits at the
    # edge of the bisection bracket and must still be found
    res = run({"H_m": 2.0, "T_zeo_in_C": 200, "m_zeo_kgh": 1500})
    assert res["capture_frac"] > 0.9999
    assert res["capture_frac"] <= 1.0
    assert abs(res["water_closure_rel"]) < 1e-6
    assert abs(res["energy_closure_rel"]) < 1e-5


def test_temperature_criterion():
    res = run({"T_zeo_in_C": 250, "m_zeo_kgh": 1500})    # bed well above 250 C
    assert res["T_bed_max_C"] > 250
    assert res["pass_temperature"] == False         # noqa: E712
    assert res["feasible"] == False                 # noqa: E712


def test_max_feed_temperature():
    case = dz.updated_case(BASE, {"H_m": 2.0, "m_zeo_kgh": 3000})
    T_feed, res = dz.max_feed_temperature(case, PREVIEW, PARAMS, CRIT, gas_source="stub")
    assert T_feed is not None
    assert 200.0 <= T_feed <= 250.0
    assert res["T_bed_max_C"] <= CRIT["T_bed_max_C"]
    # 1 C hotter must break the limit (bisection tolerance 0.5 C)
    hotter = run({"H_m": 2.0, "m_zeo_kgh": 3000, "T_zeo_in_C": T_feed + 1.0})
    assert hotter["T_bed_max_C"] > CRIT["T_bed_max_C"]


def test_more_zeolite_lowers_peak_temperature():
    a = run({"m_zeo_kgh": 1200})
    b = run({"m_zeo_kgh": 2400})
    assert b["T_bed_max_C"] < a["T_bed_max_C"]


# --------------------------------------------------------------------------- regression
# values with the interface contract (dH -60.1 kJ/mol, rho_bed 640, eps 0.372, cp 960,
# q_in = 0 on the Sips scale, spec defaults X 0.30, S 0.60, zeolite feed 200 C,
# zeolite 3000 kg/h)
REGRESSION = {
    "preview_base": [1.0, 234.524, 0.005897],           # capture, T max (C), dP (bar)
    "preview_200kgh": [0.568477, 454.265, 0.008072],
    "stub_base": [1.0, 234.524, 0.005927],
}


def test_regression_values():
    results = {
        "preview_base": run({}),
        "preview_200kgh": run({"m_zeo_kgh": 200}),
        "stub_base": run({}, zeolite=STUB),
    }
    for key in results:
        res = results[key]
        capture, T_max, dp = REGRESSION[key]
        assert abs(res["capture_frac"] - capture) < 2e-4, key
        # stored values are the peak of the marched profile (before the bottom zone
        # where the zeolite meets the inlet gas was added to T_bed_max_C)
        assert abs(res["T_profile_max_C"] - T_max) < 0.1, key
        assert res["T_bed_max_C"] >= res["T_profile_max_C"], key
        assert abs(res["dp_bar"] - dp) / dp < 0.05, key


def table_error(fn, model, table, limits, n_points):
    """Largest relative error of a PropertyTable against the direct call."""
    worst = 0.0
    for i in range(n_points):
        for j in range(n_points):
            # points between the grid nodes, over the whole range
            T_K = limits["T_min_K"] + (i + 0.37) / n_points * (limits["T_max_K"]
                                                                - limits["T_min_K"])
            lnp = math.log(limits["p_min_bar"]) + (j + 0.61) / n_points * (
                math.log(limits["p_max_bar"]) - math.log(limits["p_min_bar"]))
            p_bar = math.exp(lnp)
            if model is None:
                exact = fn(T_K, p_bar)
            else:
                exact = fn(T_K, p_bar, model)
            worst = max(worst, abs(table.value(T_K, p_bar) / exact - 1.0))
    return worst


def check_tables(zeolite):
    """Tables of q* and k_LDF agree with the zeolite object to 1e-4."""
    limits, q_fn, k_fn = dz.fast_functions(zeolite, "sips", PARAMS)
    entry = dz.TABLE_CACHE[(id(zeolite), "sips")]
    error_q = table_error(zeolite.loading, "sips", entry[2], limits["loading"], 25)
    error_k = table_error(zeolite.k_ldf, None, entry[3], limits["k_ldf"], 25)
    assert error_q < 1e-4, "q* table error " + str(error_q) + " (run check_tables.py)"
    assert error_k < 1e-4, "k_LDF table error " + str(error_k) + " (run check_tables.py)"


def test_lookup_tables_accurate():
    check_tables(PREVIEW)
    # a k_LDF with a numerical Darken factor, like Part 1 (curved in T and ln p)
    preview = zs.Zeolite3ASipsPreview(allow_extrapolation=True)
    lim = {"T_min_K": 473.15, "T_max_K": 623.15, "p_min_bar": 0.05, "p_max_bar": 4.5}
    table = property_tables.PropertyTable(darken_like_kldf, None, lim,
                                          PARAMS["table_dT_K"], PARAMS["table_dlnp"])
    assert table_error(darken_like_kldf, None, table, lim, 30) < 1e-4
    # the core result hardly changes
    a = run({"T_zeo_in_C": 200.0})
    case = dz.updated_case(BASE, {"T_zeo_in_C": 200.0})
    b = dz.run_check(case, PREVIEW, PARAMS, CRIT, gas_source="stub", use_tables=False)
    assert abs(a["capture_frac"] - b["capture_frac"]) < 1e-4
    assert abs(a["T_bed_max_C"] - b["T_bed_max_C"]) < 0.01


DARKEN_ZEOLITE = zs.Zeolite3ASipsPreview(allow_extrapolation=True)


def darken_like_kldf(T_K, p_bar):
    """Test function: Arrhenius x numerical Darken factor of the preview Sips."""
    step = 1e-4
    q_up = DARKEN_ZEOLITE.loading(T_K, p_bar * math.exp(step))
    q_down = DARKEN_ZEOLITE.loading(T_K, p_bar * math.exp(-step))
    dlnq_dlnp = (math.log(q_up) - math.log(q_down)) / (2.0 * step)
    return 5.0e-3 * math.exp(-6500.0 * (1.0 / T_K - 1.0 / 523.15)) / dlnq_dlnp


# ------------------------------------------------------- design mode (rate first)
SPEC_CASE = dz.default_case(PARAMS)      # D 0.30, H 2.0, zeolite fed at 200 C


def test_peak_T_falls_as_zeolite_rate_rises():
    m_T, res = dz.min_rate_for_temperature(SPEC_CASE, PREVIEW, PARAMS, CRIT, "stub")
    assert m_T is not None
    assert res["T_bed_max_C"] <= CRIT["T_bed_max_C"]
    below = dz.run_check(dz.updated_case(SPEC_CASE, {"m_zeo_kgh": 0.98 * m_T}), PREVIEW,
                         PARAMS, CRIT, "stub")
    assert below["T_bed_max_C"] > CRIT["T_bed_max_C"]      # it really is the lowest rate
    T_list = []
    for factor in [1.0, 1.5, 2.0, 3.0]:
        r = dz.run_check(dz.updated_case(SPEC_CASE, {"m_zeo_kgh": factor * m_T}), PREVIEW,
                         PARAMS, CRIT, "stub")
        T_list.append(r["T_bed_max_C"])
    for i in range(len(T_list) - 1):
        assert T_list[i + 1] < T_list[i]
    # the helper used inside the bisection must catch a rise
    try:
        dz.check_T_falls_with_rate([[1000.0, 240.0], [2000.0, 245.0]])
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_feed_at_250C_has_no_rate():
    # zeolite fed at the limit itself: any heat of adsorption pushes the bed above it
    case = dz.updated_case(SPEC_CASE, {"T_zeo_in_C": 250.0})
    m_T, res = dz.min_rate_for_temperature(case, PREVIEW, PARAMS, CRIT, "stub")
    assert m_T is None
    assert res["T_bed_max_C"] > CRIT["T_bed_max_C"]


def test_more_cp_needs_less_zeolite():
    df = dz.cp_sensitivity(SPEC_CASE, PREVIEW, PARAMS, CRIT, gas_source="stub")
    m = list(df["m_min_T_kgh"])
    assert m[0] > m[1] > m[2]                      # Cp x 0.8, x 1.0, x 1.2
    assert abs(df.loc[1, "change_vs_base_pct"]) < 1e-9


def test_k_ldf_factor():
    a = run({"T_zeo_in_C": 200.0})
    b = run({"T_zeo_in_C": 200.0, "k_ldf_factor": 1.0})
    c = run({"T_zeo_in_C": 200.0, "k_ldf_factor": 0.5})
    assert a["capture_frac"] == b["capture_frac"]          # default is 1.0
    assert c["capture_frac"] < a["capture_frac"]           # slower uptake, less captured
    assert abs(c["water_closure_rel"]) < 1e-6
    assert abs(c["profile"]["k_ldf_1s"].iloc[0] / a["profile"]["k_ldf_1s"].iloc[0]
               - 0.5) < 0.05                               # k used in the model is halved


def test_recommended_size_has_margin():
    out = dz.design_with_rate(SPEC_CASE, PREVIEW, PARAMS, CRIT, [0.15, 0.20, 0.25, 0.30],
                              [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0], gas_source="stub")
    rec = out["recommended"]
    assert rec is not None
    smallest = out["design"]["refined"]
    assert smallest is not None
    V_small = 3.141592653589793 / 4 * smallest["D_m"] ** 2 * smallest["H_m"]
    V_rec = 3.141592653589793 / 4 * rec["D_m"] ** 2 * rec["H_m"]
    assert V_rec >= V_small                                # never smaller
    assert rec["result_nominal"]["capture_frac"] >= CRIT["capture_min"]   # k_LDF x 1
    assert rec["result_nominal"]["feasible"]
    assert rec["result_slow"]["feasible"]                  # all five with k_LDF x 0.5
    assert rec["result_slow"]["case"]["k_ldf_factor"] == 0.5


def test_profile_plot_reaches_full_height():
    import plots
    for H_m in [0.75, 1.5, 2.0]:
        case = dz.updated_case(SPEC_CASE, {"H_m": H_m})
        res = dz.run_check(case, PREVIEW, PARAMS, CRIT, gas_source="stub")
        fig = plots.plot_core_profiles(res["profile"], res, 250.0, 0.05)
        for axis in fig.axes:
            assert axis.get_ylim()[1] >= H_m, H_m        # the axis goes up to z = H
            assert axis.get_ylim()[0] <= 0.0
        plots.plt.close(fig)


def test_residence_time_is_a_note_and_lift_is_local_max():
    res = dz.run_check(SPEC_CASE, PREVIEW, PARAMS, CRIT, gas_source="stub")
    for text in res["warnings"]:
        assert "residence" not in text                 # information, not a warning
    found = False
    for text in res["notes"]:
        if "residence time" in text and "hand rule" not in text:
            found = True
    assert found
    # the lift criterion uses the largest local value, found at the gas inlet
    assert res["lift_ratio_max"] >= res["lift_ratio_avg"]
    assert res["z_lift_max_m"] < 0.05 * SPEC_CASE["H_m"]


def test_format_capture():
    assert dz.format_capture(1.0) == "> 99.9%"
    assert dz.format_capture(0.9995) == "> 99.9%"
    assert dz.format_capture(0.999) == "99.9%"
    assert dz.format_capture(0.95) == "95.0%"


def test_no_typed_densities_and_no_chdir():
    # densities come from the zeolite object only (spec, 21 Nov 2026)
    for name in ["design.py", "app.py", "core_column.py", "core_parameters.csv"]:
        with open(os.path.join(dz.HERE, name), encoding="utf-8") as file:
            text = file.read()
        for number in ["640", "1019", "1200"]:
            assert number not in text, number + " typed in " + name
        assert "os.chdir" not in text, "os.chdir used in " + name
    assert dz.PART1_DIR.is_absolute()
    assert dz.PART1_DIR == dz.HERE.parent / "part1_zeolite"


# ------------------------------------------------------- integration with real Part 1
BASE_SPEC = dz.default_case(PARAMS)      # spec base case: D 0.30, H 2.0, 3000 kg/h


def test_part1_files_found():
    need_part1()
    old_folder = os.getcwd()
    os.chdir(tempfile.gettempdir())     # start from another folder, as streamlit may
    try:
        zeolite, used = dz.load_zeolite("part1")
        flows, gas_used = dz.get_inlet_gas(BASE_SPEC, "part1")
    finally:
        os.chdir(old_folder)
    assert used == "part1" and gas_used == "part1"
    assert type(zeolite).__module__ == "zeolite"          # the real class, not the stub


def test_part1_contract_values():
    need_part1()
    zeolite, used = dz.load_zeolite("part1")
    # claude/interface-contract.md, "Part 1 ให้อะไร"
    assert abs(zeolite.heat_of_adsorption() + 60100.0) < 1.0
    assert abs(zeolite.heat_of_adsorption(method="constant") + 45950.0) < 1.0
    assert abs(zeolite.rho_bed_kg_m3 - 640.0) < 0.5
    assert abs(zeolite.eps_bed - 0.372) < 1e-4
    assert abs(zeolite.rho_pellet_kg_m3 - 1019.0) < 1.0
    assert abs(zeolite.cp_J_kgK - 960.0) < 0.5
    k = zeolite.k_ldf(523.15, 0.5)
    assert 2e-3 < k < 1.2e-2                              # about 5e-3 1/s at 250 C
    # bed properties used by Part 2 come from the object, not from a typed number
    props = dz.bed_data(zeolite, PARAMS)
    assert props["source"] == "zeolite model"
    assert props["rho_pellet_kgm3"] == zeolite.rho_pellet_kg_m3
    assert props["eps_bed"] == zeolite.eps_bed


def test_part1_inlet_gas_base_case():
    need_part1()
    gas = dz.part1_inlet_gas_dict(BASE_SPEC)              # full dict from feed_gas.py
    assert abs(gas["F_total_mol_s"] - 3.682) < 2e-3
    assert abs(gas["y"]["H2O"] - 0.1071) < 2e-4
    assert abs(gas["p_H2O_bar"] - 3.21) < 0.01
    assert abs(gas["water_kg_h"] - 25.58) < 0.02
    flows, used = dz.get_inlet_gas(BASE_SPEC, "part1")
    assert used == "part1"
    assert abs(flows["H2O"] - 0.3945) < 5e-4


def test_part1_data_range_detected():
    need_part1()
    zeolite, used = dz.load_zeolite("part1")
    lim = dz.zeolite_limits(zeolite, "sips", PARAMS)
    assert abs(lim["loading"]["T_min_K"] - 473.15) < 0.5    # Sips 200-250 C
    assert abs(lim["loading"]["T_max_K"] - 523.15) < 0.5
    assert abs(lim["loading"]["p_min_bar"] - 0.05) < 1e-3
    assert abs(lim["loading"]["p_max_bar"] - 4.5) < 0.01


def test_part1_lookup_tables_accurate():
    need_part1()
    zeolite, used = dz.load_zeolite("part1")
    check_tables(zeolite)


def test_part1_henry_continuous_at_p_min():
    need_part1()
    zeolite, used = dz.load_zeolite("part1")
    col = guarded_col(zeolite)
    col["limits_q"] = dz.zeolite_limits(zeolite, "sips", PARAMS)["loading"]
    p_min = col["limits_q"]["p_min_bar"]
    for T_C in [200.0, 225.0, 250.0]:
        T_K = T_C + 273.15
        q_edge = zeolite.loading(T_K, p_min)
        q_henry = cc.qstar_safe(T_K, p_min * (1.0 - 1e-9), col)
        assert abs(q_henry - q_edge) / q_edge < 1e-6, T_C
        assert abs(cc.qstar_safe(T_K, 0.5 * p_min, col) - 0.5 * q_edge) / q_edge < 1e-6


def test_part1_full_core_balances():
    need_part1()
    zeolite, used = dz.load_zeolite("part1")
    res = dz.run_check(BASE_SPEC, zeolite, PARAMS, CRIT, gas_source="part1")
    assert res["gas_source"] == "part1"
    assert abs(res["water_closure_rel"]) < 1e-6
    assert abs(res["energy_closure_rel"]) < 1e-5
    # water picked up by the zeolite = water removed from the gas
    on_zeolite = BASE_SPEC["m_zeo_kgh"] * (res["q_out_kgkg"] - BASE_SPEC["q_in_kgkg"])
    assert abs(on_zeolite - res["water_captured_kgh"]) / res["water_in_kgh"] < 1e-4
    half = dz.run_check(BASE_SPEC, zeolite, PARAMS, CRIT, gas_source="part1",
                        n_steps=int(PARAMS["n_steps"] / 2))
    assert abs(half["capture_frac"] - res["capture_frac"]) < 0.01


def test_part1_minimum_rate_161():
    need_part1()
    zeolite, used = dz.load_zeolite("part1")
    flows, used = dz.get_inlet_gas(BASE_SPEC, "part1")
    m_min = dz.equilibrium_min_rate(BASE_SPEC, zeolite, flows)
    assert abs(m_min - 161.0) < 3.0                       # interface contract: ~161 kg/h
    assert BASE_SPEC["m_zeo_kgh"] > m_min
    res = dz.run_check(BASE_SPEC, zeolite, PARAMS, CRIT, gas_source="part1")
    # the zeolite can never leave wetter than equilibrium with the inlet gas
    q_eq_in = zeolite.loading(BASE_SPEC["T_zeo_in_C"] + 273.15, res["y_H2O_in"]
                              * BASE_SPEC["p_bar"])
    assert res["q_out_kgkg"] <= q_eq_in * 1.001


# --------------------------------------------------------------------------- runner
def test_lift_safety_factor_column():
    df = dz.design_sweep(BASE, PREVIEW, PARAMS, CRIT, [0.3], [1.0])
    assert abs(df.loc[0, "lift_safety_factor"] * df.loc[0, "lift_ratio_max"] - 1) < 1e-12


ALL_TESTS = [
    test_screw_rpm_hand_value, test_screw_rejects_bad_eta, test_ergun_hand_value,
    test_cp_interpolation, test_stub_feed_gas_matches_spec_and_atoms,
    test_stub_follows_contract, test_preview_isotherm_check_value,
    test_out_of_range_policies, test_henry_branch_is_continuous_at_p_min,
    test_henry_warning_on_web_page, test_regime_selection,
    test_bed_data_taken_from_zeolite_model, test_isotherm_model_is_passed_on,
    test_data_range_is_detected, test_pick_smallest, test_water_balance_closes,
    test_energy_balance_closes,
    test_step_halving_changes_less_than_1pct, test_analytic_constant_capacity,
    test_analytic_temperature_rise, test_cross_check_with_scipy_solve_bvp,
    test_colder_zeolite_feed_helps, test_outlet_not_drier_than_top_equilibrium,
    test_q_in_lowers_capture, test_gsta_with_zero_q_in_is_flagged,
    test_complete_capture_case_solves, test_temperature_criterion,
    test_max_feed_temperature,
    test_more_zeolite_lowers_peak_temperature, test_regression_values,
    test_lift_safety_factor_column, test_peak_T_falls_as_zeolite_rate_rises,
    test_feed_at_250C_has_no_rate, test_more_cp_needs_less_zeolite, test_format_capture,
    test_k_ldf_factor, test_recommended_size_has_margin,
    test_profile_plot_reaches_full_height, test_residence_time_is_a_note_and_lift_is_local_max,
    test_no_typed_densities_and_no_chdir, test_lookup_tables_accurate,
    test_part1_files_found, test_part1_contract_values, test_part1_inlet_gas_base_case,
    test_part1_data_range_detected, test_part1_lookup_tables_accurate,
    test_part1_henry_continuous_at_p_min,
    test_part1_full_core_balances, test_part1_minimum_rate_161,
]

if __name__ == "__main__":
    n_passed = 0
    n_failed = 0
    for test in ALL_TESTS:                       # functions in a list, called one by one
        try:
            test()
            n_passed = n_passed + 1
            print("PASS ", test.__name__)
        except AssertionError as error:
            n_failed = n_failed + 1
            print("FAIL ", test.__name__, error)
        except Exception as error:               # any other error: report, keep going
            n_failed = n_failed + 1
            print("ERROR", test.__name__, type(error).__name__ + ":", error)
    print()
    print(str(n_passed) + "/" + str(len(ALL_TESTS)) + " passed, " + str(n_failed)
          + " failed (no test is skipped)")
