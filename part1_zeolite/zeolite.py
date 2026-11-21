"""
zeolite.py - Water (steam) adsorption on zeolite 3A pellets.
Part 1 of the DSQ water-capture code (CPE 100 project).

Models
------
Sips working isotherm (MAIN model)
    van Kampen et al., React. Chem. Eng. 2021, 6, 244-257, Eqs. 17-18, Table 3
Dual-site GSTA + capillary condensation (comparison, and Darken factor)
    van Kampen et al., Adsorption 2021, 27, 577-589, Eqs. 2, 7, 9, Tables 1, 3, 4
Linear driving force (LDF): micropore diffusion x Darken factor
    van Kampen et al., Adsorption 2021, Eqs. 10, 16, 17, Table 5

Units are written in every variable name:
    T_K       temperature (K)
    p_bar     H2O PARTIAL pressure (bar), not the total pressure
    q_kgkg    loading (kg H2O per kg pellet)
    k_1_s     LDF rate constant (1/s)
    dH_J_mol  heat of adsorption (J/mol H2O), negative = heat is released

Validity (data in the papers)
    Sips:                 200-250 degC  (fitted on 200 and 250 degC data only)
    GSTA, capillary, LDF: 200-350 degC
    all models:           pH2O 0.05-4.5 bar

Sips is a WORKING-capacity isotherm (loading gained per cycle after regeneration),
so zeolite fed to the core has q = 0 on the Sips scale. GSTA + capillary is a true
equilibrium isotherm: use it for comparison only.
"""

import numpy as np
import pandas as pd


class OutOfRangeError(ValueError):
    """T or pH2O is outside the range covered by the paper data."""


# ------------------------------------------------------------------ parameters
def load_parameters(filename):
    """Read parameters.csv (name,value,unit,description,source) into a dict {name: value}."""
    table = pd.read_csv(filename)
    params = {}
    for i in range(len(table)):
        params[table.loc[i, "name"]] = float(table.loc[i, "value"])
    return params


def capillary_table(params):
    """Return two lists (T_K, muRT_mol_kg) from the rows cap_muRT_mol_kg_T<T in K>."""
    prefix = "cap_muRT_mol_kg_T"
    T_list_K = []
    muRT_list_mol_kg = []
    for name, value in params.items():              # rows are in increasing T in the CSV
        if name.startswith(prefix):
            T_list_K.append(float(name[len(prefix):]))
            muRT_list_mol_kg.append(value)
    return T_list_K, muRT_list_mol_kg


def interpolate_linear(x, x_list, y_list):
    """Linear interpolation of y at x; outside the table the end value is used."""
    if x <= x_list[0]:
        return y_list[0]
    if x >= x_list[-1]:
        return y_list[-1]
    for i in range(len(x_list) - 1):
        if x_list[i] <= x <= x_list[i + 1]:
            fraction = (x - x_list[i]) / (x_list[i + 1] - x_list[i])
            return y_list[i] + (y_list[i + 1] - y_list[i]) * fraction


def water_psat_bar(T_K):
    """Saturation pressure of water in bar (IAPWS-IF97 Eq. 30), valid 273.15-647.096 K."""
    n1, n2, n3, n4, n5 = 0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, \
        0.12020824702470e5, -0.32325550322333e7
    n6, n7, n8, n9, n10 = 0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6, \
        -0.23855557567849, 0.65017534844798e3
    theta = T_K + n9 / (T_K - n10)
    A = theta**2 + n1 * theta + n2
    B = n3 * theta**2 + n4 * theta + n5
    C = n6 * theta**2 + n7 * theta + n8
    p_MPa = (2 * C / (-B + (B**2 - 4 * A * C) ** 0.5)) ** 4
    return 10 * p_MPa


# ------------------------------------------------------------------- isotherms
# Every isotherm has the same parameters (T_K, p_bar, params) and returns q_kgkg,
# so an isotherm can be passed to another function as an argument.

def sips_kgkg(T_K, p_bar, params):
    """Sips working isotherm: q = qs (b p)^(1/n) / (1 + (b p)^(1/n)), b = b0 exp(-Ea/(R T))."""
    b_1_bar = params["sips_b0_1_bar"] * np.exp(-params["sips_Ea_J_mol"] / (params["R_J_molK"] * T_K))
    x = (b_1_bar * p_bar) ** (1 / params["sips_n"])
    return params["sips_qs_kgkg"] * x / (1 + x)


def gsta_kgkg(T_K, p_bar, params):
    """Dual-site GSTA part only (crystal fraction f included), kg/kg.

    q = f qs / m * sum(n K_n x^n) / (1 + sum(K_n x^n)),  x = p / P0,
    K_n = exp(-dH_n / (R T) + dS_n / R)
    """
    R = params["R_J_molK"]
    m = int(params["gsta_m"])
    x = p_bar / params["gsta_P0_bar"]
    numerator = 0.0
    denominator = 1.0
    for n in range(1, m + 1):
        dH_J_mol = params["gsta_dH" + str(n) + "_J_mol"]
        dS_J_molK = params["gsta_dS" + str(n) + "_J_molK"]
        K_n = np.exp(-dH_J_mol / (R * T_K) + dS_J_molK / R)
        numerator = numerator + n * K_n * x**n
        denominator = denominator + K_n * x**n
    return params["gsta_f"] * params["gsta_qs_kgkg"] / m * numerator / denominator


def capillary_mol_kg(T_K, p_bar, params):
    """Capillary condensation in mol/kg (NOT kg/kg): q = (mu/RT) / ln(ps/p)  (Eq. 7).

    mu/RT (Table 4) is interpolated linearly between the tabulated temperatures.
    """
    if p_bar == 0:
        return 0.0
    T_list_K, muRT_list_mol_kg = capillary_table(params)
    muRT_mol_kg = interpolate_linear(T_K, T_list_K, muRT_list_mol_kg)
    return muRT_mol_kg / np.log(water_psat_bar(T_K) / p_bar)


def capillary_kgkg(T_K, p_bar, params):
    """Capillary condensation converted to kg/kg (mol/kg x M_H2O)."""
    return capillary_mol_kg(T_K, p_bar, params) * params["M_H2O_kg_mol"]


def gsta_capillary_kgkg(T_K, p_bar, params):
    """Full equilibrium isotherm of the Adsorption 2021 paper: GSTA + capillary (Eq. 9), kg/kg."""
    return gsta_kgkg(T_K, p_bar, params) + capillary_kgkg(T_K, p_bar, params)


# ---------------------------------------------- functions that take an isotherm
def darken_factor(isotherm, T_K, p_bar, params, step=1e-4):
    """Darken factor d ln p / d ln q at constant T (Eq. 16), central difference in ln p."""
    p = max(p_bar, 1e-12)                            # ln q needs p > 0
    lnq_up = np.log(isotherm(T_K, p * np.exp(step), params))
    lnq_down = np.log(isotherm(T_K, p * np.exp(-step), params))
    dlnq_dlnp = (lnq_up - lnq_down) / (2 * step)
    return 1 / dlnq_dlnp


def isosteric_heat_J_mol(isotherm, T_K, p_bar, params, step=1e-4):
    """Isosteric heat dH = R (d ln p / d(1/T)) at constant q, J/mol (negative = released).

    Uses (d ln p/d(1/T))_q = -(d ln q/d(1/T))_p / (d ln q/d ln p)_T, so the isotherm
    does not need to be inverted.
    """
    p = max(p_bar, 1e-12)
    inv_T = 1 / T_K
    d_inv_T = step * inv_T
    lnq_cold = np.log(isotherm(1 / (inv_T + d_inv_T), p, params))
    lnq_hot = np.log(isotherm(1 / (inv_T - d_inv_T), p, params))
    dlnq_dinvT = (lnq_cold - lnq_hot) / (2 * d_inv_T)
    dlnq_dlnp = 1 / darken_factor(isotherm, T_K, p, params, step)
    return -params["R_J_molK"] * dlnq_dinvT / dlnq_dlnp


def equilibrium_pressure_bar(isotherm, q_kgkg, T_K, params, p_low_bar, p_high_bar,
                             tolerance=1e-10, max_iterations=200):
    """Pressure (bar) that gives loading q_kgkg at T_K: bisection on ln p with a while loop."""
    q_low = isotherm(T_K, p_low_bar, params)
    q_high = isotherm(T_K, p_high_bar, params)
    if q_kgkg < q_low or q_kgkg > q_high:
        raise OutOfRangeError("q = " + str(q_kgkg) + " kg/kg cannot be reached between "
                              + str(p_low_bar) + " and " + str(p_high_bar) + " bar")
    ln_low = np.log(p_low_bar)
    ln_high = np.log(p_high_bar)
    iteration = 0
    while ln_high - ln_low > tolerance and iteration < max_iterations:
        ln_mid = (ln_low + ln_high) / 2
        if isotherm(T_K, np.exp(ln_mid), params) < q_kgkg:
            ln_low = ln_mid
        else:
            ln_high = ln_mid
        iteration = iteration + 1
    return float(np.exp((ln_low + ln_high) / 2))


# ---------------------------------------------------------------------- class
class Zeolite3A:
    """Zeolite 3A pellets for in-situ steam removal in the DSQ core.

    zeo = Zeolite3A("parameters.csv")
    zeo.loading(T_K, p_bar)               -> q_kgkg   (Sips by default)
    zeo.k_ldf(T_K, p_bar)                 -> k_1_s
    zeo.heat_of_adsorption()              -> dH_J_mol (-60100 by default)
    zeo.equilibrium_pressure(q_kgkg, T_K) -> p_bar

    allow_extrapolation=False raises OutOfRangeError outside the data range;
    True computes anyway (label such results as extrapolated).
    """

    def __init__(self, filename="parameters.csv", allow_extrapolation=False):
        try:
            self.params = load_parameters(filename)
        except FileNotFoundError:
            raise FileNotFoundError(filename + " not found: run from the part1_zeolite folder "
                                    "or give the full path, e.g. Zeolite3A('../part1_zeolite/parameters.csv')")
        self.allow_extrapolation = allow_extrapolation
        self.R_J_molK = self.params["R_J_molK"]
        self.M_H2O_kg_mol = self.params["M_H2O_kg_mol"]
        self.Rc_m = self.params["d_crystal_m"] / 2
        self.d_pellet_m = self.params["d_pellet_m"]
        self.eps_bed = self.params["eps_bed"]
        self.rho_bed_kg_m3 = self.params["rho_bed_kg_m3"]
        self.rho_pellet_kg_m3 = self.params["rho_bed_kg_m3"] / (1 - self.params["eps_bed"])
        self.cp_J_kgK = self.params["cp_pellet_J_kgK"]

    def isotherm(self, model):
        """Return the isotherm function for model 'sips', 'gsta' or 'gsta_cap'."""
        match model:
            case "sips":
                return sips_kgkg
            case "gsta":
                return gsta_kgkg
            case "gsta_cap":
                return gsta_capillary_kgkg
            case _:
                raise ValueError("unknown model " + str(model) + "; use 'sips', 'gsta' or 'gsta_cap'")

    def check_range(self, T_K, p_bar, model="gsta_cap"):
        """Raise OutOfRangeError if T or pH2O is outside the data range of the model."""
        if p_bar < 0:
            raise ValueError("H2O partial pressure must be >= 0 bar")
        if p_bar >= water_psat_bar(T_K):
            raise ValueError("pH2O >= saturation pressure: liquid water, isotherms do not apply")
        if self.allow_extrapolation:
            return
        match model:
            case "sips":
                T_max_K = self.params["T_max_sips_K"]
            case _:
                T_max_K = self.params["T_max_K"]
        T_min_K = self.params["T_min_K"]
        tolerance = 1e-9
        if T_K < T_min_K - tolerance or T_K > T_max_K + tolerance:
            raise OutOfRangeError("T = " + str(round(T_K - 273.15, 1)) + " degC is outside the "
                                  + model + " data range " + str(round(T_min_K - 273.15)) + "-"
                                  + str(round(T_max_K - 273.15)) + " degC")
        if p_bar < self.params["p_min_bar"] - tolerance or p_bar > self.params["p_max_bar"] + tolerance:
            raise OutOfRangeError("pH2O = " + str(round(p_bar, 3)) + " bar is outside the data range "
                                  + str(self.params["p_min_bar"]) + "-" + str(self.params["p_max_bar"]) + " bar")

    def loading(self, T_K, p_bar, model="sips"):
        """Equilibrium loading q_kgkg at T_K and H2O partial pressure p_bar."""
        self.check_range(T_K, p_bar, model)
        return self.isotherm(model)(T_K, p_bar, self.params)

    def equilibrium_pressure(self, q_kgkg, T_K, model="sips"):
        """H2O partial pressure (bar) in equilibrium with loading q_kgkg at T_K."""
        if self.allow_extrapolation:
            p_low_bar = 1e-8
            p_high_bar = 0.99 * water_psat_bar(T_K)
        else:
            self.check_range(T_K, self.params["p_min_bar"], model)
            p_low_bar = self.params["p_min_bar"]
            p_high_bar = self.params["p_max_bar"]
        return equilibrium_pressure_bar(self.isotherm(model), q_kgkg, T_K, self.params,
                                        p_low_bar, p_high_bar)

    def micropore_diffusivity_m2_s(self, T_K):
        """Corrected micropore diffusivity Dc* = D0c exp(-Ea / (R T)) in m2/s (Eq. 17)."""
        return self.params["ldf_D0c_m2_s"] * np.exp(-self.params["ldf_Ea_J_mol"] / (self.R_J_molK * T_K))

    def darken_factor(self, T_K, p_bar, model="gsta_cap"):
        """Darken factor d ln p / d ln q from the chosen isotherm."""
        self.check_range(T_K, p_bar, model)
        return darken_factor(self.isotherm(model), T_K, p_bar, self.params)

    def k_ldf(self, T_K, p_bar, model="gsta_cap"):
        """LDF rate constant (1/s): k = 15 Dc / Rc^2 with Dc = Dc*(T) x Darken factor.

        Only the micropore term of Eq. 10. The Darken factor uses GSTA + capillary by
        default because D0c and Ea (Table 5) were fitted with that isotherm.
        """
        Dc_m2_s = self.micropore_diffusivity_m2_s(T_K) * self.darken_factor(T_K, p_bar, model)
        return 15 * Dc_m2_s / self.Rc_m**2

    def heat_of_adsorption(self, T_K=None, p_bar=None, method="sips"):
        """Heat of adsorption in J/mol H2O (negative = released).

        'sips'                default, Ea of the Sips isotherm (-60.1 kJ/mol)
        'constant'            -45.95 kJ/mol for comparison (Guffanti et al. 2021, Table 4)
        'isosteric_sips', 'isosteric_gsta', 'isosteric_gsta_cap'
                              Clausius-Clapeyron from an isotherm (needs T_K and p_bar)
        """
        match method:
            case "sips":
                return self.params["sips_Ea_J_mol"]
            case "constant":
                return self.params["dH_ads_J_mol"]
            case "isosteric_sips":
                model = "sips"
            case "isosteric_gsta":
                model = "gsta"
            case "isosteric_gsta_cap":
                model = "gsta_cap"
            case _:
                raise ValueError("unknown method " + str(method))
        if T_K is None or p_bar is None:
            raise ValueError("method " + method + " needs T_K and p_bar")
        self.check_range(T_K, p_bar, model)
        return isosteric_heat_J_mol(self.isotherm(model), T_K, p_bar, self.params)
