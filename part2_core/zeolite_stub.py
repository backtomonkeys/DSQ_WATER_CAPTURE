"""
zeolite_stub.py - Stand-ins for the Part 1 code, used until Part 1 is plugged in.

Names, arguments and units follow claude/interface-contract.md:
    OutOfRangeError                          (a kind of ValueError)
    Zeolite3A(allow_extrapolation=False)
        loading(T_K, p_bar, model="sips")    -> q [kg H2O / kg pellet]
        k_ldf(T_K, p_bar)                    -> k_LDF [1/s]
        heat_of_adsorption(method="sips")    -> -60100.0 J/mol
        rho_bed_kg_m3, rho_pellet_kg_m3, eps_bed, d_pellet_m, cp_J_kgK, M_H2O_kg_mol
    inlet_gas(X, S, P_bar, T_C, CO2_fresh_t_day, H2_CO2_ratio, basis) -> dict
p_bar is always the WATER PARTIAL PRESSURE.

Zeolite3A            : STUB, constant capacity 0.1 kg/kg and k_LDF 6e-3 1/s.
                       Temperature has no effect on capacity.
Zeolite3ASipsPreview : PREVIEW, Sips isotherm (paper parameters) and a simple
                       k_LDF fit. Use it only until the real Part 1 class is ready.
"""

import math

R_JMOLK = 8.314462618

# molar masses [kg/mol]
M_KGMOL = {"CO2": 0.04401, "H2": 0.002016, "CO": 0.02801, "DME": 0.046069,
           "H2O": 0.01801528}


class OutOfRangeError(ValueError):
    """T or pH2O is outside the range where the zeolite data are valid."""
    pass


def check_range(T_K, p_bar):
    """Raise OutOfRangeError outside 200-350 C or pH2O 0.05-4.5 bar."""
    if T_K < 473.15 or T_K > 623.15:
        raise OutOfRangeError("T = " + str(round(T_K - 273.15, 1))
                              + " C is outside 200-350 C")
    if p_bar < 0.05 or p_bar > 4.5:
        raise OutOfRangeError("pH2O = " + str(round(p_bar, 4))
                              + " bar is outside 0.05-4.5 bar")


def heat_of_adsorption_value(method):
    """Heat of adsorption [J/mol] for the two methods in the contract."""
    match method:
        case "sips":
            return -60100.0
        case "constant":
            return -45950.0
        case _:
            raise ValueError("Unknown method: " + str(method))


class Zeolite3A:
    """STUB zeolite 3A: constant loading 0.1 kg/kg and constant k_LDF 6e-3 1/s."""

    def __init__(self, allow_extrapolation=False):
        self.allow_extrapolation = allow_extrapolation
        # bed data from the contract (van Kampen et al. 2021)
        self.rho_bed_kg_m3 = 640.0
        self.eps_bed = 0.372
        self.rho_pellet_kg_m3 = self.rho_bed_kg_m3 / (1.0 - self.eps_bed)   # 1019
        self.d_pellet_m = 1.6e-3
        self.cp_J_kgK = 960.0          # contract value (approximate)
        self.M_H2O_kg_mol = 0.01801528

    def loading(self, T_K, p_bar, model="sips"):
        """Equilibrium loading q* [kg/kg]; the stub gives 0.1 for every model."""
        if not self.allow_extrapolation:
            check_range(T_K, p_bar)
        return 0.10

    def k_ldf(self, T_K, p_bar):
        """LDF coefficient [1/s]."""
        if not self.allow_extrapolation:
            check_range(T_K, p_bar)
        return 6.0e-3

    def heat_of_adsorption(self, method="sips"):
        """Heat of adsorption [J/mol], negative = releases heat."""
        return heat_of_adsorption_value(method)


class Zeolite3ASipsPreview:
    """PREVIEW zeolite 3A: Sips isotherm + fitted k_LDF (not the Part 1 model).

    Sips (van Kampen et al., React. Chem. Eng. 2021, 6, 244):
        b = b0 * exp(-Ea / (R T)),   Ea = -60.1 kJ/mol
        q* = qs * (b p)^(1/n) / (1 + (b p)^(1/n))
    Check: 250 C and 1 bar give 0.075 kg/kg.

    k_LDF fit through the check values in part1-zeolite-brief:
        250 C: 5e-3 1/s at 0.5 bar, 9e-3 1/s at 3 bar;  200 C, 0.5 bar: 2.7e-3 1/s
    """

    def __init__(self, allow_extrapolation=False):
        self.allow_extrapolation = allow_extrapolation
        self.rho_bed_kg_m3 = 640.0
        self.eps_bed = 0.372
        self.rho_pellet_kg_m3 = self.rho_bed_kg_m3 / (1.0 - self.eps_bed)
        self.d_pellet_m = 1.6e-3
        self.cp_J_kgK = 960.0
        self.M_H2O_kg_mol = 0.01801528
        # Sips parameters
        self.qs_kgkg = 0.21
        self.b0_1bar = 4.10e-7
        self.Ea_Jmol = -60.1e3
        self.n_sips = 1.53
        # k_LDF fit: k = 5e-3 * (p/0.5)^m * exp(-E/R * (1/T - 1/523.15))
        self.m_p = math.log(9.0 / 5.0) / math.log(3.0 / 0.5)                  # 0.328
        self.E_k_over_R_K = math.log(5.0 / 2.7) / (1 / 473.15 - 1 / 523.15)  # ~3050 K

    def affinity_1bar(self, T_K):
        """Sips affinity constant b(T) [1/bar]."""
        return self.b0_1bar * math.exp(-self.Ea_Jmol / (R_JMOLK * T_K))

    def loading(self, T_K, p_bar, model="sips"):
        """Sips equilibrium loading q* [kg/kg]. The preview has no GSTA."""
        if model != "sips":
            raise ValueError("The Sips preview only has model='sips'")
        if not self.allow_extrapolation:
            check_range(T_K, p_bar)
        x = (self.affinity_1bar(T_K) * p_bar) ** (1.0 / self.n_sips)
        return self.qs_kgkg * x / (1.0 + x)

    def k_ldf(self, T_K, p_bar):
        """Preview fit of k_LDF [1/s]."""
        if not self.allow_extrapolation:
            check_range(T_K, p_bar)
        k_p = 5.0e-3 * (p_bar / 0.5) ** self.m_p
        return k_p * math.exp(-self.E_k_over_R_K * (1.0 / T_K - 1.0 / 523.15))

    def heat_of_adsorption(self, method="sips"):
        """Heat of adsorption [J/mol]."""
        return heat_of_adsorption_value(method)

    def equilibrium_pressure(self, q_kgkg, T_K):
        """Inverse Sips: water partial pressure [bar] in equilibrium with q."""
        if q_kgkg <= 0.0 or q_kgkg >= self.qs_kgkg:
            raise ValueError("q must be between 0 and qs")
        x = q_kgkg / (self.qs_kgkg - q_kgkg)
        return x ** self.n_sips / self.affinity_1bar(T_K)


def inlet_gas(X=0.30, S=0.60, P_bar=30.0, T_C=250.0, CO2_fresh_t_day=1.0,
              H2_CO2_ratio=3.0, basis="recycle"):
    """STUB of Part 1 inlet_gas(): gas entering the core (same dict as the contract).

    Method from claude/water-capture-code-spec.md:
    * total water is fixed by the fresh CO2: with full recycle all carbon ends as DME,
      so 1.5 mol H2O per mol fresh CO2 (25.6 kg/h for 1 t CO2/day), whatever X and S;
    * the gas COMPOSITION is the reactor outlet of one pass with feed CO2 : H2 = 1 : 3,
        2 CO2 + 6 H2 -> DME + 3 H2O   (fraction S of the converted carbon)
        CO2 + H2 -> CO + H2O          (fraction 1 - S, RWGS)
    * total flow = water flow / water mole fraction.
    Base case X 0.30, S 0.60: y_H2O 0.1071, 3.68 mol/s.
    """
    if basis != "recycle":
        raise ValueError("The stub only supports basis='recycle'")
    if X <= 0.0 or X > 1.0:
        raise ValueError("X must be in (0, 1]")
    if S < 0.0 or S > 1.0:
        raise ValueError("S must be in [0, 1]")

    # one pass, per mol CO2 fed
    per_pass = {}
    per_pass["CO2"] = 1.0 - X
    per_pass["H2"] = H2_CO2_ratio - 3.0 * X * S - X * (1.0 - S)
    per_pass["CO"] = X * (1.0 - S)
    per_pass["DME"] = 0.5 * X * S
    per_pass["H2O"] = 1.5 * X * S + X * (1.0 - S)
    if per_pass["H2"] < 0.0:
        raise ValueError("Not enough H2 for this X and H2/CO2 ratio")

    # scale so that the water flow equals 1.5 mol per mol fresh CO2
    F0 = CO2_fresh_t_day * 1000.0 / M_KGMOL["CO2"] / 86400.0       # mol/s fresh CO2
    scale = 1.5 * F0 / per_pass["H2O"]
    F = {}
    for species in per_pass:
        F[species] = per_pass[species] * scale

    F_total = 0.0
    for species in F:
        F_total = F_total + F[species]
    y = {}
    M_mix = 0.0
    for species in F:
        y[species] = F[species] / F_total
        M_mix = M_mix + y[species] * M_KGMOL[species]

    return {
        "F_total_mol_s": F_total,
        "F_mol_s": F,
        "y": y,
        "p_H2O_bar": y["H2O"] * P_bar,
        "water_kg_h": F["H2O"] * M_KGMOL["H2O"] * 3600.0,
        "M_mix_kg_mol": M_mix,
        "m_gas_kg_s": F_total * M_mix,
        # whole plant: all fresh CO2 ends as DME, needing 3 mol H2 per mol CO2
        "fresh_H2_kg_day": 3.0 * F0 * M_KGMOL["H2"] * 86400.0,
        "DME_kg_day": 0.5 * F0 * M_KGMOL["DME"] * 86400.0,
        "T_K": T_C + 273.15,
        "P_bar": P_bar,
    }
