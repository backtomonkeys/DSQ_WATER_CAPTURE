"""
feed_gas.py - Gas entering the DSQ core (zeolite moving bed), estimated from stoichiometry.
Part 1 of the DSQ water-capture code (CPE 100 project).

The reactor kinetics are NOT simulated. The gas leaving the catalyst donuts (= gas entering
the core) is estimated from two assumed numbers (sliders in the web app):

    X   CO2 conversion per pass (fraction of the CO2 in the reactor feed that reacts)
    S   carbon selectivity to DME (fraction of the converted carbon that ends in DME);
        the rest (1 - S) leaves as CO through the reverse water-gas shift (RWGS)

Reactions
    2 CO2 + 6 H2 -> CH3OCH3 + 3 H2O      direct DME synthesis
      CO2 +   H2 -> CO      +   H2O      RWGS

Flow basis: 1 t fresh CO2 per day (claude/water-capture-code-spec.md).
    basis "recycle"       full recycle, no purge: CO2, H2 and CO go back to the reactor, so over
                          the plant all carbon ends in DME and the water to remove is fixed at
                          1.5 mol H2O per mol fresh CO2 (~25.6 kg/h), independent of X and S.
                          X and S set the composition of the gas leaving the catalyst per pass
                          (and so the H2O mole fraction); the total flow is scaled so that the
                          water flow equals the fixed value. The recycle loop is not simulated.
    basis "once_through"  reactor CO2 feed = fresh CO2; water = what one pass makes

Base case (spec): X = 0.30, S = 0.60, 30 bar, 250 degC. S = 1.0 is not used because RWGS
always makes some CO.
"""

import numpy as np
import pandas as pd

SPECIES = ["CO2", "H2", "CO", "DME", "H2O"]

M_kg_mol = {
    "CO2": 44.0095e-3,
    "H2": 2.01588e-3,
    "CO": 28.0101e-3,
    "DME": 46.0684e-3,
    "H2O": 18.01528e-3,
}

# number of (C, H, O) atoms in one molecule
ATOMS_CHO = {
    "CO2": (1, 0, 2),
    "H2": (0, 2, 0),
    "CO": (1, 0, 1),
    "DME": (2, 6, 1),
    "H2O": (0, 2, 1),
}

SECONDS_PER_DAY = 86400


def fresh_co2_mol_s(CO2_fresh_t_day=1.0):
    """Fresh CO2 molar flow (mol/s) for a capacity in tonnes CO2 per day."""
    return CO2_fresh_t_day * 1000 / M_kg_mol["CO2"] / SECONDS_PER_DAY


def reactor_feed_mol_s(X, S, CO2_fresh_t_day=1.0, H2_CO2_ratio=3.0, basis="recycle"):
    """Molar flows (mol/s) entering the catalyst beds (CO2 + H2 at H2_CO2_ratio)."""
    F_fresh_mol_s = fresh_co2_mol_s(CO2_fresh_t_day)
    match basis:
        case "recycle":
            # one pass makes X (1 + S/2) mol H2O per mol CO2 fed; scale the feed so the
            # water flow equals 1.5 x fresh CO2 (all carbon ends in DME over the plant)
            F_CO2_mol_s = 1.5 * F_fresh_mol_s / (X * (1 + S / 2))
        case "once_through":
            F_CO2_mol_s = F_fresh_mol_s
        case _:
            raise ValueError("unknown basis " + str(basis) + "; use 'recycle' or 'once_through'")
    return {"CO2": F_CO2_mol_s, "H2": H2_CO2_ratio * F_CO2_mol_s, "CO": 0.0, "DME": 0.0, "H2O": 0.0}


def react(feed_mol_s, X, S):
    """Apply conversion X and DME carbon selectivity S to a feed; return outlet flows (mol/s)."""
    CO2_reacted_mol_s = X * feed_mol_s["CO2"]
    xi_DME_mol_s = S * CO2_reacted_mol_s / 2       # extent of 2 CO2 + 6 H2 -> DME + 3 H2O
    xi_RWGS_mol_s = (1 - S) * CO2_reacted_mol_s    # extent of CO2 + H2 -> CO + H2O

    out = dict(feed_mol_s)                          # copy, so the feed is not changed
    out["CO2"] = out["CO2"] - 2 * xi_DME_mol_s - xi_RWGS_mol_s
    out["H2"] = out["H2"] - 6 * xi_DME_mol_s - xi_RWGS_mol_s
    out["DME"] = out["DME"] + xi_DME_mol_s
    out["CO"] = out["CO"] + xi_RWGS_mol_s
    out["H2O"] = out["H2O"] + 3 * xi_DME_mol_s + xi_RWGS_mol_s

    if out["H2"] < -1e-12:
        raise ValueError("not enough H2 in the feed for this X and S; raise H2_CO2_ratio")
    return out


def atom_flows_mol_s(flows_mol_s):
    """Atom flows [C, H, O] in mol atoms/s for a dict of molar flows."""
    atoms = np.array([0.0, 0.0, 0.0])
    for species, F_mol_s in flows_mol_s.items():
        atoms = atoms + F_mol_s * np.array(ATOMS_CHO[species])
    return atoms


def inlet_gas(X=0.30, S=0.60, P_bar=30.0, T_C=250.0, CO2_fresh_t_day=1.0,
              H2_CO2_ratio=3.0, basis="recycle"):
    """Gas entering the zeolite core, estimated from stoichiometry.

    Returns a dict with: F_mol_s (dict per species), F_total_mol_s, y (dict of mole
    fractions), p_H2O_bar, water_mol_s, water_kg_h, M_mix_kg_mol, m_gas_kg_s, T_K, P_bar,
    feed_mol_s, fresh_CO2_mol_s, fresh_H2_mol_s, fresh_H2_kg_day, DME_kg_day, X, S, basis.
    """
    if not 0 < X <= 1:
        raise ValueError("X must be in (0, 1]")
    if not 0 <= S <= 1:
        raise ValueError("S must be in [0, 1]")
    if P_bar <= 0:
        raise ValueError("P_bar must be > 0")

    feed_mol_s = reactor_feed_mol_s(X, S, CO2_fresh_t_day, H2_CO2_ratio, basis)
    out_mol_s = react(feed_mol_s, X, S)

    F_total_mol_s = 0.0
    for species in SPECIES:
        F_total_mol_s = F_total_mol_s + out_mol_s[species]

    y = {}
    M_mix_kg_mol = 0.0
    for species in SPECIES:
        y[species] = out_mol_s[species] / F_total_mol_s
        M_mix_kg_mol = M_mix_kg_mol + y[species] * M_kg_mol[species]

    F_fresh_mol_s = fresh_co2_mol_s(CO2_fresh_t_day)
    match basis:
        case "recycle":                            # plant: 2 CO2 + 6 H2 -> DME + 3 H2O
            fresh_H2_mol_s = 3 * F_fresh_mol_s
            DME_product_mol_s = F_fresh_mol_s / 2
        case "once_through":
            fresh_H2_mol_s = feed_mol_s["H2"]
            DME_product_mol_s = out_mol_s["DME"]

    return {
        "X": X,
        "S": S,
        "basis": basis,
        "T_K": T_C + 273.15,
        "P_bar": P_bar,
        "F_mol_s": out_mol_s,
        "F_total_mol_s": F_total_mol_s,
        "y": y,
        "p_H2O_bar": y["H2O"] * P_bar,
        "water_mol_s": out_mol_s["H2O"],
        "water_kg_h": out_mol_s["H2O"] * M_kg_mol["H2O"] * 3600,
        "M_mix_kg_mol": M_mix_kg_mol,
        "m_gas_kg_s": F_total_mol_s * M_mix_kg_mol,
        "feed_mol_s": feed_mol_s,
        "fresh_CO2_mol_s": F_fresh_mol_s,
        "fresh_H2_mol_s": fresh_H2_mol_s,
        "fresh_H2_kg_day": fresh_H2_mol_s * M_kg_mol["H2"] * SECONDS_PER_DAY,
        "DME_kg_day": DME_product_mol_s * M_kg_mol["DME"] * SECONDS_PER_DAY,
    }


def gas_table(gas):
    """DataFrame of the core inlet gas: molar flow, mole fraction, partial pressure, mass flow."""
    rows = []
    for species in SPECIES:
        F_mol_s = gas["F_mol_s"][species]
        rows.append([species, F_mol_s, gas["y"][species], gas["y"][species] * gas["P_bar"],
                     F_mol_s * M_kg_mol[species] * 3600])
    table = pd.DataFrame(rows, columns=["species", "F_mol_s", "y", "p_bar", "m_kg_h"])
    table = table.set_index("species")
    table.loc["total"] = table.sum()
    return table


def sweep_inlet_gas(X_values, S_values, P_bar=30.0):
    """DataFrame of inlet_gas() over a grid of X and S (one row per pair; use pivot for maps)."""
    rows = []
    for X in X_values:
        for S in S_values:
            gas = inlet_gas(X=X, S=S, P_bar=P_bar)
            rows.append([X, S, gas["water_kg_h"], gas["F_total_mol_s"], gas["y"]["H2O"],
                         gas["p_H2O_bar"], gas["DME_kg_day"]])
    return pd.DataFrame(rows, columns=["X", "S", "water_kg_h", "F_total_mol_s", "y_H2O",
                                       "p_H2O_bar", "DME_kg_day"])
