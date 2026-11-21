"""
gas_props.py - Ideal-gas properties of the gas in the core.

* cp from a table (gas_properties.csv) at 400, 500, 600, 700 K, linear interpolation
* viscosity of each gas (Sutherland law) and of the mixture (Wilke mixing rule)

Accuracy: cp about 1% (JANAF) for H2, CO2, CO, H2O, N2; DME and CH3OH values are
approximate. Viscosity about 10-15%, good enough for the Ergun pressure drop.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))      # folder of this file
DATA_FILE = os.path.join(HERE, "gas_properties.csv")
T_TABLE_K = [400.0, 500.0, 600.0, 700.0]


def load_gas_data(csv_path=DATA_FILE):
    """Read gas_properties.csv into a dictionary {species: {property: value}}."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    data = {}
    for i in range(len(df)):
        name = df.loc[i, "species"]
        data[name] = {
            "M_kgmol": float(df.loc[i, "M_kgmol"]),
            "cp_table_JmolK": [float(df.loc[i, "cp400_JmolK"]),
                               float(df.loc[i, "cp500_JmolK"]),
                               float(df.loc[i, "cp600_JmolK"]),
                               float(df.loc[i, "cp700_JmolK"])],
            "mu_ref_Pas": float(df.loc[i, "mu_ref_Pas"]),
            "T_mu_ref_K": float(df.loc[i, "T_mu_ref_K"]),
            "S_suth_K": float(df.loc[i, "S_suth_K"]),
        }
    return data


def check_species(flows_mols, gas_data):
    """Return a clean copy of the flows {species: mol/s}; every species must be known.

    Species names follow the interface contract: CO2, H2, CO, DME, H2O (also CH3OH, N2).
    """
    clean = {}
    for species in flows_mols:
        if species not in gas_data:
            raise KeyError("Species " + str(species) + " is not in gas_properties.csv")
        F = float(flows_mols[species])
        if F < 0.0:
            raise ValueError("Negative flow for " + str(species))
        clean[species] = F
    if "H2O" not in clean:
        clean["H2O"] = 0.0
    return clean


def interpolate(x, x0, x1, y0, y1):
    """Straight line through (x0, y0) and (x1, y1), evaluated at x."""
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def cp_JmolK(species, T_K, gas_data):
    """Ideal-gas cp [J/(mol K)] by linear interpolation in the 400-700 K table.

    Outside the table the end value is used.
    """
    table = gas_data[species]["cp_table_JmolK"]
    if T_K <= T_TABLE_K[0]:
        return table[0]
    if T_K >= T_TABLE_K[-1]:
        return table[-1]
    for i in range(len(T_TABLE_K) - 1):
        if T_K <= T_TABLE_K[i + 1]:
            return interpolate(T_K, T_TABLE_K[i], T_TABLE_K[i + 1], table[i], table[i + 1])


def viscosity_Pas(species, T_K, gas_data):
    """Viscosity of one gas [Pa s], Sutherland law (T_K may be a NumPy array)."""
    d = gas_data[species]
    T0 = d["T_mu_ref_K"]
    S = d["S_suth_K"]
    return d["mu_ref_Pas"] * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)


def mixture_viscosity_Pas(y, T_K, gas_data):
    """Viscosity of a gas mixture [Pa s], Wilke mixing rule.

    y    : dictionary species -> mole fraction (number or NumPy array)
    T_K  : temperature (number or NumPy array of the same shape)
    """
    mu = {}
    for species in y:
        mu[species] = viscosity_Pas(species, T_K, gas_data)

    mu_mix = 0.0
    for i in y:
        denominator = 0.0
        for j in y:
            Mi = gas_data[i]["M_kgmol"]
            Mj = gas_data[j]["M_kgmol"]
            phi_ij = ((1.0 + np.sqrt(mu[i] / mu[j]) * (Mj / Mi) ** 0.25) ** 2
                      / np.sqrt(8.0 * (1.0 + Mi / Mj)))
            denominator = denominator + y[j] * phi_ij
        mu_mix = mu_mix + y[i] * mu[i] / denominator
    return mu_mix


def mixture_molar_mass_kgmol(y, gas_data):
    """Mean molar mass [kg/mol] of a mixture with mole fractions y."""
    M_mix = 0.0
    for species in y:
        M_mix = M_mix + y[species] * gas_data[species]["M_kgmol"]
    return M_mix
