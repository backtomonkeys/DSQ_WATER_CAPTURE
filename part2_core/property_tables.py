"""
property_tables.py - lookup tables for q*(T, p) and k_LDF(T, p) to make the core
model fast.

Why: one core solution calls the isotherm ~120 000 times and k_LDF ~90 000 times
(shooting = many marches). Part 1 k_LDF has a numerical Darken factor with several
isotherm calls inside, so a design sweep of 100 sizes took many minutes. Here each
function is called once per grid point (a few seconds), then read back by bilinear
interpolation in T and ln(p) of ln(value).

Accuracy (checked in test_core.py): relative error < 1e-4 against the direct call;
core results change < 0.01 %. Only used with the "clip" policy, where T and p are
always moved inside the data range before the call.
"""

import math


class PropertyTable:
    """Grid of ln f(T, p) over T_min..T_max and ln p_min..ln p_max, read back bilinearly.

    fn       : the zeolite function, e.g. zeolite.loading or zeolite.k_ldf
    model    : isotherm model passed to fn, or None for k_ldf(T, p)
    limits   : {"T_min_K", "T_max_K", "p_min_bar", "p_max_bar"} (data range)
    """

    def __init__(self, fn, model, limits, dT_K, dlnp):
        self.T_min_K = limits["T_min_K"]
        self.lnp_min = math.log(limits["p_min_bar"])
        self.n_T = max(2, int(math.ceil((limits["T_max_K"] - self.T_min_K) / dT_K)) + 1)
        lnp_span = math.log(limits["p_max_bar"]) - self.lnp_min
        self.n_p = max(2, int(math.ceil(lnp_span / dlnp)) + 1)
        self.dT_K = (limits["T_max_K"] - self.T_min_K) / (self.n_T - 1)
        self.dlnp = lnp_span / (self.n_p - 1)
        self.ln_values = []                         # list of rows (one row per T)
        for i in range(self.n_T):
            T_K = self.T_min_K + i * self.dT_K
            row = []
            for j in range(self.n_p):
                p_bar = math.exp(self.lnp_min + j * self.dlnp)
                if model is None:
                    value = fn(T_K, p_bar)
                else:
                    value = fn(T_K, p_bar, model)
                row.append(math.log(value))
            self.ln_values.append(row)

    def value(self, T_K, p_bar):
        """Bilinear interpolation of ln f in (T, ln p); T, p must be inside the grid."""
        x = (T_K - self.T_min_K) / self.dT_K
        i = min(max(int(x), 0), self.n_T - 2)
        fx = x - i
        y = (math.log(p_bar) - self.lnp_min) / self.dlnp
        j = min(max(int(y), 0), self.n_p - 2)
        fy = y - j
        row_a = self.ln_values[i]
        row_b = self.ln_values[i + 1]
        ln_a = (1.0 - fy) * row_a[j] + fy * row_a[j + 1]
        ln_b = (1.0 - fy) * row_b[j] + fy * row_b[j + 1]
        return math.exp((1.0 - fx) * ln_a + fx * ln_b)

    def loading(self, T_K, p_bar, model="sips"):
        """Same call as Zeolite3A.loading (the model is fixed when the table is built)."""
        return self.value(T_K, p_bar)

    def k_ldf(self, T_K, p_bar):
        """Same call as Zeolite3A.k_ldf."""
        return self.value(T_K, p_bar)
