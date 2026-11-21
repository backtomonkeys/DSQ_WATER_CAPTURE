"""
check_tables.py - how well the lookup tables (property_tables.py) follow the real
Part 1 functions, and how much that changes the core results.

Run (inside part2_core):   python check_tables.py
Takes about 1-3 minutes (it also runs the core once without tables).
"""

import math
import time

import design as dz
import property_tables

GRIDS = [[2.0, 0.05], [1.0, 0.02], [0.5, 0.01]]      # [dT_K, d ln p]
N_TEST = 40                                         # 40 x 40 test points per table


def worst_error(fn, model, table, limits):
    """Largest relative error between the table and the direct call, and where."""
    worst = 0.0
    where = [0.0, 0.0]
    lnp_low = math.log(limits["p_min_bar"])
    lnp_high = math.log(limits["p_max_bar"])
    for i in range(N_TEST):
        for j in range(N_TEST):
            T_K = limits["T_min_K"] + (i + 0.37) / N_TEST * (limits["T_max_K"]
                                                             - limits["T_min_K"])
            p_bar = math.exp(lnp_low + (j + 0.61) / N_TEST * (lnp_high - lnp_low))
            if model is None:
                exact = fn(T_K, p_bar)
            else:
                exact = fn(T_K, p_bar, model)
            error = abs(table.value(T_K, p_bar) / exact - 1.0)
            if error > worst:
                worst = error
                where = [T_K - 273.15, p_bar]
    return worst, where


def main():
    params = dz.load_params()
    criteria = dz.default_criteria(params)
    zeolite, used = dz.load_zeolite("part1")
    limits = dz.zeolite_limits(zeolite, "sips", params)
    print("Zeolite model:", used)
    print("loading range:", limits["loading"])
    print("k_ldf range:  ", limits["k_ldf"])

    t0 = time.time()
    for i in range(200):
        zeolite.k_ldf(500.0, 1.0)
    print("one k_ldf call:", round((time.time() - t0) / 200 * 1e6), "microseconds")
    t0 = time.time()
    for i in range(200):
        zeolite.loading(500.0, 1.0, "sips")
    print("one loading call:", round((time.time() - t0) / 200 * 1e6), "microseconds")
    print()

    for name, fn, model, lim in [["loading", zeolite.loading, "sips", limits["loading"]],
                                 ["k_ldf", zeolite.k_ldf, None, limits["k_ldf"]]]:
        for dT_K, dlnp in GRIDS:
            t0 = time.time()
            table = property_tables.PropertyTable(fn, model, lim, dT_K, dlnp)
            build_s = time.time() - t0
            error, where = worst_error(fn, model, table, lim)
            print(name + " grid dT " + str(dT_K) + " K, dlnp " + str(dlnp) + ": "
                  + str(table.n_T * table.n_p) + " points, built in "
                  + str(round(build_s, 1)) + " s, worst error " + str(round(error, 7))
                  + " at " + str(round(where[0], 2)) + " C, " + str(round(where[1], 4))
                  + " bar")
    print()

    case = dz.default_case(params)
    case = dz.updated_case(case, {"m_zeo_kgh": 2204.0})
    case = dz.sized_case(case, 0.20, 0.50, params["d_shaft_ratio"], params["pitch_ratio"])
    t0 = time.time()
    a = dz.run_check(case, zeolite, params, criteria, use_tables=True)
    t_table = time.time() - t0
    t0 = time.time()
    b = dz.run_check(case, zeolite, params, criteria, use_tables=False)
    t_direct = time.time() - t0
    print("Core D 0.20 x H 0.50 m, 2204 kg/h: with tables " + str(round(t_table, 1))
          + " s, direct " + str(round(t_direct, 1)) + " s")
    for key in ["capture_frac", "T_bed_max_C", "q_out_kgkg", "dp_bar", "lift_ratio_max"]:
        print("  " + key + ": table " + str(a[key]) + "  direct " + str(b[key])
              + "  relative difference " + str(abs(a[key] / b[key] - 1.0)))


if __name__ == "__main__":
    main()
