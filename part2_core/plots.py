"""
plots.py - matplotlib figures for the core model (used by app.py).

Every function returns a figure object (fig). One y-scale per panel; a legend when a
panel has two or more lines.
"""

import math

import matplotlib

matplotlib.use("Agg")                   # draw into images, no pop-up windows
import matplotlib.pyplot as plt        # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0ca30c"
RED = "#d03b3b"
GREY = "#52514e"


def plot_profiles(profile, res, criteria):
    """Four panels along the core height (height on the vertical axis, like the column)."""
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    z = profile["z_m"]

    ax[0].plot(profile["T_C"], z, color=BLUE)
    ax[0].set_xlabel("T (C)")
    ax[0].set_ylabel("Height above gas inlet z (m)")
    ax[0].set_title("Bed temperature")

    ax[1].plot(profile["pH2O_bar"], z, color=BLUE)
    ax[1].set_xscale("log")             # water drops from ~3 bar to ~0.03 bar
    set_log_ticks(ax[1], profile["pH2O_bar"].min(), profile["pH2O_bar"].max())
    ax[1].set_xlabel("pH2O (bar), log scale")
    ax[1].set_title("Water partial pressure")

    ax[2].plot(profile["q_kgkg"], z, color=BLUE, label="loading q")
    ax[2].plot(profile["qstar_kgkg"], z, color=ORANGE, linestyle="--",
               label="equilibrium q*")
    ax[2].set_xlabel("q (kg/kg)")
    ax[2].set_title("Zeolite water loading")
    ax[2].legend()

    ax[3].plot(profile["lift_ratio"], z, color=BLUE, label="gas drag / weight")
    ax[3].axvline(criteria["lift_max"], color=RED, linestyle=":",
                  label="limit " + str(criteria["lift_max"]))
    ax[3].set_xlim(0, 1.35 * max(criteria["lift_max"], profile["lift_ratio"].max()))
    ax[3].set_xlabel("drag / bed weight (-)")
    ax[3].set_title("Lift check")
    ax[3].legend(loc="center right")

    for axis in ax:
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_criteria(res, criteria):
    """Bar chart: each criterion as value / limit (a bar below 1 passes)."""
    H_over_D = res["case"]["H_m"] / res["case"]["D_m"]
    names = ["Water left\n/ allowed", "Lift ratio\n/ limit", "Pressure drop\n/ limit",
             "Peak bed T\n/ limit (C)", "Required H/D\n/ actual H/D"]
    values = [(1 - res["capture_frac"]) / max(1 - criteria["capture_min"], 1e-9),
              res["lift_ratio_max"] / criteria["lift_max"],
              res["dp_bar"] / criteria["dp_max_bar"],
              res["T_bed_max_C"] / criteria["T_bed_max_C"],
              criteria["HD_min"] / max(H_over_D, 1e-9)]
    passed = [res["pass_capture"], res["pass_lift"], res["pass_dp"],
              res["pass_temperature"], res["pass_aspect"]]

    colors = []
    for i in range(len(names)):
        if passed[i]:
            colors.append(GREEN)
            names[i] = names[i] + "\nPASS"
        else:
            colors.append(RED)
            names[i] = names[i] + "\nFAIL"

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(names, values, color=colors)
    ax.axhline(1.0, color=GREY, linestyle=":")
    ax.text(len(names) - 0.55, 1.03, "limit", color=GREY, ha="right")
    ax.set_ylim(0, max(1.2, max(values) * 1.15))
    ax.set_ylabel("value / limit (-)")
    ax.set_title("Design criteria (bar below 1 passes)")
    plt.tight_layout()
    return fig


def plot_capture_heatmap(df, best=None):
    """Water captured [%] over the D x H grid (imshow); x = fails a criterion."""
    data = df.copy()
    data["capture_pct"] = 100 * data["capture_frac"]
    capture = data.pivot(index="H_m", columns="D_m", values="capture_pct")
    feasible = data.pivot(index="H_m", columns="D_m", values="feasible")
    D_values = list(capture.columns)
    H_values = list(capture.index)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(capture, origin="lower", aspect="auto", cmap="Blues",
                      vmin=80, vmax=100)
    fig.colorbar(image, label="captured (%) - scale starts at 80")

    for i in range(len(H_values)):
        for j in range(len(D_values)):
            if feasible.loc[H_values[i], D_values[j]] != True:  # noqa: E712
                ax.text(j, i, "x", ha="center", va="center", color=RED)
    if best is not None:
        j = D_values.index(best["D_m"])
        i = H_values.index(best["H_m"])
        ax.plot(j, i, marker="s", markersize=22, markerfacecolor="none",
                markeredgecolor=ORANGE, markeredgewidth=3)

    ax.set_xticks(range(len(D_values)))
    ax.set_xticklabels(D_values)
    ax.set_yticks(range(len(H_values)))
    ax.set_yticklabels(H_values)
    ax.set_xlabel("Core diameter D (m)")
    ax.set_ylabel("Bed height H (m)")
    ax.set_title("Water captured (x = fails a criterion, square = smallest passing)")
    plt.tight_layout()
    return fig


def plot_feed_temperature(df):
    """Capture and peak bed temperature vs zeolite feed temperature (two panels)."""
    ok = df[df["error"] == ""]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ok["T_zeo_in_C"], ok["capture_pct"], color=BLUE, marker="o")
    ax[0].set_xlabel("Zeolite feed temperature (C)")
    ax[0].set_ylabel("Water captured (%)")
    ax[0].set_title("Water captured")
    ax[1].plot(ok["T_zeo_in_C"], ok["T_bed_max_C"], color=BLUE, marker="o")
    ax[1].set_xlabel("Zeolite feed temperature (C)")
    ax[1].set_ylabel("T max (C)")
    ax[1].set_title("Peak bed temperature")
    for axis in ax:
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_zeolite_rate(df):
    """Capture vs zeolite circulation; one line per thermal regime."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    if "regime" in df.columns:
        gas_side = df[df["regime"] == "gas_carries_heat"]
        solid_side = df[df["regime"] == "solid_carries_heat"]
        ax.plot(gas_side["m_zeo_kgh"], gas_side["capture_pct"], color=ORANGE, marker="o",
                label="W_solid < W_gas (gas carries the heat)")
        ax.plot(solid_side["m_zeo_kgh"], solid_side["capture_pct"], color=BLUE,
                marker="o", label="W_solid > W_gas (zeolite carries the heat)")
    failed = df[df["error"] != ""]
    if len(failed) > 0:
        ax.plot(failed["m_zeo_kgh"], [0] * len(failed), color=RED, marker="x",
                linestyle="none", label="no steady solution (W_solid ~ W_gas)")
    ax.set_xlabel("Zeolite circulation (kg/h)")
    ax.set_ylabel("Water captured (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Water captured vs zeolite circulation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Design mode: pressure drop and lift safety factor vs core size
# Limit lines are grey and dashed with a text label; ordered series (bed height H)
# use one blue ramp, light -> dark.
# ---------------------------------------------------------------------------
def blue_ramp(n):
    """n blues from light to dark (for ordered series such as bed height)."""
    colors = []
    for i in range(n):
        if n == 1:
            level = 0.8
        else:
            level = 0.45 + 0.5 * i / (n - 1)
        colors.append(plt.cm.Blues(level))
    return colors


def pick_some(values, n_max):
    """Up to n_max values spread evenly over a sorted list (always keeps both ends)."""
    if len(values) <= n_max:
        return list(values)
    picked = []
    for i in range(n_max):
        k = int(round(i * (len(values) - 1) / (n_max - 1)))
        picked.append(values[k])
    return picked


def set_log_ticks(axis, low, high):
    """Readable ticks on a log x-axis: 1-2-5 steps over a short range, decades otherwise.

    At complete capture pH2O falls to ~0 at the top of the core; values below
    1e-6 x the largest one are numerical noise, so the axis starts there.
    """
    if high <= 0:
        return                          # nothing to show on a log axis
    if low <= 0 or low < 1e-6 * high:
        low = 1e-6 * high
    k_low = int(math.floor(math.log10(low)))
    k_high = int(math.ceil(math.log10(high)))
    span = k_high - k_low
    if span <= 3:
        multiples = [1, 2, 5]
    else:
        multiples = [1]
    if span <= 4:
        decade_step = 1
    else:
        decade_step = 2                 # every other decade so the labels do not overlap
    ticks = []
    labels = []
    for k in range(k_high, k_low - 1, -decade_step):
        for m in multiples:
            value = m * 10.0 ** k
            if 0.8 * low <= value <= 1.25 * high:
                ticks.append(value)
                if value >= 0.01:
                    labels.append(str(round(value, 2)).rstrip("0").rstrip("."))
                else:
                    labels.append(str(m) + "e" + str(k))
    axis.set_xlim(0.8 * low, 1.25 * high)
    axis.minorticks_off()
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)


def plot_size_check(df, criteria, n_lines=4):
    """Figure 5: pressure drop and lift safety factor vs core diameter (lines = H).

    Safety factor = bed weight / gas drag = 1 / lift ratio. Two panels, one y-scale each.
    """
    ok = df[df["error"] == ""]
    heights = pick_some(sorted(list(ok["H_m"].unique())), n_lines)
    colors = blue_ramp(len(heights))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for i in range(len(heights)):
        part = ok[ok["H_m"] == heights[i]].sort_values("D_m")
        label = "H = " + str(round(heights[i], 2)) + " m"
        ax[0].plot(part["D_m"], 1000 * part["dp_bar"], color=colors[i], marker="o",
                   markersize=4, label=label)
        ax[1].plot(part["D_m"], part["lift_safety_factor"], color=colors[i], marker="o",
                   markersize=4, label=label)
    ax[0].set_yscale("log")             # dP falls steeply with D (about D^-4 or faster)
    ax[0].axhline(1000 * criteria["dp_max_bar"], color=GREY, linestyle="--",
                  label="limit " + str(round(1000 * criteria["dp_max_bar"])) + " mbar")
    ax[0].set_xlabel("Core diameter D (m)")
    ax[0].set_ylabel("Pressure drop in the core (mbar, log scale)")
    ax[0].set_title("(a) Pressure drop")

    sf_min = 1.0 / criteria["lift_max"]
    ax[1].set_ylim(bottom=0)
    ax[1].axhline(sf_min, color=GREY, linestyle="--",
                  label="minimum " + str(round(sf_min, 2)) + " (drag <= "
                  + str(round(100 * criteria["lift_max"])) + " % of weight)")
    ax[1].axhline(1.0, color=RED, linestyle=":", label="1 = bed starts to lift")
    ax[1].set_xlabel("Core diameter D (m)")
    ax[1].set_ylabel("Lift safety factor = bed weight / gas drag (-)")
    ax[1].set_title("(b) Safety factor against lifting the bed")
    ax[1].text(0.98, 0.35, "lines for different H lie on top of each other:\nthe largest "
               "drag/weight is at the gas inlet,\nso it depends on D, not on H",
               transform=ax[1].transAxes, ha="right", va="bottom", color=GREY, fontsize=8)
    for axis in ax:
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Web page figures (a) profiles and (b) lowest zeolite rate vs feed temperature
# ---------------------------------------------------------------------------
def plot_core_profiles(profile, res, T_edge_C, p_henry_bar):
    """(a) pH2O, loading and temperatures along the core height (three panels).

    The model uses one temperature for gas and pellets (local thermal equilibrium),
    so the bed curve is both. The stream end temperatures are marked: gas in at the
    bottom and out at the top, zeolite in at the top and out at the bottom.
    """
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    z = profile["z_m"]
    case = res["case"]
    H_m = case["H_m"]                   # the axis and the shading always reach z = H

    low = profile[profile["pH2O_bar"] < p_henry_bar]
    if len(low) > 0:
        z_henry = float(low["z_m"].min())
        for axis in ax:
            axis.axhspan(z_henry, H_m, color=GREY, alpha=0.10, linewidth=0)
        ax[0].text(0.03, 0.5 * (z_henry + H_m), "Henry's law\n(pH2O < "
                   + str(p_henry_bar) + " bar)", transform=ax[0].get_yaxis_transform(),
                   color=GREY, va="center", fontsize=8)

    ax[0].plot(profile["pH2O_bar"], z, color=BLUE)
    ax[0].set_xscale("log")             # water drops by orders of magnitude
    set_log_ticks(ax[0], profile["pH2O_bar"].min(), profile["pH2O_bar"].max())
    ax[0].set_xlabel("pH2O (bar, log scale; below 1e-6 x inlet not drawn)")
    ax[0].set_ylabel("Height above the gas inlet z (m)")
    ax[0].set_title("(a1) Water in the gas")

    ax[1].plot(profile["q_kgkg"], z, color=BLUE, label="loading q")
    ax[1].plot(profile["qstar_kgkg"], z, color=ORANGE, linestyle="--",
               label="equilibrium q*")
    ax[1].set_xlabel("Water loading (kg H2O / kg zeolite)")
    ax[1].set_title("(a2) Water on the zeolite")
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].plot(profile["T_C"], z, color=BLUE, label="bed: gas = zeolite (one T)")
    ax[2].plot([case["T_gas_in_C"]], [0.0], marker="^", markersize=9, color=ORANGE,
               linestyle="none", label="gas in " + str(round(case["T_gas_in_C"])) + " °C")
    ax[2].plot([res["T_gas_out_C"]], [H_m], marker="^", markersize=9, color=ORANGE,
               fillstyle="none", linestyle="none",
               label="gas out " + str(round(res["T_gas_out_C"], 1)) + " °C")
    ax[2].plot([case["T_zeo_in_C"]], [H_m], marker="v", markersize=9, color=GREY,
               linestyle="none", label="zeolite in " + str(round(case["T_zeo_in_C"]))
               + " °C")
    ax[2].plot([res["T_zeo_out_C"]], [0.0], marker="v", markersize=9, color=GREY,
               fillstyle="none", linestyle="none",
               label="zeolite out " + str(round(res["T_zeo_out_C"], 1)) + " °C")
    ax[2].axvline(T_edge_C, color=RED, linestyle=":")
    ax[2].text(T_edge_C, 0.5 * H_m, " " + str(round(T_edge_C)) + " °C\n edge of\n Sips data",
               color=RED, fontsize=8, va="center")
    ax[2].set_xlabel("Temperature (°C)")
    ax[2].set_title("(a3) Gas and zeolite temperature")
    ax[2].legend(loc="upper right", fontsize=7)

    for axis in ax:
        axis.grid(True, alpha=0.3)
    ax[0].set_ylim(-0.02 * H_m, 1.02 * H_m)
    fig.suptitle("D " + str(round(case["D_m"], 3)) + " m x H " + str(round(H_m, 3))
                 + " m, zeolite " + str(round(case["m_zeo_kgh"])) + " kg/h", fontsize=10)
    plt.tight_layout()
    return fig


def plot_min_rate_vs_feed(df, criteria):
    """(b) Lowest zeolite rate for peak bed T <= limit vs zeolite feed temperature,
    with the capacity minimum (equilibrium) for comparison. Log y: the two differ ~100x.
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ok = df[df["m_min_T_kgh"].notna()]
    none = df[df["m_min_T_kgh"].isna()]
    ax.plot(ok["T_zeo_in_C"], ok["m_min_T_kgh"], color=BLUE, marker="o",
            label="lowest rate for peak T <= " + str(round(criteria["T_bed_max_C"])) + " °C")
    ax.plot(df["T_zeo_in_C"], df["m_min_capacity_kgh"], color=ORANGE, marker="s",
            linestyle="--", label="capacity minimum (equilibrium with inlet gas)")
    for i in none.index:
        T_C = none.loc[i, "T_zeo_in_C"]
        ax.axvline(T_C, color=RED, linestyle=":")
        ax.text(T_C, ax.get_ylim()[1], " no rate\n passes ", color=RED, fontsize=8,
                ha="right", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("Zeolite feed temperature (°C)")
    ax.set_ylabel("Zeolite circulation rate (kg/h, log scale)")
    ax.set_title("(b) Lowest zeolite rate vs zeolite feed temperature")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig
