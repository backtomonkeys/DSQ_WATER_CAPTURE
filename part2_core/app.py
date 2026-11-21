"""
app.py - Streamlit web page for the DSQ water-capture core (Part 2).

Run (from the repo folder):   streamlit run part2_core/app.py
      (or inside part2_core):  streamlit run app.py
Streamlit is outside CPE 100; every calculation lives in design.py / core_column.py.

Check mode : the user gives size and zeolite rate -> one core solution (a few seconds).
Design mode: runs only when the button is pressed (about 1-3 minutes); results cached.
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core_column as cc
import design as dz
import plots

st.set_page_config(page_title="DSQ core - water capture", layout="wide")

PARAMS = dz.load_params()
ZEOLITE_CHOICES = {
    "Part 1 Zeolite3A (real, default)": "auto",
    "Sips preview (no Part 1 needed)": "preview",
    "Stub: constant 0.1 kg/kg (program check only)": "stub",
}
HEATS = {"-60.1 kJ/mol (Sips Ea, default)": "sips",
         "-45.95 kJ/mol (Guffanti 2021, compare)": "constant"}
ISOTHERMS = {"Sips (main)": "sips", "GSTA (compare only)": "gsta",
             "GSTA + capillary (compare only)": "gsta_cap"}
HD_NOTE = "assumed rule of thumb, no specific source"
LIFT_HELP = ("Largest LOCAL value along the height of (Ergun pressure gradient) / (weight "
             "of the bed per metre). It is at the gas inlet (bottom), where the gas is "
             "hottest and wettest and flows fastest. The height average is shown below.")
DESIGN_FEED_T_LIST_C = [200.0, 210.0, 220.0, 230.0, 240.0, 245.0, 250.0]  # 245: rate ~2x


# ---------------------------------------------------------------------------
# Cached objects and results
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the zeolite model and building lookup tables...")
def get_zeolite(source, isotherm_model):
    """Zeolite3A (or stub) + its q*/k_LDF lookup tables, built once per session."""
    zeolite, used = dz.load_zeolite(source)
    dz.fast_functions(zeolite, isotherm_model, PARAMS)
    limits = dz.zeolite_limits(zeolite, isotherm_model, PARAMS)
    return zeolite, used, limits


@st.cache_data(show_spinner=False)
def cached_check(case, criteria, source, gas_source):
    zeolite, used, limits = get_zeolite(source, case["isotherm_model"])
    return dz.run_check(case, zeolite, PARAMS, criteria, gas_source)


@st.cache_data(show_spinner=False)
def cached_design(case, criteria, source, gas_source, D_list, H_list, T_list):
    """Design steps (1)-(4); the same inputs come back from the cache at once."""
    zeolite, used, limits = get_zeolite(source, case["isotherm_model"])
    t_start = time.time()
    out = dz.design_with_rate(case, zeolite, PARAMS, criteria, D_list, H_list,
                              gas_source=gas_source)
    out["df_feed_T"] = dz.min_rate_vs_feed_temperature(case, zeolite, PARAMS, criteria,
                                                       T_list, gas_source)
    out["seconds"] = time.time() - t_start
    return out


# ---------------------------------------------------------------------------
# Small helpers for the page
# ---------------------------------------------------------------------------
def show_figure(fig):
    """Show a matplotlib figure on the page and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def excel_bytes(sheets, filename):
    """Write the sheets to an Excel file and return its bytes for the download button."""
    path = os.path.join(dz.HERE, filename)
    dz.results_to_excel(path, sheets)
    with open(path, "rb") as file:
        return file.read()


def pass_fail(flag):
    if flag:
        return "PASS"
    return "FAIL"


def criteria_table(res, criteria):
    """(c) The five criteria as a table: value, limit, PASS/FAIL."""
    case = res["case"]
    rows = [
        ["Water captured", dz.format_capture(res["capture_frac"]),
         ">= " + str(round(100 * criteria["capture_min"], 1)) + "%",
         pass_fail(res["pass_capture"])],
        ["Gas drag / bed weight (largest local value, z = "
         + str(round(res["z_lift_max_m"], 2)) + " m)", str(round(res["lift_ratio_max"], 3)),
         "<= " + str(criteria["lift_max"]), pass_fail(res["pass_lift"])],
        ["Pressure drop", str(round(res["dp_bar"], 4)) + " bar",
         "<= " + str(criteria["dp_max_bar"]) + " bar", pass_fail(res["pass_dp"])],
        ["Peak temperature (gas = zeolite)", str(round(res["T_bed_max_C"], 1)) + " °C",
         "<= " + str(criteria["T_bed_max_C"]) + " °C", pass_fail(res["pass_temperature"])],
        ["H/D (" + HD_NOTE + ")", str(round(case["H_m"] / case["D_m"], 2)),
         ">= " + str(criteria["HD_min"]), pass_fail(res["pass_aspect"])],
    ]
    return pd.DataFrame(rows, columns=["criterion", "value", "limit", "result"])


def show_warnings(res, T_edge_C):
    """Labels for extrapolation: T above the Sips data and Henry's law (plus others)."""
    if res["T_bed_max_C"] > T_edge_C + 0.05:
        st.warning("Peak temperature " + str(round(res["T_bed_max_C"], 1)) + " °C is above "
                   + str(round(T_edge_C)) + " °C, the edge of the Sips data: q* is "
                   "extrapolated there (isosteric shift = Sips with "
                   "allow_extrapolation=True).")
    for text in res["warnings"]:
        st.warning(text)
    for text in res["notes"]:
        st.info(text)


def show_size_block(title, res, criteria, T_edge_C, note=None):
    """One size: heading, key numbers, criteria table (c), warnings."""
    case = res["case"]
    st.markdown("#### " + title)
    st.markdown("**D " + str(round(case["D_m"], 3)) + " m × H " + str(round(case["H_m"], 3))
                + " m** · volume " + str(round(res["volume_m3"], 4)) + " m³ · screw "
                + str(round(res["screw_rpm"], 1)) + " rpm · zeolite residence "
                + str(round(res["tau_zeo_s"] / 60, 2)) + " min")
    if note is not None:
        st.caption(note)
    st.dataframe(criteria_table(res, criteria), hide_index=True)
    show_warnings(res, T_edge_C)


# ---------------------------------------------------------------------------
# Sidebar inputs (spec: "ตัวเลือกบนหน้าเว็บ")
# ---------------------------------------------------------------------------
st.sidebar.header("Zeolite model")
zeo_label = st.sidebar.selectbox("Zeolite model", list(ZEOLITE_CHOICES))
zeo_source = ZEOLITE_CHOICES[zeo_label]
if zeo_source == "auto":
    gas_source = "auto"
else:
    gas_source = "stub"

st.sidebar.header("Process")
p_bar = st.sidebar.radio("Pressure (bar)", [30.0, 40.0], horizontal=True,
                         help="30 bar = default; 40 bar = proposal value (sorbent data "
                              "measured up to 30 bar, so 40 bar is an extrapolation)")
X = st.sidebar.slider("CO2 conversion per pass X (%)", 10, 60,
                      int(round(100 * PARAMS["X_conv"])), 1) / 100.0
S = st.sidebar.slider("Carbon to DME per pass S (%)", 0, 100,
                      int(round(100 * PARAMS["S_dme"])), 5) / 100.0
dH_method = HEATS[st.sidebar.selectbox("Heat of adsorption", list(HEATS))]

st.sidebar.header("Zeolite and screw")
d_pellet_mm = st.sidebar.slider("Pellet diameter (mm)", 1.0, 5.0, PARAMS["d_pellet_mm"],
                                0.1, help="Default 3 mm; the paper measured 1.6 mm pellets")
eta = st.sidebar.slider("Screw volumetric efficiency", 0.5, 1.0, PARAMS["eta_screw"], 0.05,
                        help="Default 0.9 (screw feeders, Bortolamasi & Fottner 2001); "
                             "0.5-1.0 not measured")
T_zeo_in_C = st.sidebar.slider("Zeolite feed temperature (°C)", 200.0, 250.0,
                               PARAMS["T_zeolite_in_C"], 5.0,
                               help="200-250 °C = range of the Sips data")
q_in = st.sidebar.slider("Loading after regeneration q_in (kg/kg)", 0.0, 0.05,
                         PARAMS["q_in_kgkg"], 0.005, format="%.3f",
                         help="0 = regenerated as in van Kampen (RCE 2021): PSA to 1-3 "
                              "bar(a) + dry inert purge (basis of the Sips working "
                              "isotherm). Positive = less complete regeneration.")

st.sidebar.header("Criteria")
criteria = {
    "capture_min": st.sidebar.slider("Water captured >= (%)", 80.0, 99.5,
                                     100 * PARAMS["capture_min"], 0.5) / 100.0,
    "lift_max": st.sidebar.slider("Gas drag / bed weight <=", 0.3, 1.0,
                                  PARAMS["lift_max_ratio"], 0.05),
    "dp_max_bar": st.sidebar.slider("Pressure drop <= (bar)", 0.1, 2.0,
                                    PARAMS["dp_max_bar"], 0.1),
    "T_bed_max_C": st.sidebar.slider("Peak temperature <= (°C)", 200.0, 300.0,
                                     PARAMS["T_bed_max_C"], 5.0),
    "HD_min": st.sidebar.slider("H/D >= (" + HD_NOTE + ")", 0.0, 5.0, PARAMS["HD_min"],
                                0.5),
}

with st.sidebar.expander("Comparison options (not the main model)"):
    if zeo_source == "auto":
        iso_model = ISOTHERMS[st.selectbox("Isotherm", list(ISOTHERMS))]
    else:
        iso_model = "sips"
        st.caption("GSTA choices need the Part 1 Zeolite3A")
    U_wall = st.number_input("Wall heat loss U (W/(m2 K)), default 0", 0.0, 200.0,
                             PARAMS["U_wall_Wm2K"], 5.0)

base_case = dz.updated_case(dz.default_case(PARAMS), {
    "X": X, "S": S, "p_bar": p_bar, "T_zeo_in_C": T_zeo_in_C, "q_in_kgkg": q_in,
    "d_pellet_mm": d_pellet_mm, "eta_screw": eta, "U_wall_Wm2K": U_wall,
    "isotherm_model": iso_model, "dH_method": dH_method})

try:
    zeolite_now, zeo_used, zeo_limits = get_zeolite(zeo_source, iso_model)
except dz.Part1NotFoundError:
    st.error("Part 1 not found: put zeolite.py, feed_gas.py and parameters.csv in "
             + str(dz.PART1_DIR) + ", or choose the Sips preview in the sidebar.")
    st.stop()
T_EDGE_C = zeo_limits["loading"]["T_max_K"] - 273.15        # 250 °C for Sips
P_HENRY_BAR = zeo_limits["loading"]["p_min_bar"]            # 0.05 bar

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("DSQ core: in-vessel water capture by moving zeolite 3A")
st.caption("Counter-current moving bed (gas up, zeolite down in the Archimedes screw), "
           "steady state, LDF kinetics, one temperature for gas and pellets, Ergun "
           "pressure drop. Basis 1 t CO2/day with recycle; gas in at "
           + str(round(PARAMS["T_gas_in_C"])) + " °C. Zeolite model: " + zeo_used + ".")
if zeo_used != "part1":
    st.info("Not the Part 1 Zeolite3A - use it only to try the program, not for the "
            "report.")

mode = st.radio("Mode", ["Check (I give the size)", "Design (find the size)"],
                horizontal=True)

if mode == "Check (I give the size)":
    c1, c2, c3 = st.columns(3)
    D_m = c1.number_input("Core diameter D (m)", 0.05, 1.5, PARAMS["D_m"], 0.01)
    H_m = c2.number_input("Bed height H (m)", 0.1, 10.0, PARAMS["H_m"], 0.05)
    m_zeo_kgh = c3.number_input("Zeolite circulation (kg/h)", 100.0, 50000.0,
                                PARAMS["m_zeolite_kgh"], 100.0,
                                help="Design mode finds the lowest rate that keeps the "
                                     "peak temperature at or below the limit")
    case = dz.sized_case(dz.updated_case(base_case, {"m_zeo_kgh": m_zeo_kgh}), D_m, H_m,
                         PARAMS["d_shaft_ratio"], PARAMS["pitch_ratio"])
    t_start = time.time()
    try:
        with st.spinner("Solving the core..."):
            res = cached_check(case, criteria, zeo_source, gas_source)
    except cc.CoreModelError as error:
        st.error(str(error))
        st.stop()
    except ValueError as error:          # also OutOfRangeError from the zeolite model
        st.error(str(error))
        st.stop()
    seconds = time.time() - t_start

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Water captured", dz.format_capture(res["capture_frac"]))
    m2.metric("Peak temperature (°C)", round(res["T_bed_max_C"], 1))
    m3.metric("Pressure drop (bar)", round(res["dp_bar"], 4))
    m4.metric("Gas drag / bed weight (max)", round(res["lift_ratio_max"], 3),
              help=LIFT_HELP)
    m5.metric("Screw speed (rpm)", round(res["screw_rpm"], 1))
    st.caption("Gas drag / bed weight: largest local value " + str(round(res["lift_ratio_max"], 3))
               + " at z = " + str(round(res["z_lift_max_m"], 2)) + " m (gas inlet); height "
               "average " + str(round(res["lift_ratio_avg"], 3)) + ". The criterion uses "
               "the largest local value.")
    st.caption("Solved in " + str(round(seconds, 2)) + " s (shaft d = "
               + str(round(case["d_shaft_m"], 3)) + " m, pitch = "
               + str(round(case["pitch_m"], 3)) + " m, both scale with D).")
    show_warnings(res, T_EDGE_C)

    st.markdown("**(c) Design criteria**")
    st.dataframe(criteria_table(res, criteria), hide_index=True)
    with st.expander("Summary of all results"):
        st.dataframe(dz.summary_table(res), hide_index=True)
    st.markdown("**(a) Profiles along the core height**")
    show_figure(plots.plot_core_profiles(res["profile"], res, T_EDGE_C, P_HENRY_BAR))
    with st.expander("Profile table"):
        st.dataframe(res["profile"])
    st.download_button(
        "Download results (Excel)",
        excel_bytes({"summary": dz.summary_table(res), "criteria":
                     criteria_table(res, criteria), "profile": res["profile"],
                     "inputs": pd.DataFrame([case]), "parameters": dz.load_param_table()},
                    "core_check.xlsx"),
        file_name="core_check.xlsx")

else:
    n_sizes = len(dz.DESIGN_D_LIST_M) * len(dz.DESIGN_H_LIST_M)
    st.markdown("Steps: (1) lowest zeolite rate with peak temperature ≤ "
                + str(round(criteria["T_bed_max_C"])) + " °C (bisection at D "
                + str(PARAMS["D_m"]) + " m × H " + str(PARAMS["H_m"]) + " m, where all "
                "water is captured = most heat); (2) at that rate, " + str(n_sizes)
                + " sizes (D " + str(dz.DESIGN_D_LIST_M[0]) + "–"
                + str(dz.DESIGN_D_LIST_M[-1]) + " m × H " + str(dz.DESIGN_H_LIST_M[0])
                + "–" + str(dz.DESIGN_H_LIST_M[-1]) + " m) → smallest passing all five "
                "criteria, height refined by bisection, and the recommended size; "
                "(3) lowest rate vs zeolite feed temperature (" + str(len(DESIGN_FEED_T_LIST_C))
                + " temperatures).")
    design_key = [base_case, criteria, zeo_source]
    if st.button("Run design mode (about 1–3 minutes)", type="primary"):
        with st.spinner("Design mode running: about 1–3 minutes on a laptop "
                        "(rate bisection, " + str(n_sizes) + " sizes, recommended size, "
                        "feed temperature table). Please wait..."):
            try:
                out = cached_design(base_case, criteria, zeo_source, gas_source,
                                    dz.DESIGN_D_LIST_M, dz.DESIGN_H_LIST_M,
                                    DESIGN_FEED_T_LIST_C)
            except cc.CoreModelError as error:
                st.error("Design mode stopped: " + str(error))
                st.stop()
            except ValueError as error:
                st.error("Design mode stopped: " + str(error))
                st.stop()
        st.session_state["design"] = [design_key, out]

    if "design" not in st.session_state:
        st.info("Press the button to run design mode. Results are kept (cached) for the "
                "same inputs, so pressing it again is instant.")
    else:
        saved_key, out = st.session_state["design"]
        if saved_key != design_key:
            st.info("Inputs changed since the last run - press the button again.")
        else:
            st.caption("Design mode took " + str(round(out["seconds"])) + " s on the "
                       "first run (later runs with the same inputs come from the cache).")
            if out["m_T_kgh"] is None:
                st.error(out["reason"] + ". No size can meet the temperature criterion "
                         "at this zeolite feed temperature.")
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric("Lowest zeolite rate (kg/h)", round(out["m_T_kgh"]))
                r2.metric("Capacity minimum (kg/h)", round(out["m_capacity_kgh"], 1))
                r3.metric("Ratio", round(out["ratio_T_to_capacity"], 1))
                st.caption("The zeolite rate is set by heat, not by capacity: the zeolite "
                           "carries the heat of adsorption away.")
                design = out["design"]
                rate_case = dz.updated_case(base_case, {"m_zeo_kgh": out["m_T_kgh"]})
                if design["best"] is None:
                    st.error(out["reason"])
                else:
                    if design["refined"] is not None:
                        res_small = design["refined"]["result"]
                    else:
                        res_small = cached_check(
                            dz.sized_case(rate_case, float(design["best"]["D_m"]),
                                          float(design["best"]["H_m"]),
                                          PARAMS["d_shaft_ratio"], PARAMS["pitch_ratio"]),
                            criteria, zeo_source, gas_source)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        show_size_block("(1) Smallest size meeting the criteria",
                                        res_small, criteria, T_EDGE_C,
                                        "No margin: k_LDF as given by the model.")
                    with col_b:
                        rec = out["recommended"]
                        if rec is None:
                            st.markdown("#### (2) Recommended size")
                            st.error("No size in the grid passes all criteria with k_LDF "
                                     "× " + str(PARAMS["k_ldf_factor_recommended"]))
                        else:
                            show_size_block(
                                "(2) Recommended size", rec["result_slow"], criteria,
                                T_EDGE_C, dz.K_LDF_MARGIN_TEXT + ". Criteria below are "
                                "with k_LDF × " + str(rec["k_ldf_factor"]) + "; with the "
                                "model k_LDF it captures "
                                + dz.format_capture(rec["result_nominal"]["capture_frac"])
                                + ".")
                    st.markdown("**(a) Profiles of the recommended size (model k_LDF)**")
                    if rec is not None:
                        res_rec = rec["result_nominal"]
                    else:
                        res_rec = res_small
                    show_figure(plots.plot_core_profiles(res_rec["profile"], res_rec,
                                                         T_EDGE_C, P_HENRY_BAR))

            st.markdown("**(b) Lowest zeolite rate vs zeolite feed temperature**")
            df_T = out["df_feed_T"]
            show_figure(plots.plot_min_rate_vs_feed(df_T, criteria))
            st.dataframe(df_T.round(2), hide_index=True)
            st.caption("Full 5 °C table: run python design.py. Where no rate passes, the "
                       "note says so; the criterion is not relaxed.")

            if out["design"] is not None:
                df = out["design"]["sweep"]
                with st.expander("All sizes in the sweep"):
                    tab_a, tab_b, tab_c = st.tabs(["Capture map", "Pressure drop & lift",
                                                   "Table"])
                    with tab_a:
                        show_figure(plots.plot_capture_heatmap(df, out["design"]["best"]))
                    with tab_b:
                        show_figure(plots.plot_size_check(df, criteria))
                    with tab_c:
                        st.dataframe(df, hide_index=True)
                st.download_button("Download design results (Excel)",
                                   excel_bytes({"design_sweep": df,
                                                "min_rate_vs_feed_T": df_T},
                                               "core_design.xlsx"),
                                   file_name="core_design.xlsx")

with st.expander("Parameters and sources (core_parameters.csv)"):
    st.dataframe(dz.load_param_table(), hide_index=True)
