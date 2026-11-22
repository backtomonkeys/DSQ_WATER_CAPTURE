"""
app.py - Streamlit web page for the DSQ water-capture core (Part 2).

Run (from the repo folder):   streamlit run part2_core/app.py
      (or inside part2_core):  streamlit run app.py
Streamlit is outside CPE 100; every calculation lives in design.py / core_column.py.

Check mode : the user gives size and zeolite rate -> one core solution (a few seconds),
             recomputed when "Run simulation" is pressed (the inputs sit in a form).
Design mode: runs only when its button is pressed (about 1-3 minutes); results cached.
Look (22 Nov 2026): sidebar "DESIGN INPUTS", black result cards, white panels
(PAGE_CSS below; colours also in .streamlit/config.toml).
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core_column as cc
import design as dz
import plots

# Logo: part2_core/logo.png if the team uploaded it (browser tab + page header);
# without the file the tab keeps Streamlit's icon and the header draws LOGO_SVG
LOGO_FILE = os.path.join(dz.HERE, "logo.png")
if os.path.isfile(LOGO_FILE):
    PAGE_ICON = LOGO_FILE
else:
    PAGE_ICON = None
st.set_page_config(page_title="DSQ Water-Capture Core", page_icon=PAGE_ICON,
                   layout="wide")

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
MODES = ["Check (I give the size)", "Design (find the size)"]
LOGO_SVG = ('<svg width="56" height="56" viewBox="0 0 56 56">'
            '<circle cx="28" cy="28" r="22" fill="none" stroke="#f8c45c" stroke-width="6"/>'
            '<path d="M28 6 A22 22 0 0 0 8 36" fill="none" stroke="#ff8d46" '
            'stroke-width="6"/>'
            '<path d="M28 15 C28 15 19 26 19 31 A9 9 0 0 0 37 31 C37 26 28 15 28 15 Z" '
            'fill="#000"/></svg>')

# Styling only (outside CPE 100): colours and shapes of the restyled page
PAGE_CSS = """
<style>
[data-testid="stSidebar"] {background-color: #f6f3ea;}
[data-testid="stSidebar"] [data-testid="stForm"] {border: none; padding: 0;}
.side-title {font-size: 0.8rem; letter-spacing: 0.18em; color: #6b6b6b;
             font-weight: 600; margin: 0.4rem 0 0.4rem 0;}
.dsq-title {font-family: Georgia, "Times New Roman", serif; font-size: 2.5rem;
            font-weight: 400; margin: 0; line-height: 1.15; color: #111;}
.dsq-subtitle {color: #6b6b6b; margin: 0.1rem 0 0.4rem 0; font-size: 1.05rem;}
.dsq-card {background: #000; color: #fff; border-radius: 14px; padding: 16px 20px;
           min-height: 118px; margin-bottom: 0.6rem;}
.dsq-card-label {font-size: 0.85rem; color: #cfcfcf;}
.dsq-card-value {font-size: 1.9rem; font-weight: 700; margin: 2px 0 6px 0;
                 white-space: nowrap; color: #fff;}
.dsq-pill {display: inline-block; border-radius: 999px; padding: 1px 10px;
           font-size: 0.75rem; font-weight: 700;}
.dsq-pill-pass {background: #ff8d46; color: #000;}
.dsq-pill-fail {background: #d03b3b; color: #fff;}
.dsq-panel {background: #fff; border: 1px solid #ece6cf; border-radius: 16px;
            padding: 18px 22px; margin-bottom: 0.6rem;}
.dsq-panel-title {font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;}
.dsq-row {display: flex; justify-content: space-between; padding: 9px 0;
          border-bottom: 1px solid #eeeeee;}
.dsq-ok {font-weight: 700;}
.dsq-no {font-weight: 700; color: #d03b3b;}
.dsq-all-pass {margin-top: 16px; text-align: center; font-weight: 700; font-size: 1.1rem;
               padding: 12px; border-radius: 10px; color: #000;
               background: linear-gradient(90deg, #f29a5c, #f8d96b);}
.dsq-some-fail {margin-top: 16px; text-align: center; font-weight: 700; font-size: 1.1rem;
                padding: 12px; border-radius: 10px; color: #fff; background: #d03b3b;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border-radius: 16px;}
.st-key-dsq_panel_check, .st-key-dsq_panel_design, .st-key-dsq_panel_rate {
    background: #fff; border-radius: 16px; padding: 4px 8px;}
.stButton, .stButton button, .stButton div, [data-testid="stFormSubmitButton"],
[data-testid="stFormSubmitButton"] div, [data-testid="stFormSubmitButton"] button,
.stFormSubmitButton, .stFormSubmitButton div, .stFormSubmitButton button {
    width: 100% !important;}
.stButton button, [data-testid="stFormSubmitButton"] button {
    background: #000; color: #fff; border: none; border-radius: 10px;
    font-weight: 700; padding: 0.6rem 0;}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
    background: #222; color: #fff;}
</style>
"""


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


def full_width_button(label, in_form=False):
    """Button across the whole column (black via PAGE_CSS). Newer Streamlit takes
    width="stretch"; older versions only know use_container_width=True."""
    try:
        if in_form:
            return st.form_submit_button(label, width="stretch")
        return st.button(label, type="primary", width="stretch")
    except TypeError:
        if in_form:
            return st.form_submit_button(label, use_container_width=True)
        return st.button(label, type="primary", use_container_width=True)


def nice_number(x):
    """95.0 -> "95", 0.7 -> "0.7" (for limits written in labels)."""
    value = round(x, 2)
    if value == int(value):
        return str(int(value))
    return str(value)


def card_html(label, value, passed=None, tooltip=""):
    """One black result card; passed = True / False adds a PASS / FAIL pill."""
    html = ('<div class="dsq-card" title="' + tooltip + '"><div class="dsq-card-label">'
            + label + '</div><div class="dsq-card-value">' + value + '</div>')
    if passed is True:
        html = html + '<span class="dsq-pill dsq-pill-pass">PASS</span>'
    elif passed is False:
        html = html + '<span class="dsq-pill dsq-pill-fail">FAIL</span>'
    return html + '</div>'


def show_result_cards(res):
    """The four black cards; values rounded for display (tables keep full precision)."""
    cards = [
        ["Water captured", dz.format_capture(res["capture_frac"]), res["pass_capture"]],
        ["Max core temperature", str(round(res["T_bed_max_C"], 1)) + " °C",
         res["pass_temperature"]],
        ["Pressure drop", str(round(res["dp_bar"], 3)) + " bar", res["pass_dp"]],
        ["Gas drag ratio", str(round(res["lift_ratio_max"], 2)), res["pass_lift"]],
    ]
    tooltips = ["", "", "", LIFT_HELP]
    columns = st.columns(4)
    for i in range(4):
        label, value, passed = cards[i]
        columns[i].markdown(card_html(label, value, passed, tooltips[i]),
                            unsafe_allow_html=True)


def criteria_panel_html(res, criteria):
    """Design-check panel: each criterion with the user's limit and a tick or cross."""
    rows = [
        ["Water removal ≥ " + nice_number(100 * criteria["capture_min"]) + "%",
         res["pass_capture"]],
        ["Gas drag ≤ " + nice_number(100 * criteria["lift_max"]) + "%", res["pass_lift"]],
        ["ΔP ≤ " + nice_number(criteria["dp_max_bar"]) + " bar", res["pass_dp"]],
        ["T ≤ " + nice_number(criteria["T_bed_max_C"]) + " °C", res["pass_temperature"]],
        ["H/D ≥ " + nice_number(criteria["HD_min"]), res["pass_aspect"]],
    ]
    html = '<div class="dsq-panel"><div class="dsq-panel-title">Design check</div>'
    n_pass = 0
    for label, passed in rows:
        if passed:
            n_pass = n_pass + 1
            mark = '<span class="dsq-ok">✓</span>'
        else:
            mark = '<span class="dsq-no">✗</span>'
        html = html + '<div class="dsq-row"><span>' + label + '</span>' + mark + '</div>'
    if n_pass == len(rows):
        html = html + '<div class="dsq-all-pass">All 5 criteria pass</div>'
    else:
        html = (html + '<div class="dsq-some-fail">' + str(n_pass) + ' of 5 criteria pass'
                '</div>')
    return html + '</div>'


def show_temperature_panel(res, criteria, T_edge_C, p_henry_bar, title, key):
    """White panel: "Temperature along the core" + the full profiles (a) in a tab.
    key gives the panel the CSS class st-key-<key> (white background)."""
    with st.container(border=True, key=key):
        st.markdown('<div class="dsq-panel-title">' + title + '</div>',
                    unsafe_allow_html=True)
        tab_T, tab_all = st.tabs(["Temperature", "All profiles (a)"])
        with tab_T:
            show_figure(plots.plot_temperature_along_core(res["profile"], res,
                                                          criteria["T_bed_max_C"]))
        with tab_all:
            show_figure(plots.plot_core_profiles(res["profile"], res, T_edge_C,
                                                 p_henry_bar))


def show_size_block(title, res, criteria, T_edge_C, note=None):
    """One size: heading, key numbers, design check panel, full table, warnings."""
    case = res["case"]
    st.markdown("#### " + title)
    st.markdown("**D " + str(round(case["D_m"], 3)) + " m × H " + str(round(case["H_m"], 3))
                + " m** · volume " + str(round(res["volume_m3"], 4)) + " m³ · screw "
                + str(round(res["screw_rpm"], 1)) + " rpm · zeolite residence "
                + str(round(res["tau_zeo_s"] / 60, 2)) + " min")
    if note is not None:
        st.caption(note)
    st.markdown(criteria_panel_html(res, criteria), unsafe_allow_html=True)
    with st.expander("Criteria table (full precision)"):
        st.dataframe(criteria_table(res, criteria), hide_index=True)
    show_warnings(res, T_edge_C)


# ---------------------------------------------------------------------------
# Sidebar: mode, then DESIGN INPUTS (5 inputs as in the mockup) and, folded away,
# every other input of the spec under "Advanced settings". The inputs sit in a
# form, so the page recomputes when "Run simulation" is pressed.
# ---------------------------------------------------------------------------
st.markdown(PAGE_CSS, unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", MODES)

with st.sidebar.form("design_inputs"):
    st.markdown('<div class="side-title">DESIGN INPUTS</div>', unsafe_allow_html=True)
    m_zeo_kgh = st.slider("Zeolite rate (kg/h)", 100.0, 50000.0, PARAMS["m_zeolite_kgh"],
                          100.0, format="%.0f",
                          help="Check mode only: design mode finds the lowest rate "
                               "that keeps the peak temperature at the limit")
    D_m = st.slider("Core diameter D (m)", 0.05, 1.5, PARAMS["D_m"], 0.01,
                    help="Check mode only (design mode sweeps D)")
    H_m = st.slider("Core height H (m)", 0.1, 10.0, PARAMS["H_m"], 0.05,
                    help="Check mode only (design mode sweeps H)")
    X = st.slider("CO2 conversion X (%)", 10, 60, int(round(100 * PARAMS["X_conv"])),
                  1) / 100.0
    S = st.slider("DME selectivity S (%)", 0, 100, int(round(100 * PARAMS["S_dme"])), 5,
                  help="Share of the converted carbon that goes to DME per pass; "
                       "the rest leaves as CO") / 100.0

    with st.expander("Advanced settings"):
        zeo_label = st.selectbox("Zeolite model", list(ZEOLITE_CHOICES))
        p_bar = st.radio("Pressure (bar)", [30.0, 40.0], horizontal=True,
                         help="30 bar = default; 40 bar = proposal value (sorbent data "
                              "measured up to 30 bar, so 40 bar is an extrapolation)")
        dH_method = HEATS[st.selectbox("Heat of adsorption", list(HEATS))]
        d_pellet_mm = st.slider("Pellet diameter (mm)", 1.0, 5.0, PARAMS["d_pellet_mm"],
                                0.1, help="Default 3 mm; the paper measured 1.6 mm")
        eta = st.slider("Screw volumetric efficiency", 0.5, 1.0, PARAMS["eta_screw"], 0.05,
                        help="Default 0.9 (screw feeders, Bortolamasi & Fottner 2001); "
                             "0.5-1.0 not measured")
        T_zeo_in_C = st.slider("Zeolite feed temperature (°C)", 200.0, 250.0,
                               PARAMS["T_zeolite_in_C"], 5.0,
                               help="200-250 °C = range of the Sips data")
        q_in = st.slider("Loading after regeneration q_in (kg/kg)", 0.0, 0.05,
                         PARAMS["q_in_kgkg"], 0.005, format="%.3f",
                         help="0 = regenerated as in van Kampen (RCE 2021): PSA to 1-3 "
                              "bar(a) + dry inert purge (basis of the Sips working "
                              "isotherm). Positive = less complete regeneration.")
        st.markdown("**Criteria**")
        criteria = {
            "capture_min": st.slider("Water captured >= (%)", 80.0, 99.5,
                                     100 * PARAMS["capture_min"], 0.5) / 100.0,
            "lift_max": st.slider("Gas drag / bed weight <=", 0.3, 1.0,
                                  PARAMS["lift_max_ratio"], 0.05),
            "dp_max_bar": st.slider("Pressure drop <= (bar)", 0.1, 2.0,
                                    PARAMS["dp_max_bar"], 0.1),
            "T_bed_max_C": st.slider("Peak temperature <= (°C)", 200.0, 300.0,
                                     PARAMS["T_bed_max_C"], 5.0),
            "HD_min": st.slider("H/D >= (" + HD_NOTE + ")", 0.0, 5.0, PARAMS["HD_min"],
                                0.5),
        }
        st.markdown("**Comparison options (not the main model)**")
        iso_label = st.selectbox("Isotherm (GSTA choices need the Part 1 Zeolite3A)",
                                 list(ISOTHERMS))
        U_wall = st.number_input("Wall heat loss U (W/(m2 K)), default 0", 0.0, 200.0,
                                 PARAMS["U_wall_Wm2K"], 5.0)

    full_width_button("Run simulation", in_form=True)

zeo_source = ZEOLITE_CHOICES[zeo_label]
if zeo_source == "auto":
    gas_source = "auto"
    iso_model = ISOTHERMS[iso_label]
else:
    gas_source = "stub"
    iso_model = "sips"                  # GSTA choices need the Part 1 Zeolite3A

base_case = dz.updated_case(dz.default_case(PARAMS), {
    "X": X, "S": S, "p_bar": p_bar, "T_zeo_in_C": T_zeo_in_C, "q_in_kgkg": q_in,
    "d_pellet_mm": d_pellet_mm, "eta_screw": eta, "U_wall_Wm2K": U_wall,
    "isotherm_model": iso_model, "dH_method": dH_method})

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 13])
with col_logo:
    if os.path.isfile(LOGO_FILE):
        st.image(LOGO_FILE, width=60)
    else:
        st.markdown(LOGO_SVG, unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="dsq-title">DSQ Water-Capture Core</div>'
                '<div class="dsq-subtitle">Moving-bed zeolite 3A core for Green-DME '
                'synthesis</div>', unsafe_allow_html=True)

try:
    zeolite_now, zeo_used, zeo_limits = get_zeolite(zeo_source, iso_model)
except dz.Part1NotFoundError:
    st.error("Part 1 not found: put zeolite.py, feed_gas.py and parameters.csv in "
             + str(dz.PART1_DIR) + ", or choose the Sips preview under Advanced settings.")
    st.stop()
T_EDGE_C = zeo_limits["loading"]["T_max_K"] - 273.15        # 250 °C for Sips
P_HENRY_BAR = zeo_limits["loading"]["p_min_bar"]            # 0.05 bar
if zeo_used != "part1":
    st.info("Zeolite model: " + zeo_used + " - not the Part 1 Zeolite3A; use it only to "
            "try the program, not for the report.")

MODEL_NOTE = ("Counter-current moving bed (gas up, zeolite down in the Archimedes screw), "
              "steady state, LDF kinetics, one temperature for gas and pellets, Ergun "
              "pressure drop. Basis 1 t CO2/day with recycle; gas in at "
              + str(round(PARAMS["T_gas_in_C"])) + " °C. Zeolite model: " + zeo_used + ".")

if mode == MODES[0]:
    # ------------------------------------------------------------------ check mode
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

    show_result_cards(res)
    col_plot, col_check = st.columns([2, 1])
    with col_plot:
        show_temperature_panel(res, criteria, T_EDGE_C, P_HENRY_BAR,
                               "Temperature along the core", "dsq_panel_check")
    with col_check:
        st.markdown(criteria_panel_html(res, criteria), unsafe_allow_html=True)

    show_warnings(res, T_EDGE_C)
    with st.expander("Model notes"):
        st.caption(MODEL_NOTE)
        st.caption("Gas drag ratio = gas drag / bed weight: largest local value "
                   + str(round(res["lift_ratio_max"], 3)) + " at z = "
                   + str(round(res["z_lift_max_m"], 2)) + " m (gas inlet); height average "
                   + str(round(res["lift_ratio_avg"], 3)) + ". " + LIFT_HELP)
        st.caption("Solved in " + str(round(seconds, 2)) + " s (shaft d = "
                   + str(round(case["d_shaft_m"], 3)) + " m, pitch = "
                   + str(round(case["pitch_m"], 3)) + " m, both scale with D; screw "
                   + str(round(res["screw_rpm"], 1)) + " rpm).")
    with st.expander("Criteria table (full precision)"):
        st.dataframe(criteria_table(res, criteria), hide_index=True)
    with st.expander("Summary of all results"):
        st.dataframe(dz.summary_table(res), hide_index=True)
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
    # ------------------------------------------------------------------ design mode
    n_sizes = len(dz.DESIGN_D_LIST_M) * len(dz.DESIGN_H_LIST_M)
    st.caption("Design mode finds the zeolite rate, D and H itself: the first three "
               "DESIGN INPUTS are not used here (X, S and Advanced settings are).")
    design_key = [base_case, criteria, zeo_source]
    if full_width_button("Run design mode (about 1–3 minutes)"):
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
            if out["m_T_kgh"] is None:
                st.error(out["reason"] + ". No size can meet the temperature criterion "
                         "at this zeolite feed temperature.")
            else:
                columns = st.columns(3)
                columns[0].markdown(card_html("Lowest zeolite rate",
                                              str(round(out["m_T_kgh"])) + " kg/h"),
                                    unsafe_allow_html=True)
                columns[1].markdown(card_html("Capacity minimum",
                                              str(round(out["m_capacity_kgh"], 1))
                                              + " kg/h"), unsafe_allow_html=True)
                columns[2].markdown(card_html("Ratio (heat / capacity)",
                                              str(round(out["ratio_T_to_capacity"], 1))
                                              + " ×"), unsafe_allow_html=True)
                design = out["design"]
                rate_case = dz.updated_case(base_case, {"m_zeo_kgh": out["m_T_kgh"]})
                rec = None
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
                    rec = out["recommended"]
                    col_a, col_b = st.columns(2)
                    with col_a:
                        show_size_block("(1) Smallest size meeting the criteria",
                                        res_small, criteria, T_EDGE_C,
                                        "No margin: k_LDF as given by the model.")
                    with col_b:
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
                    if rec is not None:
                        res_rec = rec["result_nominal"]
                    else:
                        res_rec = res_small
                    show_temperature_panel(res_rec, criteria, T_EDGE_C, P_HENRY_BAR,
                                           "Recommended size: temperature along the core "
                                           "(model k_LDF)", "dsq_panel_design")

            df_T = out["df_feed_T"]
            with st.container(border=True, key="dsq_panel_rate"):
                st.markdown('<div class="dsq-panel-title">(b) Lowest zeolite rate vs '
                            'zeolite feed temperature</div>', unsafe_allow_html=True)
                show_figure(plots.plot_min_rate_vs_feed(df_T, criteria))
            st.dataframe(df_T.round(2), hide_index=True)
            with st.expander("Model notes"):
                st.caption(MODEL_NOTE)
                st.caption("Steps: (1) lowest zeolite rate with peak temperature ≤ "
                           + str(round(criteria["T_bed_max_C"])) + " °C (bisection at D "
                           + str(PARAMS["D_m"]) + " m × H " + str(PARAMS["H_m"]) + " m, "
                           "where all water is captured = most heat); (2) at that rate, "
                           + str(n_sizes) + " sizes (D " + str(dz.DESIGN_D_LIST_M[0]) + "–"
                           + str(dz.DESIGN_D_LIST_M[-1]) + " m × H "
                           + str(dz.DESIGN_H_LIST_M[0]) + "–" + str(dz.DESIGN_H_LIST_M[-1])
                           + " m) → smallest passing all five criteria, height refined by "
                           "bisection, and the recommended size; (3) lowest rate vs "
                           "zeolite feed temperature (" + str(len(DESIGN_FEED_T_LIST_C))
                           + " temperatures; full 5 °C table: python design.py). Where no "
                           "rate passes, the table says so; the criterion is not relaxed.")
                st.caption("Design mode took " + str(round(out["seconds"])) + " s on the "
                           "first run (later runs with the same inputs come from the "
                           "cache).")
                st.caption("The zeolite rate is set by heat, not by capacity: the zeolite "
                           "carries the heat of adsorption away.")

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
