"""
core_column.py - Steady-state counter-current moving-bed water-capture core.

Physical picture
----------------
* Gas flows UP through the core; zeolite 3A pellets move DOWN (plug flow) in the
  fully packed channel of the Archimedes screw (annulus between casing D and shaft d).
* Water uptake by the linear driving force (LDF) model, dq/dt = k_LDF (q* - q).
* Local thermal equilibrium: gas and pellets have one temperature T(z)
  (the gas-solid heat-transfer length for 3 mm beads is only a few mm).
* Constant heat capacities, evaluated at the gas inlet temperature.

Balances per unit height (z upward, N = adsorption rate [mol H2O / (s m)])
    N        = rho_bulk * A * k_LDF(T, pH2O) * (q*(T, pH2O) - q) / M_H2O
    dF_w/dz  = -N                        water in the gas [mol/s]
    dq/dz    = -N * M_H2O / m_zeo        zeolite loading [kg/kg] (zeolite moves down)
    dW * dT/dz = N * (-dH_ads) - U * pi * D * (T - T_wall)
dW = W_gas - W_solid [W/K] is the NET UPWARD heat-capacity flow. Water moved from the
gas to the solid carries the same cp, so dW is constant along the column.

Thermal regimes (chosen with match-case)
----------------------------------------
* dW > 0 "gas_carries_heat"  : bed temperature fixed at the bottom by the gas inlet;
  the bed heats up towards the top by Q_ads / dW.
* dW < 0 "solid_carries_heat": bed temperature fixed at the top by the zeolite feed;
  the bottom is hotter by Q_ads / |dW|.
* dW ~ 0 "balanced"           : the heat is trapped -> ThermalPinchError.
So the temperature rise is Q / |W_gas - W_solid|, not Q / (W_gas + W_solid).

Numerical method
----------------
Shooting: a for-loop marches up the column one height step at a time (RK4); a
while-loop bisection finds the zeolite OUTLET loading q_out that makes the loading at
the top equal the feed loading q_in. If round-off makes the march drift before the
top (long columns), the march restarts from the last height where the two bracketing
marches still agree. An outer while-loop updates the pressure profile (Ergun) and the
wall heat loss until nothing changes.

All inputs are dictionaries (see solve_column). The isotherm and the LDF functions are
passed in as arguments.
"""

import math

import numpy as np
import pandas as pd

import gas_props

R_JMOLK = 8.314462618
G_MS2 = 9.80665
M_W_KGMOL = 0.01801528
BAR_PA = 1.0e5

# marching / shooting controls
LAM_H_MAX = 1.0       # largest (stiffness x substep) allowed for RK4
DT_MAX_K = 2.0        # largest temperature change per substep [K]
MAX_SUB = 400         # largest number of substeps per height step
T_FLOOR_K = 250.0     # leaving these limits tells us the guess was too low / too high
T_CEIL_K = 2000.0
Q_CAP_KGKG = 1.0
DIV_TOL = 1e-6        # scaled difference at which two marches have "drifted apart"
OK_TOL = 1e-9         # scaled difference still accepted for a restart height
MAX_SEGMENTS = 60

# data range of the zeolite functions (interface contract) and extrapolation constants;
# design.py replaces the ranges with the ones the real zeolite model accepts
DEFAULT_LIMITS = {
    "loading": {"T_min_K": 473.15, "T_max_K": 623.15, "p_min_bar": 0.05, "p_max_bar": 4.5},
    "k_ldf": {"T_min_K": 473.15, "T_max_K": 623.15, "p_min_bar": 0.05, "p_max_bar": 4.5},
    "E_iso_shift_Jmol": 60.1e3,
    "Ea_kldf_Jmol": 54.2e3,
}


class CoreModelError(Exception):
    """Base class for errors of the core-column model."""
    pass


class ThermalPinchError(CoreModelError):
    """Gas and zeolite heat-capacity flows are almost equal (no steady solution)."""
    pass


class BracketError(CoreModelError):
    """The bisection bracket has no sign change."""
    pass


class ShootingError(CoreModelError):
    """The march could not be carried through the column."""
    pass


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def check_geometry(geom):
    """Raise ValueError for an impossible core / screw geometry."""
    if geom["D_m"] <= 0 or geom["H_m"] <= 0 or geom["pitch_m"] <= 0:
        raise ValueError("D, H and pitch must be positive")
    if geom["d_shaft_m"] < 0 or geom["d_shaft_m"] >= geom["D_m"]:
        raise ValueError("Shaft diameter must satisfy 0 <= d < D")


def flow_area_m2(geom):
    """Cross-section of the packed annulus between casing D and shaft d [m2]."""
    return math.pi / 4.0 * (geom["D_m"] ** 2 - geom["d_shaft_m"] ** 2)


def outer_volume_m3(geom):
    """pi/4 D^2 H [m3], used as the 'size' of the core in design mode."""
    return math.pi / 4.0 * geom["D_m"] ** 2 * geom["H_m"]


# ---------------------------------------------------------------------------
# Zeolite properties, also outside the data range
# ---------------------------------------------------------------------------
def shifted_point(T_K, p_bar, lim, E_shift_Jmol):
    """Clip T to the data range lim and shift p so that b(T) * p stays the same.

    For the Sips isotherm, q* depends on b(T) * p only, so
    q*(T, p) = q*(T_c, p * b(T) / b(T_c)) is exact. Returns (T_c, p_eq).
    """
    if T_K < lim["T_min_K"]:
        T_c = lim["T_min_K"]
    elif T_K > lim["T_max_K"]:
        T_c = lim["T_max_K"]
    else:
        return T_K, p_bar
    p_eq = p_bar * math.exp(E_shift_Jmol / R_JMOLK * (1.0 / T_K - 1.0 / T_c))
    return T_c, p_eq


def qstar_safe(T_K, p_bar, col):
    """Equilibrium loading q* [kg/kg] with the chosen out-of-range policy.

    "raise": call the isotherm directly (OutOfRangeError reaches the caller)
    "clip" : stay inside the data range; T by the shift above, pH2O < p_min by
             Henry's law (q* proportional to p), pH2O > p_max held at p_max
    """
    loading = col["loading_fn"]
    model = col["isotherm_model"]
    match col["policy"]:
        case "raise":
            return loading(T_K, p_bar, model)
        case "clip":
            if p_bar <= 0.0:
                return 0.0
            lim = col["limits_q"]
            T_c, p_eq = shifted_point(T_K, p_bar, lim, col["E_iso_shift_Jmol"])
            if p_eq < lim["p_min_bar"]:
                # Henry's law: K_H = q_Sips(T, p_min) / p_min at the LOCAL temperature,
                # so q* is continuous at p_min and goes to 0 as p goes to 0
                return loading(T_c, lim["p_min_bar"], model) * p_eq / lim["p_min_bar"]
            if p_eq > lim["p_max_bar"]:
                return loading(T_c, lim["p_max_bar"], model)
            return loading(T_c, p_eq, model)
        case _:
            raise ValueError("Unknown policy: " + str(col["policy"]))


def kldf_safe(T_K, p_bar, col):
    """LDF coefficient [1/s] with the chosen out-of-range policy (Arrhenius in T),
    times col["k_ldf_factor"] (Part 2 only; Part 1 is not changed)."""
    return col.get("k_ldf_factor", 1.0) * kldf_policy(T_K, p_bar, col)


def kldf_policy(T_K, p_bar, col):
    """k_LDF [1/s] from the zeolite function (or table) with the out-of-range policy."""
    match col["policy"]:
        case "raise":
            return col["kldf_fn"](T_K, p_bar)
        case "clip":
            if p_bar < 0.0:
                p_bar = 0.0
            lim = col["limits_k"]
            T_c, p_eq = shifted_point(T_K, p_bar, lim, col["E_iso_shift_Jmol"])
            p_c = min(max(p_eq, lim["p_min_bar"]), lim["p_max_bar"])
            k_1s = col["kldf_fn"](T_c, p_c)
            if T_c != T_K:
                k_1s = k_1s * math.exp(-col["Ea_kldf_Jmol"] / R_JMOLK
                                       * (1.0 / T_K - 1.0 / T_c))
            return k_1s
        case _:
            raise ValueError("Unknown policy: " + str(col["policy"]))


# ---------------------------------------------------------------------------
# Small engineering helpers
# ---------------------------------------------------------------------------
def ergun_dpdz_Pa_m(u_ms, mu_Pas, rho_kgm3, eps, d_p_m):
    """Ergun pressure gradient [Pa/m] (u = superficial velocity relative to the bed)."""
    viscous = 150.0 * mu_Pas * (1.0 - eps) ** 2 * u_ms / (eps ** 3 * d_p_m ** 2)
    inertial = 1.75 * rho_kgm3 * (1.0 - eps) * u_ms ** 2 / (eps ** 3 * d_p_m)
    return viscous + inertial


def classify_regime(dW_hi_WK, dW_lo_WK, W_ref_WK, tol):
    """Thermal regime from the range of dW = W_gas - W_solid over the q_out bracket."""
    if dW_lo_WK > tol * W_ref_WK:
        return "gas_carries_heat"
    if dW_hi_WK < -tol * W_ref_WK:
        return "solid_carries_heat"
    return "balanced"


def gas_heat_capacity(gas_mols, T_ref_K, gas_data):
    """Return (W of the dry gas [W/K], cp of water vapour [J/(mol K)]) at T_ref_K."""
    W_dry_WK = 0.0
    for species in gas_mols:
        if species != "H2O":
            W_dry_WK = W_dry_WK + gas_mols[species] * gas_props.cp_JmolK(
                species, T_ref_K, gas_data)
    cp_v_JmolK = gas_props.cp_JmolK("H2O", T_ref_K, gas_data)
    return W_dry_WK, cp_v_JmolK


def balanced_zeolite_rate_kgs(gas_mols, T_ref_K, cp_zeolite_JkgK, q_in_kgkg,
                              gas_data=None):
    """Zeolite rate [kg/s] at which W_solid = W_gas at the inlets (AVOID this rate)."""
    if gas_data is None:
        gas_data = gas_props.load_gas_data()
    gas = gas_props.check_species(gas_mols, gas_data)
    W_dry_WK, cp_v = gas_heat_capacity(gas, T_ref_K, gas_data)
    W_gas_in_WK = W_dry_WK + cp_v * gas["H2O"]
    return W_gas_in_WK / (cp_zeolite_JkgK + cp_v * q_in_kgkg / M_W_KGMOL)


def net_heat_capacity_flow(q_out, col):
    """dW = W_gas - W_solid [W/K] for a zeolite outlet loading q_out (same at every z)."""
    return (col["W_dry"] - col["W_s0"]
            + col["cp_v"] * (col["F_w_in"] - col["m_s"] * q_out / M_W_KGMOL))


def anchor_temperature(q_out, col):
    """Bed temperature at the bottom (z = 0) for a guessed q_out."""
    match col["regime"]:
        case "gas_carries_heat":
            return col["T_gas_in_K"]
        case "solid_carries_heat":
            # energy balance: the top must end at the zeolite feed temperature
            Q_ads_W = col["mdH"] * col["m_s"] * (q_out - col["q_in"]) / M_W_KGMOL
            dW = net_heat_capacity_flow(q_out, col)
            return col["T_zeo_in_K"] + (Q_ads_W - col["Qw_est"]) / (-dW)
        case _:
            raise ThermalPinchError("No anchor temperature in the balanced regime")


# ---------------------------------------------------------------------------
# One height step (RK4)
# ---------------------------------------------------------------------------
def pressure_at(z_m, col):
    """Total pressure [bar] at height z by linear interpolation between the nodes."""
    p_nodes = col["p_nodes"]
    x = z_m / col["h"]
    i = int(x)
    if i >= col["n"]:
        return p_nodes[col["n"]]
    if i < 0:
        return p_nodes[0]
    return p_nodes[i] + (x - i) * (p_nodes[i + 1] - p_nodes[i])


def derivatives(z_m, F_w, q, T_K, col, dW):
    """Right-hand side of the balances: [dF_w/dz, dq/dz, dT/dz, wall heat loss/dz]."""
    p_bar = pressure_at(z_m, col)
    if F_w > 0.0:
        Fw_pos = F_w
    else:
        Fw_pos = 0.0
    pw_bar = Fw_pos * p_bar / (col["F_dry"] + Fw_pos)
    # keep the property calls in a sensible temperature window
    if T_K < T_FLOOR_K:
        T_prop = T_FLOOR_K
    elif T_K > T_CEIL_K:
        T_prop = T_CEIL_K
    else:
        T_prop = T_K
    N = (col["rhoA"] / M_W_KGMOL * kldf_safe(T_prop, pw_bar, col)
         * (qstar_safe(T_prop, pw_bar, col) - q))
    wall_W_m = col["UP"] * (T_K - col["T_wall_K"])
    dFw_dz = -N
    dq_dz = -N * M_W_KGMOL / col["m_s"]
    dT_dz = (N * col["mdH"] - wall_W_m) / dW
    return [dFw_dz, dq_dz, dT_dz, wall_W_m]


def stiffness(z_m, F_w, q, T_K, col, dW):
    """Rough largest rate of change [1/m] of the system, used to choose the substep."""
    p_bar = pressure_at(z_m, col)
    if F_w > 0.0:
        Fw_pos = F_w
    else:
        Fw_pos = 0.0
    F_tot = col["F_dry"] + Fw_pos
    pw_bar = Fw_pos * p_bar / F_tot
    if T_K < T_FLOOR_K:
        T_K = T_FLOOR_K
    elif T_K > T_CEIL_K:
        T_K = T_CEIL_K
    k_1s = kldf_safe(T_K, pw_bar, col)
    q_eq = qstar_safe(T_K, pw_bar, col)
    dp = 1e-3 * pw_bar + 1e-7
    dq_dp = (qstar_safe(T_K, pw_bar + dp, col) - q_eq) / dp
    dq_dT = (qstar_safe(T_K + 0.5, pw_bar, col) - q_eq) / 0.5
    rhoA_M = col["rhoA"] / M_W_KGMOL
    lam_gas = rhoA_M * k_1s * abs(dq_dp) * p_bar * col["F_dry"] / (F_tot * F_tot)
    lam_solid = rhoA_M * k_1s * M_W_KGMOL / col["m_s"]
    lam_heat = rhoA_M * k_1s * abs(dq_dT) * col["mdH"] / abs(dW)
    lam_wall = col["UP"] / abs(dW)
    return lam_gas + lam_solid + lam_heat + lam_wall


def rk4_step(z_m, state, h_m, col, dW):
    """One classical Runge-Kutta step. state = [F_w, q, T, Qw] -> new state."""
    k1 = derivatives(z_m, state[0], state[1], state[2], col, dW)
    s2 = []
    for i in range(4):
        s2.append(state[i] + 0.5 * h_m * k1[i])
    k2 = derivatives(z_m + 0.5 * h_m, s2[0], s2[1], s2[2], col, dW)
    s3 = []
    for i in range(4):
        s3.append(state[i] + 0.5 * h_m * k2[i])
    k3 = derivatives(z_m + 0.5 * h_m, s3[0], s3[1], s3[2], col, dW)
    s4 = []
    for i in range(4):
        s4.append(state[i] + h_m * k3[i])
    k4 = derivatives(z_m + h_m, s4[0], s4[1], s4[2], col, dW)
    new_state = []
    for i in range(4):
        new_state.append(state[i] + h_m / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    return new_state


# ---------------------------------------------------------------------------
# March up the column (for-loop over height)
# ---------------------------------------------------------------------------
def march_up(i0, start_state, col):
    """March from node i0 to the top.

    start_state = [F_w, q, T, Qw] at node i0.
    Returns (residual, rows); residual = q(top) - q_in, or -1 / +1 when the march
    leaves the physical range early (the guess was clearly too low / too high).
    rows = [[node, F_w, q, T, Qw], ...] for every node reached.
    """
    n = col["n"]
    h = col["h"]
    F_w, q, T_K, Qw = start_state
    # (F_w - m_s q / M) does not change with z, so dW is the same for the whole march
    dW = col["W_dry"] - col["W_s0"] + col["cp_v"] * (F_w - col["m_s"] * q / M_W_KGMOL)
    rows = [[i0, F_w, q, T_K, Qw]]

    for i in range(i0, n):
        z_m = i * h
        # choose substeps so that RK4 stays accurate for stiff parts of the column
        lam = stiffness(z_m, F_w, q, T_K, col, dW)
        slope = derivatives(z_m, F_w, q, T_K, col, dW)
        m_sub = int(h * lam / LAM_H_MAX) + 1
        dT_estimate = abs(slope[2]) * h
        if dT_estimate > DT_MAX_K:
            m_sub = max(m_sub, int(dT_estimate / DT_MAX_K) + 1)
        m_sub = min(m_sub, MAX_SUB)
        h_sub = h / m_sub

        state = [F_w, q, T_K, Qw]
        for j in range(m_sub):
            state = rk4_step(z_m, state, h_sub, col, dW)
            z_m = z_m + h_sub
            F_w, q, T_K, Qw = state
            # small negative F_w is round-off near complete capture (treated as 0)
            if q < -1e-6 or F_w < -1e-3 * col["F_w_in"] or T_K < T_FLOOR_K:
                return -1.0, rows
            if q > Q_CAP_KGKG or F_w > 50.0 * col["F_w_in"] or T_K > T_CEIL_K:
                return 1.0, rows
            if not (math.isfinite(q) and math.isfinite(T_K) and math.isfinite(F_w)):
                raise ShootingError("The march produced NaN (system too stiff)")
        rows.append([i + 1, F_w, q, T_K, Qw])

    return q - col["q_in"], rows


def start_state(q_guess, F_start, T_fixed, Qw_start, col):
    """State at the start node; T comes from the anchor rule unless it is fixed."""
    if T_fixed is None:
        T_start = anchor_temperature(q_guess, col)
    else:
        T_start = T_fixed
    return [F_start, q_guess, T_start, Qw_start]


def state_difference(row_a, row_b, F_w_in):
    """Scaled difference between two marched states at the same node."""
    d_q = abs(row_a[2] - row_b[2]) / 0.1
    d_F = abs(row_a[1] - row_b[1]) / F_w_in
    d_T = abs(row_a[3] - row_b[3]) / 100.0
    d = max(d_q, d_F, d_T)
    if not math.isfinite(d):
        return float("inf")
    return d


def middle_row(row_a, row_b):
    """Average of two marched states at the same node."""
    row = [row_a[0]]
    for i in range(1, 5):
        row.append(0.5 * (row_a[i] + row_b[i]))
    return row


# ---------------------------------------------------------------------------
# Shooting: while-loop bisection on the zeolite outlet loading
# ---------------------------------------------------------------------------
def shoot(col, q_lo, q_hi):
    """Find the zeolite outlet loading by bisection; restart the march if it drifts.

    Returns (rows, stats); rows hold the solution at nodes 0..n.
    """
    n = col["n"]
    i0 = 0
    F_start = col["F_w_in"]
    Qw_start = 0.0
    T_fixed = None
    accepted = []
    stats = {"n_march": 0, "n_bisect": 0, "n_segments": 0}

    segment = 0
    while segment < MAX_SEGMENTS:
        segment = segment + 1
        stats["n_segments"] = segment

        # 1) the bracket must contain a sign change
        r_lo = march_up(i0, start_state(q_lo, F_start, T_fixed, Qw_start, col), col)[0]
        r_hi, rows_hi = march_up(i0, start_state(q_hi, F_start, T_fixed, Qw_start, col),
                                 col)
        stats["n_march"] = stats["n_march"] + 2

        # complete capture: the upper end (all water taken up) already fits the top
        if segment == 1 and q_hi == col["q_hi_max"] and r_lo < 0.0 and r_hi <= 0.0:
            if len(rows_hi) == n + 1 and abs(r_hi) < 1e-7:
                return rows_hi, stats

        n_widen = 0
        while r_lo * r_hi > 0.0:
            if segment == 1 or n_widen >= 8:
                raise BracketError("No sign change for q between " + str(q_lo)
                                   + " and " + str(q_hi) + " kg/kg")
            middle = 0.5 * (q_lo + q_hi)
            width = 10.0 * max(q_hi - q_lo, 1e-13)
            q_lo = middle - width
            q_hi = middle + width
            r_lo = march_up(i0, start_state(q_lo, F_start, T_fixed, Qw_start, col), col)[0]
            r_hi = march_up(i0, start_state(q_hi, F_start, T_fixed, Qw_start, col), col)[0]
            stats["n_march"] = stats["n_march"] + 2
            n_widen = n_widen + 1

        # 2) bisection
        while q_hi - q_lo > col["tol_q"]:
            q_mid = 0.5 * (q_lo + q_hi)
            if q_mid <= q_lo or q_mid >= q_hi:
                break                                   # no more digits to gain
            r_mid = march_up(i0, start_state(q_mid, F_start, T_fixed, Qw_start, col),
                             col)[0]
            stats["n_march"] = stats["n_march"] + 1
            stats["n_bisect"] = stats["n_bisect"] + 1
            if r_mid == 0.0:
                q_lo = q_mid
                q_hi = q_mid
                break
            if (r_mid > 0.0) == (r_lo > 0.0):
                q_lo = q_mid
                r_lo = r_mid
            else:
                q_hi = q_mid
                r_hi = r_mid

        # 3) march with both ends of the final bracket and compare them
        rows_lo = march_up(i0, start_state(q_lo, F_start, T_fixed, Qw_start, col), col)[1]
        rows_hi = march_up(i0, start_state(q_hi, F_start, T_fixed, Qw_start, col), col)[1]
        stats["n_march"] = stats["n_march"] + 2
        n_common = min(len(rows_lo), len(rows_hi))
        k_drift = None
        k_ok = 0
        for k in range(n_common):
            d = state_difference(rows_lo[k], rows_hi[k], col["F_w_in"])
            if d > DIV_TOL:
                k_drift = k
                break
            if d < OK_TOL:
                k_ok = k

        complete = len(rows_lo) == n - i0 + 1 and len(rows_hi) == n - i0 + 1
        if k_drift is None and complete:
            mids = []
            for k in range(n_common):
                mids.append(middle_row(rows_lo[k], rows_hi[k]))
            if len(accepted) == 0:
                accepted.extend(mids)
            else:
                accepted.extend(mids[1:])
            return accepted, stats

        # 4) the marches drifted apart: keep the good part and restart from there
        if k_ok == 0:
            raise ShootingError("Shooting drifts within one height step; "
                                "increase n_steps")
        mids = []
        for k in range(k_ok + 1):
            mids.append(middle_row(rows_lo[k], rows_hi[k]))
        if len(accepted) == 0:
            accepted.extend(mids)
        else:
            accepted.extend(mids[1:])
        last = accepted[-1]
        i0 = last[0]
        F_start = last[1]
        T_fixed = last[3]
        Qw_start = last[4]
        q_a = rows_lo[k_ok][2]
        q_b = rows_hi[k_ok][2]
        middle = 0.5 * (q_a + q_b)
        width = max(abs(q_b - q_a), 1e-12)
        q_lo = middle - 5.0 * width
        q_hi = middle + 5.0 * width

    raise ShootingError("More than " + str(MAX_SEGMENTS) + " restart segments needed")


# ---------------------------------------------------------------------------
# Profiles along the column (NumPy arrays): pressure drop and lift check
# ---------------------------------------------------------------------------
def trapezoid_integral(y, x):
    """Integral of y over x by the trapezoid rule (for-loop)."""
    total = 0.0
    for i in range(1, len(x)):
        total = total + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def profile_arrays(rows, col):
    """Turn the marched rows into NumPy arrays and compute the Ergun profile."""
    table = np.array(rows)                      # columns: node, F_w, q, T, Qw
    z_m = table[:, 0] * col["h"]
    F_w = table[:, 1]
    q = table[:, 2]
    T_K = table[:, 3]
    Qw = table[:, 4]
    p_bar = np.array(col["p_nodes"])

    F_w_pos = F_w.copy()
    F_w_pos[F_w_pos < 0.0] = 0.0                # boolean mask: no negative water
    F_tot = col["F_dry"] + F_w_pos
    pw_bar = F_w_pos / F_tot * p_bar

    qstar_list = []
    k_list = []
    for i in range(len(T_K)):
        qstar_list.append(qstar_safe(T_K[i], pw_bar[i], col))
        k_list.append(kldf_safe(T_K[i], pw_bar[i], col))
    qstar = np.array(qstar_list)
    k_1s = np.array(k_list)
    N_mol_sm = col["rhoA"] * k_1s * (qstar - q) / M_W_KGMOL

    # gas composition, density, viscosity, velocity
    y = {}
    for species in col["gas"]:
        if species != "H2O":
            y[species] = col["gas"][species] / F_tot
    y["H2O"] = F_w_pos / F_tot
    M_mix = gas_props.mixture_molar_mass_kgmol(y, col["gas_data"])
    rho_g = p_bar * BAR_PA * M_mix / (R_JMOLK * T_K)
    mu_g = gas_props.mixture_viscosity_Pas(y, T_K, col["gas_data"])
    u_g = F_tot * R_JMOLK * T_K / (p_bar * BAR_PA * col["A"])
    v_p = col["m_s"] / (col["rho_bulk"] * col["A"])
    u_rel = u_g + col["eps_bed"] * v_p
    dpdz = ergun_dpdz_Pa_m(u_rel, mu_g, rho_g, col["eps_bed"], col["d_pellet_m"])
    weight_grad = (1.0 - col["eps_bed"]) * (col["rho_pellet"] - rho_g) * G_MS2
    lift = dpdz / weight_grad

    # new pressure profile: integrate Ergun upward from the gas inlet
    p_new = [col["p_in_bar"]]
    for i in range(1, len(z_m)):
        dp_Pa = 0.5 * (dpdz[i] + dpdz[i - 1]) * (z_m[i] - z_m[i - 1])
        p_new.append(p_new[-1] - dp_Pa / BAR_PA)

    return {
        "z_m": z_m, "F_w": F_w, "q": q, "T_K": T_K, "Qw": Qw, "p_bar": p_bar,
        "pw_bar": pw_bar, "qstar": qstar, "k_1s": k_1s, "N": N_mol_sm,
        "y_w": y["H2O"], "u_g": u_g, "dpdz": dpdz, "lift": lift,
        "p_new": np.array(p_new), "v_p": v_p, "weight_grad": weight_grad,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def solve_column(geom, op, bed, loading_fn, kldf_fn, limits=None, isotherm_model="sips",
                 policy="clip", n_steps=200, include_dp=True, pinch_tol=0.15,
                 gas_data=None, tol_q=1e-13, tol_p_bar=2e-3, max_outer=20):
    """Solve the steady counter-current core.

    geom   : {"D_m", "H_m", "d_shaft_m", "pitch_m"}
    op     : {"gas_in_mols", "T_gas_in_K", "p_in_bar", "m_zeo_kgs", "T_zeo_in_K",
              "q_in_kgkg"}
    bed    : {"rho_pellet_kgm3", "eps_bed", "d_pellet_m", "cp_zeolite_JkgK",
              "dH_ads_Jmol", "U_wall_Wm2K", "T_wall_K"} and optional "k_ldf_factor"
    loading_fn(T_K, p_bar, model) -> q* [kg/kg]   (e.g. Zeolite3A().loading)
    kldf_fn(T_K, p_bar)           -> k [1/s]      (e.g. Zeolite3A().k_ldf)
    limits : data range of the zeolite model (default DEFAULT_LIMITS)
    policy : "clip" (extrapolate safely and report it) or "raise"

    Returns a dictionary of results, with "profile" (DataFrame) and "warnings" (list).
    """
    match policy:
        case "clip" | "raise":
            pass
        case _:
            raise ValueError("policy must be 'clip' or 'raise'")
    check_geometry(geom)
    if limits is None:
        limits = DEFAULT_LIMITS
    if gas_data is None:
        gas_data = gas_props.load_gas_data()
    gas = gas_props.check_species(op["gas_in_mols"], gas_data)
    F_w_in = gas["H2O"]
    if F_w_in <= 0.0:
        raise ValueError("The inlet gas contains no water")
    if op["m_zeo_kgs"] <= 0.0:
        raise ValueError("The zeolite rate must be positive")
    if n_steps < 10:
        raise ValueError("n_steps must be at least 10")

    F_dry = 0.0
    for species in gas:
        if species != "H2O":
            F_dry = F_dry + gas[species]
    W_dry, cp_v = gas_heat_capacity(gas, op["T_gas_in_K"], gas_data)
    m_s = op["m_zeo_kgs"]
    A = flow_area_m2(geom)
    rho_bulk = bed["rho_pellet_kgm3"] * (1.0 - bed["eps_bed"])

    # everything the marching functions need, in one dictionary
    col = {
        "n": int(n_steps), "h": geom["H_m"] / int(n_steps), "gas": gas,
        "gas_data": gas_data, "F_dry": F_dry, "F_w_in": F_w_in,
        "q_in": op["q_in_kgkg"], "m_s": m_s, "A": A, "rho_bulk": rho_bulk,
        "rhoA": rho_bulk * A, "rho_pellet": bed["rho_pellet_kgm3"],
        "eps_bed": bed["eps_bed"], "d_pellet_m": bed["d_pellet_m"],
        "mdH": -bed["dH_ads_Jmol"], "UP": bed["U_wall_Wm2K"] * math.pi * geom["D_m"],
        "T_wall_K": bed["T_wall_K"], "W_dry": W_dry, "W_s0": m_s * bed["cp_zeolite_JkgK"],
        "cp_v": cp_v, "T_gas_in_K": op["T_gas_in_K"], "T_zeo_in_K": op["T_zeo_in_K"],
        "p_in_bar": op["p_in_bar"], "loading_fn": loading_fn, "kldf_fn": kldf_fn,
        "isotherm_model": isotherm_model, "policy": policy, "tol_q": tol_q,
        "Qw_est": 0.0,
        # multiplies every k_LDF value (1.0 = model as given; 0.5 = recommended size)
        "k_ldf_factor": bed.get("k_ldf_factor", 1.0),
    }
    col["limits_q"] = limits["loading"]
    col["limits_k"] = limits["k_ldf"]
    col["E_iso_shift_Jmol"] = limits["E_iso_shift_Jmol"]
    col["Ea_kldf_Jmol"] = limits["Ea_kldf_Jmol"]

    # thermal regime over the whole possible range of q_out
    q_hi_max = op["q_in_kgkg"] + F_w_in * M_W_KGMOL / m_s
    col["q_hi_max"] = q_hi_max
    W_gas_in = W_dry + cp_v * F_w_in
    W_solid_in = m_s * (bed["cp_zeolite_JkgK"] + cp_v * op["q_in_kgkg"] / M_W_KGMOL)
    col["regime"] = classify_regime(net_heat_capacity_flow(0.0, col),
                                    net_heat_capacity_flow(q_hi_max, col),
                                    max(W_gas_in, W_solid_in), pinch_tol)
    m_bal = balanced_zeolite_rate_kgs(gas, op["T_gas_in_K"], bed["cp_zeolite_JkgK"],
                                      op["q_in_kgkg"], gas_data)
    if col["regime"] == "balanced":
        raise ThermalPinchError(
            "W_gas (" + str(round(W_gas_in)) + " W/K) ~ W_solid ("
            + str(round(W_solid_in)) + " W/K): the heat of adsorption is trapped in "
            "the counter-current core. Move the zeolite rate away from ~"
            + str(round(m_bal * 3600)) + " kg/h (higher is better).")

    # first pressure profile: Ergun gradient at the inlet, straight line in z
    n = col["n"]
    if include_dp:
        col["p_nodes"] = [op["p_in_bar"]]
        inlet = profile_arrays([[0, F_w_in, op["q_in_kgkg"], op["T_gas_in_K"], 0.0]], col)
        dpdz0 = float(inlet["dpdz"][0])
        col["p_nodes"] = []
        for i in range(n + 1):
            col["p_nodes"].append(op["p_in_bar"] - dpdz0 * i * col["h"] / BAR_PA)
    else:
        col["p_nodes"] = []
        for i in range(n + 1):
            col["p_nodes"].append(op["p_in_bar"])

    # outer while-loop: pressure profile and wall heat loss
    n_outer = 0
    q_prev = None
    converged = False
    totals = {"n_march": 0, "n_bisect": 0, "n_segments": 0}
    while n_outer < max_outer:
        n_outer = n_outer + 1
        rows = None
        if q_prev is not None:
            try:                                 # narrow bracket around the last answer
                rows, stats = shoot(col, max(0.0, q_prev - 2e-4),
                                    min(q_hi_max, q_prev + 2e-4))
            except BracketError:
                rows = None
        if rows is None:
            rows, stats = shoot(col, 0.0, q_hi_max)
        for key in totals:
            totals[key] = totals[key] + stats[key]
        q_prev = rows[0][2]
        prof = profile_arrays(rows, col)

        dp_change = float(abs(prof["p_new"] - prof["p_bar"]).max())
        Qw_top = rows[-1][4]
        Q_ads_est = col["mdH"] * (F_w_in - rows[-1][1])
        wall_ok = (col["regime"] == "gas_carries_heat"
                   or abs(Qw_top - col["Qw_est"]) <= 1e-7 * max(Q_ads_est, 1.0))
        dp_ok = (not include_dp) or dp_change < tol_p_bar
        if wall_ok and dp_ok:
            converged = True
            break
        if include_dp:
            col["p_nodes"] = []
            for value in prof["p_new"]:
                col["p_nodes"].append(float(value))
        col["Qw_est"] = Qw_top

    return collect_results(rows, prof, col, geom, bed, op, W_gas_in, W_solid_in, m_bal,
                           converged, n_outer, totals)


def collect_results(rows, prof, col, geom, bed, op, W_gas_in, W_solid_in, m_bal,
                    converged, n_outer, totals):
    """Scalar results, balance checks, warnings and the profile table."""
    m_s = col["m_s"]
    cp_s = bed["cp_zeolite_JkgK"]
    cp_v = col["cp_v"]
    q_in = op["q_in_kgkg"]
    F_w_in = col["F_w_in"]
    mdH = col["mdH"]
    T_ref_K = op["T_gas_in_K"]

    q_out = rows[0][2]
    F_w_out = max(rows[-1][1], 0.0)      # round-off can leave a tiny negative value
    T_bottom = rows[0][3]
    T_top = rows[-1][3]
    Q_wall_W = rows[-1][4]
    captured_mols = F_w_in - F_w_out
    captured_solid_mols = m_s * (q_out - q_in) / M_W_KGMOL
    Q_ads_W = mdH * captured_mols

    # temperatures of the streams leaving the column (thin thermal layer at one end)
    W_g_top = col["W_dry"] + cp_v * F_w_out
    W_s_top = m_s * (cp_s + cp_v * q_in / M_W_KGMOL)
    W_g_bottom = col["W_dry"] + cp_v * F_w_in
    W_s_bottom = m_s * (cp_s + cp_v * q_out / M_W_KGMOL)
    match col["regime"]:
        case "gas_carries_heat":
            T_zeo_out = T_bottom
            T_gas_out = T_top - W_s_top / W_g_top * (T_top - op["T_zeo_in_K"])
        case "solid_carries_heat":
            T_gas_out = T_top
            T_zeo_out = T_bottom - W_g_bottom / W_s_bottom * (T_bottom - op["T_gas_in_K"])

    # energy balance (reference: gas inlet temperature, water vapour)
    H_in = W_s_top * (op["T_zeo_in_K"] - T_ref_K) - m_s * q_in / M_W_KGMOL * mdH
    H_out = (W_g_top * (T_gas_out - T_ref_K) + W_s_bottom * (T_zeo_out - T_ref_K)
             - m_s * q_out / M_W_KGMOL * mdH)
    energy_closure = (H_in - H_out - Q_wall_W) / max(abs(Q_ads_W), 1.0)
    water_closure = (captured_mols - captured_solid_mols) / F_w_in

    # how much of the column is outside the data range
    T_K = prof["T_K"]
    pw = prof["pw_bar"]
    lim = col["limits_q"]
    T_lim = col["limits_k"]
    T_low_C = round(max(lim["T_min_K"], T_lim["T_min_K"]) - 273.15, 1)
    T_high_C = round(min(lim["T_max_K"], T_lim["T_max_K"]) - 273.15, 1)
    # 0.05 K margin: the edges found by bisection sit up to 0.01 K inside the range
    T_out = (T_K - 273.15 < T_low_C - 0.05) | (T_K - 273.15 > T_high_C + 0.05)
    frac_T_out = float(T_out.mean())                 # fraction of nodes (True = 1)
    frac_p_low = float((pw < lim["p_min_bar"]).mean())
    frac_p_high = float((pw > lim["p_max_bar"]).mean())
    holdup_kg = col["rhoA"] * geom["H_m"]
    tau_s = holdup_kg / m_s
    dp_bar = op["p_in_bar"] - float(prof["p_new"][-1])
    weight_Pa = trapezoid_integral(prof["weight_grad"], prof["z_m"])

    warnings = []
    if frac_T_out > 0:
        warnings.append("T outside the data range (" + str(T_low_C) + "-"
                        + str(T_high_C) + " C) over "
                        + str(round(100 * frac_T_out)) + "% of the height: "
                        "q* and k_LDF extrapolated")
    if frac_p_low > 0:
        warnings.append("pH2O < " + str(lim["p_min_bar"]) + " bar over "
                        + str(round(100 * frac_p_low)) + "% of the height: "
                        "Henry's-law extrapolation of q*")
    if frac_p_high > 0:
        warnings.append("pH2O > " + str(lim["p_max_bar"]) + " bar over "
                        + str(round(100 * frac_p_high)) + "% of the height: "
                        "q* held at its value at " + str(lim["p_max_bar"]) + " bar")
    if col["regime"] == "gas_carries_heat":
        warnings.append("W_gas > W_solid: the bed temperature is set by the gas inlet; "
                        "the zeolite feed temperature does not cool the bed")
    if not converged:
        warnings.append("Outer iteration (pressure / wall heat) did not converge")
    # information, not a warning (shown with st.info on the web page)
    notes = ["Zeolite residence time " + str(round(tau_s / 60, 2)) + " min; the LDF "
             "model accounts for the uptake kinetics"]

    profile = pd.DataFrame({
        "z_m": prof["z_m"],
        "T_C": T_K - 273.15,
        "p_bar": prof["p_bar"],
        "y_H2O": prof["y_w"],
        "pH2O_bar": pw,
        "q_kgkg": prof["q"],
        "qstar_kgkg": prof["qstar"],
        "k_ldf_1s": prof["k_1s"],
        "N_mol_s_m": prof["N"],
        "F_H2O_mols": prof["F_w"],
        "u_gas_ms": prof["u_g"],
        "dpdz_Pa_m": prof["dpdz"],
        "lift_ratio": prof["lift"],
    })

    return {
        "regime": col["regime"],
        "W_gas_in_WK": W_gas_in,
        "W_solid_in_WK": W_solid_in,
        "dW_WK": net_heat_capacity_flow(q_out, col),
        "W_ratio_solid_gas": W_solid_in / W_gas_in,
        "m_zeo_balanced_kgh": m_bal * 3600.0,
        "capture_frac": captured_mols / F_w_in,
        "water_in_kgh": F_w_in * M_W_KGMOL * 3600.0,
        "water_captured_kgh": captured_mols * M_W_KGMOL * 3600.0,
        "water_out_kgh": F_w_out * M_W_KGMOL * 3600.0,
        "y_H2O_in": float(prof["y_w"][0]),
        "y_H2O_out": float(prof["y_w"][-1]),
        "pH2O_out_bar": float(pw[-1]),
        "q_out_kgkg": q_out,
        "q_top_kgkg": rows[-1][2],
        # hottest zeolite: the marched profile, or the zeolite leaving at the bottom
        # after the thin zone where it meets the inlet gas (solid_carries_heat)
        "T_bed_max_C": max(float(T_K.max()), T_zeo_out) - 273.15,
        "T_profile_max_C": float(T_K.max()) - 273.15,
        "T_bed_min_C": float(T_K.min()) - 273.15,
        "T_bed_bottom_C": T_bottom - 273.15,
        "T_bed_top_C": T_top - 273.15,
        "T_gas_out_C": T_gas_out - 273.15,
        "T_zeo_out_C": T_zeo_out - 273.15,
        "Q_ads_kW": Q_ads_W / 1e3,
        "Q_wall_kW": Q_wall_W / 1e3,
        "dp_bar": dp_bar,
        # largest LOCAL value of (Ergun pressure gradient) / (bed weight per metre);
        # it is at the gas inlet (bottom), where the gas is hottest and wettest
        "lift_ratio_max": float(prof["lift"].max()),
        "z_lift_max_m": float(prof["z_m"][int(np.argmax(prof["lift"]))]),
        # height average: whole-core pressure drop / whole-bed weight per area
        "lift_ratio_avg": dp_bar * BAR_PA / weight_Pa,
        "u_gas_in_ms": float(prof["u_g"][0]),
        "u_gas_max_ms": float(prof["u_g"].max()),
        "v_zeo_ms": float(prof["v_p"]),
        "tau_zeo_s": tau_s,
        "holdup_kg": holdup_kg,
        "energy_closure_rel": energy_closure,
        "water_closure_rel": water_closure,
        "frac_T_extrapolated": frac_T_out,
        "T_data_range_C": str(T_low_C) + "-" + str(T_high_C),
        "frac_p_extrapolated": frac_p_low + frac_p_high,
        "frac_henry": frac_p_low,           # share of the height on Henry's law
        "p_henry_bar": lim["p_min_bar"],
        "converged": converged,
        "n_outer": n_outer,
        "n_march": totals["n_march"],
        "n_bisect": totals["n_bisect"],
        "n_segments": totals["n_segments"],
        "warnings": warnings,
        "notes": notes,
        "profile": profile,
    }
