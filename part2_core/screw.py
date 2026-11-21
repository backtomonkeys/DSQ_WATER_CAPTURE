"""
screw.py - Archimedes screw that moves the zeolite down through the core.

Volumetric throughput of a screw (fully packed channel):
    Q_bulk = eta * pi/4 * (D^2 - d^2) * pitch * N_rps
so the speed needed for a given zeolite (bulk) volume flow is
    rpm = 60 * Q_bulk / (pi/4 * (D^2 - d^2) * pitch * eta)

eta = volumetric efficiency, default 0.9 (screw feeders, Bortolamasi & Fottner 2001);
the range 0.5-1.0 has not been measured for this design.
"""

import math


def swept_volume_per_rev_m3(D_m, d_m, pitch_m):
    """Volume swept by one turn of the screw [m3/rev]."""
    if not (0.0 <= d_m < D_m) or pitch_m <= 0.0:
        raise ValueError("Need 0 <= d < D and pitch > 0")
    return math.pi / 4.0 * (D_m ** 2 - d_m ** 2) * pitch_m


def screw_rpm(Q_bulk_m3s, D_m, d_m, pitch_m, eta=0.9):
    """Screw speed [rpm] for a zeolite bulk volume flow Q_bulk_m3s [m3/s]."""
    if not (0.0 < eta <= 1.0):
        raise ValueError("Volumetric efficiency eta must be in (0, 1]")
    if Q_bulk_m3s < 0.0:
        raise ValueError("Volume flow must be >= 0")
    return 60.0 * Q_bulk_m3s / (swept_volume_per_rev_m3(D_m, d_m, pitch_m) * eta)


def zeolite_bulk_flow_m3s(m_zeo_kgs, rho_bulk_kgm3):
    """Bulk volume flow of zeolite [m3/s] from mass flow and bulk density."""
    return m_zeo_kgs / rho_bulk_kgm3


def screw_summary(m_zeo_kgs, rho_bulk_kgm3, D_m, d_m, pitch_m, eta=0.9):
    """Screw numbers for the report: bulk flow, rpm, axial pellet speed."""
    Q_bulk_m3s = zeolite_bulk_flow_m3s(m_zeo_kgs, rho_bulk_kgm3)
    rpm = screw_rpm(Q_bulk_m3s, D_m, d_m, pitch_m, eta)
    return {
        "Q_bulk_m3h": Q_bulk_m3s * 3600.0,
        "screw_rpm": rpm,
        "swept_m3_per_rev": swept_volume_per_rev_m3(D_m, d_m, pitch_m),
        "axial_speed_mm_s": eta * pitch_m * rpm / 60.0 * 1e3,
    }
