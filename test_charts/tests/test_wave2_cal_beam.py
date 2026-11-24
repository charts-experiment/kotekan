"""Unit tests for Wave 2B calibration and chromatic Zernike primary beam (CAL-2.1..2.3)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CHARTS_DIR = REPO_ROOT / "charts"
V2_DIR = CHARTS_DIR / "commissioning_20261119_v2"
for p in (str(CHARTS_DIR), str(V2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from charts_constants import C_LIGHT  # noqa: E402
from calibrate_corr_epoch import (  # noqa: E402
    POS4,
    solve_antenna_sefd_from_visibilities,
    solve_eig_spatial_filter_gains,
)
from primary_beam import (  # noqa: E402
    ZernikePrimaryBeamModel,
    charts_element_fwhm_deg,
    effective_beam_response_sq,
    eval_zernike_basis,
    super_gaussian_primary_beam,
)

C_LIGHT_MPS = float(C_LIGHT.value) if hasattr(C_LIGHT, "value") else float(C_LIGHT)


def test_super_gaussian_and_zernike_primary_beam():
    """CAL-2.3: Super-Gaussian FWHMs and GAMBIT Zernike compression/interpolation."""
    we_345, wh_345 = charts_element_fwhm_deg(345.0)
    assert abs(float(we_345) - 101.0) < 0.2
    assert abs(float(wh_345) - 74.0) < 0.2

    # At zenith B_elem == 1.0; at half-FWHM along E/H axes B_elem ~ 0.5
    b0 = float(super_gaussian_primary_beam(0.0, 0.0, 345.0))
    assert abs(b0 - 1.0) < 1e-12
    l_half = math.sin(math.radians(float(we_345) * 0.5))
    m_half = math.sin(math.radians(float(wh_345) * 0.5))
    assert abs(float(super_gaussian_primary_beam(l_half, 0.0, 345.0)) - 0.5) < 1e-3
    assert abs(float(super_gaussian_primary_beam(0.0, m_half, 345.0)) - 0.5) < 1e-3
    assert float(super_gaussian_primary_beam(0.9, 0.9, 345.0)) == 0.0

    # Fit ZernikePrimaryBeamModel to chromatic super-Gaussian snapshots and test out-of-sample
    g1d = np.linspace(-0.75, 0.75, 19)
    ll, mm = np.meshgrid(g1d, g1d)
    disk = (ll * ll + mm * mm) <= 0.78 ** 2
    lp, mp = ll[disk], mm[disk]
    freqs_fit = np.linspace(300.0, 500.0, 11)
    y_fit = super_gaussian_primary_beam(lp[None, :], mp[None, :], freqs_fit[:, None])

    basis = eval_zernike_basis(lp, mp, max_n=4)
    assert basis.shape == (len(lp), 15)

    model = ZernikePrimaryBeamModel.fit(lp, mp, freqs_fit, y_fit, max_n=4)
    y_pred = model.evaluate(lp[None, :], mp[None, :], freqs_fit[:, None])
    rmse = float(np.sqrt(np.mean((y_pred - y_fit) ** 2)))
    assert rmse < 0.025, f"Zernike fit RMSE too large: {rmse}"

    reloaded = ZernikePrimaryBeamModel.from_dict(model.to_dict())
    y_mid = reloaded.evaluate(0.25, -0.30, 412.5)
    y_ref = super_gaussian_primary_beam(0.25, -0.30, 412.5)
    assert abs(float(y_mid - y_ref)) < 0.02

    eff = effective_beam_response_sq(0.0, 0.0, l_half, 0.0, np.array([345.0]), pos_enu=POS4)
    assert float(eff[0]) <= 0.505


def test_solve_antenna_sefd_from_visibilities():
    """CAL-2.1: Exact log-least-squares SEFD recovery from visibility correlation magnitudes."""
    freqs = np.linspace(310.0, 450.0, 16)
    true_sefd = np.array([2.0e6, 8.0e6, 3.0e6, 5.0e6], dtype=np.float64)
    s_sun = np.full_like(freqs, 2.0e5)
    beam = super_gaussian_primary_beam(0.2, -0.1, freqs)
    df_hz = 300.0e3
    tau_s = 0.05
    bt = math.sqrt(df_hz * tau_s)

    n_ant = 4
    triu_i, triu_j = np.triu_indices(n_ant)
    vis = np.zeros((len(freqs), len(triu_i)), dtype=np.complex128)
    for fi in range(len(freqs)):
        sig = float(s_sun[fi] * beam[fi]) * bt
        for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
            if i == j:
                vis[fi, p_idx] = true_sefd[i] + sig
            else:
                sefd_ij = math.sqrt(true_sefd[i] * true_sefd[j])
                rho = sig / (sefd_ij + sig)
                auto_prod = math.sqrt((true_sefd[i] + sig) * (true_sefd[j] + sig))
                vis[fi, p_idx] = rho * auto_prod * np.exp(1j * 0.2 * (i - j))

    est = solve_antenna_sefd_from_visibilities(
        vis, freqs, sun_flux_jy=s_sun, beam_elem=beam, df_hz=df_hz, tau_s=tau_s
    )
    rel_err = np.max(np.abs(est - true_sefd[None, :]) / true_sefd[None, :])
    assert rel_err < 0.05, f"Relative SEFD recovery error {rel_err}"


def test_solve_eig_spatial_filter_gains():
    """CAL-2.2: Reject dominant off-target horizon RFI eigenmode and recover solar gains."""
    freqs = np.linspace(320.0, 460.0, 8)
    # Compact 4x4 CHARTS grid (d = 0.6 m): unambiguous main beam without sparse 2x2 grating aliasing
    cols, rows = np.meshgrid(np.arange(4), np.arange(4))
    pos_16 = np.column_stack([-cols.ravel() * 0.6, rows.ravel() * 0.6, np.zeros(16)])
    n_ant = len(pos_16)
    true_gains = np.exp(
        1j * np.outer(np.linspace(0.0, 0.25, len(freqs)), np.linspace(-0.2, 0.2, n_ant))
    )
    true_gains *= np.conj(true_gains[:, :1])

    # Solar source at target (l=0.1, m=0.15) with power 10, plus 15x stronger RFI at horizon (l=-0.85, m=-0.35)
    l_sun, m_sun = 0.10, 0.15
    n_sun = math.sqrt(1.0 - l_sun * l_sun - m_sun * m_sun)
    l_rfi, m_rfi = -0.85, -0.35
    n_rfi = math.sqrt(1.0 - l_rfi * l_rfi - m_rfi * m_rfi)

    s_sun = np.array([l_sun, m_sun, n_sun], dtype=np.float64)
    s_rfi = np.array([l_rfi, m_rfi, n_rfi], dtype=np.float64)
    triu_i, triu_j = np.triu_indices(n_ant)
    vis = np.zeros((len(freqs), len(triu_i)), dtype=np.complex128)
    for fi, f_mhz in enumerate(freqs):
        k = 2.0 * math.pi * f_mhz * 1.0e6 / C_LIGHT_MPS
        a_sun = true_gains[fi] * np.exp(-1j * k * (pos_16 @ s_sun))
        a_rfi = true_gains[fi] * np.exp(-1j * k * (pos_16 @ s_rfi))
        R = 10.0 * np.outer(a_sun, np.conj(a_sun)) + 150.0 * np.outer(a_rfi, np.conj(a_rfi))
        R += 0.25 * np.eye(n_ant)
        for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
            vis[fi, p_idx] = R[i, j]

    # Compare against true gains with target_lm centered at (l_sun, m_sun)
    est = solve_eig_spatial_filter_gains(
        vis, freqs, pos_enu=pos_16, target_lm=(l_sun, m_sun), cone_deg=20.0, n_iter=6
    )
    for fi, f_mhz in enumerate(freqs):
        k = 2.0 * math.pi * f_mhz * 1.0e6 / C_LIGHT_MPS
        expected = true_gains[fi] * np.exp(-1j * k * (pos_16 @ s_sun))
        expected *= np.exp(-1j * np.angle(expected[0]))
        phase_err_deg = np.rad2deg(np.abs(np.angle(est[fi] * np.conj(expected))))
        rms_deg = float(np.sqrt(np.mean(phase_err_deg ** 2)))
        assert rms_deg < 6.0 and float(np.max(phase_err_deg)) < 10.0, (
            f"Channel {fi} phase err too large: rms={rms_deg:.2f}, max={np.max(phase_err_deg):.2f}"
        )
