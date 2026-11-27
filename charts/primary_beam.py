#!/usr/bin/env python3
"""Chromatic Super-Gaussian (p=3) & GAMBIT Zernike Primary Beam Models for CHARTS.

Literature & observatory basis:
- CHARTS System Temperature Estimate (330-360 MHz Memo): super-Gaussian (p=3) element
  pattern with E/H FWHM = 96x70 deg (300 MHz), 101x74 deg (345 MHz), 107x78 deg (400 MHz).
- GAMBIT Generative Antenna Model (Pashapour-Ahmadabadi et al., arXiv:2611.26713v1 Eq. 11-23):
  M_f(l, m) = sum_k c_k Z_k(s_f (l - dl_{0,f}), s_f (m - dm_{0,f})).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from charts_constants import C_LIGHT

C_LIGHT_MPS = float(C_LIGHT.value) if hasattr(C_LIGHT, "value") else float(C_LIGHT)

# Anchor FWHMs (deg) from CHARTS T_sys Memo (w_e: E-plane / East-West, w_h: H-plane / North-South)
_ANCHOR_FREQS_MHZ = np.array([300.0, 345.0, 400.0], dtype=np.float64)
_ANCHOR_FWHM_E_DEG = np.array([96.0, 101.0, 107.0], dtype=np.float64)
_ANCHOR_FWHM_H_DEG = np.array([70.0, 74.0, 78.0], dtype=np.float64)


def _piecewise_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    hi = x > xp[-1]
    if np.any(lo):
        s_lo = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y = np.where(lo, fp[0] + s_lo * (x - xp[0]), y)
    if np.any(hi):
        s_hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y = np.where(hi, fp[-1] + s_hi * (x - xp[-1]), y)
    return y


def charts_element_fwhm_deg(freq_mhz: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return chromatic E-plane and H-plane FWHMs (deg) across 300-501.3 MHz."""
    f = np.asarray(freq_mhz, dtype=np.float64)
    w_e = np.clip(_piecewise_linear_extrap(f, _ANCHOR_FREQS_MHZ, _ANCHOR_FWHM_E_DEG), 60.0, 150.0)
    w_h = np.clip(_piecewise_linear_extrap(f, _ANCHOR_FREQS_MHZ, _ANCHOR_FWHM_H_DEG), 50.0, 130.0)
    return w_e, w_h


def super_gaussian_primary_beam(
    l: Union[float, np.ndarray],
    m: Union[float, np.ndarray],
    freq_mhz: Union[float, np.ndarray],
    p: float = 3.0,
    fwhm_e_deg: Optional[Union[float, np.ndarray]] = None,
    fwhm_h_deg: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """Evaluate CHARTS chromatic Super-Gaussian element power pattern B_elem(l, m, f).

    Parameters accept scalar or broadcastable numpy arrays. Returns 0 below horizon (l^2+m^2 > 1).
    """
    l_arr, m_arr, f_arr = np.broadcast_arrays(
        np.asarray(l, dtype=np.float64),
        np.asarray(m, dtype=np.float64),
        np.asarray(freq_mhz, dtype=np.float64),
    )
    rho2 = l_arr * l_arr + m_arr * m_arr
    above = rho2 <= 1.0
    theta_rad = np.arcsin(np.sqrt(np.clip(rho2, 0.0, 1.0)))
    phi_rad = np.arctan2(m_arr, l_arr)

    default_e, default_h = charts_element_fwhm_deg(f_arr)
    w1_rad = np.deg2rad(default_e if fwhm_e_deg is None else np.asarray(fwhm_e_deg, dtype=np.float64))
    w2_rad = np.deg2rad(default_h if fwhm_h_deg is None else np.asarray(fwhm_h_deg, dtype=np.float64))

    cos_p = np.cos(phi_rad)
    sin_p = np.sin(phi_rad)
    r = 2.0 * theta_rad * np.sqrt((cos_p * cos_p) / np.maximum(w1_rad * w1_rad, 1e-12)
                                  + (sin_p * sin_p) / np.maximum(w2_rad * w2_rad, 1e-12))
    out = np.exp(-math.log(2.0) * (r ** float(p)))
    return np.where(above, out, 0.0)


def _zernike_nm_list(max_n: int) -> List[Tuple[int, int]]:
    """Noll-like ordering of (n, m) Zernike modes for 0 <= n <= max_n."""
    modes: List[Tuple[int, int]] = []
    for n in range(max_n + 1):
        for abs_m in range(n % 2, n + 1, 2):
            if abs_m == 0:
                modes.append((n, 0))
            else:
                modes.append((n, abs_m))
                modes.append((n, -abs_m))
    return modes


def _radial_zernike(n: int, abs_m: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial R_n^|m|(rho)."""
    res = np.zeros_like(rho, dtype=np.float64)
    half = (n - abs_m) // 2
    for s in range(half + 1):
        num = ((-1) ** s) * math.factorial(n - s)
        den = math.factorial(s) * math.factorial((n + abs_m) // 2 - s) * math.factorial(
            (n - abs_m) // 2 - s
        )
        res += (float(num) / float(den)) * (rho ** (n - 2 * s))
    return res


def eval_zernike_basis(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    max_n: int = 4,
    clamp_unit_disk: bool = True,
) -> np.ndarray:
    """Evaluate orthonormal Zernike modes up to radial order `max_n`.

    Returns shape `(..., N_modes)`. When `clamp_unit_disk=True`, coordinates outside
    `rho > 1` are smoothly clamped to unit radius for extrapolation safety under frequency scaling.
    """
    u_arr, v_arr = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    rho = np.sqrt(u_arr * u_arr + v_arr * v_arr)
    if clamp_unit_disk:
        rho = np.minimum(rho, 1.0)
    phi = np.arctan2(v_arr, u_arr)

    modes = _zernike_nm_list(max_n)
    basis = np.empty(u_arr.shape + (len(modes),), dtype=np.float64)
    for k, (n, m) in enumerate(modes):
        r_nm = _radial_zernike(n, abs(m), rho)
        if m == 0:
            norm = math.sqrt(n + 1)
            basis[..., k] = norm * r_nm
        elif m > 0:
            norm = math.sqrt(2.0 * (n + 1))
            basis[..., k] = norm * r_nm * np.cos(m * phi)
        else:
            norm = math.sqrt(2.0 * (n + 1))
            basis[..., k] = norm * r_nm * np.sin(abs(m) * phi)
    return basis


@dataclass
class ZernikePrimaryBeamModel:
    """GAMBIT-style generative chromatic Zernike primary beam model.

    Models `M_f(l, m) = sum_k c_k Z_k(s_f (l - dl_{0,f}), s_f (m - dm_{0,f}))` where
    `s_f`, `dl_{0,f}`, and `dm_{0,f}` are low-order polynomials in `f_norm = (f - f_ref) / 100 MHz`.
    """

    max_n: int = 4
    f_ref_mhz: float = 400.0
    coeffs: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))
    scale_poly: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], dtype=np.float64))
    dl_poly: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float64))
    dm_poly: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float64))

    def _transform_params(self, freq_mhz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = (np.asarray(freq_mhz, dtype=np.float64) - self.f_ref_mhz) / 100.0
        s_f = np.maximum(np.polyval(self.scale_poly, x), 0.25)
        dl_f = np.polyval(self.dl_poly, x)
        dm_f = np.polyval(self.dm_poly, x)
        return s_f, dl_f, dm_f

    def evaluate(
        self,
        l: Union[float, np.ndarray],
        m: Union[float, np.ndarray],
        freq_mhz: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Evaluate primary beam power gain M_f(l, m) clipped cleanly to [0, inf)."""
        l_arr, m_arr, f_arr = np.broadcast_arrays(
            np.asarray(l, dtype=np.float64),
            np.asarray(m, dtype=np.float64),
            np.asarray(freq_mhz, dtype=np.float64),
        )
        above = (l_arr * l_arr + m_arr * m_arr) <= 1.0
        s_f, dl_f, dm_f = self._transform_params(f_arr)
        u = s_f * (l_arr - dl_f)
        v = s_f * (m_arr - dm_f)
        basis = eval_zernike_basis(u, v, max_n=self.max_n)
        vals = np.tensordot(basis, self.coeffs, axes=([-1], [0]))
        return np.where(above, np.maximum(vals, 0.0), 0.0)

    @classmethod
    def fit(
        cls,
        l_pts: np.ndarray,
        m_pts: np.ndarray,
        freqs_mhz: np.ndarray,
        patterns: np.ndarray,
        max_n: int = 4,
        f_ref_mhz: float = 400.0,
        reg_l2: float = 1e-6,
    ) -> "ZernikePrimaryBeamModel":
        """Fit Zernike coefficients + linear-in-frequency scale/shift (s_f, dl_f, dm_f).

        Parameters:
            l_pts, m_pts: 1D arrays of direction cosines (length N_pts).
            freqs_mhz: 1D array of frequencies (length N_freq).
            patterns: 2D array of shape (N_freq, N_pts) target primary beam gains.
        """
        l_pts = np.asarray(l_pts, dtype=np.float64).ravel()
        m_pts = np.asarray(m_pts, dtype=np.float64).ravel()
        freqs = np.asarray(freqs_mhz, dtype=np.float64).ravel()
        y_obs = np.asarray(patterns, dtype=np.float64)
        if y_obs.shape != (len(freqs), len(l_pts)):
            raise ValueError(
                f"Expected patterns shape {(len(freqs), len(l_pts))}, got {y_obs.shape}"
            )

        n_modes = len(_zernike_nm_list(max_n))
        x_f = (freqs - f_ref_mhz) / 100.0
        eye_reg = reg_l2 * np.eye(n_modes, dtype=np.float64)

        def _solve_coeffs(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # theta = [s_slope, s_int, dl_slope, dl_int, dm_slope, dm_int]
            s_f = np.maximum(theta[0] * x_f + theta[1], 0.25)
            dl_f = theta[2] * x_f + theta[3]
            dm_f = theta[4] * x_f + theta[5]
            u = s_f[:, None] * (l_pts[None, :] - dl_f[:, None])
            v = s_f[:, None] * (m_pts[None, :] - dm_f[:, None])
            basis = eval_zernike_basis(u, v, max_n=max_n).reshape(-1, n_modes)
            rhs = y_obs.ravel()
            coeffs, *_ = np.linalg.lstsq(
                basis.T @ basis + eye_reg, basis.T @ rhs, rcond=None
            )
            resid = basis @ coeffs - rhs
            return coeffs, resid

        x0 = np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bounds = (
            np.array([-1.5, 0.35, -0.5, -0.35, -0.5, -0.35], dtype=np.float64),
            np.array([1.5, 1.50, 0.5, 0.35, 0.5, 0.35], dtype=np.float64),
        )
        opt = least_squares(lambda th: _solve_coeffs(th)[1], x0=x0, bounds=bounds, max_nfev=40)
        coeffs_opt, _ = _solve_coeffs(opt.x)
        th = opt.x
        return cls(
            max_n=max_n,
            f_ref_mhz=float(f_ref_mhz),
            coeffs=coeffs_opt.astype(np.float64),
            scale_poly=np.array([th[0], th[1]], dtype=np.float64),
            dl_poly=np.array([th[2], th[3]], dtype=np.float64),
            dm_poly=np.array([th[4], th[5]], dtype=np.float64),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_n": int(self.max_n),
            "f_ref_mhz": float(self.f_ref_mhz),
            "coeffs": [float(x) for x in self.coeffs],
            "scale_poly": [float(x) for x in self.scale_poly],
            "dl_poly": [float(x) for x in self.dl_poly],
            "dm_poly": [float(x) for x in self.dm_poly],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZernikePrimaryBeamModel":
        return cls(
            max_n=int(data["max_n"]),
            f_ref_mhz=float(data["f_ref_mhz"]),
            coeffs=np.asarray(data["coeffs"], dtype=np.float64),
            scale_poly=np.asarray(data["scale_poly"], dtype=np.float64),
            dl_poly=np.asarray(data["dl_poly"], dtype=np.float64),
            dm_poly=np.asarray(data["dm_poly"], dtype=np.float64),
        )


def effective_beam_response_sq(
    l_beam: float,
    m_beam: float,
    l_src: float,
    m_src: float,
    freqs_mhz: Sequence[float],
    pos_enu: np.ndarray,
    primary_beam: Union[str, ZernikePrimaryBeamModel, None] = "super_gaussian",
) -> np.ndarray:
    """Compute effective forward beam response |AF(beam; src)|^2 * B_elem(src, f)."""
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    n_b = float(math.sqrt(max(0.0, 1.0 - l_beam * l_beam - m_beam * m_beam)))
    n_s = float(math.sqrt(max(0.0, 1.0 - l_src * l_src - m_src * m_src)))
    ds = np.array([l_beam - l_src, m_beam - m_src, n_b - n_s], dtype=np.float64)
    delays = np.asarray(pos_enu, dtype=np.float64) @ ds
    k = 2.0 * math.pi * freqs * 1.0e6 / C_LIGHT_MPS
    af_sq = np.abs(np.mean(np.exp(1j * k[:, None] * delays[None, :]), axis=1)) ** 2

    if primary_beam is None or primary_beam == "none":
        b_elem = np.ones_like(freqs)
    elif isinstance(primary_beam, ZernikePrimaryBeamModel):
        b_elem = primary_beam.evaluate(l_src, m_src, freqs)
    elif primary_beam == "super_gaussian":
        b_elem = super_gaussian_primary_beam(l_src, m_src, freqs)
    else:
        raise ValueError(f"Unknown primary_beam model: {primary_beam!r}")
    return af_sq * b_elem
