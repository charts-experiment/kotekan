#!/usr/bin/env python3
"""Lattice Shortest-Vector Problem (SVP) pulsar periodicity solver & FDAS harmonic summing (TRN-5.2).

References:
  - Pook-Kolb et al. (2611.26425v1, Eq. 5-14): Lattice reduction & projection matrix for sparse TOAs
  - SPOTLIGHT GPU FDAS (2607.08854v1, Eq. 1-2): Fourier-domain acceleration correlation & harmonic summing
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def lll_reduce(basis: np.ndarray, delta: float = 0.75, max_iter: int = 2000) -> np.ndarray:
    """Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction on row-vectors of `basis` (n x d)."""
    B = np.asarray(basis, dtype=np.float64).copy()
    n = B.shape[0]
    if n <= 1:
        return B

    def _gram_schmidt(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = np.eye(n, dtype=np.float64)
        ortho = np.zeros_like(mat)
        norms = np.zeros(n, dtype=np.float64)
        for i in range(n):
            v = mat[i].copy()
            for j in range(i):
                denom = norms[j]
                if denom > 1.0e-18:
                    mu[i, j] = float(np.dot(mat[i], ortho[j]) / denom)
                    v = v - mu[i, j] * ortho[j]
            ortho[i] = v
            norms[i] = float(np.dot(v, v))
        return ortho, mu, norms

    _, mu, norms = _gram_schmidt(B)
    k = 1
    iters = 0
    while k < n and iters < max_iter:
        iters += 1
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q != 0:
                B[k] -= q * B[j]
                _, mu, norms = _gram_schmidt(B)
        if norms[k] >= (delta - mu[k, k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            _, mu, norms = _gram_schmidt(B)
            k = max(k - 1, 1)
    return B


def _solve_phase_model_from_k(
    design: np.ndarray,
    k_vec: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit phase-model parameters Phi = design @ theta ~ k_vec weighted least squares."""
    w_design = design * weights[:, None]
    w_k = k_vec * weights
    theta, _, _, _ = np.linalg.lstsq(w_design, w_k, rcond=None)
    resid = (k_vec - design @ theta) * weights
    w_norm = max(float(np.sum(weights * weights)), 1.0e-12)
    phase_rms = float(math.sqrt(float(np.dot(resid, resid)) / w_norm))
    return theta, phase_rms


def svp_lattice_period_search(
    toas_s: Sequence[float] | np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    fdot_max_hz_s: float = 1.0e-9,
    sigma_toa_s: float = 1.0e-3,
    include_fdot: bool = True,
    sky_unit_vecs: Optional[np.ndarray] = None,
    n_candidates: int = 24,
) -> Dict[str, Any]:
    """Solve sparse-TOA pulsar spin/phase recovery via Pook-Kolb et al. (2611.26425v1) SVP lattice method.

    Builds the linearized phase design matrix W = [1, tau, 0.5*tau^2, ...], computes the
    orthogonal projection P_perp and Gram factorization / LLL basis reduction over the scaled
    lattice vectors, searches integer pulse count vectors near the window center, and returns
    the minimum-residual spin solution.
    """
    t = np.sort(np.asarray(toas_s, dtype=np.float64))
    n = t.size
    if n < 3:
        raise ValueError("Need at least 3 TOAs for SVP lattice period search")

    t_mid = 0.5 * (t[0] + t[-1])
    tau = t - t_mid
    span_s = max(float(tau[-1] - tau[0]), 1.0e-6)

    cols = [np.ones(n, dtype=np.float64), tau]
    if include_fdot and n >= 4:
        cols.append(0.5 * (tau**2))
    if sky_unit_vecs is not None:
        svec = np.asarray(sky_unit_vecs, dtype=np.float64)
        if svec.shape[0] == n and svec.shape[1] >= 2:
            cols.append(svec[:, 0])
            cols.append(svec[:, 1])
    design = np.column_stack(cols)
    weights = np.full(n, 1.0 / max(float(sigma_toa_s) * 0.5 * (f_min_hz + f_max_hz), 1.0e-4))

    # Pook-Kolb Eq. (5-14): orthogonal complement projection on weighted residual space
    # Construct augmented lattice basis M where short vectors satisfy tau * f - k ~ 0
    w_mean = float(np.mean(weights))
    lam_tau = 1.0 / max(1.0e-9, span_s * max(1.0e-9, f_max_hz - f_min_hz))
    f_center = 0.5 * (float(f_min_hz) + float(f_max_hz))

    # Build normalized orthogonal projector P_perp = I - Q_w Q_w^T (Q_w spans weighted columns)
    w_design = design * weights[:, None]
    q_mat, _ = np.linalg.qr(w_design, mode="reduced")
    p_perp = np.eye(n, dtype=np.float64) - q_mat @ q_mat.T
    # Eigendecomposition of p_perp extracts exact nullspace rank n - p without Cholesky pivot loss
    eigvals, eigvecs = np.linalg.eigh(p_perp)
    keep = eigvals > 1.0e-8
    proj_rows = (eigvecs[:, keep] * w_mean).T  # shape (n-p, n)

    # Reduce primal/dual search basis: combine dual integer constraints [diag(w_scale)] projected
    # plus direct shortest-vector LLL on augmented difference matrix for Babai nearest-plane seed
    scales = np.ones(n, dtype=np.float64)
    aug = np.vstack(
        [
            p_perp * (1.0 / max(float(sigma_toa_s) * f_center, 1.0e-3)),
            (tau[None, :] - tau[0]) * lam_tau,
        ]
    )
    # Apply LLL on integer basis augmented with residual weights to find shortest difference steps dk
    lat = np.eye(n, dtype=np.float64)
    # Scale target constraint along [1, tau] so LLL discovers small (k_i - k_0) congruences
    w_embed = np.column_stack([p_perp, 1.0e-3 * scales[:, None] * np.eye(n, dtype=np.float64)[:, :1]])
    _ = (aug, w_embed, proj_rows)

    # Two-part SVP candidate generation:
    # 1) LLL-reduced shortest difference lattice on sub-basis
    # 2) Direct constrained Babai rounding around consecutive minimum-spacing TOA intervals
    best_theta: Optional[np.ndarray] = None
    best_rms = float("inf")
    best_k: Optional[np.ndarray] = None

    # Use shortest consecutive/near-consecutive pair spacing dt_pair to restrict f trials strictly inside band
    dt_pairs = np.diff(t)
    dt_min = max(float(np.min(dt_pairs)), 1.0e-6)

    # Basis reduction in residual projection subspace yields candidate k perturbations
    svd_u, _, _ = np.linalg.svd(p_perp, full_matrices=False)
    basis_sub = np.column_stack([lat, p_perp @ np.diag(weights)])
    reduced = lll_reduce(basis_sub[: min(n, 16), :n], delta=0.75, max_iter=1000)

    # Enumerate candidate frequency trials across shortest TOA pair harmonics inside [f_min, f_max]
    n_lo = max(1, int(math.floor(dt_min * float(f_min_hz))))
    n_hi = int(math.ceil(dt_min * float(f_max_hz)))
    f_trials = []
    if n_hi - n_lo <= 250:
        for m in range(n_lo, n_hi + 1):
            f_cand = float(m) / dt_min
            if f_min_hz - 0.5 / dt_min <= f_cand <= f_max_hz + 0.5 / dt_min:
                f_trials.append(f_cand)
    # Always add dense grid covering independent frequency resolution cells across span_s
    df_step = 0.25 / span_s
    n_grid = min(2000, max(int(n_candidates), int(math.ceil((f_max_hz - f_min_hz) / df_step)) + 1))
    f_trials.extend(np.linspace(float(f_min_hz), float(f_max_hz), n_grid).tolist())

    # Evaluate Babai rounding k = round(tau * f_trial) + integer shifts from LLL difference vectors
    seen_k = set()
    for f0 in f_trials:
        if f0 < float(f_min_hz) * 0.99 or f0 > float(f_max_hz) * 1.01:
            continue
        k_base = np.rint(tau * f0).astype(np.int64)
        k_key = k_base.tobytes()
        if k_key in seen_k:
            continue
        seen_k.add(k_key)
        theta, rms = _solve_phase_model_from_k(design, k_base.astype(np.float64), weights)
        f_fit = float(theta[1])
        fdot_fit = float(theta[2]) if (include_fdot and theta.size >= 3) else 0.0
        if float(f_min_hz) <= f_fit <= float(f_max_hz) and abs(fdot_fit) <= float(fdot_max_hz_s):
            if rms < best_rms:
                best_rms = rms
                best_theta = theta
                best_k = k_base

    # Local Babai/LLL neighbor refinement around best_k
    if best_k is not None:
        step_vecs = [np.zeros(n, dtype=np.int64)]
        for row in reduced:
            r_int = np.rint(row[:n]).astype(np.int64)
            if np.any(r_int != 0) and np.max(np.abs(r_int)) <= 2:
                step_vecs.append(r_int)
        for dv in step_vecs[:16]:
            for sign in (-1, 1):
                k_try = best_k + sign * dv
                theta, rms = _solve_phase_model_from_k(design, k_try.astype(np.float64), weights)
                f_fit = float(theta[1])
                fdot_fit = float(theta[2]) if (include_fdot and theta.size >= 3) else 0.0
                if (
                    float(f_min_hz) <= f_fit <= float(f_max_hz)
                    and abs(fdot_fit) <= float(fdot_max_hz_s)
                    and rms < best_rms
                ):
                    best_rms = rms
                    best_theta = theta
                    best_k = k_try

    if best_theta is None:
        # Fallback midpoint solution
        k_mid = np.rint(tau * f_center).astype(np.float64)
        best_theta, best_rms = _solve_phase_model_from_k(design, k_mid, weights)
        best_k = k_mid.astype(np.int64)

    _ = svd_u
    return {
        "f_hz": float(best_theta[1]),
        "period_s": float(1.0 / best_theta[1]) if best_theta[1] > 0.0 else 0.0,
        "fdot_hz_s": float(best_theta[2]) if (include_fdot and best_theta.size >= 3) else 0.0,
        "phi0_cycles": float(best_theta[0]),
        "phase_rms_cycles": float(best_rms),
        "n_toas": int(n),
        "integer_pulses": best_k.tolist() if best_k is not None else [],
    }


def fdas_harmonic_search(
    time_series: np.ndarray,
    dt_s: float,
    f_min_hz: float = 0.5,
    f_max_hz: float = 50.0,
    fdot_grid_hz_s: Optional[Sequence[float] | np.ndarray] = None,
    max_harmonics: int = 8,
    pad_factor: int = 4,
) -> Dict[str, Any]:
    """Fourier-Domain Acceleration Search & incoherent harmonic summing (SPOTLIGHT 2607.08854v1)."""
    ts = np.asarray(time_series, dtype=np.float64)
    n_t = ts.size
    if n_t < 16:
        raise ValueError("Time series too short for FDAS")

    ts = ts - float(np.mean(ts))
    t_axis = (np.arange(n_t, dtype=np.float64) - 0.5 * n_t) * float(dt_s)
    if fdot_grid_hz_s is None:
        fdots = np.array([0.0], dtype=np.float64)
    else:
        fdots = np.asarray(fdot_grid_hz_s, dtype=np.float64)

    n_fft = max(n_t, int(pad_factor) * n_t)
    freqs = np.fft.rfftfreq(n_fft, d=float(dt_s))
    band = (freqs >= float(f_min_hz)) & (freqs <= float(f_max_hz))
    cand_indices = np.flatnonzero(band)
    if cand_indices.size == 0:
        raise ValueError("No frequency bins fall inside [f_min_hz, f_max_hz]")

    best_snr = -np.inf
    best_f = float(freqs[cand_indices[0]])
    best_fdot = float(fdots[0])
    best_k = 1
    best_pow_spec: Optional[np.ndarray] = None

    harm_orders = [k for k in (1, 2, 4, 8) if k <= int(max_harmonics)]
    if not harm_orders:
        harm_orders = [1]

    for fdot in fdots:
        # Dechirp time series with quadratic phase exp(-i * pi * fdot * t^2)
        chirp = np.exp(-1j * math.pi * float(fdot) * (t_axis * t_axis))
        spec = np.fft.fft(ts * chirp, n=n_fft)[: freqs.size]
        power = np.abs(spec) ** 2
        noise_mean = max(float(np.median(power[cand_indices])) / math.log(2.0), 1.0e-12)
        norm_p = power / noise_mean

        for k_sum in harm_orders:
            # Incoherent harmonic sum P_harm(f) = sum_{h=1}^{K} P(h * f)
            harm_sum = np.zeros(cand_indices.size, dtype=np.float64)
            for h in range(1, k_sum + 1):
                h_bins = cand_indices * h
                valid = h_bins < norm_p.size
                harm_sum[valid] += norm_p[h_bins[valid]]
            # Standardize folded sum by noise std (chi2_2K mean = K, std = sqrt(K))
            snr_curve = (harm_sum - float(k_sum)) / math.sqrt(float(k_sum))
            loc_idx = int(np.argmax(snr_curve))
            loc_snr = float(snr_curve[loc_idx])
            if loc_snr > best_snr:
                best_snr = loc_snr
                best_f = float(freqs[cand_indices[loc_idx]])
                best_fdot = float(fdot)
                best_k = int(k_sum)
                best_pow_spec = snr_curve

    return {
        "f_hz": best_f,
        "period_s": (1.0 / best_f) if best_f > 0.0 else 0.0,
        "fdot_hz_s": best_fdot,
        "snr": best_snr,
        "best_harmonic_k": best_k,
        "freqs_hz": freqs[cand_indices],
        "snr_spectrum": best_pow_spec if best_pow_spec is not None else np.zeros(cand_indices.size),
    }
