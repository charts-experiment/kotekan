#!/usr/bin/env python3
"""Build full 672-channel complex calibration gains from a post-epoch GPU
correlator run for CHARTS 4 corner antennas [0, 4, 32, 36].

Uses 2nd-order Butterworth Fringe-Rate High-Pass filtering (fc = 0.08 mHz)
to separate moving Sun fringes from stationary horizon clutter across
300.0 - 501.3 MHz, solves WALS rank-1 complex gains with Ant 0 as phase
reference, and fits lower/upper linear dispersion + 8-channel PFB sawtooth
to bridge RFI-contaminated channels seamlessly.

Supports --verify-only-gains to evaluate 24-hour phase/delay drift on a
later correlator run against previously saved gains without overwriting them.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, filtfilt

from charts_constants import CHARTS_ANTENNA_REF_LOCATION, C_LIGHT, FREQ
from charts_utils.cal.coordinates import CHARTSArray, geometric_visibility_phase
from charts_utils.core.antenna_indices import charts_to_kotekan_raw
from charts_utils.core.gpu_frames import read_gpu_corr_frame_header
from charts_utils.core.io import extract_gpu_corr_visibility, load_gpu_corr_frame

_CHARTS_DIR = str(Path(__file__).resolve().parents[1])
if _CHARTS_DIR not in sys.path:
    sys.path.insert(0, _CHARTS_DIR)

from primary_beam import super_gaussian_primary_beam  # noqa: E402

C_LIGHT_MPS = float(C_LIGHT.value) if hasattr(C_LIGHT, "value") else float(C_LIGHT)

ANTS = [0, 4, 32, 36]
ANT_LABELS = ["Ant 0 (D00, SE)", "Ant 4 (D07, SW)", "Ant 32 (A10, NE)", "Ant 36 (A17, NW)"]
POS4 = np.array(
    [
        [0.0, 0.0, 0.0],
        [-4.24775, 0.0275, 0.0],
        [0.15, 4.30625, 0.0],
        [-4.12725, 4.42725, 0.0],
    ],
    dtype=np.float64,
)
N_FREQ = 672
N_TOTAL_ANTS = 64
FREQS_MHZ = np.asarray(FREQ, dtype=np.float64)
SUBBAND_MAX_DT_US = 2000.0
DEFAULT_EPOCH_JSON = Path("/data/commissioning_20261119_v2/rfsoc_epoch.json")


def solve_wals_gains_batch(fs_triu: np.ndarray, niter: int = 6) -> np.ndarray:
    """Vectorized WALS rank-1 complex gain solver across (N_freq, 10, N_blocks)."""
    n_f, _, n_b = fs_triu.shape
    triu_i, triu_j = np.triu_indices(4)
    R = np.zeros((n_f, n_b, 4, 4), dtype=np.complex64)
    for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
        if i != j:
            v = fs_triu[:, p_idx, :]
            R[:, :, i, j] = v
            R[:, :, j, i] = np.conj(v)

    diag_idx = np.arange(4)
    for _ in range(niter):
        evals, evecs = np.linalg.eigh(R)
        u1 = evecs[:, :, :, -1]
        l1 = np.maximum(evals[:, :, -1], 0.0)
        R[:, :, diag_idx, diag_idx] = l1[:, :, None] * (np.abs(u1) ** 2)

    evals, evecs = np.linalg.eigh(R)
    u1 = evecs[:, :, :, -1]
    l1 = np.maximum(evals[:, :, -1], 0.0)
    ref_ph = np.exp(-1j * np.angle(u1[:, :, 0:1]))
    gains = u1 * ref_ph * np.sqrt(l1[:, :, None])
    return np.transpose(gains, (0, 2, 1)).astype(np.complex64)


def solve_eig_spatial_filter_gains(
    vis_triu: np.ndarray,
    freqs_mhz: np.ndarray,
    pos_enu: np.ndarray = POS4,
    target_lm: Tuple[float, float] = (0.0, 0.0),
    cone_deg: float = 25.0,
    n_iter: int = 6,
) -> np.ndarray:
    """Solve complex antenna gains via covariance eigen-decomposition spatial filtering (CAL-2.2).

    Separates the target (solar) rank-1 eigenmode from orthogonal horizon/RFI subspaces by
    projecting onto eigenmodes whose dirty-beam peak lies within `cone_deg` of `target_lm`,
    then iteratively refining diagonal noise power via WALS. Supports 2D `(N_freq, N_pairs)`
    or 3D `(N_freq, N_pairs, N_blocks)` input.
    """
    arr_in = np.asarray(vis_triu, dtype=np.complex128)
    squeezed = arr_in.ndim == 2
    if squeezed:
        arr_in = arr_in[:, :, None]
    n_f, n_pairs, n_b = arr_in.shape
    n_ant = int(round((math.sqrt(1 + 8 * n_pairs) - 1) / 2.0))
    triu_i, triu_j = np.triu_indices(n_ant)

    ll_1d = np.linspace(-0.98, 0.98, 49)
    ll, mm = np.meshgrid(ll_1d, ll_1d)
    disk = (ll * ll + mm * mm) <= 1.0
    l_v = ll[disk]
    m_v = mm[disk]
    n_v = np.sqrt(np.maximum(0.0, 1.0 - l_v * l_v - m_v * m_v))
    lt, mt = float(target_lm[0]), float(target_lm[1])
    nt = math.sqrt(max(0.0, 1.0 - lt * lt - mt * mt))
    sep_deg = np.rad2deg(np.arccos(np.clip(l_v * lt + m_v * mt + n_v * nt, -1.0, 1.0)))
    sky_dirs = np.stack([l_v, m_v, n_v], axis=-1) # [N_sky, 3]
    proj = sky_dirs @ np.asarray(pos_enu[:n_ant], dtype=np.float64).T # [N_sky, N_ant]

    gains_out = np.zeros((n_f, n_ant, n_b), dtype=np.complex64)
    diag_idx = np.arange(n_ant)

    for fi in range(n_f):
        k_wave = 2.0 * math.pi * float(freqs_mhz[fi]) * 1.0e6 / C_LIGHT_MPS
        # Match localize_rfi_emitters convention: V_ij = a_i conj(a_j) with a_i = exp(-j k r_i.s)
        steer = np.exp(-1j * k_wave * proj)
        for bi in range(n_b):
            R = np.zeros((n_ant, n_ant), dtype=np.complex128)
            for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
                val = arr_in[fi, p_idx, bi]
                R[i, j] = val
                R[j, i] = np.conj(val)
            # Zero-diag off-diagonal WALS warm-up if diagonal is missing/unpopulated
            diag_abs = np.mean(np.abs(np.diag(R)))
            off_abs = np.max(np.abs(R - np.diag(np.diag(R)))) if n_ant > 1 else 0.0
            if diag_abs <= 1e-12 and off_abs > 0.0:
                R[diag_idx, diag_idx] = off_abs * (n_ant - 1)
                for _ in range(3):
                    ev, evec = np.linalg.eigh(R)
                    u1 = evec[:, -1]
                    l1 = max(float(ev[-1]), 0.0)
                    R[diag_idx, diag_idx] = l1 * (np.abs(u1) ** 2)

            evals, evecs = np.linalg.eigh(R)
            sigma_floor = max(0.0, float(np.min(evals)))
            in_cone = sep_deg <= cone_deg
            has_cone = bool(np.any(in_cone))
            keep_modes = []
            rejected_modes = []
            best_mode = int(np.argmax(evals))
            best_ratio = -1.0
            for k_mode in range(n_ant - 1, -1, -1):
                lam = float(evals[k_mode])
                if lam <= 0.0:
                    continue
                u_k = evecs[:, k_mode]
                dirty = np.abs(steer @ np.conj(u_k)) ** 2
                pk_global = max(float(np.max(dirty)), 1e-30)
                pk_cone = float(np.max(dirty[in_cone])) if has_cone else 0.0
                ratio = pk_cone / pk_global
                if ratio > best_ratio and lam > 1.2 * sigma_floor:
                    best_ratio = ratio
                    best_mode = k_mode
                if ratio >= 0.70 and lam > 1.2 * sigma_floor:
                    keep_modes.append((lam, u_k))
                elif lam > 1.2 * sigma_floor:
                    rejected_modes.append((lam, u_k))

            if not keep_modes and evals[best_mode] > 0.0:
                keep_modes.append((float(evals[best_mode]), evecs[:, best_mode]))
                rejected_modes = [
                    (lam, u_k) for (lam, u_k) in rejected_modes
                    if not np.allclose(u_k, evecs[:, best_mode])
                ]

            if keep_modes and rejected_modes:
                R_filt = R.copy()
                for lam, u_k in rejected_modes:
                    R_filt -= (lam - sigma_floor) * np.outer(u_k, np.conj(u_k))
                ev_c, evec_c = np.linalg.eigh(R_filt)
                ev_c = np.maximum(ev_c, 0.0)
                R_filt = (evec_c * ev_c[None, :]) @ evec_c.conj().T
            else:
                R_filt = np.zeros((n_ant, n_ant), dtype=np.complex128)
                for lam, u_k in keep_modes:
                    R_filt += lam * np.outer(u_k, np.conj(u_k))
            for _ in range(max(1, n_iter)):
                ev, evec = np.linalg.eigh(R_filt)
                u_sol = evec[:, -1]
                l_sol = max(float(ev[-1]), 0.0)
                R_filt[diag_idx, diag_idx] = l_sol * (np.abs(u_sol) ** 2)

            ev, evec = np.linalg.eigh(R_filt)
            u_sol = evec[:, -1]
            l_sol = max(float(ev[-1]), 0.0)
            ref_ph = np.exp(-1j * np.angle(u_sol[0]))
            gains_out[fi, :, bi] = (u_sol * ref_ph * math.sqrt(l_sol)).astype(np.complex64)

    return gains_out[:, :, 0] if squeezed else gains_out


def solve_antenna_sefd_from_visibilities(
    vis_triu: np.ndarray,
    freqs_mhz: np.ndarray,
    sun_flux_jy: Union[float, np.ndarray],
    beam_elem: Optional[np.ndarray] = None,
    df_hz: float = 300.0e3,
    tau_s: float = 1.0,
) -> np.ndarray:
    """Solve per-antenna SEFD(nu, ant) in Jy from solar cross-correlations (BURSTT-256 Eq. 8).

    Expects `vis_triu` shape `(N_freq, N_pairs)` or `(N_freq, N_pairs, N_t)`. Returns `(N_freq, N_ant)`
    float64 array in Jy (NaN where fewer than N_ant - 1 valid cross baselines exist).
    """
    import math as _m

    arr_in = np.asarray(vis_triu, dtype=np.complex128)
    if arr_in.ndim == 3:
        vis_mean = np.mean(arr_in, axis=-1)
    else:
        vis_mean = arr_in
    n_f, n_pairs = vis_mean.shape
    n_ant = int(round((_m.sqrt(1 + 8 * n_pairs) - 1) / 2.0))
    triu_i, triu_j = np.triu_indices(n_ant)
    diag_pairs = [idx for idx, (i, j) in enumerate(zip(triu_i, triu_j)) if i == j]
    cross_pairs = [(idx, int(i), int(j)) for idx, (i, j) in enumerate(zip(triu_i, triu_j)) if i != j]

    s_sun = np.broadcast_to(np.asarray(sun_flux_jy, dtype=np.float64), (n_f,))
    b_el = (
        np.ones(n_f, dtype=np.float64)
        if beam_elem is None
        else np.broadcast_to(np.asarray(beam_elem, dtype=np.float64), (n_f,))
    )
    bt_sqrt = math.sqrt(max(df_hz * tau_s, 1.0))

    sefd_out = np.full((n_f, n_ant), np.nan, dtype=np.float64)
    for fi in range(n_f):
        autos = np.array([max(float(np.real(vis_mean[fi, dp])), 0.0) for dp in diag_pairs], dtype=np.float64)
        rows_A = []
        rows_b = []
        for p_idx, i, j in cross_pairs:
            denom = math.sqrt(max(autos[i] * autos[j], 0.0))
            if denom <= 0.0:
                continue
            rho = abs(vis_mean[fi, p_idx]) / denom
            if not (1.0e-4 < rho < 0.9999):
                continue
            sefd_ij = ((1.0 - rho) / rho) * float(s_sun[fi]) * float(b_el[fi]) * bt_sqrt
            if not (np.isfinite(sefd_ij) and sefd_ij > 0.0):
                continue
            row = np.zeros(n_ant, dtype=np.float64)
            row[i] = 0.5
            row[j] = 0.5
            rows_A.append(row)
            rows_b.append(math.log(sefd_ij))
        if len(rows_A) >= n_ant:
            A_mat = np.asarray(rows_A, dtype=np.float64)
            b_vec = np.asarray(rows_b, dtype=np.float64)
            sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            sefd_out[fi] = np.exp(sol)
    return sefd_out


def load_aligned_corr_pair(f0_path: str, f1_path: str, raws: list[int], triu_i, triu_j) -> np.ndarray:
    """Load one (672, 10) visibility frame for the 4 active antennas."""
    v0 = load_gpu_corr_frame(f0_path, n_freq=336, num_antennas=64)
    v1 = load_gpu_corr_frame(f1_path, n_freq=336, num_antennas=64)
    out = np.empty((N_FREQ, 10), dtype=np.complex64)
    for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
        out[:336, p_idx] = extract_gpu_corr_visibility(v0, raws[i], raws[j])
        out[336:, p_idx] = extract_gpu_corr_visibility(v1, raws[i], raws[j])
    return out


def compute_residual_stats(V_fs_mean: np.ndarray, gains_sub: np.ndarray, freqs_mask: np.ndarray) -> dict:
    """Compute per-baseline residual phase, circular coherence, and outlier-safe delay error (ns)."""
    triu_i, triu_j = np.triu_indices(4)
    stats = {}
    freqs_sel = FREQS_MHZ[freqs_mask]
    for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
        if i == j:
            continue
        res = V_fs_mean[freqs_mask, p_idx] / (gains_sub[freqs_mask, i] * np.conj(gains_sub[freqs_mask, j]))
        unit = res / np.maximum(np.abs(res), 1e-12)
        mean_phasor = np.mean(unit)
        mean_rad = float(np.angle(mean_phasor))
        mean_deg = float(np.rad2deg(mean_rad))
        circ_r = float(np.abs(mean_phasor))
        # Center residuals around the circular mean before fitting delay slope to avoid 2*pi unwrap slips.
        rel = np.angle(unit * np.exp(-1j * mean_rad))
        keep = np.abs(rel) < 2.0
        if int(np.sum(keep)) >= 3:
            slope = float(np.polyfit(freqs_sel[keep], rel[keep], 1)[0])
        else:
            slope = float(np.polyfit(freqs_sel, rel, 1)[0])
        delay_ns = slope / (2.0 * np.pi) * 1e3
        stats[f"{ANTS[i]}-{ANTS[j]}"] = {
            "mean_res_deg": round(mean_deg, 3),
            "circular_R": round(circ_r, 4),
            "delay_err_ns": round(delay_ns, 4),
        }
    return stats


def localize_rfi_emitters(V_raw_mean: np.ndarray, gains_sub: np.ndarray) -> dict:
    """Compute 2D dirty-beam sky/horizon maps for stationary RFI emitters."""
    triu_i, triu_j = np.triu_indices(4)
    ll, mm = np.meshgrid(np.linspace(-0.999, 0.999, 251), np.linspace(-0.999, 0.999, 251))
    mask_sky = (ll ** 2 + mm ** 2) <= 1.0
    target_freqs = [
        ("MUOS_360.9", 360.9),
        ("MUOS_376.2", 376.2),
        ("RFI_380.1", 380.1),
        ("RFI_415.8", 415.8),
        ("RFI_423.6", 423.6),
        ("RFI_425.4", 425.4),
        ("RFI_428.4", 428.4),
        ("RFI_432.6", 432.6),
        ("RFI_472.2", 472.2),
        ("RFI_472.5", 472.5),
    ]
    results = {}
    for label, f_mhz in target_freqs:
        fi = int(np.argmin(np.abs(FREQS_MHZ - f_mhz)))
        f_actual = float(FREQS_MHZ[fi])
        R = np.zeros((4, 4), dtype=np.complex128)
        for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
            if i == j:
                continue
            val = V_raw_mean[fi, p_idx] / (gains_sub[fi, i] * np.conj(gains_sub[fi, j]))
            R[i, j] = val
            R[j, i] = np.conj(val)
        k_wave = 2.0 * np.pi * f_actual * 1e6 / 299792458.0
        score = np.zeros_like(ll)
        for i in range(4):
            for j in range(i + 1, 4):
                dx, dy = POS4[i, 0] - POS4[j, 0], POS4[i, 1] - POS4[j, 1]
                ph = -k_wave * (dx * ll + dy * mm)
                score += 2.0 * np.real(R[i, j] * np.exp(-1j * ph))
        score[~mask_sky] = -1e30
        bi, bj = np.unravel_index(np.argmax(score), score.shape)
        lp, mp = float(ll[bi, bj]), float(mm[bi, bj])
        el = float(np.rad2deg(np.arcsin(np.sqrt(max(0.0, 1.0 - lp * lp - mp * mp)))))
        az = float(np.rad2deg(np.arctan2(lp, mp)) % 360.0)
        score_hi = score.copy()
        score_hi[(ll ** 2 + mm ** 2) > np.cos(np.deg2rad(25.0)) ** 2] = -1e30
        hi_i, hi_j = np.unravel_index(np.argmax(score_hi), score_hi.shape)
        lp_hi, mp_hi = float(ll[hi_i, hi_j]), float(mm[hi_i, hi_j])
        el_hi = float(np.rad2deg(np.arcsin(np.sqrt(max(0.0, 1.0 - lp_hi ** 2 - mp_hi ** 2)))))
        az_hi = float(np.rad2deg(np.arctan2(lp_hi, mp_hi)) % 360.0)
        results[label] = {
            "freq_mhz": round(f_actual, 3),
            "channel": fi,
            "peak_all_sky": {"l": round(lp, 4), "m": round(mp, 4), "az_deg": round(az, 2), "el_deg": round(el, 2)},
            "peak_el_ge_25": {"l": round(lp_hi, 4), "m": round(mp_hi, 4), "az_deg": round(az_hi, 2), "el_deg": round(el_hi, 2)},
        }
    return results


def default_epoch_min_unix() -> float:
    if DEFAULT_EPOCH_JSON.exists():
        try:
            d = json.loads(DEFAULT_EPOCH_JSON.read_text(encoding="utf-8"))
            ts = float(d.get("timestamp_sec", 0.0))
            if ts > 0.0:
                return ts
        except Exception:
            pass
    return 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corr-dir", default="/data/corr/corr_1119_cal")
    p.add_argument("--out-dir", default="/data/calibration/corr_1119")
    p.add_argument("--rfi-out", default="/data/commissioning_20261119_v2/localized_rfi_targets.json")
    p.add_argument("--epoch-min-unix", type=float, default=None)
    p.add_argument("--min-sun-el", type=float, default=20.0)
    p.add_argument("--verify-only-gains", default=None, help="Existing gains .npy to audit for phase/delay drift only")
    p.add_argument("--gains-filename", default="gains_corr_1119_0_4_32_36_fringehp_672x64.npy")
    p.add_argument(
        "--method",
        default="fringe_hp",
        choices=["fringe_hp", "eig_spatial_filter"],
        help="Calibration mode: temporal fringe high-pass + WALS, or covariance eigen-decomposition spatial filter",
    )
    p.add_argument("--save-sefd", default=None, help="Optional output path for solved per-antenna inverse-SEFD weights .npy (672x64 float32)")
    args = p.parse_args()

    corr_dir = Path(args.corr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rfi_out = Path(args.rfi_out)
    rfi_out.parent.mkdir(parents=True, exist_ok=True)

    epoch_min = args.epoch_min_unix if args.epoch_min_unix is not None else default_epoch_min_unix()

    f0s = sorted(glob.glob(str(corr_dir / "*freq_0_*.bin")))
    f1s = sorted(glob.glob(str(corr_dir / "*freq_1_*.bin")))
    if not f0s or not f1s:
        raise FileNotFoundError(f"Missing corr_freq_0/1 binaries in {corr_dir}")

    # Match frame pairs by exact trailing numeric index
    def _idx_map(paths):
        mp = {}
        for fp in paths:
            stem = Path(fp).stem
            try:
                idx = int(stem.rsplit("_", 1)[-1])
                mp[idx] = fp
            except ValueError:
                pass
        return mp

    m0 = _idx_map(f0s)
    m1 = _idx_map(f1s)
    common_idxs = sorted(set(m0.keys()) & set(m1.keys()))
    if not common_idxs:
        raise RuntimeError(f"No matching freq_0/freq_1 frame indices in {corr_dir}")

    raws = [charts_to_kotekan_raw(a, 64) for a in ANTS]
    triu_i, triu_j = np.triu_indices(4)
    loc = EarthLocation(
        lat=CHARTS_ANTENNA_REF_LOCATION[0] * u.deg,
        lon=CHARTS_ANTENNA_REF_LOCATION[1] * u.deg,
        height=CHARTS_ANTENNA_REF_LOCATION[2] * u.m,
    )

    print(f"[1/6] Scanning {len(common_idxs)} frame pairs in {corr_dir} (epoch_min={epoch_min}) ...")
    valid_meta = []
    for idx in common_idxs:
        p0, p1 = m0[idx], m1[idx]
        h0 = read_gpu_corr_frame_header(p0)
        h1 = read_gpu_corr_frame_header(p1)
        t0 = h0.time0_fpga * 1e-6
        if abs(h1.time0_fpga - h0.time0_fpga) > SUBBAND_MAX_DT_US or t0 < epoch_min:
            continue
        valid_meta.append((idx, t0, p0, p1))
    if len(valid_meta) < 4:
        raise RuntimeError(f"Too few aligned post-epoch frames in {corr_dir}: {len(valid_meta)}")

    t_all = np.array([m[1] for m in valid_meta], dtype=np.float64)
    sun_el = get_sun(Time(t_all, format="unix")).transform_to(
        AltAz(obstime=Time(t_all, format="unix"), location=loc)
    ).alt.deg
    keep_idx = [valid_meta[k] for k in range(len(valid_meta)) if sun_el[k] >= args.min_sun_el]
    if len(keep_idx) < 4:
        print(f"  Warning: only {len(keep_idx)} frames have Sun El >= {args.min_sun_el}; using all {len(valid_meta)} frames.")
        keep_idx = valid_meta

    print(
        f"  Aligned frames used: {len(keep_idx)} "
        f"({datetime.fromtimestamp(keep_idx[0][1], tz=timezone.utc).strftime('%H:%M:%S')}Z -> "
        f"{datetime.fromtimestamp(keep_idx[-1][1], tz=timezone.utc).strftime('%H:%M:%S')}Z, "
        f"Sun El {sun_el.min():.1f}..{sun_el.max():.1f} deg)"
    )

    clean_low = (FREQS_MHZ >= 315.0) & (FREQS_MHZ <= 355.0)
    clean_high = (FREQS_MHZ >= 440.0) & (FREQS_MHZ <= 462.0)
    arr = CHARTSArray(
        CHARTS_ANTENNA_REF_LOCATION[0],
        CHARTS_ANTENNA_REF_LOCATION[1],
        CHARTS_ANTENNA_REF_LOCATION[2],
        POS4,
    )

    if args.verify_only_gains:
        gains_64 = np.load(args.verify_only_gains)
        gains_sub = gains_64[:, ANTS]
        t_list = []
        v_list = []
        for _, t0, p0, p1 in keep_idx:
            v_pair = load_aligned_corr_pair(p0, p1, raws, triu_i, triu_j)
            if np.median(np.real(v_pair[:, 0])) < 1e5:
                continue
            t_list.append(t0)
            v_list.append(v_pair)
        t_arr = np.asarray(t_list, dtype=np.float64)
        V_raw = np.transpose(np.asarray(v_list, dtype=np.complex128), (1, 2, 0))
        geom = -geometric_visibility_phase(t_arr, FREQS_MHZ, arr, triu_i, triu_j, src_name="sun")
        V_fs = V_raw * np.exp(-1j * geom)
        V_hp = V_fs.copy()
        span_s = float(t_arr[-1] - t_arr[0]) if len(t_arr) >= 2 else 0.0
        hp_method = "unfiltered"
        if span_s >= 2400.0 and len(t_arr) >= 24:
            dt_step = float(np.median(np.diff(t_arr)))
            fs_hz = 1.0 / max(dt_step, 1.0)
            b_hp, a_hp = butter(2, min(0.08e-3 / (0.5 * fs_hz), 0.4), btype="high")
            hp_method = "butter_2nd_0.08mHz_highpass"
            for p_idx in range(10):
                if triu_i[p_idx] != triu_j[p_idx]:
                    V_hp[:, p_idx, :] = (
                        filtfilt(b_hp, a_hp, np.real(V_raw[:, p_idx, :]), axis=-1)
                        + 1j * filtfilt(b_hp, a_hp, np.imag(V_raw[:, p_idx, :]), axis=-1)
                    ) * np.exp(-1j * geom[:, p_idx, :])
        elif len(t_arr) >= 4:
            hp_method = "temporal_mean_subtract"
            for p_idx in range(10):
                if triu_i[p_idx] != triu_j[p_idx]:
                    v_ac = V_raw[:, p_idx, :] - np.mean(V_raw[:, p_idx, :], axis=-1, keepdims=True)
                    V_hp[:, p_idx, :] = v_ac * np.exp(-1j * geom[:, p_idx, :])
        V_mean_unfilt = np.mean(V_fs, axis=-1)
        V_mean_hp = np.mean(V_hp, axis=-1)
        drift_rep = {
            "corr_dir": str(corr_dir),
            "gains_file": str(args.verify_only_gains),
            "frames_used": len(t_arr),
            "start_utc": datetime.fromtimestamp(t_arr[0], tz=timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(t_arr[-1], tz=timezone.utc).isoformat(),
            "sun_el_min_deg": round(float(sun_el.min()), 2),
            "sun_el_max_deg": round(float(sun_el.max()), 2),
            "hp_method": hp_method,
            "residuals_unfilt_315_355": compute_residual_stats(V_mean_unfilt, gains_sub, clean_low),
            "residuals_unfilt_440_462": compute_residual_stats(V_mean_unfilt, gains_sub, clean_high),
            "residuals_hp_315_355": compute_residual_stats(V_mean_hp, gains_sub, clean_low),
            "residuals_hp_440_462": compute_residual_stats(V_mean_hp, gains_sub, clean_high),
        }
        out_json = out_dir / "drift_verification.json"
        tmp_json = out_json.with_name(f".tmp.{os.getpid()}.{out_json.name}")
        tmp_json.write_text(json.dumps(drift_rep, indent=2), encoding="utf-8")
        os.chmod(tmp_json, 0o644)
        os.replace(tmp_json, out_json)
        print(json.dumps(drift_rep, indent=2))
        return

    # Solve calibration gains: uniform 61.44 s binning
    t_start = keep_idx[0][1]
    t_end = keep_idx[-1][1]
    dt_bin = 61.44 if (t_end - t_start) >= 1200.0 else max(20.48, (t_end - t_start) / 24.0)
    n_bins = int(np.floor((t_end - t_start) / dt_bin)) + 1
    vis_binned = np.zeros((N_FREQ, 10, n_bins), dtype=np.complex128)
    counts_bin = np.zeros(n_bins, dtype=np.int32)
    t_bin = t_start + (np.arange(n_bins) + 0.5) * dt_bin

    for _, t0, p0, p1 in keep_idx:
        b = int(np.floor((t0 - t_start) / dt_bin))
        if b < 0 or b >= n_bins:
            continue
        v_pair = load_aligned_corr_pair(p0, p1, raws, triu_i, triu_j)
        if np.median(np.real(v_pair[:, 0])) < 1e5:
            continue
        vis_binned[:, :, b] += v_pair
        counts_bin[b] += 1

    good_bins = counts_bin > 0
    for b in np.where(good_bins)[0]:
        vis_binned[:, :, b] /= counts_bin[b]

    empty_bins = np.where(~good_bins)[0]
    print(f"  Uniform {dt_bin:.2f}s bins: {n_bins} total, {int(good_bins.sum())} populated, {len(empty_bins)} interpolated gap bins")
    if len(empty_bins) > 0:
        x_good = np.where(good_bins)[0]
        x_all = np.arange(n_bins)
        for p_idx in range(10):
            for f in range(N_FREQ):
                re_i = np.interp(x_all, x_good, np.real(vis_binned[f, p_idx, x_good]))
                im_i = np.interp(x_all, x_good, np.imag(vis_binned[f, p_idx, x_good]))
                vis_binned[f, p_idx, empty_bins] = re_i[empty_bins] + 1j * im_i[empty_bins]

    print("[2/6] Computing geometric solar phase & applying Fringe-Rate High-Pass filter ...")
    geom_triu = -geometric_visibility_phase(t_bin, FREQS_MHZ, arr, triu_i, triu_j, src_name="sun")
    demod_phasor = np.exp(-1j * geom_triu).astype(np.complex64)

    fs_hz = 1.0 / dt_bin
    nyq_hz = 0.5 * fs_hz
    fc_hz = 0.08e-3
    padlen = min(9, max(0, n_bins // 4))
    b_hp, a_hp = butter(2, min(fc_hz / nyq_hz, 0.4), btype="high")

    vis_hp = vis_binned.astype(np.complex64).copy()
    if n_bins >= 12:
        for p_idx in range(10):
            if triu_i[p_idx] != triu_j[p_idx]:
                re_f = filtfilt(b_hp, a_hp, np.real(vis_binned[:, p_idx, :]), axis=-1, padlen=padlen)
                im_f = filtfilt(b_hp, a_hp, np.imag(vis_binned[:, p_idx, :]), axis=-1, padlen=padlen)
                vis_hp[:, p_idx, :] = (re_f + 1j * im_f).astype(np.complex64)
    else:
        vis_hp -= np.mean(vis_hp, axis=-1, keepdims=True)

    valid_solve_mask = good_bins.copy()
    if n_bins >= 16:
        valid_solve_mask[:2] = False
        valid_solve_mask[-2:] = False
    solve_bins = np.where(valid_solve_mask)[0]
    blk_sz = 4 if len(solve_bins) >= 24 else max(1, len(solve_bins) // 6)
    n_blocks = max(1, len(solve_bins) // blk_sz)
    solve_bins = solve_bins[: n_blocks * blk_sz]

    fs_unfilt = (vis_binned[:, :, solve_bins] * demod_phasor[:, :, solve_bins]).reshape(N_FREQ, 10, n_blocks, blk_sz).mean(axis=-1)
    fs_hp = (vis_hp[:, :, solve_bins] * demod_phasor[:, :, solve_bins]).reshape(N_FREQ, 10, n_blocks, blk_sz).mean(axis=-1)

    print(f"[3/6] Solving complex gains across {n_blocks} blocks (blk_sz={blk_sz}, method={args.method}) ...")
    gains_cube_unfilt = solve_wals_gains_batch(fs_unfilt, niter=6)
    if args.method == "eig_spatial_filter":
        gains_cube_hp = solve_eig_spatial_filter_gains(
            fs_unfilt, FREQS_MHZ, pos_enu=POS4, target_lm=(0.0, 0.0), cone_deg=25.0, n_iter=6
        )
    else:
        gains_cube_hp = solve_wals_gains_batch(fs_hp, niter=6)

    unit_unfilt = gains_cube_unfilt / np.maximum(np.abs(gains_cube_unfilt), 1e-12)
    coh_unfilt_min = np.min(np.abs(np.mean(unit_unfilt[:, 1:, :], axis=-1)), axis=1)
    unit_hp = gains_cube_hp / np.maximum(np.abs(gains_cube_hp), 1e-12)
    coh_hp_min = np.min(np.abs(np.mean(unit_hp[:, 1:, :], axis=-1)), axis=1)

    phasor_unfilt = np.mean(unit_unfilt, axis=-1)
    phasor_unfilt /= np.maximum(np.abs(phasor_unfilt), 1e-12)
    phasor_hp = np.mean(unit_hp, axis=-1)
    phasor_hp /= np.maximum(np.abs(phasor_hp), 1e-12)

    print("[4/6] Constructing seamless 672-channel gain tensor ...")
    diag_indices = [0, 4, 7, 9]
    auto_sqrt = np.sqrt(np.maximum(np.median(np.real(vis_binned[:, diag_indices, :][:, :, solve_bins]), axis=-1), 1.0))
    clean_amp_mask = (coh_unfilt_min >= 0.85) | (coh_hp_min >= 0.75)
    if np.sum(clean_amp_mask) < 50:
        clean_amp_mask = (coh_unfilt_min >= 0.65) | (coh_hp_min >= 0.60)
    if np.sum(clean_amp_mask) < 30:
        clean_amp_mask = np.ones(N_FREQ, dtype=bool)

    ch_grid = np.arange(N_FREQ, dtype=np.float64)
    amp_smooth = np.zeros((N_FREQ, 4), dtype=np.float32)
    for k in range(4):
        xp = ch_grid[clean_amp_mask]
        fp = auto_sqrt[clean_amp_mask, k]
        interp_a = np.interp(ch_grid, xp, fp)
        interp_a = median_filter(interp_a, size=15, mode="nearest")
        interp_a = uniform_filter1d(interp_a, size=21, mode="nearest")
        amp_smooth[:, k] = interp_a.astype(np.float32)
    amp_scale = 0.62 / float(np.median(amp_smooth))
    amp_smooth *= amp_scale

    lower_anchor = (ch_grid < 367) & (coh_unfilt_min >= 0.90)
    if np.sum(lower_anchor) < 30:
        lower_anchor = (ch_grid < 367) & (coh_unfilt_min >= 0.80)
    if np.sum(lower_anchor) < 20:
        lower_anchor = (ch_grid < 367) & (coh_hp_min >= 0.60)

    upper_anchor = (ch_grid >= 367) & (coh_hp_min >= 0.72)
    if np.sum(upper_anchor) < 30:
        upper_anchor = (ch_grid >= 367) & (coh_hp_min >= 0.55)
    print(f"  Anchor channels: lower={int(lower_anchor.sum())}/367, upper={int(upper_anchor.sum())}/305")

    lower_model = np.ones((N_FREQ, 4), dtype=np.complex64)
    upper_model = np.ones((N_FREQ, 4), dtype=np.complex64)
    for k in range(1, 4):
        ch_low = ch_grid[lower_anchor]
        ph_low = np.unwrap(np.angle(phasor_unfilt[lower_anchor, k]))
        w_low = np.maximum(coh_unfilt_min[lower_anchor], 0.1) ** 2
        slope_l, int_l = np.polyfit(ch_low, ph_low, 1, w=w_low)
        res_l = np.angle(phasor_unfilt[lower_anchor, k] * np.exp(-1j * (slope_l * ch_low + int_l)))
        saw_l = np.zeros(8, dtype=np.float64)
        for m8 in range(8):
            sel = (ch_low.astype(int) % 8) == m8
            if np.any(sel):
                saw_l[m8] = np.angle(np.mean(np.exp(1j * res_l[sel])))
        saw_l -= np.mean(saw_l)
        lower_model[:, k] = np.exp(1j * (slope_l * ch_grid + int_l + saw_l[ch_grid.astype(int) % 8])).astype(np.complex64)

        if np.sum(upper_anchor) >= 12:
            ch_up = ch_grid[upper_anchor]
            diff_up = phasor_hp[upper_anchor, k] * np.conj(lower_model[upper_anchor, k])
            res_up = np.unwrap(np.angle(diff_up))
            w_up = np.maximum(coh_hp_min[upper_anchor], 0.1) ** 2
            slope_d, int_d = np.polyfit(ch_up, res_up, 1, w=w_up)
            rem_up = np.angle(diff_up * np.exp(-1j * (slope_d * ch_up + int_d)))
            saw_u = np.zeros(8, dtype=np.float64)
            for m8 in range(8):
                sel = (ch_up.astype(int) % 8) == m8
                if np.any(sel):
                    saw_u[m8] = np.angle(np.mean(np.exp(1j * rem_up[sel])))
            saw_u -= np.mean(saw_u)
            upper_model[:, k] = np.exp(
                1j * (slope_l * ch_grid + int_l + slope_d * ch_grid + int_d + saw_u[ch_grid.astype(int) % 8])
            ).astype(np.complex64)
        else:
            upper_model[:, k] = lower_model[:, k]

    final_phasor = np.ones((N_FREQ, 4), dtype=np.complex64)
    channel_source = np.empty(N_FREQ, dtype="<U24")
    for ch in range(N_FREQ):
        if ch < 367:
            if coh_unfilt_min[ch] >= 0.85:
                final_phasor[ch, 1:] = phasor_unfilt[ch, 1:]
                channel_source[ch] = "sun_unfilt"
            elif coh_hp_min[ch] >= 0.72:
                final_phasor[ch, 1:] = phasor_hp[ch, 1:]
                channel_source[ch] = "fringe_hp_low"
            else:
                final_phasor[ch, 1:] = lower_model[ch, 1:]
                channel_source[ch] = "lower_model_interp"
        else:
            if coh_hp_min[ch] >= 0.72:
                final_phasor[ch, 1:] = phasor_hp[ch, 1:]
                channel_source[ch] = "fringe_hp_sun"
            else:
                final_phasor[ch, 1:] = upper_model[ch, 1:]
                channel_source[ch] = "upper_model_interp"

    gains_4 = (amp_smooth * final_phasor).astype(np.complex64)
    gains_64 = np.zeros((N_FREQ, N_TOTAL_ANTS), dtype=np.complex64)
    gains_64[:, ANTS] = gains_4

    assert np.all(np.isfinite(gains_64)), "Non-finite gains produced!"
    assert np.all(np.abs(gains_64[:, ANTS]) > 0.1), "Zero/tiny gains produced!"

    out_npy = out_dir / args.gains_filename
    np.save(out_npy, gains_64)
    np.save(out_dir / "gains_channel_source.npy", channel_source)

    sefd_out_path = None
    if args.save_sefd:
        # Approximate quiet-Sun continuum flux density spectrum S_sun(nu) ~ 20 kJy * (nu/400)^0.75 * 10^4 Jy
        sun_aa = get_sun(Time(t_bin[solve_bins], format="unix")).transform_to(
            AltAz(obstime=Time(t_bin[solve_bins], format="unix"), location=loc)
        )
        alt_r = np.deg2rad(np.clip(sun_aa.alt.deg, 1.0, 90.0))
        az_r = np.deg2rad(sun_aa.az.deg)
        l_sun = float(np.median(np.cos(alt_r) * np.sin(az_r)))
        m_sun = float(np.median(np.cos(alt_r) * np.cos(az_r)))
        beam_sun = super_gaussian_primary_beam(l_sun, m_sun, FREQS_MHZ)
        s_sun_jy = 2.0e5 * (FREQS_MHZ / 400.0) ** 0.75
        v_for_sefd = vis_binned[:, :, solve_bins] * demod_phasor[:, :, solve_bins]
        sefd_4 = solve_antenna_sefd_from_visibilities(
            v_for_sefd, FREQS_MHZ, sun_flux_jy=s_sun_jy, beam_elem=beam_sun, df_hz=300.0e3, tau_s=dt_bin
        )
        sefd_64 = np.zeros((N_FREQ, N_TOTAL_ANTS), dtype=np.float32)
        for k_idx, ant_id in enumerate(ANTS):
            valid_s = np.isfinite(sefd_4[:, k_idx]) & (sefd_4[:, k_idx] > 0)
            if np.any(valid_s):
                sefd_64[:, ant_id] = np.interp(ch_grid, ch_grid[valid_s], sefd_4[valid_s, k_idx]).astype(np.float32)
        sefd_out_path = Path(args.save_sefd)
        sefd_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(sefd_out_path, sefd_64)

    print("[5/6] Validating residuals & localizing stationary RFI emitters ...")
    V_mean_hp = np.mean(fs_hp, axis=-1)
    V_mean_unfilt = np.mean(fs_unfilt, axis=-1)
    res_low = compute_residual_stats(V_mean_unfilt, gains_4, clean_low)
    res_high = compute_residual_stats(V_mean_hp, gains_4, clean_high)

    V_raw_mean = np.mean(vis_binned[:, :, solve_bins], axis=-1)
    rfi_locs = localize_rfi_emitters(V_raw_mean, gains_4)
    rfi_out.write_text(json.dumps(rfi_locs, indent=2), encoding="utf-8")

    u_src, c_src = np.unique(channel_source, return_counts=True)
    report = {
        "gains_file": str(out_npy),
        "corr_dir": str(corr_dir),
        "epoch_min_unix": epoch_min,
        "frames_used": len(keep_idx),
        "solve_blocks": int(n_blocks),
        "channel_provenance": {str(k): int(v) for k, v in zip(u_src, c_src)},
        "residuals_315_355": res_low,
        "residuals_440_462": res_high,
        "localized_rfi_targets": rfi_locs,
    }
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[6/6] Saving diagnostic plot ...")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    ax0, ax1, ax2 = axes
    ax0.plot(FREQS_MHZ, coh_unfilt_min, color="#64748b", alpha=0.65, lw=1.0, label="Unfiltered EVD")
    ax0.plot(FREQS_MHZ, coh_hp_min, color="#0284c7", lw=1.2, label="Fringe-HP (fc=0.08 mHz)")
    ax0.axhline(0.72, color="#dc2626", ls="--", lw=1.0, label="Threshold (0.72)")
    ax0.axvspan(356.0, 382.0, color="#fef08a", alpha=0.35, label="MUOS 356-382 MHz")
    ax0.set_ylabel("Min Ant Coherence")
    ax0.set_ylim(0, 1.05)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="lower left", ncol=2, fontsize=8.5)
    ax0.set_title(f"CHARTS Epoch Solar Calibration ({corr_dir.name}) for Antennas [0, 4, 32, 36]")

    colors = ["#0f172a", "#2563eb", "#16a34a", "#dc2626"]
    for k in range(1, 4):
        ax1.plot(FREQS_MHZ, np.rad2deg(np.angle(gains_4[:, k])), color=colors[k], lw=1.0, label=ANT_LABELS[k])
    ax1.set_ylabel("Gain Phase (deg)")
    ax1.set_ylim(-190, 190)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", ncol=3, fontsize=8.5)

    for k in range(4):
        ax2.plot(FREQS_MHZ, np.abs(gains_4[:, k]), color=colors[k], lw=1.3, label=ANT_LABELS[k])
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Gain |g|")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", ncol=4, fontsize=8.5)
    fig.savefig(out_dir / "calibration_diagnostics.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
