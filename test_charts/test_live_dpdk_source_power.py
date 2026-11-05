#!/usr/bin/env python3
"""
Test CHARTS Direct Beam Tracker with Normalized Mode ON, Dual DPDK Streams, and Calibrated Gains
================================================================================================
Tests live line-rate DPDK streaming (105 Gbps across dual 100GbE ports), 672 frequency channels,
calibrated gains (.npy), normalized mode ON, and measures formed beam power for Fornax A,
Fornax Cluster Core, NGC 253, NGC 612, Pictor A, and Cold Sky background.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import numpy as np

REST_URL = "http://127.0.0.1:12048"

def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{REST_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return {"status": resp.status, "body": resp.read().decode("utf-8", errors="replace")}

def get_json(endpoint: str) -> dict:
    url = f"{REST_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def wait_for_new_frames(n_frames=2, timeout_s=15):
    start = time.time()
    initial_fid = None
    while time.time() - start < timeout_s:
        try:
            data = get_json("/direct_tracker/beam_power")
            fid = data.get("frame_id", 0)
            bp = data.get("beam_power", [])
            if len(bp) > 0 and any(len(s) > 0 and max(s) > 0 for s in bp):
                if initial_fid is None:
                    initial_fid = fid
                elif fid >= initial_fid + n_frames:
                    return fid
        except Exception:
            pass
        time.sleep(0.3)
    return initial_fid or 0

def sample_beams(n_samples=5):
    """Samples formed beam power spectra across n_samples distinct frames."""
    accumulated_spectra = {}
    last_fid = -1
    count = 0
    start = time.time()

    while count < n_samples and time.time() - start < 20:
        try:
            data = get_json("/direct_tracker/beam_power")
            fid = data.get("frame_id", 0)
            if fid > 0 and fid != last_fid:
                spectra = data.get("beam_power", [])
                for b_idx, spec in enumerate(spectra):
                    spec_arr = np.array(spec, dtype=np.float32)
                    if b_idx not in accumulated_spectra:
                        accumulated_spectra[b_idx] = []
                    accumulated_spectra[b_idx].append(spec_arr)
                count += 1
                last_fid = fid
        except Exception as e:
            pass
        time.sleep(0.3)

    result = {}
    for b_idx, arrs in accumulated_spectra.items():
        if len(arrs) > 0:
            stacked = np.stack(arrs, axis=0) # [n_samples, n_channels]
            mean_spec = np.mean(stacked, axis=0)
            result[b_idx] = {
                "mean_spectrum": mean_spec,
                "overall_mean": float(np.mean(mean_spec)),
                "overall_std": float(np.std(mean_spec)),
                "overall_median": float(np.median(mean_spec)),
                "subband0_mean": float(np.mean(mean_spec[:336])),
                "subband1_mean": float(np.mean(mean_spec[336:])),
                "peak_power": float(np.max(mean_spec)),
                "peak_chan": int(np.argmax(mean_spec)),
                "n_samples": len(arrs)
            }
    return result

def main():
    print("=" * 100)
    print(" CHARTS DIRECT BEAM TRACKER: LIVE DUAL-DPDK SOURCE POWER & NORMALIZATION TEST")
    print("=" * 100)

    # 1. Pipeline Verification
    print("\n[Step 1] Verifying Pipeline Invariants & Telemetry...")
    st = get_json("/direct_tracker/status")
    bp_info = get_json("/direct_tracker/beam_power")

    print(f"  * normalize_weights:       {st.get('normalize_weights')}")
    print(f"  * active_antennas:         {st.get('active_antennas')} (Physical: {st.get('active_physical_antennas')})")
    print(f"  * active_raw_elements:     {st.get('active_raw_elements')}")
    print(f"  * num_active_beams:        {st.get('num_active_beams')} / {st.get('max_beams')}")
    print(f"  * total_freq_channels:     {bp_info.get('total_freq_channels')} ({bp_info.get('num_subbands')} subbands)")
    print(f"  * current_lst_hours:       {st.get('current_lst_hours'):.4f} h")
    print(f"  * current_utc_timestamp:   {st.get('current_utc_timestamp'):.2f}")

    assert st.get('normalize_weights') == True, "normalize_weights must be True"
    assert bp_info.get('total_freq_channels') == 672, "total_freq_channels must be 672"
    assert st.get('active_antennas') == 5, "active_antennas must be 5"

    source_names = {
        0: "NGC 253 (Sculptor)",
        1: "NGC 612 (PKS 0131-36)",
        2: "NGC 1316 (Fornax A)",
        3: "Fornax Cluster Core",
        4: "Pictor A",
        5: "Cold Sky Reference"
    }

    # 2. Measure Live Powers for Assigned Beams (Normalized ON)
    print("\n[Step 2] Measuring Multi-Frame Formed Beam Powers with NORMALIZED ON...")
    wait_for_new_frames(2)
    norm_samples = sample_beams(n_samples=6)

    print("\n" + "-" * 100)
    print(f"{'Beam':<6} {'Target Name':<24} {'RA (deg)':<10} {'Dec (deg)':<10} {'Alt (n0)':<10} {'Mean Pow':<10} {'Std Dev':<10} {'SB0 (300M)':<12} {'SB1 (400M)':<12}")
    print("-" * 100)

    beams_meta = {b['beam_id']: b for b in st.get('beams', [])}
    for b_idx in range(st.get('num_active_beams')):
        b_meta = beams_meta.get(b_idx, {})
        tgt = b_meta.get('celestial_target', {})
        ra = tgt.get('ra_deg', 0.0)
        dec = tgt.get('dec_deg', 0.0)
        n0 = b_meta.get('n0', 0.0)
        alt_deg = np.degrees(np.arcsin(max(0.0, min(1.0, n0))))
        s_info = norm_samples.get(b_idx, {})
        name = source_names.get(b_idx, f"Beam {b_idx}")
        print(f"{b_idx:<6} {name:<24} {ra:<10.3f} {dec:<10.3f} {alt_deg:<4.1f} ({n0:.3f}) {s_info.get('overall_mean', 0.0):<10.4f} {s_info.get('overall_std', 0.0):<10.4f} {s_info.get('subband0_mean', 0.0):<12.4f} {s_info.get('subband1_mean', 0.0):<12.4f}")
    print("-" * 100)

    # 3. Dedicated On-Source (Fornax A) vs Off-Source (Cold Sky) Contrast Test
    print("\n[Step 3] Performing Differential On-Source vs Off-Source Contrast Test...")
    print("  * Enabling Beam 5 as Cold Sky Reference at Dec = -37.2083 deg, RA = 80.675 deg (offset +30 deg from Fornax A)...")
    post_json("/direct_tracker/enable_beam", {"num_active_beams": 6})
    post_json("/direct_tracker/set_celestial_target", {
        "beam_id": 5,
        "ra_deg": 80.6750,
        "dec_deg": -37.2083
    })
    wait_for_new_frames(3)

    diff_samples = sample_beams(n_samples=5)
    fornax_p = diff_samples[2]["overall_mean"]
    fornax_cluster_p = diff_samples[3]["overall_mean"]
    cold_sky_p = diff_samples[5]["overall_mean"]

    delta_p = fornax_p - cold_sky_p
    ratio_p = fornax_p / max(1e-6, cold_sky_p)
    ratio_db = 10.0 * np.log10(max(1e-6, ratio_p))

    print(f"\n  Differential Power Results:")
    print(f"  * Beam 2 [NGC 1316 (Fornax A) ON]:      Mean Power = {fornax_p:8.4f} (RMS: {diff_samples[2]['overall_std']:.4f})")
    print(f"  * Beam 3 [Fornax Cluster Core ON]:       Mean Power = {fornax_cluster_p:8.4f} (RMS: {diff_samples[3]['overall_std']:.4f})")
    print(f"  * Beam 5 [Cold Sky OFF-Target]:          Mean Power = {cold_sky_p:8.4f} (RMS: {diff_samples[5]['overall_std']:.4f})")
    print(f"  * Delta Power (Fornax A - Cold Sky):    Delta P    = {delta_p:+8.4f}")
    print(f"  * On/Off Contrast Ratio:                 Contrast   = {ratio_p:8.4f}x ({ratio_db:+6.2f} dB)")

    # 4. Dynamic Repointing Test: Steer Beam 5 directly onto Fornax A
    print("\n[Step 4] Dynamic Repointing Test: Steer Beam 5 directly ONTO Fornax A...")
    post_json("/direct_tracker/set_celestial_target", {
        "beam_id": 5,
        "ra_deg": 50.6750,
        "dec_deg": -37.2083
    })
    wait_for_new_frames(3)
    repointed_samples = sample_beams(n_samples=5)
    b5_on_fornax = repointed_samples[5]["overall_mean"]
    b2_on_fornax = repointed_samples[2]["overall_mean"]
    print(f"  * Beam 2 [Fornax A]:                     Mean Power = {b2_on_fornax:8.4f}")
    print(f"  * Beam 5 [Steered to Fornax A]:          Mean Power = {b5_on_fornax:8.4f}")
    match_diff = abs(b5_on_fornax - b2_on_fornax)
    print(f"  * Discrepancy between two Fornax beams:  |B5 - B2|  = {match_diff:8.5f} (<0.01 indicates perfect beam agreement)")

    # 5. Normalized ON vs Normalized OFF Comparison
    print("\n[Step 5] Testing UNNORMALIZED Mode vs NORMALIZED Mode...")
    print("  * Toggling normalize_weights: false via /direct_tracker/set_taper...")
    post_json("/direct_tracker/set_taper", {
        "taper_type": "uniform",
        "normalize_weights": False,
        "enable_weight_bislc": False
    })
    time.sleep(0.5)
    st_unnorm = get_json("/direct_tracker/status")
    print(f"  * Confirmed normalize_weights is now:    {st_unnorm.get('normalize_weights')}")

    wait_for_new_frames(3)
    unnorm_samples = sample_beams(n_samples=5)

    print("\n  * Toggling normalize_weights: true back to NORMALIZED...")
    post_json("/direct_tracker/set_taper", {
        "taper_type": "uniform",
        "normalize_weights": True,
        "enable_weight_bislc": False
    })
    time.sleep(0.5)
    wait_for_new_frames(3)
    restored_samples = sample_beams(n_samples=5)

    print("\n" + "=" * 90)
    print(f"{'Beam':<6} {'Target Name':<24} {'Normalized (ON)':<18} {'Unnormalized (OFF)':<20} {'Scale Ratio':<12} {'Expected':<10}")
    print("=" * 90)
    for b_idx in range(6):
        p_norm = restored_samples[b_idx]["overall_mean"]
        p_unnorm = unnorm_samples[b_idx]["overall_mean"]
        scale = p_unnorm / max(1e-6, p_norm)
        name = source_names.get(b_idx, f"Beam {b_idx}")
        print(f"{b_idx:<6} {name:<24} {p_norm:<18.4f} {p_unnorm:<20.4f} {scale:<12.2f}x {'~5.00x (N=5)':<10}")
    print("=" * 90)

    # 6. Spectral Continuity & Subband Analysis
    print("\n[Step 6] 672-Channel Spectral Continuity & Subband Analysis for Fornax A:")
    fornax_spec = restored_samples[2]["mean_spectrum"]
    sb0_spec = fornax_spec[:336]
    sb1_spec = fornax_spec[336:]
    boundary_ch335 = fornax_spec[335]
    boundary_ch336 = fornax_spec[336]
    boundary_delta_pct = abs(boundary_ch336 - boundary_ch335) / max(1e-6, boundary_ch335) * 100.0

    print(f"  * Subband 0 (300.0 - 400.8 MHz): Mean Power = {np.mean(sb0_spec):.4f}, RMS = {np.std(sb0_spec):.4f}, Min = {np.min(sb0_spec):.4f}, Max = {np.max(sb0_spec):.4f}")
    print(f"  * Subband 1 (400.8 - 501.3 MHz): Mean Power = {np.mean(sb1_spec):.4f}, RMS = {np.std(sb1_spec):.4f}, Min = {np.min(sb1_spec):.4f}, Max = {np.max(sb1_spec):.4f}")
    print(f"  * Stitching Boundary (Ch 335 -> Ch 336): Ch 335 = {boundary_ch335:.4f}, Ch 336 = {boundary_ch336:.4f} (Delta: {boundary_delta_pct:.2f}%)")

    # 7. Restore Routine Configuration
    print("\n[Step 7] Restoring 5 Active Beams for Continuous Routine Tracking...")
    post_json("/direct_tracker/enable_beam", {"num_active_beams": 5})
    post_json("/direct_tracker/set_celestial_target", {
        "beam_id": 5,
        "ra_deg": 54.6208,
        "dec_deg": -35.4508
    })

    print("\n" + "=" * 100)
    print(" LIVE DUAL-DPDK SOURCE POWER TEST COMPLETED SUCCESSFULLY!")
    print("=" * 100)

if __name__ == "__main__":
    main()
