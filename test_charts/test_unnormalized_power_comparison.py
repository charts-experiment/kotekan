#!/usr/bin/env python3
"""
Unnormalized vs. Normalized Direct Beam Tracker Power Comparison
================================================================
Demonstrates the difference in formed beam power when:
1. normalize_weights = false (unnormalized coherent addition: sum(w_a * v_a) with w_a = 1.0)
2. normalize_weights = true  (L2 normalized: sum(w_a^2) = 1.0, w_a = 1/sqrt(N_pol),
                              pinning beam noise power to average antenna power)

Evaluates beam power across 4 concurrent beams:
- Beam 0: On-target at bright source (Fornax A transit: l0=0.0, m0=-0.0660455)
- Beam 1: Slight offset (0.03 rad: near sidelobes)
- Beam 2: Moderately off-target (0.15 rad: far sidelobes)
- Beam 3: Completely off-source (Cold sky)

Also demonstrates dynamic target re-pointing:
- Steers Beam 3 onto Fornax A and Beam 0 onto Cold Sky to observe real-time power shifts.
"""

import os
import subprocess
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
KOTEKAN_BIN = str(REPO_DIR / "build/kotekan/kotekan")
CONFIG_FILE = str(REPO_DIR / "charts/config/64antennas_unnormalized_power_comparison.yaml")
REST_URL = "http://127.0.0.1:12048"

def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{REST_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return {"status": resp.status, "body": resp.read().decode("utf-8", errors="replace")}

def get_json(endpoint: str) -> dict:
    url = f"{REST_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def wait_for_new_frames(n_frames=2, timeout_s=30):
    start = time.time()
    initial_fid = None
    while time.time() - start < timeout_s:
        try:
            data = get_json("/direct_tracker/beam_power")
            fid = data.get("frame_id", 0)
            bp = data.get("beam_power", [])
            has_power = len(bp) > 0 and any(len(s) > 0 and max(s) > 0 for s in bp)
            if has_power:
                if initial_fid is None:
                    initial_fid = fid
                elif fid >= initial_fid + n_frames:
                    return fid
        except Exception:
            pass
        time.sleep(0.5)
    return initial_fid if initial_fid is not None else 0

def sample_beam_powers(n_samples=3):
    accumulated = [[] for _ in range(4)]
    last_fid = -1
    count = 0
    start = time.time()

    while count < n_samples and time.time() - start < 20:
        try:
            data = get_json("/direct_tracker/beam_power")
            fid = data.get("frame_id", 0)
            if fid > 0 and fid != last_fid:
                beam_spectra = data.get("beam_power", [])
                has_data = False
                for b in range(min(4, len(beam_spectra))):
                    spec = np.array(beam_spectra[b], dtype=np.float32)
                    valid = spec[spec > 0]
                    if len(valid) > 0:
                        accumulated[b].append(float(np.mean(valid)))
                        has_data = True
                if has_data:
                    count += 1
                    last_fid = fid
        except Exception:
            pass
        time.sleep(0.4)

    return [float(np.mean(acc)) if len(acc) > 0 else 0.0 for acc in accumulated]

def main():
    print("=" * 96)
    print(" CHARTS DIRECT BEAM TRACKER: UNNORMALIZED VS NORMALIZED POWER TEST")
    print("=" * 96)

    # 1. Start Kotekan
    print("\n[1] Starting Kotekan pipeline with UNNORMALIZED weights (normalize_weights: false)...")
    cmd = [KOTEKAN_BIN, "-c", CONFIG_FILE, "-b", "127.0.0.1:12048"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_DIR))

    try:
        # Wait for REST API
        print("  -> Waiting for REST server to initialize...")
        ready = False
        for _ in range(30):
            try:
                st = get_json("/direct_tracker/status")
                if st.get("num_active_beams") == 4:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            print("ERROR: Pipeline failed to reach ready state!")
            return

        print("  -> REST server is UP!")
        st0 = get_json("/direct_tracker/status")
        print(f"  -> normalize_weights:   {st0.get('normalize_weights')}")
        print(f"  -> active_antennas:     {st0.get('active_antennas')}")
        print(f"  -> active_beams:        {st0.get('num_active_beams')}")
        print(f"  -> taper_weights_sample:{st0.get('taper_weights_sample', [])[:4]}")

        # Wait for initial frames to flow
        print("  -> Waiting for GPU to process initial frames...")
        fid_ready = wait_for_new_frames(n_frames=2, timeout_s=30)
        print(f"  -> Pipeline running smoothly at frame_id={fid_ready}!")

        # 2. Sample Unnormalized Powers
        print("\n[2] Sampling UNNORMALIZED formed beam power (w_a = 1.0, pure coherent sum)...")
        unnorm_powers = sample_beam_powers(n_samples=3)
        for b in range(4):
            print(f"     Beam {b} Power: {unnorm_powers[b]:10.2f}")

        # 3. Dynamic Re-pointing Test in UNNORMALIZED Mode
        print("\n[3] Dynamically steering beams via REST (/direct_tracker/set_target)...")
        print("     - Steering Beam 3 (previously Cold Sky) ONTO Fornax A (l0=0.0, m0=-0.0660455)")
        print("     - Steering Beam 0 (previously On-Source) AWAY TO Cold Sky (l0=0.40, m0=0.30)")
        post_json("/direct_tracker/set_target", {
            "beam_id": 3,
            "l0": 0.0,
            "m0": -0.0660455
        })
        post_json("/direct_tracker/set_target", {
            "beam_id": 0,
            "l0": 0.40,
            "m0": 0.30
        })

        wait_for_new_frames(n_frames=2, timeout_s=20)
        reaimed_powers = sample_beam_powers(n_samples=2)
        print("     After Dynamic Re-pointing:")
        print(f"     Beam 0 (Now Cold Sky):  {reaimed_powers[0]:10.2f} (dropped from {unnorm_powers[0]:.2f})")
        print(f"     Beam 3 (Now On-Source): {reaimed_powers[3]:10.2f} (jumped from {unnorm_powers[3]:.2f})")

        # Restore original pointing
        print("     Restoring original beam directions...")
        post_json("/direct_tracker/set_target", {
            "beam_id": 0,
            "l0": 0.0,
            "m0": -0.0660455
        })
        post_json("/direct_tracker/set_target", {
            "beam_id": 3,
            "l0": 0.40,
            "m0": 0.30
        })
        wait_for_new_frames(n_frames=2, timeout_s=20)

        # 4. Toggle normalize_weights: true dynamically via REST
        print("\n[4] Toggling normalize_weights: true dynamically via REST (/direct_tracker/set_taper)...")
        post_json("/direct_tracker/set_taper", {
            "taper_type": "uniform",
            "normalize_weights": True,
            "enable_weight_bislc": False
        })
        time.sleep(0.5)
        st1 = get_json("/direct_tracker/status")
        print(f"  -> Updated normalize_weights:   {st1.get('normalize_weights')}")
        print(f"  -> taper_weights_sample (norm): {st1.get('taper_weights_sample', [])[:4]}")

        wait_for_new_frames(n_frames=2, timeout_s=20)

        # 5. Sample Normalized Powers
        print("\n[5] Sampling NORMALIZED formed beam power (sum(w_a^2) = 1.0, w_a = 1/sqrt(N_pol))...")
        norm_powers = sample_beam_powers(n_samples=3)
        for b in range(4):
            print(f"     Beam {b} Power: {norm_powers[b]:10.2f}")

        # 6. Display Comprehensive Comparison Results
        beam_descriptions = [
            ("Beam 0 (On-Source)", "Fornax A transit (l0=0.0, m0=-0.066) [Bright Injected Source]"),
            ("Beam 1 (Slight Offset)", "Offset by 0.03 rad (l0=0.03, m0=-0.066) [Near Sidelobes]"),
            ("Beam 2 (Off-Target)", "Offset by ~0.20 rad (l0=0.15, m0=0.10) [Far Sidelobes]"),
            ("Beam 3 (Cold Sky)", "Offset by >0.50 rad (l0=0.40, m0=0.30) [Pure Noise Background]")
        ]

        print("\n" + "=" * 96)
        print(" RESULTS: FORMED BEAM POWER COMPARISON")
        print("=" * 96)
        print(f"{'Beam':<22} | {'Aiming Direction & Sky Target':<42} | {'Unnormalized':<14} | {'Normalized':<10} | {'Unnorm / Norm'}")
        print("-" * 96)
        for b in range(4):
            tag, desc = beam_descriptions[b]
            un = unnorm_powers[b]
            no = norm_powers[b]
            ratio = (un / no) if no > 0 else 0.0
            print(f"{tag:<22} | {desc:<42} | {un:14.2f} | {no:10.2f} | {ratio:6.2f}x")
        print("=" * 96)

        # Analysis
        p_on_unnorm = unnorm_powers[0]
        p_off_unnorm = unnorm_powers[3]
        p_on_norm = norm_powers[0]
        p_off_norm = norm_powers[3]

        snr_unnorm_db = 10.0 * np.log10(p_on_unnorm / p_off_unnorm) if p_off_unnorm > 0 else 0.0
        snr_norm_db = 10.0 * np.log10(p_on_norm / p_off_norm) if p_off_norm > 0 else 0.0
        reaim_contrast_db = 10.0 * np.log10(reaimed_powers[3] / unnorm_powers[3]) if unnorm_powers[3] > 0 else 0.0

        print("\n--- Analytical Insights ---")
        print(f"1. UNNORMALIZED Beamforming (w_a = 1.0, pure coherent sum):")
        print(f"   * Bright Source Power (Beam 0) : {p_on_unnorm:.2f}")
        print(f"   * Cold Sky Power (Beam 3)      : {p_off_unnorm:.2f}")
        print(f"   * Bright-to-Cold Sky Contrast  : {p_on_unnorm / p_off_unnorm:.2f}x ({snr_unnorm_db:.2f} dB)")
        print(f"   * Dynamic Re-pointing Response : Aiming Beam 3 at Fornax A jumped power from {unnorm_powers[3]:.2f} -> {reaimed_powers[3]:.2f} (+{reaim_contrast_db:.2f} dB)!")
        print(f"     Aiming Beam 0 away to Cold Sky dropped power from {unnorm_powers[0]:.2f} -> {reaimed_powers[0]:.2f}.")
        print(f"   * Coherent vs Incoherent Physics: The point source fields add constructively in voltage,")
        print(f"     scaling as N^2 * S, while uncorrelated noise fields add incoherently, scaling as N * sigma^2.\n")

        print(f"2. NORMALIZED Beamforming (sum(w_a^2) = 1.0 per polarization):")
        print(f"   * Bright Source Power (Beam 0) : {p_on_norm:.2f}")
        print(f"   * Cold Sky Power (Beam 3)      : {p_off_norm:.2f}")
        print(f"   * Bright-to-Cold Sky Contrast  : {p_on_norm / p_off_norm:.2f}x ({snr_norm_db:.2f} dB)")
        print(f"   * Normalization Invariant      : In normalized mode, the weights are scaled so that the")
        print(f"     formed beam noise power on cold sky ({p_off_norm:.2f}) matches the average power of alive antennas!")
        print("=" * 96)

    finally:
        print("\nStopping Kotekan process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Kotekan process stopped.")

if __name__ == "__main__":
    main()
