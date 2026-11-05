#!/usr/bin/env python3
"""
================================================================================
 CHARTS 3-BEAM CELESTIAL TARGET RECORDER
 Targets:
   - Beam 0: Galactic Center (Sagittarius A*)
   - Beam 1: The Sun (Dynamic Astropy Ephemeris)
   - Beam 2: Vela Pulsar (PSR B0833-45 / J0835-4510)
 Observatory:
   - Caren Observatory (-33.4211146° S, -70.8634710° W, 458m)
 Data Output:
   - 3 concurrent beams of raw complex voltages (float2)
   - 672 frequency channels (300.0 MHz to 501.3 MHz)
   - Duration: 2.048 seconds (4 frames @ 153,600 samples/frame)
   - Size: ~9.91 GB total (~2.48 GB per frame file)
================================================================================
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
    from astropy.time import Time
except ImportError:
    print("[ERROR] astropy is required. Install via: pip install astropy")
    sys.exit(1)

import numpy as np

# Observatory Location
CAREN_LAT_DEG = -33.4211146
CAREN_LON_DEG = -70.8634710
CAREN_ALT_M = 458.0
CAREN_LOC = EarthLocation(
    lat=CAREN_LAT_DEG * u.deg, lon=CAREN_LON_DEG * u.deg, height=CAREN_ALT_M * u.m
)

# Pipeline Sampling Constants
SAMPLE_PERIOD_S = 3.333333333333e-6  # 300 kSPS PFB output
SAMPLES_PER_FRAME = 153600
FRAME_DURATION_S = SAMPLES_PER_FRAME * SAMPLE_PERIOD_S  # 0.512 seconds
NUM_LOCAL_FREQ = 672
FREQ_START_MHZ = 300.0
FREQ_STEP_MHZ = 0.3
MAX_BEAMS = 3


def calculate_target_trajectories(obstime: Time) -> List[Dict]:
    """Compute instantaneous direction cosines and tracking rates for the 3 celestial targets."""
    altaz_frame = AltAz(obstime=obstime, location=CAREN_LOC)
    dt = 1.0 * u.s
    altaz_next = AltAz(obstime=obstime + dt, location=CAREN_LOC)

    # 1. Beam 0: Galactic Center (Sagittarius A*)
    sgr_a = SkyCoord("17h45m40.04s", "-29d00m28.1s", frame="icrs")

    # 2. Beam 1: The Sun
    sun = get_sun(obstime)

    # 3. Beam 2: Vela Pulsar (PSR B0833-45)
    vela = SkyCoord("08h35m20.61s", "-45d10m34.9s", frame="icrs")

    target_defs = [
        ("Galactic_Center_SgrA", sgr_a),
        ("Sun", sun),
        ("Vela_Pulsar_B0833-45", vela),
    ]

    results = []
    lst_hours = float(obstime.sidereal_time("apparent", longitude=CAREN_LOC.lon).hour)

    for beam_idx, (name, coord) in enumerate(target_defs):
        aa = coord.transform_to(altaz_frame)
        alt_rad = float(aa.alt.rad)
        az_rad = float(aa.az.rad)

        l0 = float(np.cos(alt_rad) * np.sin(az_rad))
        m0 = float(np.cos(alt_rad) * np.cos(az_rad))

        aa_next = coord.transform_to(altaz_next)
        l_next = float(np.cos(aa_next.alt.rad) * np.sin(aa_next.az.rad))
        m_next = float(np.cos(aa_next.alt.rad) * np.cos(aa_next.az.rad))

        dl_sample = float((l_next - l0) * SAMPLE_PERIOD_S)
        dm_sample = float((m_next - m0) * SAMPLE_PERIOD_S)

        results.append(
            {
                "beam_id": beam_idx,
                "name": name,
                "ra_deg": float(coord.ra.deg),
                "dec_deg": float(coord.dec.deg),
                "alt_deg": float(aa.alt.deg),
                "az_deg": float(aa.az.deg),
                "l0": l0,
                "m0": m0,
                "dl": dl_sample,
                "dm": dm_sample,
                "above_horizon": bool(aa.alt.deg > 0.0),
                "lst_hours": lst_hours,
            }
        )

    return results


def generate_pipeline_yaml(
    targets: List[Dict], out_yaml_path: Path, output_dir: Path, n_files: int = 4
):
    """Generate specialized Kotekan YAML configuration for 3-beam celestial capture."""
    yaml_content = f"""# ==============================================================================
# Auto-generated 3-Beam Celestial Target Recording Pipeline
# Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
# Targets:
#   Beam 0: {targets[0]['name']} (l0={targets[0]['l0']:+.5f}, m0={targets[0]['m0']:+.5f})
#   Beam 1: {targets[1]['name']} (l0={targets[1]['l0']:+.5f}, m0={targets[1]['m0']:+.5f})
#   Beam 2: {targets[2]['name']} (l0={targets[2]['l0']:+.5f}, m0={targets[2]['m0']:+.5f})
# ==============================================================================

type: config
log_level: INFO

buffer_depth: 30
gpu_buffer_depth: 2

samples_per_data_set: {SAMPLES_PER_FRAME}
integration_spectra: 320
capture_n_frames: 45000
cpu_affinity: [0, 1]

sample_period_s: {SAMPLE_PERIOD_S}
frame_arrival_period: samples_per_data_set * sample_period_s

num_elements: 32
num_local_freq: {NUM_LOCAL_FREQ}
n_channels_per_packet: 168
packets_per_spectrum: 4
bytes_per_dataset: n_channels_per_packet * packets_per_spectrum * samples_per_data_set

spacing_m: 0.6
max_beams: {MAX_BEAMS}
initial_active_beams: {MAX_BEAMS}
sizeof_complex_float: 8

main_pool:
  kotekan_metadata_pool: chartsMetadata
  num_metadata_objects: buffer_depth * 15

network_capture_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: bytes_per_dataset * num_elements
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

mask_buffer_zeros:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set
  numa_node: 0
  metadata_pool: main_pool

host_formed_beams_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * max_beams * sizeof_complex_float
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

dpdk:
  kotekan_stage: dpdkCore
  lcore_cpu_map: [3]
  main_lcore_cpu: 4
  lcore_port_map:
    - [0]
  max_rx_pkt_len: 9000
  num_mbufs: 9600
  handlers:
    - dpdk_handler: rfsocHandlerNoClk
      out_buffer: network_capture_buf
      alignment: 4000
      mask_buf: mask_buffer_zeros
    - dpdk_handler: none

zero_samples:
  kotekan_stage: zeroSamples
  out_buf: network_capture_buf
  lost_samples_buf: mask_buffer_zeros
  zero_value: 0
  sample_size: 21504

gpu:
  profiling: false
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaBeamTrackerCommand
      gpu_mem_voltage: voltage
      gpu_mem_formed_beams: formed_beams
      num_elements: 32
      num_local_freq: {NUM_LOCAL_FREQ}
      samples_per_data_set: {SAMPLES_PER_FRAME}
      integration_spectra: 320
      spacing_m: 0.6
      max_beams: {MAX_BEAMS}
      initial_active_beams: {MAX_BEAMS}
      buffer_depth: 2
      site_lat_deg: {CAREN_LAT_DEG}
      site_lon_deg: {CAREN_LON_DEG}
      site_alt_m: {CAREN_ALT_M}
      freq_start_hz: {FREQ_START_MHZ * 1e6}
      freq_step_hz: {FREQ_STEP_MHZ * 1e6}
      # Antenna mapping convention outside kernel: descending (raw 31 -> physical 0)
      antenna_order: "descending"
      # Working antennas: raw elements 31 down to 24 (physical antennas 0 to 7)
      active_raw_elements: [24, 25, 26, 27, 28, 29, 30, 31]
      # Beam 0: {targets[0]['name']}
      source_l0: {targets[0]['l0']:.8f}
      source_m0: {targets[0]['m0']:.8f}
      source_dl: {targets[0]['dl']:.12e}
      source_dm: {targets[0]['dm']:.12e}
      # Beam 1: {targets[1]['name']}
      source_l0_1: {targets[1]['l0']:.8f}
      source_m0_1: {targets[1]['m0']:.8f}
      source_dl_1: {targets[1]['dl']:.12e}
      source_dm_1: {targets[1]['dm']:.12e}
      # Beam 2: {targets[2]['name']}
      source_l0_2: {targets[2]['l0']:.8f}
      source_m0_2: {targets[2]['m0']:.8f}
      source_dl_2: {targets[2]['dl']:.12e}
      source_dm_2: {targets[2]['dm']:.12e}
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: formed_beams
      out_buf: host_formed_beams
  gpu_0:
    kotekan_stage: cudaProcess
    gpu_id: 0
    buffer_depth: 2
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_formed_beams: host_formed_beams_buffer

inspect_beams:
  kotekan_stage: restInspectFrame
  in_buf: host_formed_beams_buffer
  len: 4194304

raw_file_write_beams:
  kotekan_stage: rawFileWrite
  in_buf: host_formed_beams_buffer
  file_name: formed_beams_celestial
  base_dir: {output_dir.as_posix()}
  file_ext: bin
  num_frames_per_file: 1
  prefix_hostname: false
  exit_after_n_files: {n_files}
"""
    out_yaml_path.write_text(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description="Record 3-beam celestial observation with CHARTS Kotekan Beam Tracker"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Observation duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/tracker/celestial_3beams_2s",
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate targets and write YAML without running Kotekan",
    )
    parser.add_argument(
        "--kotekan-bin",
        type=str,
        default="./build/kotekan/kotekan",
        help="Path to kotekan binary",
    )
    args = parser.parse_args()

    # Calculate frame count (each frame is 0.512s)
    n_files = max(1, int(np.ceil(args.duration / FRAME_DURATION_S)))
    exact_duration_s = n_files * FRAME_DURATION_S
    frame_bytes = SAMPLES_PER_FRAME * NUM_LOCAL_FREQ * MAX_BEAMS * 8
    total_bytes = n_files * frame_bytes

    out_dir = Path(args.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out_dir = Path("/tmp/celestial_3beams_2s")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] Cannot create {args.output_dir}, falling back to {out_dir}")

    now = Time.now()
    targets = calculate_target_trajectories(now)

    print("=" * 80)
    print(" CHARTS 3-BEAM CELESTIAL TARGET RECORDING PLAN")
    print("=" * 80)
    print(f" Observation Epoch (UTC) : {now.iso}")
    print(f" Caren Sidereal Time (LST): {targets[0]['lst_hours']:.4f} hours")
    print(
        f" Requested Duration       : {args.duration:.2f} s -> Exact: {exact_duration_s:.3f} s ({n_files} frames)"
    )
    print(
        f" Frequency Band           : {FREQ_START_MHZ:.1f} MHz to {FREQ_START_MHZ + NUM_LOCAL_FREQ * FREQ_STEP_MHZ:.1f} MHz (672 channels)"
    )
    print(
        f" Total Expected Data Size : {total_bytes / (1024**3):.2f} GB ({frame_bytes / (1024**3):.2f} GB per file)"
    )
    print(f" Output Directory         : {out_dir}")
    print("-" * 80)
    print(
        f" {'Beam':<5} {'Target Name':<24} {'RA (deg)':<10} {'Dec (deg)':<10} {'Alt (deg)':<10} {'Az (deg)':<10} {'Horizon'}"
    )
    print("-" * 80)
    for t in targets:
        horizon_str = "[ABOVE]" if t["above_horizon"] else "[BELOW]"
        print(
            f" {t['beam_id']:<5} {t['name']:<24} {t['ra_deg']:<10.3f} {t['dec_deg']:<10.3f} {t['alt_deg']:<10.2f} {t['az_deg']:<10.2f} {horizon_str}"
        )
    print("-" * 80)

    # Write runtime YAML
    runtime_yaml = Path("charts/32antennas_3beams_record.yaml")
    generate_pipeline_yaml(targets, runtime_yaml, out_dir, n_files=n_files)
    print(f"[OK] Generated Kotekan configuration: {runtime_yaml}")

    # Write metadata sidecar JSON
    sidecar_meta = {
        "observation_utc": now.iso,
        "mjd": float(now.mjd),
        "site": {
            "name": "Observatorio Caren",
            "lat_deg": CAREN_LAT_DEG,
            "lon_deg": CAREN_LON_DEG,
            "alt_m": CAREN_ALT_M,
        },
        "frequency": {
            "num_channels": NUM_LOCAL_FREQ,
            "start_mhz": FREQ_START_MHZ,
            "step_mhz": FREQ_STEP_MHZ,
            "end_mhz": FREQ_START_MHZ + NUM_LOCAL_FREQ * FREQ_STEP_MHZ,
        },
        "sampling": {
            "sample_period_s": SAMPLE_PERIOD_S,
            "samples_per_frame": SAMPLES_PER_FRAME,
            "frame_duration_s": FRAME_DURATION_S,
            "num_frames": n_files,
            "total_duration_s": exact_duration_s,
            "max_beams": MAX_BEAMS,
            "sample_format": "complex64 (float2: 4-byte real + 4-byte imag)",
            "frame_bytes": frame_bytes,
        },
        "targets": targets,
    }
    meta_json_path = out_dir / "celestial_3beams_metadata.json"
    with open(meta_json_path, "w") as f:
        json.dump(sidecar_meta, f, indent=2)
    print(f"[OK] Saved metadata sidecar: {meta_json_path}")

    if args.dry_run:
        print("\n[DRY RUN] Config and metadata generated. To execute manually:")
        print(f"  sudo {args.kotekan_bin} -c {runtime_yaml} -b 127.0.0.1:12048")
        return

    print("\n[STARTING] Launching Kotekan 3-beam recording pipeline...")
    cmd = ["sudo", args.kotekan_bin, "-c", str(runtime_yaml), "-b", "127.0.0.1:12048"]
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    t_start = time.perf_counter()
    proc = subprocess.run(cmd)
    t_elapsed = time.perf_counter() - t_start

    print("=" * 80)
    if proc.returncode == 0:
        print(f" [SUCCESS] Capture completed cleanly in {t_elapsed:.2f} seconds!")
        bin_files = sorted(out_dir.glob("*.bin"))
        print(f" Recorded {len(bin_files)} files in {out_dir}:")
        for bf in bin_files:
            size_mb = bf.stat().st_size / (1024 * 1024)
            print(f"   - {bf.name} ({size_mb:.2f} MB)")
        print(f"\n To analyze and view formed beams:")
        print(f"   python charts/analyze_recorded_beams.py --dir {out_dir}")
    else:
        print(f" [ERROR] Kotekan exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
