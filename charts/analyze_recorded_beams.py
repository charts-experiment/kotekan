#!/usr/bin/env python3
"""
================================================================================
 CHARTS 3-BEAM RECORDED VOLTAGE ANALYZER & HDF5 EXPORTER
 Analyzes raw complex voltage files recorded by Kotekan Beam Tracker.
 Extracts:
   - Beam 0: Galactic Center (Sagittarius A*)
   - Beam 1: The Sun
   - Beam 2: Vela Pulsar (PSR B0833-45)
 Displays:
   - Terminal power spectrum comparison & ASCII graphs
   - Signal stats (RMS voltage, dB power, peak channels)
   - Optional export to structured HDF5 dataset (.h5)
================================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Import ASCII spectrum renderer from tracker control if available
try:
    from test_charts.kotekan_tracker_control import render_ascii_spectrum
except ImportError:
    render_ascii_spectrum = None


def read_formed_beam_file(
    file_path: Path, n_time: int = 153600, n_freq: int = 672, n_beams: int = 3
) -> np.ndarray:
    """Read formed beams complex voltage array from raw .bin file."""
    payload_bytes = n_time * n_freq * n_beams * 8  # 8 bytes per complex64
    file_size = file_path.stat().st_size
    if file_size < payload_bytes:
        raise ValueError(
            f"File {file_path.name} is too small ({file_size} bytes, expected >= {payload_bytes})"
        )

    offset = file_size - payload_bytes
    # Memmap array: shape (n_time, n_freq, n_beams) of complex64
    arr = np.memmap(
        file_path,
        dtype=np.complex64,
        mode="r",
        offset=offset,
        shape=(n_time, n_freq, n_beams),
    )
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 3-beam recorded voltages from Kotekan"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/data/tracker/celestial_3beams_2s",
        help="Directory with .bin files",
    )
    parser.add_argument(
        "--export-h5", action="store_true", help="Export combined dataset to HDF5 file"
    )
    parser.add_argument(
        "--h5-output",
        type=str,
        default="celestial_3beams_combined.h5",
        help="HDF5 output filename",
    )
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        print(f"[ERROR] Directory not found: {dir_path}")
        sys.exit(1)

    bin_files = sorted(dir_path.glob("*.bin"))
    if not bin_files:
        print(f"[ERROR] No .bin files found in {dir_path}")
        sys.exit(1)

    # Load metadata sidecar if available
    meta_path = dir_path / "celestial_3beams_metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    targets = meta.get(
        "targets",
        [
            {"beam_id": 0, "name": "Galactic_Center_SgrA"},
            {"beam_id": 1, "name": "Sun"},
            {"beam_id": 2, "name": "Vela_Pulsar_B0833-45"},
        ],
    )

    n_time = meta.get("sampling", {}).get("samples_per_frame", 153600)
    n_freq = meta.get("frequency", {}).get("num_channels", 672)
    n_beams = meta.get("sampling", {}).get("max_beams", 3)
    freq_start_mhz = meta.get("frequency", {}).get("start_mhz", 300.0)
    freq_step_mhz = meta.get("frequency", {}).get("step_mhz", 0.3)

    print("=" * 80)
    print(" CHARTS RECORDED 3-BEAM VOLTAGE ANALYSIS")
    print("=" * 80)
    print(f" Directory      : {dir_path}")
    print(
        f" Frame Files    : {len(bin_files)} ({len(bin_files) * 0.512:.3f} seconds total)"
    )
    print(
        f" Format         : {n_time} time samples x {n_freq} channels x {n_beams} beams (complex64)"
    )
    print(
        f" Frequency Band : {freq_start_mhz:.1f} MHz - {freq_start_mhz + n_freq * freq_step_mhz:.1f} MHz"
    )
    print("=" * 80)

    # Accumulate power spectra across all files
    power_spectra = np.zeros((n_beams, n_freq), dtype=np.float64)
    total_time_samples = 0

    for idx, bf in enumerate(bin_files):
        print(f"Reading frame [{idx+1}/{len(bin_files)}] : {bf.name} ...")
        arr = read_formed_beam_file(bf, n_time, n_freq, n_beams)
        # Power per beam: mean over time
        for b in range(n_beams):
            beam_v = arr[:, :, b]
            power_spectra[b] += np.sum(np.abs(beam_v) ** 2, axis=0)
        total_time_samples += n_time

    power_spectra /= max(1, total_time_samples)

    print("\n" + "=" * 80)
    print(
        f" {'Beam':<5} {'Target Name':<25} {'RMS Volt':<12} {'Mean Power':<12} {'Peak Bin':<10} {'Peak Freq':<12} {'Peak dB'}"
    )
    print("-" * 80)

    db_spectra = []
    for b in range(n_beams):
        name = targets[b]["name"] if b < len(targets) else f"Beam_{b}"
        p = power_spectra[b]
        db_p = 10.0 * np.log10(p + 1e-12)
        db_spectra.append(db_p)

        rms_v = float(np.sqrt(np.mean(p)))
        mean_db = float(np.mean(db_p))
        peak_bin = int(np.argmax(db_p))
        peak_freq = freq_start_mhz + peak_bin * freq_step_mhz
        peak_db = float(db_p[peak_bin])

        print(
            f" {b:<5} {name:<25} {rms_v:10.4f}   {mean_db:8.2f} dB   {peak_bin:<10d} {peak_freq:8.2f} MHz {peak_db:8.2f} dB"
        )

    print("-" * 80)

    # Render ASCII Power Spectrum for each beam
    if render_ascii_spectrum is not None:
        for b in range(n_beams):
            name = targets[b]["name"] if b < len(targets) else f"Beam {b}"
            print(f"\n[SPECTRUM] Beam {b}: {name}")
            print(
                render_ascii_spectrum(
                    db_spectra[b],
                    height=8,
                    width=64,
                    freq_start_mhz=freq_start_mhz,
                    freq_step_mhz=freq_step_mhz,
                )
            )

    # Export to HDF5 if requested
    if args.export_h5:
        try:
            import h5py

            h5_path = dir_path / args.h5_output
            print(
                f"\n[EXPORT] Consolidating raw complex voltages into HDF5: {h5_path} ..."
            )
            with h5py.File(h5_path, "w") as hf:
                # Metadata
                hf.attrs["total_frames"] = len(bin_files)
                hf.attrs["total_time_samples"] = total_time_samples
                hf.attrs["sample_period_s"] = meta.get("sampling", {}).get(
                    "sample_period_s", 3.333333e-6
                )
                hf.attrs["total_duration_s"] = total_time_samples * meta.get(
                    "sampling", {}
                ).get("sample_period_s", 3.333333e-6)

                freq_axis = freq_start_mhz + np.arange(n_freq) * freq_step_mhz
                hf.create_dataset("frequencies_mhz", data=freq_axis)

                for b in range(n_beams):
                    t_name = targets[b]["name"] if b < len(targets) else f"beam_{b}"
                    grp = hf.create_group(f"beam_{b}_{t_name}")
                    grp.attrs["ra_deg"] = targets[b].get("ra_deg", 0.0)
                    grp.attrs["dec_deg"] = targets[b].get("dec_deg", 0.0)
                    grp.create_dataset("power_spectrum_db", data=db_spectra[b])

                    # Stream voltages into dataset
                    dset_v = grp.create_dataset(
                        "voltages",
                        shape=(total_time_samples, n_freq),
                        dtype=np.complex64,
                        chunks=(n_time, min(128, n_freq)),
                        compression="gzip",
                        compression_opts=4,
                    )
                    t_offset = 0
                    for bf in bin_files:
                        chunk = read_formed_beam_file(bf, n_time, n_freq, n_beams)
                        dset_v[t_offset : t_offset + n_time, :] = chunk[:, :, b]
                        t_offset += n_time

            print(
                f"[OK] Successfully exported {h5_path} ({h5_path.stat().st_size / (1024**3):.2f} GB)"
            )
        except ImportError:
            print(
                "[WARN] h5py not installed, skipping HDF5 export. Run: pip install h5py"
            )

    print("\n[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
