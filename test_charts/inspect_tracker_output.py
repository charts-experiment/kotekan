#!/usr/bin/env python3
import os
import struct
import math
import array


def inspect_complex_voltages_file(filepath, max_beams=4):
    print("==================================================")
    print(f"Inspecting complex voltages file: {filepath}")
    file_size = os.path.getsize(filepath)
    print(f"File size on disk: {file_size:,} bytes")

    with open(filepath, "rb") as f:
        meta_size_data = f.read(4)
        if len(meta_size_data) < 4:
            print("Error: file too small")
            return
        meta_size = struct.unpack("I", meta_size_data)[0]
        print(f"Metadata header size: {meta_size} bytes")
        if meta_size > 0:
            meta_bytes = f.read(meta_size)
            print(f"Metadata read: {len(meta_bytes)} bytes")

        payload_bytes = f.read()
        print(f"Payload size: {len(payload_bytes):,} bytes")

        # Read float32 array (float2 = 2 floats per sample)
        floats = array.array("f")
        floats.frombytes(payload_bytes)
        n_floats = len(floats)
        expected_floats = 15360 * 336 * max_beams * 2

        print(
            f"Total float elements: {n_floats:,} (Expected: {expected_floats:,} for {max_beams} complex beams)"
        )

        if n_floats != expected_floats:
            print(
                f"[WARNING] Float count mismatch! Expected {expected_floats}, got {n_floats}"
            )
            return

        # Check active beam 0 (first beam)
        min_real = float("inf")
        max_real = float("-inf")
        min_imag = float("inf")
        max_imag = float("-inf")
        total_power = 0.0
        nan_count = 0
        inf_count = 0

        beam0_samples = 15360 * 336
        stride = max_beams * 2

        for idx in range(0, n_floats, stride):
            er = floats[idx]
            ei = floats[idx + 1]

            if math.isnan(er) or math.isnan(ei):
                nan_count += 1
            elif math.isinf(er) or math.isinf(ei):
                inf_count += 1
            else:
                if er < min_real:
                    min_real = er
                if er > max_real:
                    max_real = er
                if ei < min_imag:
                    min_imag = ei
                if ei > max_imag:
                    max_imag = ei
                total_power += er * er + ei * ei

        mean_power = total_power / beam0_samples
        rms_amp = math.sqrt(mean_power)

        print(f"\n--- Beam 0 Complex Voltage Statistics ---")
        print(f"  Real Range     : [{min_real:.4f} .. {max_real:.4f}]")
        print(f"  Imag Range     : [{min_imag:.4f} .. {max_imag:.4f}]")
        print(f"  Synthesized Power (Mean |E|²): {mean_power:.4f}")
        print(f"  RMS Amplitude  : {rms_amp:.4f}")
        print(f"  Contains NaN?  : {'YES [FAIL]' if nan_count > 0 else 'NO [PASS]'}")
        print(f"  Contains Inf?  : {'YES [FAIL]' if inf_count > 0 else 'NO [PASS]'}")

        print(
            "\nSample first 4 Complex Voltage pairs (Real, Imag) [time=0, freq=0, beam 0..3]:"
        )
        for b in range(min(4, max_beams)):
            r = floats[b * 2]
            i = floats[b * 2 + 1]
            mag = math.sqrt(r * r + i * i)
            phase_deg = math.degrees(math.atan2(i, r))
            print(
                f"  Beam {b}: {r:+.4f} + {i:+.4f}j  (|E|={mag:.4f}, phase={phase_deg:+.1f}°)"
            )
        print("==================================================\n")


if __name__ == "__main__":
    files = [
        "/tmp/test_charts_tracker/multibeam_complex_voltages_0000000.bin",
        "/tmp/test_charts_tracker/multibeam_complex_voltages_0000001.bin",
    ]
    for f in files:
        if os.path.exists(f):
            inspect_complex_voltages_file(f, max_beams=4)
        else:
            print(f"File not found: {f}")
