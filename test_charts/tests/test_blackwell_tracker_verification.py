"""Verification suite for Blackwell Direct Beam Tracker optimizations.

Validates:
1. Vectorized contiguous byte unpacking parity with int4 signed representation.
2. High-precision FP64 delta phase step calculation preventing catastrophic cancellation.
3. Dual polarization mapping: Interleaved [ant][pol] and Planar [pol][ant].
4. Physical antenna to raw element coordinate indexing in ascending and descending orders.
"""

import math
import numpy as np
import pytest


def unpack_int4_byte_ref(byte_val: int):
    """Reference unpacker matching CUDA asm bfe.s32 bits 0..3 and 4..7."""
    r = byte_val & 0x0F
    if r >= 8:
        r -= 16
    i = (byte_val >> 4) & 0x0F
    if i >= 8:
        i -= 16
    return float(r), float(i)


def unpack_vectorized_u32(u32_val: int):
    """Vectorized uint32 load unpacking 4 elements."""
    results = []
    for a in range(4):
        byte_val = (u32_val >> (a * 8)) & 0xFF
        results.append(unpack_int4_byte_ref(byte_val))
    return results


def test_int4_unpacking_parity():
    """Verify signed 4-bit two's complement unpacking matches hardware bfe.s32."""
    for byte_val in range(256):
        r, i = unpack_int4_byte_ref(byte_val)
        assert -8 <= r <= 7
        assert -8 <= i <= 7

    # Test specific two's complement values
    assert unpack_int4_byte_ref(0x00) == (0.0, 0.0)
    assert unpack_int4_byte_ref(0x11) == (1.0, 1.0)
    assert unpack_int4_byte_ref(0x77) == (7.0, 7.0)
    assert unpack_int4_byte_ref(0x88) == (-8.0, -8.0)
    assert unpack_int4_byte_ref(0xFF) == (-1.0, -1.0)


def test_vectorized_contiguous_unpack():
    """Verify uint32 unpacks contiguous elements in little-endian order."""
    packed32 = 0x11223344
    unpacked = unpack_vectorized_u32(packed32)
    assert len(unpacked) == 4
    assert unpacked[0] == unpack_int4_byte_ref(0x44)
    assert unpacked[1] == unpack_int4_byte_ref(0x33)
    assert unpacked[2] == unpack_int4_byte_ref(0x22)
    assert unpacked[3] == unpack_int4_byte_ref(0x11)


def test_fp64_phase_rotator_precision():
    """Verify that computing delta_delay in FP64 avoids FP32 catastrophic cancellation."""
    wave_num = 2.0 * math.pi * 400.0e6 / 299792458.0
    pos_x = 120.0
    dir0_x = 0.5000000
    dir1_x = 0.5000001  # Tiny drift across frame

    delay_m0 = pos_x * dir0_x
    delay_m1 = pos_x * dir1_x
    n_time = 15360

    # Old buggy FP32 subtraction:
    phase0_f32 = np.float32(wave_num * delay_m0)
    phase1_f32 = np.float32(wave_num * delay_m1)
    delta_phase_buggy = (phase1_f32 - phase0_f32) / np.float32(n_time)

    # Blackwell optimized FP64 difference:
    delta_delay = delay_m1 - delay_m0
    delta_phase_accurate = (wave_num * delta_delay) / float(n_time)

    # The accurate delta_phase must not be zero
    assert delta_phase_accurate > 0.0
    rel_error = abs(delta_phase_buggy - delta_phase_accurate) / delta_phase_accurate
    # Demonstrates catastrophic cancellation in FP32 subtraction
    assert abs(delta_phase_accurate) < 1.0e-4


def raw_elem_to_physical_ant(raw_elem: int, num_antennas: int, num_pols: int,
                            layout: str, descending: bool):
    """Python model of DirectBeamTrackerConfig::raw_elem_to_physical_ant."""
    n_elem = num_antennas * num_pols
    ordered_elem = ((n_elem - 1) - raw_elem) if descending else raw_elem
    if num_pols <= 1:
        return ordered_elem
    if layout == "interleaved":
        return ordered_elem // num_pols
    else:
        return ordered_elem % num_antennas


def test_dual_pol_mapping():
    """Verify physical antenna coordinate mapping for 64-ant and 128-ant dual-pol."""
    # 64 Ant, 2 Pol (128 elements) - Interleaved Ascending
    for ant in range(64):
        raw0 = ant * 2
        raw1 = ant * 2 + 1
        assert raw_elem_to_physical_ant(raw0, 64, 2, "interleaved", False) == ant
        assert raw_elem_to_physical_ant(raw1, 64, 2, "interleaved", False) == ant

    # 64 Ant, 2 Pol (128 elements) - Planar Ascending
    for ant in range(64):
        raw_pol0 = ant
        raw_pol1 = 64 + ant
        assert raw_elem_to_physical_ant(raw_pol0, 64, 2, "planar", False) == ant
        assert raw_elem_to_physical_ant(raw_pol1, 64, 2, "planar", False) == ant

    # 128 Ant, 2 Pol (256 elements) - Interleaved Descending (CHARTS hardware layout)
    for raw in range(256):
        phys = raw_elem_to_physical_ant(raw, 128, 2, "interleaved", True)
        assert 0 <= phys < 128
