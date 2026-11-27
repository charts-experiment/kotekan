"""Writes pirate_network_send_golden.bin, the bytes test_pirateNetworkSend expects from the stage.

The bytes are built with the reference sender of the CHARTS pirate fork (github.com/GonUchile/pirate,
branch charts-64beams, charts/07_emitter/protocol.py), so the test checks the stage against it:

    python3 pirate_network_send_golden.py <path to charts/07_emitter>

The frames, beams, channels, seqs and metadata must match test_pirateNetworkSend.cpp.
"""

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import protocol  # noqa: E402

NUM_FREQ = 4
BEAMS = range(1, 3)  # beam_range of the test
SEQ_PER_MINICHUNK = 256 * 320

METADATA = {
    "zone_nfreq": [21504],
    "zone_freq_edges": [299.85, 501.75],
    "beamset": 1,
    "beam_ids": [7, 8],
    "beam_positions_x": [0.0, 0.1],
    "beam_positions_y": [0.0, 0.0],
    "unix_ns_at_seq_0": 1772483060000000000,
    "dt_ns_per_seq": 3333,
    "seq_per_frb_time_sample": 320,
    "noise_variance": 1.0,
    "version": 2,
    "freq_channels": [100, 101, 102, 103],
}


def minichunk(index, seq):
    """Minichunk of frame `index` of the test, as protocol.Connection.send_minichunk sends it."""
    b = np.array(BEAMS)[:, None, None]
    f = np.arange(NUM_FREQ)[None, :, None]
    t = np.arange(256)[None, None, :]
    codes = (7 * b + 3 * f + t + 5 * index) % 16 - 8
    bits = 16 * b[..., 0] + f[..., 0] + index
    scales_offsets = np.stack([0x3800 + bits, 0x2000 + bits], axis=-1).astype(np.uint16)
    data = protocol.pack_int4((codes & 0x0F).astype(np.uint8))
    return (
        struct.pack("<IQ", protocol.MAGIC, seq)
        + scales_offsets.view(np.float16).tobytes()
        + data.tobytes()
    )


# The stage sends the metadata as compact JSON with sorted keys (nlohmann::json::dump).
text = json.dumps(METADATA, sort_keys=True, separators=(",", ":"))
golden = (
    protocol.connection_header(text)
    + minichunk(0, 3 * SEQ_PER_MINICHUNK)
    + minichunk(1, 5 * SEQ_PER_MINICHUNK)
)
out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pirate_network_send_golden.bin"
)
with open(out, "wb") as fh:
    fh.write(golden)
print(f"wrote {len(golden)} bytes to {out}")
