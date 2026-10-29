"""Summarize a live beamformer run of charts/config/64antennas_beamform_live.yaml.

    /usr/bin/python3 -m test_charts.sim.check_live_run --run-dir /dev/shm/charts_bf_live \
        [--results-dir test_charts/results/live_<date>] [--decode-frames 20]

Per shard: frame count, FPGA_SEQ_NUM continuity (step T_raw), arrival rate from file
mtimes, intensity / int8-parameter / exposure pairing by (shard, FPGA_SEQ_NUM), exposure
(valid-spectra fraction, i.e. packet loss seen by the beamformer), invalid int8 chunks and
decoded intensity statistics for the first frames. Writes report.json; prints a summary.
Not a numerical validation: it checks that the live product is complete and consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from . import intensity_product as ip
from .packet_loss import read_product
from .real_data import read_charts_metadata

FRAME_SECONDS = 15360 / 300000.0


def frames(run_dir: Path, prefix: str) -> list[Path]:
    rx = re.compile(rf"^{re.escape(prefix)}_(\d+)\.bin$")
    return sorted((p for p in run_dir.iterdir() if rx.match(p.name)),
                  key=lambda p: int(rx.match(p.name).group(1)))


def check_shard(run_dir: Path, shard: int, decode_frames: int) -> dict:
    d = ip.load_descriptor(run_dir / f"intensity.shard{shard}.product.json")
    inten = frames(run_dir, f"intensity.shard{shard}")
    qpar = frames(run_dir, f"qparams.shard{shard}")
    expo = frames(run_dir, f"exposure.shard{shard}")
    seq = np.array([read_charts_metadata(p).fpga_seq_num for p in inten], np.int64)
    step = np.diff(seq)
    mtimes = np.array([p.stat().st_mtime for p in inten])
    keys_q = [read_charts_metadata(p).fpga_seq_num for p in qpar]
    keys_e = [read_charts_metadata(p).fpga_seq_num for p in expo]
    t_raw = d["T_raw"]
    exp_shape = tuple(d["exposure"]["shape"]) if d.get("exposure") else None
    valid = []
    for p in expo:
        _, e = read_product(p, d["exposure"]["dtype"], exp_shape)
        valid.append(e.astype(np.float64))
    valid = np.array(valid)                                   # [N][windows][336]
    k = d["K"]
    decoded = []
    nan_chunks = 0
    for n in range(min(decode_frames, len(inten), len(qpar))):
        q = ip.read_intensity(d, inten[n])
        pr = ip.read_quant_params(d, qpar[n])
        x = ip.decode_int8(d, q, pr, shard=shard)
        nan_chunks += int(np.isnan(x).any(axis=(1, 2)).sum())
        decoded.append(np.nanmedian(x, axis=(0, 1)))         # per beam
    return {
        "shard": shard,
        "descriptor": {"schema": d["schema"], "input_mode": d["input_mode"], "K": k,
                       "dtype": d["intensity"]["dtype"], "shape": d["intensity"]["shape"],
                       "reconstructable": bool(d.get("int8") and d["int8"]["reconstructable"]),
                       "weights": d["weights"]},
        "frames": {"intensity": len(inten), "qparams": len(qpar), "exposure": len(expo)},
        "seq_first": int(seq[0]) if seq.size else None,
        "seq_last": int(seq[-1]) if seq.size else None,
        "seq_steps": {str(int(s)): int(c) for s, c in zip(*np.unique(step, return_counts=True))},
        "missing_frames_from_seq": int(np.sum(np.maximum(step // t_raw - 1, 0))),
        "products_paired_by_seq": keys_q == seq.tolist() and keys_e == seq.tolist(),
        "write_rate_hz": float((len(mtimes) - 1) / (mtimes[-1] - mtimes[0]))
        if len(mtimes) > 2 and mtimes[-1] > mtimes[0] else None,
        "exposure": {
            "valid_fraction": float(valid.sum() / (valid.size * k)) if valid.size else None,
            "frames_with_loss": int(np.sum((valid < k).reshape(len(valid), -1).any(axis=1)[1:]))
            if valid.size else None,
            "first_frame_first_window_min": float(valid[0, 0].min()) if valid.size else None,
        },
        "decoded_frames": len(decoded),
        "decoded_invalid_window_rows": nan_chunks,
        "decoded_median_per_beam": np.median(decoded, axis=0).round(6).tolist() if decoded else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--decode-frames", type=int, default=20)
    args = ap.parse_args()
    report = {"run_dir": str(args.run_dir),
              "shards": [check_shard(args.run_dir, s, args.decode_frames) for s in (0, 1)]}
    for name in ("kotekan_config_*.json", "run_metadata_*.json"):
        found = sorted(args.run_dir.glob(name))
        report[name.split("_*")[0]] = found[-1].name if found else None
    s0, s1 = report["shards"]
    report["shards_aligned"] = s0["seq_first"] == s1["seq_first"]
    if args.results_dir:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    for s in report["shards"]:
        print(f"shard {s['shard']}: frames {s['frames']}, seq steps {s['seq_steps']}, "
              f"missing frames {s['missing_frames_from_seq']}, paired {s['products_paired_by_seq']}, "
              f"rate {s['write_rate_hz']} Hz, valid fraction {s['exposure']['valid_fraction']}, "
              f"invalid int8 rows {s['decoded_invalid_window_rows']}")
    print(f"shards aligned: {report['shards_aligned']}")


if __name__ == "__main__":
    main()
