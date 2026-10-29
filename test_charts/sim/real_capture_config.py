"""Capture configuration for an exact beamformer-vs-correlator comparison on real data.

Derives a Kotekan YAML from the production correlator + triggered-baseband config
(`charts/config/64antennas_corr_triggered_baseband.yaml`, left unchanged) with only:

* `integration_frames` (default 2: 1.024 s correlator integrations, so a 5 s capture
  contains at least 3 whole integrations whatever its phase);
* `output_dir` (every alias of it, e.g. rawFileWrite / BasebandWriter / run metadata);
* the baseband antenna selection (`selected_antennas`, `num_saved_antennas`) and the
  run metadata `signal_antennas`;
* `correlation_output_frames` (run length; Kotekan exits after that many files).

    /usr/bin/python3 -m test_charts.sim.real_capture_config --output-dir /data/corr/real_p1_<date> \
        --antennas 4 5 --minutes 10 --yaml /tmp/real_p1_capture.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "charts/config/64antennas_corr_triggered_baseband.yaml"
SPECTRUM_RATE = 300000.0


def _replace(node, old, new):
    if isinstance(node, dict):
        return {k: _replace(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return new if node == old else [_replace(v, old, new) for v in node]
    return new if node == old else node


def capture_config(output_dir: str, antennas: list[int], minutes: float,
                   integration_frames: int = 2, source: Path = PRODUCTION) -> dict:
    cfg = yaml.safe_load(source.read_text())
    if sorted(set(antennas)) != list(antennas) or not 1 <= len(antennas) <= 5:
        raise ValueError("antennas: 1..5 distinct charts_logical indices in increasing order")
    if any(not 0 <= a < 64 for a in antennas):
        raise ValueError("antennas must be charts_logical 0..63")
    old_dir = cfg["output_dir"]
    old_sel = cfg["baseband_capture"]["selected_antennas"]
    cfg = _replace(cfg, old_dir, output_dir)
    cfg = _replace(cfg, old_sel, list(antennas))
    cfg["num_saved_antennas"] = len(antennas)
    cfg["integration_frames"] = integration_frames
    t_raw = int(cfg["samples_per_data_set"])
    seconds = integration_frames * t_raw / SPECTRUM_RATE
    cfg["correlation_output_frames"] = max(1, round(minutes * 60 / seconds))
    meta = cfg.get("run_metadata")
    if isinstance(meta, dict):
        meta["signal_antennas"] = list(antennas)
    for stage in ("baseband_capture", "baseband_save"):
        if cfg[stage].get("selected_antennas") != list(antennas):
            raise RuntimeError(f"{stage} selection was not replaced")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", required=True, help="data directory of the run")
    ap.add_argument("--antennas", type=int, nargs="+", required=True,
                    help="working antennas, charts_logical, increasing")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--integration-frames", type=int, default=2)
    ap.add_argument("--source", type=Path, default=PRODUCTION)
    ap.add_argument("--yaml", type=Path, required=True, help="where to write the config")
    args = ap.parse_args()
    cfg = capture_config(args.output_dir, args.antennas, args.minutes,
                         args.integration_frames, args.source)
    args.yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.yaml}: integration_frames {cfg['integration_frames']}, "
          f"{cfg['correlation_output_frames']} integrations, antennas {args.antennas}, "
          f"output {args.output_dir}")


if __name__ == "__main__":
    main()
