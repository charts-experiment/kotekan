"""Compare live fine-mode runs with different upchannelizer gains G(u).

    ~/charts-utils/.venv/bin/python -m test_charts.sim.compare_upchan_gain_runs \
        --run g1=/mnt/A02/beamform/muos_gain_20261102/g1 \
        --run g0H=/mnt/A02/beamform/muos_gain_20261102/g0H \
        --run ref1030=/mnt/A02/beamform/muos_fine_20261030 \
        --results-dir test_charts/results/muos_gain_live_20261102

For each run and shard, it decodes every Nth int8 frame with the published decoder
(test_charts.sim.intensity_product) and uses the one-hot beams (labels "one-hot antenna a"
in the weights directions file, else beams 7-10). The quantities follow
FINE_PATH_SCALLOPING_QUANTIZATION.md §4 step 9:

* Output scalloping. The median over quiet coarse channels of I(f, u)/mean_u I(f, u), as
  centre/edge dB and max |dB|. "Output" is the product as written (it includes G²);
  "ungained" divides by G(u)² from the descriptor.
* Fine level. σ per component at the channel centre from the mean one-hot window power,
  assuming the linear regime: σ ≈ sqrt(mean / (2K)). With G = 1 this is a lower bound of
  the true σ (rounding to zero dominates), so the predicted zero fraction is also given.
* MUOS band vs off band. Per coarse channel, the ungained mean power over u in the
  360-380 MHz band, relative to the median of the off band (340-355, 385-395 MHz), in dB:
  median, 5-95 % range and minimum per beam. The f-engine equalizes each coarse bin's RMS
  to its target, so a linear fine path gives values near 0 dB within the calibration
  tolerance. On 2026-10-30 (G = 1) the band showed -3 to -7 dB boxes. A plausible cause is
  the int4 rounding at σ ≈ 0.2: non-Gaussian MUOS signals with lighter tails produce fewer
  nonzero codes at the same power.
* int8 floor. The fraction of finite codes at -127 (a chunk minimum) per one-hot beam.
Observed data only; differences between runs also contain sky, RFI and level changes, so
run the A/B variants back to back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import intensity_product as ip

QUIET = ((340.0, 358.0), (385.0, 399.0))
BAND = (360.0, 380.0)
OFF = ((340.0, 355.0), (385.0, 395.0))


def onehot_beams(d: dict) -> tuple[list[int], list[str]]:
    """One-hot beams from the directions file next to the weights (also after the
    /data/beamform -> /mnt/A02/beamform move); beams 7-10 otherwise."""
    wfile = str(d.get("weights", {}).get("file", ""))
    for base in (wfile, wfile.replace("/data/beamform/", "/mnt/A02/beamform/")):
        try:
            labels = json.loads(Path(base).with_name("muos_directions.json").read_text())["beam_labels"]
        except (OSError, KeyError, ValueError):
            continue
        beams = [i for i, lab in enumerate(labels) if lab.startswith("one-hot")]
        if beams:
            return beams, [labels[i] for i in beams]
    return [7, 8, 9, 10], ["beam 7", "beam 8", "beam 9", "beam 10"]


def in_ranges(nu, ranges):
    m = np.zeros(nu.shape, bool)
    for lo, hi in ranges:
        m |= (nu > lo) & (nu < hi)
    return m


def analyse_shard(run_dir: Path, shard: int, every: int, max_frames: int):
    d = ip.load_descriptor(run_dir / f"intensity.shard{shard}.product.json")
    files = sorted(run_dir.glob(f"intensity.shard{shard}_*.bin"))
    if not files:
        raise SystemExit(f"no intensity.shard{shard} files in {run_dir}")
    files = files[::every][:max_frames]
    beams, labels = onehot_beams(d)
    cols = ip.beam_positions(d, beams)          # weight-file beams -> output positions
    g = ip.upchan_gain(d).astype(np.float64)                         # [10752]
    nu = ip.frequencies_hz(d) / 1e6
    k = d["K"]
    acc = np.zeros((nu.size, len(beams)))
    cnt = np.zeros((nu.size, len(beams)))
    floor = np.zeros(len(beams))
    finite = np.zeros(len(beams))
    for p in files:
        codes = ip.read_intensity(d, p)
        params = ip.read_quant_params(d, Path(str(p).replace("intensity.", "qparams.")))
        i = ip.decode_int8(d, codes, params, shard=shard)[:, :, cols]   # [W][FF][b]
        acc += np.nansum(i, axis=0)
        cnt += np.sum(np.isfinite(i), axis=0)
        q = codes[1][:, :, cols]
        floor += np.sum(q == -127, axis=(0, 1))
        finite += np.sum(q != ip.INVALID_CODE, axis=(0, 1))
    mean = acc / np.maximum(cnt, 1)                                    # [FF][b] window power
    out = {"run_dir": str(run_dir), "shard": shard, "frames_decoded": len(files),
           "schema": d["schema"], "beams": dict(zip(map(str, beams), labels)),
           "upchan_gain": d.get("history", {}).get("upchan_gain",
                                                    {"unity": True, "source": "v1 descriptor"}),
           "int8_floor_fraction": (floor / np.maximum(finite, 1)).tolist()}
    grid = mean.reshape(336, 32, len(beams))
    gg = g.reshape(336, 32)[:, :, None]
    coarse_nu = nu.reshape(336, 32).mean(axis=1)
    quiet = in_ranges(coarse_nu, QUIET)
    if quiet.any():
        for name, x in (("output", grid), ("ungained", grid / gg ** 2)):
            norm = x / x.mean(axis=1, keepdims=True)
            prof = np.median(norm[quiet], axis=0)                      # [32][b]
            prof_db = 10 * np.log10(prof / prof[15:17].mean(axis=0))
            out[f"scalloping_{name}"] = {
                "quiet_coarse_channels": int(quiet.sum()),
                "centre_edge_dB": (-prof_db[[0, 31]].mean(axis=0)).tolist(),
                "max_abs_dB": np.max(np.abs(prof_db), axis=0).tolist(),
                "profile_dB_beam_mean": prof_db.mean(axis=1).round(3).tolist()}
        centre = grid[quiet][:, 15:17].mean(axis=(0, 1))               # [b]
        sigma = np.sqrt(centre / (2 * k))
        out["fine_sigma_centre_linear_estimate"] = sigma.tolist()
        out["fine_sigma_note"] = ("sqrt(mean window power / 2K) at u = 15, 16 in quiet "
                                  "channels; a lower bound when most codes are 0 (G = 1)")
    band, off = in_ranges(coarse_nu, (BAND,)), in_ranges(coarse_nu, OFF)
    if band.any() and off.any():
        coarse_power = (grid / gg ** 2).mean(axis=1)                   # [336][b]
        ref = np.median(coarse_power[off], axis=0)
        steps = 10 * np.log10(coarse_power[band] / ref)
        out["muos_band_vs_off_dB"] = {"median": np.median(steps, axis=0).tolist(),
                                "p05": np.percentile(steps, 5, axis=0).tolist(),
                                "p95": np.percentile(steps, 95, axis=0).tolist(),
                                "min": steps.min(axis=0).tolist(),
                                "band_coarse_channels": int(band.sum())}
    return out, coarse_nu, mean


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--every", type=int, default=20, help="decode every Nth frame")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report, spectra = {}, {}
    for item in args.run:
        label, path = item.split("=", 1)
        report[label] = {}
        for shard in args.shards:
            r, coarse_nu, mean = analyse_shard(Path(path), shard, args.every, args.max_frames)
            report[label][shard] = r
            spectra[f"{label}_shard{shard}"] = mean
            spectra[f"{label}_shard{shard}_coarse_nu"] = coarse_nu
    (args.results_dir / "compare_upchan_gain.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(args.results_dir / "compare_upchan_gain_spectra.npz", **spectra)
    for label, shards in report.items():
        for shard, r in shards.items():
            print(f"== {label} shard {shard}: {r['frames_decoded']} frames, "
                  f"G {'unity' if r['upchan_gain'].get('unity') else r['upchan_gain'].get('source')}")
            for key in ("scalloping_output", "scalloping_ungained"):
                if key in r:
                    print(f"   {key:20s} centre/edge dB {np.round(r[key]['centre_edge_dB'], 2)}"
                          f"  max|dB| {np.round(r[key]['max_abs_dB'], 2)}")
            if "fine_sigma_centre_linear_estimate" in r:
                print(f"   fine sigma (centre) {np.round(r['fine_sigma_centre_linear_estimate'], 3)}")
            if "muos_band_vs_off_dB" in r:
                s = r["muos_band_vs_off_dB"]
                print(f"   MUOS band vs off dB median {np.round(s['median'], 2)}  "
                      f"p05 {np.round(s['p05'], 2)}  min {np.round(s['min'], 2)}")
            print(f"   int8 floor fraction {np.round(r['int8_floor_fraction'], 4)}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for label, shards in report.items():
            if 0 in shards and "scalloping_output" in shards[0]:
                ax[0].plot(shards[0]["scalloping_output"]["profile_dB_beam_mean"], ".-", label=label)
            key = f"{label}_shard0"
            if key in spectra:
                m = spectra[key].reshape(336, 32, -1)
                g = ip.upchan_gain(ip.load_descriptor(Path(shards[0]["run_dir"])
                                                      / "intensity.shard0.product.json"))
                p = (m / g.reshape(336, 32, 1) ** 2).mean(axis=(1, 2))
                ax[1].semilogy(spectra[f"{key}_coarse_nu"], p / np.median(p), lw=0.7, label=label)
        ax[0].set_xlabel("fine index u")
        ax[0].set_ylabel("output dB re centre (quiet channels, one-hot mean)")
        ax[0].legend()
        ax[1].set_xlabel("MHz (shard 0)")
        ax[1].set_ylabel("ungained coarse power / median")
        ax[1].axvspan(*BAND, color="y", alpha=0.15)
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(args.results_dir / "compare_upchan_gain.png", dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
