"""Time-integrated beam spectra of live fine-mode runs: power vs frequency per beam.

    # 1. integrate (decodes every frame; cached per run and shard)
    ~/charts-utils/.venv/bin/python -m test_charts.sim.plot_beam_spectra integrate \
        --run g1=/mnt/A02/beamform/muos_gain_20261102/g1 \
        --run g0H=/mnt/A02/beamform/muos_gain_20261102/g0H --out DIR
    # 2. plot from the cache
    ~/charts-utils/.venv/bin/python -m test_charts.sim.plot_beam_spectra plot --out DIR

Integration: the mean window power over all decoded windows of all frames, NaN-aware.
Invalid int8 chunks are skipped. The result is [10752 fine channels][B beams] per shard (B
from the descriptor: 64 for runs before 2026-11-14, 128 since),
with the frequencies from the descriptor and G(u) from history.upchan_gain.

Plots:
* `beam_spectra_coarse.png`. Small multiples, one panel per labelled beam, both shards
  (300-501 MHz). Each panel shows the coarse-channel mean of the ungained power (÷G²) in
  dB re that beam's median, for every run. The panels compare spectral shape; the absolute
  level differs by run.
* `beam_spectra_fine_zoom.png`. Fine resolution in a 1.5 MHz window, the output power as
  written (including G²), normalized to its median. It shows the scalloping with G = 1
  and its removal with G = g0/H.
* `beam_frequency_map_<run>.png`. All B beams × coarse channel, in dB re each beam's
  median, on a diverging scale centred at 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import intensity_product as ip

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (fixed order)
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e4e3de", "#ffffff"


# --------------------------------------------------------------------------- integrate

def integrate_shard(run_dir: Path, shard: int, every: int = 1):
    d = ip.load_descriptor(run_dir / f"intensity.shard{shard}.product.json")
    files = sorted(run_dir.glob(f"intensity.shard{shard}_*.bin"))[::every]
    acc = np.zeros(tuple(d["intensity"]["shape"][1:]))                 # [FF][B]
    cnt = np.zeros_like(acc)
    for n, p in enumerate(files):
        codes = ip.read_intensity(d, p)
        params = ip.read_quant_params(d, Path(str(p).replace("intensity.", "qparams.")))
        i = ip.decode_int8(d, codes, params, shard=shard)
        acc += np.nansum(i, axis=0)
        cnt += np.sum(np.isfinite(i), axis=0)
        if n % 100 == 0:
            print(f"  {run_dir.name} shard {shard}: {n}/{len(files)}", flush=True)
    return {"power": acc / np.maximum(cnt, 1), "windows": cnt, "beams": ip.beam_indices(d),
            "freq_mhz": ip.frequencies_hz(d) / 1e6, "gain": ip.upchan_gain(d),
            "frames": len(files)}


def beam_labels(run_dir: Path) -> list[str]:
    """Label of every output beam; the directions file labels weight-file beams, so it is
    indexed through beam_indices (output beam j = weight beam written[j])."""
    d = ip.load_descriptor(run_dir / "intensity.shard0.product.json")
    beams = ip.beam_indices(d)
    wfile = str(d.get("weights", {}).get("file", ""))
    for base in (wfile, wfile.replace("/data/beamform/", "/mnt/A02/beamform/")):
        try:
            labels = json.loads(Path(base).with_name("muos_directions.json").read_text())["beam_labels"]
            return [labels[b] for b in beams]
        except (OSError, KeyError, ValueError, IndexError):
            continue
    return [f"beam {b}" for b in beams]


def cmd_integrate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for item in args.run:
        label, path = item.split("=", 1)
        path = Path(path)
        for shard in (0, 1):
            cache = args.out / f"integrated_{label}_shard{shard}.npz"
            if cache.exists() and not args.force:
                print(f"cached {cache}")
                continue
            r = integrate_shard(path, shard, args.every)
            np.savez_compressed(cache, **r)
            print(f"wrote {cache} ({r['frames']} frames)")
        runs[label] = {"dir": str(path), "labels": beam_labels(path)}
    meta = args.out / "runs.json"
    old = json.loads(meta.read_text()) if meta.exists() else {}
    old.update(runs)
    meta.write_text(json.dumps(old, indent=1))


# --------------------------------------------------------------------------- plot

def load(out: Path, label: str):
    """Returns power [21504][B'] with the weight-file beam of every column (beams); both
    shards must have written the same beams (output_beam_groups)."""
    s = [np.load(out / f"integrated_{label}_shard{k}.npz") for k in (0, 1)]
    beams = [x["beams"] if "beams" in x else np.arange(x["power"].shape[1]) for x in s]
    if not np.array_equal(beams[0], beams[1]):
        raise SystemExit(f"run {label}: shards wrote different beams {beams[0].tolist()} / "
                         f"{beams[1].tolist()}; plot them separately")
    power = np.concatenate([x["power"] for x in s])                     # [21504][B']
    gain = np.concatenate([x["gain"] for x in s]).astype(np.float64)
    freq = np.concatenate([x["freq_mhz"] for x in s])
    frames = int(s[0]["frames"])
    return power, gain, freq, frames, beams[0].astype(int)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7, length=2)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(x)


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    meta = json.loads((args.out / "runs.json").read_text())
    labels_run = list(meta)
    data = {r: load(args.out, r) for r in labels_run}
    # Beams are weight-file indices (--beams, --zoom-beams); each run maps them to its own
    # output columns. runs.json labels are per output column of that run.
    label_of = {}
    col_of = {}
    for r in labels_run:
        run_beams = data[r][4]
        col_of[r] = {int(b): j for j, b in enumerate(run_beams)}
        for j, b in enumerate(run_beams):
            label_of.setdefault(int(b), meta[r]["labels"][j])
    common = sorted(set.intersection(*(set(c) for c in col_of.values())))
    beams = args.beams if args.beams else common[:11]

    # ---- coarse small multiples -----------------------------------------------------
    ncol = 3
    nrow = -(-len(beams) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.3 * nrow), sharex=True, sharey=True,
                             squeeze=False)
    for ax, b in zip(axes.flat, beams):
        style(ax)
        for i, r in enumerate(labels_run):
            power, gain, freq, frames, _ = data[r]
            if b not in col_of[r]:
                continue                                    # beam not written in this run
            coarse = (power[:, col_of[r][b]] / gain ** 2).reshape(-1, 32).mean(axis=1)
            cf = freq.reshape(-1, 32).mean(axis=1)
            ax.plot(cf, db(coarse / np.nanmedian(coarse)), color=SERIES[i], lw=0.9,
                    label=f"{r} ({frames} frames)")
        ax.axvspan(360, 380, color=INK2, alpha=0.07, lw=0)
        ax.set_title(f"{b}: {label_of.get(b, 'not written')}", fontsize=8, color=INK, loc="left")
    for ax in axes.flat[len(beams):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("frequency (MHz)", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("dB re median", fontsize=8, color=INK2)
    lo, hi = args.ylim
    axes[0, 0].set_ylim(lo, hi)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", fontsize=8, frameon=False, ncol=len(names))
    fig.suptitle("Time-integrated beam power per coarse channel (÷G², mean over the 32 fine "
                 "bins); shaded: MUOS 360-380 MHz", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out / "beam_spectra_coarse.png", dpi=130)
    plt.close(fig)

    # ---- fine zoom ----------------------------------------------------------------
    z0, z1 = args.zoom
    fig, axes = plt.subplots(1, len(args.zoom_beams), figsize=(5 * len(args.zoom_beams), 3),
                             sharey=True, squeeze=False)
    for ax, b in zip(axes[0], args.zoom_beams):
        style(ax)
        for i, r in enumerate(labels_run):
            power, gain, freq, frames, _ = data[r]
            if b not in col_of[r]:
                continue
            m = (freq > z0) & (freq < z1)
            p = power[m, col_of[r][b]]
            ax.plot(freq[m], db(p / np.nanmedian(p)), color=SERIES[i], lw=0.9, label=r)
        ax.set_title(f"{b}: {label_of.get(b, 'not written')}", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("frequency (MHz)", fontsize=8, color=INK2)
    axes[0, 0].set_ylabel("output power, dB re median", fontsize=8, color=INK2)
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Fine channels {z0}-{z1} MHz as written (includes G²): coarse-channel "
                 "scalloping", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out / "beam_spectra_fine_zoom.png", dpi=130)
    plt.close(fig)

    # ---- beam x frequency maps ----------------------------------------------------
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#e8e7e2",
                                                     "#f2a07c", "#b5431b"])
    for r in labels_run:
        power, gain, freq, frames, run_beams = data[r]
        n_beams = len(run_beams)
        coarse = (power / gain[:, None] ** 2).reshape(-1, 32, n_beams).mean(axis=1)  # [672][B']
        rel = db(coarse / np.nanmedian(coarse, axis=0, keepdims=True)).T              # [B][672]
        cf = freq.reshape(-1, 32).mean(axis=1)
        fig, ax = plt.subplots(figsize=(13, 6))
        im = ax.imshow(rel, aspect="auto", origin="lower", cmap=cmap,
                       norm=TwoSlopeNorm(0, args.map_range[0], args.map_range[1]),
                       extent=(cf[0], cf[-1], -0.5, n_beams - 0.5), interpolation="nearest")
        ax.set_yticks(range(n_beams))
        ax.set_yticklabels([f"{b} {meta[r]['labels'][j]}" if j < 11 else str(b)
                            for j, b in enumerate(run_beams)], fontsize=5)
        ax.set_xlabel("frequency (MHz)", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=INK2)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("dB re beam median (÷G²)", fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(f"{r}: time-integrated power, all beams ({frames} frames per shard)",
                     fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(args.out / f"beam_frequency_map_{r}.png", dpi=130)
        plt.close(fig)
    print("plots in", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("integrate")
    a.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--every", type=int, default=1)
    a.add_argument("--force", action="store_true")
    p = sub.add_parser("plot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--beams", type=int, nargs="*")
    p.add_argument("--zoom", type=float, nargs=2, default=(385.0, 386.5))
    p.add_argument("--zoom-beams", type=int, nargs="+", default=[0, 7])
    p.add_argument("--ylim", type=float, nargs=2, default=(-6.0, 6.0))
    p.add_argument("--map-range", type=float, nargs=2, default=(-4.0, 4.0))
    args = ap.parse_args()
    cmd_integrate(args) if args.cmd == "integrate" else cmd_plot(args)


if __name__ == "__main__":
    main()
