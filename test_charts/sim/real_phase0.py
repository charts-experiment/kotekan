"""Inspect a recorded correlator + triggered-baseband run.

Data inspection only (no Kotekan, no beamformer): correlator frame timing and loss,
baseband capture span and int4 levels, correlator/baseband time overlap, and which
charts_logical inputs carry antenna signal. Writes report.json and .npy summaries.

    /usr/bin/python3 -m test_charts.sim.real_phase0 --run-dir /data/corr_1020 --event 44 \
        --results-dir test_charts/results/real_inspect_<date>
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from .real_data import (BYTE_IM, BYTE_POWER, BYTE_RE, corr_auto_imag, corr_matrix,
                        open_baseband, read_charts_metadata, read_corr_raw)

SPECTRUM_RATE = 300000.0  # raw spectra per second (10/3 us)


def utc(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, timezone.utc).isoformat(timespec="milliseconds")


def load_config(run_dir: Path) -> dict:
    (path,) = sorted(run_dir.glob("kotekan_config_*.json"))[-1:]
    return json.loads(path.read_text())


def corr_files(run_dir: Path, shard: int) -> list[Path]:
    rx = re.compile(rf".*_corr_freq_{shard}_(\d+)\.bin$")
    files = [p for p in run_dir.iterdir() if rx.match(p.name)]
    return sorted(files, key=lambda p: int(rx.match(p.name).group(1)))


def correlator_timing(files: list[Path], integration: int) -> dict:
    seq = np.empty(len(files), np.int64)
    lost = np.empty(len(files), np.int64)
    t0 = np.empty(len(files), np.int64)
    cf0 = set()
    for i, p in enumerate(files):
        m = read_charts_metadata(p)
        seq[i], lost[i], t0[i] = m.fpga_seq_num, m.lost_timesamples, m.time0_fpga
        cf0.add((int(m.nfreq), int(m.coarse_freq[0])))
    d = np.diff(seq)
    return {
        "frames": len(files), "seq": seq, "time0_us": t0, "lost": lost,
        "summary": {
            "frames": len(files),
            "first_seq": int(seq[0]), "last_seq": int(seq[-1]),
            "integration_spectra_config": integration,
            "seq_step_values": {str(int(k)): int(v) for k, v in zip(*np.unique(d, return_counts=True))},
            "gaps": int(np.sum(d != integration)),
            "nfreq_coarse_freq0": sorted(cf0),
            "first_utc": utc(t0[0] / 1e6), "last_utc": utc(t0[-1] / 1e6),
            "lost_timesamples": {"frames_with_loss": int(np.sum(lost > 0)),
                                 "total": int(lost.sum()), "max": int(lost.max())},
        },
    }


def baseband_levels(bb) -> dict:
    """int4 statistics per antenna (beamformer convention, low nibble real)."""
    n_ant, n_freq = bb.num_elements, bb.num_freq
    hist = np.zeros((n_ant, 256), np.int64)
    chan_power = np.zeros((n_freq, n_ant))
    spectra = 0
    zero_spectra = 0
    seq_ok = True
    power_lut = BYTE_POWER.astype(np.float64)
    for i in range(len(bb.meta)):
        x = bb.record(i)
        spectra += x.shape[0]
        zero_spectra += int(np.sum(~x.reshape(x.shape[0], -1).any(axis=1)))
        for a in range(n_ant):
            hist[a] += np.bincount(x[..., a].ravel(), minlength=256)
        chan_power += power_lut[x].sum(axis=0)
        if i and bb.meta["frame_fpga_seq"][i] != (bb.meta["frame_fpga_seq"][i - 1]
                                                   + bb.meta["valid_to"][i - 1]):
            seq_ok = False
    chan_power /= spectra
    comp = np.arange(-8, 8)
    out = []
    for a in range(n_ant):
        h = hist[a]
        n = h.sum()
        re_h = np.array([h[BYTE_RE == c].sum() for c in comp]) / n
        im_h = np.array([h[BYTE_IM == c].sum() for c in comp]) / n
        small = (np.abs(BYTE_RE) <= 1) & (np.abs(BYTE_IM) <= 1)
        cp = chan_power[:, a]
        out.append({
            "mean_power_code2": float((h * BYTE_POWER).sum() / n),
            "rms_re": float(np.sqrt((h * BYTE_RE ** 2).sum() / n)),
            "rms_im": float(np.sqrt((h * BYTE_IM ** 2).sum() / n)),
            "frac_zero_byte": float(h[0] / n),
            "frac_both_components_within_1": float(h[small].sum() / n),
            "frac_minus8_re": float(re_h[0]), "frac_minus8_im": float(im_h[0]),
            "component_hist_re": dict(zip(comp.tolist(), np.round(re_h, 6).tolist())),
            "component_hist_im": dict(zip(comp.tolist(), np.round(im_h, 6).tolist())),
            "channel_power": {"min": float(cp.min()), "median": float(np.median(cp)),
                              "max": float(cp.max()), "zero_channels": int(np.sum(cp == 0))},
        })
    return {"spectra": spectra, "all_zero_spectra": zero_spectra,
            "records_contiguous": seq_ok, "per_antenna": out,
            "channel_power": chan_power}


def overlap(bb, corr: dict, integration: int) -> dict:
    start = int(bb.meta["frame_fpga_seq"][0])
    end = int(bb.meta["frame_fpga_seq"][-1] + bb.meta["valid_to"][-1])
    seq = corr["seq"]
    inside = np.nonzero((seq >= start) & (seq + integration <= end))[0]
    touching = np.nonzero((seq < end) & (seq + integration > start))[0]
    return {
        "baseband_seq": [start, end], "baseband_spectra": end - start,
        "baseband_seconds": (end - start) / SPECTRUM_RATE,
        "event_start_end_fpga": [int(bb.meta["event_start_fpga"][0]),
                                 int(bb.meta["event_end_fpga"][0])],
        "baseband_first_utc": utc(float(bb.meta["time0_ctime"][0])),
        "correlator_integrations_fully_inside": inside.tolist(),
        "correlator_integrations_touching": [
            {"file_index": int(k), "seq": [int(seq[k]), int(seq[k] + integration)],
             "covered_fraction": (min(end, seq[k] + integration) - max(start, seq[k]))
             / integration} for k in touching],
    }


def antenna_activity(files_by_shard: list[list[Path]], indices: list[int],
                     integration: int) -> dict:
    """Autocorrelation ratio and coherence |V'_ab|/sqrt(V'_aa V'_bb) over sampled frames."""
    autos, coh, unit, imag_frac = [], [], [], []
    for k in indices:
        per_shard_auto, per_shard_coh, per_shard_unit, per_shard_imag = [], [], [], []
        for files in files_by_shard:
            raw = read_corr_raw(files[k])
            m = corr_matrix(raw)
            p = m[:, np.arange(64), np.arange(64)].real
            norm = np.sqrt(np.maximum(p[:, :, None] * p[:, None, :], 1e-30))
            c = m / norm
            per_shard_auto.append(p)
            per_shard_coh.append(np.abs(c))
            per_shard_unit.append(np.where(np.abs(m) > 0, m / np.maximum(np.abs(m), 1e-30), 0))
            per_shard_imag.append(corr_auto_imag(raw) != 0)
        autos.append(np.concatenate(per_shard_auto))      # [672][64]
        coh.append(np.concatenate(per_shard_coh))         # [672][64][64]
        unit.append(np.concatenate(per_shard_unit))
        imag_frac.append(np.concatenate(per_shard_imag))
    autos, coh, unit = np.array(autos), np.array(coh), np.array(unit)
    auto_med = np.median(autos, axis=(0, 1))
    ratio = auto_med / np.median(auto_med)
    coh_med = np.median(coh, axis=(0, 1))                 # median over frames and channels
    np.fill_diagonal(coh_med, 0)
    stability = np.median(np.abs(unit.mean(axis=0)), axis=0)  # |<V/|V|>_frames|, channel median
    np.fill_diagonal(stability, 0)
    floor = 1 / np.sqrt(integration)
    off = coh_med[np.triu_indices(64, 1)]
    return {
        "frames_used": indices,
        "noise_only_coherence_expected": floor,
        "offdiagonal_coherence_median": float(np.median(off)),
        "per_antenna": [{"antenna": a, "kotekan_raw": 63 - a,
                         "auto_ratio_to_median": float(ratio[a]),
                         "max_coherence": float(coh_med[a].max()),
                         "max_coherence_with": int(coh_med[a].argmax()),
                         "auto_imag_nonzero_frac": float(np.mean(np.array(imag_frac)[..., a]))}
                        for a in range(64)],
        "coherence": coh_med, "phase_stability": stability,
    }


def pair_table(act: dict, antennas: list[int]) -> list[dict]:
    rows = []
    for i, a in enumerate(antennas):
        for b in antennas[i + 1:]:
            rows.append({"pair": [a, b], "coherence": float(act["coherence"][a, b]),
                         "phase_stability": float(act["phase_stability"][a, b])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=Path("/data/corr_1020"))
    ap.add_argument("--event", type=int, default=44)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--activity-frames", type=int, default=80,
                    help="correlator frames sampled across the run for antenna activity")
    args = ap.parse_args()
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.run_dir)
    t_raw = int(cfg["samples_per_data_set"])
    integration = int(cfg["integration_frames"]) * t_raw
    bb_spectra = int(cfg["baseband_spectra_per_frame"])
    run_meta = sorted(args.run_dir.glob("run_metadata_*.json"))
    report = {"run_dir": str(args.run_dir), "event": args.event,
              "config": {"samples_per_data_set": t_raw,
                         "integration_frames": int(cfg["integration_frames"]),
                         "integration_spectra": integration,
                         "integration_seconds": integration / SPECTRUM_RATE,
                         "baseband_spectra_per_frame": bb_spectra,
                         "element_order": cfg.get("element_order")},
              "run_metadata": json.loads(run_meta[-1].read_text()) if run_meta else None}

    files = [corr_files(args.run_dir, s) for s in (0, 1)]
    corr = [correlator_timing(f, integration) for f in files]
    report["correlator"] = [c["summary"] for c in corr]
    if not np.array_equal(corr[0]["seq"][:min(map(len, files))],
                          corr[1]["seq"][:min(map(len, files))]):
        report["correlator_shards_aligned"] = False
    else:
        report["correlator_shards_aligned"] = True

    sel = json.loads((args.run_dir / f"baseband_raw_{args.event}/antenna_selection.json")
                     .read_text())
    antennas = sel["selected_antennas"]
    report["baseband"] = {"selected_antennas": antennas,
                          "index_convention": sel["index_convention"], "shards": []}
    channel_power = []
    for shard, freq_id in enumerate((0, 336)):
        bb = open_baseband(args.run_dir / f"baseband_raw_{args.event}"
                           / f"baseband_{args.event}_{freq_id}.data", bb_spectra)
        lv = baseband_levels(bb)
        channel_power.append(lv.pop("channel_power"))
        for a, row in zip(antennas, lv["per_antenna"]):
            row["antenna"] = a
        ov = overlap(bb, corr[shard], integration)
        report["baseband"]["shards"].append({"freq_id": freq_id, "records": len(bb.meta),
                                             **lv, "overlap": ov})
    np.save(out / "baseband_channel_power.npy", np.concatenate(channel_power))

    n = min(map(len, files))
    touching = report["baseband"]["shards"][0]["overlap"]["correlator_integrations_touching"]
    indices = sorted(set(np.linspace(0, n - 1, args.activity_frames).astype(int).tolist())
                     | {t["file_index"] for t in touching})
    act = antenna_activity(files, indices, integration)
    act["selected_antenna_pairs"] = pair_table(act, antennas)
    np.save(out / "correlator_coherence.npy", act.pop("coherence"))
    np.save(out / "correlator_phase_stability.npy", act.pop("phase_stability"))
    ranked = sorted(act["per_antenna"], key=lambda r: -r["auto_ratio_to_median"])
    act["top_by_auto_ratio"] = [(r["antenna"], round(r["auto_ratio_to_median"], 2))
                                for r in ranked[:10]]
    ranked = sorted(act["per_antenna"], key=lambda r: -r["max_coherence"])
    act["top_by_max_coherence"] = [(r["antenna"], round(r["max_coherence"], 4),
                                    r["max_coherence_with"]) for r in ranked[:10]]
    report["antenna_activity"] = act

    (out / "report.json").write_text(json.dumps(report, indent=1, default=int) + "\n")
    print(json.dumps({k: report[k] for k in ("config", "correlator")}, indent=1, default=int))


if __name__ == "__main__":
    main()
