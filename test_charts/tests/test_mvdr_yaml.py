"""Config generation for the chartsMVDRBeams replay pipeline (test_charts/sim/mvdr.py)."""

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.mvdr import create_mvdr_yaml, expected_weight_updates  # noqa: E402


def _make(tmp_path, **kw):
    xs = [-0.6 * (i % 8) for i in range(64)]
    ys = [0.6 * (i // 8) for i in range(64)]
    path = create_mvdr_yaml(
        tmp_path / "mvdr.yaml",
        baseband_dir=tmp_path / "win",
        baseband_name="charts_sim_64ant",
        output_dir=tmp_path / "out",
        num_frames=99,
        positions_x_m=xs,
        positions_y_m=ys,
        beams=[(0.4677, 0.4801), (-0.1, 0.25)],
        **kw,
    )
    return yaml.safe_load(path.read_text())


def test_stage_block_and_buffer_sizes(tmp_path):
    cfg = _make(tmp_path)
    st = cfg["mvdr_beams"]
    assert st["kotekan_stage"] == "chartsMVDRBeams"
    assert st["beams"] == [[0.4677, 0.4801], [-0.1, 0.25]]
    assert len(st["positions_x_m"]) == 64 and len(st["positions_y_m"]) == 64
    assert st["flush_after_frames"] == 99
    assert "gains_file" not in st
    # the stage holds 2 H + 1 frames and needs at least 2 H + 2
    assert cfg["host_correlation_buffer"]["num_frames"] >= 2 * st["cov_half_window"] + 2
    nb, nf, ne = 2, 336, 64
    assert cfg["mvdr_power_buffer"]["frame_size"] == 2 * nf * nb * 4
    assert (
        cfg["mvdr_weights_buffer"]["frame_size"] == 16 + nf * nb * ne * 8 + nf * nb * 4
    )
    assert cfg["correlator_dump"]["in_buf"] == "host_correlation_buffer"
    assert cfg["correlator_dump"]["base_dir"].endswith("out/corr")
    assert cfg["mvdr_power_dump"]["base_dir"].endswith("out/mvdr")
    assert cfg["mvdr_weights_dump"]["base_dir"].endswith("out/mvdr")


def test_options(tmp_path):
    cfg = _make(
        tmp_path,
        cov_half_window=40,
        gains_file=tmp_path / "g.npy",
        write_correlation=False,
    )
    assert cfg["host_correlation_buffer"]["num_frames"] == 84
    assert cfg["mvdr_beams"]["gains_file"].endswith("g.npy")
    assert cfg["mvdr_beams"]["antenna_order"] == "descending"
    assert "correlator_dump" not in cfg


def test_rejects_bad_positions(tmp_path):
    with pytest.raises(ValueError):
        create_mvdr_yaml(
            tmp_path / "m.yaml",
            tmp_path,
            "w",
            tmp_path,
            10,
            [0.0] * 3,
            [0.0] * 3,
            [(0, 0)],
        )


def test_shutdown_counts(tmp_path):
    cfg = _make(tmp_path)
    assert cfg["baseband_reader"]["end_interrupt"] is False
    assert cfg["mvdr_power_dump"]["exit_after_n_files"] == 99
    assert cfg["correlator_dump"]["exit_after_n_files"] == 99
    assert cfg["mvdr_weights_dump"]["exit_after_n_files"] == 99  # update_every 1
    cfg = _make(tmp_path, update_every=4)
    assert cfg["mvdr_weights_dump"]["exit_after_n_files"] == expected_weight_updates(
        99, 10, 1, 9, 4
    )


def test_expected_weight_updates():
    # 40 frames, H 4, g 1, min 6, U 4: first window with 6 frames at t = 4, then every 4th
    # frame while the window keeps 6 frames (t = 36 has 5): 4, 8, ..., 32 -> 8 updates, as
    # measured on the stage with the same parameters
    assert expected_weight_updates(40, 4, 1, 6, 4) == 8
    assert expected_weight_updates(30, 10, 1, 9, 1) == 30
    assert expected_weight_updates(3, 10, 1, 9, 1) == 0
