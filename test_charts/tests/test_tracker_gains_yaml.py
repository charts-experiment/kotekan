"""Config generation for the calibrated tracker replay (test_charts/sim/tracker_gains.py)."""

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.tracker_gains import create_tracker_gains_yaml  # noqa: E402


def _cmd(cfg):
    return [
        c for c in cfg["gpu"]["commands"] if c["name"] == "cudaDirectBeamTrackerCommand"
    ][0]


def test_beams_gains_and_shutdown(tmp_path):
    p = create_tracker_gains_yaml(
        tmp_path / "t.yaml",
        tmp_path / "win",
        "w",
        tmp_path / "out",
        "beams",
        20,
        beams=[(0.4677, 0.4801), (-0.1, 0.25)],
        gains_file=tmp_path / "g.npy",
        gains_first_channel=0,
    )
    cfg = yaml.safe_load(p.read_text())
    c = _cmd(cfg)
    assert (c["source_l0"], c["source_m0"]) == (0.4677, 0.4801)
    assert (c["source_l0_1"], c["source_m0_1"]) == (-0.1, 0.25)
    assert c["max_beams"] == 2 and c["initial_active_beams"] == 2
    assert c["gains_file"].endswith("g.npy") and c["gains_first_channel"] == 0
    assert c["antenna_order"] == "descending"
    assert cfg["baseband_reader"]["end_interrupt"] is False
    assert cfg["tracker_dump"]["exit_after_n_files"] == 20
    assert not any(
        x["name"] == "cudaAntennaMaskCommand" for x in cfg["gpu"]["commands"]
    )


def test_without_gains(tmp_path):
    p = create_tracker_gains_yaml(
        tmp_path / "t.yaml", tmp_path, "w", tmp_path / "o", "b", 5, beams=[(0.0, 0.0)]
    )
    c = _cmd(yaml.safe_load(p.read_text()))
    assert "gains_file" not in c and "gains_first_channel" not in c
    with pytest.raises(ValueError):
        create_tracker_gains_yaml(
            tmp_path / "t.yaml", tmp_path, "w", tmp_path, "b", 5, beams=[]
        )
