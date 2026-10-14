"""Tier-1 test: events are rendered in every written frame overlapping their span.

Before this fix an event was attached only to the frames selected for it, so a
frame selected for another event (or for the background) omitted it and the
event appeared to switch on and off between written frames.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

_test_charts_dir = Path(__file__).resolve().parent.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.generator import SimulatedEvent, build_frame_selection_schedule

FRAME_S = 0.00512
TOTAL = int(5.0 / FRAME_S)


def _events():
    return [
        SimulatedEvent(event_id="leo", event_type="rfi_leo", t_start_s=1.585,
                       duration_s=3.415, nominal_amp=12.0, l0=0.0, m0=0.0),
        SimulatedEvent(event_id="rfi", event_type="rfi_narrow", t_start_s=4.012,
                       duration_s=0.988, nominal_amp=7.0, l0=0.5, m0=0.5),
    ]


class TestEventFrameMembership(unittest.TestCase):
    def _check(self, dense_s):
        events = _events()
        idx, frame_events = build_frame_selection_schedule(
            TOTAL, FRAME_S, events, event_dense_s=dense_s)
        for ev in events:
            for k in idx:
                t = k * FRAME_S
                overlaps = ev.t_start_s - FRAME_S < t < ev.t_start_s + ev.duration_s
                self.assertEqual(ev.event_id in frame_events.get(k, []), overlaps,
                                 f"{ev.event_id} at frame {k} (t = {t:.3f} s)")

    def test_overlapping_events_dense_window(self):
        self._check(dense_s=1.0)

    def test_overlapping_events_sparse_window(self):
        self._check(dense_s=0.05)

    def test_selection_unchanged(self):
        idx, _ = build_frame_selection_schedule(TOTAL, FRAME_S, _events(), event_dense_s=0.05)
        self.assertEqual(len(idx), len(set(idx)))
        self.assertTrue(all(0 <= k < TOTAL for k in idx))


if __name__ == "__main__":
    unittest.main()
