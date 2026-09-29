#!/usr/bin/env python3
"""CHARTS Analog-Chain Thermal Noise Model Compatibility Layer.

Maintains backward-compatibility by re-exporting all noise models,
analog chain parameters, and solar ephemeris from `sim.noise_model`.
"""

from __future__ import annotations

from pathlib import Path
import sys

_test_charts_dir = Path(__file__).resolve().parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.noise_model import *
