#!/usr/bin/env python3
"""CHARTS Constants Compatibility Layer for Kotekan test_charts.

Maintains backward-compatibility by re-exporting all physical, instrumental,
and observatory constants from the centralized `sim.constants` module.
"""

from __future__ import annotations

from pathlib import Path
import sys

_test_charts_dir = Path(__file__).resolve().parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.constants import *
from sim.constants import (
    _external_module,
    _get_external,
    _has_external,
    _source_description,
)
