"""CHARTS Simulation and Verification Suite for Kotekan.

A unified framework for simulating, executing, inspecting, and visualizing
CHARTS baseband data, tensor-core correlations, and beam tracking.
"""

from .constants import (
    C_LIGHT,
    CHARTS_CHANNEL_WIDTH_HZ,
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    CHARTS_N_ANTENNAS,
    CHARTS_N_FREQ,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    FPGA_TIME_RESOLUTION_US,
    K_DM,
    LOCAL_FREQUENCY_CHANNELS,
    SPEED_OF_LIGHT,
    get_antenna_positions,
)
from .generator import (
    SimulatedEvent,
    build_frame_selection_schedule,
    compute_analytic_lightcurve,
    generate_simulation_window,
    schedule_random_events,
)
from .inspector import (
    inspect_correlator_matrix,
    inspect_tracker_dump,
    load_astron_correlator_dump,
)
from .noise_model import (
    AnalogChainParams,
    ChartsNoiseModel,
)
from .pipeline import (
    create_beam_tracker_yaml,
    create_correlator_yaml,
    execute_kotekan,
    parse_beam_targets,
)
from .presets import (
    SimulationConfig,
    find_kotekan_binary,
    get_default_scratch_dir,
    get_default_workers,
    get_preset_config,
)
from .visualizer import (
    plot_casm_correlation_matrix,
    plot_correlator_waterfalls,
    plot_tracker_waterfall,
)
from .writer import (
    BasebandWriter,
    HDF5Writer,
    RawBinWriter,
    WindowMetadata,
    create_writer,
    fiducial_5d_to_kotekan,
    kotekan_to_fiducial_5d,
    pack_int4x2,
    read_raw_bin_frame,
    unpack_int4x2,
)

__all__ = [
    "C_LIGHT",
    "SPEED_OF_LIGHT",
    "K_DM",
    "CHARTS_CHANNEL_WIDTH_HZ",
    "CHARTS_CHANNEL_WIDTH_MHZ",
    "CHARTS_LATITUDE_DEG",
    "CHARTS_LONGITUDE_DEG",
    "CHARTS_N_ANTENNAS",
    "CHARTS_N_FREQ",
    "DEFAULT_FREQUENCY_START_MHZ",
    "DEFAULT_SPACING_M",
    "FPGA_TIME_RESOLUTION_US",
    "LOCAL_FREQUENCY_CHANNELS",
    "get_antenna_positions",
    "AnalogChainParams",
    "ChartsNoiseModel",
    "SimulatedEvent",
    "generate_simulation_window",
    "schedule_random_events",
    "build_frame_selection_schedule",
    "compute_analytic_lightcurve",
    "create_correlator_yaml",
    "create_beam_tracker_yaml",
    "execute_kotekan",
    "parse_beam_targets",
    "load_astron_correlator_dump",
    "inspect_correlator_matrix",
    "inspect_tracker_dump",
    "plot_casm_correlation_matrix",
    "plot_correlator_waterfalls",
    "plot_tracker_waterfall",
    "SimulationConfig",
    "get_preset_config",
    "find_kotekan_binary",
    "get_default_scratch_dir",
    "get_default_workers",
    "BasebandWriter",
    "RawBinWriter",
    "HDF5Writer",
    "WindowMetadata",
    "pack_int4x2",
    "unpack_int4x2",
    "read_raw_bin_frame",
    "create_writer",
    "fiducial_5d_to_kotekan",
    "kotekan_to_fiducial_5d",
]
