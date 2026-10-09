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
    create_accumulate_yaml,
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
from .sky import (
    VerifiedTarget,
    catalog_transit_summary,
    direction_cosines_track,
    find_verified_target,
    geometric_phase_track,
    load_verified_catalog,
    resolve_window_start,
)
from .fengine import (
    FEngineConfig,
    measure_quantization_snr_loss,
    optimal_noise_sigma_lsb,
    pfb_channelize,
    quantize_int4x2,
    simulate_fengine_frame,
    sinc_hanning_window,
)
from .verify import (
    reference_beamform,
    reference_visibilities,
    verify_pointing,
    verify_quantization_snr_loss,
    verify_transit_lightcurve,
    verify_visibility_phasing,
)
from .visualizer import (
    plot_casm_correlation_matrix,
    plot_correlator_waterfalls,
    plot_tracker_waterfall,
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
    "create_accumulate_yaml",
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
    "VerifiedTarget",
    "load_verified_catalog",
    "find_verified_target",
    "direction_cosines_track",
    "geometric_phase_track",
    "resolve_window_start",
    "catalog_transit_summary",
    "FEngineConfig",
    "pfb_channelize",
    "quantize_int4x2",
    "simulate_fengine_frame",
    "measure_quantization_snr_loss",
    "optimal_noise_sigma_lsb",
    "sinc_hanning_window",
    "reference_visibilities",
    "reference_beamform",
    "verify_visibility_phasing",
    "verify_pointing",
    "verify_quantization_snr_loss",
    "verify_transit_lightcurve",
]
