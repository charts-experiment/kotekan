
# Documentation (CHARTS)

# Contributing

If you are contributing, or even just a kotekan user, please have a look at the documentation! You can find an [overview of kotekan](https://kotekan.readthedocs.io/latest/overview.html) and [theory of operation](https://kotekan.readthedocs.io/latest/overview_theory_of_operation.html) in the documentation, which is available at [kotekan.readthedocs.io](https://kotekan.readthedocs.io/).

[![Documentation Status](https://app.readthedocs.org/projects/kotekan/badge/)](https://kotekan.readthedocs.io/)

Please send changes that meet an operational need, not refactoring or reformatting on its own. Before asking for human review, it helps to ask an LLM for an adversarial review of the pull request. Write PR descriptions for human reviewers, in a few sentences: what changed, why, and how it was tested. An AGENTS.md file in this repository supports these goals.

# Repository layout

- `kotekan/` - the `kotekan` executable entry point.
- `lib/core/` - framework: `Stage` base class and factory, `buffer`/frame management, `ringbuffer`, `Config`, REST server, logging, metrics.
- `lib/stages/` - CPU stages (network I/O, file writers, N2 processing, RFI, beamforming support).
- `lib/metadata/` - per-frame metadata types and frame descriptors (`chordMetadata`, `N2FrameDesc`, `N2Layout`, `NDArray`).
- `lib/utils/` - shared helpers: telescope definitions, dataset manager and states, frame views, file formats, time utilities.
- `lib/gpu/`, `lib/cuda/`, `lib/hip/`, `lib/opencl/` - GPU framework and backend-specific commands. `lib/cuda/generated/` holds kernels produced by `julia/`; regenerate them there rather than editing by hand.
- `lib/dpdk/` - DPDK packet capture stages.
- `lib/testing/` - synthetic data and checking stages used by test configs (built with `-DWITH_TESTS=ON`).
- `julia/` - Julia CUDA kernel generator (see its README).
- `config/` - pipeline configs. Top level and `fengine/` hold production and telescope configs (`.yaml`, or `.j2` Jinja templates); `ci-tests/` holds the configs run by CI; `examples/` holds minimal starters.
- `python/kotekan/` - Python helpers for reading and running kotekan buffers and configs; used by the pytests.
- `tests/` - pytests, `boost/` unit tests, and `ci-scripts/` standalone shell tests.
- `docs/sphinx/`, `docs/doxygen/` - user and developer guides, API reference.
- `tools/` - lint scripts, docker images, debugging helpers.
- `external/` - vendored dependencies.

# Glossary

- **Stage** - a pipeline unit with its own thread, configured under `kotekan_stage` in a yaml config. Consumes and produces buffers. Older docs call these "processes".
- **Buffer / frame** - a buffer is a fixed set of equal-size frames shared between stages; a frame is the unit a producer fills and a consumer releases. Each frame carries a metadata object drawn from a **metadata pool**.
- **Ring buffer** - a byte-addressed circular buffer with a cursor, used between GPU stages instead of framed buffers.
- **Element / input** - one correlator input, that is one dish-polarization pair. `num_elements` counts inputs. Labels are the dish label followed by the polarization letter, for example `A01X` and `A01Y`.
- **Dish / feed** - the physical antenna. A dish has two inputs, one per polarization.
- **Product** - one visibility, a pair of inputs. Autocorrelations are the products of an input with itself.
- **N2 layout** (also visibility layout) - the arrangement of products within an N2 frame, described by `N2Layout`. Use this term rather than "frame order".
- **Coarse frequency** - one FPGA channel. Upchannelization splits it into finer channels.
- **Metadata** - per-frame header (timestamps, sequence numbers, frequency, dataset id). Telescope-specific types live in `lib/metadata/`.

# Build Instructions

| `develop` |
|------|
| [![kotekan-ci-tests](https://github.com/kotekan/kotekan/actions/workflows/main.yaml/badge.svg?branch=develop)](https://github.com/kotekan/kotekan/actions/workflows/main.yaml) |


Detailed instructions are available at https://kotekan.readthedocs.io/latest/compiling/general.html

Full list of CMake options: https://kotekan.readthedocs.io/latest/compiling/cmake_options.html

This project is built using cmake, so you will need to install cmake
before starting a build.

To build just the base framework:

	cd build
	cmake <options> ..
	make

Building minimun kotekan for CHARTS

Cmake build options (defaults shown in parentheses; most feature toggles accept `AUTO`, `ON`, or `OFF`, with `AUTO` probing for dependencies and falling back gracefully):

* `-DUSE_CUDA=<AUTO|ON|OFF>` (`AUTO`) - Build the CUDA backend and enable CUDA stages when `nvcc` and the CUDA toolkit are available. Adds `-DWITH_CUDA` on success.

* `-DUSE_DPDK=<AUTO|ON|OFF>` (`AUTO`) - Enable DPDK stages when `libdpdk>=19.11` is present via `pkg-config`.

* `-DUSE_HDF5=<AUTO|ON|OFF>` (`AUTO`) - Enable HDF5 output stages when HDF5, HighFive, and the runtime plugin directory are *all* available. Populates `KOTEKAN_HDF5_PLUGIN_DIR` for runtime use.

* `-DUSE_NUMA=<AUTO|ON|OFF>` (`ON`) - Link libnuma and enable NUMA-aware buffer handling. Required when DPDK is enabled.

* `-DWITH_TESTS=<AUTO|ON|OFF>` (`OFF`) - Build and link the helper stages from `lib/testing` into the kotekan binary (used by QA/example configs). Does not build unit tests.

**Examples (implemented on CHARTS):**

    cmake -DUSE_CUDA=ON -DUSE_DPDK=ON -DUSE_HDF5=ON -DUSE_NUMA=ON -DWITH_TESTS=ON ..

At the end of configuration, CMake prints a colorized feature summary indicating which features were enabled (found) or disabled (missing/explicitly off). Each feature row shows its toggle flag, e.g. `CUDA: ON (found, toggle: -DUSE_CUDA=ON/OFF)`. Use `-D<OPTION>=AUTO|ON|OFF` to auto-detect, require, or disable a feature present on your system.

To install kotekan:

	make install

# Running kotekan

**Using systemd (full install)**

To start kotekan

    sudo systemctl start kotekan

To stop kotekan

    sudo systemctl stop kotekan

**To run in debug mode, run from `ch_gpu/build/kotekan/`**

    sudo ./kotekan -c <config_file>.yaml

For example:

    sudo ./kotekan -c ../../kotekan/kotekan_gpu_replay.yaml

When installed kotekan's config files are located at /etc/kotekan/

**This read me is based on the official Kotekan Readme