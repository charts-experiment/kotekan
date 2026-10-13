#!/usr/bin/env python3
r"""CHARTS Baseband Serialization and File Writer Framework.

Extracts baseband frame packing and serialization from the simulation generator
into modular writer backends implementing the `BasebandWriter` protocol:
  - `RawBinWriter`: Kotekan rawFileWrite-compatible sequential .bin files with
    4-byte metadata header and packed int4x2 payloads.
  - `HDF5Writer`: Analysis-friendly HDF5 baseband container.

Theory & Literature Foundations:
  1. Buschmann, B. A. P. (2025). "Design and Implementation of the F-Engine
     for the CHARTS Project". Master's thesis / Technical Report, Section 4.
     Continuous mathematical formulation:
       q(x) = \text{clip}\left(\text{round}\left(\frac{x}{\Delta}\right), -7, 7\right)
       \text{Byte} = (\text{Re} \ \& \ 0x0F) \ | \ ((\text{Im} \ \& \ 0x0F) \ll 4)
  2. Smith, K. M. (2023). "Notes on CHORD F -> X Packet Format".
     Discrete 5D fiducial tensor indexing:
       (t_\text{pkt}, \text{dish}, \text{pol}, \text{freq}, t_\text{samp})
  3. CHIME Collaboration (2022). "CHIME / CHARTS Archive Data Format Specification".
     HDF5 metadata structure and telescope layout schema.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import h5py
import numpy as np

from .constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
    FPGA_TIME_RESOLUTION_US,
)
from .reference import ReferenceWindowManifest, compute_file_sha256

logger = logging.getLogger("kotekan.charts.sim.writer")

WindowManifest = ReferenceWindowManifest


# ---------------------------------------------------------------------------
# Metadata Container
# ---------------------------------------------------------------------------


@dataclass
class WindowMetadata:
    """Complete metadata description of a baseband observation window.

    Follows the CHIME/CHARTS archive schema for telescope geometry,
    analog receiver parameters, solar state, and scheduled transient events.
    """

    window_name: str
    target_dir: Path
    utc_start: str
    duration_s: float
    antennas: int
    num_freq: int
    samples_per_frame: int
    frame_duration_s: float
    frequencies_mhz: np.ndarray
    antenna_pos_x_m: np.ndarray
    antenna_pos_y_m: np.ndarray
    sun_elevation_deg: float = 0.0
    sun_is_up: bool = False
    total_physical_frames: int = 0
    num_written_frames: int = 0
    antenna_sigma_base: Optional[np.ndarray] = None
    bandpass_shape: Optional[np.ndarray] = None
    physical_frame_indices: Optional[np.ndarray] = None
    output_file_indices: Optional[np.ndarray] = None
    timestamps_s: Optional[np.ndarray] = None
    mean_powers_lsb2: Optional[np.ndarray] = None
    max_powers_lsb2: Optional[np.ndarray] = None
    clip_fractions: Optional[np.ndarray] = None
    lightcurve_time_s: Optional[np.ndarray] = None
    lightcurve_power: Optional[np.ndarray] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    active_sources: List[Dict[str, Any]] = field(default_factory=list)
    extra_attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        res = asdict(self)
        res["target_dir"] = str(self.target_dir)
        for key in [
            "frequencies_mhz",
            "antenna_pos_x_m",
            "antenna_pos_y_m",
            "antenna_sigma_base",
            "bandpass_shape",
            "physical_frame_indices",
            "output_file_indices",
            "timestamps_s",
            "mean_powers_lsb2",
            "max_powers_lsb2",
            "clip_fractions",
            "lightcurve_time_s",
            "lightcurve_power",
        ]:
            if res.get(key) is not None and isinstance(res[key], np.ndarray):
                res[key] = res[key].tolist()
        return res


# ---------------------------------------------------------------------------
# int4x2 Quantization and Packing Functions
# ---------------------------------------------------------------------------


def pack_int4x2(voltages: np.ndarray) -> np.ndarray:
    r"""Pack complex voltages into 4-bit complex integers stored in uint8.

    Continuous and Discrete Formulation:
      Given complex electric field voltage v = v_r + j v_i:
        r = \text{clip}\left(\text{round}(v_r), -7, 7\right) \in [-7, 7]
        i = \text{clip}\left(\text{round}(v_i), -7, 7\right) \in [-7, 7]
      Byte packing:
        \text{Byte} = (r \ \& \ 0x0F) \ | \ ((i \ \& \ 0x0F) \ll 4)

    References:
      - Buschmann (2025), "Design and Implementation of the F-Engine for CHARTS", §4.
      - Smith (2023), "Notes on CHORD F -> X Packet Format".

    Parameters
    ----------
    voltages : np.ndarray
        Complex array (float32, complex64, or int). If already uint8, returned as-is.

    Returns
    -------
    packed : np.ndarray
        uint8 array with shape matching the input voltage array.
    """
    if voltages.dtype == np.uint8:
        return voltages

    if np.iscomplexobj(voltages):
        v_real = voltages.real
        v_imag = voltages.imag
    else:
        # Assume trailing dimension 2 represents real and imag
        if voltages.shape[-1] == 2:
            v_real = voltages[..., 0]
            v_imag = voltages[..., 1]
        else:
            raise ValueError(
                f"Cannot interpret array with shape {voltages.shape} and dtype {voltages.dtype} as complex"
            )

    r_quant = np.clip(np.round(v_real), -8, 7).astype(np.int8)
    i_quant = np.clip(np.round(v_imag), -8, 7).astype(np.int8)

    r_nibble = (r_quant & 0x0F).astype(np.uint8)
    i_nibble = ((i_quant & 0x0F) << 4).astype(np.uint8)

    return (r_nibble | i_nibble).astype(np.uint8)


def unpack_int4x2(packed: np.ndarray) -> np.ndarray:
    """Unpack int4x2 uint8 array into complex64 values.

    Extracts signed 4-bit real (low nibble) and imaginary (high nibble) components
    using two's complement sign extension:
      r = (Byte & 0x0F), sign-extended from 4 to 8 bits
      i = ((Byte >> 4) & 0x0F), sign-extended from 4 to 8 bits
      v = r + 1j * i

    References:
      - Buschmann (2025), "Design and Implementation of the F-Engine for CHARTS", §4.
      - Smith (2023), "Notes on CHORD F -> X Packet Format".

    Parameters
    ----------
    packed : np.ndarray
        uint8 packed array.

    Returns
    -------
    unpacked : np.ndarray
        complex64 array with the same shape as `packed`.
    """
    u8 = np.asarray(packed, dtype=np.uint8)
    r = (u8 & 0x0F).astype(np.int8)
    r[r >= 8] -= 16

    i = ((u8 >> 4) & 0x0F).astype(np.int8)
    i[i >= 8] -= 16

    return r.astype(np.float32) + 1j * i.astype(np.float32)


# ---------------------------------------------------------------------------
# 5D Fiducial Layout Transforms
# ---------------------------------------------------------------------------


def fiducial_5d_to_kotekan(arr_5d: np.ndarray) -> np.ndarray:
    """Convert fiducial 5D ordering to Kotekan 3D frame layout.

    Fiducial 5D Ordering (AGENTS.md §2.2, Smith 2023):
      Shape: (t_pkt, dish, pol, freq, t_samp)
    Kotekan Streaming Buffer Layout:
      Shape: (t_samp, freq, num_elements) where num_elements = dish * pol.

    Parameters
    ----------
    arr_5d : np.ndarray
        Array with 5 dimensions (t_pkt, dish, pol, freq, t_samp) or
        4 dimensions (dish, pol, freq, t_samp) for a single packet.

    Returns
    -------
    arr_kotekan : np.ndarray
        Array with Kotekan layout (t_samp, freq, num_elements) or
        (t_pkt, t_samp, freq, num_elements).
    """
    if arr_5d.ndim == 4:
        # (dish, pol, freq, t_samp) -> (t_samp, freq, dish, pol) -> (t_samp, freq, num_elements)
        dish, pol, freq, t_samp = arr_5d.shape
        transposed = np.transpose(arr_5d, (3, 2, 0, 1))
        return transposed.reshape(t_samp, freq, dish * pol)
    elif arr_5d.ndim == 5:
        # (t_pkt, dish, pol, freq, t_samp) -> (t_pkt, t_samp, freq, dish, pol) -> (t_pkt, t_samp, freq, num_elements)
        t_pkt, dish, pol, freq, t_samp = arr_5d.shape
        transposed = np.transpose(arr_5d, (0, 4, 3, 1, 2))
        return transposed.reshape(t_pkt, t_samp, freq, dish * pol)
    else:
        raise ValueError(
            f"Expected 4D or 5D array for fiducial conversion, got {arr_5d.ndim}D"
        )


def kotekan_to_fiducial_5d(
    arr_kotekan: np.ndarray, num_dishes: int, num_pol: int = 1
) -> np.ndarray:
    """Convert Kotekan 3D frame layout to fiducial 5D ordering.

    Kotekan Streaming Buffer Layout:
      Shape: (t_samp, freq, num_elements) or (t_pkt, t_samp, freq, num_elements)
    Fiducial 5D Ordering:
      Shape: (dish, pol, freq, t_samp) or (t_pkt, dish, pol, freq, t_samp)

    Parameters
    ----------
    arr_kotekan : np.ndarray
        Array with Kotekan shape (t_samp, freq, num_elements) or
        (t_pkt, t_samp, freq, num_elements).
    num_dishes : int
        Number of physical antenna dishes.
    num_pol : int, default 1
        Number of polarizations per dish.

    Returns
    -------
    arr_5d : np.ndarray
        Array in fiducial order (dish, pol, freq, t_samp) or (t_pkt, dish, pol, freq, t_samp).
    """
    if arr_kotekan.ndim == 3:
        t_samp, freq, num_elements = arr_kotekan.shape
        if num_elements != num_dishes * num_pol:
            raise ValueError(
                f"num_elements ({num_elements}) != num_dishes ({num_dishes}) * num_pol ({num_pol})"
            )
        reshaped = arr_kotekan.reshape(t_samp, freq, num_dishes, num_pol)
        # Transpose to (dish, pol, freq, t_samp)
        return np.transpose(reshaped, (2, 3, 1, 0))
    elif arr_kotekan.ndim == 4:
        t_pkt, t_samp, freq, num_elements = arr_kotekan.shape
        if num_elements != num_dishes * num_pol:
            raise ValueError(
                f"num_elements ({num_elements}) != num_dishes ({num_dishes}) * num_pol ({num_pol})"
            )
        reshaped = arr_kotekan.reshape(t_pkt, t_samp, freq, num_dishes, num_pol)
        # Transpose to (t_pkt, dish, pol, freq, t_samp)
        return np.transpose(reshaped, (0, 3, 4, 2, 1))
    else:
        raise ValueError(
            f"Expected 3D or 4D array for kotekan layout, got {arr_kotekan.ndim}D"
        )


# ---------------------------------------------------------------------------
# BasebandWriter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BasebandWriter(Protocol):
    """Protocol for baseband voltage serialization and window manifest tracking.

    Generation code must depend only on this protocol, remaining completely
    decoupled from the physical storage representation (.bin vs .h5).
    """

    def write_frame(self, frame_id: int, voltages: np.ndarray) -> Path:
        """Serialize and write a single baseband voltage frame to disk.

        Parameters
        ----------
        frame_id : int
            Sequential output index of the frame (0-indexed).
        voltages : np.ndarray
            Baseband voltage data, either complex float (pre-quantization)
            or int4x2-packed uint8.

        Returns
        -------
        Path
            Path to the written file.
        """
        ...

    def write_metadata(self, meta: WindowMetadata) -> Path:
        """Write companion metadata, antenna positions, and event catalogs.

        Parameters
        ----------
        meta : WindowMetadata
            Complete observation window metadata container.

        Returns
        -------
        Path
            Path to the primary metadata file.
        """
        ...

    def finalize(self) -> WindowManifest:
        """Finalize the written window, compute checksums, and emit manifest.

        Returns
        -------
        WindowManifest
            Structured manifest with checksums, frame counts, and git metadata.
        """
        ...


# ---------------------------------------------------------------------------
# RawBinWriter Implementation
# ---------------------------------------------------------------------------


class RawBinWriter:
    """Kotekan rawFileWrite-compatible sequential .bin frame writer.

    Produces sequential binary files matching Kotekan's `rawFileRead` stage:
      - Header: 4-byte uint32 metadata size (= 0 for standard capture)
      - Payload: packed uint8 int4x2 frames of shape (samples_per_frame, num_freq, num_elements)
      - Filename format: <window_name>_<frame_id:07d>.bin
    """

    def __init__(
        self,
        target_dir: Union[Path, str],
        window_name: str,
        samples_per_frame: int = 1536,
        num_freq: int = 336,
        num_elements: int = 64,
        num_polarizations: int = 1,
    ):
        self.target_dir = Path(target_dir)
        self.window_name = window_name
        self.samples_per_frame = samples_per_frame
        self.num_freq = num_freq
        self.num_elements = num_elements
        self.num_polarizations = num_polarizations

        self.target_dir.mkdir(parents=True, exist_ok=True)
        self._written_frames: Dict[int, Path] = {}
        self._metadata: Optional[WindowMetadata] = None

    def write_frame(self, frame_id: int, voltages: np.ndarray) -> Path:
        """Packs and writes one sequential .bin frame.

        Parameters
        ----------
        frame_id : int
            Output frame index.
        voltages : np.ndarray
            Complex float or uint8 packed voltages.
        """
        # Convert fiducial 5D/4D if necessary
        if voltages.ndim in (4, 5):
            voltages = fiducial_5d_to_kotekan(voltages)

        packed = pack_int4x2(voltages)
        expected_shape = (self.samples_per_frame, self.num_freq, self.num_elements)
        if packed.shape != expected_shape:
            # Reshape if total byte size matches
            if (
                packed.size
                == self.samples_per_frame * self.num_freq * self.num_elements
            ):
                packed = packed.reshape(expected_shape)
            else:
                raise ValueError(
                    f"Frame size mismatch: expected {expected_shape} ({np.prod(expected_shape)} bytes), "
                    f"got {packed.shape} ({packed.size} bytes)"
                )

        out_path = self.target_dir / f"{self.window_name}_{frame_id:07d}.bin"
        with open(out_path, "wb") as f:
            # Kotekan rawFileRead expects 4-byte metadata_size header
            np.uint32(0).tofile(f)
            packed.tofile(f)

        self._written_frames[frame_id] = out_path
        return out_path

    def write_metadata(self, meta: WindowMetadata) -> Path:
        """Write companion HDF5 metadata and events JSON."""
        self._metadata = meta
        meta_h5_path = self.target_dir / f"{self.window_name}_meta.h5"
        events_json_path = self.target_dir / f"{self.window_name}_events.json"

        # 1. Events JSON
        with open(events_json_path, "w", encoding="utf-8") as f:
            json.dump(meta.events, f, indent=2)

        # 2. Companion events.json at root of window dir for contract parity
        root_events_path = self.target_dir / "events.json"
        if not root_events_path.exists():
            with open(root_events_path, "w", encoding="utf-8") as f:
                json.dump(meta.events, f, indent=2)

        # 3. Companion HDF5 metadata
        with h5py.File(meta_h5_path, "w") as h5:
            h5.attrs["utc_start"] = meta.utc_start
            h5.attrs["duration_s"] = meta.duration_s
            h5.attrs["antennas"] = meta.antennas
            h5.attrs["num_freq"] = meta.num_freq
            h5.attrs["samples_per_frame"] = meta.samples_per_frame
            h5.attrs["frame_duration_s"] = meta.frame_duration_s
            h5.attrs["sun_elevation_deg"] = meta.sun_elevation_deg
            h5.attrs["sun_is_up"] = meta.sun_is_up
            h5.attrs["total_physical_frames"] = meta.total_physical_frames
            h5.attrs["num_written_frames"] = meta.num_written_frames
            for k, v in meta.extra_attrs.items():
                h5.attrs[k] = v

            h5.create_dataset(
                "frequencies_mhz",
                data=np.asarray(meta.frequencies_mhz, dtype=np.float64),
            )
            h5.create_dataset(
                "antenna_pos_x_m",
                data=np.asarray(meta.antenna_pos_x_m, dtype=np.float64),
            )
            h5.create_dataset(
                "antenna_pos_y_m",
                data=np.asarray(meta.antenna_pos_y_m, dtype=np.float64),
            )

            if meta.antenna_sigma_base is not None:
                h5.create_dataset(
                    "antenna_sigma_base",
                    data=np.asarray(meta.antenna_sigma_base, dtype=np.float32),
                )
            if meta.bandpass_shape is not None:
                h5.create_dataset(
                    "bandpass_shape",
                    data=np.asarray(meta.bandpass_shape, dtype=np.float32),
                )

            frames_grp = h5.create_group("frames")
            if meta.physical_frame_indices is not None:
                frames_grp.create_dataset(
                    "physical_frame_index",
                    data=np.asarray(meta.physical_frame_indices, dtype=np.int32),
                )
            if meta.output_file_indices is not None:
                frames_grp.create_dataset(
                    "output_file_index",
                    data=np.asarray(meta.output_file_indices, dtype=np.int32),
                )
            if meta.timestamps_s is not None:
                frames_grp.create_dataset(
                    "timestamp_s", data=np.asarray(meta.timestamps_s, dtype=np.float64)
                )
            if meta.mean_powers_lsb2 is not None:
                frames_grp.create_dataset(
                    "mean_power_lsb2",
                    data=np.asarray(meta.mean_powers_lsb2, dtype=np.float32),
                )
            if meta.max_powers_lsb2 is not None:
                frames_grp.create_dataset(
                    "max_power_lsb2",
                    data=np.asarray(meta.max_powers_lsb2, dtype=np.float32),
                )
            if meta.clip_fractions is not None:
                frames_grp.create_dataset(
                    "clip_fraction",
                    data=np.asarray(meta.clip_fractions, dtype=np.float32),
                )

            if meta.lightcurve_time_s is not None and meta.lightcurve_power is not None:
                lc_grp = h5.create_group("lightcurve")
                lc_grp.create_dataset(
                    "time_s", data=np.asarray(meta.lightcurve_time_s, dtype=np.float64)
                )
                lc_grp.create_dataset(
                    "power_analytic",
                    data=np.asarray(meta.lightcurve_power, dtype=np.float64),
                )

        return meta_h5_path

    def finalize(self) -> WindowManifest:
        """Scan written .bin frames, compute SHA-256 prefixes, and emit window_manifest.json."""
        bin_files = sorted(self.target_dir.glob("*.bin"))
        frame_entries = []
        total_size = 0

        for bf in bin_files:
            size = bf.stat().st_size
            total_size += size
            frame_entries.append(
                {
                    "name": bf.name,
                    "size_bytes": size,
                    "sha256_prefix": compute_file_sha256(bf),
                }
            )

        frame_count = len(bin_files)
        frame_period_ms = (self.samples_per_frame * FPGA_TIME_RESOLUTION_US) / 1000.0

        obs_start_utc = (
            self._metadata.utc_start
            if self._metadata
            else datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        duration_s = (
            self._metadata.duration_s
            if self._metadata
            else (frame_count * frame_period_ms * 1e-3)
        )
        freq_start_mhz = (
            float(self._metadata.frequencies_mhz[0])
            if (self._metadata and len(self._metadata.frequencies_mhz) > 0)
            else DEFAULT_FREQUENCY_START_MHZ
        )

        manifest = ReferenceWindowManifest(
            tag=self.window_name,
            description=f"Baseband observation window: {self.window_name}",
            created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            observation_start_utc=obs_start_utc,
            observation_start_lst_hours=0.0,
            duration_s=duration_s,
            total_frames=frame_count,
            samples_per_frame=self.samples_per_frame,
            frame_period_ms=frame_period_ms,
            antennas=self.num_elements,
            num_freq=self.num_freq,
            frequency_start_mhz=freq_start_mhz,
            channel_width_mhz=CHARTS_CHANNEL_WIDTH_MHZ,
            active_sources=self._metadata.active_sources if self._metadata else [],
            injected_events=self._metadata.events if self._metadata else [],
            frame_files=frame_entries,
            total_size_bytes=total_size,
            total_size_mb=round(total_size / (1024.0 * 1024.0), 3),
        )

        manifest_path = self.target_dir / "window_manifest.json"
        manifest.save(manifest_path)
        return manifest


# ---------------------------------------------------------------------------
# HDF5Writer Implementation
# ---------------------------------------------------------------------------


class HDF5Writer:
    """Analysis-friendly HDF5 baseband container writer.

    Stores sequential baseband frames inside an HDF5 dataset `baseband`
    along with accompanying metadata datasets and attributes.
    """

    def __init__(
        self,
        target_dir: Union[Path, str],
        window_name: str,
        samples_per_frame: int = 1536,
        num_freq: int = 336,
        num_elements: int = 64,
        num_polarizations: int = 1,
        h5_path: Optional[Union[Path, str]] = None,
    ):
        self.target_dir = Path(target_dir)
        self.window_name = window_name
        self.samples_per_frame = samples_per_frame
        self.num_freq = num_freq
        self.num_elements = num_elements
        self.num_polarizations = num_polarizations

        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = (
            Path(h5_path) if h5_path else (self.target_dir / f"{self.window_name}.h5")
        )
        self._written_count = 0
        self._metadata: Optional[WindowMetadata] = None

    def write_frame(self, frame_id: int, voltages: np.ndarray) -> Path:
        """Pack and store baseband frame in HDF5 dataset `baseband`."""
        if voltages.ndim in (4, 5):
            voltages = fiducial_5d_to_kotekan(voltages)

        packed = pack_int4x2(voltages)
        expected_shape = (self.samples_per_frame, self.num_freq, self.num_elements)
        if packed.shape != expected_shape:
            if (
                packed.size
                == self.samples_per_frame * self.num_freq * self.num_elements
            ):
                packed = packed.reshape(expected_shape)
            else:
                raise ValueError(
                    f"Frame size mismatch: expected {expected_shape}, got {packed.shape}"
                )

        with h5py.File(self.h5_path, "a") as h5:
            if "baseband" not in h5:
                # Shape: (frames, samples_per_frame, num_freq, num_elements)
                dset = h5.create_dataset(
                    "baseband",
                    shape=(0, self.samples_per_frame, self.num_freq, self.num_elements),
                    maxshape=(
                        None,
                        self.samples_per_frame,
                        self.num_freq,
                        self.num_elements,
                    ),
                    dtype=np.uint8,
                    chunks=(
                        1,
                        min(self.samples_per_frame, 256),
                        min(self.num_freq, 64),
                        self.num_elements,
                    ),
                )
                dset.attrs["dimension_labels"] = ["frame", "t_samp", "freq", "element"]
                dset.attrs["encoding"] = "int4x2_packed"
                dset.attrs["packing_formula"] = (
                    "Byte = (Re & 0x0F) | ((Im & 0x0F) << 4)"
                )
            else:
                dset = h5["baseband"]

            target_idx = frame_id
            if target_idx >= dset.shape[0]:
                dset.resize(target_idx + 1, axis=0)

            dset[target_idx] = packed
            self._written_count = max(self._written_count, target_idx + 1)

        return self.h5_path

    def write_metadata(self, meta: WindowMetadata) -> Path:
        """Write metadata to HDF5 file and sidecars."""
        self._metadata = meta
        events_json_path = self.target_dir / f"{self.window_name}_events.json"
        with open(events_json_path, "w", encoding="utf-8") as f:
            json.dump(meta.events, f, indent=2)

        root_events_path = self.target_dir / "events.json"
        if not root_events_path.exists():
            with open(root_events_path, "w", encoding="utf-8") as f:
                json.dump(meta.events, f, indent=2)

        # Also emit standard _meta.h5 sidecar for tooling parity
        meta_h5_path = self.target_dir / f"{self.window_name}_meta.h5"
        with h5py.File(meta_h5_path, "w") as h5:
            h5.attrs["utc_start"] = meta.utc_start
            h5.attrs["duration_s"] = meta.duration_s
            h5.attrs["antennas"] = meta.antennas
            h5.attrs["num_freq"] = meta.num_freq
            h5.attrs["samples_per_frame"] = meta.samples_per_frame
            h5.attrs["frame_duration_s"] = meta.frame_duration_s
            h5.attrs["sun_elevation_deg"] = meta.sun_elevation_deg
            h5.attrs["sun_is_up"] = meta.sun_is_up
            h5.attrs["total_physical_frames"] = meta.total_physical_frames
            h5.attrs["num_written_frames"] = meta.num_written_frames
            for k, v in meta.extra_attrs.items():
                h5.attrs[k] = v

            h5.create_dataset(
                "frequencies_mhz",
                data=np.asarray(meta.frequencies_mhz, dtype=np.float64),
            )
            h5.create_dataset(
                "antenna_pos_x_m",
                data=np.asarray(meta.antenna_pos_x_m, dtype=np.float64),
            )
            h5.create_dataset(
                "antenna_pos_y_m",
                data=np.asarray(meta.antenna_pos_y_m, dtype=np.float64),
            )

            if meta.antenna_sigma_base is not None:
                h5.create_dataset(
                    "antenna_sigma_base",
                    data=np.asarray(meta.antenna_sigma_base, dtype=np.float32),
                )
            if meta.bandpass_shape is not None:
                h5.create_dataset(
                    "bandpass_shape",
                    data=np.asarray(meta.bandpass_shape, dtype=np.float32),
                )

            frames_grp = h5.create_group("frames")
            if meta.physical_frame_indices is not None:
                frames_grp.create_dataset(
                    "physical_frame_index",
                    data=np.asarray(meta.physical_frame_indices, dtype=np.int32),
                )
            if meta.output_file_indices is not None:
                frames_grp.create_dataset(
                    "output_file_index",
                    data=np.asarray(meta.output_file_indices, dtype=np.int32),
                )
            if meta.timestamps_s is not None:
                frames_grp.create_dataset(
                    "timestamp_s", data=np.asarray(meta.timestamps_s, dtype=np.float64)
                )
            if meta.mean_powers_lsb2 is not None:
                frames_grp.create_dataset(
                    "mean_power_lsb2",
                    data=np.asarray(meta.mean_powers_lsb2, dtype=np.float32),
                )
            if meta.max_powers_lsb2 is not None:
                frames_grp.create_dataset(
                    "max_power_lsb2",
                    data=np.asarray(meta.max_powers_lsb2, dtype=np.float32),
                )
            if meta.clip_fractions is not None:
                frames_grp.create_dataset(
                    "clip_fraction",
                    data=np.asarray(meta.clip_fractions, dtype=np.float32),
                )

            if meta.lightcurve_time_s is not None and meta.lightcurve_power is not None:
                lc_grp = h5.create_group("lightcurve")
                lc_grp.create_dataset(
                    "time_s", data=np.asarray(meta.lightcurve_time_s, dtype=np.float64)
                )
                lc_grp.create_dataset(
                    "power_analytic",
                    data=np.asarray(meta.lightcurve_power, dtype=np.float64),
                )

        return meta_h5_path

    def finalize(self) -> WindowManifest:
        """Emit window manifest for HDF5 container."""
        frame_period_ms = (self.samples_per_frame * FPGA_TIME_RESOLUTION_US) / 1000.0
        size = self.h5_path.stat().st_size if self.h5_path.exists() else 0
        sha_pref = compute_file_sha256(self.h5_path) if self.h5_path.exists() else ""

        frame_entries = [
            {
                "name": self.h5_path.name,
                "size_bytes": size,
                "sha256_prefix": sha_pref,
            }
        ]

        obs_start_utc = (
            self._metadata.utc_start
            if self._metadata
            else datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        duration_s = (
            self._metadata.duration_s
            if self._metadata
            else (self._written_count * frame_period_ms * 1e-3)
        )
        freq_start_mhz = (
            float(self._metadata.frequencies_mhz[0])
            if (self._metadata and len(self._metadata.frequencies_mhz) > 0)
            else DEFAULT_FREQUENCY_START_MHZ
        )

        manifest = ReferenceWindowManifest(
            tag=self.window_name,
            description=f"HDF5 baseband observation window: {self.window_name}",
            created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            observation_start_utc=obs_start_utc,
            observation_start_lst_hours=0.0,
            duration_s=duration_s,
            total_frames=self._written_count,
            samples_per_frame=self.samples_per_frame,
            frame_period_ms=frame_period_ms,
            antennas=self.num_elements,
            num_freq=self.num_freq,
            frequency_start_mhz=freq_start_mhz,
            channel_width_mhz=CHARTS_CHANNEL_WIDTH_MHZ,
            active_sources=self._metadata.active_sources if self._metadata else [],
            injected_events=self._metadata.events if self._metadata else [],
            frame_files=frame_entries,
            total_size_bytes=size,
            total_size_mb=round(size / (1024.0 * 1024.0), 3),
        )

        manifest_path = self.target_dir / "window_manifest.json"
        manifest.save(manifest_path)
        return manifest


# ---------------------------------------------------------------------------
# Reading Utility Functions
# ---------------------------------------------------------------------------


def read_raw_bin_frame(
    file_path: Union[Path, str],
    shape: Optional[Tuple[int, int, int]] = None,
    unpack: bool = True,
) -> np.ndarray:
    """Read a Kotekan rawFileWrite binary frame.

    Parses the 4-byte metadata_size header, skips any metadata payload,
    and returns the frame payload either as packed uint8 or unpacked complex64.

    Parameters
    ----------
    file_path : Path | str
        Path to .bin file.
    shape : tuple of (samples_per_frame, num_freq, num_ant), optional
        If provided, reshapes the returned array.
    unpack : bool, default True
        If True, unpacks int4x2 bytes to complex64. If False, returns uint8 bytes.

    Returns
    -------
    np.ndarray
        Frame data (complex64 or uint8).
    """
    path = Path(file_path)
    with open(path, "rb") as f:
        meta_size_arr = np.fromfile(f, dtype=np.uint32, count=1)
        if len(meta_size_arr) == 0:
            raise EOFError(f"File {path} is empty")
        meta_size = int(meta_size_arr[0])
        if meta_size > 0:
            f.seek(meta_size, 1)  # Skip metadata bytes

        packed = np.fromfile(f, dtype=np.uint8)

    if shape is not None:
        packed = packed.reshape(shape)

    if unpack:
        return unpack_int4x2(packed)
    return packed


# ---------------------------------------------------------------------------
# Factory Helper
# ---------------------------------------------------------------------------


def create_writer(
    writer_type: str,
    target_dir: Union[Path, str],
    window_name: str,
    samples_per_frame: int = 1536,
    num_freq: int = 336,
    num_elements: int = 64,
    num_polarizations: int = 1,
    **kwargs,
) -> BasebandWriter:
    """Factory creating a BasebandWriter instance.

    Supported types: 'raw_bin', 'bin', 'raw', 'hdf5', 'h5'.
    """
    w_type = writer_type.lower().strip()
    if w_type in ("raw_bin", "bin", "raw"):
        return RawBinWriter(
            target_dir=target_dir,
            window_name=window_name,
            samples_per_frame=samples_per_frame,
            num_freq=num_freq,
            num_elements=num_elements,
            num_polarizations=num_polarizations,
        )
    elif w_type in ("hdf5", "h5"):
        return HDF5Writer(
            target_dir=target_dir,
            window_name=window_name,
            samples_per_frame=samples_per_frame,
            num_freq=num_freq,
            num_elements=num_elements,
            num_polarizations=num_polarizations,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown writer type: {writer_type}. Supported: 'raw_bin', 'hdf5'"
        )
