#!/usr/bin/env python3
r"""CHARTS F-Engine Reference Simulation (Julia parity).

NumPy port of the gold-standard F-Engine reference
`RadioTelescopeFEngine.jl` (src/RadioTelescopeFEngine.jl), implementing the
real CHARTS signal chain:

  real ADC stream -> 4-tap sinc-Hanning PFB -> channel selection -> 4-bit
  complex quantization -> int4x2 packing

This module is the *reference* model: it is slower than the direct channel
synthesis used by `sim.generator` (the "fast path") but physically faithful.
It is used to

  1. validate the fast path (channel noise statistics, quantizer operating
     point, tone response), and
  2. generate small gold-standard windows for X-engine equivalence tests.

Continuous Mathematical Model
----------------------------
PFB window (sinc-Hanning, eq. (11) of Shaw's PFB notes, with `N = U + 1`;
the Erik convention correct in the limit `M -> 1`, `U -> 1`):

.. math::

    s' = \frac{2s - (M U - 1)}{2 M U} \in (-\tfrac{1}{2}, \tfrac{1}{2}),
    \qquad
    w(s) = \cos^2(\pi s')\,\mathrm{sinc}(M s')

Channelization of a real ADC stream $x_n$ (Julia `channelize!`):

.. math::

    X_k^{(m)} = \frac{1}{U/2} \sum_{n=0}^{M U - 1} w_n\,
        x_{m U + n}\, e^{-2\pi i k n / (M U)},
    \qquad k = M \cdot f_{\mathrm{id}}

Quantization (Julia `quantize!`):

.. math::

    q = \mathrm{round}\left(\mathrm{clamp}(7.5\,X,\ -7,\ +7)\right)

Theory & Literature:
  - Shaw, R. (PFB notes) via `RadioTelescopeFEngine.jl`, `sinc_hanning` (eq. 11).
  - Buschmann, B. A. P. (2025). "Design and Implementation of the F-Engine for
    CHARTS", §4 (PFB design and 4-bit quantization).
  - Thompson, Moran & Swenson (2017), §8 (filter-bank channelizers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .constants import (
    ADC_SAMPLING_FREQ_HZ,
    C_LIGHT,
    FPGA_NUM_SAMP_FFT,
    LOCAL_FREQUENCY_CHANNELS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FEngineConfig:
    """CHARTS F-Engine parameters (RFSoC PFB channelizer).

    Defaults follow `charts-constants` (ADC 2457.6 MHz, 8192-point FFT ->
    300 kHz channels) and the Julia reference (4 taps, quantizer scale 7.5,
    channels 1000..1335 -> 300.0 - 400.8 MHz).
    """

    adc_rate_hz: float = ADC_SAMPLING_FREQ_HZ
    ntaps: int = 4
    nsamples: int = FPGA_NUM_SAMP_FFT
    freq_ids: np.ndarray = field(
        default_factory=lambda: np.arange(1000, 1000 + LOCAL_FREQUENCY_CHANNELS)
    )
    quant_scale: float = 7.5

    def __post_init__(self) -> None:
        self.freq_ids = np.asarray(self.freq_ids, dtype=np.int64)

    @property
    def channel_width_hz(self) -> float:
        return self.adc_rate_hz / self.nsamples

    @property
    def freqs_hz(self) -> np.ndarray:
        return self.freq_ids.astype(np.float64) * self.channel_width_hz

    @property
    def adc_dt_s(self) -> float:
        return 1.0 / self.adc_rate_hz


@dataclass(frozen=True)
class ToneSource:
    """A monochromatic point source on the sky (Julia `MonochromaticSource`).

    The source direction is given by topocentric direction cosines (l, m);
    the geometric delay at antenna position (x, y) is
    tau = (x*l + y*m) / c (Julia `calc_delay` with sin(angle) == direction
    cosine for small angles; here the exact cosine form is used).
    """

    freq_hz: float
    amp: float
    l: float
    m: float


# ---------------------------------------------------------------------------
# PFB primitives (Julia parity)
# ---------------------------------------------------------------------------


def sinc_hanning_window(ntaps: int, nsamples: int) -> np.ndarray:
    r"""sinc-Hanning PFB window, eq. (11), Erik convention (Julia `sinc_hanning`).

    .. math:: w(s) = \cos^2(\pi s')\,\mathrm{sinc}(M s'), \quad
        s' = \frac{2s - (MU - 1)}{2 MU}, \quad s = 0..MU-1
    """
    s = np.arange(ntaps * nsamples, dtype=np.float64)
    s_prime = (2.0 * s - (ntaps * nsamples - 1)) / (2.0 * ntaps * nsamples)
    x = ntaps * s_prime
    sinc_term = np.sinc(x)  # numpy sinc(x) = sin(pi x)/(pi x) == sinc1(x)
    return np.cos(np.pi * s_prime) ** 2 * sinc_term


def pfb_channelize(
    adc_stream: np.ndarray,
    config: Optional[FEngineConfig] = None,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Channelizes a real ADC stream through the ntaps-point PFB.

    Faithful port of Julia `channelize!`: consecutive overlapping blocks of
    `ntaps * nsamples` ADC samples, weighted by `w / (nsamples / 2)`, passed
    through a real FFT, keeping every `ntaps`-th bin at the configured channel
    ids. Block `m` starts at ADC sample `m * nsamples`, so the output cadence
    is one spectrum per `nsamples` ADC samples (= 1 / 300 kHz = 10/3 us).

    Parameters
    ----------
    adc_stream:
        Real ADC samples, shape (n_samples,) or (n_samples, n_ant).
    config:
        FEngineConfig; default CHARTS parameters.
    window:
        Precomputed sinc-Hanning window (optional).

    Returns
    -------
    channels:
        Complex128 array of shape (n_spectra, n_channels) or
        (n_spectra, n_channels, n_ant).
    """
    cfg = config or FEngineConfig()
    win = window if window is not None else sinc_hanning_window(cfg.ntaps, cfg.nsamples)
    win = win / (cfg.nsamples / 2.0)

    x = np.asarray(adc_stream, dtype=np.float64)
    multi_ant = x.ndim == 2
    if not multi_ant:
        x = x[:, None]

    n_samples, n_ant = x.shape
    block = cfg.ntaps * cfg.nsamples
    if n_samples < block:
        raise ValueError(
            f"ADC stream too short: {n_samples} samples < one PFB window ({block})"
        )
    n_spectra = n_samples // cfg.nsamples - (cfg.ntaps - 1)
    if n_spectra <= 0:
        raise ValueError("ADC stream does not contain a full PFB window")

    # Build sliding blocks: (n_spectra, block, n_ant)
    starts = (np.arange(n_spectra) * cfg.nsamples)[:, None] + np.arange(block)[None, :]
    blocks = x[starts] * win[None, :, None]

    spectra = np.fft.rfft(blocks, n=block, axis=1)  # (n_spectra, block//2+1, n_ant)
    channels = spectra[:, cfg.ntaps * cfg.freq_ids, :]  # (n_spectra, n_ch, n_ant)

    if not multi_ant:
        return channels[:, :, 0]
    return channels


def quantize_int4x2(
    channels: np.ndarray,
    scale: float = 7.5,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Quantizes complex channel voltages to int4x2 (Julia `quantize!`).

    .. math:: q = \mathrm{round}(\mathrm{clamp}(\mathrm{scale} \cdot X, -7, +7))

    Parameters
    ----------
    channels:
        Complex channel voltages (arbitrary shape).
    scale:
        Digitizer full-scale mapping (Julia convention: 7.5).

    Returns
    -------
    (packed, quantized):
        ``packed`` is uint8 with Byte = (Re & 0x0F) | ((Im & 0x0F) << 4);
        ``quantized`` is the complex-valued quantized voltage (for SNR analysis).
    """
    x = np.asarray(channels)
    re_q = np.clip(np.round(scale * x.real), -7, 7).astype(np.int8)
    im_q = np.clip(np.round(scale * x.imag), -7, 7).astype(np.int8)

    packed = (re_q & 0x0F).astype(np.uint8) | ((im_q & 0x0F).astype(np.uint8) << 4)
    quantized = re_q.astype(np.float64) + 1j * im_q.astype(np.float64)
    return packed, quantized


# ---------------------------------------------------------------------------
# ADC stream synthesis
# ---------------------------------------------------------------------------


def synth_adc_stream(
    tone_sources: Sequence[ToneSource],
    noise_sigma: float,
    pos_x_m: np.ndarray,
    pos_y_m: np.ndarray,
    t0_s: float,
    n_samples: int,
    adc_dt_s: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""Synthesizes the real ADC voltage stream per antenna.

    Each tone source contributes (Julia `calc_field` + `calc_delay`):

    .. math::

        E_a(t) = A \sin\!\left(2\pi f (t - \tau_a)\right), \qquad
        \tau_a = \frac{x_a l + y_a m}{c}

    and receiver noise is additive white Gaussian at the ADC.

    Returns
    -------
    adc:
        Float64 array (n_samples, n_ant).
    """
    if rng is None:
        rng = np.random.default_rng()
    pos_x = np.asarray(pos_x_m, dtype=np.float64)
    pos_y = np.asarray(pos_y_m, dtype=np.float64)
    n_ant = pos_x.size

    t = t0_s + np.arange(n_samples, dtype=np.float64) * adc_dt_s
    adc = rng.standard_normal((n_samples, n_ant)) * noise_sigma

    for src in tone_sources:
        delays = (src.l * pos_x + src.m * pos_y) / C_LIGHT  # (n_ant,)
        # E_a(t) = A sin(2 pi f (t - tau_a)); vectorized over (t, ant)
        phase = (2.0 * math.pi) * src.freq_hz * (t[:, None] - delays[None, :])
        adc += src.amp * np.sin(phase)

    return adc


# ---------------------------------------------------------------------------
# Full reference chain
# ---------------------------------------------------------------------------


def simulate_fengine_frame(
    tone_sources: Sequence[ToneSource],
    noise_sigma_adc: float,
    pos_x_m: np.ndarray,
    pos_y_m: np.ndarray,
    t0_s: float,
    n_spectra: int,
    config: Optional[FEngineConfig] = None,
    rng: Optional[np.random.Generator] = None,
    return_float: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Runs the full reference F-Engine chain for one frame.

    real ADC -> PFB -> quantize -> int4x2 packing.

    Parameters
    ----------
    tone_sources:
        Monochromatic sky sources (see :class:`ToneSource`).
    noise_sigma_adc:
        Receiver noise sigma in ADC units.
    pos_x_m, pos_y_m:
        Antenna positions (m).
    t0_s:
        Absolute start time of the ADC stream (seconds).
    n_spectra:
        Number of output channelized time samples.
    config:
        FEngineConfig (CHARTS defaults).
    return_float:
        Also return the unquantized float channels and the quantized complex
        values (for quantization SNR analysis).

    Returns
    -------
    packed:
        uint8 array (n_spectra, n_channels, n_ant) in Kotekan frame layout
        [time][freq][antenna], int4x2 packed.
    [channels_float, channels_quant]:
        Only when ``return_float=True``.
    """
    cfg = config or FEngineConfig()
    n_adc = (n_spectra + cfg.ntaps - 1) * cfg.nsamples

    adc = synth_adc_stream(
        tone_sources,
        noise_sigma_adc,
        pos_x_m,
        pos_y_m,
        t0_s,
        n_adc,
        cfg.adc_dt_s,
        rng=rng,
    )
    channels = pfb_channelize(adc, config=cfg)  # (n_spectra, n_ch, n_ant)
    packed, quantized = quantize_int4x2(channels, scale=cfg.quant_scale)

    if return_float:
        return packed, channels, quantized
    return packed


def measure_quantization_snr_loss(
    channels_float: np.ndarray,
    channels_quant: np.ndarray,
) -> float:
    r"""Quantization SNR loss in dB between float and quantized channels.

    .. math::

        \mathcal{L} = -10 \log_{10}
        \frac{\langle q \cdot x \rangle^2}{\langle q^2 \rangle \langle x^2 \rangle}

    computed over all samples (the loss of the quantized signal relative to
    the ideal correlation with the float reference). AGENTS.md §3 requires
    this to stay below 0.5 dB for the operating point chosen by the digitizer.
    """
    x = np.asarray(channels_float).ravel()
    q = np.asarray(channels_quant).ravel()
    cross = np.abs(np.vdot(q, x)) ** 2  # |<q, x>|^2
    denom = (np.vdot(q, q).real) * (np.vdot(x, x).real)
    if denom <= 0.0:
        return float("inf")
    return -10.0 * math.log10(cross / denom)


def optimal_noise_sigma_lsb(
    n_trials: int = 400_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Numerically finds the per-component noise sigma (in LSB) minimizing
    quantization SNR loss for the 15-level (±7) digitizer.

    The optimum for the round-clamp quantizer is ~2.7 LSB per component
    (loss ~0.06 dB); used to justify the fast-path operating point
    (`DIGITIZER_NOMINAL_SIGMA_LSB` in `sim.constants`).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sigmas = np.linspace(1.0, 4.0, 31)
    losses = []
    for sigma in sigmas:
        x = rng.standard_normal(n_trials) + 1j * rng.standard_normal(n_trials)
        x = x * sigma
        _, q = quantize_int4x2(x, scale=1.0)
        losses.append(measure_quantization_snr_loss(x, q))
    return float(sigmas[int(np.argmin(losses))])
