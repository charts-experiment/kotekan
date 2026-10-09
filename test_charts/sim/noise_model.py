#!/usr/bin/env python3
"""CHARTS Analog-Chain Thermal Noise Model.

Implements a physically motivated noise model for the CHARTS receiver chain:
  - Antenna: Dual-polarization differential patch (100 Ohm, 300-500 MHz, HPBW ~100 deg)
  - Stage 1 ULNA: QPL9547 at antenna feed (Gain = 19.3 dB, NF = 0.3 dB)
  - Stage 2 LNA:  PSA4-5043+ (Gain = 20.0 dB, NF = 0.65 dB @ 400 MHz)
  - Friis cascade: T_rx = T_1 + T_2 / G_1
  - Sky background: CMB (2.725 K) + Galactic synchrotron (Haslam 408 MHz scaling) + atmosphere (~1.5 K)
  - Ground spillover: Modeled via ~100 deg HPBW patch pattern (~2% effective spillover)
  - Solar radiation: Quiet / active Sun radio flux, topocentric ephemeris at Observatorio Carén,
    primary beam attenuation, and antenna temperature calculation
  - Digitizer mapping: Converts system temperature and celestial sources into 4-bit ADC voltage units
    ([-7, +7]), incorporating per-antenna gain dispersion and analog bandpass shape.
"""

from __future__ import annotations

import argparse
import datetime
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .constants import DIGITIZER_NOMINAL_SIGMA_LSB

# Physical constants
K_BOLTZMANN = 1.380649e-23     # J / K
C_LIGHT = 299_792_458.0        # m / s
T_PHYS_DEFAULT = 290.0         # Standard physical temperature in Kelvin
T_CMB = 2.725                  # Cosmic Microwave Background temperature in Kelvin
T_ATM_400MHZ = 1.5             # Typical atmospheric zenith brightness temperature in Kelvin


def db_to_lin_power(val_db: float) -> float:
    """Converts dB to linear power ratio."""
    return 10.0 ** (val_db / 10.0)


def lin_to_db_power(val_lin: float) -> float:
    """Converts linear power ratio to dB."""
    return 10.0 * math.log10(val_lin)


def noise_figure_to_temp(nf_db: float, t_phys: float = T_PHYS_DEFAULT) -> float:
    """Converts Noise Figure (dB) to equivalent noise temperature (K)."""
    f_factor = db_to_lin_power(nf_db)
    return (f_factor - 1.0) * t_phys


@dataclass
class AnalogChainParams:
    """Hardware parameters for CHARTS analog signal chain."""
    antenna_impedance_ohm: float = 100.0
    band_min_mhz: float = 300.0
    band_max_mhz: float = 500.0
    antenna_hpbw_deg: float = 100.0

    # Stage 1: Feed ULNA (QPL9547)
    ulna_gain_db: float = 19.3
    ulna_nf_db: float = 0.3

    # Stage 2: 2nd stage LNA (PSA4-5043+)
    lna2_gain_db: float = 20.0
    lna2_nf_db: float = 0.65

    # Ground and environment
    t_ground_k: float = 290.0
    ground_spillover_fraction: float = 0.02

    # Digitizer calibration: T_ref maps to DIGITIZER_NOMINAL_SIGMA_LSB per
    # component (Re/Im), matching the F-engine reference operating point
    # (sim/fengine.py; quantizer SNR loss ~0.09 dB, AGENTS.md §3 bound < 0.5 dB).
    t_ref_k: float = 50.0
    sigma_ref_lsb: float = DIGITIZER_NOMINAL_SIGMA_LSB


class ChartsNoiseModel:
    """Computes system temperatures and digitizer voltage statistics for CHARTS."""

    def __init__(
        self,
        params: Optional[AnalogChainParams] = None,
        site_lat_deg: float = -33.4211146,
        site_lon_deg: float = -70.8634710,
        site_alt_m: float = 458.0,
    ):
        self.params = params or AnalogChainParams()
        self.lat_deg = site_lat_deg
        self.lon_deg = site_lon_deg
        self.alt_m = site_alt_m

        # Precompute receiver noise temperature via Friis cascade
        self.t_ulna = noise_figure_to_temp(self.params.ulna_nf_db, self.params.t_ground_k)
        self.g_ulna_lin = db_to_lin_power(self.params.ulna_gain_db)
        self.t_lna2 = noise_figure_to_temp(self.params.lna2_nf_db, self.params.t_ground_k)

        # T_rx = T_1 + T_2 / G_1
        self.t_rx = self.t_ulna + (self.t_lna2 / self.g_ulna_lin)

        # Antenna directivity from HPBW (Gaussian beam approximation)
        hpbw_rad = math.radians(self.params.antenna_hpbw_deg)
        self.omega_a_sr = 1.133 * (hpbw_rad ** 2)
        self.directivity = (4.0 * math.pi) / self.omega_a_sr
        self.directivity_dbi = lin_to_db_power(self.directivity)

        # Primary beam Gaussian sigma in degrees
        self.sigma_pb_deg = self.params.antenna_hpbw_deg / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def receiver_temperature(self) -> float:
        """Returns the total receiver noise temperature in Kelvin."""
        return self.t_rx

    def effective_area(self, freq_mhz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes antenna effective aperture A_eff(nu) = D * lambda^2 / (4 * pi)."""
        wavelength_m = C_LIGHT / (np.asarray(freq_mhz) * 1e6)
        return (self.directivity * (wavelength_m ** 2)) / (4.0 * math.pi)

    def primary_beam_attenuation(self, theta_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes Gaussian primary beam attenuation for an angle theta (deg) off zenith."""
        th = np.asarray(theta_deg, dtype=np.float64)
        return np.exp(-0.5 * ((th / self.sigma_pb_deg) ** 2))

    def galactic_temperature(
        self, freq_mhz: Union[float, np.ndarray], galactic_factor: float = 0.5
    ) -> Union[float, np.ndarray]:
        """Scales Haslam 408 MHz Galactic synchrotron background."""
        f = np.asarray(freq_mhz, dtype=np.float64)
        haslam_ref = 25.5  # Kelvin at 408 MHz
        return haslam_ref * ((f / 408.0) ** -2.75) * galactic_factor

    def sky_temperature(
        self, freq_mhz: Union[float, np.ndarray], galactic_factor: float = 0.5
    ) -> Union[float, np.ndarray]:
        """Computes total sky background temperature: T_sky = T_CMB + T_gal(nu) + T_atm."""
        t_gal = self.galactic_temperature(freq_mhz, galactic_factor=galactic_factor)
        return T_CMB + t_gal + T_ATM_400MHZ

    def ground_spillover_temperature(self, spillover_fraction: Optional[float] = None) -> float:
        """Computes ground spillover noise pickup."""
        frac = self.params.ground_spillover_fraction if spillover_fraction is None else spillover_fraction
        return self.params.t_ground_k * frac

    def sun_position(self, utc_dt: datetime.datetime) -> Tuple[float, float, float, float, float]:
        """Computes topocentric solar coordinates for Observatorio Carén.
        Returns: (l, m, n, elevation_deg, azimuth_deg)
        """
        doy = utc_dt.timetuple().tm_yday
        hour_frac = utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0 + utc_dt.microsecond / 3.6e9

        gamma = 2.0 * math.pi * (doy - 1) / 365.0
        decl_deg = (
            0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2.0 * gamma)
            + 0.000907 * math.sin(2.0 * gamma)
            - 0.002697 * math.cos(3.0 * gamma)
            + 0.001480 * math.sin(3.0 * gamma)
        ) * (180.0 / math.pi)

        eot_min = 229.18 * (
            0.000075
            + 0.001868 * math.cos(gamma)
            - 0.032077 * math.sin(gamma)
            - 0.014615 * math.cos(2.0 * gamma)
            - 0.040849 * math.sin(2.0 * gamma)
        )

        solar_time_h = (hour_frac + (self.lon_deg / 15.0) + (eot_min / 60.0)) % 24.0
        ha_deg = (solar_time_h - 12.0) * 15.0

        ha_rad = math.radians(ha_deg)
        dec_rad = math.radians(decl_deg)
        lat_rad = math.radians(self.lat_deg)

        l = -math.cos(dec_rad) * math.sin(ha_rad)
        m = math.sin(dec_rad) * math.cos(lat_rad) - math.cos(dec_rad) * math.sin(lat_rad) * math.cos(ha_rad)
        n = math.sin(dec_rad) * math.sin(lat_rad) + math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad)

        elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, n))))
        azimuth_rad = math.atan2(l, m)
        azimuth_deg = (math.degrees(azimuth_rad) + 360.0) % 360.0

        return float(l), float(m), float(n), float(elevation_deg), float(azimuth_deg)

    def sun_flux_jy(
        self, freq_mhz: Union[float, np.ndarray], activity: str = "quiet"
    ) -> Union[float, np.ndarray]:
        """Computes Sun radio flux density in Jansky across 300-500 MHz."""
        f = np.asarray(freq_mhz, dtype=np.float64)
        base_flux_400 = 1.5e5  # Jy
        scale = {"quiet": 1.0, "moderate": 2.5, "active": 6.0}.get(activity.lower(), 1.0)
        return base_flux_400 * scale * ((f / 400.0) ** -1.0)

    def sun_antenna_temperature(
        self,
        freq_mhz: Union[float, np.ndarray],
        utc_dt: datetime.datetime,
        activity: str = "quiet",
    ) -> Tuple[Union[float, np.ndarray], float, float, float]:
        """Computes Sun antenna temperature after primary beam attenuation."""
        l, m, n, elev_deg, _ = self.sun_position(utc_dt)
        if elev_deg <= 0.0:
            return (np.zeros_like(freq_mhz, dtype=np.float64) if isinstance(freq_mhz, np.ndarray) else 0.0,
                    elev_deg, l, m)

        theta_sun_deg = 90.0 - elev_deg
        pb_gain = self.primary_beam_attenuation(theta_sun_deg)

        a_eff = self.effective_area(freq_mhz)
        flux_jy = self.sun_flux_jy(freq_mhz, activity=activity)
        flux_si = flux_jy * 1e-26  # W / m^2 / Hz

        t_sun_unattenuated = (a_eff * flux_si) / (2.0 * K_BOLTZMANN)
        t_sun_pb = t_sun_unattenuated * pb_gain

        return t_sun_pb, elev_deg, l, m

    def system_temperature(
        self,
        freq_mhz: Union[float, np.ndarray],
        utc_dt: Optional[datetime.datetime] = None,
        include_sun: bool = True,
        galactic_factor: float = 0.5,
        sun_activity: str = "quiet",
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Computes full system temperature components: T_sys = T_rx + T_sky + T_ground + T_sun_pb."""
        t_rx = self.receiver_temperature()
        t_sky = self.sky_temperature(freq_mhz, galactic_factor=galactic_factor)
        t_ground = self.ground_spillover_temperature()

        t_sun_pb = 0.0
        elev_deg = -90.0
        l_sun, m_sun = 0.0, 0.0

        if include_sun and utc_dt is not None:
            t_sun_pb, elev_deg, l_sun, m_sun = self.sun_antenna_temperature(
                freq_mhz, utc_dt, activity=sun_activity
            )

        t_noise_incoherent = t_rx + t_sky + t_ground
        t_sys_total = t_noise_incoherent + t_sun_pb

        return {
            "t_rx": t_rx,
            "t_sky": t_sky,
            "t_ground": t_ground,
            "t_sun_pb": t_sun_pb,
            "t_noise_incoherent": t_noise_incoherent,
            "t_sys_total": t_sys_total,
            "sun_elevation_deg": elev_deg,
            "sun_l": l_sun,
            "sun_m": m_sun,
        }

    def temp_to_adc_sigma(self, temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Converts temperature (K) to 4-bit ADC voltage standard deviation (LSB)."""
        t = np.asarray(temp_k, dtype=np.float64)
        t_safe = np.maximum(t, 0.0)
        return self.params.sigma_ref_lsb * np.sqrt(t_safe / self.params.t_ref_k)

    def flux_jy_to_antenna_temp(
        self, flux_jy: Union[float, np.ndarray], freq_mhz: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Converts celestial flux density S (Jy) to unattenuated antenna temperature T_A (K)."""
        a_eff = self.effective_area(freq_mhz)
        flux_si = np.asarray(flux_jy, dtype=np.float64) * 1e-26
        return (a_eff * flux_si) / (2.0 * K_BOLTZMANN)

    def generate_antenna_gain_dispersion(
        self, num_antennas: int = 64, seed: int = 42
    ) -> np.ndarray:
        """Generates realistic per-antenna receiver gain dispersion."""
        rng = np.random.default_rng(seed)
        gains = rng.normal(1.0, 0.05, size=num_antennas).astype(np.float32)
        gains = np.clip(gains, 0.85, 1.15)

        n_degraded = max(1, int(num_antennas * 0.05))
        bad_indices = rng.choice(num_antennas, size=n_degraded, replace=False)
        for idx in bad_indices:
            gains[idx] *= rng.uniform(1.3, 1.6)

        return gains

    def generate_bandpass_shape(
        self, freqs_hz: np.ndarray, tilt_db: float = 1.0, ripple_amp: float = 0.03
    ) -> np.ndarray:
        """Generates realistic analog filter bandpass response."""
        f_mhz = freqs_hz / 1e6
        f_mid = 0.5 * (f_mhz[0] + f_mhz[-1])
        bw = max(1.0, f_mhz[-1] - f_mhz[0])

        tilt_lin = 1.0 + (db_to_lin_power(tilt_db) - 1.0) * ((f_mhz - f_mid) / bw)
        ripple = 1.0 + ripple_amp * np.sin(2.0 * np.pi * (f_mhz - f_mhz[0]) / 25.0)

        shape = (tilt_lin * ripple).astype(np.float32)
        return shape / np.mean(shape)
