# CHARTS Simulation Framework — Review & Improvement Plan

Status: **implemented** (see `sim/sky.py`, `sim/fengine.py`, `sim/verify.py`, reworked
`sim/generator.py` / `sim/pipeline.py`, tests in `tests/`).

## 1. Review findings (why the framework was not planned properly)

### F1 — The sky model is fake: sources never transit
`sim/generator.py` rendered every source with a **fixed** `(l0, m0)` for the whole
window plus an arbitrary per-event-type drift term
(`base_phase = 2π·(t·κ)·(f·1e-8)` with κ = 0.005 / 0.02 / 0.01 / 0.003 / 0.05).
Real transits follow Earth rotation:

$$
\phi_a(f, t) = -2\pi f\,\frac{x_a\,l(t) + y_a\,m(t)}{c},
\qquad (l, m)(t) \text{ from } (\alpha, \delta, \mathrm{LST}(t))
$$

The X-engine trackers (`cudaDirectBeamTrackerCommand.cpp`) compute exactly that —
per-frame direction updates from LST plus sub-frame phase interpolation — so the
sim could never validate tracking against the transit trajectories confirmed with
the tracker-viewer routine.

### F2 — Three disconnected time systems
- Sim: `HH:MM` / float hours anchored to equinox 2026-03-20 (`astro.parse_observation_time`).
- Viewer routine (`direct-beam-tracker-viewer/src/viewer/routine.py`): local solar set-hour slots.
- Verified catalog (`tools/verified_targets.json`, date 2026-10-07): real SIMBAD-verified
  transit UTC times — the confirmed ground truth.

There was no way to say "simulate the window at Vela's confirmed transit".

### F3 — No F-engine model (Julia reference not used)
The real chain (per `RadioTelescopeFEngine.jl` `channelize!` / `quantize!` and the
CHARTS RFSoC: ADC 2457.6 MHz, 8192-pt PFB, 4 taps, 336 × 300 kHz channels) is:

$$
X_k = \sum_{n} \frac{w_n}{N/2}\, x_n\, e^{-2\pi i k n / N},
\qquad w_n = \cos^2(\pi s')\,\mathrm{sinc}(M s')
$$

$$
q = \mathrm{round}\left(\mathrm{clamp}(7.5\,X,\ -7,\ +7)\right) \rightarrow \text{int4x2}
$$

The Python sim synthesized channel voltages directly with `clip(round(v), ±7)`
(scale 1.0) — a different quantizer operating point, no PFB channel leakage, and no
gold reference to validate the fast path against.

### F4 — Writer-independence refactor half-done
`sim/writer.py` implements the full `BasebandWriter` protocol (`RawBinWriter`,
`HDF5Writer`, manifests) with tests, but `generate_simulation_window()` still wrote
files directly and did not accept a `writer` kwarg →
`tests/test_writer_independence.py::test_02` failed.

### F5 — Invalid Kotekan YAML + partial X-engine coverage
`cudaCorrelatorAstron`, `cudaShuffleAstron`, `cudaDirectBeamTrackerCommand`,
`cudaBiSLCCommand`, `cudaTransientTriggerCommand` are
`cudaCommand`s that must run inside a `cudaProcess` stage command list.
`sim/pipeline.py` emitted them as top-level `kotekan_stage:` entries — **invalid
configs that kotekan cannot start**. Coverage also stopped at correlator + direct
tracker: no BiSLC, no transient trigger, no accumulate (legacy Beam Tracker V5
is deprecated and unsupported; CHARTS exclusively uses Direct Beam Tracker).

### F6 — No ground-truth verification of X-engine outputs
`inspector.py` checked Hermitian symmetry and power statistics, but nothing verified
pointing accuracy, visibility phasing vs. geometry, transit timing, or quantization
SNR loss. AGENTS.md §3 requires equivalence tests (pointing < 1e-4 rad, quantization
SNR loss < 0.5 dB).

### F7 — Duplicate, disagreeing catalogs
`astro.CELESTIAL_CATALOG` hardcoded RA/Dec values that disagree with the
SIMBAD-verified catalog (e.g. Puppis A 125.6875° vs. verified 126.029°).

## 2. Improvements implemented

### 2.1 Unified sky & time model — `sim/sky.py`
- `load_verified_catalog()` reads `tools/verified_targets.json` (the catalog
  confirmed via the tracker-viewer routine); `find_verified_target()` does fuzzy
  name lookup ("vela" → "Vela SNR / PSR B0833-45").
- `direction_cosines_track(ra, dec, unix_s, lat, lon)` — vectorized per-sample
  `(l, m, n)(t)`, bit-identical formula to kotekan's `compute_celestial_direction`
  (`cudaDirectBeamTracker.hpp`).
- `resolve_window_start("transit:Vela", duration_s)` anchors windows at the
  **confirmed transit UTC** (centered, with optional `transit+Δs:` / `transit-Δs:`
  offsets); also accepts ISO, `HH:MM`, float hour, and `slot:HH:MM` local routine
  slots (matching the viewer routine's set-hour table).

### 2.2 Physical source rendering — `sim/generator.py` rework
- All celestial events (FRB, pulsar, narrowband RFI, Sun) now carry `ra_deg` /
  `dec_deg` (resolved from the verified catalog) and are rendered with the true
  time-evolving geometric phase — the fake drift constants are gone. LEO satellites
  keep a physical linear `(l, m)` drift.
- The Sun is rendered as a celestial source at its computed RA/Dec for the window
  time (was: fixed l/m for the whole window).
- `generate_simulation_window(cfg, writer=None)` now goes through the
  `BasebandWriter` protocol (default `RawBinWriter`) — fixes F4.
- Digitizer operating point calibrated so nominal T_sys → σ ≈ 2 LSB
  (`DIGITIZER_NOMINAL_SIGMA_LSB`), matching the Julia scale-7.5 convention;
  quantization SNR loss verified < 0.5 dB.

### 2.3 F-engine reference (Julia parity) — `sim/fengine.py`
NumPy port of `RadioTelescopeFEngine.jl`:
- `sinc_hanning_window(ntaps, nsamples)` — eq. (11), Erik convention.
- `pfb_channelize(adc, ...)` — 4-tap PFB, `w/(N/2)` normalization, rfft,
  every-ntaps bin pick, CHARTS channel map (bins 1000..1335 → 300–400.8 MHz).
- `quantize_int4x2(x, scale=7.5)` — Julia `quantize!` convention.
- `simulate_fengine_frame(...)` — full real-ADC → PFB → quantize → int4x2 chain
  (gold reference for small windows / calibration of the fast path).

### 2.4 Ground-truth X-engine verification — `sim/verify.py`
Pure-NumPy reference implementations + equivalence checks (no kotekan binary needed):
- `reference_visibilities()` / `reference_beamform()` — same conventions as
  `cudaCorrelatorAstron` (V_ij = Σ v_i conj(v_j)) and `cudaDirectBeamTracker`
  (w = exp(+j k·pos·dir)/√N_active, formed = Σ w·v).
- `verify_visibility_phasing()` — visibility phase vs. geometric prediction.
- `verify_pointing()` — formed-beam power vs. pointing offset; recovers the
  injected source direction to < 1e-4 rad.
- `verify_quantization_snr_loss()` — < 0.5 dB vs. float reference.
- `verify_transit_lightcurve()` — tracker lightcurve peaks at the confirmed
  transit time.

### 2.5 Valid Kotekan YAML for all X-engine components — `sim/pipeline.py`
All generators now emit the correct `cudaProcess` command-list structure
(`rawFileRead → cudaInputData → cudaSyncInput → … → cudaSyncOutput → cudaOutputData
→ rawFileWrite`), mirroring the live configs:
- `create_correlator_yaml()` — fixed (was invalid).
- `create_beam_tracker_yaml()` — direct tracker (`cudaDirectBeamTrackerCommand`),
  fixed; optional BiSLC + transient-trigger commands (Beam Tracker V5 is legacy
  and unsupported in CHARTS).
- `create_accumulate_yaml()` — correlator + `chartsAccumulate` integration.

### 2.6 Tests
- `tests/test_sky_transit_model.py` — catalog loading, transit anchoring,
  kotekan-formula parity, transit trajectory (l(t), m(t)) continuity.
- `tests/test_fengine_parity.py` — PFB channelization recovers tones at the right
  channels with the Julia leakage pattern; quantizer convention; ADC→PFB→quantize
  end-to-end frame.
- `tests/test_xengine_verification.py` — pointing accuracy < 1e-4 rad, visibility
  phasing, quantization SNR loss < 0.5 dB, transit lightcurve.
- `tests/test_writer_independence.py` — passes again (writer kwarg).

## 3. Verification matrix (X-engine component → sim test path)

| X-engine component | Sim input | Ground-truth check |
|---|---|---|
| `cudaCorrelatorAstron` | transit-anchored window | `verify_visibility_phasing` (NumPy reference) |
| `cudaDirectBeamTracker` | transit-anchored window | `verify_pointing`, `verify_transit_lightcurve` |
| `cudaBeamTrackerV5` | *(Legacy / Deprecated)* | **Unsupported** (CHARTS exclusively uses Direct Beam Tracker) |
| `cudaBiSLCCommand` | tracker beams | YAML chain (direct → BiSLC) |
| `cudaTransientTriggerCommand` | FRB/pulsar events | YAML chain + event catalog |
| `cudaAntennaMask` | saturated/dead antennas | YAML chain + clip-fraction stats |
| `chartsAccumulate` | correlator dumps | `create_accumulate_yaml` |
| Quantizer (F-engine out) | `fengine.py` reference | `verify_quantization_snr_loss` < 0.5 dB |
