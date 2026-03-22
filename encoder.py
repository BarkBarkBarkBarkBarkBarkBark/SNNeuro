"""
encoder.py — Preprocessing, temporal receptive field encoder, and attention neuron.

Derived from MB2018 ANNet (MATLAB/C MEX).
See annet_architecture.yaml §1 (encoding) and §1 (attention_neuron).

Preprocessor
    Optional causal IIR bandpass filter (300–6 000 Hz) followed by integer
    decimation.  Runs sample-by-sample, maintaining filter state across calls.

SpikeEncoder
    Converts a raw electrode sample (float) into a binary afferent vector.
    Maintains a shift register for temporal delay taps.

AttentionNeuron
    Single LIF neuron with synaptic depression (pRel, no learnable weights).
    Fires when signal energy exceeds the noise floor.
    Provides:  1) excitatory boost for the template layer
               2) gating / confidence signal for the control decoder
"""

import numpy as np
from scipy.signal import butter, sosfilt


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessor — causal bandpass + decimation
# ─────────────────────────────────────────────────────────────────────────────
class Preprocessor:
    """
    Optional front-end: bandpass filter then decimate.

    Bandpass uses second-order sections (SOS) applied sample-by-sample via
    sosfilt with carried state, so it is causal and streaming-safe.

    Decimation simply keeps every N-th sample (after the anti-alias bandpass
    has already removed energy above Nyquist/N).

    Usage:
        pp = Preprocessor(cfg)
        for raw_sample in stream:
            for sample in pp.step(raw_sample):
                encoder.step(sample)
    """

    def __init__(self, cfg: dict):
        fs = cfg["sampling_rate_hz"]

        # ── Bandpass ──────────────────────────────────────────────────────
        self.do_bandpass = cfg.get("enable_bandpass", False)
        if self.do_bandpass:
            lo = cfg["bandpass_lo_hz"]
            hi = cfg["bandpass_hi_hz"]
            order = cfg.get("bandpass_order", 2)
            self._sos = butter(order, [lo, hi], btype="band",
                               fs=fs, output="sos")
            # sosfilt state: (n_sections, 2)
            self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)

        # ── Decimation ────────────────────────────────────────────────────
        self.do_decimate = cfg.get("enable_decimation", False)
        self.dec_factor  = cfg.get("decimation_factor", 1)
        self._dec_count  = 0

        # Effective sample rate after decimation (informational)
        self.effective_fs = fs // self.dec_factor if self.do_decimate else fs

    def step(self, sample: float) -> list[float]:
        """
        Ingest one raw sample.  Returns a list of output samples:
          - [] if the sample was decimated away
          - [filtered_sample] otherwise
        """
        x = sample

        # Bandpass (sample-by-sample via sosfilt with state)
        if self.do_bandpass:
            out, self._zi = sosfilt(self._sos, np.array([x]),
                                    zi=self._zi)
            x = float(out[0])

        # Decimation
        if self.do_decimate:
            self._dec_count += 1
            if self._dec_count < self.dec_factor:
                return []
            self._dec_count = 0

        return [x]


# ─────────────────────────────────────────────────────────────────────────────
#  SpikeEncoder — temporal receptive field encoding
# ─────────────────────────────────────────────────────────────────────────────
class SpikeEncoder:
    """
    Converts a stream of scalar electrode samples into a flat boolean
    afferent vector of shape (n_afferents,) at each timestep.

    Architecture (from ANNet):
        1. Estimate noise via running Median Absolute Deviation (MAD).
        2. Build a set of overlapping amplitude bins (centers).
        3. At each sample, activate the bins that bracket the amplitude.
        4. Feed activations into a shift register to retain a short
           temporal history (twindow delay taps, subsampled by step_size).
        5. Flatten the [nCenters × twindow] matrix into a 1-D bool vector.
    """

    def __init__(self, cfg: dict):
        self.overlap      = cfg["enc_overlap"]
        self.dvm_factor   = cfg["enc_dvm_factor"]
        self.step_size    = cfg["enc_step_size"]
        self.twindow      = cfg["enc_window_depth"]
        self.ema_alpha    = cfg["enc_noise_ema_alpha"]
        self.init_samples = cfg["enc_noise_init_samples"]

        # State — populated on first call or after calibrate()
        self.centers: np.ndarray | None = None
        self.dvm:     float             = 0.0
        self.n_centers: int             = 0
        self.n_afferents: int           = 0

        # Shift register: [nCenters × (step_size × twindow)]
        self._shift_reg: np.ndarray | None = None

        # Running noise / signal stats for online MAD
        self._abs_buf: list[float] = []   # accumulates |sample| during init
        self._noise_est: float     = 1.0  # running noise σ
        self._sig_min: float       = 0.0
        self._sig_max: float       = 0.0
        self._calibrated: bool     = False
        self._sample_count: int    = 0

    # ── public API ────────────────────────────────────────────────────────────
    def step(self, sample: float) -> np.ndarray:
        """
        Ingest one electrode sample.
        Returns a flat boolean array of shape (n_afferents,).
        During calibration (first `init_samples` samples) returns an empty
        array while accumulating noise statistics.
        """
        self._sample_count += 1

        if not self._calibrated:
            self._abs_buf.append(abs(sample))
            if sample < self._sig_min:
                self._sig_min = sample
            if sample > self._sig_max:
                self._sig_max = sample
            if self._sample_count >= self.init_samples:
                self._calibrate()
                # fall through to encode this first sample
            else:
                return np.zeros(0, dtype=bool)

        # Update running noise estimate (exponential moving average of |x|)
        self._noise_est += self.ema_alpha * (abs(sample) - self._noise_est)

        assert self.centers is not None and self._shift_reg is not None

        # Activate amplitude bins
        active = np.abs(self.centers - sample) <= self.dvm   # bool [nCenters]

        # Shift register: shift left by nCenters, insert new at the right
        self._shift_reg = np.roll(self._shift_reg, -self.n_centers)
        self._shift_reg[-self.n_centers:] = active

        # Subsample: pick every step_size-th column from the register
        # Register shape is flat [nCenters × (step_size × twindow)].
        # We want positions [0, step_size, 2*step_size, ...] per center.
        out = self._shift_reg.reshape(self.step_size * self.twindow,
                                      self.n_centers).T
        # out is [nCenters × (step_size × twindow)]; subsample rows
        current = out[:, ::self.step_size]   # [nCenters × twindow]
        return current.ravel()               # flat [n_afferents]

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ── internals ─────────────────────────────────────────────────────────────
    def _calibrate(self) -> None:
        """Build amplitude bins from the calibration buffer."""
        arr = np.array(self._abs_buf, dtype=np.float64)
        self._noise_est = float(np.median(arr) / 0.6745)
        self.dvm = self.dvm_factor * self._noise_est

        dvi = self.dvm * 2.0 / self.overlap  # spacing between centers
        self.centers = np.arange(self._sig_min - self.dvm,
                                 self._sig_max + self.dvm + dvi,
                                 dvi, dtype=np.float64)
        self.n_centers   = len(self.centers)
        self.n_afferents = self.n_centers * self.twindow

        reg_len = self.n_centers * self.step_size * self.twindow
        self._shift_reg = np.zeros(reg_len, dtype=bool)

        self._abs_buf.clear()
        self._calibrated = True


# ─────────────────────────────────────────────────────────────────────────────
#  AttentionNeuron — activity detector with synaptic depression
# ─────────────────────────────────────────────────────────────────────────────
class AttentionNeuron:
    """
    Single LIF neuron (no learnable weights).
    Each afferent contributes its release probability pRel as the weight.
    pRel models short-term synaptic depression:
        Recovery:   pRel(t) = 1 − (1 − pRel_last) × exp(−Δt / τ_d)
        Depression: pRel(t) *= (1 − f_d)

    Fires when V ≥ threshold.  Soft reset (V += reset_potential).
    """

    def __init__(self, cfg: dict, n_afferents: int):
        tm = cfg["dn_tm_samples"]

        # Threshold: factor × overlap × window_depth / (1 − exp(−1/tm))
        self.threshold = (cfg["dn_threshold_factor"]
                          * cfg["enc_overlap"]
                          * cfg["enc_window_depth"]
                          / (1.0 - np.exp(-1.0 / tm)))

        # Reset potential (additive after spike)
        self.reset_potential = (cfg["dn_reset_potential_factor"]
                                * (np.exp(1.0 / tm) - 1.0)
                                * self.threshold)

        self.tau_d = cfg["dn_depression_tau"]     # pRel recovery τ (samples)
        self.f_d   = cfg["dn_depression_frac"]    # pRel depletion fraction

        # Precomputed decay factor for membrane potential
        self.decay = np.exp(-1.0 / tm)

        # State
        self.v = 0.0                                         # membrane potential
        self.p_rel = np.ones(n_afferents, dtype=np.float64)  # release probs
        self.last_pre = np.full(n_afferents, -9999, dtype=np.int64)
        self.t = 0                                           # sample counter
        self.n_spikes = 0

        # Precompute exp LUT for depression recovery (up to 20×τ_d)
        max_dt = int(self.tau_d * 20)
        self._exp_td = np.exp(-np.arange(max_dt + 1, dtype=np.float64)
                              / self.tau_d)

    # ── public API ────────────────────────────────────────────────────────────
    def step(self, afferents: np.ndarray) -> bool:
        """
        Ingest the flat afferent vector (bool).
        Returns True if the attention neuron fires this timestep.
        """
        self.t += 1
        active = np.flatnonzero(afferents)

        if len(active) == 0:
            # Pure decay — no input
            self.v *= self.decay
            return False

        # Update pRel for active afferents (recovery + depression)
        dt_arr = self.t - self.last_pre[active]
        dt_arr = np.clip(dt_arr, 0, len(self._exp_td) - 1)
        exp_vals = self._exp_td[dt_arr]

        self.p_rel[active] = 1.0 - (1.0 - self.p_rel[active]) * exp_vals
        contributions = self.p_rel[active].copy()
        self.p_rel[active] *= (1.0 - self.f_d)
        self.last_pre[active] = self.t

        # Integrate: V decays then receives summed pRel contributions
        self.v = self.v * self.decay + contributions.sum()

        # Fire?
        fired = self.v >= self.threshold
        if fired:
            self.v = self.reset_potential   # soft reset (V set to sub-threshold)
            self.n_spikes += 1

        return bool(fired)
