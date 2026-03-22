# config.py — All parameters for the SNN agent.
# Edit here only; no other files need to change for topology/parameter adjustments.

CFG = {

    # ── Mode ─────────────────────────────────────────────────────────────────
    # "electrode" : real-time neural signal processing (ANNet-derived, UDP in)
    # "lsl"       : Lab Streaming Layer input (lsl_player.py → mne-lsl)
    # "synthetic" : SpikeInterface ground-truth recording (no hardware needed)
    "mode": "lsl",

    # ── Network ports ────────────────────────────────────────────────────────
    "ws_port":        8765,     # WS      — browser spike stream
    "http_port":      8080,     # HTTP    — serves index.html
    "udp_electrode_port": 9001, # UDP in  — raw electrode samples
    "udp_control_port":   9002, # UDP out — control signal to experiment
    "control_target_host": "127.0.0.1",

    # ── Electrode signal ─────────────────────────────────────────────────────
    "sampling_rate_hz": 80_000,

    # ── Preprocessing (bandpass + decimation) ────────────────────────────────
    "enable_bandpass":     True,        # causal IIR bandpass before encoding
    "bandpass_lo_hz":      300,         # high-pass corner
    "bandpass_hi_hz":      6000,        # low-pass corner
    "bandpass_order":      2,           # SOS filter order (per section)
    "enable_decimation":   True,        # decimate after bandpass
    "decimation_factor":   4,           # 80 kHz → 20 kHz (must satisfy Nyquist)

    # ── Temporal receptive field encoding ─────────────────────────────────────
    "enc_overlap":       9,       # receptive field overlap factor          (optimized trial #65)
    "enc_dvm_factor":    2.055,   # noise scale factor for RF half-width    (optimized trial #65)
    "enc_step_size":     2,       # temporal subsampling (samples between delays) (optimized trial #65)
    "enc_window_depth":  10,      # number of delay taps
    "enc_noise_init_samples": 8000,  # initial MAD estimation window
    "enc_noise_ema_alpha":  0.001,   # online MAD tracking rate

    # ── Attention neuron (DN) ────────────────────────────────────────────────
    "dn_tm_samples":             2,       # membrane time constant (samples)
    "dn_threshold_factor":       0.3825,  # optimized trial #65 (was 0.4252)
    "dn_reset_potential_factor": 0.15,    # soft reset = factor × (e^(1/tm)-1) × threshold
    "dn_depression_tau":         400,     # pRel recovery (samples)
    "dn_depression_frac":        0.01613, # pRel depletion per spike (~1.6%)

    # ── Template layer (L1) ──────────────────────────────────────────────────
    "l1_n_neurons":          110,        # optimized trial #65 (was 60)
    "l1_tm_samples":         2,          # membrane time constant (samples)
    "l1_reset_potential":    0.0,        # hard reset
    "l1_refractory_samples": 1,
    "l1_dn_weight":          113.26,     # DN excitatory boost (optimized trial #65, was 45.0)
    "l1_init_w_min":         0.4,
    "l1_init_w_max":         1.0,
    "l1_w_bounds":           (0.0, 1.0),
    "l1_stdp_ltp":           0.016,      # LTP amplitude                       (optimized trial #65)
    "l1_stdp_ltp_window":    4,          # LTP window (samples)
    "l1_stdp_ltd":          -0.00763,    # global LTD at post-spike             (optimized trial #65)
    "l1_freeze_stdp":        False,      # True → lock weights for deployment

    # ── Control decoder ──────────────────────────────────────────────────────
    # strategy: "rate" | "population" | "trigger"
    "ctrl_strategy":       "rate",
    "ctrl_window_ms":      5.0,         # sliding window for rate strategy
    "ctrl_weights":        None,        # None = uniform; or list[float] len=n_l1
    "ctrl_threshold":      0.5,         # population/trigger threshold
    "ctrl_leaky_tau_ms":   10.0,        # integrator time constant
    "ctrl_dn_confidence_window_ms": 5.0,

    # ── LSL input ────────────────────────────────────────────────────────────
    "lsl_stream_name":     "NCS-Replay",  # must match lsl_player.py --name
    "lsl_pick_channel":    None,           # None = first channel; or "CSC1"
    "lsl_bufsize_sec":     5.0,            # StreamLSL ring buffer (seconds)
    "lsl_poll_interval_s": 0.0005,         # async poll sleep when idle (500 µs)

    # ── Synthetic (SpikeInterface ground truth) ───────────────────────────────
    "synth_duration_s":   20.0,          # recording length
    "synth_fs":           30_000,        # native sampling rate (Hz)
    "synth_num_units":    2,             # number of ground-truth neurons
    "synth_noise_level":  8.0,           # additive noise amplitude
    "synth_seed":         42,            # RNG seed for reproducibility
    "synth_realtime":     True,          # True = pace to wall-clock; False = fast

    # ── Viz ──────────────────────────────────────────────────────────────────
    "broadcast_every": 5,               # send to browser every N sim steps
}
