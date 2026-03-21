# config.py — All parameters for the SNN agent.
# Edit here only; no other files need to change for topology/parameter adjustments.
# Keep the N constant in index.html in sync with n_hidden below.

CFG = {

    # ── Mode ─────────────────────────────────────────────────────────────────
    # "binary"    : original bit-encoded I/O (test_sender.py)
    # "electrode" : real-time neural signal processing (ANNet-derived)
    # "lsl"       : Lab Streaming Layer input (lsl_player.py → mne-lsl)
    "mode": "lsl",

    # ── Network topology (binary mode) ────────────────────────────────────────
    "input_bit_width":  16,
    "n_hidden":         64,
    "output_bit_width":  8,

    # ── LIF neuron dynamics (binary mode) ─────────────────────────────────────
    "tau_m":   20e-3,
    "V_th":   -50e-3,
    "V_reset": -65e-3,
    "dt":        1e-3,

    # ── Reward-modulated STDP (binary mode) ───────────────────────────────────
    "A_plus":   0.010,
    "A_minus":  0.012,
    "tau_pre":  20e-3,
    "tau_post": 20e-3,
    "W_max":     1.0,

    # ── Network ports ────────────────────────────────────────────────────────
    "udp_port":       9000,     # UDP in  — binary mode input frames
    "ws_port":        8765,     # WS      — browser spike stream + reward
    "http_port":      8080,     # HTTP    — serves index.html
    "udp_electrode_port": 9001, # UDP in  — raw electrode samples
    "udp_control_port":   9002, # UDP out — control signal to experiment
    "control_target_host": "127.0.0.1",

    # ── Electrode signal ─────────────────────────────────────────────────────
    "sampling_rate_hz": 80_000,

    # ── Temporal receptive field encoding ─────────────────────────────────────
    "enc_overlap":       10,      # receptive field overlap factor
    "enc_dvm_factor":    1.75,    # noise scale factor for RF half-width
    "enc_step_size":     4,       # temporal subsampling (samples between delays)
    "enc_window_depth":  10,      # number of delay taps
    "enc_noise_init_samples": 8000,  # initial MAD estimation window
    "enc_noise_ema_alpha":  0.001,   # online MAD tracking rate

    # ── Attention neuron (DN) ────────────────────────────────────────────────
    "dn_tm_samples":             2,       # membrane time constant (samples)
    "dn_threshold_factor":       0.4252,  # threshold = factor × overlap × window × ...
    "dn_reset_potential_factor": 0.15,    # soft reset = factor × (e^(1/tm)-1) × threshold
    "dn_depression_tau":         400,     # pRel recovery (samples)
    "dn_depression_frac":        0.01613, # pRel depletion per spike (~1.6%)

    # ── Template layer (L1) ──────────────────────────────────────────────────
    "l1_n_neurons":          60,
    "l1_tm_samples":         2,          # membrane time constant (samples)
    "l1_reset_potential":    0.0,        # hard reset
    "l1_refractory_samples": 1,
    "l1_dn_weight":          45.0,       # DN excitatory boost to all L1 neurons
    "l1_init_w_min":         0.4,
    "l1_init_w_max":         1.0,
    "l1_w_bounds":           (0.0, 1.0),
    "l1_stdp_ltp":           0.005,      # LTP amplitude
    "l1_stdp_ltp_window":    4,          # LTP window (samples)
    "l1_stdp_ltd":          -0.00275,    # global LTD at post-spike
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

    # ── Viz ──────────────────────────────────────────────────────────────────
    "broadcast_every": 5,               # send to browser every N sim steps
}
