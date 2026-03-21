"""
snn.py — Spiking Neural Network modules (NumPy only).

Contains:
  Network       — Original binary I/O SNN (rate encoding, R-STDP)
  TemplateLayer — ANNet-derived L1 pattern matching layer (electrode mode)

See annet_architecture.yaml for the full derivation.
"""

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  TemplateLayer — ANNet L1 (pattern matching via STDP + WTA)
# ═════════════════════════════════════════════════════════════════════════════
class TemplateLayer:
    """
    N LIF neurons that learn recurring spatio-temporal patterns in the
    encoded electrode signal via competitive asymmetric STDP.

    Each neuron becomes a template matcher for a particular waveform shape.
    Winner-take-all competition ensures distinct templates.

    Inputs per step:
        afferents : bool array [n_afferents]  — from SpikeEncoder
        dn_spike  : bool                      — from AttentionNeuron

    Output per step:
        spikes    : bool array [n_neurons]    — which L1 neurons fired
    """

    def __init__(self, cfg: dict, n_afferents: int):
        n = cfg["l1_n_neurons"]
        tm = cfg["l1_tm_samples"]

        # Threshold: (dn_weight + overlap×(window-k)) × (1-e^(-stepSize/tm))
        #            / (1-e^(-1/tm))   where k=3
        k = 3
        overlap = cfg["enc_overlap"]
        window  = cfg["enc_window_depth"]
        step_sz = cfg["enc_step_size"]
        self.threshold = ((cfg["l1_dn_weight"]
                           + overlap * (window - k))
                          * (1.0 - np.exp(-step_sz / tm))
                          / (1.0 - np.exp(-1.0 / tm)))

        self.reset_v   = cfg["l1_reset_potential"]
        self.dn_weight = cfg["l1_dn_weight"]
        self.refractory = cfg["l1_refractory_samples"]
        self.decay     = np.exp(-1.0 / tm)
        self.freeze    = cfg["l1_freeze_stdp"]

        # STDP params
        self.ltp       = cfg["l1_stdp_ltp"]
        self.ltp_win   = cfg["l1_stdp_ltp_window"]
        self.ltd       = cfg["l1_stdp_ltd"]
        w_lo, w_hi     = cfg["l1_w_bounds"]
        self.w_lo      = w_lo
        self.w_hi      = w_hi

        # Weights: [n_afferents × n_neurons]  uniform in [init_min, init_max]
        rng = np.random.default_rng(seed=7)
        self.W = rng.uniform(cfg["l1_init_w_min"], cfg["l1_init_w_max"],
                             (n_afferents, n)).astype(np.float64)

        # Neuron state
        self.n = n
        self.n_aff = n_afferents
        self.V = np.zeros(n, dtype=np.float64)
        self.last_post_spike = np.full(n, -9999, dtype=np.int64)
        self.last_pre_spike  = np.full(n_afferents, -9999, dtype=np.int64)
        self.spikes = np.zeros(n, dtype=bool)
        self.t = 0

    # ── public API ────────────────────────────────────────────────────────────
    def step(self, afferents: np.ndarray, dn_spike: bool) -> np.ndarray:
        """
        Advance one simulation step.
        Returns bool array of shape (n_neurons,) — which L1 neurons fired.
        """
        self.t += 1
        self.spikes[:] = False

        # DN excitation — boost ALL neurons
        if dn_spike:
            self.V = self.V * self.decay + self.dn_weight

        # Active afferents
        active = np.flatnonzero(afferents)
        if len(active) > 0:
            self.last_pre_spike[active] = self.t

            # Integrate: V_i += sum(W[active, i])  with decay
            # Vectorised: each neuron receives its own weighted sum
            input_current = self.W[active, :].sum(axis=0)  # [n_neurons]
            self.V = self.V * self.decay + input_current

        # Check thresholds — WTA: only the highest-V neuron above threshold
        above = self.V >= self.threshold
        # Enforce refractory period
        refractory_mask = (self.t - self.last_post_spike) <= self.refractory
        above &= ~refractory_mask

        if np.any(above):
            winner = int(np.argmax(self.V * above))  # highest V among eligible
            self.spikes[winner] = True
            self.last_post_spike[winner] = self.t

            # STDP on winner
            if not self.freeze:
                self._stdp(winner)

            # Hard reset
            self.V[winner] = self.reset_v

        return self.spikes.copy()

    def get_weights(self) -> np.ndarray:
        """Return a copy of the weight matrix [n_afferents × n_neurons]."""
        return self.W.copy()

    # ── STDP ──────────────────────────────────────────────────────────────────
    def _stdp(self, winner: int) -> None:
        """
        Asymmetric Hebbian STDP on the winning neuron's weights.
          - LTD: all synapses depressed by ltd at post-spike
          - LTP: afferents active within ltp_window before post-spike potentiated
        """
        w = self.W[:, winner]

        # Global LTD on all synapses
        w += self.ltd

        # LTP for recently active afferents
        dt = self.t - self.last_pre_spike
        causal = dt <= self.ltp_win
        # Also must have fired at least once (last_pre_spike > 0)
        causal &= self.last_pre_spike > 0
        w[causal] += self.ltp

        np.clip(w, self.w_lo, self.w_hi, out=w)
        self.W[:, winner] = w


# ═════════════════════════════════════════════════════════════════════════════
#  Network — Original binary-I/O SNN (unchanged from v1)
# ═════════════════════════════════════════════════════════════════════════════


class Network:
    def __init__(self, cfg: dict):
        n_in  = cfg["input_bit_width"]   # one input  neuron per bit
        n_h   = cfg["n_hidden"]
        n_out = cfg["output_bit_width"]  # one output neuron per bit

        # Weights — uniform, small, non-negative (STDP keeps W_in ≥ 0)
        rng   = np.random.default_rng(seed=42)
        scale = 0.5 / np.sqrt(n_in)
        self.W_in  = rng.uniform(0, scale, (n_in,  n_h )).astype(np.float32)
        self.W_out = rng.uniform(0, scale, (n_h,   n_out)).astype(np.float32)

        # LIF membrane potentials
        Vr = float(cfg["V_reset"])
        self.V     = np.full(n_h,   Vr, dtype=np.float32)
        self.V_out = np.full(n_out, Vr, dtype=np.float32)

        # Spike state (updated each step)
        self.spikes_h   = np.zeros(n_h,   dtype=bool)
        self.spikes_out = np.zeros(n_out,  dtype=bool)

        # STDP eligibility traces
        self.x_pre  = np.zeros(n_in, dtype=np.float32)
        self.x_post = np.zeros(n_h,  dtype=np.float32)

        self.cfg = cfg
        self.t   = 0  # global timestep counter

    # ── LIF step ──────────────────────────────────────────────────────────────
    def step(self, in_spikes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance one simulation timestep.

        in_spikes : bool array (input_bit_width,)
        Returns   : (hidden_spikes, output_spikes) — both bool arrays
        """
        cfg = self.cfg
        dt, tau = cfg["dt"], cfg["tau_m"]
        Vr, Vth = cfg["V_reset"], cfg["V_th"]

        # Hidden layer — LIF Euler update
        I_h            = self.W_in.T @ in_spikes.astype(np.float32)
        self.V        += (dt / tau) * (-(self.V - Vr) + I_h)
        self.spikes_h  = self.V >= Vth
        self.V[self.spikes_h] = Vr

        # Output layer — LIF Euler update
        I_out              = self.W_out.T @ self.spikes_h.astype(np.float32)
        self.V_out        += (dt / tau) * (-(self.V_out - Vr) + I_out)
        self.spikes_out    = self.V_out >= Vth
        self.V_out[self.spikes_out] = Vr

        self.t += 1
        return self.spikes_h, self.spikes_out

    # ── Reward-modulated STDP ─────────────────────────────────────────────────
    def stdp_update(self, in_spikes: np.ndarray, reward: float = 0.0):
        """
        One step of reward-modulated STDP on W_in.
        Traces always evolve; weights only change when |reward| > 0.
        """
        cfg = self.cfg
        dt  = cfg["dt"]

        # Exponential trace decay
        self.x_pre  *= np.exp(-dt / cfg["tau_pre"])
        self.x_post *= np.exp(-dt / cfg["tau_post"])

        # Accumulate traces at spike times
        self.x_pre [in_spikes]      += 1.0
        self.x_post[self.spikes_h]  += 1.0

        if reward == 0.0:
            return  # traces evolve but no weight change

        r = float(reward)
        # LTP: pre-trace × current post-spike  (causal pairings)
        self.W_in += r * cfg["A_plus"]  * np.outer(self.x_pre,
                                                     self.spikes_h.astype(np.float32))
        # LTD: current pre-spike × post-trace  (acausal pairings)
        self.W_in -= r * cfg["A_minus"] * np.outer(in_spikes.astype(np.float32),
                                                     self.x_post)
        np.clip(self.W_in, 0.0, cfg["W_max"], out=self.W_in)

    # ── Encoding / decoding ───────────────────────────────────────────────────
    @staticmethod
    def encode(value: int, bit_width: int) -> np.ndarray:
        """
        Rate encoding: bit k of `value` drives input neuron k.
          bit = 1  →  Bernoulli spike with p = 0.9
          bit = 0  →  Bernoulli spike with p = 0.1
        Returns bool array of shape (bit_width,).
        """
        bits  = np.array([(value >> i) & 1 for i in range(bit_width)], dtype=np.float32)
        probs = bits * 0.8 + 0.1          # map:  0 → 0.1,  1 → 0.9
        return np.random.rand(bit_width) < probs

    @staticmethod
    def decode(spikes_out: np.ndarray) -> int:
        """
        Bit-threshold decoding: output neuron k spikes → bit k = 1 in result.
        Returns the reconstructed integer.
        """
        return int(sum(int(s) << i for i, s in enumerate(spikes_out)))
