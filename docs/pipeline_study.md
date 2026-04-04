# How the Pipeline Actually Works

> A plain-English walkthrough for the scrutinising PI who wants to know
> exactly what happens between "electrode wiggle" and "control output" —
> without wading through 3 000 lines of Python.

---

## The One-Sentence Version

Raw voltage comes in at 80 kHz, gets bandpass-filtered and decimated to
20 kHz, encoded into a boolean spike pattern, run through an excitatory
attention gate and an inhibitory noise gate in parallel, fed to a
competitive spiking template layer that learns waveform shapes via STDP,
optionally refined by a 16-neuron decoder layer, and finally collapsed
into a scalar control signal with a confidence estimate.

That's it. Everything below is the "how" and the "why."

---

## 0 · Two-Phase Boot

The pipeline cannot be built in one shot. The encoder needs ~8 000
samples of raw noise before it knows how many amplitude bins to create,
and every downstream layer's input dimensionality depends on that number.

So construction is split:

| Phase | What gets built | Trigger |
|-------|-----------------|---------|
| **Early** | Preprocessor + Encoder | Immediately |
| **Late** | Attention, NoiseGate, Inhibitor, L1 Templates, DEC, Decoder | After encoder calibrates |

Until calibration finishes, the server just streams raw waveform data
to the browser with zeroed-out spike fields. No learning, no detection,
no control output. Patience.

---

## 1 · Preprocessing — Kill the Garbage, Keep the Spikes

**File:** `preprocessor.py`

Two operations, both optional but both on by default:

### 1.1 Bandpass Filter

A 2nd-order Butterworth IIR (SOS form) that passes 300–6 000 Hz. This
kills DC drift, slow LFP oscillations, and high-frequency thermal noise
in one step. The filter is causal — no future-peeking — and maintains
state across calls, so you can feed it one sample at a time without
artefacts.

Why SOS? Because direct transfer-function form is numerically unstable
for high-order IIR filters. SOS chains pairs of biquads. Textbook
stuff, but the kind of textbook stuff that saves you a week of debugging
when someone tries to run this at 80 kHz.

### 1.2 Decimation (÷4)

After bandpass removes everything above 6 kHz, we're massively
oversampled. Decimation by 4 drops us from 80 kHz → 20 kHz. Three out
of every four samples are simply discarded. The bandpass acts as the
anti-aliasing filter — no separate decimation filter needed.

**Net effect:** downstream processing runs at $f_s = 20\,\text{kHz}$, 
which is 4× less work per second.

---

## 2 · Spike Encoding — Turning Voltage into Binary Patterns

**File:** `encoder.py`

This is where the signal stops being a waveform and starts being a
spike code. The encoder builds a *temporal receptive field* — a grid of
amplitude bins × time delays — and at every timestep emits a flat
boolean vector saying which bins are active.

### 2.1 Calibration (First 8 000 Samples)

During calibration the encoder collects absolute sample values, then
computes the noise floor using the Median Absolute Deviation:

$$\hat{\sigma} = \frac{\text{median}(|x|)}{0.6745}$$

The 0.6745 factor converts MAD to a Gaussian-equivalent standard
deviation. We multiply by a tunable `dvm_factor` (default ≈ 2.06) to
get the detection voltage margin:

$$\text{dvm} = \texttt{dvm\_factor} \times \hat{\sigma}$$

This sets the half-width of each amplitude bin. The number of bins
depends on the signal's dynamic range during calibration — more range
means more bins means more afferents means a wider input vector for
everything downstream. That's why the pipeline is two-phase.

### 2.2 Amplitude Binning

Bins are placed at regular intervals spanning
$[\min(x) - \text{dvm},\;\max(x) + \text{dvm}]$ with spacing:

$$\Delta = \frac{2 \cdot \text{dvm}}{\texttt{overlap}}$$

The `overlap` parameter (default 9) controls how many bins a single
sample can activate — higher overlap gives finer amplitude resolution
but more afferents.

At each timestep, bin $i$ fires if:

$$|c_i - x_t| \leq \text{dvm}$$

where $c_i$ is the bin centre and $x_t$ is the current sample.

### 2.3 Shift Register (Temporal Delay Taps)

Bin activations are pushed into a 2-D shift register of shape
$[\texttt{n\_centres} \times \texttt{step\_size} \cdot \texttt{window\_depth}]$.
Only every `step_size`-th column is read out, giving `window_depth`
temporal snapshots.

The output — flattened to a 1-D boolean array — has length:

$$n_{\text{afferents}} = n_{\text{centres}} \times \texttt{window\_depth}$$

Typical values: ~227 centres × 10 delays = **2 270 afferents**.

Each afferent encodes "bin $i$ was active $k$ steps ago" — a
spatio-temporal code of the recent waveform shape.

---

## 3 · The Push–Pull Gate: Attention vs. Noise Gate

Two parallel pathways modulate the template layer's input. One says
"pay attention." The other says "ignore that." Together they form a
push–pull gain control that sharpens spike detection.

### 3.1 Attention Neuron (DN) — "Something Interesting Is Here"

**File:** `attention.py`

A single leaky integrate-and-fire (LIF) neuron with no learned weights.
Instead, each afferent contributes its *release probability* $p_{\text{rel}}$
as a dynamic weight:

$$p_{\text{rel},i}(t) = 1 - \bigl(1 - p_{\text{rel},i}^{-}\bigr) \cdot e^{-\Delta t / \tau_d}$$

After each pre-synaptic spike, the release probability is depressed:

$$p_{\text{rel},i}^{+} = p_{\text{rel},i} \cdot (1 - f_d)$$

This is short-term synaptic depression — the neuron habituates to
constant input (noise) but responds strongly to novel, structured input
(spikes). The membrane voltage integrates the sum of active release
probabilities and fires when it crosses threshold.

When DN fires, two things happen:
1. The template layer gets an excitatory current boost (`dn_weight ≈ 113`).
2. The decoder's confidence estimate goes up.

The inner loop is JIT-compiled via Numba with a pre-computed
$e^{-\Delta t/\tau_d}$ lookup table, because this runs 20 000 times per
second and Python is not invited to that party.

### 3.2 Noise Gate — "This Is Just Noise"

**File:** `noise_gate.py`

A 1-D Kalman filter that tracks instantaneous signal variance in real
time. The state is the estimated variance $\hat{\sigma}^2$; the
measurement is $x_t^2$ (squared sample amplitude).

Standard Kalman predict-update:

$$\hat{x}^- = \hat{x}, \quad P^- = P + Q$$
$$K = P^- / (P^- + R)$$
$$\hat{x} = \hat{x}^- + K(z - \hat{x}^-)$$

When the estimated $\sigma$ is close to the calibrated noise floor, the
gate emits a suppression factor $s \in [s_{\min}, 1]$ that scales down
all template-layer input current. An exponential moving average smooths
the output to prevent rapid toggling.

**In plain English:** if the signal looks like noise, throttle the
template layer so it doesn't hallucinate spikes.

---

## 4 · Global Inhibition — The Refractory Bouncer

**File:** `inhibition.py`

After any L1 neuron fires, a countdown timer suppresses *all* L1 input
for a blanking window (default 5 ms = 100 samples at 20 kHz). During
this window, input current is zeroed.

One exception: if the raw current magnitude exceeds a
`strength_threshold`, the signal breaks through anyway. Big spikes get
VIP treatment.

This prevents the template layer from double-counting the same spike
event across consecutive timesteps — a kind of network-level refractory
period.

The suppression from the noise gate and the inhibitor are multiplied
together into a single factor before being applied to the template
layer input:

$$I_{\text{effective}} = I_{\text{raw}} \times s_{\text{noise}} \times s_{\text{inhibition}}$$

---

## 5 · Template Layer (L1) — Learning What Spikes Look Like

**File:** `template.py`

This is the core of the sorter. 110 LIF neurons (via snnTorch), each
receiving the full 2 270-dimensional afferent vector through a learned
weight matrix $W \in \mathbb{R}^{n_{\text{aff}} \times 110}$.

### 5.1 Input Current

At each timestep:

$$I_j = \sum_{i \in \text{active}} W_{ij} + \begin{cases} w_{\text{DN}} & \text{if DN fired} \\ 0 & \text{otherwise} \end{cases}$$

The DN boost is a flat additive term (~113) shared across all neurons.
It doesn't help any particular neuron win — it just makes sure *someone*
fires when there's a real spike in the signal.

### 5.2 Winner-Take-All

snnTorch's `inhibition=True` flag implements WTA: only the neuron with
the highest membrane potential fires. If two neurons are close, tough
luck for the runner-up. This forces each neuron to specialise on a
different waveform template.

### 5.3 Competitive Asymmetric STDP

When a neuron wins:

- **LTD (global):** All its afferent weights decrease by a small amount
  ($\delta_{\text{LTD}} \approx -0.008$). This is the "forgetting" pressure.
- **LTP (causal):** Afferents that were recently active (within
  `ltp_window` steps) get a *larger* increase
  ($\delta_{\text{LTP}} \approx +0.016$). This is the "remember what
  caused me to fire" reinforcement.

Net effect: weights converge toward the spatio-temporal pattern that
most reliably triggers each neuron. Neurons that fire together wire
together; neurons that don't, slowly unwire.

Weights are clamped to $[0, 1]$. STDP is fully vectorised across all
simultaneous winners — no Python for-loop.

### 5.4 Refractory Period

Each neuron has a per-neuron refractory counter (default 1 sample).
A neuron that just fired cannot fire again until the counter expires.
This is enforced *after* the LIF forward pass by zeroing out
refractory-violating spikes.

---

## 6 · DEC Layer — "Which Unit Was That?"

**File:** `dec_layer.py`

Optional (enabled by `use_dec` flag). 16 LIF neurons that receive L1
spike vectors and learn to identify *which* putative neural unit just
fired.

### 6.1 Architecture

| Neuron | Role | Weights | STDP |
|--------|------|---------|------|
| 0 | Any-fire detector | Fixed = 1 (all L1 inputs) | None |
| 1–15 | Unit identity neurons | Random init, learned | Competitive asymmetric |

Neuron 0 is a pure presence detector — "did anything fire?" — gated by
DN activity. Neurons 1–15 compete via WTA to each claim a cluster of
correlated L1 responses.

### 6.2 DN Gating

The DEC layer only integrates when the DN attention neuron is active (or
within a configurable post-DN window, default ~2 ms). No DN activity →
no DEC integration → no output. This prevents noise-driven L1 leakage
from contaminating unit identification.

### 6.3 Optional Delay Expansion

When `use_delays=True`, L1 spike vectors are pushed into a shift
register of `n_delay_taps` entries, giving each DEC neuron a
spatio-temporal receptive field over recent L1 history. The flattened
input size becomes $n_{\text{L1}} \times n_{\text{taps}}$.

### 6.4 Output Format

A 16-bit boolean vector, also encoded as a hex bitmask (`uint16`). Bit
0 = any-fire, bits 1–15 = unit neurons. The hex word is what gets sent
over UDP to experiment hardware.

---

## 7 · Control Decoder — Turning Spikes into Actions

**File:** `decoder.py`

Takes the spike output (from L1 or DEC) and produces a scalar control
signal $c \in [-1, 1]$ plus a confidence $\gamma \in [0, 1]$.

### 7.1 Confidence

Always computed, regardless of strategy. It's the fraction of recent
timesteps where DN fired, tracked via a sliding window with $O(1)$
running sum:

$$\gamma = \frac{\sum_{k=t-W}^{t} \mathbb{1}[\text{DN fired at } k]}{W}$$

### 7.2 Strategies

| Strategy | Behaviour | Use Case |
|----------|-----------|----------|
| `discrete` | 1 if any spike ∧ DN active, else 0 | Simplest; event detection |
| `ttl` | Goes HIGH for a fixed-width pulse after spike+DN, then LOW | Hardware trigger lines |
| `rate` | Sliding-window spike rate, normalised to $[-1, 1]$ | Proportional control |
| `population` | Leaky integrator; emits on threshold crossing, then resets | Accumulate-to-threshold |
| `trigger` | Exponentially decaying pulse on spike+DN | Smooth impulse response |

### 7.3 UDP Output

If DEC is active, the hex bitmask is sent as a big-endian `uint16`. If
not, a `(control, confidence)` float pair is sent. Either way, it's a
single UDP datagram per event — fire and forget.

---

## 8 · Data Flow Summary

```
80 kHz raw electrode
  │
  ▼ Bandpass 300–6000 Hz + decimate ÷4
  │
  │  20 kHz filtered signal
  ▼
Encoder: amplitude bins × delay taps → bool[2270]
  │
  ├──────────────────┐
  ▼                  ▼
  DN (excite)        Noise Gate (inhibit)
  │ fires? bool      │ suppression ∈ [0,1]
  │                  │
  ├──────────────────┤
  ▼                  ▼
  Inhibitor: 5 ms blanking (strong signals bypass)
  │
  │  combined suppression factor
  ▼
  L1 Templates: 110 LIF + WTA + STDP → bool[110]
  │
  ▼ (optional)
  DEC: 16 LIF + DN gating → bool[16] / hex
  │
  ▼
  Decoder: strategy-dependent → (control, confidence)
  │
  ▼
  UDP datagram to hardware
```

---

## 9 · Numbers Worth Knowing

| Quantity | Typical Value | Where Set |
|----------|---------------|-----------|
| Raw sample rate | 80 kHz | Hardware / synthetic config |
| Effective sample rate | 20 kHz | After ÷4 decimation |
| Bandpass | 300–6 000 Hz | `PreprocessConfig` |
| Calibration window | 8 000 samples (0.4 s) | `EncoderConfig.noise_init_samples` |
| Amplitude bins | ~227 (signal-dependent) | Computed from dynamic range |
| Afferents | ~2 270 | `n_centres × window_depth` |
| L1 neurons | 110 | `L1Config.n_neurons` |
| DEC neurons | 16 (1 fixed + 15 learned) | Hardcoded in `DECLayer` |
| Inhibition blanking | 5 ms (100 samples) | `InhibitionConfig.duration_ms` |
| DN depression τ | 400 samples (20 ms) | `DNConfig.depression_tau` |
| STDP LTP / LTD | +0.016 / −0.008 | `L1Config.stdp_ltp/ltd` |

---

## 10 · What Could Go Wrong (and How to Tell)

| Symptom | Likely Cause | Diagnostic |
|---------|--------------|------------|
| No L1 spikes ever | Threshold unreachable | Check $V_{ss} < 0.8 \cdot \theta$; raise `dn_weight` |
| Every sample triggers L1 | Noise gate too permissive / DN too sensitive | Lower `dn_threshold_factor` or raise `ng_inhibit_below_sd` |
| All L1 neurons fire the same pattern | STDP too weak or WTA broken | Increase LTP/LTD ratio; check `inhibition=True` in snnTorch |
| DEC layer silent | DN not firing during real spikes | Check DN threshold; verify `dn_window_ms` is wide enough |
| Control signal jitters | Decoder strategy too twitchy | Switch to `ttl` or `population`; increase `window_ms` |

---

*Last updated from codebase state as of the commit you're reading this in.
If the code has moved on and this doc hasn't, blame the last person who
touched `pipeline.py` without updating `docs/`.*
