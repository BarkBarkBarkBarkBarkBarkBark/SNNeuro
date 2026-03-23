# Scientific Principles — SNN Agent

> This document is the complete mathematical and scientific reference for the
> SNN Agent pipeline. Each component includes a plain-English explanation,
> pseudocode, formal equations, the scientific basis with literature citations,
> and an implementation audit.
>
> **For the accessible overview**, see the [README](../README.md).
> **Source code**: [`src/snn_agent/core/`](../src/snn_agent/core/)

---

## Table of Contents

1. [Signal Preprocessing](#1-signal-preprocessing)
2. [Spike Encoding (Temporal Receptive Field)](#2-spike-encoding-temporal-receptive-field)
3. [Attention Neuron (Detection Neuron)](#3-attention-neuron-detection-neuron)
4. [Template Matching (L1 Layer)](#4-template-matching-l1-layer)
5. [Synaptic Plasticity (STDP)](#5-synaptic-plasticity-stdp)
6. [Noise Gate (Kalman Filter)](#6-noise-gate-kalman-filter)
7. [Global Inhibition](#7-global-inhibition)
8. [Classification Layer (L2)](#8-classification-layer-l2)
9. [Control Decoder](#9-control-decoder)
10. [References](#10-references)
11. [Scientific Claims Audit Summary](#11-scientific-claims-audit-summary)

---

## 1. Signal Preprocessing

### In plain English

Before the neural network can see the signal, we need to clean it up. Raw
electrode recordings contain frequencies we don't care about — slow drifts
from breathing, high-frequency electrical noise from equipment. The
preprocessor acts like a selective filter: it keeps only the frequency band
where neural spikes live (300 Hz to 6,000 Hz) and throws everything else
away. It also reduces the data rate by a factor of 4 (e.g., 80,000 → 20,000
samples per second) to make the downstream processing feasible in real time.

### How it works

```
for each raw_sample:
    filtered = apply_bandpass(raw_sample, lo=300 Hz, hi=6000 Hz)
    decimation_counter += 1
    if decimation_counter == 4:
        decimation_counter = 0
        emit filtered
    else:
        discard (filtered but not emitted)
```

### The mathematics

The bandpass filter uses second-order sections (SOS) form of an IIR
Butterworth filter:

$$H(z) = \prod_{k=1}^{K} \frac{b_{0k} + b_{1k}z^{-1} + b_{2k}z^{-2}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

where $K$ = `bandpass_order` (default 2). The SOS representation avoids the
numerical instability of the direct transfer-function (b, a) form.

Filter state is carried across sample-by-sample calls via `sosfilt` with
persistent `zi`, making it **causal** and **streaming-safe**.

### Scientific basis

✅ **Sound.** Using SOS form avoids the numerical instability of
transfer-function form for higher-order filters. Carrying the filter state
across calls ensures correct streaming operation — this is a textbook
implementation of real-time IIR filtering. The 300–6,000 Hz passband is
standard for extracellular spike sorting (Quiroga et al. 2004).

### Implementation

- **File:** `core/preprocessor.py` → `Preprocessor`
- **Config:** `PreprocessConfig` — `bandpass_lo_hz`, `bandpass_hi_hz`,
  `bandpass_order`, `decimation_factor`

---

## 2. Spike Encoding (Temporal Receptive Field)

### In plain English

The encoder converts the continuous electrode signal into a pattern of
on/off activations — like turning a waveform into a barcode. It divides the
amplitude range into overlapping bins (think of them as "sensitivity bands")
and records which bands are active at each moment. These activations are fed
into a shift register that remembers the last several timesteps, creating a
two-dimensional snapshot: amplitude × time. This snapshot is what the
downstream neurons "see" — it's how the network perceives each moment of the
signal.

Before any of this can work, the encoder needs to understand the noise level.
It spends the first 8,000 samples measuring background noise using a robust
statistical method (MAD), then sets up its bins accordingly. This means the
system automatically adapts to different electrode impedances, amplifier
gains, and noise environments.

### How it works

```
# Calibration phase (first 8000 samples)
collect |sample| values
noise_sigma = median(|samples|) / 0.6745    # robust noise estimate
bin_halfwidth = dvm_factor × noise_sigma
bin_spacing = 2 × bin_halfwidth / overlap
centres = range(signal_min - bin_halfwidth, signal_max + bin_halfwidth, bin_spacing)

# Encoding phase (every sample after calibration)
for each sample:
    active_bins = [j for j in centres if |centre_j - sample| ≤ bin_halfwidth]
    shift_register.push(active_bins)
    afferent_vector = shift_register.flatten()    # [n_centres × window_depth] booleans
    emit afferent_vector
```

### The mathematics

**Noise estimation** — Median Absolute Deviation during calibration:

$$\hat{\sigma} = \frac{\mathrm{median}(|x_1|, |x_2|, \dots, |x_N|)}{0.6745}$$

where $N$ = `encoder.noise_init_samples`. The divisor 0.6745 converts MAD
to a consistent estimator of standard deviation for Gaussian noise.

**Amplitude bin half-width:**

$$\Delta v_m = \gamma \cdot \hat{\sigma}$$

where $\gamma$ = `encoder.dvm_factor`. Bin centres are spaced by
$\frac{2 \Delta v_m}{O}$ with $O$ = `encoder.overlap`.

**Afferent activation** — bin $j$ fires when:

$$|c_j - x_t| \leq \Delta v_m$$

The afferent vector is the flattened shift register
$\mathbf{a} \in \{0,1\}^{N_c \times D}$ where $N_c$ is the number of centres
and $D$ = `encoder.window_depth`.

### Scientific basis

✅ **Sound.** The MAD scaled by $1/\Phi^{-1}(3/4) \approx 1/0.6745$ is a
well-established consistent estimator of $\sigma$ under Gaussian noise.
It is robust to outliers (spike waveforms), which is exactly the motivation:
the spikes are the "outliers" the estimator should ignore.

The temporal receptive field approach converts a 1-D continuous signal into a
2-D binary pattern (amplitude × time), enabling pattern matching via
downstream STDP. Analogous to cochlear-style place coding, where each "bin"
acts as a frequency/amplitude channel.

- Donoho & Johnstone (1994) — *Ideal spatial adaptation by wavelet shrinkage.*
  Introduced MAD / 0.6745 for denoising.
- Quiroga, Nadasdy & Ben-Shaul (2004) — *Unsupervised spike detection and
  sorting with wavelets and superparamagnetic clustering.* Uses the identical
  MAD formula for extracellular noise estimation.
- Thorpe, Delorme & Van Rullen (2001) — *Spike-based strategies for rapid
  processing.* Establishes the framework of rank-order and first-spike coding.

⚠️ **Caveat:** A post-calibration EMA tracks `|x_t|` but is never used
downstream — `dvm` and `centres` are fixed at calibration time. Consider
removing the unused EMA or wiring it for adaptive bin updates.

### Implementation

- **File:** `core/encoder.py` → `SpikeEncoder`
- **Config:** `EncoderConfig` — `overlap`, `dvm_factor`, `step_size`,
  `window_depth`, `noise_init_samples`
- **Key state:** `n_afferents` is only known *after* calibration. This is why
  the pipeline factory is two-phase.

---

## 3. Attention Neuron (Detection Neuron)

### In plain English

The attention neuron is like a lookout. It watches the encoded signal and
fires whenever something "interesting" happens — meaning the signal energy
rises above the background noise. When it fires, it tells the template layer
"pay attention now!" by boosting the input to all template neurons.

To avoid crying wolf on every bit of noise, it uses a biological trick called
**synaptic depression**: each time a synapse contributes to a spike, it gets
temporarily weakened (like a muscle getting tired). This means the neuron
naturally ignores sustained, repetitive input (noise) and responds only to
sudden changes in energy (real spikes).

### How it works

```
for each afferent_vector:
    # Recover synapses that haven't fired recently
    for each active afferent j:
        p_rel[j] = 1 - (1 - p_rel_last[j]) × exp(-Δt / τ_depression)
        contribution += p_rel[j]
        p_rel[j] *= (1 - depression_fraction)    # deplete after use

    membrane = membrane × decay + contribution
    if membrane ≥ threshold:
        fire!
        membrane = reset_potential
```

### The mathematics

**Membrane dynamics** (leaky integrate-and-fire):

$$V_t = V_{t-1} \cdot e^{-1/\tau_m} + \sum_{j \in \mathcal{A}_t} p_{\mathrm{rel},j}(t)$$

where $\mathcal{A}_t$ is the set of active afferents at time $t$ and
$\tau_m$ = `dn.tm_samples`.

**Synaptic release probability** (short-term depression):

$$p_{\mathrm{rel},j}(t) = 1 - \left(1 - p_{\mathrm{rel},j}(t_{\mathrm{last}})\right) e^{-\Delta t / \tau_d}$$

After contributing, each synapse is depressed:

$$p_{\mathrm{rel},j} \leftarrow p_{\mathrm{rel},j} \cdot (1 - f_d)$$

where $\tau_d$ = `dn.depression_tau` and $f_d$ = `dn.depression_frac`.

**Threshold:**

$$\theta_{\mathrm{DN}} = \frac{F_{\mathrm{th}} \cdot O \cdot D}{1 - e^{-1/\tau_m}}$$

where $F_{\mathrm{th}}$ = `dn.threshold_factor`. This computes the
steady-state membrane potential under maximum sustained input, scaled by the
threshold factor.

**Reset** (set, not additive):

$$V \leftarrow V_{\mathrm{reset}} = F_r \cdot \left(e^{1/\tau_m} - 1\right) \cdot \theta_{\mathrm{DN}}$$

where $F_r$ = `dn.reset_potential_factor`. This provides a small positive
"memory" after firing, biasing the neuron to re-fire if the signal persists.

### Scientific basis

✅ **Sound.** The synaptic depression model is a simplified version of
Tsodyks & Markram (1997). The full T–M model tracks three variables
(utilisation $u$, available resources $x$, post-synaptic current $I$). SNN
Agent collapses this into a single variable $p_{\mathrm{rel}}$ that recovers
exponentially and is depleted multiplicatively. This is a valid simplification
when the utilisation parameter $U$ is fixed.

- Tsodyks & Markram (1997) — *The neural code between neocortical pyramidal
  neurons depends on neurotransmitter release probability.* PNAS 94(2):
  719–723.

⚠️ **Caveat on threshold:** The formula computes a steady-state threshold
from the input dimensionality. The denominator $1 - e^{-1/\tau_m}$ is the
geometric series sum factor for a leaky integrator.

⚠️ **Caveat on reset:** The precise derivation of the reset formula is not
documented in any reference. It appears to be an empirical choice from the
MB2018 work. The behaviour is reasonable — it implements a soft reset.

### Implementation

- **File:** `core/attention.py` → `AttentionNeuron`
- **Config:** `DNConfig` — `threshold_factor`, `depression_tau`,
  `depression_frac`, `tm_samples`, `reset_potential_factor`
- **Note:** Pre-computed exponential LUT (`_exp_td`) for fast depression
  recovery, truncated at $20\tau_d$ (negligible error: $e^{-20} \approx 2 \times 10^{-9}$).

---

## 4. Template Matching (L1 Layer)

### In plain English

This is the heart of the spike sorter. 110 spiking neurons compete to become
"template matchers" — each one learns to recognise a different spike
waveform shape. When a waveform arrives that matches a neuron's template, that
neuron fires and claims the spike. A winner-take-all rule ensures only the
best-matching neuron fires, preventing double-counting.

Think of it like having 110 lock-and-key detectors. Over time, each detector
reshapes its "lock" to perfectly fit one type of "key" (spike waveform).
When that waveform appears, the matching detector fires and the others stay
quiet.

The attention neuron's signal acts as a "boost" — when the attention neuron
says "something interesting is here," all template neurons get extra input
current, making them more likely to fire and learn.

### How it works

```
for each afferent_vector, dn_spike:
    # Compute input current for all neurons
    current = afferents @ W + (dn_spike × dn_weight)
    current *= suppression    # from noise gate + inhibitor

    # LIF membrane update + WTA
    membrane = β × membrane + current
    winner = argmax(membrane) if max(membrane) ≥ threshold
    fire winner, reset its membrane to 0

    # STDP learning on the winner's weights
    winner.weights += LTD    # global depression (all synapses)
    winner.weights[recently_active] += LTP    # potentiate causal synapses
    clip weights to [w_min, w_max]
```

### The mathematics

**Input current:**

$$I_t = \mathbf{a}_t^{\top} \mathbf{W} + s_{\mathrm{DN}}(t) \cdot w_{\mathrm{DN}}$$

where $\mathbf{W} \in \mathbb{R}^{N_a \times N}$ are the learnable weights
and $w_{\mathrm{DN}}$ = `l1.dn_weight`.

**LIF membrane** (via snnTorch):

$$V_{i,t} = \beta \, V_{i,t-1} + I_{i,t}, \qquad \beta = e^{-1/\tau_m}$$

where $\tau_m$ = `l1.tm_samples`.

**Spike and reset:**

$$s_{i,t} = \begin{cases} 1 & \text{if } V_{i,t} \geq \theta_{L1} \\ 0 & \text{otherwise} \end{cases}, \qquad V_{i,t} \leftarrow 0 \;\text{ on spike (WTA)}$$

**L1 threshold derivation:**

$$\theta_{L1} = \frac{\left(w_{\mathrm{DN}} + O \cdot (D - k)\right) \left(1 - e^{-S/\tau_m}\right)}{1 - e^{-1/\tau_m}}$$

where $k = 3$ (spike waveform half-width in delay taps) and
$S$ = `encoder.step_size`. The numerator models the expected peak current
when a coherent spike pattern fills the receptive field. The factor
$(1 - e^{-S/\tau_m})$ accounts for temporal subsampling.

**Winner-take-all:** snnTorch's `inhibition=True` — only the neuron with the
highest membrane potential fires per timestep.

### Scientific basis

✅ **Sound.** The competitive spiking network with WTA is a standard
architecture for unsupervised pattern recognition. The threshold derivation
is principled — it models the steady-state membrane under expected spike
input.

- Oster, Douglas & Liu (2009) — *Computation with spikes in a winner-take-all
  network.* Neural Computation 21(9): 2437–2465.

⚠️ **Caveat:** snnTorch marks `inhibition=True` as "unstable." The behaviour
is functionally correct for template matching. The constant $k = 3$ is
hardcoded; for data with substantially different waveform morphology, it may
need tuning.

### Implementation

- **File:** `core/template.py` → `TemplateLayer`
- **Config:** `L1Config` — `n_neurons`, `dn_weight`, `tm_samples`,
  `refractory_samples`, `init_w_min`, `init_w_max`
- **Weight matrix:** `W` shape `[n_afferents, n_neurons]` (torch.Tensor)

---

## 5. Synaptic Plasticity (STDP)

### In plain English

The template neurons aren't programmed with specific waveform shapes — they
**learn** them. Every time a neuron fires, its connections are adjusted using
a learning rule called STDP (spike-timing-dependent plasticity).

The rule has two parts:
- **Weakening (LTD):** When a neuron fires, *all* of its input connections
  get slightly weaker. This prevents the neuron from responding to everything.
- **Strengthening (LTP):** Input connections that were recently active
  (i.e., contributed to making the neuron fire) get strengthened. This
  reinforces the pattern that triggered the spike.

Over time, this competitive process causes each neuron to specialise: it
strengthens connections to the waveform features it responds to best and
weakens connections to everything else. Combined with winner-take-all, each
neuron ends up "owning" a distinct waveform template.

### How it works

```
function stdp(winner_neuron):
    # Weaken all connections (global LTD)
    weights[winner] += η⁻    # η⁻ is negative

    # Strengthen recently-active connections (causal LTP)
    for each afferent j:
        if (now - last_pre_spike[j]) ≤ LTP_window:
            weights[winner][j] += η⁺    # η⁺ is positive

    clip weights to [w_min, w_max]
```

### The mathematics

**LTD** — global depression at every post-synaptic spike:

$$\Delta W_{j,w}^{-} = \eta^{-} \quad \forall\; j$$

**LTP** — potentiation for recently active pre-synaptic afferents:

$$\Delta W_{j,w}^{+} = \eta^{+} \quad \text{if } (t - t_j^{\mathrm{pre}}) \leq T_{\mathrm{LTP}}$$

**Weight bounds:**

$$W_{j,w} \in [w_{\min},\; w_{\max}]$$

where $\eta^{+}$ = `l1.stdp_ltp`, $\eta^{-}$ = `l1.stdp_ltd`,
$T_{\mathrm{LTP}}$ = `l1.stdp_ltp_window`.

### Scientific basis

✅ **Sound.** This is a well-established competitive STDP rule. The global
LTD ensures unused synapses decay, while causal LTP reinforces afferents that
fired shortly before the post-synaptic spike. Combined with WTA, this drives
specialisation — effectively implementing unsupervised template matching.

- Masquelier, Guyonneau & Thorpe (2008) — *Spike timing dependent plasticity
  finds the start of repeating patterns in continuous spike trains.* PLoS ONE
  3(1): e1377. Uses the same global-LTD / causal-LTP asymmetric STDP rule.
- Song, Miller & Abbott (2000) — *Competitive Hebbian learning through
  spike-timing-dependent synaptic plasticity.* Nature Neuroscience 3(9):
  919–926. Foundational work on competitive STDP.

### Implementation

- **File:** `core/template.py` → `TemplateLayer._stdp()`
- **Config:** `L1Config` — `stdp_ltp`, `stdp_ltd`, `stdp_ltp_window`,
  `w_lo`, `w_hi`, `freeze_stdp`
- **Note:** The guard `last_pre_spike > 0` correctly prevents LTP on
  afferents that have never fired.

---

## 6. Noise Gate (Kalman Filter)

### In plain English

The noise gate is the inhibitory counterpart to the attention neuron. While
the attention neuron says *"something interesting is here — boost the signal,"*
the noise gate says *"this is just background noise — suppress the signal."*

It works by continuously estimating the signal's variance using a Kalman
filter (a mathematical tool for tracking a changing quantity from noisy
measurements). When the estimated variance is close to the known noise
baseline, the gate suppresses the template layer's input. When a real spike
appears and pushes the variance above baseline, the gate opens up and lets
the signal through.

Together, the attention neuron and noise gate provide a push–pull modulation:
excitation during spikes, suppression during noise. This dramatically
improves the signal-to-noise ratio.

### How it works

```
for each sample:
    measurement = sample²    # instantaneous power

    # Kalman filter predict + update
    predicted_variance = estimated_variance + process_noise
    innovation = measurement - predicted_variance
    kalman_gain = predicted_variance / (predicted_variance + measurement_noise)
    estimated_variance += kalman_gain × innovation

    # Gating decision
    estimated_sigma = √estimated_variance
    if estimated_sigma < threshold × noise_baseline_sigma:
        suppression = interpolate(suppression_factor, 1.0)    # suppress
    else:
        suppression = 1.0    # pass through

    suppression = EMA_smooth(suppression)    # avoid rapid toggling
    emit suppression
```

### The mathematics

**State model:** 1-D Kalman filter tracking signal variance ($\sigma^2$).

- State: $x_t = \hat{\sigma}^2_t$ (estimated variance)
- Measurement: $z_t = x_t^2$ (squared sample = instantaneous power)
- Process noise: $Q$ = `noise_gate.process_noise`
- Measurement noise: $R$ = `noise_gate.measurement_noise`

**Predict:**

$$\hat{x}_{t|t-1} = \hat{x}_{t-1}, \quad P_{t|t-1} = P_{t-1} + Q$$

**Update:**

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}, \quad \hat{x}_t = \hat{x}_{t|t-1} + K_t (z_t - \hat{x}_{t|t-1}), \quad P_t = (1 - K_t) P_{t|t-1}$$

**Gating:** When $\sqrt{\hat{x}_t} < \alpha \cdot \sigma_{\mathrm{noise}}$
(where $\alpha$ = `inhibit_below_sd`), apply suppression via linear
interpolation between `suppression_factor` and 1.0.

### Scientific basis

✅ **Sound.** The 1-D Kalman filter is optimal for tracking a scalar state
under Gaussian noise. Using it for variance estimation is a pragmatic
engineering choice. The push–pull modulation with the attention neuron is
analogous to excitatory/inhibitory balance in cortical circuits.

### Implementation

- **File:** `core/noise_gate.py` → `NoiseGateNeuron`
- **Config:** `NoiseGateConfig` — `process_noise`, `measurement_noise`,
  `inhibit_below_sd`, `suppression_factor`
- **Baseline:** Uses the encoder's MAD-calibrated `noise_sigma` as the
  reference noise floor.

---

## 7. Global Inhibition

### In plain English

After any template neuron fires, the global inhibitor temporarily blocks all
input to the template layer for a short period (default 5 ms). This prevents
the same spike waveform from triggering multiple neurons in rapid succession
— it's like a "cooldown" period.

However, if a very strong signal arrives during the cooldown, it breaks
through the block. This ensures that closely spaced, high-amplitude spikes
are not missed.

This mechanism mirrors how inhibitory interneurons in the brain create
refractory periods and enforce temporal separation between neural responses.

### How it works

```
for each timestep:
    if any_neuron_fired_last_step:
        start countdown = blanking_samples    # 5 ms worth

    if countdown > 0:
        countdown -= 1
        if current_magnitude ≥ strength_threshold:
            pass through (strong-signal bypass)
        else:
            suppress all input (factor = 0.0)
    else:
        pass through (factor = 1.0)
```

### The mathematics

**Blanking window:** $T_{\mathrm{blank}} = \lfloor \Delta t_{\mathrm{ms}} \times 10^{-3} \times f_s \rfloor$ samples.

**Gate function:**

$$g(t) = \begin{cases}
1.0 & \text{if countdown} = 0 \\
1.0 & \text{if countdown} > 0 \;\wedge\; |I_t| \geq \theta_{\mathrm{str}} \\
0.0 & \text{if countdown} > 0 \;\wedge\; |I_t| < \theta_{\mathrm{str}}
\end{cases}$$

### Scientific basis

✅ **Sound.** Post-spike inhibition is a well-established mechanism in
cortical circuits, mediated by inhibitory interneurons. The strong-signal
bypass is a practical engineering choice that prevents the inhibitor from
masking genuinely distinct, high-amplitude spikes that happen to occur
during the blanking window.

### Implementation

- **File:** `core/inhibition.py` → `GlobalInhibitor`
- **Config:** `InhibitionConfig` — `enabled`, `duration_ms`,
  `strength_threshold`

---

## 8. Classification Layer (L2)

### In plain English

The optional L2 layer sits on top of the template layer and learns to group
template neurons into clusters that represent the same neural unit. While L1
neurons specialise on individual waveform *phases* or *shapes*, L2 neurons
learn to recognise that certain sets of L1 responses always occur together
and therefore belong to the same neuron.

Think of L1 as recognising individual words and L2 as recognising sentences —
it spots patterns in the patterns.

This layer uses the same competitive learning (STDP + WTA) as L1, plus
stronger lateral inhibition to enforce sharper competition between the 10
output neurons.

### How it works

```
for each l1_spike_vector:
    current = l1_spikes @ W_l2    # feed-forward
    # Amplify competition via lateral inhibition factor
    current = mean(current) + (current - mean(current)) × wi_factor

    membrane = β × membrane + current
    winner = WTA(membrane, threshold)
    if winner: apply STDP to winner's weights
```

### Scientific basis

The L2 architecture mirrors the MATLAB ANNet Output_Layer (10 neurons,
`wiFactor=10`). Lateral inhibition via amplified WTA competition is a
standard mechanism for forcing discrete categorisation in neural networks.

### Implementation

- **File:** `core/output_layer.py` → `ClassificationLayer`
- **Config:** `L2Config` — `n_neurons`, `wi_factor`, `threshold_factor`,
  STDP params
- **Feature flag:** `Config.use_l2 = True` to enable

---

## 9. Control Decoder

### In plain English

The decoder converts the sorted spike activity into a single control signal
that can drive experiment hardware — for example, triggering a stimulator
or adjusting a cursor position. It also computes a confidence score based on
how active the attention neuron has been recently (high attention activity =
high confidence that real spikes are being detected).

Three decoding strategies are available:

- **Rate:** Counts spikes in a sliding time window — more spikes = stronger
  control signal. Good for graded, proportional control.
- **Population:** A leaky accumulator that builds up with each spike and
  fires when it crosses a threshold. Good for detecting bursts.
- **Trigger:** Simple binary pulse whenever a spike occurs and the attention
  neuron is active. Good for discrete event detection.

### How it works

```
# Rate strategy
spike_counts = sliding_window_count(spikes, window_ms)
control = clip(weights · spike_rates, -1, 1)

# Population strategy
integrator = integrator × decay + weights · spikes
if integrator ≥ threshold:
    emit control signal, reset integrator

# Trigger strategy
if any_spike AND dn_active:
    emit 1.0
```

### The mathematics

**Rate strategy** — sliding-window weighted sum:

$$u_t = \mathrm{clip}\!\left(\sum_{i=1}^{N} w_i \cdot \frac{n_{i,t}}{T_w},\; -1,\; 1\right)$$

where $n_{i,t}$ counts spikes of neuron $i$ over window length $T_w$.

**Population strategy** — leaky integrator with reset:

$$\mathcal{I}_t = \mathcal{I}_{t-1} \cdot e^{-1/\tau_c} + \mathbf{w}^{\top} \mathbf{s}_t$$

Emits when $\mathcal{I}_t \geq \theta_c$, then resets $\mathcal{I}_t \leftarrow 0$.

**Trigger strategy** — binary pulse:

$$u_t = \begin{cases} 1 & \text{if } \exists\, i : s_{i,t}=1 \;\wedge\; s_{\mathrm{DN}}(t)=1 \\ \mathrm{None} & \text{otherwise} \end{cases}$$

**DN confidence** — sliding-window attention activity:

$$\mathrm{conf}_t = \frac{\sum_{k} s_{\mathrm{DN}}(k)}{\min(t,\; T_{\mathrm{dn}})}$$

where $T_{\mathrm{dn}}$ = `decoder.dn_confidence_window_ms`.

### Scientific basis

✅ **Sound.** Rate-coded readouts and leaky integrators are standard motor
cortex decoding models.

⚠️ **Caveat on confidence:** The DN confidence is a heuristic fraction, not a
formal Bayesian posterior. During warm-up (`len(buf) < T_dn`), the
denominator grows from 1, making early values noisy.

### Implementation

- **File:** `core/decoder.py` → `ControlDecoder`
- **Config:** `DecoderConfig` — `strategy`, `window_ms`, `threshold`,
  `leaky_tau_ms`, `dn_confidence_window_ms`

---

## 10. References

1. Donoho, D.L. & Johnstone, I.M. (1994). *Ideal spatial adaptation by
   wavelet shrinkage.* Biometrika 81(3): 425–455.

2. Quiroga, R.Q., Nadasdy, Z. & Ben-Shaul, Y. (2004). *Unsupervised spike
   detection and sorting with wavelets and superparamagnetic clustering.*
   Neural Computation 16(8): 1661–1687.

3. Thorpe, S., Delorme, A. & Van Rullen, R. (2001). *Spike-based strategies
   for rapid processing.* Neural Networks 14(6–7): 715–725.

4. Tsodyks, M.V. & Markram, H. (1997). *The neural code between neocortical
   pyramidal neurons depends on neurotransmitter release probability.* PNAS
   94(2): 719–723.

5. Masquelier, T., Guyonneau, R. & Thorpe, S. (2008). *Spike timing
   dependent plasticity finds the start of repeating patterns in continuous
   spike trains.* PLoS ONE 3(1): e1377.

6. Song, S., Miller, K.D. & Abbott, L.F. (2000). *Competitive Hebbian
   learning through spike-timing-dependent synaptic plasticity.* Nature
   Neuroscience 3(9): 919–926.

7. Oster, M., Douglas, R. & Liu, S.C. (2009). *Computation with spikes in a
   winner-take-all network.* Neural Computation 21(9): 2437–2465.

---

## 11. Scientific Claims Audit Summary

Every major scientific claim in the codebase has been evaluated against the
literature and the implementation. Full details of each audit are inline
above.

| # | Component | Claim | Verdict |
|---|---|---|---|
| 1 | Encoder | MAD / 0.6745 noise estimator | ✅ Sound |
| 2 | Encoder | Temporal receptive field encoding | ✅ Sound |
| 3 | DN | LIF + Tsodyks-Markram synaptic depression | ✅ Sound |
| 4 | DN | Threshold = steady-state formula | ✅ Sound |
| 5 | DN | Reset potential formula | ⚠️ Derivation undocumented |
| 6 | L1 | Threshold derivation (k=3) | ✅ Sound (k hardcoded) |
| 7 | L1 | Asymmetric STDP (global LTD + causal LTP) | ✅ Sound |
| 8 | L1 | WTA via snnTorch inhibition | ⚠️ snnTorch stability caveat |
| 9 | Decoder | Rate — sliding-window weighted sum | ✅ Sound |
| 10 | Decoder | Population — leaky integrator | ✅ Sound |
| 11 | Decoder | DN confidence = sliding-window average | ⚠️ Heuristic, not Bayesian |
| 12 | Pipeline | Biologically-inspired architecture | ✅ Sound (engineering abstraction) |
| 13 | L1 | snnTorch LIF zero reset mechanism | ✅ Sound |
| 14 | Preprocessor | SOS causal bandpass filter | ✅ Sound |

**Overall assessment:** The scientific foundations are solid. All equations
match the code implementation. No claims require correction.
