# Neuron & Component Guide — Plain English

> **Purpose:** Explain what every neuron type and processing stage in the SNN
> spike-sorting pipeline actually does, what each hyperparameter controls, and
> how the pieces wire together. No jargon without a definition, no maths
> without a plain-English translation.

---

## Table of Contents

1. [Spike Encoder (Input Layer)](#1-spike-encoder-input-layer)
2. [Attention Neuron (DN)](#2-attention-neuron-dn)
3. [Noise Gate Neuron](#3-noise-gate-neuron)
4. [Global Inhibitor](#4-global-inhibitor)
5. [Template Layer (L1)](#5-template-layer-l1)
6. [DEC Layer (Output)](#6-dec-layer-output)
7. [How They Wire Together](#7-how-they-wire-together)

---

## 1. Spike Encoder (Input Layer)

**File:** `src/snn_agent/core/encoder.py`

### What it does

The encoder converts the raw voltage signal (after bandpass filtering and
decimation to 20 kHz) into a grid of binary spikes that the downstream spiking
neurons can consume.

It works in **two phases:**

#### Phase 1 — Calibration (first 8 000 samples ≈ 0.4 s)

The encoder watches the incoming signal and measures how "noisy" it is by
computing the **Median Absolute Deviation (MAD)** — a robust measure of
spread that isn't thrown off by spike outliers the way standard deviation
would be. From that MAD it sets each receptive-field centre's threshold:
a bin fires only when the signal deviates far enough from the median.

Think of it as the encoder "listening to the room" before deciding what
counts as a real signal versus background noise.

#### Phase 2 — Encoding (all subsequent samples)

Once calibrated, the encoder tiles a set of overlapping receptive-field
centres across the signal's amplitude range. At every time step each
centre checks whether the current sample falls inside its sensitive zone.
If it does, a `True` is pushed into that centre's shift register (a short
memory window of the last few time steps). The output is a 2-D boolean
array: `[n_centres, window_depth]` — one row per centre, one column per
recent time step.

This is the **afferent spike vector** that feeds into the Template Layer,
the Attention Neuron, and the Noise Gate simultaneously.

### Hyperparameters

| Parameter | Config key | Default | What it does |
|-----------|-----------|---------|--------------|
| **Number of centres** | `n_centres` | 20 | How many amplitude bins tile the signal range. More centres = finer amplitude resolution but more afferents for L1 to handle. |
| **Overlap** | `overlap` | 8 | How many neighbouring centres share sensitivity. Higher overlap means each voltage value activates more centres simultaneously, giving the network a richer (but noisier) input. |
| **Window depth** | `window_depth` | 3 | How many past time steps each centre remembers. Deeper windows let the network see short temporal patterns but increase the total afferent count (`n_centres × window_depth`). |
| **Step size** | `step_size` | 2 | Stride between adjacent centres in amplitude space. Smaller steps pack centres more tightly. |
| **dvm factor** | `dvm_factor` | 1.0 | Scales the MAD-derived threshold that each centre uses to decide "active or not." Larger values raise the bar, so only stronger deviations produce spikes. |
| **Calibration samples** | `calibration_samples` | 8 000 | How many samples the encoder watches before locking in its thresholds. |

---

## 2. Attention Neuron (DN)

**File:** `src/snn_agent/core/attention.py`

### What it does

The Attention Neuron (called **DN** for "Deviation Neuron") acts as an
**energy detector / novelty gate**. It watches the afferent spike vector
from the encoder and fires when it sees an unusually strong burst of
activity — the kind you'd expect from a real neural spike rather than
background noise.

It is a single **Leaky Integrate-and-Fire (LIF)** neuron with one extra
trick: **pRel (probabilistic release) depression**. Every time the DN
fires, its ability to fire again is temporarily reduced (its "release
probability" drops). This prevents the DN from machine-gunning during a
sustained spike and ensures it fires once per event.

The depression model is a simplified **Tsodyks–Markram** synapse: after
each firing the release probability drops by a fraction
(`depression_frac`), then recovers exponentially back to 1.0 with a time
constant of `depression_tau` samples.

**When the DN fires, two things happen:**

1. The Template Layer (L1) gets a strong excitatory boost, making it much
   more likely to recognise the incoming pattern as a spike.
2. The DEC Layer's confidence counter increments, contributing to the
   output control signal.

Think of the DN as a bouncer at a club: it only lets strong, novel
events through the door, and after letting one in it takes a breather
before admitting the next.

### Hyperparameters

| Parameter | Config key | Default | What it does |
|-----------|-----------|---------|--------------|
| **Threshold factor** | `threshold_factor` | 0.85 | Multiplied by the encoder's MAD-derived noise estimate to set the DN's firing threshold. Lower = more sensitive (fires on weaker events). |
| **DN weight to L1** | `l1_dn_weight` | 80.0 | How strongly a DN spike excites the Template Layer. This is a large direct current injection — it's the main "pay attention now" signal. |
| **Beta (leak)** | `beta` | 0.7 | Membrane leak factor (0–1). Higher values mean the neuron remembers past input longer; lower values make it more "forgetful" and responsive only to sharp transients. |
| **Depression tau** | `depression_tau` | 100 | Recovery time constant for pRel depression, in samples. Larger values mean the DN stays suppressed longer after firing. At 20 kHz, 100 samples ≈ 5 ms. |
| **Depression fraction** | `depression_frac` | 0.5 | How much the release probability drops after each spike. 0.5 means it halves. Closer to 1.0 = deeper depression. |
| **DN window (DEC)** | `dec_dn_window_ms` | 5.0 | Sliding window (ms) in the decoder that counts recent DN spikes to compute confidence. Not a DN parameter per se, but tightly coupled. |

---

## 3. Noise Gate Neuron

**File:** `src/snn_agent/core/noise_gate.py`

### What it does

The Noise Gate runs **in parallel** with the Attention Neuron but does
the opposite job: instead of detecting interesting events, it detects
**boring** ones (noise) and suppresses the input to L1 accordingly.

It uses a tiny **Kalman filter** to track the variance of the incoming
signal over time. At each step it:

1. **Predicts** the next variance estimate (adds a small "process noise"
   term to account for the world changing).
2. **Updates** the estimate using the actual observed sample variance
   (weighted by a Kalman gain that balances trust in the prediction vs
   the measurement).

If the current signal variance is close to or below what the Kalman
filter considers "just noise" (i.e. within `inhibit_below_sd` standard
deviations of the noise baseline), the noise gate outputs a
**suppression factor** between 0 and 1. This factor multiplies the
afferent input before it reaches L1, effectively turning down the volume
during quiet periods.

Think of it as an automatic gain control on a microphone: when nobody is
talking, it dials the sensitivity way down so you don't hear hiss.

### Hyperparameters

| Parameter | Config key | Default | What it does |
|-----------|-----------|---------|--------------|
| **Process noise** | `process_noise` | 1e-3 | How much the Kalman filter expects the "true" noise variance to wander between steps. Larger values make the filter more reactive but jumpier. |
| **Measurement noise** | `measurement_noise` | 0.05 | How much the filter distrusts any single variance measurement. Larger = smoother but slower to react. |
| **Inhibit below SD** | `inhibit_below_sd` | 2.0 | The threshold (in standard deviations above the Kalman-estimated noise floor) below which the gate starts suppressing. Higher = more aggressive suppression. |
| **Suppression factor** | `suppression_factor` | 0.3 | When the gate decides "this is noise," it multiplies the afferent vector by this value (0.3 = reduce to 30%). Lower = stronger muting. |
| **EMA alpha** | `ema_alpha` | 0.001 | Smoothing factor for the exponential moving average that tracks the long-term noise baseline. Smaller = slower adaptation, more stable baseline. |

---

## 4. Global Inhibitor

**File:** `src/snn_agent/core/inhibition.py`

### What it does

The Global Inhibitor enforces a **refractory blanking period** after any
L1 neuron fires. For a configurable duration (default 5 ms) after a
spike, it blocks all new L1 input — unless the incoming signal is
exceptionally strong (above `strength_threshold`).

This serves two purposes:

1. **Prevents double-counting:** Without blanking, a single neural spike
   waveform (which lasts ~1 ms) could trigger the same L1 neuron
   multiple times as the waveform slides through the encoder's shift
   registers.
2. **Lets strong signals through:** If a second real spike arrives very
   shortly after the first, the inhibitor's strength bypass ensures it
   isn't lost. This handles the case of overlapping spikes from
   different neurons.

The implementation is a simple countdown timer: when any L1 neuron
fires, the timer resets to `duration_samples`. Each step it decrements.
While the timer is > 0 and the afferent magnitude is below the bypass
threshold, the input to L1 is zeroed out.

### Hyperparameters

| Parameter | Config key | Default | What it does |
|-----------|-----------|---------|--------------|
| **Duration (ms)** | `duration_ms` | 5.0 | How long the blanking window lasts after a spike, in milliseconds. At 20 kHz this is 100 samples. Longer = more conservative, may miss closely spaced spikes. |
| **Strength threshold** | `strength_threshold` | 3.5 | If the afferent vector's sum exceeds this value during the blanking window, the input is allowed through anyway. This is the "strong signal bypass." Higher = harder to bypass. |

---

## 5. Template Layer (L1)

**File:** `src/snn_agent/core/template.py`

### What it does

The Template Layer is where the actual **spike sorting** happens. It
contains a population of LIF neurons (default 110), each of which learns
to recognise one particular spike waveform shape via **STDP
(Spike-Timing-Dependent Plasticity)**.

#### How a single L1 neuron works

Each L1 neuron receives the full afferent vector (from the encoder, gated
by the noise gate and inhibitor) through a set of learned weights. At
each time step:

1. **Weighted input** = dot product of afferents × weight vector, plus
   the DN excitation (if the DN fired).
2. The neuron's **membrane potential** integrates this input with a leak
   factor `beta`. If it crosses `threshold`, the neuron fires.

#### Winner-Take-All (WTA)

When multiple L1 neurons fire simultaneously, only the one with the
**highest membrane potential** is allowed to actually spike. All others
are reset. This prevents every neuron from learning the same template —
it forces specialisation.

#### STDP Learning

After each time step, weights are updated:

- **LTP (Long-Term Potentiation):** If a neuron fires and an afferent
  was active just before, the synapse connecting them is strengthened.
  "Neurons that fire together wire together."
- **LTD (Long-Term Depression):** If a neuron fires but an afferent was
  *not* active, that synapse is weakened. This sharpens selectivity.
- **Weight clamping:** Weights are hard-clamped to [0, 1] after every
  update to prevent runaway growth or negative weights.

Over time, each neuron's weight vector converges to match one particular
spike waveform template. When that waveform appears in the input, that
neuron fires; other waveforms activate other neurons.

### Hyperparameters

| Parameter | Config key | Default | What it does |
|-----------|-----------|---------|--------------|
| **Number of neurons** | `n_neurons` | 110 | Size of the L1 population. More neurons = capacity to learn more templates, but slower and may waste neurons on noise. |
| **Beta (leak)** | `beta` | 0.75 | Membrane leak. Higher = longer memory, more temporal integration. Lower = responds only to sharp coincident input. |
| **Threshold** | `threshold` | 25.0 | Membrane voltage that triggers a spike. Higher = neuron needs stronger / more coherent input to fire. |
| **STDP LTP rate** | `stdp_ltp` | 0.05 | Learning rate for strengthening synapses. Larger = faster learning but less stable templates. |
| **STDP LTD rate** | `stdp_ltd` | 0.02 | Learning rate for weakening synapses. Should generally be smaller than LTP to allow net learning. |
| **DN weight** | (from `DNConfig.l1_dn_weight`) | 80.0 | Direct current injection from DN spike. This is what makes L1 "pay attention" when the DN fires. |
| **Initial weight** | `init_weight` | 0.5 | Starting value for all synaptic weights before STDP shapes them. |

---

## 6. DEC Layer (Output)

**File:** `src/snn_agent/core/dec_layer.py`

### What it does

The DEC (Decision / Event Classification) Layer sits on top of L1 and
translates the template-layer spikes into a classified output. It
contains **16 LIF neurons** with distinct roles:

#### Neuron 0 — The OR Gate

Neuron 0 receives a fixed weight of **1.0** from *every* L1 neuron, plus
the DN excitation. It fires whenever *any* L1 neuron fires and the DN
agrees that something interesting is happening. It answers the question:
"Did a spike occur?" (regardless of which unit produced it).

#### Neurons 1–15 — Unit Identity Learners

These 15 neurons learn to associate specific **groups of L1 neurons**
with specific neural units via STDP, the same way L1 neurons learn
waveform templates. Over time, neuron 3 might learn that "when L1
neurons 12, 47, and 83 fire together, that's Unit A," while neuron 7
learns the pattern for Unit B.

Key differences from L1 STDP:

- **Stronger lateral inhibition** (`wi_factor = 8`): the WTA competition
  is much fiercer, forcing neurons to specialise on different unit
  identities quickly.
- **DN gating**: weight updates only happen when the DN has fired
  recently (within `dn_window_ms`). This ensures the DEC only learns
  from real spike events, not noise.
- **Optional delay expansion**: L1 spikes can be smeared over a short
  time window (`delay_expansion` steps) to tolerate jitter in spike
  timing.

#### Output Format

The DEC layer's output is a **16-bit hex bitmask** (`uint16`): bit *i*
is set if DEC neuron *i* fired. This bitmask is sent via UDP to
experiment hardware and broadcast over WebSocket to the GUI.

### Hyperparameters

| Parameter | Config key | Default | What it does |
|-----------|-----------|---------|--------------|
| **Number of neurons** | `n_neurons` | 16 | Fixed at 16 (1 OR gate + 15 unit learners). |
| **Beta (leak)** | `beta` | 0.6 | Membrane leak for DEC neurons. |
| **Threshold** | `threshold` | 15.0 | Firing threshold. Lower than L1 because DEC receives fewer inputs. |
| **STDP LTP** | `stdp_ltp` | 0.08 | Learning rate for strengthening. Slightly faster than L1 defaults. |
| **STDP LTD** | `stdp_ltd` | 0.04 | Learning rate for weakening. |
| **WI factor** | `wi_factor` | 8.0 | Lateral inhibition multiplier for WTA. 8× means the winner strongly suppresses all losers. |
| **DN window (ms)** | `dn_window_ms` | 5.0 | Time window after a DN spike during which STDP updates are allowed. |
| **Delay expansion** | `delay_expansion` | 2 | How many extra time steps L1 spikes are "smeared" over to tolerate timing jitter. |
| **DN weight** | `dn_weight` | 60.0 | Direct current from DN spike into DEC neurons (especially neuron 0). |

---

## 7. How They Wire Together

```
Raw Signal (80 kHz electrode / synthetic)
    │
    ▼
┌────────────────────┐
│  Preprocessor      │  Bandpass 300–6000 Hz, then decimate ÷4 → 20 kHz
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Spike Encoder     │  Calibrates (MAD), then converts voltage → binary
│                    │  afferent grid [n_centres × window_depth]
└────────┬───────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
┌────────────────────┐       ┌────────────────────┐
│  Attention Neuron  │       │  Noise Gate        │
│  (DN)              │       │  (Kalman filter)   │
│                    │       │                    │
│  "Is this a real   │       │  "Is this just     │
│   spike event?"    │       │   background hiss?"|
└────────┬───────────┘       └────────┬───────────┘
         │ dn_spike (bool)            │ suppression (0–1)
         │                            │
         ├────────────────────────────┤
         ▼                            ▼
┌──────────────────────────────────────────────┐
│  Global Inhibitor                            │
│  Blanks input for 5 ms after any L1 spike    │
│  Strong signals bypass the blanking          │
└────────┬─────────────────────────────────────┘
         │ gated afferents
         ▼
┌────────────────────┐
│  Template Layer    │  110 LIF neurons + WTA + STDP
│  (L1)              │  Each neuron learns one waveform shape
│                    │  DN spike → big excitatory boost
└────────┬───────────┘
         │ l1_spikes [110 bools]
         ▼
┌────────────────────┐
│  DEC Layer         │  16 LIF neurons
│                    │  N0: "did anything fire?" (OR gate)
│                    │  N1–15: "which unit was it?" (STDP)
│                    │  Output: 16-bit hex bitmask
└────────┬───────────┘
         │ dec_spikes + hex bitmask
         ▼
┌────────────────────┐
│  Control Decoder   │  Converts spikes → (control, confidence)
│                    │  Sends UDP to hardware
│                    │  Broadcasts via WebSocket
└────────────────────┘
```

### Signal Flow Summary

1. **Preprocessor** cleans and down-samples the raw signal.
2. **Encoder** converts it to binary spikes (after a brief calibration).
3. **DN** and **Noise Gate** run in parallel — DN flags interesting
   events, Noise Gate suppresses boring ones.
4. **Global Inhibitor** enforces a refractory period to prevent
   double-counting.
5. **L1** learns waveform templates via STDP, boosted by DN excitation.
6. **DEC** classifies which neural unit produced each spike.
7. **Decoder** translates the classification into a control signal.

### Key Interactions

- **DN → L1:** Large excitatory weight (`l1_dn_weight = 80`) makes L1
  fire almost exclusively when the DN agrees "this is real."
- **DN → DEC:** Provides gating for STDP learning and contributes to
  neuron 0's OR decision.
- **Noise Gate → L1:** Multiplicative suppression reduces false
  activations during quiet periods.
- **L1 → Inhibitor → L1:** Feedback loop — when L1 fires, it triggers
  its own blanking period.
- **L1 → DEC:** Feed-forward — DEC neurons receive L1 spike patterns
  and learn to group them into unit identities.

---

*Generated for the SNN spike-sorting pipeline. See
`docs/scientific_principles.md` for the full mathematical treatment and
`docs/optimization_guide.md` for the hyperparameter tuning methodology.*
