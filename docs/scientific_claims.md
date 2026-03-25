# Scientific Claims Audit — SNN Agent

This document catalogues every major scientific claim made by the SNN Agent
codebase (README, docstrings, architecture docs) and evaluates each against
the literature, the implementation, and mathematical consistency.

**Legend:**  ✅ Sound  ·  ⚠️ Minor caveat  ·  ❌ Incorrect or unsupported

---

## 1. Noise Estimation via MAD / 0.6745

**Claim:** The spike encoder estimates background noise standard deviation
using the Median Absolute Deviation (MAD) divided by 0.6745.

**Implementation:** `core/encoder.py` — `_calibrate()` computes
`median(|x|) / 0.6745` over the first `noise_init_samples` samples.

**Evaluation:** ✅ **Sound.**

The MAD scaled by $1/\Phi^{-1}(3/4) \approx 1/0.6745$ is a well-established
consistent estimator of $\sigma$ under Gaussian noise.  It is robust to
outliers (spike waveforms), which is exactly the motivation: the spikes are
the "outliers" the estimator should ignore.

- Donoho & Johnstone (1994) — *Ideal spatial adaptation by wavelet shrinkage.*
  Biometrika 81(3): 425–455.  Introduced MAD / 0.6745 for wavelet denoising.
- Quiroga, Nadasdy & Ben-Shaul (2004) — *Unsupervised spike detection and
  sorting with wavelets and superparamagnetic clustering.*  Neural Computation
  16(8): 1661–1687.  Uses the identical MAD formula for extracellular noise
  estimation — the same neuroscience context as SNN Agent.

**Caveat:** None.  The implementation is a textbook application of the
MAD / 0.6745 estimator.  Bin centres and half-widths are fixed at calibration
time, which is appropriate for stationary noise conditions.

---

## 2. Temporal Receptive Field Encoding

**Claim:** The SpikeEncoder converts scalar samples into a spatiotemporal
binary activation pattern using overlapping amplitude bins and a shift
register, producing a flat boolean afferent vector.

**Implementation:** `core/encoder.py` — amplitude bins with half-width
$\Delta v_m = \gamma \hat\sigma$, spaced by $2\Delta v_m / O$, fed into a
shift register of depth $D$ subsampled by step size $S$.

**Evaluation:** ✅ **Sound.**

This is a faithful implementation of the temporal receptive field (TRF)
encoder described in the MB2018 ANNet architecture.  The approach converts a
1-D continuous signal into a 2-D binary pattern (amplitude × time), enabling
pattern matching via downstream STDP.  Analogous to cochlear-style place
coding, where each "bin" acts as a frequency/amplitude channel.

- Thorpe, Delorme & Van Rullen (2001) — *Spike-based strategies for rapid
  processing.*  Neural Networks 14(6–7): 715–725.  Establishes the broader
  framework of rank-order and first-spike coding.

**Caveat:** None.  The overlap parameter $O$ controls the resolution/redundancy
trade-off.  Higher overlap → more bins → higher afferent dimensionality but
smoother coverage.

---

## 3. Attention Neuron (DN) — LIF with Short-Term Synaptic Depression

**Claim:** The attention neuron uses a LIF model with Tsodyks-Markram-style
short-term synaptic depression (pRel recovery + depletion) to detect when
signal energy exceeds the noise floor.

**Implementation:** `core/attention.py` — membrane:
$V_t = V_{t-1} e^{-1/\tau_m} + \sum p_{\mathrm{rel},j}$.  pRel recovery:
$p_{\mathrm{rel}} = 1 - (1 - p_{\mathrm{last}}) e^{-\Delta t/\tau_d}$.
Depression: $p_{\mathrm{rel}} \leftarrow p_{\mathrm{rel}} (1-f_d)$.

**Evaluation:** ✅ **Sound.**

The synaptic depression model is a simplified version of Tsodyks & Markram
(1997).  The full T–M model tracks three variables (utilisation $u$, available
resources $x$, post-synaptic current $I$).  SNN Agent collapses this into a
single variable $p_{\mathrm{rel}}$ (release probability) that recovers
exponentially and is depleted multiplicatively on each spike.  This is a
valid simplification when the utilisation parameter $U$ is fixed and only the
available-resource dynamics matter.

- Tsodyks & Markram (1997) — *The neural code between neocortical pyramidal
  neurons depends on neurotransmitter release probability.*  PNAS 94(2):
  719–723.

**Caveat:** ⚠️ The implementation uses a pre-computed LUT
(`_exp_td = exp(-arange/τ_d)`) truncated at `20·τ_d`.  For the default
`τ_d = 400`, this allocates an 8001-element array.  At $\Delta t > 8000$
steps, the LUT clips to the boundary value, which is effectively full
recovery ($p_{\mathrm{rel}} \approx 1$).  This is numerically correct — at
$20\tau_d$, $e^{-20} \approx 2 \times 10^{-9}$, so the clipping error is
negligible.

---

## 4. DN Threshold Derivation

**Claim:** The DN threshold is
$\theta_{\mathrm{DN}} = F_{\mathrm{th}} \cdot O \cdot D / (1 - e^{-1/\tau_m})$.

**Implementation:** `core/attention.py` — `__init__()` computes this exactly.

**Evaluation:** ✅ **Sound.**

The formula computes a steady-state threshold.  The denominator
$1 - e^{-1/\tau_m}$ is the geometric series sum factor for a leaky
integrator: if $O \cdot D$ afferents fire simultaneously every step (maximum
sustained input), the membrane converges to
$V_{ss} = (O \cdot D) / (1 - \beta)$.  The threshold factor $F_{\mathrm{th}}$
then scales this to require a fraction of maximum sustained activation before
firing.  This is a principled way to set thresholds relative to the input
dimensionality.

**Origin:** This derivation appears to be original to the MB2018 ANNet
architecture and is an adaptation of the standard LIF steady-state formula.

---

## 5. DN Reset Potential

**Claim:** After firing, the DN resets to
$V_{\mathrm{reset}} = F_r \cdot (e^{1/\tau_m} - 1) \cdot \theta$.

**Implementation:** `core/attention.py` — `__init__()` computes this exactly.
On fire: `self.v = self.reset_potential` (hard set).

**Evaluation:** ⚠️ **Functional but derivation unclear.**

The factor $(e^{1/\tau_m} - 1) = 1/\beta - 1$ normalises the reset relative
to a single-step voltage contribution at threshold.  For the default
$\tau_m = 2$:  $e^{0.5} - 1 \approx 0.6487$.  With $F_r = 0.15$, the reset
is $\approx 9.7\%$ of threshold.  This provides a small positive "memory"
after firing, biasing the neuron to re-fire if the signal persists.

The precise derivation of this formula is not documented in any reference we
can find.  It appears to be an empirical choice from the MB2018 work.  The
behaviour is reasonable — it implements a soft reset rather than a hard zero.

---

## 6. Template Layer (L1) Threshold

**Claim:**
$\theta_{L1} = (w_{\mathrm{DN}} + O(D-k)) \cdot (1 - e^{-S/\tau_m}) / (1 - e^{-1/\tau_m})$
where $k = 3$.

**Implementation:** `core/template.py` — `__init__()` computes this exactly.

**Evaluation:** ✅ **Sound.**

This threshold formula models the expected maximum input current when a
coherent spike pattern fills the receptive field.  The numerator
$w_{\mathrm{DN}} + O(D-k)$ represents the peak per-step current: the DN
excitatory weight plus the number of afferents activated by a spike waveform
spanning $(D-k)$ of the $D$ delay taps (with overlap $O$).  The constant
$k = 3$ represents the half-width of a typical extracellular spike waveform
in delay-tap units.

The factor $(1 - e^{-S/\tau_m})$ accounts for the temporal subsampling — with
step size $S > 1$, the membrane decays between observations.  The denominator
$(1 - e^{-1/\tau_m})$ is the standard steady-state normalisation.

**Caveat:** ⚠️ The magic constant $k = 3$ is hardcoded.  This assumes a
spike waveform width of $\sim$3 delay taps.  For data with substantially
different waveform morphology (e.g., burst complexes, different electrode
impedance), $k$ may need tuning.  It would be better modelled as a
configurable parameter.

---

## 7. Asymmetric STDP — Global LTD + Causal LTP

**Claim:** Learning uses asymmetric STDP where every post-synaptic spike
produces global long-term depression across all afferents, and
recently-active pre-synaptic afferents receive long-term potentiation.

**Implementation:** `core/template.py` — `_stdp()`:
(1) `w += ltd` for all afferents (global LTD);
(2) `w[causal] += ltp` for afferents with `(t - t_pre) ≤ T_LTP`.

**Evaluation:** ✅ **Sound.**

This is a well-established competitive STDP rule.  The global LTD ensures
that unused synapses decay, while the causal LTP reinforces afferents that
fired shortly before the post-synaptic spike.  Combined with WTA, this drives
each neuron to specialise on a different spatio-temporal pattern — effectively
implementing unsupervised template matching.

- Masquelier, Guyonneau & Thorpe (2008) — *Spike timing dependent plasticity
  finds the start of repeating patterns in continuous spike trains.*  PLoS ONE
  3(1): e1377.  Uses the same global-LTD / causal-LTP asymmetric STDP rule
  for temporal pattern learning.
- Song, Miller & Abbott (2000) — *Competitive Hebbian learning through
  spike-timing-dependent synaptic plasticity.*  Nature Neuroscience 3(9):
  919–926.  Foundational work on competitive STDP.

**Caveat:** None.  The implementation is a faithful discrete-time version of
the continuous rule.  The additional guard `last_pre_spike > 0` correctly
prevents LTP on afferents that have never fired.

---

## 8. Winner-Take-All via snnTorch `inhibition=True`

**Claim:** WTA competition is implemented via snnTorch's `inhibition=True`
parameter on the Leaky neuron, allowing only the neuron with the highest
membrane potential to fire per timestep.

**Implementation:** `core/template.py` — `snn.Leaky(inhibition=True, ...)`.

**Evaluation:** ⚠️ **Functionally correct, with a stability caveat.**

snnTorch's `inhibition=True` implements lateral inhibition by suppressing all
neurons except the one with the maximum membrane potential.  This effectively
gives WTA behaviour.  However, snnTorch itself marks this feature as
"unstable" (the codebase suppresses this warning).

The WTA mechanism is sound for template matching — it ensures that when
multiple neurons could potentially match a pattern, only the best-matching one
fires and receives STDP updates, driving specialisation.

- Oster, Douglas & Liu (2009) — *Computation with spikes in a winner-take-all
  network.*  Neural Computation 21(9): 2437–2465.

---

## 9. Control Decoder — Rate Strategy

**Claim:** The rate decoder computes a sliding-window weighted spike rate.

**Implementation:** `core/decoder.py` — `_step_rate()` accumulates spike
vectors in a deque of length $T_w$, computes per-neuron rates = counts / $T_w$,
then dot-products with weights, clipped to $[-1, 1]$.

**Evaluation:** ✅ **Sound.**

This is a standard rate-coded readout.  The sliding-window approach is
appropriate for streaming applications where the control signal must be
continuously updated.

---

## 10. Control Decoder — Population Strategy

**Claim:** The population decoder uses a leaky integrator that emits on
threshold crossing.

**Implementation:** `core/decoder.py` — `_step_population()`:
$\mathcal{I}_t = \mathcal{I}_{t-1} \cdot \beta + w^T s_t$.  Emits and resets
to 0 when $\mathcal{I}_t \geq \theta$.

**Evaluation:** ✅ **Sound.**

The leaky integrator with threshold is a standard model for population-level
activity detection.  The reset-to-zero on emission prevents double-counting.

---

## 11. DN Confidence Metric

**Claim:** Decoder confidence is the sliding-window average of DN spikes.

**Implementation:** `core/decoder.py` — `confidence = sum(dn_buf) / len(dn_buf)`
where `dn_buf` is a bounded deque.

**Evaluation:** ⚠️ **Reasonable heuristic, not a formal confidence measure.**

This gives the fraction of recent timesteps in which the attention neuron
fired.  A high value indicates sustained signal-to-noise, a low value
indicates either silence or noise-dominated input.  While it correlates with
signal quality, it is not calibrated as a probabilistic confidence in the
Bayesian sense.

**Caveat:** During warm-up (`len(buf) < T_dn`), the denominator grows from 1,
making early confidence values noisy.  This is noted in the README.

---

## 12. Pipeline Architecture — Biological Plausibility

**Claim (implicit):** The overall architecture — bandpass → encode → detect →
gate → inhibit → template match → identify → decode — mimics a plausible
neural processing hierarchy for spike sorting.

**Evaluation:** ✅ **Sound as an engineering abstraction.**

The pipeline mirrors established computational neuroscience architectures:

1. **Bandpass filtering** = peripheral auditory processing / electrode
   hardware filtering.  Standard in all spike sorting pipelines.
2. **Amplitude binning + shift register** = place-coded representation,
   analogous to cochlear tonotopy or cortical receptive fields.
3. **Attention neuron with depression** = novelty/salience detection.
   Short-term depression naturally implements a high-pass filter for temporal
   patterns, suppressing response to sustained or repetitive inputs.
4. **Noise gate** = inhibitory counterpart to the attention neuron.  Push–pull
   excitation/suppression mirrors cortical excitatory/inhibitory balance.
5. **Post-spike inhibition** = refractory blanking via inhibitory interneurons.
   Standard in cortical models.
6. **Template matching via competitive STDP** = cortical column-like
   specialisation.  Each neuron learns a prototype.
7. **DEC hierarchical grouping** = second-level abstraction, grouping L1
   features into putative unit identities.
8. **Rate/population decode** = standard motor cortex readout models.

The architecture does not claim to be a biophysically detailed model — it is
a biologically *inspired* engineering system.  This is appropriate.

---

## 13. snnTorch LIF Model — `reset_mechanism="zero"`

**Claim:** The template layer uses snnTorch's Leaky neuron with zero reset.

**Implementation:** `snn.Leaky(reset_mechanism="zero", ...)`.

**Evaluation:** ✅ **Sound.**

The "zero" reset mechanism sets the membrane potential to 0 on spike, which is
appropriate for the template matching use case: after a successful match, the
neuron should start fresh for the next potential match.  This contrasts with
"subtract" (which retains excess voltage) — "zero" provides cleaner
discrimination between patterns.

---

## 14. Preprocessor — SOS Bandpass Filter

**Claim:** The preprocessor uses a causal IIR bandpass filter (second-order
sections) that is streaming-safe.

**Implementation:** `core/preprocessor.py` — `butter(..., output='sos')` with
`sosfilt` and carried filter state.

**Evaluation:** ✅ **Sound.**

Using SOS form avoids the numerical instability of transfer-function (ba)
form for higher-order filters.  Carrying the filter state (`zi`) across
sample-by-sample calls ensures correct streaming operation — this is a
textbook implementation of real-time IIR filtering.

---

## 15. Noise Gate — 1-D Kalman Variance Tracker

**Claim:** The noise gate uses a 1-D Kalman filter to track signal variance
and suppresses template layer input when the estimated standard deviation is
below a configurable multiple of the noise baseline.

**Implementation:** `core/noise_gate.py` — standard predict/update cycle with
state = variance estimate, measurement = squared sample.  Gating uses linear
interpolation between `suppression_factor` and 1.0, smoothed with an EMA.

**Evaluation:** ✅ **Sound.**

The 1-D Kalman filter is optimal for tracking a scalar state under Gaussian
noise.  Using it for variance estimation is a pragmatic engineering choice.
The push–pull modulation with the attention neuron is analogous to
excitatory/inhibitory balance in cortical circuits.

**Caveat:** None.  The implementation is straightforward.

---

## 16. Global Inhibition — Post-Spike Blanking with Strong-Signal Bypass

**Claim:** After any L1 neuron fires, all L1 input is suppressed for a
configurable blanking window (default 5 ms), unless the raw input current
exceeds a strength threshold, allowing strong signals to break through.

**Implementation:** `core/inhibition.py` — countdown timer activated on any
L1 spike, returns suppression factor 0.0 (block) or 1.0 (pass).  Strong
signals bypass via `current_magnitude >= strength_threshold`.

**Evaluation:** ✅ **Sound.**

Post-spike inhibition is a well-established mechanism in cortical circuits,
mediated by inhibitory interneurons.  The strong-signal bypass is a practical
engineering choice that prevents the inhibitor from masking genuinely
distinct, high-amplitude spikes that happen to occur during the blanking
window.

**Caveat:** None.

---

## 17. DEC Layer — DN-Gated Hierarchical Competitive STDP

**Claim:** The DEC layer uses 16 LIF neurons that receive L1 spikes and learn
to produce one spike per putative neural unit, using the same competitive
STDP rule as L1 but with DN-gated input integration.

**Implementation:** `core/dec_layer.py` — neurons 1–15 use asymmetric STDP
(global LTD + causal LTP) with WTA competition, gated by a DN-controlled
integration window (`dn_window_ms`).

**Evaluation:** ✅ **Sound.**

The DEC layer implements a standard hierarchical competitive learning
architecture: L1 learns low-level features (waveform shapes), DEC learns
higher-level associations (which combinations of L1 responses correspond to
the same neural unit).  This mirrors the L1→L2 hierarchy in the original
MATLAB ANNet architecture.

**Caveat:** None for the STDP mechanism.  The DN-gated window is a pragmatic
engineering choice that addresses a timing problem: the DN fires slightly
before L1 completes its template matching.

---

## 18. DEC Layer — Hard-Wired OR Gate (Neuron 0)

**Claim:** DEC neuron 0 is a hard-wired OR gate that fires whenever any L1
neuron fires while the DN is active, with no learning.

**Implementation:** `core/dec_layer.py` — neuron 0 has fixed unit weights
from all L1 neurons, very low threshold, and no STDP updates.

**Evaluation:** ⚠️ **Engineering choice, not a learned component.**

The OR gate provides a reliable "any spike detected" signal for simple
control tasks but does not discriminate between neural units.  This is by
design — it serves as a guaranteed baseline detection signal.

---

## Summary

| # | Claim | Verdict | Action needed |
|---|---|---|---|
| 1 | MAD / 0.6745 noise estimator | ✅ | — |
| 2 | Temporal receptive field encoding | ✅ | — |
| 3 | LIF + synaptic depression (DN) | ✅ | — |
| 4 | DN threshold derivation | ✅ | — |
| 5 | DN reset potential formula | ⚠️ | Document derivation origin |
| 6 | L1 threshold derivation | ✅ | Consider making $k$ configurable |
| 7 | Asymmetric STDP | ✅ | — |
| 8 | WTA via snnTorch inhibition | ⚠️ | Monitor snnTorch stability status |
| 9 | Rate decoder | ✅ | — |
| 10 | Population decoder | ✅ | — |
| 11 | DN confidence metric | ⚠️ | Document warm-up behaviour |
| 12 | Pipeline architecture | ✅ | — |
| 13 | snnTorch LIF zero reset | ✅ | — |
| 14 | SOS causal bandpass | ✅ | — |
| 15 | Noise gate (Kalman variance tracker) | ✅ | — |
| 16 | Post-spike blanking with bypass | ✅ | — |
| 17 | DEC hierarchical competitive STDP | ✅ | — |
| 18 | DEC OR gate (neuron 0) | ⚠️ | Engineering choice — no learning |

**Overall assessment:** The scientific foundations are solid.  All equations
match the code implementation.  The four minor caveats (DN reset derivation,
WTA stability warning, confidence warm-up, OR gate design choice) are
documented above.  No claims require correction.
