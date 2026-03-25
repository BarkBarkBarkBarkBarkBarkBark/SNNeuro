# Optimization Guide — SNN Agent

> Complete reference for hyperparameter optimization of the SNN spike-sorting
> pipeline. Covers the evaluation methodology, search strategy, metrics,
> genetic optimizer, and practical usage.
>
> **Last updated:** 2026-03-24  
> **Related files:** [`optimization_manifest.yaml`](optimization_manifest.yaml),
> [`evaluate.py`](../src/snn_agent/eval/evaluate.py),
> [`optimize.py`](../src/snn_agent/eval/optimize.py),
> [`genetic.py`](../src/snn_agent/eval/genetic.py),
> [`config.py`](../src/snn_agent/config.py)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Evaluation Methodology](#2-evaluation-methodology)
   - 2.1 [Multi-Scenario Evaluation](#21-multi-scenario-evaluation)
   - 2.2 [Spike Matching Tolerance](#22-spike-matching-tolerance-delta_time)
   - 2.3 [Train/Test Temporal Split](#23-traintest-temporal-split)
   - 2.4 [Metrics](#24-metrics)
   - 2.5 [Early Termination](#25-early-termination)
3. [Stage 1: Optuna TPE Search](#3-stage-1-optuna-tpe-search)
   - 3.1 [How It Works](#31-how-it-works)
   - 3.2 [Search Space](#32-search-space)
   - 3.3 [Seeding from Prior Best](#33-seeding-from-prior-best)
   - 3.4 [Running](#34-running)
4. [Stage 2: Genetic Crossover Optimizer](#4-stage-2-genetic-crossover-optimizer)
   - 4.1 [Design Rationale](#41-design-rationale)
   - 4.2 [Gene Blocks](#42-gene-blocks)
   - 4.3 [Selection, Crossover, and Mutation](#43-selection-crossover-and-mutation)
   - 4.4 [Running](#44-running)
5. [Interpreting Results](#5-interpreting-results)
   - 5.1 [Output Files](#51-output-files)
   - 5.2 [What to Look For](#52-what-to-look-for)
   - 5.3 [Dead Trials](#53-dead-trials)
6. [Pipeline Architecture for Optimization](#6-pipeline-architecture-for-optimization)
   - 6.1 [Parameter Interactions](#61-parameter-interactions)
   - 6.2 [The Threshold Reachability Constraint](#62-the-threshold-reachability-constraint)
7. [Design Decisions and Rationale](#7-design-decisions-and-rationale)
8. [Extending the Search Space](#8-extending-the-search-space)
9. [Performance History](#9-performance-history)

---

## 1. Overview

The SNN Agent uses a two-stage optimization workflow:

```
Stage 1: Optuna TPE Search          Stage 2: Genetic Crossover
─────────────────────────           ────────────────────────────
  snn-optimize (80 trials)   →       snn-genetic (160 offspring)
  Broad exploration via                Fine-tuning via block-level
  Tree-structured Parzen               crossover + bounded Gaussian
  Estimator. Discovers                  mutation. Breeds top-K parents
  viable parameter regions.             from Stage 1.
         │                                       │
         └── data/trials.csv ──→ top K ──→       │
         └── data/best_config.json               │
                                                  └── data/best_config.json
                                                  └── data/genetic_trials.csv
```

Both stages evaluate each parameter configuration against **multiple
synthetic neural recordings** with a **train/test temporal split** and a
**precision-weighted F₀.₅ objective**, preventing overfitting to any single
noise realization and penalizing false positives.

---

## 2. Evaluation Methodology

### 2.1 Multi-Scenario Evaluation

Each trial is evaluated across **4 diverse synthetic recordings**, and the
final score is the **arithmetic mean** of per-scenario metrics. This prevents
the optimizer from overfitting to a single noise realization, unit count, or
firing rate pattern.

| Scenario | Seed | Noise Level | Units | Firing Rates (Hz) | Character |
|----------|------|-------------|-------|--------------------|-----------|
| S1 | 42 | 8.0 | 2 | [6, 10] | Baseline — moderate noise, typical rates |
| S2 | 137 | 10.0 | 2 | [6, 10] | High noise — tests robustness |
| S3 | 256 | 6.0 | 2 | [8, 14] | Low noise, fast rates — tests temporal precision |
| S4 | 789 | 12.0 | 3 | [5, 8, 12] | Very high noise, 3 units — stress test |

Each recording is 20 seconds at 30 kHz, generated via SpikeInterface's
`generate_ground_truth_recording()` with known spike train positions. The
scenarios are defined in `optimization_manifest.yaml` under `objective.scenarios`
and can be modified or extended.

**Implementation:** `multi_evaluate()` in
[`evaluate.py`](../src/snn_agent/eval/evaluate.py) loops over scenarios,
calling `evaluate_pipeline()` for each. If the first scenario scores 0.0
(dead configuration), remaining scenarios are skipped to save time.

### 2.2 Spike Matching Tolerance (`delta_time`)

When comparing detected spikes against ground truth, SpikeInterface uses a
time window to match them. A detected spike within ±`delta_time` ms of a
ground truth spike counts as a **true positive**; otherwise it's a **false
positive**.

| Setting | Value | Context |
|---------|-------|---------|
| SpikeInterface default | 0.4 ms | Designed for high-precision offline sorters |
| **Our setting** | **2.0 ms** | Appropriate for an online LIF-based system |
| Previous setting | 10.0 ms | Was 25× too generous — masked timing imprecision |

**Why 2.0 ms?** Empirical testing revealed a sharp accuracy cliff between
1.5 ms and 2.0 ms. The pipeline's inherent LIF membrane integration
introduces ~1.5–2.0 ms jitter in spike timing (relative to the true spike
peak). At 2.0 ms, accuracy reflects genuine template matching quality. At
10.0 ms, a spike detected almost 10 ms late still counted as correct, hiding
poor temporal precision.

**Impact on numbers:** The same configuration that scored 0.88 accuracy at
`delta_time=10.0` drops to 0.26 at 1.5 ms and holds at 0.88 at 2.0 ms.
This demonstrates that the pipeline's template matching is temporally precise
to within ~2 ms — a physically reasonable result for single-compartment LIF
neurons.

### 2.3 Train/Test Temporal Split

STDP learning and accuracy measurement now happen on **different temporal
windows** of the same recording:

```
├─── Train window (0–15 s) ──────────────────────┤──── Test window (15–20 s) ───┤
│ STDP active, neurons learn waveform templates   │ Scoring happens here only    │
│ Spikes detected but NOT scored                  │ STDP still active (online)   │
│                                                 │ but accuracy computed here   │
└─────────────────────────────────────────────────┴─────────────────────────────┘
```

**Why?** Previously, the same 20-second window was used for both learning and
scoring — equivalent to evaluating a classifier on its training set. The
split ensures that accuracy reflects the pipeline's ability to generalize
learned templates to new data, not just memorize the training signal.

**Implementation:** `evaluate_pipeline(score_after_s=15.0)` filters both
detected spikes and ground truth spikes to only include those at
$t \geq 15.0 \text{ s}$ before running SpikeInterface comparison.

**Note:** STDP remains active during the test window (this is an online
system, not a batch learner). The split tests whether templates learned in
the first 15 s continue to work on fresh signal.

### 2.4 Metrics

#### Primary objective: F₀.₅

The optimizer maximizes $F_{0.5}$, a precision-weighted harmonic mean:

$$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

where:
- $P = \frac{TP}{TP + FP}$ (precision — fraction of detections that are real spikes)
- $R = \frac{TP}{TP + FN}$ (recall — fraction of real spikes that are detected)

**Why F₀.₅ instead of accuracy?** F₀.₅ weights precision **twice** as heavily
as recall. This directly penalizes false positives — in a closed-loop BCI
context, a false spike detection triggers an unintended control action, which
is worse than missing a spike (which just means a slightly slower response).

**Comparison:**

| Metric | Formula | FP sensitivity | Use case |
|--------|---------|----------------|----------|
| Accuracy | $\frac{TP}{TP+FP+FN}$ | Moderate | General sorting quality |
| Precision | $\frac{TP}{TP+FP}$ | High | False positive rate only |
| Recall | $\frac{TP}{TP+FN}$ | None | Missed spike rate only |
| **F₀.₅** | $\frac{1.25 \cdot P \cdot R}{0.25P + R}$ | **High** | **BCI control — FPs are costly** |
| F₁ | $\frac{2PR}{P+R}$ | Equal to FN | Balanced applications |

#### All reported metrics

Every trial records and exports these values (averaged across scenarios):

| Metric | Key | Description |
|--------|-----|-------------|
| F₀.₅ | `f_half` | Primary objective (precision-weighted) |
| Accuracy | `accuracy` | TP/(TP+FP+FN) — SpikeInterface standard |
| Precision | `precision` | TP/(TP+FP) — false positive rate |
| Recall | `recall` | TP/(TP+FN) — missed spike rate |
| Active neurons | `n_active` | L1 neurons that fired ≥1 spike |
| Total spikes | `total_spikes` | Total L1 spike events across all neurons |
| Runtime | `runtime_s` | Wall-clock time for the trial |

### 2.5 Early Termination

Two mechanisms prevent wasting compute on dead configurations:

1. **Threshold reachability check** — Before processing any data,
   `evaluate_pipeline()` estimates the maximum steady-state membrane
   potential $V_{ss}$ given the current parameters. If
   $V_{ss} < 0.8 \times \text{threshold}$, the trial immediately returns
   `accuracy=0.0` without processing the full recording. This catches
   configurations where the L1 neurons can never fire.

2. **Multi-scenario early exit** — If the first scenario (S1) scores 0.0,
   `multi_evaluate()` skips the remaining 3 scenarios and returns zeros.
   Since S1 is the easiest scenario, if it fails, the others will too.

Together, these save ~50–60% of the optimization budget that would otherwise
be wasted on non-viable configurations.

---

## 3. Stage 1: Optuna TPE Search

### 3.1 How It Works

[Optuna](https://optuna.org/) uses a **Tree-structured Parzen Estimator**
(TPE) to model the relationship between hyperparameters and performance.
Unlike grid or random search, TPE builds a probabilistic model of which
parameter regions produce good results and samples more densely from those
regions as the study progresses.

```
Trial 1–10:   Mostly random exploration
Trial 10–30:  TPE model forms, starts biasing toward good regions
Trial 30–80:  Focused exploitation of promising parameter subspaces
```

Each trial:
1. TPE suggests a parameter vector
2. `multi_evaluate()` runs the pipeline on 4 synthetic recordings
3. The mean F₀.₅ is returned to Optuna
4. TPE updates its density model

### 3.2 Search Space

17 parameters across 6 functional groups:

| Group | Parameter | Type | Range | Description |
|-------|-----------|------|-------|-------------|
| **Encoder** | `enc_overlap` | int | 4–14 | Amplitude bin overlap factor |
| | `enc_dvm_factor` | float | 1.0–3.0 | Noise scale for RF half-width |
| | `enc_step_size` | int | 2–4 | Temporal subsampling stride |
| **DN** | `dn_threshold_factor` | log | 0.15–0.60 | Detection threshold scaling |
| | `dn_depression_tau` | int | 100–800 (step 50) | pRel recovery time constant |
| | `dn_depression_frac` | float | 0.005–0.05 | pRel depression depth per spike |
| **L1/STDP** | `l1_dn_weight` | float | 40–120 | DN excitatory boost to L1 |
| | `l1_stdp_ltp` | log | 0.001–0.02 | Long-term potentiation amplitude |
| | `l1_stdp_ltd` | float | −0.012–−0.0005 | Long-term depression amplitude |
| | `l1_n_neurons` | int | 20–120 (step 10) | Number of template neurons |
| **Inhibition** | `inh_duration_ms` | float | 1.0–10.0 | Post-spike blanking window |
| | `inh_strength_threshold` | float | 50–300 | Blanking bypass current |
| **Noise gate** | `ng_process_noise` | log | 0.001–0.1 | Kalman filter process noise Q |
| | `ng_inhibit_below_sd` | float | 1.2–3.5 | Noise gate sensitivity |
| | `ng_suppression_factor` | float | 0.0–0.3 | Max suppression strength |
| | `ng_ema_alpha` | float | 0.05–0.3 | EMA smoothing speed |
| **DEC** | `dec_dn_window_ms` | float | 0.5–5.0 | Post-DN integration window |

**Parameter types:**
- `float` — uniform continuous sampling
- `log` — log-uniform sampling (for parameters spanning orders of magnitude)
- `int` — integer sampling with optional step size

### 3.3 Seeding from Prior Best

By default, the study's **first trial** (trial 0) is seeded from
`data/best_config.json`. This gives TPE a known-good starting point so it
doesn't waste early trials exploring dead regions. For any parameters in the
manifest that don't exist in the seed file, the midpoint of the search range
is used.

Disable with `--no-seed` if you want a clean slate.

### 3.4 Running

```bash
# Standard run (80 trials, seeded from best_config.json)
rm data/snn-spike-sorting.db && snn-optimize --n-trials 80

# Clean start without seeding
rm data/snn-spike-sorting.db && snn-optimize --n-trials 80 --no-seed

# Larger budget
rm data/snn-spike-sorting.db && snn-optimize --n-trials 200

# Custom manifest
snn-optimize --manifest path/to/manifest.yaml --n-trials 50
```

**Runtime estimate:** ~220 s/trial (4 scenarios × ~55 s each), so 80 trials ≈
**5 hours**. Dead trials terminate early (~5–15 s), so effective time is lower.

**Output:**
- `data/snn-spike-sorting.db` — Optuna SQLite study (can resume)
- `data/best_config.json` — Best trial parameters + metrics
- `data/trials.csv` — All completed trials with parameters and metrics

---

## 4. Stage 2: Genetic Crossover Optimizer

### 4.1 Design Rationale

TPE is excellent at broad exploration but can struggle with fine-tuning
because it samples each parameter independently. The genetic optimizer
addresses this by:

1. **Preserving good parameter *combinations*** — Block-level crossover keeps
   functionally coupled parameters together (e.g., STDP LTP and LTD always
   move as a pair).
2. **Small perturbations around known-good points** — Gaussian mutation with
   bounded strength explores the local neighborhood of elite configurations.
3. **Combinatorial exploration** — By crossing gene blocks from different
   parents, the optimizer tests whether combining the encoder settings from
   parent A with the STDP settings from parent B yields something better than
   either alone.

### 4.2 Gene Blocks

Parameters are grouped into **6 functional blocks** that reflect the pipeline
architecture. During crossover, entire blocks are inherited from one parent
or the other — individual parameters within a block always travel together.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Gene Block    │  Parameters                                       │
├────────────────┼───────────────────────────────────────────────────┤
│  ENC (encoder) │  enc_overlap, enc_dvm_factor, enc_step_size       │
│  DN (attention)│  dn_threshold_factor, l1_dn_weight,               │
│                │  dn_depression_tau, dn_depression_frac            │
│  STDP (learn)  │  l1_stdp_ltp, l1_stdp_ltd, l1_n_neurons          │
│  INH (blanking)│  inh_duration_ms, inh_strength_threshold          │
│  NG (noise)    │  ng_process_noise, ng_inhibit_below_sd,           │
│                │  ng_suppression_factor, ng_ema_alpha               │
│  DEC (decoder) │  dec_dn_window_ms                                 │
└────────────────┴───────────────────────────────────────────────────┘
```

**Why these groupings?** Parameters within each block interact physically:
- **ENC:** Overlap and dvm_factor together determine the number of afferents
  and their spatial resolution. Changing one without the other produces
  incoherent encodings.
- **DN:** Threshold factor and dn_weight together determine whether the
  attention neuron fires and how strongly it drives L1. Depression parameters
  control how quickly it recovers after firing.
- **STDP:** LTP and LTD amplitudes must be balanced — a high LTP with no
  corresponding LTD saturates all weights at the ceiling. Neuron count
  interacts because more neurons means more competition for waveform templates.
- **INH:** Blanking duration and bypass threshold are tightly coupled — a long
  blanking window with a low bypass threshold effectively disables inhibition.
- **NG:** All four noise gate parameters jointly determine the Kalman filter's
  aggressiveness and smoothing behavior.

### 4.3 Selection, Crossover, and Mutation

**Selection:** The top-K trials (by F₀.₅ score) from Stage 1's `trials.csv`
are loaded as the elite pool. Parents are chosen via **fitness-proportional
selection** using a temperature-scaled softmax over scores:

$$w_i = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)}, \quad T = 0.1$$

where $s_i$ is the score of parent $i$. The low temperature ($T = 0.1$)
strongly favors the best parents without completely excluding weaker ones.

**Crossover:** For each of the 6 gene blocks, flip a biased coin
(default $p = 0.5$). If heads, take that block from parent B; otherwise keep
parent A's block. This produces offspring that are *chimeras* of two
successful configurations.

**Mutation:** Each parameter has a probability `mutation_rate` (default 0.3)
of being independently perturbed:

- **Float parameters:** $x' = x + \mathcal{N}(0,\; \sigma)$ where
  $\sigma = \text{strength} \times (x_{\max} - x_{\min})$, clamped to bounds.
- **Log parameters:** Mutation in log space (multiplicative):
  $x' = \exp(\ln x + \mathcal{N}(0,\; \sigma_{\log}))$
- **Integer parameters:** Gaussian perturbation rounded to nearest step.
- **Boolean parameters:** Flip with probability `mutation_rate`.

Default `mutation_strength=0.15` means the standard deviation of perturbation
is 15% of the parameter's total range — small enough to stay near good
configurations, large enough to escape local optima.

**Elitism:** Trial 0 always re-evaluates the best parent (no mutation).
This ensures the elite solution is preserved even if all offspring are worse.

### 4.4 Running

```bash
# Standard run (10 parents, 160 offspring)
snn-genetic --top-k 10 --n-offspring 160

# Aggressive exploration (more mutation)
snn-genetic --top-k 15 --n-offspring 200 --mutation-rate 0.5 --mutation-strength 0.25

# Conservative refinement (small perturbations around best)
snn-genetic --top-k 5 --n-offspring 80 --mutation-rate 0.2 --mutation-strength 0.08

# From a specific trials file
snn-genetic --trials data/some_other_trials.csv --top-k 10 --n-offspring 100
```

**Runtime estimate:** Same as Stage 1 (~220 s/trial), so 160 offspring ≈
**10 hours**. Dead-config early termination reduces effective time.

**Output:**
- `data/best_config.json` — Best overall (overwrites Stage 1's)
- `data/genetic_trials.csv` — All genetic trials with parameters and metrics

---

## 5. Interpreting Results

### 5.1 Output Files

| File | Format | Content |
|------|--------|---------|
| `data/best_config.json` | JSON | Best parameters, score, and all metrics |
| `data/trials.csv` | CSV | Stage 1 (Optuna) results: trial, value, params, metrics |
| `data/genetic_trials.csv` | CSV | Stage 2 (genetic) results: same format |
| `data/snn-spike-sorting.db` | SQLite | Optuna study database (can resume) |

**`best_config.json` schema:**
```json
{
  "study_name": "snn-spike-sorting",
  "best_trial": 42,
  "best_score": 0.65,
  "metric": "f_half",
  "parameters": {
    "dn_threshold_factor": 0.264,
    "l1_dn_weight": 81.1,
    "...": "..."
  },
  "user_attrs": {
    "accuracy": 0.62,
    "precision": 0.85,
    "recall": 0.55,
    "f_half": 0.65,
    "n_active": 18,
    "total_spikes": 450,
    "runtime_s": 220.0
  }
}
```

### 5.2 What to Look For

**Healthy trial distribution:**
- **< 30% dead trials** (score = 0.0) — if higher, the search space may be
  too wide or constraints too loose.
- **Top 10 trials within ~10% of each other** — indicates the optimizer has
  found a robust region, not a lucky outlier.
- **Precision ≥ 0.80** in the best trial — the F₀.₅ objective should push
  precision up. If precision is low, the pipeline is producing false positives.
- **Consistent per-scenario scores** — large variance across S1–S4 means the
  configuration is overfit to specific conditions.

**Warning signs:**
- Best trial has high recall but low precision → too many false positives.
- Only 1–3 active neurons → the pipeline may have degenerated to a single
  template matching everything.
- 100+ active neurons → neurons are firing independently, not specializing.
- Score drops after genetic optimization → mutation strength may be too high,
  or the elite pool was too small.

### 5.3 Dead Trials

"Dead trials" (accuracy = 0.0, F₀.₅ = 0.0) occur when L1 neurons **cannot
fire** — the membrane potential never reaches threshold. This happens when:

1. `l1_dn_weight` is too low and `enc_overlap` is too high → threshold is
   high but excitatory drive is weak.
2. `enc_step_size` is too high → temporal subsampling reduces the effective
   input frequency, lowering steady-state membrane voltage.
3. `dn_threshold_factor` is too high → the attention neuron fires rarely,
   starving L1 of excitatory boosts.

The threshold reachability check catches most dead trials early, but some
marginal configurations may run to completion and still score zero (e.g.,
threshold is just barely reachable but STDP drives weights to zero).

---

## 6. Pipeline Architecture for Optimization

### 6.1 Parameter Interactions

The SNN pipeline has complex parameter interactions that make optimization
non-trivial. Understanding these dependencies explains why certain parameter
combinations fail and why gene blocks are designed the way they are.

```
enc_overlap ─────┐
enc_dvm_factor ──┤── determines n_afferents & L1 threshold
enc_step_size ───┘            │
                               ▼
                    l1_dn_weight ──── must reach threshold
                               │
                    dn_threshold_factor ──── DN must fire to provide dn_weight
                               │
                    l1_stdp_ltp/ltd ──── learned weights scale afferent current
                               │
                    inh_duration_ms ──── post-spike blanking can suppress L1
                    inh_strength_threshold ── but strong signals bypass
                               │
                    ng_inhibit_below_sd ──── noise gate can suppress afferents
                    ng_suppression_factor ── maximum suppression depth
```

**The critical chain:** For L1 to fire at all, the steady-state membrane
potential must exceed the threshold. The threshold is derived from:

$$V_{ss} \approx \frac{I_{max}}{1 - \beta}, \quad \beta = e^{-1/\tau_m}$$

where $I_{max}$ is the peak input current (afferent drive + DN weight). The
threshold is set to $k \times V_{ss}^{nominal}$ with $k = 3$ (hardcoded).
If STDP drives weights too low, $I_{max}$ drops and firing stops entirely.

### 6.2 The Threshold Reachability Constraint

Before processing data, the evaluation function estimates whether L1 can
possibly fire:

$$V_{ss}^{est} = \frac{n_{active} \cdot w_{max} + w_{DN}}{1 - \beta}$$

If $V_{ss}^{est} < 0.8 \times \text{threshold}$, the trial is immediately
terminated with score 0.0. This prevents wasting ~60 seconds on a configuration
that can never produce output.

---

## 7. Design Decisions and Rationale

### Why F₀.₅ instead of accuracy?

Accuracy ($\frac{TP}{TP+FP+FN}$) treats false positives and missed spikes
equally. In a BCI control loop, false positives trigger unintended actions
(movement, stimulation), while a missed spike just means slightly reduced
responsiveness. F₀.₅ weights precision 2× over recall, aligning the
optimization objective with real-world deployment priorities.

### Why 2.0 ms tolerance and not the default 0.4 ms?

The SpikeInterface default of 0.4 ms is designed for high-resolution offline
sorters (wavelet-based, template matching on raw waveforms). Our pipeline uses
LIF membrane integration, which inherently introduces ~1.5–2 ms timing jitter
relative to the true spike peak. Using 0.4 ms would score the system at 0%
regardless of how good the templates are — the jitter is physical, not a bug.
2.0 ms captures the system's true sorting accuracy without inflating numbers.

### Why 4 scenarios and not 1 or 10?

- **1 scenario** (the old approach) allows overfitting to a single noise
  realization. We proved this: the old best config scored 0.95 on its training
  scenario but 0.28–0.31 on the other three.
- **10+ scenarios** would make each trial ~15 minutes, limiting the budget to
  ~20 trials in a reasonable time. With 17 parameters, 20 trials is far too few
  for meaningful optimization.
- **4 scenarios** is the sweet spot: diverse enough (varying noise, unit count,
  firing rates) to prevent overfitting, efficient enough (~220 s/trial) to
  allow 80+ trial budgets within ~5 hours.

### Why train/test split at 15 s of a 20 s recording?

- **75%/25% split** is standard in machine learning.
- **15 s of training** gives STDP enough time to learn waveform templates
  (~90–150 spikes per unit at 6–10 Hz).
- **5 s of testing** gives ~30–50 GT spikes per unit — enough for statistically
  meaningful precision/recall estimates.
- STDP remains active during testing (this is an online system), so the split
  tests online generalization, not offline-trained performance.

### Why block-level crossover instead of parameter-level?

Parameter-level crossover (swapping individual parameters) can break
functional couplings. For example, taking `l1_stdp_ltp=0.018` from parent A
and `l1_stdp_ltd=−0.001` from parent B creates a severe LTP/LTD imbalance
that neither parent had. Block-level crossover ensures that LTP and LTD
always come from the same parent, preserving the balance that made that parent
successful.

---

## 8. Extending the Search Space

To add a new parameter to the optimization:

### Step 1: Add the config field

In `src/snn_agent/config.py`, add the parameter to the relevant `*Config`
dataclass:

```python
@dataclass(frozen=True, slots=True)
class L1Config:
    n_neurons: int = 110
    my_new_param: float = 0.5    # ← add here with a sensible default
```

### Step 2: Add the flat-key mapping

In `config.py`, add an entry to `_FLAT_MAP`:

```python
_FLAT_MAP = {
    ...
    "l1_my_new_param": ("l1", "my_new_param"),    # ← add this line
}
```

### Step 3: Add to the manifest

In `docs/optimization_manifest.yaml`, under `parameters:`:

```yaml
  l1_my_new_param:
    type: float          # float, log, int, or bool
    low: 0.1
    high: 1.0
    description: "Description of what this controls"
```

### Step 4: Add to a gene block (optional, for genetic optimizer)

In `src/snn_agent/eval/genetic.py`, add the parameter to the appropriate
`GENE_BLOCKS` entry:

```python
GENE_BLOCKS = {
    "STDP": ["l1_stdp_ltp", "l1_stdp_ltd", "l1_n_neurons", "l1_my_new_param"],
    ...
}
```

### Step 5: Wire it in the pipeline

Make sure the parameter is actually used. Check `pipeline.py`,
`complete_pipeline()`, and the relevant core module.

---

## 9. Performance History

### Evaluation methodology evolution

| Version | delta_time | Scenarios | Train/test split | Metric | Notes |
|---------|-----------|-----------|------------------|--------|-------|
| v1 (initial) | 10.0 ms | 1 (seed=42) | None | accuracy | Inflated numbers |
| **v2 (current)** | **2.0 ms** | **4 diverse** | **15s/5s** | **F₀.₅** | Honest evaluation |

### Best known results

| Stage | Metric | Best F₀.₅ | Accuracy | Precision | Recall | Config |
|-------|--------|-----------|----------|-----------|--------|--------|
| v1 Optuna | accuracy@10ms | — | 0.824 | 0.973 | 0.846 | Trial 77 |
| v1 → v2 re-eval | f_half@2ms×4scen | 0.488 | 0.459 | 0.472 | 0.568 | Same config |

**Interpretation:** The v1 best config achieved 0.82 accuracy on a single
scenario at 10 ms tolerance. Under the honest v2 evaluation (4 scenarios,
2 ms, train/test split), the same config scores 0.49 F₀.₅. This 40-point
gap is the "overfitting tax" — the difference between memorizing one signal
and generalizing to diverse conditions.

The v2 optimization is designed to close this gap by finding configurations
that perform well across all 4 scenarios simultaneously. Expected behavior:
v2 scores will start lower than v1 but represent genuine, transferable
performance.
