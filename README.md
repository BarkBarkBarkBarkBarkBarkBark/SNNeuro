# ⚡ SNN Agent — Real-time Spiking Neural Network for Neural Signals

A streaming spiking neural network that detects and classifies extracellular
action potentials in real time, derived from the
[MB2018 ANNet](MB2018%20-%20ANNet/read_me.txt) architecture.

```
raw signal → bandpass → decimate → encode → attention neuron → template layer → control decoder
              300–6 kHz   80→20 kHz   temporal RF   LIF + pRel     60 LIF + STDP    rate/pop/trigger
```

## Quick start

```bash
# 1. Install (once)
./run.sh install

# 2. Edit parameters, then auto-launch
./run.sh config

# 3. Open the live raster in your browser
open http://localhost:8080
```

## Run modes

| Command | What it does |
|---|---|
| `./run.sh` | Start the agent (reads `mode` from `config.py`) |
| `./run.sh config` | Open `config.py` in `$EDITOR`, launch on save |
| `./run.sh lsl <ncs_path>` | Replay a `.ncs` file over LSL + start agent |
| `./run.sh electrode` | Start agent + synthetic test sender |
| `./run.sh evaluate [json]` | Offline pipeline evaluation with optional config overrides |
| `./run.sh optimize [opts]` | Hyperparameter optimization (Optuna) |
| `./run.sh install` | Create `.venv`, install dependencies |
| `./run.sh help` | Show all options |

### LSL mode (replay a recording)

```bash
./run.sh lsl raw_data/CSC285_0001.ncs
```

This launches **two processes** in one terminal:
1. `lsl_player.py` — streams the NCS file over Lab Streaming Layer
2. `server.py` — connects to the LSL stream and runs the full pipeline

### Electrode mode (synthetic or live UDP)

```bash
# With the built-in synthetic test signal:
./run.sh electrode --snr 5 --duration 30

# Or start agent alone and feed it real UDP samples:
./run.sh
# In another terminal, send samples to UDP port 9001
```

**UDP frame format:** `struct.pack('!Hf', 0xABCD, sample_float32)`

## Configuration

All parameters live in a single file: **`config.py`**.

```bash
./run.sh config   # opens in $EDITOR, relaunches after save
```

Key sections:

| Section | What to tweak |
|---|---|
| **Mode** | `"electrode"` (UDP input) or `"lsl"` (LSL stream) |
| **Preprocessing** | `enable_bandpass`, `bandpass_lo/hi_hz`, `enable_decimation`, `decimation_factor` |
| **Encoding** | `enc_overlap`, `enc_window_depth`, `enc_step_size` — receptive field shape |
| **Attention (DN)** | `dn_threshold_factor`, `dn_depression_tau` — spike detection sensitivity |
| **Template (L1)** | `l1_n_neurons`, `l1_stdp_ltp/ltd`, `l1_freeze_stdp` — learning and capacity |
| **Control decoder** | `ctrl_strategy` (`"rate"` / `"population"` / `"trigger"`), window, threshold |
| **Ports** | WebSocket, HTTP, UDP in/out — all configurable |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Input                                              │
│  • UDP electrode samples  (electrode mode)          │
│  • LSL stream / .ncs file (lsl mode)                │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  Preprocessor            │  encoder.py
│  IIR bandpass 300–6 kHz  │
│  Decimation 80→20 kHz    │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  SpikeEncoder            │  encoder.py
│  Amplitude bins + shift  │
│  register → bool vector  │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  AttentionNeuron (DN)    │  encoder.py
│  LIF + synaptic release  │
│  probability (pRel)      │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  TemplateLayer (L1)      │  snn.py
│  60 LIF neurons, WTA     │
│  snnTorch + custom STDP  │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  ControlDecoder          │  decoder.py
│  rate / population /     │
│  trigger → UDP out       │
└──────────────┬───────────┘
          ┌────┴────┐
          ▼         ▼
     UDP out     WebSocket
   (port 9002)  (port 8765)
       │            │
   Experiment    Browser
   hardware     spike raster
               localhost:8080
```

## Files

| File | Purpose |
|---|---|
| `run.sh` | Launch helper — install, config, run, evaluate, optimize |
| `config.py` | All tunable parameters (single source of truth) |
| `server.py` | Async event loop — glues the pipeline, serves UI |
| `encoder.py` | `Preprocessor`, `SpikeEncoder`, `AttentionNeuron` |
| `snn.py` | `TemplateLayer` — snnTorch LIF + STDP |
| `decoder.py` | `ControlDecoder` — spike → control signal |
| `evaluate.py` | Offline pipeline scorer against ground truth |
| `optimize.py` | Optuna hyperparameter search driver |
| `optimization_manifest.yaml` | Search space + trial budget definition |
| `index.html` | Browser-based live spike raster visualisation |
| `lsl_player.py` | Replay `.ncs` files over Lab Streaming Layer |
| `test_electrode.py` | Synthetic electrode signal generator (UDP) |

## Dependencies

```
numpy  scipy  torch  snntorch  websockets  mne  mne-lsl  spikeinterface  optuna  pyyaml
```

Install everything with `./run.sh install` or `pip install -r requirements.txt`.

## Activation Functions & Dynamics

Mathematical specification of every learnable and non-learnable activation
in the pipeline, derived from the [ANNet architecture](annet_architecture.yaml).

### Temporal Receptive Field Encoding (SpikeEncoder)

**Noise estimation** — Median Absolute Deviation during calibration:

$$\hat{\sigma} = \frac{\mathrm{median}(|x_1|, |x_2|, \dots, |x_N|)}{0.6745}$$

After calibration, $\hat{\sigma}$ is tracked with an exponential moving average:

$$\hat{\sigma}_{t} = \hat{\sigma}_{t-1} + \alpha \left(|x_t| - \hat{\sigma}_{t-1}\right)$$

**Amplitude bin half-width:**

$$\Delta v_m = \gamma \cdot \hat{\sigma}$$

where $\gamma$ = `enc_dvm_factor`. Bin centres are spaced by $\frac{2 \Delta v_m}{O}$ with $O$ = `enc_overlap`.

**Afferent activation** — bin $j$ fires when:

$$|c_j - x_t| \leq \Delta v_m$$

The afferent vector is the flattened shift register $\mathbf{a} \in \{0,1\}^{N_c \times D}$
where $N_c$ is the number of centres and $D$ = `enc_window_depth`.

---

### Attention Neuron (DN) — LIF with Synaptic Depression

**Membrane dynamics** (leaky integrate-and-fire):

$$V_t = V_{t-1} \cdot e^{-1/\tau_m} + \sum_{j \in \mathcal{A}_t} p_{\mathrm{rel},j}(t)$$

where $\mathcal{A}_t$ is the set of active afferents at time $t$.

**Synaptic release probability** (short-term depression):

$$p_{\mathrm{rel},j}(t) = 1 - \left(1 - p_{\mathrm{rel},j}(t_{\mathrm{last}})\right) e^{-\Delta t / \tau_d}$$

After contributing, each synapse is depressed:

$$p_{\mathrm{rel},j} \leftarrow p_{\mathrm{rel},j} \cdot (1 - f_d)$$

where $\tau_d$ = `dn_depression_tau` and $f_d$ = `dn_depression_frac`.

**Threshold:**

$$\theta_{\mathrm{DN}} = \frac{F_{\mathrm{th}} \cdot O \cdot D}{1 - e^{-1/\tau_m}}$$

where $F_{\mathrm{th}}$ = `dn_threshold_factor`.

**Reset** (set, not additive):

$$V \leftarrow V_{\mathrm{reset}} = F_r \cdot \left(e^{1/\tau_m} - 1\right) \cdot \theta_{\mathrm{DN}}$$

---

### Template Layer (L1) — LIF + WTA + STDP

**Input current:**

$$I_t = \mathbf{a}_t^{\top} \mathbf{W} + s_{\mathrm{DN}}(t) \cdot w_{\mathrm{DN}}$$

where $\mathbf{W} \in \mathbb{R}^{N_a \times N}$ are the learnable weights and
$w_{\mathrm{DN}}$ = `l1_dn_weight`.

**LIF membrane** (via snnTorch):

$$V_{i,t} = \beta \, V_{i,t-1} + I_{i,t}, \qquad \beta = e^{-1/\tau_m}$$

**Spike and reset:**

$$s_{i,t} = \begin{cases} 1 & \text{if } V_{i,t} \geq \theta_{L1} \\ 0 & \text{otherwise} \end{cases}, \qquad V_{i,t} \leftarrow 0 \;\text{ on spike (WTA)}$$

**L1 threshold:**

$$\theta_{L1} = \frac{\left(w_{\mathrm{DN}} + O \cdot (D - k)\right) \left(1 - e^{-S/\tau_m}\right)}{1 - e^{-1/\tau_m}}$$

where $k = 3$ and $S$ = `enc_step_size`.

**Winner-take-all:** snnTorch's `inhibition=True` — only the neuron with the
highest membrane potential fires per timestep.

**Asymmetric STDP** (applied to the winner $w$):

- *LTD* — global depression at every post-synaptic spike:

$$\Delta W_{j,w}^{-} = \eta^{-} \quad \forall\; j$$

- *LTP* — potentiation for recently active pre-synaptic afferents:

$$\Delta W_{j,w}^{+} = \eta^{+} \quad \text{if } (t - t_j^{\mathrm{pre}}) \leq T_{\mathrm{LTP}}$$

- Weight clip: $W_{j,w} \in [w_{\min},\; w_{\max}]$

where $\eta^{+}$ = `l1_stdp_ltp`, $\eta^{-}$ = `l1_stdp_ltd`,
$T_{\mathrm{LTP}}$ = `l1_stdp_ltp_window`.

---

### Control Decoder

**Rate strategy** — sliding-window weighted sum:

$$u_t = \mathrm{clip}\!\left(\sum_{i=1}^{N} w_i \cdot \frac{n_{i,t}}{T_w},\; -1,\; 1\right)$$

where $n_{i,t}$ counts spikes of neuron $i$ over the window of length $T_w$.

**Population strategy** — leaky integrator:

$$\mathcal{I}_t = \mathcal{I}_{t-1} \cdot e^{-1/\tau_c} + \mathbf{w}^{\top} \mathbf{s}_t$$

Emits when $\mathcal{I}_t \geq \theta_c$.

**DN confidence** — attention-gated quality measure:

$$\mathrm{conf}_t = \frac{1}{T_{\mathrm{dn}}} \sum_{k=t-T_{\mathrm{dn}}}^{t} s_{\mathrm{DN}}(k)$$

## Hyperparameter Optimization

Automated search over the 8 most impactful parameters using
[Optuna](https://optuna.org/) (TPE sampler) scored against SpikeInterface
ground-truth synthetic recordings.

### Quick start

```bash
# One-shot evaluation with specific overrides
./run.sh evaluate '{"l1_n_neurons": 40}'

# Full optimization (80 trials, 4 parallel workers)
./run.sh optimize

# Custom budget
./run.sh optimize --n-trials 40 --n-jobs 2
```

### How it works

1. **`optimization_manifest.yaml`** defines the search space (8 parameters,
   bounds, types) and trial budget.
2. **`evaluate.py`** generates a synthetic recording via SpikeInterface,
   runs the full pipeline offline (Preprocessor → Encoder → DN → L1 → Decoder),
   and scores L1 clusters against ground truth.
3. **`optimize.py`** drives an Optuna study: each trial samples a parameter
   set, calls `evaluate_pipeline()`, and reports the accuracy score back.
   TPE learns which regions of the space are promising.

### Outputs

| File | Contents |
|---|---|
| `best_config.json` | Best trial's parameters + score |
| `trials.csv` | Full history — every trial with params and metrics |
| `snn-spike-sorting.db` | SQLite Optuna study (supports resume) |

### Search space

| Parameter | Type | Range | Description |
|---|---|---|---|
| `dn_threshold_factor` | log | [0.15, 0.80] | DN firing threshold scale |
| `l1_dn_weight` | float | [15, 80] | DN excitatory boost to L1 |
| `l1_stdp_ltp` | log | [0.001, 0.02] | LTP amplitude |
| `l1_stdp_ltd` | float | [−0.012, −0.0005] | LTD amplitude |
| `enc_overlap` | int | [4, 20] | Amplitude bin overlap |
| `enc_dvm_factor` | float | [1.0, 3.5] | Noise scale for RF width |
| `enc_step_size` | int | [2, 8] | Temporal subsampling stride |
| `l1_n_neurons` | int | [20, 120] | Template neuron count |

## Multi-terminal workflow

If you prefer separate terminals for each process:

```bash
# Terminal 1 — LSL player (skip if using electrode mode)
.venv/bin/python lsl_player.py raw_data/CSC285_0001.ncs

# Terminal 2 — SNN agent
.venv/bin/python server.py

# Terminal 3 — browser
open http://localhost:8080
```

## License

Research use. See project documentation for details.
