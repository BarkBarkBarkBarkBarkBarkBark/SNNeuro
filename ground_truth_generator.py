"""
ground_truth_generator.py — Benchmarking tool for the SNN spike-sorting pipeline.

Usage patterns
--------------
1. **Standalone benchmark** (no server, no browser):
       python ground_truth_generator.py
   Generates synthetic data, runs the full SNN pipeline offline,
   and scores detected L1 clusters against ground truth.

2. **As a library** (import individual helpers):
       from ground_truth_generator import make_single_channel_ground_truth, score_detector
       recording, sorting, trains = make_single_channel_ground_truth(...)

3. **Live visualization** (via server.py):
       ./run.sh synthetic
   Uses the same spikeinterface recording generation but streams through
   the full server pipeline with live browser raster & waveform.
"""

import time
import numpy as np

import spikeinterface.full as si
import spikeinterface.comparison as sc
from spikeinterface.core import NumpySorting


# -----------------------------
# 1) Build a single-channel synthetic dataset
# -----------------------------
def make_single_channel_ground_truth(
    duration_s=20.0,
    fs=30_000.0,
    num_units=2,
    firing_rates=(6.0, 12.0),
    refractory_period_ms=4.0,
    noise_level=8.0,
    seed=42,
):
    """
    Returns
    -------
    recording : SpikeInterface Recording
        Single-channel synthetic extracellular recording
    sorting_true : SpikeInterface Sorting
        Ground-truth spike trains
    gt_unit_trains : dict
        unit_id -> spike sample indices
    """

    # Explicitly generate the spike trains first so the truth is easy to inspect
    sorting = si.generate_sorting(
        num_units=num_units,
        sampling_frequency=fs,
        durations=[duration_s],
        firing_rates=list(firing_rates),
        refractory_period_ms=refractory_period_ms,
        seed=seed,
    )

    # Build a 1-channel recording with injected templates + noise
    # ms_before / ms_after must be large enough for the randomly-drawn
    # template waveforms — 3 ms each side is safe for 30 kHz.
    recording, sorting_true = si.generate_ground_truth_recording(
        durations=[duration_s],
        sampling_frequency=fs,
        num_channels=1,
        sorting=sorting,
        ms_before=3.0,
        ms_after=4.0,
        noise_kwargs={
            "noise_levels": noise_level,
            "strategy": "on_the_fly",
        },
        dtype="float32",
        seed=seed,
    )

    gt_unit_trains = {
        unit_id: sorting_true.get_unit_spike_train(unit_id=unit_id, segment_index=0)
        for unit_id in sorting_true.unit_ids
    }

    return recording, sorting_true, gt_unit_trains


# -----------------------------
# 2) Iterate over the recording in real-time-like chunks
# -----------------------------
def iter_realtime_chunks(recording, chunk_ms=1.0, pace_realtime=False):
    """
    Yields (start_frame, chunk) where chunk has shape (num_samples,).
    """
    fs = recording.get_sampling_frequency()
    chunk_size = int(round(fs * chunk_ms / 1000.0))
    n_frames = recording.get_num_frames(segment_index=0)

    t0 = time.perf_counter()

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)

        # SpikeInterface returns shape (num_samples, num_channels) for traces
        traces = recording.get_traces(
            segment_index=0,
            start_frame=start,
            end_frame=end,
            channel_ids=recording.channel_ids,
        )

        # single-channel -> flatten to 1D
        chunk = traces[:, 0]

        if pace_realtime:
            expected_elapsed = end / fs
            actual_elapsed = time.perf_counter() - t0
            sleep_time = expected_elapsed - actual_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        yield start, chunk


# -----------------------------
# 3) Example detector placeholder
#    Replace with your real detector/classifier
# -----------------------------
def simple_negative_threshold_detector(recording, threshold_uV=-35.0, refractory_ms=1.0):
    """
    Very naive one-unit detector.
    Returns a dict suitable for NumpySorting.from_unit_dict().
    """
    fs = recording.get_sampling_frequency()
    traces = recording.get_traces(segment_index=0)[:, 0]

    refractory_samples = int(round(refractory_ms * fs / 1000.0))

    spike_samples = []
    last_spike = -np.inf

    for i in range(1, len(traces) - 1):
        is_local_min = traces[i] < traces[i - 1] and traces[i] <= traces[i + 1]
        crosses = traces[i] < threshold_uV

        if crosses and is_local_min:
            if i - last_spike >= refractory_samples:
                spike_samples.append(i)
                last_spike = i

    # one detected unit called 0
    return {0: np.asarray(spike_samples, dtype=np.int64)}


# -----------------------------
# 4) Compare any detector output to ground truth
# -----------------------------
def score_detector(recording, sorting_true, detected_units_dict):
    """
    Wraps spikeinterface's compare_sorter_to_ground_truth.

    Parameters
    ----------
    detected_units_dict : dict[int, np.ndarray]
        unit_id → array of spike-time sample indices.

    Returns
    -------
    cmp  : GroundTruthComparison
    perf : pd.DataFrame   (columns: accuracy, recall, precision, …)
    """
    fs = recording.get_sampling_frequency()

    tested_sorting = NumpySorting.from_unit_dict(
        units_dict_list=detected_units_dict,
        sampling_frequency=fs,
    )

    cmp = sc.compare_sorter_to_ground_truth(
        gt_sorting=sorting_true,
        tested_sorting=tested_sorting,
        exhaustive_gt=True,
        delta_time=10.0,
    )

    perf = cmp.get_performance()
    return cmp, perf


# -----------------------------
# 5) Run the full SNN pipeline offline and score it
# -----------------------------
def benchmark_snn_pipeline(recording, sorting_true, cfg_overrides=None):
    """
    Imports the SNN modules, feeds every sample through
    Preprocessor → Encoder → AttentionNeuron → TemplateLayer,
    logs L1 spike times, and scores them against *sorting_true*.

    Returns
    -------
    cmp       : GroundTruthComparison | None
    perf      : pd.DataFrame | None
    spike_log : dict[int, list[int]]   neuron_id → [frame_indices]
    """
    from config import CFG
    from encoder import Preprocessor, SpikeEncoder, AttentionNeuron
    from snn import TemplateLayer
    from decoder import ControlDecoder

    cfg = dict(CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    fs = recording.get_sampling_frequency()
    cfg["sampling_rate_hz"] = int(fs)

    preproc = Preprocessor(cfg)
    enc_cfg = dict(cfg)
    enc_cfg["sampling_rate_hz"] = preproc.effective_fs
    encoder = SpikeEncoder(enc_cfg)

    dn = None
    l1 = None
    decoder_obj = None

    traces = recording.get_traces(segment_index=0)[:, 0]
    n_total = len(traces)
    l1_spike_log: dict[int, list[int]] = {}
    step_count = 0
    t0 = time.perf_counter()

    for frame_idx in range(n_total):
        raw = float(traces[frame_idx])
        processed = preproc.step(raw)
        if not processed:
            continue
        for pp_sample in processed:
            step_count += 1
            afferents = encoder.step(pp_sample)
            if not encoder.is_calibrated:
                continue
            if dn is None:
                n_aff = encoder.n_afferents
                dn = AttentionNeuron(enc_cfg, n_aff)
                l1 = TemplateLayer(cfg, n_aff)
                decoder_obj = ControlDecoder(enc_cfg, cfg["l1_n_neurons"])
                print(f"  ✅ Encoder calibrated: {n_aff} afferents  "
                      f"(at {preproc.effective_fs} Hz)")

            dn_spike = dn.step(afferents)
            l1_spikes = l1.step(afferents, dn_spike)
            decoder_obj.step(l1_spikes, dn_spike)

            for idx in np.flatnonzero(l1_spikes):
                nid = int(idx)
                l1_spike_log.setdefault(nid, []).append(frame_idx)

        if frame_idx % 50_000 == 0 and frame_idx > 0:
            pct = 100 * frame_idx / n_total
            print(f"  ⏳ {pct:.0f}% ({frame_idx}/{n_total} frames)")

    elapsed = time.perf_counter() - t0
    n_active = sum(1 for v in l1_spike_log.values() if v)
    total_spikes = sum(len(v) for v in l1_spike_log.values())
    print(f"  🏁 Done in {elapsed:.1f}s — {step_count} processed samples, "
          f"{n_active} active L1 neurons, {total_spikes} total spikes")

    if not l1_spike_log:
        print("  ⚠ No L1 spikes detected — nothing to score.")
        return None, None, l1_spike_log

    l1_sorting_dict = {
        nid: np.array(times, dtype=np.int64)
        for nid, times in l1_spike_log.items()
        if len(times) > 0
    }

    l1_sorting = NumpySorting.from_unit_dict(
        units_dict_list=l1_sorting_dict,
        sampling_frequency=fs,
    )

    cmp = sc.compare_sorter_to_ground_truth(
        gt_sorting=sorting_true,
        tested_sorting=l1_sorting,
        exhaustive_gt=True,
        delta_time=10.0,
    )
    perf = cmp.get_performance()
    return cmp, perf, l1_spike_log


# -----------------------------
# 6) Demo run
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SNN Agent — Ground-Truth Benchmark")
    print("=" * 60)

    recording, sorting_true, gt_unit_trains = make_single_channel_ground_truth(
        duration_s=20.0,
        fs=30_000.0,
        num_units=2,
        firing_rates=(6.0, 10.0),
        refractory_period_ms=4.0,
        noise_level=8.0,
        seed=7,
    )

    print(f"\nRecording : {recording}")
    print(f"Sorting   : {sorting_true}")
    for uid, train in gt_unit_trains.items():
        rate = len(train) / 20.0
        print(f"  Unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")

    # ── Baseline: naive threshold detector ────────────────────────────
    print("\n── Naive threshold detector ──")
    detected_units = simple_negative_threshold_detector(
        recording,
        threshold_uV=-35.0,
        refractory_ms=1.0,
    )
    for uid, train in detected_units.items():
        print(f"  Detected unit {uid}: {len(train)} spikes")

    cmp_naive, perf_naive = score_detector(recording, sorting_true, detected_units)
    print("\nNaive detector performance:")
    print(perf_naive)

    # ── SNN pipeline benchmark ────────────────────────────────────────
    print("\n── SNN pipeline benchmark ──")
    cmp_snn, perf_snn, spike_log = benchmark_snn_pipeline(recording, sorting_true)

    if perf_snn is not None:
        print("\nSNN pipeline performance:")
        print(perf_snn)

        print(f"\n  Active L1 neurons: {len(spike_log)}")
        for nid in sorted(spike_log):
            print(f"    L1[{nid:2d}]: {len(spike_log[nid]):5d} spikes")
    else:
        print("  (no spikes to score)")

    # ── Save ground truth for external analysis ───────────────────────
    np.savez(
        "single_channel_ground_truth.npz",
        sampling_frequency=recording.get_sampling_frequency(),
        unit_ids=np.asarray(list(gt_unit_trains.keys())),
        **{f"unit_{u}": st for u, st in gt_unit_trains.items()},
    )