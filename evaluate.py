"""
evaluate.py — Offline SNN pipeline evaluation against ground truth.

Standalone scorer: generates (or loads cached) synthetic data, runs the full
SNN pipeline sample-by-sample, and compares detected L1 clusters to the
ground-truth spike trains via SpikeInterface.

Can be used:
    1. Directly:     python evaluate.py                    (uses default CFG)
    2. With overrides: python evaluate.py --overrides '{"l1_n_neurons": 40}'
    3. As a library:   from evaluate import evaluate_pipeline
"""

import json
import hashlib
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress snntorch's "Inhibition is an unstable feature" warning
warnings.filterwarnings("ignore", message="Inhibition is an unstable feature")
import spikeinterface.comparison as sc
from spikeinterface.core import NumpySorting

from ground_truth_generator import make_single_channel_ground_truth


# ─────────────────────────────────────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".cache"


class _ThresholdUnreachable(Exception):
    """Raised when L1 threshold is provably unreachable with given params."""
    pass


def _recording_cache_key(rec_params: dict) -> str:
    """Deterministic hash of recording parameters."""
    blob = json.dumps(rec_params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _get_or_generate_recording(rec_params: dict):
    """
    Return (recording, sorting_true, gt_unit_trains), using a disk cache
    so repeated evaluations skip the expensive generation step.
    """
    cache_key = _recording_cache_key(rec_params)
    cache_path = CACHE_DIR / f"synth_{cache_key}.npz"

    recording, sorting_true, gt_unit_trains = make_single_channel_ground_truth(
        duration_s=rec_params.get("duration_s", 20.0),
        fs=rec_params.get("fs", 30_000.0),
        num_units=rec_params.get("num_units", 2),
        firing_rates=tuple(rec_params.get("firing_rates", (6.0, 10.0))),
        noise_level=rec_params.get("noise_level", 8.0),
        seed=rec_params.get("seed", 42),
    )

    # Cache the traces for quick reload (informational — we regenerate
    # the SpikeInterface objects each time for full API compatibility)
    if not cache_path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        traces = recording.get_traces(segment_index=0)[:, 0]
        np.savez_compressed(
            cache_path,
            traces=traces,
            fs=recording.get_sampling_frequency(),
        )

    return recording, sorting_true, gt_unit_trains


# ─────────────────────────────────────────────────────────────────────────────
#  Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_pipeline(
    cfg_overrides: dict | None = None,
    rec_params: dict | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run the full SNN pipeline offline and score it against ground truth.

    Parameters
    ----------
    cfg_overrides : dict, optional
        Keys from config.py to override (e.g. {"l1_n_neurons": 40}).
    rec_params : dict, optional
        Synthetic recording parameters (duration_s, fs, num_units, …).
        Defaults come from optimization_manifest.yaml / config.py.
    verbose : bool
        Print progress updates.

    Returns
    -------
    dict with keys:
        accuracy   : float   mean accuracy across GT units (0–1)
        recall     : float   mean recall
        precision  : float   mean precision
        n_active   : int     number of L1 neurons that fired
        total_spikes : int   total L1 spikes
        runtime_s  : float   wall-clock seconds for the pipeline run
        perf_df    : pd.DataFrame  (full per-unit performance table)
    """
    from config import CFG
    from encoder import Preprocessor, SpikeEncoder, AttentionNeuron
    from snn import TemplateLayer
    from decoder import ControlDecoder

    # ── Build effective config ────────────────────────────────────────────
    cfg = dict(CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    # ── Generate / cache recording ────────────────────────────────────────
    if rec_params is None:
        rec_params = {
            "duration_s": cfg.get("synth_duration_s", 20.0),
            "fs": cfg.get("synth_fs", 30_000),
            "num_units": cfg.get("synth_num_units", 2),
            "noise_level": cfg.get("synth_noise_level", 8.0),
            "seed": cfg.get("synth_seed", 42),
        }

    recording, sorting_true, gt_unit_trains = _get_or_generate_recording(rec_params)
    fs = recording.get_sampling_frequency()
    cfg["sampling_rate_hz"] = int(fs)

    if verbose:
        for uid, train in gt_unit_trains.items():
            rate = len(train) / rec_params.get("duration_s", 20.0)
            print(f"  GT unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")

    # ── Instantiate pipeline components ───────────────────────────────────
    preproc = Preprocessor(cfg)
    enc_cfg = dict(cfg)
    enc_cfg["sampling_rate_hz"] = preproc.effective_fs
    encoder = SpikeEncoder(enc_cfg)

    dn = None
    l1 = None
    decoder_obj = None

    # ── Run sample-by-sample ──────────────────────────────────────────────
    traces = recording.get_traces(segment_index=0)[:, 0]
    n_total = len(traces)
    l1_spike_log: dict[int, list[int]] = {}
    step_count = 0
    t0 = time.perf_counter()
    _early_exit = False

    try:
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
                if verbose:
                    print(f"  ✅ Calibrated: {n_aff} afferents "
                          f"({preproc.effective_fs} Hz)")

                # ── Reachability check ────────────────────────────────
                # Estimate max steady-state membrane potential:
                #   V_ss ≈ max_current / (1 − β)
                # where max_current ≈ n_centers × w_max + dn_weight
                import torch as _torch
                _w_max = float(l1.W.max())
                _n_active_est = min(n_aff, int(cfg.get("enc_overlap", 10)
                                    * cfg.get("enc_window_depth", 10)))
                _max_current = _n_active_est * _w_max + l1.dn_weight
                _beta = float(np.exp(-1.0 / cfg.get("l1_tm_samples", 2)))
                _v_ss = _max_current / (1.0 - _beta)
                if _v_ss < l1.threshold * 0.8:
                    # Threshold is unreachable — skip the rest
                    if verbose:
                        print(f"  ⚠ L1 threshold {l1.threshold:.0f} "
                              f"unreachable (V_ss≈{_v_ss:.0f}) — skipping")
                    raise _ThresholdUnreachable(
                        f"L1 threshold {l1.threshold:.0f} unreachable "
                        f"(max V_ss≈{_v_ss:.0f})"
                    )

            dn_spike = dn.step(afferents)
            l1_spikes = l1.step(afferents, dn_spike)
            decoder_obj.step(l1_spikes, dn_spike)

            for idx in np.flatnonzero(l1_spikes):
                nid = int(idx)
                l1_spike_log.setdefault(nid, []).append(frame_idx)

        if verbose and frame_idx % 100_000 == 0 and frame_idx > 0:
            pct = 100 * frame_idx / n_total
            print(f"  ⏳ {pct:.0f}%")

    except _ThresholdUnreachable:
      _early_exit = True

    elapsed = time.perf_counter() - t0

    # ── Score against ground truth ────────────────────────────────────────
    n_active = sum(1 for v in l1_spike_log.values() if v)
    total_spikes = sum(len(v) for v in l1_spike_log.values())

    if verbose:
        print(f"  🏁 {elapsed:.1f}s — {n_active} active neurons, "
              f"{total_spikes} spikes")

    result = {
        "accuracy": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "n_active": n_active,
        "total_spikes": total_spikes,
        "runtime_s": elapsed,
        "perf_df": None,
    }

    if not l1_spike_log:
        return result

    l1_sorting_dict = {
        nid: np.array(times, dtype=np.int64)
        for nid, times in l1_spike_log.items()
        if len(times) > 0
    }

    l1_sorting = NumpySorting.from_unit_dict(
        units_dict_list=l1_sorting_dict,
        sampling_frequency=fs,
    )

    # delta_time (ms): matching tolerance for spike alignment.
    # The streaming pipeline introduces latency (encoder shift register,
    # DN integration, causal filter delay) so we widen from the default
    # 0.4 ms to 10 ms to allow for this systematic delay.
    cmp = sc.compare_sorter_to_ground_truth(
        gt_sorting=sorting_true,
        tested_sorting=l1_sorting,
        exhaustive_gt=True,
        delta_time=10.0,
    )
    perf = cmp.get_performance()

    result["accuracy"]  = float(perf["accuracy"].mean())
    result["recall"]    = float(perf["recall"].mean())
    result["precision"] = float(perf["precision"].mean())
    result["perf_df"]   = perf

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SNN pipeline offline")
    parser.add_argument(
        "--overrides", type=str, default="{}",
        help='JSON dict of config overrides, e.g. \'{"l1_n_neurons": 40}\'',
    )
    args = parser.parse_args()

    overrides = json.loads(args.overrides)

    print("=" * 60)
    print("SNN Agent — Offline Evaluation")
    print("=" * 60)

    if overrides:
        print(f"Config overrides: {overrides}")

    metrics = evaluate_pipeline(cfg_overrides=overrides, verbose=True)

    print("\n── Results ──")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Active L1 : {metrics['n_active']}")
    print(f"  Total spk : {metrics['total_spikes']}")
    print(f"  Runtime   : {metrics['runtime_s']:.1f}s")

    if metrics["perf_df"] is not None:
        print("\nPer-unit performance:")
        print(metrics["perf_df"].to_string())
