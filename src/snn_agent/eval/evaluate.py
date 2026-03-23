"""
snn_agent.eval.evaluate — Offline SNN pipeline evaluation against ground truth.

Generates synthetic data, runs the full pipeline sample-by-sample,
and scores L1 clusters against ground-truth spike trains via SpikeInterface.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np

# Suppress snntorch "Inhibition is an unstable feature" warning
warnings.filterwarnings("ignore", message="Inhibition is an unstable feature")
import spikeinterface.comparison as sc
from spikeinterface.core import NumpySorting

from snn_agent.config import Config, DEFAULT_CONFIG
from snn_agent.core.pipeline import build_pipeline, complete_pipeline
from snn_agent.eval.ground_truth import make_single_channel_ground_truth

__all__ = ["evaluate_pipeline"]

class _ThresholdUnreachable(Exception):
    pass


def _get_or_generate_recording(rec_params: dict):
    return make_single_channel_ground_truth(
        duration_s=rec_params.get("duration_s", 20.0),
        fs=rec_params.get("fs", 30_000.0),
        num_units=rec_params.get("num_units", 2),
        firing_rates=tuple(rec_params.get("firing_rates", (6.0, 10.0))),
        noise_level=rec_params.get("noise_level", 8.0),
        seed=rec_params.get("seed", 42),
    )


def evaluate_pipeline(
    cfg_overrides: dict | None = None,
    rec_params: dict | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run the full SNN pipeline offline and score against ground truth.

    Parameters
    ----------
    cfg_overrides : dict, optional
        Flat-key overrides (e.g. ``{"l1_n_neurons": 40}``).
    rec_params : dict, optional
        Synthetic recording parameters.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with accuracy, recall, precision, n_active, total_spikes, runtime_s, perf_df.
    """
    # Build config with overrides
    if cfg_overrides:
        cfg = Config.from_flat({**DEFAULT_CONFIG.to_dict_flat(), **cfg_overrides})
    else:
        cfg = DEFAULT_CONFIG

    # Recording params
    if rec_params is None:
        syn = cfg.synthetic
        rec_params = {
            "duration_s": syn.duration_s,
            "fs": syn.fs,
            "num_units": syn.num_units,
            "noise_level": syn.noise_level,
            "seed": syn.seed,
        }

    recording, sorting_true, gt_unit_trains = _get_or_generate_recording(rec_params)
    fs = recording.get_sampling_frequency()
    cfg = cfg.with_overrides(sampling_rate_hz=int(fs))

    if verbose:
        for uid, train in gt_unit_trains.items():
            rate = len(train) / rec_params.get("duration_s", 20.0)
            print(f"  GT unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")

    # Build pipeline
    preproc, encoder, effective_cfg = build_pipeline(cfg)
    pipeline_obj = None

    traces = recording.get_traces(segment_index=0)[:, 0]
    n_total = len(traces)
    l1_spike_log: dict[int, list[int]] = {}
    l2_spike_log: dict[int, list[int]] = {}
    step_count = 0
    t0 = time.perf_counter()
    _early_exit = False
    any_l1_fired_prev = False

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

                if pipeline_obj is None:
                    pipeline_obj = complete_pipeline(
                        cfg, effective_cfg, preproc, encoder
                    )
                    if verbose:
                        print(
                            f"  ✅ Calibrated: {encoder.n_afferents} afferents "
                            f"({preproc.effective_fs} Hz)"
                        )

                    # Reachability check
                    _w_max = float(pipeline_obj.template.W.max())
                    _n_active_est = min(
                        encoder.n_afferents,
                        int(cfg.encoder.overlap * cfg.encoder.window_depth),
                    )
                    _max_current = _n_active_est * _w_max + pipeline_obj.template.dn_weight
                    _beta = float(np.exp(-1.0 / cfg.l1.tm_samples))
                    _v_ss = _max_current / (1.0 - _beta)
                    if _v_ss < pipeline_obj.template.threshold * 0.8:
                        if verbose:
                            print(
                                f"  ⚠ L1 threshold "
                                f"{pipeline_obj.template.threshold:.0f} "
                                f"unreachable (V_ss≈{_v_ss:.0f}) — skipping"
                            )
                        raise _ThresholdUnreachable()

                dn_spike = pipeline_obj.attention.step(afferents)

                # Noise gate suppression
                suppression = 1.0
                if pipeline_obj.noise_gate is not None:
                    suppression = pipeline_obj.noise_gate.step(pp_sample)

                # Global inhibition
                if pipeline_obj.inhibitor is not None:
                    max_current = float(encoder.n_afferents)
                    inh_factor = pipeline_obj.inhibitor.gate(max_current, any_l1_fired_prev)
                    suppression *= inh_factor

                l1_spikes = pipeline_obj.template.step(afferents, dn_spike, suppression)
                any_l1_fired_prev = bool(np.any(l1_spikes))

                # Optional L2
                decoder_input = l1_spikes
                if pipeline_obj.output_layer is not None:
                    l2_spikes = pipeline_obj.output_layer.step(l1_spikes)
                    for idx in np.flatnonzero(l2_spikes):
                        nid = int(idx)
                        l2_spike_log.setdefault(nid, []).append(frame_idx)
                    decoder_input = l2_spikes

                pipeline_obj.decoder.step(decoder_input, dn_spike)

                for idx in np.flatnonzero(l1_spikes):
                    nid = int(idx)
                    l1_spike_log.setdefault(nid, []).append(frame_idx)

            if verbose and frame_idx % 100_000 == 0 and frame_idx > 0:
                pct = 100 * frame_idx / n_total
                print(f"  ⏳ {pct:.0f}%")

    except _ThresholdUnreachable:
        _early_exit = True

    elapsed = time.perf_counter() - t0

    n_active = sum(1 for v in l1_spike_log.values() if v)
    total_spikes = sum(len(v) for v in l1_spike_log.values())

    if verbose:
        print(
            f"  🏁 {elapsed:.1f}s — {n_active} active neurons, "
            f"{total_spikes} spikes"
        )

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

    cmp = sc.compare_sorter_to_ground_truth(
        gt_sorting=sorting_true,
        tested_sorting=l1_sorting,
        exhaustive_gt=True,
        delta_time=10.0,
    )
    perf = cmp.get_performance()

    result["accuracy"] = float(perf["accuracy"].mean())
    result["recall"] = float(perf["recall"].mean())
    result["precision"] = float(perf["precision"].mean())
    result["perf_df"] = perf

    return result


def main() -> None:
    """CLI entry point (``snn-evaluate``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SNN pipeline offline")
    parser.add_argument(
        "--overrides",
        type=str,
        default="{}",
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


if __name__ == "__main__":
    main()
