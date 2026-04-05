"""
snn_agent.eval.evaluate — Offline SNN pipeline evaluation against ground truth.

Generates synthetic data, runs the full pipeline sample-by-sample,
and scores L1 clusters against ground-truth spike trains via SpikeInterface.
"""

from __future__ import annotations

# ruff: noqa: E402

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

__all__ = ["evaluate_pipeline", "multi_evaluate"]

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
    delta_time: float = 2.0,
    score_after_s: float | None = None,
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
    delta_time : float
        Spike matching tolerance in ms for SpikeInterface comparison.
        Default 2.0 ms (5× tighter than prior 10 ms; SpikeInterface default 0.4 ms).
    score_after_s : float, optional
        If set, only score spikes occurring after this time (seconds).
        Enables train/test temporal split: STDP learns on all data
        but accuracy is measured only on the test window.

    Returns
    -------
    dict with accuracy, recall, precision, f_half, n_active, total_spikes, runtime_s, perf_df.
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
    dec_spike_log: dict[int, list[int]] = {}
    step_count = 0
    t0 = time.perf_counter()
    _early_exit = False
    any_l1_fired_prev = False

    # ── Pre-process entire trace in one sosfilt call ────────────────────────
    # Replaces n_total individual step() calls (each invoking sosfilt on a
    # single element) with one vectorised call — major speedup for eval runs.
    decimated_all = preproc.step_chunk(traces.astype(np.float64))
    n_dec = len(decimated_all)
    # Compute the original raw-sample frame index for each decimated output.
    # Preprocessor._dec_count starts at 0, so the first kept raw index is
    # (dec_factor - 1) and subsequent ones are spaced dec_factor apart.
    if preproc.do_decimate:
        _df = preproc.dec_factor
        _raw_frames = np.arange(_df - 1, n_total, _df, dtype=np.int64)[:n_dec]
    else:
        _raw_frames = np.arange(n_dec, dtype=np.int64)

    try:
        for dec_idx in range(n_dec):
            pp_sample = float(decimated_all[dec_idx])
            frame_idx = int(_raw_frames[dec_idx])
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
                max_current = pipeline_obj.template.last_current_magnitude
                inh_factor = pipeline_obj.inhibitor.gate(max_current, any_l1_fired_prev)
                suppression *= inh_factor

            l1_spikes = pipeline_obj.template.step(afferents, dn_spike, suppression)
            any_l1_fired_prev = bool(np.any(l1_spikes))

            # Optional DEC decoder layer
            decoder_input = l1_spikes
            if pipeline_obj.dec_layer is not None:
                dec_spikes = pipeline_obj.dec_layer.step(l1_spikes, dn_spike)
                for idx in np.flatnonzero(dec_spikes):
                    nid = int(idx)
                    dec_spike_log.setdefault(nid, []).append(frame_idx)
                decoder_input = dec_spikes

            pipeline_obj.decoder.step(decoder_input, dn_spike)

            for idx in np.flatnonzero(l1_spikes):
                nid = int(idx)
                l1_spike_log.setdefault(nid, []).append(frame_idx)

            if verbose and dec_idx % 25_000 == 0 and dec_idx > 0:
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
        "f_half": 0.0,
        "n_active": n_active,
        "total_spikes": total_spikes,
        "runtime_s": elapsed,
        "perf_df": None,
    }

    if not l1_spike_log:
        return result

    # ── Train/test split: filter spikes to scoring window ─────────────
    if score_after_s is not None and score_after_s > 0:
        t_start_sample = int(score_after_s * fs)
        l1_sorting_dict = {
            nid: np.array([t for t in times if t >= t_start_sample], dtype=np.int64)
            for nid, times in l1_spike_log.items()
        }
        l1_sorting_dict = {k: v for k, v in l1_sorting_dict.items() if len(v) > 0}
        gt_test = {
            uid: train[train >= t_start_sample]
            for uid, train in gt_unit_trains.items()
        }
        sorting_cmp = NumpySorting.from_unit_dict(gt_test, sampling_frequency=fs)
        if verbose:
            n_test_gt = sum(len(v) for v in gt_test.values())
            n_test_det = sum(len(v) for v in l1_sorting_dict.values())
            print(f"  🧪 Scoring window: {score_after_s}s–end "
                  f"({n_test_gt} GT spikes, {n_test_det} detected)")
    else:
        l1_sorting_dict = {
            nid: np.array(times, dtype=np.int64)
            for nid, times in l1_spike_log.items()
            if len(times) > 0
        }
        sorting_cmp = sorting_true

    if not l1_sorting_dict:
        return result

    l1_sorting = NumpySorting.from_unit_dict(
        units_dict_list=l1_sorting_dict,
        sampling_frequency=fs,
    )

    cmp = sc.compare_sorter_to_ground_truth(
        gt_sorting=sorting_cmp,
        tested_sorting=l1_sorting,
        exhaustive_gt=True,
        delta_time=delta_time,
    )
    perf = cmp.get_performance()

    P = float(perf["precision"].mean())
    R = float(perf["recall"].mean())

    result["accuracy"] = float(perf["accuracy"].mean())
    result["recall"] = R
    result["precision"] = P
    result["f_half"] = (1.25 * P * R) / (0.25 * P + R) if (P + R) > 0 else 0.0
    result["perf_df"] = perf

    return result


# ── Default evaluation scenarios for multi_evaluate ──────────────────────────
DEFAULT_SCENARIOS = [
    {"seed": 42,  "noise_level": 8.0,  "num_units": 2, "firing_rates": [6.0, 10.0]},
    {"seed": 137, "noise_level": 10.0, "num_units": 2, "firing_rates": [6.0, 10.0]},
    {"seed": 256, "noise_level": 6.0,  "num_units": 2, "firing_rates": [8.0, 14.0]},
    {"seed": 789, "noise_level": 12.0, "num_units": 3, "firing_rates": [5.0, 8.0, 12.0]},
]


def multi_evaluate(
    cfg_overrides: dict | None = None,
    scenarios: list[dict] | None = None,
    score_after_s: float | None = 15.0,
    delta_time: float = 2.0,
    verbose: bool = False,
) -> dict:
    """
    Evaluate the SNN pipeline across multiple synthetic scenarios.

    Runs ``evaluate_pipeline()`` for each scenario (varying seed, noise,
    unit count, firing rates) and returns averaged metrics.  This prevents
    overfitting to a single noise realisation and ensures the reported
    accuracy generalises.

    Parameters
    ----------
    cfg_overrides : dict, optional
        Flat-key config overrides (forwarded to each evaluation).
    scenarios : list[dict], optional
        List of recording parameter dicts.  Each must contain at least
        ``seed``, ``noise_level``, ``num_units``, ``firing_rates``.
        Defaults to :data:`DEFAULT_SCENARIOS` (4 diverse signals).
    score_after_s : float, optional
        Train/test split: only score spikes after this time (seconds).
        Default 15.0 s for a 20 s recording (train first 15 s, test last 5 s).
    delta_time : float
        Spike matching tolerance in ms (default 2.0; SpikeInterface default 0.4).
    verbose : bool
        Print per-scenario progress.

    Returns
    -------
    dict with averaged accuracy, recall, precision, f_half, n_active,
    total_spikes, runtime_s, and per-scenario ``scenario_results``.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    all_results: list[dict] = []

    for i, scenario in enumerate(scenarios):
        rec_params = {
            "duration_s": scenario.get("duration_s", 20.0),
            "fs": scenario.get("fs", 30_000.0),
            "num_units": scenario["num_units"],
            "firing_rates": scenario["firing_rates"],
            "noise_level": scenario["noise_level"],
            "seed": scenario["seed"],
        }
        if verbose:
            print(f"  ── Scenario {i + 1}/{len(scenarios)}: "
                  f"seed={scenario['seed']}, noise={scenario['noise_level']}, "
                  f"units={scenario['num_units']} ──")

        result = evaluate_pipeline(
            cfg_overrides=cfg_overrides,
            rec_params=rec_params,
            verbose=verbose,
            delta_time=delta_time,
            score_after_s=score_after_s,
        )
        all_results.append(result)

        # Early termination: if first scenario scores 0, skip the rest
        if i == 0 and result["accuracy"] == 0.0:
            if verbose:
                print("  ⚠ First scenario scored 0.0 — skipping remaining")
            for _ in range(len(scenarios) - 1):
                all_results.append({
                    "accuracy": 0.0, "recall": 0.0, "precision": 0.0,
                    "f_half": 0.0, "n_active": 0, "total_spikes": 0,
                    "runtime_s": 0.0, "perf_df": None,
                })
            break

    # ── Average metrics across scenarios ──────────────────────────────
    avg: dict = {}
    for key in ("accuracy", "precision", "recall", "n_active", "total_spikes", "runtime_s"):
        vals = [r.get(key, 0) for r in all_results]
        avg[key] = sum(vals) / len(vals)

    P, R = avg["precision"], avg["recall"]
    avg["f_half"] = (1.25 * P * R) / (0.25 * P + R) if (P + R) > 0 else 0.0
    avg["scenario_results"] = all_results
    avg["perf_df"] = None  # not meaningful when averaged

    if verbose:
        print(f"\n  ── Multi-scenario average ({len(scenarios)} signals) ──")
        print(f"     Accuracy  : {avg['accuracy']:.4f}")
        print(f"     Precision : {avg['precision']:.4f}")
        print(f"     Recall    : {avg['recall']:.4f}")
        print(f"     F₀.₅     : {avg['f_half']:.4f}")

    return avg


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
    print(f"  F₀.₅     : {metrics['f_half']:.4f}")
    print(f"  Active L1 : {metrics['n_active']}")
    print(f"  Total spk : {metrics['total_spikes']}")
    print(f"  Runtime   : {metrics['runtime_s']:.1f}s")

    if metrics["perf_df"] is not None:
        print("\nPer-unit performance:")
        print(metrics["perf_df"].to_string())


if __name__ == "__main__":
    main()
