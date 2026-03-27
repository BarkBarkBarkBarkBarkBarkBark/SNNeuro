# AGENT-HINT: End-to-end latency benchmark for spike detection pipeline.
# PURPOSE: Measures the delay (in ms and samples) between each ground-truth
#          spike and the first downstream detection, at three pipeline stages:
#          L1 (template), DEC (decoder layer), and control signal (output > 0).
#          Also measures wall-clock processing time per sample.
# CONFIG: Uses Config (via best_config.json or defaults) + synthetic recording.
# SEE ALSO: evaluate.py (accuracy benchmark), ground_truth.py (synthetic data),
#           decoder.py (output strategies), pipeline.py (factory)
"""
snn_agent.eval.latency — End-to-end spike detection latency benchmark.

Generates a synthetic recording with known spike times, runs the full
pipeline sample-by-sample, and measures the delay from each ground-truth
spike to the first detection event at three pipeline stages:

1. **L1** — when any template neuron fires
2. **DEC** — when any DEC neuron fires (if enabled)
3. **Control** — when the output control signal first exceeds zero

Reports both **algorithmic latency** (how many samples / ms late the
detection arrives relative to the ground-truth spike) and **wall-clock
processing time** (how fast the Python code runs per sample and per
pipeline step, critical for closed-loop human experiments).
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", message="Inhibition is an unstable feature")

from snn_agent.config import Config, DEFAULT_CONFIG
from snn_agent.core.pipeline import build_pipeline, complete_pipeline
from snn_agent.eval.ground_truth import make_single_channel_ground_truth

__all__ = ["measure_latency"]

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "data"


# ── Per-spike result ──────────────────────────────────────────────────────────
@dataclass
class SpikeLatency:
    """Latency measurement for a single ground-truth spike."""
    gt_unit: int
    gt_sample: int            # sample index at native fs
    gt_time_ms: float         # ground-truth time in ms

    l1_sample: int | None = None
    l1_latency_samples: int | None = None
    l1_latency_ms: float | None = None

    dec_sample: int | None = None
    dec_latency_samples: int | None = None
    dec_latency_ms: float | None = None

    ctrl_sample: int | None = None
    ctrl_latency_samples: int | None = None
    ctrl_latency_ms: float | None = None


# ── Main benchmark ────────────────────────────────────────────────────────────
def measure_latency(
    cfg_overrides: dict | None = None,
    rec_params: dict | None = None,
    max_window_ms: float = 5.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full pipeline on a synthetic recording and measure detection
    latency at every stage.

    Parameters
    ----------
    cfg_overrides : dict, optional
        Flat-key config overrides (e.g. from best_config.json).
    rec_params : dict, optional
        Synthetic recording parameters.
    max_window_ms : float
        Maximum window (ms) after a GT spike to search for a detection.
        Detections beyond this window are treated as misses.
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict with keys:
        spikes : list[SpikeLatency]
        summary : dict  (per-layer stats)
        wall_clock : dict  (per-sample and per-step timing)
        config_used : dict
    """
    # ── Build config ──────────────────────────────────────────────────
    if cfg_overrides:
        cfg = Config.from_flat({**DEFAULT_CONFIG.to_dict_flat(), **cfg_overrides})
    else:
        cfg = DEFAULT_CONFIG

    if rec_params is None:
        syn = cfg.synthetic
        rec_params = {
            "duration_s": syn.duration_s,
            "fs": syn.fs,
            "num_units": syn.num_units,
            "noise_level": syn.noise_level,
            "seed": syn.seed,
        }

    # ── Generate ground truth ─────────────────────────────────────────
    recording, sorting_true, gt_unit_trains = make_single_channel_ground_truth(
        duration_s=rec_params.get("duration_s", 20.0),
        fs=rec_params.get("fs", 30_000.0),
        num_units=rec_params.get("num_units", 2),
        firing_rates=tuple(rec_params.get("firing_rates", (6.0, 10.0))),
        noise_level=rec_params.get("noise_level", 8.0),
        seed=rec_params.get("seed", 42),
    )

    fs = recording.get_sampling_frequency()
    cfg = cfg.with_overrides(sampling_rate_hz=int(fs))
    max_window_samples = int(max_window_ms * 1e-3 * fs)

    # Flatten all GT spikes into a sorted list with unit labels
    gt_events: list[tuple[int, int]] = []  # (sample_idx, unit_id)
    for uid, train in gt_unit_trains.items():
        for s in train:
            gt_events.append((int(s), int(uid)))
    gt_events.sort(key=lambda x: x[0])

    if verbose:
        total_gt = len(gt_events)
        for uid, train in gt_unit_trains.items():
            rate = len(train) / rec_params.get("duration_s", 20.0)
            print(f"  GT unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")
        print(f"  Total GT spikes: {total_gt}")
        print(f"  Max matching window: {max_window_ms} ms ({max_window_samples} samples)")

    # ── Build pipeline ────────────────────────────────────────────────
    preproc, encoder, effective_cfg = build_pipeline(cfg)
    pipeline_obj = None

    traces = recording.get_traces(segment_index=0)[:, 0]
    n_total = len(traces)

    # ── Detection logs: frame_idx → event type ────────────────────────
    l1_event_times: list[int] = []    # frame_idx when any L1 neuron fires
    dec_event_times: list[int] = []   # frame_idx when any DEC neuron fires
    ctrl_event_times: list[int] = []  # frame_idx when control > 0

    # ── Wall-clock timing ─────────────────────────────────────────────
    step_count = 0
    wall_times: list[float] = []  # per-step wall-clock durations (seconds)
    any_l1_fired_prev = False

    if verbose:
        print(f"  Running pipeline on {n_total} samples ({n_total / fs:.1f}s)…")

    t_total_start = time.perf_counter()

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
                pipeline_obj = complete_pipeline(cfg, effective_cfg, preproc, encoder)
                if verbose:
                    print(
                        f"  ✅ Calibrated: {encoder.n_afferents} afferents "
                        f"({preproc.effective_fs} Hz)"
                    )

            # ── Timed pipeline step ───────────────────────────────
            t_step_start = time.perf_counter()

            dn_spike = pipeline_obj.attention.step(afferents)

            suppression = 1.0
            if pipeline_obj.noise_gate is not None:
                suppression = pipeline_obj.noise_gate.step(pp_sample)

            if pipeline_obj.inhibitor is not None:
                max_current = pipeline_obj.template.last_current_magnitude
                inh_factor = pipeline_obj.inhibitor.gate(max_current, any_l1_fired_prev)
                suppression *= inh_factor

            l1_spikes = pipeline_obj.template.step(afferents, dn_spike, suppression)
            any_l1_fired = bool(np.any(l1_spikes))
            any_l1_fired_prev = any_l1_fired

            decoder_input = l1_spikes
            any_dec_fired = False
            if pipeline_obj.dec_layer is not None:
                dec_spikes = pipeline_obj.dec_layer.step(l1_spikes, dn_spike)
                any_dec_fired = bool(np.any(dec_spikes))
                decoder_input = dec_spikes

            ctrl_val, _ = pipeline_obj.decoder.step(decoder_input, dn_spike)

            t_step_end = time.perf_counter()
            wall_times.append(t_step_end - t_step_start)

            # ── Log detections ────────────────────────────────────
            if any_l1_fired:
                l1_event_times.append(frame_idx)
            if any_dec_fired:
                dec_event_times.append(frame_idx)
            if ctrl_val > 0:
                ctrl_event_times.append(frame_idx)

        if verbose and frame_idx % 100_000 == 0 and frame_idx > 0:
            pct = 100 * frame_idx / n_total
            print(f"  ⏳ {pct:.0f}%")

    t_total_end = time.perf_counter()
    total_wall = t_total_end - t_total_start

    if verbose:
        print(f"  🏁 Done in {total_wall:.2f}s ({step_count} post-decimation steps)")

    # ── Convert detection logs to sorted arrays ───────────────────────
    l1_arr = np.array(l1_event_times, dtype=np.int64)
    dec_arr = np.array(dec_event_times, dtype=np.int64)
    ctrl_arr = np.array(ctrl_event_times, dtype=np.int64)

    # ── Match each GT spike to nearest subsequent detection ───────────
    spike_latencies: list[SpikeLatency] = []

    for gt_sample, gt_unit in gt_events:
        sl = SpikeLatency(
            gt_unit=gt_unit,
            gt_sample=gt_sample,
            gt_time_ms=gt_sample / fs * 1000.0,
        )

        # L1 match
        idx = np.searchsorted(l1_arr, gt_sample)
        if idx < len(l1_arr):
            delta = int(l1_arr[idx]) - gt_sample
            if 0 <= delta <= max_window_samples:
                sl.l1_sample = int(l1_arr[idx])
                sl.l1_latency_samples = delta
                sl.l1_latency_ms = delta / fs * 1000.0

        # DEC match
        idx = np.searchsorted(dec_arr, gt_sample)
        if idx < len(dec_arr):
            delta = int(dec_arr[idx]) - gt_sample
            if 0 <= delta <= max_window_samples:
                sl.dec_sample = int(dec_arr[idx])
                sl.dec_latency_samples = delta
                sl.dec_latency_ms = delta / fs * 1000.0

        # Control signal match
        idx = np.searchsorted(ctrl_arr, gt_sample)
        if idx < len(ctrl_arr):
            delta = int(ctrl_arr[idx]) - gt_sample
            if 0 <= delta <= max_window_samples:
                sl.ctrl_sample = int(ctrl_arr[idx])
                sl.ctrl_latency_samples = delta
                sl.ctrl_latency_ms = delta / fs * 1000.0

        spike_latencies.append(sl)

    # ── Summary statistics ────────────────────────────────────────────
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "mean": None, "median": None, "p95": None,
                    "min": None, "max": None}
        arr = np.array(values)
        return {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    total_gt = len(spike_latencies)

    l1_ms = [s.l1_latency_ms for s in spike_latencies if s.l1_latency_ms is not None]
    dec_ms = [s.dec_latency_ms for s in spike_latencies if s.dec_latency_ms is not None]
    ctrl_ms = [s.ctrl_latency_ms for s in spike_latencies if s.ctrl_latency_ms is not None]

    l1_samp = [s.l1_latency_samples for s in spike_latencies if s.l1_latency_samples is not None]
    dec_samp = [s.dec_latency_samples for s in spike_latencies if s.dec_latency_samples is not None]
    ctrl_samp = [s.ctrl_latency_samples for s in spike_latencies if s.ctrl_latency_samples is not None]

    summary = {
        "total_gt_spikes": total_gt,
        "l1": {
            "matched": len(l1_ms),
            "missed": total_gt - len(l1_ms),
            "latency_ms": _stats(l1_ms),
            "latency_samples": _stats(l1_samp),
        },
        "dec": {
            "matched": len(dec_ms),
            "missed": total_gt - len(dec_ms),
            "latency_ms": _stats(dec_ms),
            "latency_samples": _stats(dec_samp),
        },
        "ctrl": {
            "matched": len(ctrl_ms),
            "missed": total_gt - len(ctrl_ms),
            "latency_ms": _stats(ctrl_ms),
            "latency_samples": _stats(ctrl_samp),
        },
    }

    # ── Wall-clock timing stats ───────────────────────────────────────
    wt_arr = np.array(wall_times) if wall_times else np.zeros(1)
    wall_clock = {
        "total_s": total_wall,
        "samples_processed": step_count,
        "per_step_us": {
            "mean": float(np.mean(wt_arr) * 1e6),
            "median": float(np.median(wt_arr) * 1e6),
            "p95": float(np.percentile(wt_arr, 95) * 1e6),
            "p99": float(np.percentile(wt_arr, 99) * 1e6),
            "max": float(np.max(wt_arr) * 1e6),
        },
        "realtime_ratio": (step_count / preproc.effective_fs) / total_wall if total_wall > 0 else 0,
        "max_realtime_fs": step_count / total_wall if total_wall > 0 else 0,
    }

    # ── Per-unit breakdown ────────────────────────────────────────────
    unit_ids = sorted(set(s.gt_unit for s in spike_latencies))
    per_unit: dict[int, dict] = {}
    for uid in unit_ids:
        unit_spikes = [s for s in spike_latencies if s.gt_unit == uid]
        u_ctrl_ms = [s.ctrl_latency_ms for s in unit_spikes if s.ctrl_latency_ms is not None]
        u_l1_ms = [s.l1_latency_ms for s in unit_spikes if s.l1_latency_ms is not None]
        per_unit[uid] = {
            "total": len(unit_spikes),
            "l1_matched": len(u_l1_ms),
            "ctrl_matched": len(u_ctrl_ms),
            "l1_latency_ms": _stats(u_l1_ms),
            "ctrl_latency_ms": _stats(u_ctrl_ms),
        }

    # ── Print results ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 64)
        print("  LATENCY BENCHMARK RESULTS")
        print("=" * 64)

        for layer, label in [("l1", "Template (L1)"), ("dec", "DEC Layer"),
                             ("ctrl", "Control Signal")]:
            info = summary[layer]
            ms = info["latency_ms"]
            samp = info["latency_samples"]
            print(f"\n  ── {label} ──")
            print(f"     Matched: {info['matched']}/{total_gt} "
                  f"({100*info['matched']/max(total_gt,1):.1f}%)")
            if ms["n"] > 0:
                print(f"     Latency (ms):      "
                      f"median={ms['median']:.3f}  mean={ms['mean']:.3f}  "
                      f"P95={ms['p95']:.3f}  max={ms['max']:.3f}")
                print(f"     Latency (samples): "
                      f"median={samp['median']:.0f}  mean={samp['mean']:.1f}  "
                      f"P95={samp['p95']:.0f}  max={samp['max']:.0f}")

        print(f"\n  ── Wall-Clock Performance ──")
        wc = wall_clock
        ps = wc["per_step_us"]
        print(f"     Total:     {wc['total_s']:.2f}s for {wc['samples_processed']} steps")
        print(f"     Per step:  median={ps['median']:.1f}µs  "
              f"mean={ps['mean']:.1f}µs  P95={ps['p95']:.1f}µs  "
              f"P99={ps['p99']:.1f}µs  max={ps['max']:.1f}µs")
        print(f"     Realtime:  {wc['realtime_ratio']:.1f}× "
              f"(pipeline runs {wc['realtime_ratio']:.1f}× faster than real time)")
        print(f"     Max fs:    {wc['max_realtime_fs']:.0f} Hz "
              f"(could sustain this sample rate in real time)")

        if len(per_unit) > 1:
            print(f"\n  ── Per-Unit Breakdown ──")
            for uid, info in per_unit.items():
                cms = info["ctrl_latency_ms"]
                if cms["n"] > 0:
                    print(f"     Unit {uid}: {info['ctrl_matched']}/{info['total']} matched  "
                          f"median={cms['median']:.3f}ms  P95={cms['p95']:.3f}ms")
                else:
                    print(f"     Unit {uid}: {info['ctrl_matched']}/{info['total']} matched")

        print("=" * 64)

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = DATA_DIR / "latency_report.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gt_unit", "gt_sample", "gt_time_ms",
            "l1_latency_ms", "l1_latency_samples",
            "dec_latency_ms", "dec_latency_samples",
            "ctrl_latency_ms", "ctrl_latency_samples",
        ])
        for sl in spike_latencies:
            writer.writerow([
                sl.gt_unit, sl.gt_sample, f"{sl.gt_time_ms:.3f}",
                f"{sl.l1_latency_ms:.3f}" if sl.l1_latency_ms is not None else "",
                sl.l1_latency_samples if sl.l1_latency_samples is not None else "",
                f"{sl.dec_latency_ms:.3f}" if sl.dec_latency_ms is not None else "",
                sl.dec_latency_samples if sl.dec_latency_samples is not None else "",
                f"{sl.ctrl_latency_ms:.3f}" if sl.ctrl_latency_ms is not None else "",
                sl.ctrl_latency_samples if sl.ctrl_latency_samples is not None else "",
            ])
    if verbose:
        print(f"\n  📄 Per-spike CSV saved to {csv_path}")

    # ── Save JSON summary ─────────────────────────────────────────────
    json_path = DATA_DIR / "latency_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": summary,
            "wall_clock": wall_clock,
            "per_unit": {str(k): v for k, v in per_unit.items()},
            "config": cfg_overrides or {},
            "rec_params": rec_params,
            "max_window_ms": max_window_ms,
        }, f, indent=2)
    if verbose:
        print(f"  📄 Summary JSON saved to {json_path}")

    return {
        "spikes": spike_latencies,
        "summary": summary,
        "wall_clock": wall_clock,
        "per_unit": per_unit,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point (``snn-latency``)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SNN Agent — End-to-end spike detection latency benchmark"
    )
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Synthetic recording duration (seconds)")
    parser.add_argument("--num-units", type=int, default=2,
                        help="Number of neural units to simulate")
    parser.add_argument("--noise-level", type=float, default=8.0,
                        help="Noise level for synthetic recording")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-window-ms", type=float, default=5.0,
                        help="Max detection window after GT spike (ms)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to best_config.json (auto-detects data/best_config.json)")
    args = parser.parse_args()

    print("=" * 64)
    print("⚡ SNN Agent — Latency Benchmark")
    print("=" * 64)

    # Load optimized config if available
    cfg_overrides: dict | None = None
    config_path = Path(args.config) if args.config else DATA_DIR / "best_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
            params = data.get("parameters", {})
            cfg_overrides = {k: v for k, v in params.items()
                            if k not in ("f_half", "accuracy", "precision", "recall")}
            print(f"  ✓ Loaded config from {config_path.name}")
        except Exception as exc:
            print(f"  ⚠ Could not load {config_path}: {exc}")
    else:
        print("  ℹ No optimized config found — using defaults")

    rec_params = {
        "duration_s": args.duration,
        "fs": 30_000.0,
        "num_units": args.num_units,
        "noise_level": args.noise_level,
        "seed": args.seed,
    }

    measure_latency(
        cfg_overrides=cfg_overrides,
        rec_params=rec_params,
        max_window_ms=args.max_window_ms,
        verbose=True,
    )


if __name__ == "__main__":
    main()
