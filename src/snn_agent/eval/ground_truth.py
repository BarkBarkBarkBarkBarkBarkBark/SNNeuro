"""
snn_agent.eval.ground_truth — Synthetic ground-truth generation & scoring.

Wraps SpikeInterface to produce single-channel synthetic extracellular
recordings with known spike trains for benchmarking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import spikeinterface.full as si

__all__ = [
    "make_single_channel_ground_truth",
]


def make_single_channel_ground_truth(
    duration_s: float = 20.0,
    fs: float = 30_000.0,
    num_units: int = 2,
    firing_rates: tuple[float, ...] = (6.0, 12.0),
    refractory_period_ms: float = 4.0,
    noise_level: float = 8.0,
    seed: int = 42,
) -> tuple[Any, Any, dict[int, np.ndarray]]:
    """
    Build a single-channel synthetic extracellular recording.

    Returns
    -------
    recording : SpikeInterface Recording
    sorting_true : SpikeInterface Sorting
    gt_unit_trains : dict  unit_id → spike sample indices
    """
    sorting = si.generate_sorting(
        num_units=num_units,
        sampling_frequency=fs,
        durations=[duration_s],
        firing_rates=list(firing_rates),
        refractory_period_ms=refractory_period_ms,
        seed=seed,
    )

    recording, sorting_true = si.generate_ground_truth_recording(
        durations=[duration_s],
        sampling_frequency=fs,
        num_channels=1,
        sorting=sorting,
        ms_before=3.0,
        ms_after=4.0,
        noise_kwargs={"noise_levels": noise_level, "strategy": "on_the_fly"},
        dtype="float32",
        seed=seed,
    )

    gt_unit_trains = {
        uid: sorting_true.get_unit_spike_train(unit_id=uid, segment_index=0)
        for uid in sorting_true.unit_ids
    }

    return recording, sorting_true, gt_unit_trains


def main() -> None:
    """CLI entry point (``snn-ground-truth``)."""
    print("=" * 60)
    print("SNN Agent — Ground-Truth Benchmark")
    print("=" * 60)

    recording, sorting_true, gt_unit_trains = make_single_channel_ground_truth(
        duration_s=20.0, fs=30_000.0, num_units=2,
        firing_rates=(6.0, 10.0), noise_level=8.0, seed=7,
    )

    print(f"\nRecording : {recording}")
    print(f"Sorting   : {sorting_true}")
    for uid, train in gt_unit_trains.items():
        rate = len(train) / 20.0
        print(f"  Unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")


if __name__ == "__main__":
    main()
