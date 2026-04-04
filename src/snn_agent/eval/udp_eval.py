"""
snn_agent.eval.udp_eval — Real-time UDP output evaluator.

Listens on the SNN Agent's control UDP port (default 9002) and matches
received DEC hex bitmasks against a synthetic ground-truth recording.

Usage
-----
  snn-udp-eval                          # default: port 9002, seed 42, 2 units
  snn-udp-eval --port 9002 --seed 42 --units 2 --noise 8.0 --fs 30000 \\
               --duration 72 --assign-after 15 --log eval_log.json

Workflow
--------
1. Generates the same synthetic GT as `snn-serve --mode synthetic` (same seed).
2. Binds UDP socket, receives 2-byte big-endian hex bitmasks from the agent.
3. First ``assign_after_s`` seconds = warm-up: DEC neurons learn, no scoring.
4. After warm-up: Hungarian-style assignment of DEC bit indices → GT unit IDs
   using co-occurrence counts collected during warm-up.
5. Scores all subsequent detections: TP/FP/FN with 2 ms tolerance.
6. Prints a running summary every second and saves a JSON log on exit.

DEC bitmask format
------------------
  2-byte big-endian uint16.  Bit 0 = neuron 0 (any-fire / presence detector).
  Bits 1-15 = learned unit neurons.  Multiple bits may be set simultaneously.
  A detection is counted when ANY bit 1-15 fires (or bit 0 if --use-any-fire).
"""
from __future__ import annotations

import argparse
import json
import signal
import socket
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  GT generation (mirrors app.py / evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def _make_gt(
    seed: int,
    num_units: int,
    noise_level: float,
    fs: int,
    duration_s: float,
) -> tuple[dict[int, np.ndarray], float]:
    """Generate synthetic GT spike trains.  Returns {unit_id: sample_indices}."""
    try:
        import spikeinterface.core.generate as si_gen
    except ImportError:
        print("ERROR: spikeinterface not installed.  Install with: uv pip install spikeinterface")
        sys.exit(1)

    print(f"   🧪 Synthetic GT: seed={seed}  units={num_units}  noise={noise_level}"
          f"  fs={fs}  duration={duration_s}s")

    recording, sorting = si_gen.generate_ground_truth_recording(
        durations=[duration_s],
        sampling_frequency=float(fs),
        num_channels=1,
        num_units=num_units,
        seed=seed,
    )
    gt_trains: dict[int, np.ndarray] = {}
    for uid in sorting.unit_ids:
        train = sorting.get_unit_spike_train(unit_id=uid)
        gt_trains[int(uid)] = train
        rate = len(train) / duration_s
        print(f"      Unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")

    total_gt = sum(len(v) for v in gt_trains.values())
    print(f"   ✅ GT ready: {total_gt} total spikes across {num_units} units")
    return gt_trains, float(sorting.get_sampling_frequency())


# ─────────────────────────────────────────────────────────────────────────────
#  Hungarian-style bit → unit assignment using co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

class BitUnitAssigner:
    """
    Greedy maximum co-occurrence assignment of DEC bit indices to GT unit IDs.

    During warm-up we record, for each (bit_idx, unit_id) pair, how many
    times a DEC detection on that bit coincides with a GT spike within ±2 ms.
    After warm-up we assign bits to units greedily (highest co-occurrence first),
    one unit per bit (many-to-one allowed for bits).
    """

    def __init__(self, n_bits: int = 15, n_units: int = 2):
        # counts[bit_idx][unit_id] = co-occurrence count during warm-up
        self.counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.n_bits = n_bits   # bits 1..n_bits
        self.n_units = n_units
        self.assignment: dict[int, int] = {}   # bit_idx → unit_id

    def record(
        self,
        fired_bits: list[int],
        gt_trains: dict[int, np.ndarray],
        frame: int,
        delta_samp: int,
    ) -> None:
        """Called during warm-up whenever a DEC detection arrives."""
        for bit in fired_bits:
            for uid, train in gt_trains.items():
                # Check if any GT spike for this unit is within delta_samp
                lo = np.searchsorted(train, frame - delta_samp, side='left')
                hi = np.searchsorted(train, frame + delta_samp, side='right')
                if hi > lo:
                    self.counts[bit][uid] += 1

    def assign(self) -> dict[int, int]:
        """Greedy assignment: returns {bit_idx: unit_id}."""
        # Build score matrix
        all_bits = list(range(1, self.n_bits + 1))

        # Collect all (score, bit, unit) triples, sorted descending
        candidates = []
        for bit in all_bits:
            for uid, score in self.counts[bit].items():
                if score > 0:
                    candidates.append((score, bit, uid))
        candidates.sort(reverse=True)

        assigned_units: set[int] = set()
        result: dict[int, int] = {}
        for score, bit, uid in candidates:
            if uid not in assigned_units and bit not in result:
                result[bit] = uid
                assigned_units.add(uid)

        self.assignment = result
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Spike evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SpikeEvaluator:
    """Sliding-window TP/FP/FN evaluator (same logic as _process_stream)."""

    def __init__(
        self,
        gt_trains: dict[int, np.ndarray],
        native_fs: float,
        delta_ms: float = 2.0,
        debounce_ms: float = 1.0,
    ):
        self.gt_trains = gt_trains
        self.native_fs = native_fs
        self.delta_samp = int(delta_ms * 1e-3 * native_fs)
        self.debounce = int(debounce_ms * 1e-3 * native_fs)

        # Merge all units into one sorted array for detection matching
        self.all_gt = np.sort(np.concatenate(list(gt_trains.values())))

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.gt_ptr = 0
        self.gt_matched: set[int] = set()
        self.latencies: list[int] = []
        self.last_det = -99999

    def step(self, frame: int, detected: bool) -> None:
        """Process one sample frame."""
        # Expire old GT as FN
        while (self.gt_ptr < len(self.all_gt) and
               self.all_gt[self.gt_ptr] < frame - self.delta_samp):
            if self.gt_ptr not in self.gt_matched:
                self.fn += 1
            self.gt_ptr += 1

        if not detected:
            return
        if frame - self.last_det <= self.debounce:
            return  # refractory

        self.last_det = frame
        best_dist = float('inf')
        best_gi = -1
        for gi in range(self.gt_ptr, len(self.all_gt)):
            gt_t = int(self.all_gt[gi])
            if gt_t > frame + self.delta_samp:
                break
            if gi in self.gt_matched:
                continue
            dist = abs(frame - gt_t)
            if dist <= self.delta_samp and dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi >= 0:
            self.tp += 1
            self.gt_matched.add(best_gi)
            self.latencies.append(best_dist)
        else:
            self.fp += 1

    def metrics(self) -> dict:
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        fh = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
        lat = (np.mean(self.latencies) / self.native_fs * 1000) if self.latencies else 0.0
        return {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f_half': round(fh, 4),
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'latency_ms': round(lat, 3),
            'n_gt': len(self.all_gt),
            'gt_ptr': self.gt_ptr,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # Generate GT
    gt_trains, native_fs = _make_gt(
        seed=args.seed,
        num_units=args.units,
        noise_level=args.noise,
        fs=args.fs,
        duration_s=args.duration,
    )

    assign_after_samp = int(args.assign_after * native_fs)
    delta_samp = int(2.0 * 1e-3 * native_fs)

    # Bind UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    sock.settimeout(0.1)
    print(f"\n   📡 Listening on UDP :{args.port}  (start `snn-serve --mode synthetic` now)")
    print(f"   ⏳ Warm-up: first {args.assign_after:.0f}s ({assign_after_samp} samples) for bit→unit assignment")
    print("   Ctrl-C to stop and save results\n")

    assigner = BitUnitAssigner(n_bits=15, n_units=args.units)
    evaluator: SpikeEvaluator | None = None

    # Timing: we don't know what sample-rate the agent is streaming at, so we
    # track wall-clock time and convert to sample frames ourselves.
    t_start = time.perf_counter()
    frame_counter = 0        # estimated current sample frame (native_fs units)
    warmup_done = False
    assigned: dict[int, int] = {}

    n_received = 0
    n_warmup_detections = 0
    log_entries: list[dict] = []
    last_print = time.perf_counter()

    # Handle Ctrl-C gracefully
    _stop = [False]
    def _handler(sig, frame_):  # noqa: ANN001
        _stop[0] = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    while not _stop[0]:
        # Advance estimated frame counter based on wall clock
        now = time.perf_counter()
        elapsed = now - t_start
        frame_counter = int(elapsed * native_fs)

        # Receive UDP packets
        try:
            data, _ = sock.recvfrom(64)
        except socket.timeout:
            # Still advance evaluator FN counting even when no packets arrive
            if evaluator is not None:
                evaluator.step(frame_counter, detected=False)
            continue
        except OSError:
            break

        n_received += 1

        # Parse packet: 2-byte big-endian uint16 hex bitmask
        if len(data) == 2:
            (hex_word,) = struct.unpack('!H', data)
        elif len(data) == 8:
            # Legacy float32 (ctrl, conf) — treat as any detection
            hex_word = 1
        else:
            continue

        fired_bits = [i for i in range(1, 16) if (hex_word >> i) & 1]
        any_fire_bit0 = bool(hex_word & 1)

        # Decide if this counts as a detection
        if args.use_any_fire:
            detected = any_fire_bit0 or len(fired_bits) > 0
        else:
            detected = len(fired_bits) > 0

        if not warmup_done:
            n_warmup_detections += 1
            # Record co-occurrence during warm-up
            if fired_bits:
                assigner.record(fired_bits, gt_trains, frame_counter, delta_samp)

            if frame_counter >= assign_after_samp:
                warmup_done = True
                assigned = assigner.assign()
                print(f"\n   🔗 Bit→Unit assignment after {args.assign_after:.0f}s warm-up:")
                if assigned:
                    for bit, uid in sorted(assigned.items()):
                        cnt = assigner.counts[bit][uid]
                        print(f"      bit {bit:2d} → unit {uid}  (co-occurrences: {cnt})")
                else:
                    print("      ⚠ No co-occurrences found — DEC may not be firing during warm-up")
                    if args.use_any_fire:
                        print("      Using any-fire (bit 0) as detection signal")
                evaluator = SpikeEvaluator(gt_trains, native_fs)
                print(f"   🏁 Scoring phase started  ({n_warmup_detections} detections during warm-up)\n")
        else:
            if evaluator is not None:
                evaluator.step(frame_counter, detected)

        # Print running summary once per second
        if now - last_print >= 1.0:
            last_print = now
            pct_gt = frame_counter / (args.duration * native_fs) * 100
            if evaluator is not None:
                m = evaluator.metrics()
                print(
                    f"   t={elapsed:6.1f}s ({pct_gt:.0f}%) | "
                    f"P:{m['precision']:.3f}  R:{m['recall']:.3f}  F½:{m['f_half']:.3f} | "
                    f"TP:{m['tp']}  FP:{m['fp']}  FN:{m['fn']}  lat:{m['latency_ms']:.1f}ms | "
                    f"pkts:{n_received}"
                )
                log_entries.append({'t': round(elapsed, 2), **m, 'pkts': n_received})
            else:
                print(
                    f"   t={elapsed:6.1f}s ({pct_gt:.0f}%) | "
                    f"[warm-up]  detections:{n_warmup_detections}  pkts:{n_received}"
                )

        # Stop if recording is over
        if frame_counter >= int(args.duration * native_fs) + int(2.0 * native_fs):
            print("\n   ⏹  Recording duration elapsed, stopping.")
            break

    sock.close()

    # Final summary
    print("\n" + "─" * 70)
    if evaluator is not None:
        m = evaluator.metrics()
        print("   📊 Final results (scoring phase, ±2 ms tolerance)")
        print(f"      Precision : {m['precision']:.4f}  ({m['precision']*100:.1f}%)")
        print(f"      Recall    : {m['recall']:.4f}  ({m['recall']*100:.1f}%)")
        print(f"      F½        : {m['f_half']:.4f}  ({m['f_half']*100:.1f}%)")
        print(f"      TP:{m['tp']}  FP:{m['fp']}  FN:{m['fn']}  Latency:{m['latency_ms']:.2f}ms")
        print(f"      GT coverage: {m['gt_ptr']}/{m['n_gt']} spikes scanned")
        print(f"      UDP packets received (total): {n_received}")
    else:
        print("   ⚠ Scoring phase never started (not enough data received)")

    # Save JSON log
    if args.log:
        result = {
            'config': {
                'seed': args.seed,
                'units': args.units,
                'noise': args.noise,
                'fs': args.fs,
                'duration': args.duration,
                'assign_after': args.assign_after,
                'port': args.port,
                'use_any_fire': args.use_any_fire,
            },
            'assignment': {str(k): v for k, v in assigned.items()},
            'warmup_detections': n_warmup_detections,
            'final': evaluator.metrics() if evaluator else {},
            'timeline': log_entries,
        }
        Path(args.log).write_text(json.dumps(result, indent=2))
        print(f"\n   💾 Saved to {args.log}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real-time UDP evaluation of SNN Agent DEC output vs. synthetic GT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--port', type=int, default=9002,
                    help="UDP port to listen on (must match snn-serve control port)")
    ap.add_argument('--seed', type=int, default=42,
                    help="RNG seed — must match the snn-serve synthetic recording seed")
    ap.add_argument('--units', type=int, default=2,
                    help="Number of synthetic units — must match snn-serve --units")
    ap.add_argument('--noise', type=float, default=8.0,
                    help="Noise level — must match snn-serve --noise")
    ap.add_argument('--fs', type=int, default=30_000,
                    help="Native sample rate (Hz) — matches SyntheticConfig.fs")
    ap.add_argument('--duration', type=float, default=72.0,
                    help="Expected recording duration in seconds")
    ap.add_argument('--assign-after', type=float, default=15.0, metavar='SECONDS',
                    help="Warm-up period before scoring starts (seconds)")
    ap.add_argument('--use-any-fire', action='store_true',
                    help="Also count bit-0 (any-fire) as a detection (lower precision)")
    ap.add_argument('--log', type=str, default='',
                    help="Path to save JSON results (empty = no save)")
    args = ap.parse_args()

    if not args.log:
        args.log = f"udp_eval_{args.seed}_{int(time.time())}.json"

    run(args)


if __name__ == '__main__':
    main()
