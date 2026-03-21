"""
lsl_player.py — Stream a Neuralynx .ncs recording over LSL at real-time rate.

Usage:
    python lsl_player.py /path/to/ncs_folder
    python lsl_player.py /path/to/single_file.ncs
    python lsl_player.py /path/to/ncs_folder --channel CSC1 --chunk-size 64

The PlayerLSL pushes samples at the original sampling rate so downstream
consumers (e.g. server.py in LSL mode) see the same timing as a live
acquisition system.

Requires: mne, mne-lsl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mne
from mne_lsl.player import PlayerLSL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream Neuralynx .ncs files over LSL at real-time rate.",
    )
    parser.add_argument(
        "path",
        help="Path to a .ncs file or a directory containing .ncs files.",
    )
    parser.add_argument(
        "--channel", "-c",
        default=None,
        help="Channel name to stream (default: auto-detect from filename, or "
             "first channel if a directory is given).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Samples per LSL push (default: 64).  Lower = less latency, "
             "higher = less overhead.",
    )
    parser.add_argument(
        "--name",
        default="NCS-Replay",
        help="LSL stream name (must match server config).",
    )
    parser.add_argument(
        "--n-repeat",
        type=float,
        default=float("inf"),
        help="How many times to loop the file (default: inf = forever).",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve()

    # ── Resolve input path ────────────────────────────────────────────────
    if path.is_file() and path.suffix.lower() == ".ncs":
        input_dir = path.parent
        default_channel = path.stem
    elif path.is_dir():
        ncs_files = sorted(path.glob("*.ncs")) + sorted(path.glob("*.NCS"))
        if not ncs_files:
            sys.exit(f"No .ncs files found in {path}")
        input_dir = path
        default_channel = None
    else:
        sys.exit(f"Not a .ncs file or directory: {path}")

    # ── Load via MNE ──────────────────────────────────────────────────────
    print(f"\n📂 Loading Neuralynx data from: {input_dir}")
    raw = mne.io.read_raw_neuralynx(input_dir, preload=True)

    print(f"   Channels : {raw.ch_names}")
    print(f"   Sfreq    : {raw.info['sfreq']:.0f} Hz")
    print(f"   Duration : {raw.times[-1]:.2f} s  ({raw.n_times} samples)")

    # ── Channel selection ─────────────────────────────────────────────────
    if len(raw.ch_names) == 1:
        # Single channel — nothing to pick
        print(f"   Picked   : {raw.ch_names[0]}")
    else:
        pick = args.channel or default_channel
        if pick:
            if pick not in raw.ch_names:
                # Try fuzzy match: filename stem often has _0001 suffix
                close = [c for c in raw.ch_names if c.lower() in pick.lower()
                         or pick.lower() in c.lower()]
                if close:
                    print(f"   ⚠️  '{pick}' not found, using closest match: {close[0]}")
                    pick = close[0]
                else:
                    sys.exit(
                        f"Channel '{pick}' not in {raw.ch_names}. "
                        f"Use --channel with one of these names."
                    )
            raw.pick([pick])
            print(f"   Picked   : {pick}")
        else:
            print(f"   ℹ️  Multiple channels — streaming all. "
                  f"Use --channel to select one.")

    # ── Build player ──────────────────────────────────────────────────────
    n_rep = np.inf if np.isinf(args.n_repeat) else int(args.n_repeat)
    player = PlayerLSL(
        raw,
        chunk_size=args.chunk_size,
        n_repeat=n_rep,
        name=args.name,
    )
    player.start()

    sfreq = raw.info["sfreq"]
    chunk_latency_ms = args.chunk_size / sfreq * 1000

    print(f"\n▶️  LSL stream started")
    print(f"   Name         : {args.name}")
    print(f"   Channels     : {player.ch_names}")
    print(f"   Sfreq        : {sfreq:.0f} Hz")
    print(f"   Chunk size   : {args.chunk_size}  ({chunk_latency_ms:.2f} ms)")
    print(f"   Repeat       : {'∞' if np.isinf(n_rep) else int(n_rep)}")
    print(f"\n   Waiting for consumers…  (Ctrl+C to stop)\n")

    try:
        # Block until user stops
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        print("\n⏹  Player stopped.")


if __name__ == "__main__":
    main()
