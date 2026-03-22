"""
optimize.py — Automated hyperparameter optimization for the SNN agent.

Uses Optuna (TPE sampler) to search the space defined in
optimization_manifest.yaml.  Each trial runs evaluate.py's
evaluate_pipeline() with a different config override dict and
scores it against spikeinterface ground truth.

Outputs:
    best_config.json  — optimized parameter values
    trials.csv        — full history of all evaluated trials

Usage:
    python optimize.py                           # defaults from manifest
    python optimize.py --n-trials 40 --n-jobs 2  # override budget
    python optimize.py --manifest my_space.yaml  # custom search space
"""

import json
import csv
import sys
import time
from pathlib import Path

import yaml
import optuna
from optuna.samplers import TPESampler

from evaluate import evaluate_pipeline


ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
#  Load manifest
# ─────────────────────────────────────────────────────────────────────────────
def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Build Optuna trial → config overrides
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial: optuna.Trial, param_defs: dict) -> dict:
    """
    Use the manifest parameter definitions to suggest values for one trial.
    Returns a dict suitable for evaluate_pipeline(cfg_overrides=...).
    """
    overrides = {}

    for name, spec in param_defs.items():
        ptype = spec["type"]
        lo = spec["low"]
        hi = spec["high"]

        if ptype == "float":
            overrides[name] = trial.suggest_float(name, lo, hi)
        elif ptype == "log":
            overrides[name] = trial.suggest_float(name, lo, hi, log=True)
        elif ptype == "int":
            step = spec.get("step", 1)
            overrides[name] = trial.suggest_int(name, lo, hi, step=step)
        else:
            raise ValueError(f"Unknown param type '{ptype}' for {name}")

    return overrides


# ─────────────────────────────────────────────────────────────────────────────
#  Objective function
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(manifest: dict):
    """Return an Optuna objective function closed over the manifest."""

    param_defs = manifest["parameters"]
    rec_params = manifest.get("recording", {})
    obj_cfg = manifest.get("objective", {})
    metric_name = obj_cfg.get("metric", "accuracy")
    aggregation = obj_cfg.get("aggregation", "mean")

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest_params(trial, param_defs)
        print(f"  ▸ Trial {trial.number:3d} started …", end="", flush=True)
        t_start = time.perf_counter()

        try:
            result = evaluate_pipeline(
                cfg_overrides=overrides,
                rec_params=rec_params if rec_params else None,
                verbose=False,
            )
        except Exception as e:
            dt = time.perf_counter() - t_start
            print(f"  FAILED ({dt:.0f}s) — {e}")
            return 0.0

        score = result.get(metric_name, 0.0)
        dt = time.perf_counter() - t_start

        # Attach extra metrics as trial user attrs for CSV export
        for key in ("accuracy", "recall", "precision", "n_active",
                    "total_spikes", "runtime_s"):
            trial.set_user_attr(key, result.get(key, 0))

        print(f"  acc={score:.4f}  recall={result.get('recall',0):.4f}  "
              f"prec={result.get('precision',0):.4f}  "
              f"L1={result.get('n_active',0)}neurons/{result.get('total_spikes',0)}spk  "
              f"({dt:.0f}s)")

        return float(score)

    return objective


# ─────────────────────────────────────────────────────────────────────────────
#  Export helpers
# ─────────────────────────────────────────────────────────────────────────────
def export_best_config(study: optuna.Study, manifest: dict, out_path: Path):
    """Write the best trial's params as a ready-to-paste config dict."""
    best = study.best_trial
    config = dict(best.params)

    payload = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_score": best.value,
        "metric": manifest.get("objective", {}).get("metric", "accuracy"),
        "parameters": config,
        "user_attrs": dict(best.user_attrs),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  📄 Best config written to {out_path}")


def export_trials_csv(study: optuna.Study, out_path: Path):
    """Write all completed trials to a CSV for analysis."""
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]

    if not trials:
        print("  ⚠ No completed trials to export.")
        return

    # Collect all param names and user attr names
    param_names = sorted(trials[0].params.keys())
    attr_names = sorted(trials[0].user_attrs.keys()) if trials[0].user_attrs else []

    header = ["trial", "value"] + param_names + attr_names

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for t in trials:
            row = [t.number, t.value]
            row += [t.params.get(p, "") for p in param_names]
            row += [t.user_attrs.get(a, "") for a in attr_names]
            writer.writerow(row)

    print(f"  📊 {len(trials)} trials written to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def run_optimization(
    manifest_path: Path | None = None,
    n_trials: int | None = None,
    n_jobs: int | None = None,
    output_dir: Path | None = None,
):
    """
    Entry point for the optimization loop.

    Parameters
    ----------
    manifest_path : Path
        YAML file defining search space (default: optimization_manifest.yaml).
    n_trials : int
        Override the trial budget from the manifest.
    n_jobs : int
        Override parallelism from the manifest.
    output_dir : Path
        Where to write best_config.json and trials.csv.
    """
    if manifest_path is None:
        manifest_path = ROOT / "optimization_manifest.yaml"
    if output_dir is None:
        output_dir = ROOT

    manifest = load_manifest(manifest_path)
    study_cfg = manifest.get("study", {})

    actual_trials = n_trials or study_cfg.get("n_trials", 50)
    actual_jobs = n_jobs or study_cfg.get("n_jobs", 1)
    direction = study_cfg.get("direction", "maximize")
    study_name = study_cfg.get("name", "snn-optimize")
    timeout = study_cfg.get("timeout_s", None)

    print("=" * 60)
    print("SNN Agent — Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Study     : {study_name}")
    print(f"  Direction : {direction}")
    print(f"  Trials    : {actual_trials}")
    print(f"  Parallel  : {actual_jobs} workers")
    print(f"  Metric    : {manifest.get('objective', {}).get('metric', 'accuracy')}")
    print()

    # Show search space
    print("  Search space:")
    for name, spec in manifest["parameters"].items():
        print(f"    {name:30s}  {spec['type']:5s}  [{spec['low']}, {spec['high']}]")
    print()

    # Python's GIL makes threaded CPU-bound work slower, not faster.
    # Force sequential execution; print a warning if the user overrode.
    if actual_jobs > 1:
        print(f"  ⚠ n_jobs={actual_jobs} → forcing 1 "
              f"(Python GIL prevents real thread parallelism)")
        actual_jobs = 1

    est_s = actual_trials * 30  # ~30s per trial is typical
    print(f"  ⏱  Estimated time: ~{est_s // 60}m {est_s % 60}s "
          f"(~30s per trial)\n")

    # Create study with SQLite storage for resume support
    storage_path = output_dir / f"{study_name}.db"
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=TPESampler(seed=42),
        load_if_exists=True,   # resume if DB exists
    )

    objective = make_objective(manifest)

    t0 = time.perf_counter()

    study.optimize(
        objective,
        n_trials=actual_trials,
        timeout=timeout,
        n_jobs=1,
    )

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("  Optimization complete")
    print(f"  Total time   : {elapsed:.0f}s")
    print(f"  Best trial   : #{study.best_trial.number}")
    print(f"  Best score   : {study.best_value:.4f}")
    print(f"  Best params  :")
    for k, v in study.best_params.items():
        print(f"    {k:30s} = {v}")
    print("=" * 60)
    print()

    # ── Export results ────────────────────────────────────────────────────
    export_best_config(study, manifest, output_dir / "best_config.json")
    export_trials_csv(study, output_dir / "trials.csv")

    return study


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    # Suppress noisy optuna logs (keep warnings)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Optimize SNN hyperparameters with Optuna"
    )
    parser.add_argument(
        "--manifest", type=str,
        default=str(ROOT / "optimization_manifest.yaml"),
        help="Path to search-space YAML",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override number of trials",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Override number of parallel workers",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(ROOT),
        help="Directory for best_config.json and trials.csv",
    )
    args = parser.parse_args()

    run_optimization(
        manifest_path=Path(args.manifest),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        output_dir=Path(args.output_dir),
    )
