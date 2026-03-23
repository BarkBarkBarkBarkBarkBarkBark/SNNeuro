# AGENT-HINT: Optuna hyperparameter optimization.
# PURPOSE: TPE sampler searches the space defined in optimization_manifest.yaml.
# SEARCH SPACE: 13 params (8 original + 5 new: inhibition + noise gate).
# ADDING PARAMS: Add to docs/optimization_manifest.yaml, add flat-key to config.py _FLAT_MAP.
# SEE ALSO: evaluate.py (objective function), config.py (Config.from_flat), data/best_config.json
"""
snn_agent.eval.optimize — Optuna hyperparameter optimisation for the SNN.

Uses TPE sampler to search the space defined in an optimisation manifest YAML.
Each trial runs :func:`evaluate_pipeline` with a different config override dict.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import yaml
import optuna
from optuna.samplers import TPESampler

from snn_agent.eval.evaluate import evaluate_pipeline

__all__ = ["run_optimization"]

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # project root
DATA_DIR = ROOT / "data"


def _find_manifest() -> Path:
    """Locate the optimization manifest, checking docs/ first then root."""
    for candidate in [ROOT / "docs" / "optimization_manifest.yaml",
                      ROOT / "optimization_manifest.yaml"]:
        if candidate.exists():
            return candidate
    return ROOT / "docs" / "optimization_manifest.yaml"


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def suggest_params(trial: optuna.Trial, param_defs: dict) -> dict:
    overrides: dict = {}
    for name, spec in param_defs.items():
        ptype = spec["type"]
        if ptype == "bool":
            overrides[name] = trial.suggest_categorical(name, [True, False])
            continue
        lo, hi = spec["low"], spec["high"]
        if ptype == "float":
            overrides[name] = trial.suggest_float(name, lo, hi)
        elif ptype == "log":
            overrides[name] = trial.suggest_float(name, lo, hi, log=True)
        elif ptype == "int":
            overrides[name] = trial.suggest_int(name, lo, hi, step=spec.get("step", 1))
        else:
            raise ValueError(f"Unknown param type '{ptype}' for {name}")
    return overrides


def make_objective(manifest: dict):
    param_defs = manifest["parameters"]
    rec_params = manifest.get("recording", {})
    obj_cfg = manifest.get("objective", {})
    metric_name = obj_cfg.get("metric", "accuracy")

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

        for key in ("accuracy", "recall", "precision", "n_active", "total_spikes", "runtime_s"):
            trial.set_user_attr(key, result.get(key, 0))

        print(
            f"  acc={score:.4f}  recall={result.get('recall',0):.4f}  "
            f"prec={result.get('precision',0):.4f}  "
            f"L1={result.get('n_active',0)}neurons/"
            f"{result.get('total_spikes',0)}spk  ({dt:.0f}s)"
        )
        return float(score)

    return objective


def export_best_config(study: optuna.Study, manifest: dict, out_path: Path):
    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_score": best.value,
        "metric": manifest.get("objective", {}).get("metric", "accuracy"),
        "parameters": dict(best.params),
        "user_attrs": dict(best.user_attrs),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  📄 Best config written to {out_path}")


def export_trials_csv(study: optuna.Study, out_path: Path):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("  ⚠ No completed trials to export.")
        return

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


def run_optimization(
    manifest_path: Path | None = None,
    n_trials: int | None = None,
    n_jobs: int | None = None,
    output_dir: Path | None = None,
):
    if manifest_path is None:
        manifest_path = _find_manifest()
    if output_dir is None:
        output_dir = DATA_DIR

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
    print(f"  Metric    : {manifest.get('objective', {}).get('metric', 'accuracy')}")
    print()

    if actual_jobs > 1:
        print(f"  ⚠ n_jobs={actual_jobs} → forcing 1 (GIL)")
        actual_jobs = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = output_dir / f"{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction=direction,
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    study.optimize(make_objective(manifest), n_trials=actual_trials, timeout=timeout, n_jobs=1)

    print()
    print("=" * 60)
    print(f"  Best trial   : #{study.best_trial.number}")
    print(f"  Best score   : {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"    {k:30s} = {v}")
    print("=" * 60)

    export_best_config(study, manifest, output_dir / "best_config.json")
    export_trials_csv(study, output_dir / "trials.csv")
    return study


def main() -> None:
    """CLI entry point (``snn-optimize``)."""
    import argparse

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(description="Optimize SNN hyperparameters")
    parser.add_argument("--manifest", type=str, default=str(_find_manifest()))
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    run_optimization(
        manifest_path=Path(args.manifest),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
