# AGENT-HINT: Optuna hyperparameter optimization.
# PURPOSE: TPE sampler searches the space defined in optimization_manifest.yaml.
# SEARCH SPACE: 17 params across 6 groups (encoder, DN, STDP, inhibition, noise gate, DEC).
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

from snn_agent.eval.evaluate import multi_evaluate

__all__ = ["run_optimization"]

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # project root
DATA_DIR = ROOT / "data"


def _load_best_seed(param_defs: dict, best_path: Path | None = None) -> dict | None:
    """Load best_config.json and fill defaults for any new manifest params.

    Returns a dict suitable for ``study.enqueue_trial()`` or None if the
    file doesn't exist.
    """
    if best_path is None:
        best_path = DATA_DIR / "best_config.json"
    if not best_path.exists():
        return None

    with open(best_path) as f:
        best = json.load(f)
    known = best.get("parameters", {})

    seed: dict = {}
    for name, spec in param_defs.items():
        if name in known:
            seed[name] = known[name]
        else:
            # Use midpoint of search range as default for new params
            lo, hi = spec.get("low", 0), spec.get("high", 1)
            ptype = spec["type"]
            if ptype == "int":
                seed[name] = (int(lo) + int(hi)) // 2
            elif ptype == "bool":
                seed[name] = True
            elif ptype == "log":
                import math
                seed[name] = math.exp((math.log(lo) + math.log(hi)) / 2)
            else:
                seed[name] = (lo + hi) / 2.0
    return seed


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
    obj_cfg = manifest.get("objective", {})
    metric_name = obj_cfg.get("metric", "f_half")
    scenarios = obj_cfg.get("scenarios", None)
    score_after_s = obj_cfg.get("score_after_s", 15.0)
    delta_time_val = obj_cfg.get("delta_time", 2.0)

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest_params(trial, param_defs)
        print(f"  ▸ Trial {trial.number:3d} started …", end="", flush=True)
        t_start = time.perf_counter()

        try:
            result = multi_evaluate(
                cfg_overrides=overrides,
                scenarios=scenarios,
                score_after_s=score_after_s,
                delta_time=delta_time_val,
                verbose=False,
            )
        except Exception as e:
            dt = time.perf_counter() - t_start
            print(f"  FAILED ({dt:.0f}s) — {e}")
            return 0.0

        score = result.get(metric_name, 0.0)
        dt = time.perf_counter() - t_start

        for key in ("accuracy", "recall", "precision", "f_half",
                     "n_active", "total_spikes", "runtime_s"):
            trial.set_user_attr(key, result.get(key, 0))

        print(
            f"  f½={score:.4f}  acc={result.get('accuracy',0):.4f}  "
            f"prec={result.get('precision',0):.4f}  "
            f"rec={result.get('recall',0):.4f}  ({dt:.0f}s)"
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
    seed_best: bool = True,
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

    # Seed the study with the known best config so TPE starts informed
    if seed_best and len(study.trials) == 0:
        seed_params = _load_best_seed(manifest["parameters"])
        if seed_params is not None:
            study.enqueue_trial(seed_params)
            print("  🌱 Enqueued best_config.json as seed trial (trial 0)")
            for k, v in seed_params.items():
                vfmt = f"{v:.6g}" if isinstance(v, float) else str(v)
                print(f"       {k:30s} = {vfmt}")
            print()

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
    parser.add_argument(
        "--no-seed", action="store_true",
        help="Don't enqueue best_config.json as seed trial",
    )
    args = parser.parse_args()

    run_optimization(
        manifest_path=Path(args.manifest),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        output_dir=Path(args.output_dir),
        seed_best=not args.no_seed,
    )


if __name__ == "__main__":
    main()
