"""Phase 4: train + track anomaly models with MLflow (the 0.89 F1 story).

    uv run python -m signallake.train.train

Reads `data/gold` directly (no Spark -- 2.1M rows of 34 doubles fits comfortably in
pandas), time-splits it into train/validation/test by `event_date` so no future
event ever leaks into training, then logs four runs to the MLflow experiment
"signallake-anomaly":

  - one unsupervised IsolationForest baseline on all 34 features
  - three supervised HistGradientBoostingClassifier runs on the v1 (~12), v2 (~22)
    and v3 (all 34) feature tiers from `signallake.features.columns.FEATURE_TIERS`

For every run, the decision threshold is chosen on the VALIDATION split to maximize
F1, then precision/recall/F1 at that threshold are reported on the untouched TEST
split -- the number that gets printed and registered is always the one the run
actually produced, never a hardcoded target (see BUILD_PLAN §1).

The best-scoring supervised run is registered in the MLflow Model Registry as
"signallake-anomaly-detector", tagged with the lineage a future consumer needs to
trust it (data date range + row count, gold schema hash, feature-set name, git
commit) and with its chosen decision threshold.
"""

import argparse
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signallake.common.config import PROJECT_ROOT, get_settings
from signallake.common.logging import configure_logging, get_logger
from signallake.features.columns import FEATURE_NAMES, features_for_tier

log = get_logger(__name__)

EXPERIMENT_NAME = "signallake-anomaly"
REGISTERED_MODEL_NAME = "signallake-anomaly-detector"
HGBC_TIERS = ("v1", "v2", "v3")
# mlflow's skops serializer refuses to round-trip sklearn's tree node storage as "untrusted" by
# default. Both our model types (IsolationForest, HistGradientBoostingClassifier) use it, and
# every model we log here was trained by this same run, not loaded from an external source, so
# trusting it is safe -- see the skops_trusted_types docs on mlflow.sklearn.log_model.
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",  # IsolationForest's internal tree node storage
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",  # HGBC's tree storage
]
V3_HEALTHY_F1_BAND = (0.85, 0.92)  # BUILD_PLAN §5 Phase 4: re-tune --noise if v3 lands outside this
PERMUTATION_IMPORTANCE_SAMPLE = 50_000  # bound the cost of permutation_importance on huge val sets


def load_gold(gold_dir: Path) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(FEATURE_NAMES)
    return con.execute(f"""
        select {cols}, label, event_date
        from read_parquet('{gold_dir}/**/*.parquet', hive_partitioning=true)
        order by event_date
    """).df()


def time_split(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split by DAY, not by row, so no event from a later day ever lands in an earlier split."""
    dates = np.sort(df["event_date"].unique())
    n = len(dates)
    n_train = min(max(1, round(n * train_frac)), n - 2)
    n_val = min(max(1, round(n * val_frac)), n - n_train - 1)
    train_dates, val_dates, test_dates = (
        dates[:n_train],
        dates[n_train : n_train + n_val],
        dates[n_train + n_val :],
    )
    train_df = df[df["event_date"].isin(train_dates)]
    val_df = df[df["event_date"].isin(val_dates)]
    test_df = df[df["event_date"].isin(test_dates)]
    info = {
        "n_days": n,
        "train": (str(train_dates.min())[:10], str(train_dates.max())[:10], len(train_df)),
        "val": (str(val_dates.min())[:10], str(val_dates.max())[:10], len(val_df)),
        "test": (str(test_dates.min())[:10], str(test_dates.max())[:10], len(test_df)),
    }
    return train_df, val_df, test_df, info


def best_threshold_for_f1(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float]:
    """Sweep the decision threshold and return the one maximizing F1: (threshold, P, R, F1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # last point has no matching threshold
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(precision[best]), float(recall[best]), float(f1[best])


def metrics_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float, float, np.ndarray]:
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, y_pred


def gold_schema_hash(feature_names: list[str]) -> str:
    canonical = "|".join(sorted(feature_names)) + "|label"
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["normal", "anomaly"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(
    names: list[str], importances: np.ndarray, path: Path, title: str
) -> None:
    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(order))))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="#3b6fb6")
    ax.set_xlabel("permutation importance (F1 drop)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_isolation_forest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    artifacts_dir: Path,
) -> dict[str, Any]:
    features = FEATURE_NAMES  # baseline uses all 34, same as the v3 supervised run
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=args.if_n_estimators,
                    contamination=args.if_contamination,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    with mlflow.start_run(run_name="isolation_forest_baseline") as run:
        t0 = time.perf_counter()
        pipeline.fit(train[features])
        fit_s = time.perf_counter() - t0

        # decision_function: higher = more normal: negate so higher = more anomalous,
        # matching predict_proba(...)[:, 1] from the supervised runs below.
        val_scores = -pipeline.decision_function(val[features])
        threshold, val_p, val_r, val_f1 = best_threshold_for_f1(val["label"].to_numpy(), val_scores)

        test_scores = -pipeline.decision_function(test[features])
        test_p, test_r, test_f1, y_pred = metrics_at_threshold(
            test["label"].to_numpy(), test_scores, threshold
        )

        cm_path = artifacts_dir / "confusion_matrix_isolation_forest.png"
        plot_confusion_matrix(test["label"].to_numpy(), y_pred, cm_path, "IsolationForest (test)")

        mlflow.log_params(
            {
                "model_type": "IsolationForest",
                "feature_set": "v3",
                "n_features": len(features),
                "n_estimators": args.if_n_estimators,
                "contamination": args.if_contamination,
                "fit_seconds": round(fit_s, 2),
            }
        )
        mlflow.log_metrics(
            {
                "val_f1_at_threshold": val_f1,
                "val_precision_at_threshold": val_p,
                "val_recall_at_threshold": val_r,
                "chosen_threshold": threshold,
                "test_precision": test_p,
                "test_recall": test_r,
                "test_f1": test_f1,
            }
        )
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES)

        log.info(
            "run.isolation_forest",
            run_id=run.info.run_id,
            test_precision=round(test_p, 4),
            test_recall=round(test_r, 4),
            test_f1=round(test_f1, 4),
        )
        return {
            "run_id": run.info.run_id,
            "kind": "baseline",
            "name": "IsolationForest (baseline)",
            "feature_set": "v3",
            "n_features": len(features),
            "threshold": threshold,
            "test_precision": test_p,
            "test_recall": test_r,
            "test_f1": test_f1,
        }


def run_hgbc(
    tier: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    artifacts_dir: Path,
) -> dict[str, Any]:
    features = features_for_tier(tier)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=args.hgbc_max_iter,
                    learning_rate=args.hgbc_learning_rate,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    with mlflow.start_run(run_name=f"hgbc_{tier}") as run:
        t0 = time.perf_counter()
        pipeline.fit(train[features], train["label"])
        fit_s = time.perf_counter() - t0

        val_scores = pipeline.predict_proba(val[features])[:, 1]
        threshold, val_p, val_r, val_f1 = best_threshold_for_f1(val["label"].to_numpy(), val_scores)

        test_scores = pipeline.predict_proba(test[features])[:, 1]
        test_p, test_r, test_f1, y_pred = metrics_at_threshold(
            test["label"].to_numpy(), test_scores, threshold
        )

        cm_path = artifacts_dir / f"confusion_matrix_{tier}.png"
        plot_confusion_matrix(test["label"].to_numpy(), y_pred, cm_path, f"HGBC {tier} (test)")

        # Bound permutation_importance's cost (n_repeats * n_features model evaluations) with a
        # fixed-size sample of the validation set -- it's only used for the importance ranking,
        # never for threshold selection or reported metrics.
        imp_sample = val.sample(
            n=min(PERMUTATION_IMPORTANCE_SAMPLE, len(val)), random_state=args.random_state
        )
        importance = permutation_importance(
            pipeline,
            imp_sample[features],
            imp_sample["label"],
            scoring="f1",
            n_repeats=5,
            random_state=args.random_state,
            n_jobs=-1,
        )
        fi_path = artifacts_dir / f"feature_importance_{tier}.png"
        plot_feature_importance(
            features, importance.importances_mean, fi_path, f"HGBC {tier}: permutation importance"
        )

        mlflow.log_params(
            {
                "model_type": "HistGradientBoostingClassifier",
                "feature_set": tier,
                "n_features": len(features),
                "max_iter": args.hgbc_max_iter,
                "learning_rate": args.hgbc_learning_rate,
                "fit_seconds": round(fit_s, 2),
            }
        )
        mlflow.log_metrics(
            {
                "val_f1_at_threshold": val_f1,
                "val_precision_at_threshold": val_p,
                "val_recall_at_threshold": val_r,
                "chosen_threshold": threshold,
                "test_precision": test_p,
                "test_recall": test_r,
                "test_f1": test_f1,
            }
        )
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(fi_path))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            metadata={"decision_threshold": threshold, "feature_set": tier},
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        log.info(
            "run.hgbc",
            tier=tier,
            run_id=run.info.run_id,
            test_precision=round(test_p, 4),
            test_recall=round(test_r, 4),
            test_f1=round(test_f1, 4),
        )
        return {
            "run_id": run.info.run_id,
            "kind": "supervised",
            "model_uri": model_info.model_uri,
            "name": f"HGBC {tier}",
            "feature_set": tier,
            "n_features": len(features),
            "threshold": threshold,
            "test_precision": test_p,
            "test_recall": test_r,
            "test_f1": test_f1,
        }


def register_best_model(
    best: dict[str, Any],
    split_info: dict[str, Any],
    n_total_rows: int,
    registered_model_name: str,
) -> None:
    client = MlflowClient()
    mv = mlflow.register_model(model_uri=best["model_uri"], name=registered_model_name)
    date_range = f"{split_info['train'][0]}_to_{split_info['test'][1]}"
    tags = {
        "data_version": f"gold-{n_total_rows}rows-{date_range}",
        "data_date_range": f"{split_info['train'][0]} to {split_info['test'][1]}",
        "gold_schema_hash": gold_schema_hash(FEATURE_NAMES),
        "feature_set": best["feature_set"],
        "git_commit": git_commit(),
        "decision_threshold": str(best["threshold"]),
        "test_f1": str(round(best["test_f1"], 4)),
    }
    for key, value in tags.items():
        client.set_model_version_tag(registered_model_name, mv.version, key, value)
    log.info("model.registered", name=registered_model_name, version=mv.version, **tags)
    print(f"\nRegistered {registered_model_name} v{mv.version} (run {best['run_id']}), tags:")
    for key, value in tags.items():
        print(f"  {key}: {value}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(
        description="Train IsolationForest baseline + 3 HGBC feature-tier runs, log to MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gold-dir", type=Path, default=settings.gold_dir)
    p.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    p.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    p.add_argument("--registered-model-name", default=REGISTERED_MODEL_NAME)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--if-n-estimators", type=int, default=200)
    p.add_argument("--if-contamination", type=float, default=0.03)
    p.add_argument("--hgbc-max-iter", type=int, default=300)
    p.add_argument("--hgbc-learning-rate", type=float, default=0.1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    artifacts_dir = PROJECT_ROOT / "reports" / "_train_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df = load_gold(args.gold_dir)
    print(f"gold: {len(df):,} rows loaded  [{time.perf_counter() - t0:.1f}s]")

    train, val, test, split_info = time_split(df, args.train_frac, args.val_frac)
    print("\n=== Time-based split (train on earlier days, val/test on later days) ===")
    for part in ("train", "val", "test"):
        start, end, n = split_info[part]
        print(f"  {part:5s}: {start} .. {end}  ({n:,} rows)")

    results = [run_isolation_forest(train, val, test, args, artifacts_dir)]
    for tier in HGBC_TIERS:
        results.append(run_hgbc(tier, train, val, test, args, artifacts_dir))

    print("\n=== Test-set results (threshold chosen on validation) ===")
    for r in results:
        print(
            f"  {r['name']:22s} feature_set={r['feature_set']:2s} "
            f"n_features={r['n_features']:2d} threshold={r['threshold']:.3f}  "
            f"P={r['test_precision']:.3f} R={r['test_recall']:.3f} F1={r['test_f1']:.3f}"
        )

    supervised = [r for r in results if r["kind"] == "supervised"]
    best = max(supervised, key=lambda r: r["test_f1"])
    print(f"\nBest supervised run: {best['name']} (test F1={best['test_f1']:.4f})")

    v3 = next(r for r in supervised if r["feature_set"] == "v3")
    lo, hi = V3_HEALTHY_F1_BAND
    if not (lo <= v3["test_f1"] <= hi):
        direction = "lower --noise (too hard)" if v3["test_f1"] < lo else "raise --noise (too easy)"
        print(
            f"\nNOTE: v3 test F1 = {v3['test_f1']:.4f} is outside the {lo}-{hi} band. Per "
            f"BUILD_PLAN §7, don't tweak this number -- regenerate data (Phase 1) with "
            f"{direction}, then re-run `make features` and `make train`. Reporting the real "
            "number above either way."
        )
    if best["feature_set"] != "v3":
        print(
            f"\nNOTE: the best run was {best['feature_set']}, not v3 -- that's a real result, not "
            f"an error. Registering {best['feature_set']} since it's what actually performed best."
        )

    register_best_model(best, split_info, len(df), args.registered_model_name)
    print(f"\nACTUAL test F1 achieved (registered model): {best['test_f1']:.4f}")
    print(f"total wall-clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
