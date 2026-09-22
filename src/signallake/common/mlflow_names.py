"""MLflow experiment/registry names and serialization trust list shared between training
(Phase 4) and serving (Phase 5) so they can never drift apart.
"""

EXPERIMENT_NAME = "signallake-anomaly"
REGISTERED_MODEL_NAME = "signallake-anomaly-detector"

# mlflow's skops serializer refuses to round-trip sklearn's tree node storage as "untrusted" by
# default. Both our model types (IsolationForest, HistGradientBoostingClassifier) use it, and
# every model we log/load here was trained by our own Phase 4 run, not sourced externally, so
# trusting it is safe -- see the skops_trusted_types docs on mlflow.sklearn.log_model/load_model.
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",  # IsolationForest's internal tree node storage
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",  # HGBC's tree storage
]
