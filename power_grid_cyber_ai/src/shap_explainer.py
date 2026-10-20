"""Local, probability-validated SHAP explanations for Random Forest only."""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


def create_rf_explainer(model):
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("Only RandomForestClassifier is supported.")
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")


def explain_rf_prediction(model, X_ready, feature_order, encoder, explainer=None, target_class=None):
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("Only RandomForestClassifier is supported.")
    X = np.asarray(X_ready, dtype=np.float64)
    names = [str(name) for name in feature_order]
    if X.shape != (1, len(names)) or X.shape[1] != model.n_features_in_:
        raise ValueError("Expected exactly one prepared row matching all model features.")
    if not np.isfinite(X).all():
        raise ValueError("Input contains non-finite values.")
    class_names = encoder.inverse_transform(np.asarray(model.classes_).astype(int)).astype(str)
    probabilities = np.asarray(model.predict_proba(X)[0], dtype=float)
    if target_class is None:
        index = int(np.argmax(probabilities))
    else:
        matches = np.flatnonzero(class_names == str(target_class))
        if len(matches) != 1:
            raise ValueError(f"Unknown target class: {target_class}")
        index = int(matches[0])
    if explainer is None:
        explainer = create_rf_explainer(model)
    raw = explainer.shap_values(X, check_additivity=True)
    if isinstance(raw, list):
        contributions = np.asarray(raw[index], dtype=float)[0]
    else:
        values = np.asarray(raw, dtype=float)
        expected = (1, len(names), len(class_names))
        if values.shape != expected:
            raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {expected}.")
        contributions = values[0, :, index]
    if contributions.shape != (len(names),) or not np.isfinite(contributions).all():
        raise ValueError("Invalid SHAP contribution vector.")
    bases = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if len(bases) != len(class_names):
        raise ValueError("SHAP base values do not match model classes.")
    base = float(bases[index])
    probability = float(probabilities[index])
    reconstructed = float(base + contributions.sum())
    if not np.isclose(reconstructed, probability, rtol=1e-4, atol=1e-5):
        raise ValueError(f"SHAP reconstruction {reconstructed:.8f} differs from predict_proba {probability:.8f}.")
    table = pd.DataFrame({"feature": names, "input_value": X[0], "shap_value": contributions})
    table["absolute_shap"] = table.shap_value.abs()
    table["contribution_pp"] = table.shap_value * 100
    table["direction"] = np.where(contributions > 0, "يدعم الفئة", np.where(contributions < 0, "يخفض الفئة", "محايد"))
    table = table.sort_values("absolute_shap", ascending=False).reset_index(drop=True)
    return {
        "target_class": str(class_names[index]), "probability": probability,
        "base_value": base, "reconstructed_probability": reconstructed,
        "additivity_residual": abs(reconstructed-probability), "table": table,
        "explanation": shap.Explanation(values=contributions, base_values=base, data=X[0], feature_names=names),
    }
