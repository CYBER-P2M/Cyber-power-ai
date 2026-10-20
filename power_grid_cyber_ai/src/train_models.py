import json
import sys
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from src.data_loader import drop_leakage_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data/selected/final_verified_multiclass.csv"
EXPECTED_CLASSES = {"Normal", "Attack", "Fault"}
VALIDATION_SIZE = 0.20
RF_TREES = 500


def prepare_fit(X):
    rates = X.isna().mean()
    features = rates[rates <= 0.50].index.tolist()
    if not features:
        raise ValueError("No usable features in training partition.")
    imputer = SimpleImputer(strategy="median")
    ready = imputer.fit_transform(X[features])
    return features, imputer, ready, rates


def prepare_transform(X, features, imputer):
    return imputer.transform(X[features])


def metrics(y, pred, encoder):
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    attack, fault, normal = encoder.transform(["Attack", "Fault", "Normal"])
    attack_count = max(1, int((y == attack).sum()))
    fault_count = max(1, int((y == fault).sum()))
    attack_as_fault = int(((y == attack) & (pred == fault)).sum())
    fault_as_attack = int(((y == fault) & (pred == attack)).sum())
    attack_as_normal = int(((y == attack) & (pred == normal)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(
            y, pred, labels=np.arange(len(encoder.classes_)),
            average="macro", zero_division=0
        )),
        "attack_as_fault": attack_as_fault,
        "fault_as_attack": fault_as_attack,
        "attack_as_normal": attack_as_normal,
        "attack_missed": attack_as_fault + attack_as_normal,
        "normal_as_attack": int(((y == normal) & (pred == attack)).sum()),
        "attack_as_fault_rate": attack_as_fault / attack_count,
        "fault_as_attack_rate": fault_as_attack / fault_count,
        "attack_missed_rate": (attack_as_fault + attack_as_normal) / attack_count,
    }


def build_candidates():
    candidates = [
        ("baseline", {
            "max_features": "sqrt", "min_samples_leaf": 1,
            "class_weight": "balanced"
        })
    ]
    for attack_factor, fault_factor in [
        (0.85, 1.00), (1.15, 1.00), (1.00, 0.85), (1.00, 1.15)
    ]:
        candidates.append((
            f"weights_attack_{attack_factor}_fault_{fault_factor}",
            {
                "max_features": "sqrt", "min_samples_leaf": 1,
                "class_weight": "balanced",
                "weight_factors": {"Attack": attack_factor, "Fault": fault_factor},
            }
        ))
    for max_features in ["sqrt", 0.20]:
        for leaf in [1, 2]:
            if max_features == "sqrt" and leaf == 1:
                continue
            candidates.append((
                f"features_{max_features}_leaf_{leaf}",
                {
                    "max_features": max_features,
                    "min_samples_leaf": leaf, "class_weight": "balanced",
                }
            ))
    candidates.append((
        "sqrt_leaf1_depth40",
        {
            "max_features": "sqrt", "min_samples_leaf": 1,
            "max_depth": 40, "class_weight": "balanced",
        }
    ))
    return candidates


def resolve_params(spec, y, encoder):
    params = dict(spec)
    factors = params.pop("weight_factors", None)
    if factors is not None:
        counts = np.bincount(np.asarray(y, dtype=int), minlength=len(encoder.classes_))
        if (counts == 0).any():
            raise ValueError("Each training partition must contain every class.")
        params["class_weight"] = {
            int(class_id): float(
                len(y) / (len(counts) * count)
                * factors.get(str(encoder.classes_[class_id]), 1.0)
            )
            for class_id, count in enumerate(counts)
        }
    return params


def make_rf(params):
    return RandomForestClassifier(
        n_estimators=RF_TREES, random_state=RANDOM_STATE,
        n_jobs=-1, **params
    )


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    if "label" not in df.columns:
        raise ValueError("Missing label column.")
    y_raw = df["label"].astype("string").str.strip()
    if y_raw.isna().any() or y_raw.eq("").any():
        raise ValueError("Missing or empty labels.")
    if set(y_raw.unique()) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected classes: {y_raw.unique().tolist()}")

    X, removed = drop_leakage_columns(df.drop(columns="label"))
    removed = list(removed)
    forced = [
        c for c in X.columns
        if str(c).strip().lower() in {"label", "marker", "source_file"}
    ]
    X = X.drop(columns=forced)
    removed = list(dict.fromkeys(removed + forced))
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain.")

    # Preserve the existing outer split. No test data is used for selection.
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=TEST_SIZE,
        stratify=y_raw, random_state=RANDOM_STATE
    )
    encoder = LabelEncoder().fit(sorted(EXPECTED_CLASSES))
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    Xt, Xv, yt, yv = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE,
        stratify=y_train, random_state=RANDOM_STATE
    )
    inner_features, inner_imputer, At, _ = prepare_fit(Xt)
    Av = prepare_transform(Xv, inner_features, inner_imputer)
    candidates = build_candidates()
    rows = []
    validation_predictions = pd.DataFrame({
        "source_row_index": Xv.index.to_numpy(),
        "y_true": encoder.inverse_transform(yv),
    })
    print("Class distribution:\n", y_raw.value_counts())
    print("Removed leakage columns:", removed)
    print(f"RF validation candidates: {len(candidates)}", flush=True)

    for name, spec in candidates:
        print(f"Validating RF candidate: {name}", flush=True)
        trial = make_rf(resolve_params(spec, yt, encoder))
        trial.fit(At, yt)
        pred = trial.predict(Av)
        row = {"candidate": name, **metrics(yv, pred, encoder)}
        row["critical_rate_sum"] = row["attack_as_fault_rate"] + row["fault_as_attack_rate"]
        row["selection_loss"] = row["critical_rate_sum"] + 1.0 - row["macro_f1"]
        rows.append(row)
        validation_predictions[f"{name}_prediction"] = encoder.inverse_transform(pred)
        proba = trial.predict_proba(Av)
        for j, class_id in enumerate(trial.classes_):
            label = encoder.inverse_transform([int(class_id)])[0]
            validation_predictions[f"{name}_p_{label.lower()}"] = proba[:, j]
        print(row, flush=True)
        del trial

    results = pd.DataFrame(rows)
    baseline = results.loc[results["candidate"] == "baseline"].iloc[0]
    # Protect against moving missed attacks from Fault to Normal.
    results["eligible"] = (
        (results["attack_as_fault"] <= baseline["attack_as_fault"])
        & (results["fault_as_attack"] <= baseline["fault_as_attack"])
        & (results["attack_as_normal"] <= baseline["attack_as_normal"])
        & (results["normal_as_attack"] <= baseline["normal_as_attack"])
        & (results["macro_f1"] >= baseline["macro_f1"] - 1e-12)
    )
    results["strict_improvement"] = (
        (results["attack_as_fault"] < baseline["attack_as_fault"])
        | (results["fault_as_attack"] < baseline["fault_as_attack"])
    )
    improved = results.loc[results["eligible"] & results["strict_improvement"]]
    if improved.empty:
        chosen_name = "baseline"
        print("No candidate reduced critical errors within the validation safeguards.")
        print("Keeping baseline. No improvement is claimed.")
    else:
        improved = improved.sort_values(
            ["critical_rate_sum", "macro_f1", "accuracy", "candidate"],
            ascending=[True, False, False, True]
        )
        chosen_name = str(improved.iloc[0]["candidate"])
    ranked = results.sort_values(
        ["eligible", "critical_rate_sum", "macro_f1"],
        ascending=[False, True, False]
    )
    selected_spec = dict(dict(candidates)[chosen_name])
    print(f"Selected on validation only: {chosen_name}", flush=True)

    features, imputer, Atrain, missing_rates = prepare_fit(X_train)
    Atest = prepare_transform(X_test, features, imputer)
    # Recalculate custom class weights from the full training partition.
    chosen_params = resolve_params(selected_spec, y_train, encoder)
    rf = make_rf(chosen_params)
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=400,
        max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"
    )
    cat = CatBoostClassifier(
        loss_function="MultiClass", iterations=800, learning_rate=0.05,
        depth=8, l2_leaf_reg=3.0, random_seed=RANDOM_STATE,
        auto_class_weights="Balanced", verbose=100, thread_count=-1,
        allow_writing_files=False
    )
    print("Refitting Random Forest on full training partition...", flush=True)
    rf.fit(Atrain, y_train)
    print("Training XGBoost...", flush=True)
    xgb.fit(Atrain, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    print("Training CatBoost...", flush=True)
    cat.fit(Atrain, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    backup = PROJECT_ROOT / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for folder in (MODELS_DIR, RESULTS_DIR):
        if any(folder.iterdir()):
            shutil.copytree(folder, backup / folder.name)

    dump(imputer, MODELS_DIR / "imputer.joblib")
    dump(encoder, MODELS_DIR / "label_encoder.joblib")
    write_json(MODELS_DIR / "feature_order.json", features)
    metadata = {
        "dataset_file": str(DATA_FILE), "target": "label",
        "classes": encoder.classes_.tolist(), "train_rows": len(X_train),
        "test_rows": len(X_test), "features": len(features),
        "removed_columns": removed,
        "dropped_missing_columns": missing_rates[missing_rates > 0.50].index.tolist(),
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "rf_selected_candidate": chosen_name,
        "rf_selected_spec": selected_spec, "rf_selected_params": chosen_params,
        "rf_n_estimators": RF_TREES,
        "validation_size_within_training": VALIDATION_SIZE,
        "split_strategy": "stratified_random_rows_not_grouped",
        "selection_rule": (
            "Require both critical errors, attack_as_normal, normal_as_attack "
            "not to increase and macro_f1 not to decrease on validation. "
            "Require strict reduction of at least one critical error; "
            "rank by critical_rate_sum then macro_f1. Otherwise retain baseline."
        ),
        "validation_improvement_found": chosen_name != "baseline",
    }
    write_json(MODELS_DIR / "model_metadata.json", metadata)
    ranked.to_csv(RESULTS_DIR / "rf_validation_search.csv", index=False)
    validation_predictions.to_csv(RESULTS_DIR / "rf_validation_predictions.csv", index=False)
    predictions = pd.DataFrame({"y_true": encoder.inverse_transform(y_test)})
    summary = []
    all_labels = np.arange(len(encoder.classes_))
    for name, model, filename, column in [
        ("random_forest", rf, "random_forest.joblib", "rf_prediction"),
        ("xgboost", xgb, "xgboost_model.joblib", "xgb_prediction"),
        ("catboost", cat, "catboost_model.joblib", "cat_prediction"),
    ]:
        dump(model, MODELS_DIR / filename)
        pred = np.asarray(model.predict(Atest)).reshape(-1).astype(int)
        predictions[column] = encoder.inverse_transform(pred)
        report = classification_report(
            y_test, pred, labels=all_labels, target_names=encoder.classes_,
            digits=4, zero_division=0
        )
        (RESULTS_DIR / f"{name}_report.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(
            confusion_matrix(y_test, pred, labels=all_labels),
            index=encoder.classes_, columns=encoder.classes_
        ).to_csv(RESULTS_DIR / f"{name}_confusion_matrix.csv")
        summary.append({"model": name, **metrics(y_test, pred, encoder)})
        if name == "random_forest":
            details = X_test.reset_index(drop=True).copy()
            details.insert(0, "source_row_index", X_test.index.to_numpy())
            if "source_file" in df.columns:
                details["source_file"] = df.loc[X_test.index, "source_file"].to_numpy()
            details["y_true"] = encoder.inverse_transform(y_test)
            details["rf_prediction"] = encoder.inverse_transform(pred)
            proba = model.predict_proba(Atest)
            for j, class_id in enumerate(model.classes_):
                label = encoder.inverse_transform([int(class_id)])[0]
                details[f"p_{label.lower()}"] = proba[:, j]
            errors = details.loc[details["y_true"] != details["rf_prediction"]].copy()
            errors.to_csv(RESULTS_DIR / "rf_error_details.csv", index=False)
            critical = errors.loc[
                ((errors["y_true"] == "Attack") & (errors["rf_prediction"] == "Fault"))
                | ((errors["y_true"] == "Fault") & (errors["rf_prediction"] == "Attack"))
            ]
            critical.to_csv(RESULTS_DIR / "rf_critical_errors.csv", index=False)
        print(f"\n{name}:\n{report}")

    predictions.to_csv(RESULTS_DIR / "predictions_test.csv", index=False)
    pd.DataFrame({"source_row_index": X_test.index}).to_csv(
        RESULTS_DIR / "test_row_indices.csv", index=False
    )
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "training_comparison.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved models: {MODELS_DIR}\nSaved results: {RESULTS_DIR}")
    print(f"Backup location (if old outputs existed): {backup}")


if __name__ == "__main__":
    main()
