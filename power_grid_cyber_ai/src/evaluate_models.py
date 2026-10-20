import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR


MODELS = [
    ("random_forest", "rf_prediction"),
    ("xgboost", "xgb_prediction"),
    ("catboost", "cat_prediction"),
]

LABELS = ["Normal", "Fault", "Attack"]


def validate_predictions(data):
    required_columns = {
        "y_true",
        *[column for _, column in MODELS],
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}\n"
            "Run the updated src/train_models.py first."
        )

    if data.empty:
        raise ValueError("predictions_test.csv contains no rows.")

    allowed_labels = set(LABELS)

    for column in sorted(required_columns):
        values = data[column].astype("string").str.strip()

        if values.isna().any() or values.eq("").any():
            raise ValueError(
                f"Column '{column}' contains missing or empty labels."
            )

        unexpected = set(values.unique()) - allowed_labels

        if unexpected:
            raise ValueError(
                f"Column '{column}' contains unexpected labels: "
                f"{sorted(unexpected)}"
            )

        data[column] = values.astype(str)

    return data


def evaluate_model(data, model_name, prediction_col):
    y_true = data["y_true"]
    y_pred = data[prediction_col]

    # Reports for each individual class.
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T

    report_df.to_csv(
        RESULTS_DIR
        / "classification_reports"
        / f"{model_name}.csv"
    )

    # Keep the text report names used elsewhere in the project.
    text_report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    # Global metrics.
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="micro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    # Attack is treated as the positive class for these metrics.
    attack_precision = report["Attack"]["precision"]
    attack_recall = report["Attack"]["recall"]
    attack_f1 = report["Attack"]["f1-score"]

    extra_metrics = (
        "\nAdditional metrics:\n"
        f"Macro Precision: {macro_precision:.6f}\n"
        f"Macro Recall:    {macro_recall:.6f}\n"
        f"Macro F1:        {macro_f1:.6f}\n"
        f"Micro F1:        {micro_f1:.6f}\n"
        f"Weighted F1:     {weighted_f1:.6f}\n"
        f"Attack Precision: {attack_precision:.6f}\n"
        f"Attack Recall:    {attack_recall:.6f}\n"
        f"Attack F1:        {attack_f1:.6f}\n"
    )

    (RESULTS_DIR / f"{model_name}_report.txt").write_text(
        text_report + extra_metrics,
        encoding="utf-8",
    )

    # Rows = actual labels; columns = predicted labels.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    )

    matrix_df.index.name = "Actual"
    matrix_df.columns.name = "Predicted"

    matrix_df.to_csv(
        RESULTS_DIR
        / "confusion_matrices"
        / f"{model_name}.csv"
    )

    # Critical directional errors.
    attack_as_fault = int(
        ((y_true == "Attack") & (y_pred == "Fault")).sum()
    )

    attack_as_normal = int(
        ((y_true == "Attack") & (y_pred == "Normal")).sum()
    )

    fault_as_attack = int(
        ((y_true == "Fault") & (y_pred == "Attack")).sum()
    )

    normal_as_attack = int(
        ((y_true == "Normal") & (y_pred == "Attack")).sum()
    )

    return {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),
        "attack_as_fault": attack_as_fault,
        "attack_as_normal": attack_as_normal,
        "attack_missed": attack_as_fault + attack_as_normal,
        "fault_as_attack": fault_as_attack,
        "normal_as_attack": normal_as_attack,
    }


def main():
    predictions_file = RESULTS_DIR / "predictions_test.csv"

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file was not found:\n{predictions_file}\n"
            "Run src/train_models.py first."
        )

    data = pd.read_csv(predictions_file)
    data = validate_predictions(data)

    (RESULTS_DIR / "confusion_matrices").mkdir(
        parents=True,
        exist_ok=True,
    )

    (RESULTS_DIR / "classification_reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for model_name, prediction_col in MODELS:
        print(f"Evaluating {model_name}...")

        rows.append(
            evaluate_model(
                data=data,
                model_name=model_name,
                prediction_col=prediction_col,
            )
        )

    results = pd.DataFrame(rows)

    # Preserve the existing filename used by Model Health.
    results.to_csv(
        RESULTS_DIR / "fault_attack_analysis.csv",
        index=False,
    )

    # A descriptive filename for the complete metrics table.
    results.to_csv(
        RESULTS_DIR / "model_metrics.csv",
        index=False,
    )

    metric_columns = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_f1",
        "attack_precision",
        "attack_recall",
    ]

    error_columns = [
        "model",
        "attack_as_fault",
        "attack_as_normal",
        "attack_missed",
        "fault_as_attack",
        "normal_as_attack",
    ]

    print("\nModel metrics:")
    print(
        results[metric_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nCritical errors:")
    print(results[error_columns].to_string(index=False))

    print("\nSaved results:")
    print(RESULTS_DIR / "model_metrics.csv")
    print(RESULTS_DIR / "fault_attack_analysis.csv")


if __name__ == "__main__":
    main()