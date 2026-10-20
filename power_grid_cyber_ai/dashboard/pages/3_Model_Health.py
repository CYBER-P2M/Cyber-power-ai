from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.theme import (
    apply_theme,
    hero,
    section_header,
    show_metric,
    data_table,
    accuracy_compare,
    count_chart,
    show_chart,
)


st.set_page_config(
    page_title="Model Health",
    page_icon="🩺",
    layout="wide",
)

apply_theme(rtl=True)

hero(
    "Model Health",
    "حالة النماذج والبيانات ومقاييس التقييم",
    rtl=True,
)


LABELS = ["Normal", "Fault", "Attack"]

MODEL_COLUMNS = {
    "Random Forest": "rf_prediction",
    "XGBoost": "xgb_prediction",
    "CatBoost": "cat_prediction",
}

MODEL_ICONS = {
    "Random Forest": "🌲",
    "XGBoost": "🚀",
    "CatBoost": "🐱",
}

metadata_path = MODELS / "model_metadata.json"
feature_path = MODELS / "feature_order.json"
predictions_path = RESULTS / "predictions_test.csv"
analysis_path = RESULTS / "fault_attack_analysis.csv"

required_files = {
    "Random Forest": MODELS / "random_forest.joblib",
    "XGBoost": MODELS / "xgboost_model.joblib",
    "CatBoost": MODELS / "catboost_model.joblib",
    "Imputer": MODELS / "imputer.joblib",
    "Label Encoder": MODELS / "label_encoder.joblib",
    "Feature Order": feature_path,
}


def calculate_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="micro",
                zero_division=0,
            )
        ),
        "attack_precision": float(
            report["Attack"]["precision"]
        ),
        "attack_recall": float(
            report["Attack"]["recall"]
        ),
        "attack_f1": float(
            report["Attack"]["f1-score"]
        ),
    }


def validate_label_column(series):
    values = series.astype("string").str.strip()

    if values.isna().any() or values.eq("").any():
        return None, "يحتوي العمود على قيم فارغة أو مفقودة."

    unexpected = set(values.unique()) - set(LABELS)

    if unexpected:
        return (
            None,
            f"يحتوي العمود على فئات غير معروفة: {sorted(unexpected)}",
        )

    return values.astype(str), None


# ---------------- File status ----------------

section_header("حالة الملفات", "", "🗂️")

status_rows = [
    {
        "Component": name,
        "Status": "Available" if path.exists() else "Missing",
        "Path": str(path),
    }
    for name, path in required_files.items()
]

data_table(pd.DataFrame(status_rows))


# ---------------- Training metadata ----------------

if metadata_path.exists():
    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object.")

    except (OSError, ValueError) as exc:
        st.warning(f"تعذر قراءة معلومات التدريب: {exc}")

    else:
        section_header("معلومات التدريب", "", "⚙️")

        c1, c2, c3 = st.columns(3)

        show_metric(
            c1,
            "Training Rows",
            f"{metadata.get('train_rows', 0):,}",
            "🏋️",
        )

        show_metric(
            c2,
            "Test Rows",
            f"{metadata.get('test_rows', 0):,}",
            "🧪",
        )

        show_metric(
            c3,
            "Features",
            metadata.get("features", 0),
            "🧬",
        )

        # Do not display zero ignored rows when this information
        # was not recorded by the training script.
        if "ignored_unknown_rows" in metadata:
            st.write(
                "Ignored Rows:",
                metadata["ignored_unknown_rows"],
            )

        st.write("Target column:", metadata.get("target"))

        st.write(
            "Classes:",
            ", ".join(
                str(value)
                for value in metadata.get("classes", [])
            ),
        )

        if "rf_selected_candidate" in metadata:
            st.write(
                "Selected RF configuration:",
                metadata["rf_selected_candidate"],
            )

        with st.expander("تفاصيل Metadata"):
            st.json(metadata)

else:
    st.info("ملف model_metadata.json غير موجود.")


# ---------------- Test predictions ----------------

if predictions_path.exists():
    try:
        predictions = pd.read_csv(predictions_path)

    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(f"تعذر قراءة predictions_test.csv: {exc}")
        predictions = None

    if predictions is not None:
        if predictions.empty:
            st.warning("ملف التنبؤات لا يحتوي على صفوف.")

        elif "y_true" not in predictions.columns:
            st.error(
                "عمود y_true غير موجود في predictions_test.csv."
            )

        else:
            clean_true, true_error = validate_label_column(
                predictions["y_true"]
            )

            if true_error:
                st.error(f"مشكلة في y_true: {true_error}")

            else:
                predictions["y_true"] = clean_true

                model_metrics = {}

                for model_name, column in MODEL_COLUMNS.items():
                    if column not in predictions.columns:
                        st.warning(
                            f"تنبؤات {model_name} غير موجودة: "
                            f"العمود {column} مفقود."
                        )
                        continue

                    clean_pred, pred_error = validate_label_column(
                        predictions[column]
                    )

                    if pred_error:
                        st.warning(
                            f"تعذر تقييم {model_name}: {pred_error}"
                        )
                        continue

                    predictions[column] = clean_pred

                    model_metrics[model_name] = calculate_metrics(
                        predictions["y_true"],
                        predictions[column],
                    )

                # ---------------- Accuracy summary ----------------

                section_header("إحصاءات الاختبار", "", "📈")

                if model_metrics:
                    containers = st.columns(
                        len(model_metrics) + 1
                    )

                    for container, (model_name, values) in zip(
                        containers,
                        model_metrics.items(),
                    ):
                        show_metric(
                            container,
                            f"{model_name} Accuracy",
                            f"{values['accuracy']:.2%}",
                            MODEL_ICONS[model_name],
                            "Normal",
                        )

                    show_metric(
                        containers[-1],
                        "Test Samples",
                        f"{len(predictions):,}",
                        "🧾",
                    )

                else:
                    st.warning(
                        "لا توجد أعمدة تنبؤ صالحة لحساب المقاييس."
                    )

                # ---------------- Charts ----------------

                left, right = st.columns(2)

                with left:
                    section_header("مقارنة دقة النماذج")

                    if all(
                        name in model_metrics
                        for name in MODEL_COLUMNS
                    ):
                        show_chart(
                            accuracy_compare(
                                model_metrics[
                                    "Random Forest"
                                ]["accuracy"],
                                model_metrics[
                                    "XGBoost"
                                ]["accuracy"],
                                model_metrics[
                                    "CatBoost"
                                ]["accuracy"],
                            ),
                            key="mh_acc",
                        )

                    elif model_metrics:
                        # Keep comparison available even when one
                        # model's prediction column is missing.
                        accuracy_table = pd.DataFrame(
                            {
                                "Model": list(model_metrics),
                                "Accuracy": [
                                    values["accuracy"]
                                    for values in model_metrics.values()
                                ],
                            }
                        )

                        st.dataframe(
                            accuracy_table.style.format(
                                {"Accuracy": "{:.2%}"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                with right:
                    section_header("توزيع الفئات الحقيقية")

                    show_chart(
                        count_chart(
                            predictions["y_true"].value_counts()
                        ),
                        key="mh_dist",
                    )

                # ---------------- Selected model metrics ----------------

                if model_metrics:
                    st.divider()

                    section_header(
                        "مقاييس النموذج",
                        "Precision وRecall ومتوسطات F1",
                        "📊",
                    )

                    selected_model = st.selectbox(
                        "اختر النموذج لعرض مقاييسه",
                        list(model_metrics),
                        key="mh_selected_model",
                    )

                    selected = model_metrics[selected_model]

                    c1, c2, c3 = st.columns(3)

                    show_metric(
                        c1,
                        "Precision — Macro",
                        f"{selected['macro_precision']:.2%}",
                        "🎯",
                        hint="متوسط Precision للفئات بالتساوي",
                    )

                    show_metric(
                        c2,
                        "Recall — Macro",
                        f"{selected['macro_recall']:.2%}",
                        "🔎",
                        hint="متوسط Recall للفئات بالتساوي",
                    )

                    show_metric(
                        c3,
                        "Micro-F1",
                        f"{selected['micro_f1']:.2%}",
                        "⚖️",
                        hint="محسوب من المجاميع عبر الفئات",
                    )

                    c1, c2 = st.columns(2)

                    show_metric(
                        c1,
                        "Accuracy",
                        f"{selected['accuracy']:.2%}",
                        "✅",
                    )

                    show_metric(
                        c2,
                        "Macro-F1",
                        f"{selected['macro_f1']:.2%}",
                        "📐",
                    )

                    st.caption(
                        "في التصنيف الحالي، لكل صف فئة واحدة. "
                        "عند احتساب جميع الفئات دون أوزان للعينات، "
                        "يساوي Micro-F1 قيمة Accuracy."
                    )

                    # ---------------- Attack-specific metrics ----------------

                    section_header(
                        "مقاييس اكتشاف الهجوم",
                        "اعتبار Attack الفئة الإيجابية",
                        "🛡️",
                    )

                    c1, c2, c3 = st.columns(3)

                    show_metric(
                        c1,
                        "Attack Precision",
                        f"{selected['attack_precision']:.2%}",
                        "🎯",
                        hint="نسبة الهجمات الصحيحة من تنبؤات Attack",
                    )

                    show_metric(
                        c2,
                        "Attack Recall",
                        f"{selected['attack_recall']:.2%}",
                        "🔎",
                        hint="نسبة الهجمات المكتشفة من الهجمات الحقيقية",
                    )

                    show_metric(
                        c3,
                        "Attack F1",
                        f"{selected['attack_f1']:.2%}",
                        "⚖️",
                    )

                    # ---------------- Per-class report ----------------

                    selected_column = MODEL_COLUMNS[
                        selected_model
                    ]

                    per_class_report = classification_report(
                        predictions["y_true"],
                        predictions[selected_column],
                        labels=LABELS,
                        output_dict=True,
                        zero_division=0,
                    )

                    per_class = pd.DataFrame(
                        [
                            {
                                "Class": label,
                                "Precision": per_class_report[
                                    label
                                ]["precision"],
                                "Recall": per_class_report[
                                    label
                                ]["recall"],
                                "F1": per_class_report[
                                    label
                                ]["f1-score"],
                                "Support": int(
                                    per_class_report[
                                        label
                                    ]["support"]
                                ),
                            }
                            for label in LABELS
                        ]
                    )

                    with st.expander("تفاصيل كل فئة"):
                        st.dataframe(
                            per_class.style.format(
                                {
                                    "Precision": "{:.2%}",
                                    "Recall": "{:.2%}",
                                    "F1": "{:.2%}",
                                    "Support": "{:,.0f}",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                    # ---------------- Comparison table ----------------

                    section_header(
                        "مقارنة جميع المقاييس",
                        "",
                        "📋",
                    )

                    comparison = pd.DataFrame(
                        [
                            {"model": model_name, **values}
                            for model_name, values
                            in model_metrics.items()
                        ]
                    )

                    formats = {
                        column: "{:.2%}"
                        for column in comparison.columns
                        if column != "model"
                    }

                    st.dataframe(
                        comparison.style.format(formats),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "تنزيل جدول المقاييس CSV",
                        data=comparison.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="model_health_metrics.csv",
                        mime="text/csv",
                        key="mh_download_metrics",
                    )

else:
    st.warning(
        "ملف predictions_test.csv غير موجود. "
        "شغّل التدريب أولًا."
    )


# ---------------- Saved analysis ----------------

if analysis_path.exists():
    section_header(
        "نتائج التقييم المحفوظة والأخطاء الحرجة",
        "",
        "🚨",
    )

    try:
        analysis = pd.read_csv(analysis_path)

    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.warning(f"تعذر قراءة ملف التحليل: {exc}")

    else:
        st.caption(
            "هذا الجدول من fault_attack_analysis.csv. "
            "لتحديثه بعد التدريب شغّل: "
            "python src/evaluate_models.py"
        )

        st.dataframe(
            analysis,
            use_container_width=True,
            hide_index=True,
        )


st.info(
    "هذه الصفحة للتدقيق الأكاديمي ومراقبة حالة الملفات والنتائج. "
    "لا تمثل اتصالًا بنظام كهرباء حقيقي."
)