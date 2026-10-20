"""Single-row prediction, optional local RF SHAP, and grounded report UI."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


def find_project_root():
    # يدعم وجود الصفحة داخل dashboard أو dashboard/pages.
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / "dashboard").is_dir():
            return parent
    raise RuntimeError("Project root not found: expected src and dashboard directories.")


ROOT = find_project_root()
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
for directory in (ROOT, ROOT / "dashboard"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ui.theme import (
    apply_theme, hero, section_header, show_metric, result_card,
    set_state, risk_gauge, prob_chart, prob_donut, data_table,
    flow_steps, show_chart,
)

apply_theme(rtl=True)
hero("Real Prediction", "تحليل صف CSV مع تفسير SHAP وتقرير مبني على الأدلة", rtl=True)
flow_steps(1, ["رفع الملف", "اختيار الصف", "النتيجة وSHAP", "التقرير"])

MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost_model.joblib",
    "CatBoost": "catboost_model.joblib",
}


def fingerprint(path):
    path = Path(path)
    info = path.stat()
    return str(path.resolve()), info.st_mtime_ns, info.st_size


@st.cache_resource(max_entries=1)
def load_bundle(model_fp, imputer_fp, encoder_fp, features_fp):
    # نخزن حزمة واحدة لتجنب تحميل النماذج الثلاثة الكبيرة في الوقت نفسه.
    model = load(model_fp[0])
    imputer = load(imputer_fp[0])
    encoder = load(encoder_fp[0])
    features = json.loads(Path(features_fp[0]).read_text(encoding="utf-8"))
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("feature_order.json must contain unique feature names.")
    return model, imputer, encoder, features


@st.cache_resource(max_entries=1)
def cached_rf_explainer(bundle_signature, helper_fp, _model):
    from src.shap_explainer import create_rf_explainer
    return create_rf_explainer(_model)


# اختيار النموذج قبل تحميل الحزمة.
section_header("١ — رفع الملف واختيار النموذج", "", "📤")
available_models = [name for name, file in MODEL_FILES.items() if (MODELS / file).exists()]
if not available_models:
    st.error("لا يوجد نموذج محفوظ. شغّل التدريب أولًا.")
    st.code("python src/train_models.py")
    st.stop()
selected_model_name = st.selectbox("اختر النموذج", available_models)
model_path = MODELS / MODEL_FILES[selected_model_name]
imputer_path = MODELS / "imputer.joblib"
encoder_path = MODELS / "label_encoder.joblib"
features_path = MODELS / "feature_order.json"
missing_files = [str(p) for p in (imputer_path, encoder_path, features_path) if not p.exists()]
if missing_files:
    st.error("ملفات المعالجة الأساسية غير موجودة.")
    st.code("\n".join(missing_files))
    st.stop()

bundle_signature = tuple(fingerprint(p) for p in (model_path, imputer_path, encoder_path, features_path))
try:
    model, imputer, encoder, feature_order = load_bundle(*bundle_signature)
except Exception as exc:
    st.error(f"تعذر تحميل حزمة النموذج: {exc}")
    st.stop()

uploaded = st.file_uploader("ارفع ملف CSV يحتوي خصائص القياس", type=["csv"])
if uploaded is None:
    st.info("ارفع ملف CSV يحتوي الخصائص المستخدمة أثناء التدريب.")
    st.stop()
try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"تعذر قراءة CSV: {exc}")
    st.stop()
if data.empty:
    st.warning("الملف لا يحتوي صفوف بيانات.")
    st.stop()
missing_features = [name for name in feature_order if name not in data.columns]
if missing_features:
    st.error(f"الملف يفتقد {len(missing_features)} خاصية. لا نغير أسماء أعمدة التدريب تلقائيًا.")
    st.code("\n".join(missing_features[:40]))
    st.stop()
st.write(f"عدد الصفوف: {len(data):,} | عدد الأعمدة: {len(data.columns):,}")

section_header("٢ — اختيار الصف", "الترقيم يبدأ من صفر", "🎚️")
file_digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
row_number = st.number_input("رقم الصف للتحليل", min_value=0, max_value=len(data)-1, value=0, step=1, key=f"rp_row_{file_digest[:16]}")
row_number = int(row_number)
row = data.iloc[[row_number]].copy()
X = row.reindex(columns=feature_order).apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
try:
    X_ready = imputer.transform(X)
    if X_ready.shape != (1, len(feature_order)):
        raise ValueError("Feature count changed after imputation; check saved artifacts.")
    if not np.isfinite(X_ready).all():
        raise ValueError("Prepared row contains non-finite values.")
    probabilities_array = np.asarray(model.predict_proba(X_ready)[0], dtype=float)
    model_classes = np.asarray(model.classes_).astype(int)
    class_names = encoder.inverse_transform(model_classes).astype(str)
    probabilities = {str(c): float(p) for c, p in zip(class_names, probabilities_array)}
    if set(probabilities) != {"Normal", "Fault", "Attack"}:
        raise ValueError("Expected Normal, Fault and Attack classes.")
    if not np.isfinite(probabilities_array).all() or not np.isclose(probabilities_array.sum(), 1.0, atol=1e-5):
        raise ValueError("Invalid class probabilities.")
except Exception as exc:
    st.error(f"تعذر تجهيز الصف أو التنبؤ: {exc}")
    st.stop()
predicted = str(class_names[int(np.argmax(probabilities_array))])
risk_score = 1.0 - probabilities["Normal"]
record_key = (bundle_signature, file_digest, row_number, predicted)
set_state(predicted)

st.divider()
section_header("٣ — النتيجة", selected_model_name, "🛰️")
result_card(predicted, risk_score, probabilities["Attack"])
c1, c2, c3 = st.columns(3)
show_metric(c1, "الفئة المتوقعة", predicted, "🛰️", predicted)
show_metric(c2, "Non-Normal Score", f"{risk_score:.2%}", "📊", predicted, "1 - P(Normal)، وليس احتمال ضرر")
show_metric(c3, "P(Attack)", f"{probabilities['Attack']:.2%}", "🛡️", "Attack")
a, b = st.columns(2)
with a:
    show_chart(risk_gauge(risk_score, predicted), key="rp_gauge")
with b:
    show_chart(prob_donut(probabilities), key="rp_donut")
show_chart(prob_chart(probabilities), key="rp_probs")
data_table(pd.DataFrame({"Class": list(probabilities), "Probability": [f"{p:.2%}" for p in probabilities.values()]}))
with st.expander("عرض المدخلات قبل المعالجة وبعدها"):
    st.dataframe(pd.DataFrame({"feature": feature_order, "raw_value": row[feature_order].iloc[0].to_numpy(), "model_input": X_ready[0]}), use_container_width=True, hide_index=True)

# ---------------- Optional SHAP: no hard dependency for prediction ----------------
st.divider()
section_header("تفسير قرار Random Forest — SHAP", "مساهمات الخصائص ليست انحرافات Z-score ولا إثباتًا لهجوم", "🔍")
helper_path = ROOT / "src" / "shap_explainer.py"
if selected_model_name != "Random Forest":
    st.info("دعم SHAP في هذه النسخة مخصص لـRandom Forest فقط؛ التنبؤ والتقرير يعملان لبقية النماذج.")
elif not helper_path.exists():
    st.warning("ملف src/shap_explainer.py غير موجود. ضَع الملف المرفق في src.")
else:
    shap_key = (record_key, fingerprint(helper_path))
    if st.button("حساب SHAP لهذا الصف", key="rp_compute_shap"):
        st.session_state.pop("rp_local_shap", None)
        try:
            from src.shap_explainer import explain_rf_prediction
            with st.spinner("جاري حساب SHAP لصف واحد؛ قد يستغرق وقتًا مع الغابة الكبيرة..."):
                explainer = cached_rf_explainer(bundle_signature, fingerprint(helper_path), model)
                result = explain_rf_prediction(model, X_ready, feature_order, encoder, explainer=explainer, target_class=predicted)
            st.session_state["rp_local_shap"] = {"key": shap_key, "result": result}
        except ImportError as exc:
            st.error(f"مكتبة مطلوبة غير موجودة أو غير متوافقة: {exc}")
            st.code("python -m pip install shap matplotlib")
        except Exception as exc:
            st.error(f"تعذر حساب SHAP: {exc}")
    stored = st.session_state.get("rp_local_shap")
    if stored and stored.get("key") == shap_key:
        result = stored["result"]
        st.write("الفئة المفسّرة:", result["target_class"])
        a, b, c = st.columns(3)
        show_metric(a, "SHAP Base Value", f"{result['base_value']:.2%}", "📍", hint="مرجع المفسر، وليس احتمال Normal")
        show_metric(b, "Model Probability", f"{result['probability']:.2%}", "🎯")
        show_metric(c, "Base + Contributions", f"{result['reconstructed_probability']:.2%}", "🧮")
        st.caption("الموجب يرفع مخرج الفئة المختارة والسالب يخفضه نسبة إلى مرجع المفسر. لا يحدد ذلك السبب الفيزيائي.")
        top = result["table"].head(15)
        st.dataframe(top[["feature", "input_value", "contribution_pp", "direction"]].style.format({"input_value": "{:.5g}", "contribution_pp": "{:+.3f}"}), use_container_width=True, hide_index=True)
        fig = None
        try:
            import shap
            import matplotlib.pyplot as plt
            shap.plots.waterfall(result["explanation"], max_display=12, show=False)
            fig = plt.gcf()
            st.pyplot(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"تعذر رسم Waterfall؛ الجدول متاح: {exc}")
        finally:
            if fig is not None:
                plt.close(fig)
        st.caption("contribution_pp بوحدة النقاط المئوية؛ مجموع إعادة البناء يستخدم جميع الخصائص.")
        st.download_button("تنزيل جميع مساهمات SHAP", result["table"].to_csv(index=False).encode("utf-8-sig"), file_name=f"shap_row_{row_number}_{predicted}.csv", mime="text/csv", key="rp_download_shap")
        with st.expander("التحقق العددي للتفسير"):
            st.write("فرق إعادة البناء:", result["additivity_residual"])
            st.write("فضاء المخرج: احتمال الفئة، تم التحقق منه عدديًا.")
    st.caption("SHAP هنا للعرض والتنزيل فقط؛ لا يتم تمرير مساهماته إلى llm_report تلقائيًا.")

st.warning("هذه النتائج من نموذج مدرب على بيانات تاريخية/محاكاة. لا تستخدمها لإرسال أوامر إلى شبكة كهرباء حقيقية.")

# ---------------- Prediction history, with legacy schema migration ----------------
st.divider()
section_header("سجل التنبؤات", "", "🗃️")
log_path = RESULTS / "prediction_log.csv"
log_columns = ["timestamp", "model", "filename", "row_number", "predicted_class", "risk_score", "p_normal", "p_fault", "p_attack"]
if st.button("حفظ نتيجة التنبؤ", key="rp_save_prediction"):
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(timespec="seconds"), "model": selected_model_name,
            "filename": uploaded.name, "row_number": row_number, "predicted_class": predicted,
            "risk_score": risk_score, "p_normal": probabilities["Normal"],
            "p_fault": probabilities["Fault"], "p_attack": probabilities["Attack"],
        }])
        if log_path.exists() and log_path.stat().st_size:
            previous = pd.read_csv(log_path)
            if "model" not in previous:
                previous["model"] = "unknown_legacy"
            if list(previous.columns) == log_columns:
                new_row[log_columns].to_csv(log_path, mode="a", header=False, index=False)
            else:
                backup_path = log_path.with_name(f"prediction_log_backup_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
                backup_path.write_bytes(log_path.read_bytes())
                merged = pd.concat([previous, new_row], ignore_index=True, sort=False)
                ordered = log_columns + [col for col in merged if col not in log_columns]
                temp = log_path.with_suffix(".tmp")
                merged[ordered].to_csv(temp, index=False)
                temp.replace(log_path)
        else:
            new_row[log_columns].to_csv(log_path, index=False)
        st.success("تم حفظ التنبؤ مع اسم النموذج.")
    except Exception as exc:
        st.error(f"تعذر حفظ السجل؛ لم نحذف الملف القديم: {exc}")
if log_path.exists():
    try:
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df.tail(20), use_container_width=True, hide_index=True)
        st.download_button("تنزيل سجل التنبؤات", log_df.to_csv(index=False).encode("utf-8-sig"), file_name="prediction_log.csv", mime="text/csv", key="rp_download_log")
    except Exception as exc:
        st.warning(f"تعذر قراءة السجل القديم. احتفظ بنسخة منه وراجع بنية الأعمدة: {exc}")

# ---------------- Grounded report ----------------
st.divider()
section_header("٤ — التقرير التفسيري", "", "📝")
if predicted == "Normal":
    st.info("التنبؤ Normal: لا يُستدعى النموذج اللغوي لهذه الفئة.")
else:
    try:
        from src import llm_report as report_module
        report_settings = report_module.settings()
    except Exception as exc:
        st.error(f"تعذر تحميل وحدة التقرير: {exc}")
        st.stop()

    def optional_fingerprint(path):
        return fingerprint(path) if Path(path).exists() else (str(path), "missing")

    canonical_name = {"Random Forest": "random_forest", "XGBoost": "xgboost", "CatBoost": "catboost"}[selected_model_name]
    reliability_path = (report_settings["RELIABILITY_CSV"] if canonical_name == "random_forest" else Path(report_settings.get("RELIABILITY_DIR", RESULTS / "classification_reports")) / f"{canonical_name}.csv")
    report_signature = (
        record_key,
        json.dumps(report_settings, sort_keys=True, default=str),
        optional_fingerprint(report_module.__file__),
        optional_fingerprint(report_settings["BASELINE_PATH"]),
        optional_fingerprint(report_settings["FEATURE_DICT_PATH"]),
        optional_fingerprint(reliability_path),
        tuple(sorted(probabilities.items())),
    )
    entry = st.session_state.get("rp_current_report")
    if entry and entry.get("signature") != report_signature:
        st.session_state.pop("rp_current_report", None)
        entry = None
    if st.button("مسح التقرير الحالي لإعادة إنشائه", key="rp_clear_report"):
        st.session_state.pop("rp_current_report", None)
        entry = None
    if st.button("إنشاء التقرير", key="generate_evidence_report"):
        if entry is None:
            record = {name: float(value) for name, value in zip(feature_order, X_ready[0])}
            prediction = {"predicted_class": predicted, "probabilities": probabilities, "risk_score": risk_score, "index": row_number, "source": uploaded.name, "model": selected_model_name}
            try:
                with st.spinner("جاري تحليل الأدلة وإنشاء التقرير..."):
                    report = report_module.generate_report(record, prediction)
                entry = {"signature": report_signature, "report": report}
                st.session_state["rp_current_report"] = entry
            except report_module.BaselineMissing as exc:
                st.error(f"ملف Baseline مفقود: {exc}")
                st.code("python tools/build_baseline.py")
            except Exception as exc:
                st.error(f"تعذر إنشاء التقرير: {exc}")
    if entry:
        report = entry["report"]
        status = report.get("status")
        if status == "grounded_template":
            st.info("تقرير مبني مباشرة من الأدلة؛ لم يُشغّل LLM.")
        elif status == "template_fallback":
            st.warning("فشل مسار LLM؛ استُخدم القالب الاحتياطي. راجع تفاصيل التشخيص.")
        elif status == "llm" and report.get("final_narrative_source") == "packet_grounded_template":
            st.info("تم توليد مسودة LLM، لكن التقرير النهائي يستخدم صياغة مبنية من الأدلة.")
        if report.get("lang_warning"):
            st.warning("السرد النهائي في نسخة التقرير الحالية باللغة الإنجليزية.")
        text_col, charts_col = st.columns([3, 2])
        with text_col:
            with st.container(border=True):
                st.markdown(report.get("markdown") or "لا يوجد نص تقرير.")
        with charts_col:
            for chart in report.get("charts", []):
                if Path(chart).is_file():
                    st.image(str(chart))
                else:
                    st.caption("ملف رسم التقرير غير متوفر.")
        with st.expander("تفاصيل إنشاء التقرير"):
            st.write("الوضع:", report.get("mode"))
            st.write("الحالة:", status)
            st.write("تم توليد مسودة LLM:", report.get("llm_used", False))
            st.write("المدة بالثواني:", report.get("duration_seconds"))
            if report.get("error"):
                st.json(report["error"])
            for note in report.get("notes", []):
                st.write("•", note)
        draft = report.get("raw_llm_text", "")
        if draft:
            with st.expander("مسودة LLM — غير معتمدة"):
                st.warning("قد تحتوي ادعاءات غير مدعومة؛ ليست التقرير النهائي ولا تستخدم لاتخاذ إجراء تشغيلي.")
                st.text(draft)
                st.download_button("تنزيل مسودة LLM", draft.encode("utf-8"), file_name=f"llm_draft_row_{row_number}.txt", mime="text/plain", key="rp_download_draft")
        if report.get("markdown"):
            st.download_button("تنزيل التقرير النهائي (.md)", report["markdown"].encode("utf-8"), file_name=f"report_{row_number}_{canonical_name}_{predicted}.md", mime="text/markdown", key="download_evidence_report")
