import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.theme import (
    apply_theme,
    hero,
    section_header,
    show_metric,
    result_card,
    set_state,
    risk_gauge,
    prob_chart,
    prob_donut,
    count_chart,
    confusion_heatmap,
    feature_bars,
    show_chart,
    status_badge,
    render,
    accuracy_compare,
)


st.set_page_config(
    page_title="PowerGrid Cyber AI",
    page_icon="⚡",
    layout="wide",
)


apply_theme(rtl=True)


hero(
    "PowerGrid Cyber AI",
    "لوحة محاكاة آمنة — لا اتصال بشبكة حقيقية",
    rtl=True,
)


@st.cache_data
def data():
    p = R / "predictions_test.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


d = data()


ICONS = {
    "Overview": "📊",
    "Live Risk": "🎯",
    "Evaluation": "🧪",
    "Feature Importance": "🧬",
    "About": "ℹ️",
}


MODEL_COLUMNS = {
    "Random Forest": "rf_prediction",
    "XGBoost": "xgb_prediction",
    "CatBoost": "cat_prediction",
}


with st.sidebar:
    st.divider()
    page = st.radio(
        "الصفحة",
        list(ICONS),
        format_func=lambda k: f"{ICONS[k]}  {k}",
    )
    st.info("بيانات تاريخية/تجريبية فقط")


if page == "Overview":
    section_header(
        "ملخص",
        "نظرة عامة على نتائج الاختبار",
        "📊",
    )

    if d.empty:
        st.warning("شغّل التدريب والتقييم أولاً")
    else:
        e = int((d["y_true"] != d["rf_prediction"]).sum())

        a, b, c = st.columns(3)

        show_metric(a, "حالات الاختبار", len(d), "🧪")
        show_metric(
            b,
            "أخطاء RF",
            e,
            "⚠️",
            "Attack" if e else "Normal",
        )
        show_metric(
            c,
            "Accuracy",
            f"{1 - e / len(d):.2%}",
            "🎯",
            "Normal",
        )

        section_header("توزيع الفئات الحقيقية")
        show_chart(
            count_chart(d["y_true"].value_counts()),
            key="ov_counts",
        )

        if "cat_prediction" in d.columns:
            rf_accuracy = 1 - (d["y_true"] != d["rf_prediction"]).mean()
            xgb_accuracy = 1 - (d["y_true"] != d["xgb_prediction"]).mean()
            cat_accuracy = 1 - (d["y_true"] != d["cat_prediction"]).mean()

            section_header("مقارنة دقة النماذج")
            show_chart(
                accuracy_compare(
                    rf_accuracy,
                    xgb_accuracy,
                    cat_accuracy,
                ),
                key="model_accuracy_compare",
            )
        else:
            st.warning(
                "عمود cat_prediction غير موجود. "
                "شغّل src/train_models.py المعدّل لإضافة CatBoost."
            )

        section_header("آخر 30 سجلاً")
        st.dataframe(
            d.tail(30),
            use_container_width=True,
        )


elif page == "Live Risk":
    section_header(
        "محاكاة Risk Score",
        "يجب أن يكون مجموع الاحتمالات 100% بالضبط.",
        "🎯",
    )

    normal = st.slider(
        "P(Normal)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
    )

    remaining = max(0.0, 1.0 - normal)

    if remaining <= 0.0:
        fault = 0.0
        attack = 0.0
    else:
        fault = st.slider(
            "P(Fault)",
            min_value=0.0,
            max_value=remaining,
            value=min(0.25, remaining),
            step=0.01,
        )
        attack = remaining - fault

    probabilities = {
        "Normal": normal,
        "Fault": fault,
        "Attack": attack,
    }

    total = sum(probabilities.values())
    probabilities["Attack"] += 1.0 - total
    probabilities["Attack"] = max(0.0, probabilities["Attack"])

    predicted = max(probabilities, key=probabilities.get)
    risk_score = 1.0 - probabilities["Normal"]

    set_state(predicted)

    c1, c2, c3 = st.columns(3)

    show_metric(c1, "الفئة", predicted, "🛰️", predicted)
    show_metric(
        c2,
        "Risk Score",
        f"{risk_score:.2%}",
        "🔥",
        predicted,
    )
    show_metric(
        c3,
        "P(Attack)",
        f'{probabilities["Attack"]:.2%}',
        "☠️",
        "Attack",
    )

    g, p = st.columns([1, 1])

    with g:
        show_chart(
            risk_gauge(risk_score, predicted),
            key="live_gauge",
        )

    with p:
        show_chart(
            prob_donut(probabilities),
            key="live_donut",
        )

    show_chart(
        prob_chart(probabilities),
        key="live_probs",
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Class": list(probabilities.keys()),
                "Probability": [
                    f"{value:.2%}"
                    for value in probabilities.values()
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if normal >= 0.99:
        st.success("الحالة طبيعية تقريباً: Risk Score منخفض جداً.")
    elif risk_score >= 0.80:
        st.error("تنبيه محاكاة: الحالة غير طبيعية بدرجة مرتفعة.")
    elif risk_score >= 0.50:
        st.warning("تنبيه محاكاة متوسط: تحتاج الحالة إلى مراجعة.")
    else:
        st.info("الخطر المحاكى منخفض.")


elif page == "Evaluation":
    section_header(
        "التقييم",
        "مصفوفة الالتباس والأخطاء الأمنية الحرجة",
        "🧪",
    )

    if d.empty:
        st.warning("شغّل التدريب والتقييم أولاً")
    else:
        if "cat_prediction" not in d.columns:
            st.error(
                "عمود cat_prediction غير موجود في predictions_test.csv.\n"
                "شغّل src/train_models.py المعدّل لإضافة CatBoost أولاً."
            )
        else:
            model_name = st.selectbox(
                "النموذج",
                list(MODEL_COLUMNS.keys()),
            )

            col = MODEL_COLUMNS[model_name]

            m = pd.crosstab(
                d["y_true"],
                d[col],
            ).reindex(
                index=["Normal", "Fault", "Attack"],
                columns=["Normal", "Fault", "Attack"],
                fill_value=0,
            )

            confusion_heatmap(
                m,
                row_title="الحقيقي",
                col_title="المتوقع",
            )

            a, b, c = st.columns(3)

            attack_as_fault = int(
                (
                    (d["y_true"] == "Attack")
                    & (d[col] == "Fault")
                ).sum()
            )

            fault_as_attack = int(
                (
                    (d["y_true"] == "Fault")
                    & (d[col] == "Attack")
                ).sum()
            )

            errors = int((d["y_true"] != d[col]).sum())
            accuracy = 1 - errors / len(d)

            show_metric(
                a,
                "Accuracy",
                f"{accuracy:.2%}",
                "🎯",
                "Normal",
            )

            show_metric(
                b,
                "Attack → Fault",
                attack_as_fault,
                "🚨",
                "Attack",
                "هجوم صُنّف كعطل",
            )

            show_metric(
                c,
                "Fault → Attack",
                fault_as_attack,
                "🚨",
                "Fault",
                "عطل صُنّف كهجوم",
            )


elif page == "Feature Importance":
    p = R / "top_20_features.csv"

    section_header(
        "أهم 20 خاصية",
        "الخصائص الأكثر تأثيراً في قرار النموذج",
        "🧬",
    )

    if p.exists():
        x = pd.read_csv(p)

        show_chart(
            feature_bars(x),
            key="fi_bars",
        )

        st.dataframe(
            x,
            use_container_width=True,
        )
    else:
        st.info("شغّل src/explainability.py")


else:
    section_header(
        "عن النظام",
        "",
        "ℹ️",
    )

    st.write(
        "نظام أكاديمي لتصنيف Normal وFault وAttack. "
        "Risk Score مؤشر دعم قرار وليس أمراً تشغيلياً."
    )

    cols = st.columns(3)

    info = [
        (
            "Normal",
            "تشغيل طبيعي للشبكة دون مؤشرات شاذة.",
        ),
        (
            "Fault",
            "عطل فيزيائي مثل قصر أو انقطاع في الشبكة.",
        ),
        (
            "Attack",
            "نشاط سيبراني خبيث يستهدف أنظمة التحكم.",
        ),
    ]

    state_colors = {
        "Normal": "#00ff9d",
        "Fault": "#ffb020",
        "Attack": "#ff2d55",
    }

    state_rgb = {
        "Normal": "0,255,157",
        "Fault": "255,176,32",
        "Attack": "255,45,85",
    }

    for c, (k, v) in zip(cols, info):
        c.markdown(
            f'<div class="hud" style="--state:{state_colors[k]};'
            f'--state-rgb:{state_rgb[k]}">'
            f'<div class="in">{status_badge(k)}'
            f'<p style="margin-top:14px">{v}</p></div></div>',
            unsafe_allow_html=True,
        )